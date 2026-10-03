"""OWT/LM1B 蒸馏数据：保留训练尾部作内部验证，按模型约定 packing。"""

import json
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import project_path, write_json
from .models import load_tokenizer
from .backbones.langflow.text import lm1b_detokenizer
from . import distributed


def prepared_dir(config):
    data = config["data"]
    # 文档数量也写入目录名，debug 与正式实验不会误用彼此的数据缓存。
    name = (f'{data.get("corpus", "owt")}_l{data["sequence_length"]}_'
            f't{data["train_documents"]}_v{data["valid_documents"]}')
    return project_path(data["prepared_root"]) / name


def packing_boundary_ids(tokenizer, corpus):
    if corpus == "lm1b":
        # 沿用原版 wrapped 的 encode(...)[0]：BERT 自动前置 CLS，EOS 实际也取 CLS。
        # 不能无意换成 SEP，否则会改变已有 LM1B checkpoint 的数据边界约定。
        return tokenizer.encode(tokenizer.bos_token)[0], tokenizer.encode(tokenizer.eos_token)[0]
    return tokenizer.bos_token_id, tokenizer.eos_token_id


def pack_documents(documents, tokenizer, sequence_length, batch_size, corpus="owt"):
    content_length = sequence_length - 2
    bos, eos = packing_boundary_ids(tokenizer, corpus)
    buffer, cursor, blocks = [], 0, []
    for start in range(0, len(documents), batch_size):
        texts = documents[start:start + batch_size]["text"]
        if corpus == "lm1b":
            texts = [lm1b_detokenizer(text) for text in texts]
        # 文档可以超过模型长度，之后会 packing 成合法 block；不在 tokenizer 阶段截断。
        encoded = tokenizer(texts, add_special_tokens=False, verbose=False,
                            return_attention_mask=False)["input_ids"]
        for ids in encoded:
            # 每篇文档尾部插入 EOS；每个固定长度 block 再包一层 BOS/EOS，不做 padding。
            buffer.extend(ids)
            buffer.append(eos)
            while len(buffer) - cursor >= content_length:
                blocks.append([bos]
                              + buffer[cursor:cursor + content_length]
                              + [eos])
                cursor += content_length
        buffer = buffer[cursor:]
        cursor = 0
    if not blocks:
        raise ValueError("真实文本不足以组成一个完整 block，请增加文档数")
    return torch.tensor(blocks, dtype=torch.long)


def prepare_data(config, tokenizer=None):
    from datasets import DatasetDict, load_dataset, load_from_disk

    settings = config["data"]
    tokenizer = tokenizer or load_tokenizer(config)
    corpus = settings.get("corpus", "owt")
    directory = prepared_dir(config)
    signature = {key: settings[key] for key in (
        "dataset", "dataset_config", "revision", "local_dataset", "sequence_length",
        "validation_documents", "train_documents", "valid_documents")}
    signature["tokenizer"] = config["model"]["tokenizer"]
    signature["packing"] = "document_eos_block_bos_eos"
    signature["bos_token_id"], signature["eos_token_id"] = packing_boundary_ids(tokenizer, corpus)
    if corpus == "lm1b":
        signature["corpus"] = corpus
        signature["detokenizer"] = "LangFlow.lm1b_detokenizer"
        signature["packing"] = "lm1b_legacy_encoded_boundary_ids"
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["preprocessing"] != signature:
            raise ValueError(f"数据缓存的预处理配置不同，请换 prepared_root: {directory}")
        if (directory / "train.pt").exists() and (directory / "valid.pt").exists():
            print(f"[data] 复用 {directory}", flush=True)
            return directory
    print(f'[data] 加载真实 {corpus.upper()}: {settings["dataset"]}', flush=True)
    # 先划分完整语料的训练/验证范围，再取小样本；避免 debug 样本与验证集重叠。
    if settings["local_dataset"]:
        dataset = load_from_disk(str(project_path(settings["local_dataset"])))
        # LM1B 本地原始缓存可为包含官方 train/test 的 DatasetDict，仅取 train。
        if isinstance(dataset, DatasetDict):
            dataset = dataset["train"]
    else:
        hf_cache = settings["hf_cache_dir"] or os.environ.get("DATA_CACHE")
        dataset = load_dataset(
            settings["dataset"], settings["dataset_config"], revision=settings["revision"],
            split="train", cache_dir=str(project_path(hf_cache)) if hf_cache else None)
    train_end = len(dataset) - settings["validation_documents"]
    train_count, valid_count = settings["train_documents"], settings["valid_documents"]
    if not 0 < train_count <= train_end or not 0 < valid_count <= settings["validation_documents"]:
        raise ValueError("文档数量无效：训练样本必须在保留验证尾部之前，验证样本不能越过尾部")
    directory.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, start, count in (("train", 0, train_count), ("valid", train_end, valid_count)):
        documents = dataset.select(range(start, start + count))
        tokens = pack_documents(documents, tokenizer, settings["sequence_length"],
                                settings["tokenize_batch_size"], corpus)
        torch.save(tokens, directory / f"{split}.pt")
        counts[split] = {"document_start": start, "documents": count, "blocks": len(tokens)}
        print(f"[data] {split}: {count} 条文本 → {len(tokens)} blocks", flush=True)
    write_json(metadata_path, {"preprocessing": signature, "source_documents": len(dataset),
                              "splits": counts})
    return directory


def load_tokens(config, split):
    directory = prepared_dir(config)
    # prepare_data 执行缓存兼容检查；训练/评测入口会先调用它。
    tokens = torch.load(directory / f"{split}.pt", map_location="cpu", weights_only=True)
    if tokens.ndim != 2 or tokens.shape[1] != config["data"]["sequence_length"]:
        raise ValueError(f"缓存中的 token block 形状不正确: {directory}")
    limit = config["data"][f"max_{split}_blocks"]
    return tokens if limit is None else tokens[:limit]


def make_loader(config, split):
    tokens = load_tokens(config, split)
    if len(tokens) == 0:
        raise ValueError(f"{split} 没有可用的 token block")
    sampler = None
    if distributed.world_size() > 1:
        if split == "train":
            # 不补重复样本；每个 epoch 至多丢弃 world_size-1 个尾部样本。
            sampler = DistributedSampler(tokens, shuffle=True, drop_last=True,
                                         seed=config["run"]["seed"])
            if len(sampler) == 0:
                raise ValueError("训练 block 数少于 DDP 进程数，请增加 max_train_blocks")
        else:
            # 验证不补齐、不重复；少量验证样本时允许部分 rank 没有 batch。
            tokens = tokens[distributed.rank()::distributed.world_size()]
    return DataLoader(tokens, batch_size=config["training"]["batch_size"], sampler=sampler,
                      shuffle=split == "train" and sampler is None, num_workers=0,
                      pin_memory=torch.device(config["model"]["device"]).type == "cuda",
                      generator=torch.Generator().manual_seed(config["run"]["seed"]))
