"""复用原版预 tokenized OWT-T5，保留 padding mask，不自造 tokenizer。"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import snapshot_download

from .. import distributed, progress
from ..config import project_path, write_json


def prepared_dir(config):
    data = config["data"]
    return project_path(data["prepared_root"]) / (
        f'elf_t5_l{data["sequence_length"]}_t{data["train_documents"]}_v{data["valid_documents"]}')


def prepare_data(config, tokenizer):
    from datasets import DatasetDict, load_from_disk
    settings = config["data"]
    directory = prepared_dir(config)
    signature = {key: settings[key] for key in ("dataset", "local_dataset", "sequence_length",
                    "validation_documents", "train_documents", "valid_documents")}
    signature["tokenizer"] = config["model"]["encoder"]
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["preprocessing"] != signature:
            raise ValueError("ELF token 缓存配置不同，请更换 data.prepared_root")
        if (directory / "train.pt").exists() and (directory / "valid.pt").exists():
            progress.emit("elf.data.cached", path=str(directory))
            return directory
    source = settings["local_dataset"]
    if source:
        source = str(project_path(source))
    else:
        cache = settings["hf_cache_dir"]
        # 原仓库是 save_to_disk Arrow，而非文本 Parquet；完整复用原 token IDs。
        source = snapshot_download(settings["dataset"], repo_type="dataset",
                                   allow_patterns=["*.json", "*.arrow", "**/*.json", "**/*.arrow"],
                                   cache_dir=str(project_path(cache)) if cache else None)
    raw = load_from_disk(source)
    if isinstance(raw, DatasetDict):
        raw = raw["train"]
    train_end = len(raw) - settings["validation_documents"]
    if not 0 < settings["train_documents"] <= train_end or not 0 < settings["valid_documents"] <= settings["validation_documents"]:
        raise ValueError("ELF 训练/验证文本范围无效")
    directory.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, start, count in (("train", 0, settings["train_documents"]),
                                ("valid", train_end, settings["valid_documents"])):
        ids = torch.full((count, settings["sequence_length"]), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(raw.select(range(start, start + count))):
            values = row["input_ids"][:settings["sequence_length"]]
            valid = min(len(values), int(row.get("sequence_length", len(values))))
            ids[index, :len(values)] = torch.tensor(values, dtype=torch.long)
            mask[index, :valid] = 1
        if (mask.sum(dim=1) == 0).any():
            raise ValueError("ELF 数据含无有效 token 的文本")
        torch.save({"input_ids": ids, "attention_mask": mask}, directory / f"{split}.pt")
        counts[split] = {"document_start": start, "documents": count, "valid_tokens": int(mask.sum())}
        progress.emit("elf.data.prepared", split=split, documents=count)
    write_json(metadata_path, {"preprocessing": signature, "source": source, "splits": counts})
    return directory


class TokenDataset(Dataset):
    def __init__(self, values, indices):
        self.values, self.indices = values, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return {key: value[self.indices[index]] for key, value in self.values.items()}


def make_loader(config, split):
    values = torch.load(prepared_dir(config) / f"{split}.pt", weights_only=True, map_location="cpu")
    limit = config["data"][f"max_{split}_blocks"]
    total = len(values["input_ids"]) if limit is None else min(limit, len(values["input_ids"]))
    indices = range(distributed.rank(), total, distributed.world_size()) if split == "valid" else range(total)
    dataset = TokenDataset(values, indices)
    sampler = None
    if split == "train" and distributed.world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, seed=config["run"]["seed"])
        if len(sampler) == 0:
            raise ValueError("ELF 训练样本少于卡数")
    return DataLoader(dataset, batch_size=config["training"]["batch_size"], sampler=sampler,
                      shuffle=split == "train" and sampler is None, num_workers=0,
                      pin_memory=torch.device(config["model"]["device"]).type == "cuda",
                      generator=torch.Generator().manual_seed(config["run"]["seed"]))
