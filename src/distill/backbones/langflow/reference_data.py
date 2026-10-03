"""原版 OWT wrapped 缓存：复用同一验证样本及每 1000 篇分块/丢弃余数规则。"""

import functools
import itertools
import os
from pathlib import Path

import torch
import tokenizers
from transformers import AutoTokenizer

from ...config import project_path


def _group_texts(examples, block_size, bos, eos):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  new_block_size = block_size - 2  # [BOS] and [EOS] to be added
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    _values.append(
      [bos]
      + concatenated_examples[i : i + new_block_size]
      + [eos])
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result


def validation_cache_path(settings):
    if settings["validation_cache"]:
        return project_path(settings["validation_cache"])
    cache = settings["data_cache"] or os.environ.get("DATA_CACHE")
    if cache is None:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        cache = hf_home / "datasets"
    # 与原版 get_dataset("openwebtext-valid", mode="validation", wrap=True) 同名。
    return project_path(cache) / f'openwebtext-valid_validation_bs{settings["sequence_length"]}_wrapped.dat'


def reference_tokenizer(name, cache_dir=None):
    local = project_path(name)
    tokenizer = AutoTokenizer.from_pretrained(str(local) if local.is_dir() else name,
                                              cache_dir=cache_dir)
    # 对齐 duo.dataloader.get_tokenizer；新增 PAD 不参与 wrapped block。
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
        (tokenizer.bos_token, tokenizer.bos_token_id),
        (tokenizer.eos_token, tokenizer.eos_token_id))
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def pack_validation(dataset, tokenizer, sequence_length, workers):
    eos = tokenizer.encode(tokenizer.eos_token)[0]
    bos = tokenizer.encode(tokenizer.bos_token)[0]
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    def tokenize(examples):
        values = tokenizer(examples["text"], add_special_tokens=False,
                           return_attention_mask=False, return_token_type_ids=False)
        return {"input_ids": [ids + [eos] for ids in values["input_ids"]]}

    # 原版两次 batched map 的默认 batch_size 都是 1000；不能用训练的连续 buffer 替代。
    tokens = dataset.map(tokenize, batched=True, batch_size=1000, num_proc=workers or None,
                         load_from_cache_file=True, desc="Tokenizing")
    tokens = tokens.remove_columns("text")
    return tokens.map(functools.partial(_group_texts, block_size=sequence_length, bos=bos, eos=eos),
                      batched=True, batch_size=1000, num_proc=workers or None,
                      load_from_cache_file=True, desc="Grouping")


def prepare_validation(settings, tokenizer):
    from datasets import load_dataset, load_from_disk

    path = validation_cache_path(settings)
    if path.is_dir():
        dataset = load_from_disk(str(path))
    else:
        if settings["validation_cache"]:
            raise FileNotFoundError(f"指定的原版验证缓存不存在: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        # 保留原始 OWT 的最后 100000 篇；同一数据集的 Parquet 转换无需执行 dataset script。
        raw = load_dataset(settings["dataset"], settings["dataset_config"],
                           revision=settings["revision"],
                           split=f'train[-{settings["validation_documents"]}:]',
                           cache_dir=str(path.parent))
        dataset = pack_validation(raw, tokenizer, settings["sequence_length"], settings["data_workers"])
        dataset.save_to_disk(str(path))
    if not len(dataset) or len(dataset[0]["input_ids"]) != settings["sequence_length"]:
        raise ValueError("验证缓存为空或 block 长度与 flow_ppl.sequence_length 不一致")
    return path


def validation_loader(settings, rank, world_size, cuda):
    from datasets import load_from_disk

    dataset = load_from_disk(str(validation_cache_path(settings))).with_format("torch")
    total = min(len(dataset), settings["first_n"]) if settings["first_n"] else len(dataset)
    subset = torch.utils.data.Subset(dataset, range(rank, total, world_size))
    return torch.utils.data.DataLoader(subset, batch_size=settings["batch_size"], shuffle=False,
                                      num_workers=settings["data_workers"], pin_memory=cuda)
