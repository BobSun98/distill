"""统一 Gen PPL v1：固定样本、可寻址噪声、文本清理及 token 加权汇总。"""

import math
from collections import Counter

import torch

SEED = 42
NUM_SAMPLES = 1024
SCORER = "gpt2-large"
MAX_LENGTH = 1024
VERSION = "gen_ppl_v1"


def sample_ids(rank, world_size, count=NUM_SAMPLES):
    return list(range(rank, count, world_size))


def gaussian_noise(ids, length, dimension, stream=0):
    if not 0 < length <= MAX_LENGTH or dimension < 1:
        raise ValueError("噪声长度必须在 1–1024，latent 维度必须为正")
    values = []
    for sample_id in ids:
        generator = torch.Generator(device="cpu").manual_seed(
            (SEED + sample_id * 1_000_003 + stream * 1_000_000_007) % (2**63 - 1))
        # 每个 latent 维度固定占 1024 个随机位置，再转置为 [length,dim]。
        # 长度/维度变化仍共享重叠区域，且不依赖卡数、batch 或模型执行顺序。
        values.append(torch.randn((dimension, MAX_LENGTH), generator=generator,
                                  dtype=torch.float32).T[:length].contiguous())
    return torch.stack(values) if values else torch.empty((0, length, dimension))


def content_token_ids(ids, bos_token_id, eos_token_id):
    ids = list(ids)
    # GPT-2 的 BOS 与 EOS 同号；先去掉至多一个开头 BOS，避免整批误判为空文本。
    if ids and bos_token_id is not None and ids[0] == bos_token_id:
        ids = ids[1:]
    if eos_token_id in ids:
        ids = ids[:ids.index(eos_token_id)]
    return ids


def decode_text(ids, tokenizer):
    ids = content_token_ids(ids, tokenizer.bos_token_id, tokenizer.eos_token_id)
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def text_statistics(ids):
    if not ids:
        return {"sample_entropy": None, "repetition_4gram": None}
    counts = Counter(ids)
    entropy = -sum((count / len(ids)) * math.log(count / len(ids)) for count in counts.values())
    grams = [tuple(ids[index:index + 4]) for index in range(len(ids) - 3)]
    return {"sample_entropy": entropy,
            "repetition_4gram": 1 - len(set(grams)) / len(grams) if grams else None}


def aggregate_scores(shards, expected_count):
    scores = sorted((row for shard in shards for row in shard), key=lambda row: row["sample_id"])
    if [row["sample_id"] for row in scores] != list(range(expected_count)):
        raise ValueError("统一评测 sample_id 必须恰好覆盖所有样本，不缺失、不重复")
    tokens = sum(row["num_scored_tokens"] for row in scores)
    nll = math.fsum(row["nll_sum"] for row in scores)
    mean_nll = nll / tokens if tokens else None
    ppl = math.exp(mean_nll) if mean_nll is not None and mean_nll < 709 else None
    def mean(key):
        values = [row[key] for row in scores if row[key] is not None]
        return math.fsum(values) / len(values) if values else None
    return {"gen_ppl": ppl, "nll_per_token": mean_nll, "num_samples": len(scores),
            "num_scored_tokens": tokens, "num_scored_samples": sum(row["num_scored_tokens"] > 0 for row in scores),
            "num_empty_samples": sum(row["empty"] for row in scores),
            "sample_entropy": mean("sample_entropy"), "repetition_4gram": mean("repetition_4gram"),
            "score_status": ("completed" if all(row["num_scored_tokens"] > 0 for row in scores)
                             else "partial_coverage") if ppl is not None else "no_finite_ppl", "per_sample": scores}


def manifest(models):
    return {"run": {"name": "unified_gen_ppl", "root": "run"},
            "protocol": {"version": VERSION, "seed": SEED, "num_samples": NUM_SAMPLES,
                         "scorer": SCORER, "scorer_max_length": MAX_LENGTH},
            "gradient_moment": {"enabled": False, "reference_config": "configs/gpt2_corpus_ppl.yaml"},
            "runtime": {"device": "cuda", "cache_dir": None, "generation_batch_size": 8,
                        "timing_warmup_forwards": 5,
                        "control_timeout_seconds": 7200}, "models": models}


def write_manifest(path, models):
    import yaml
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest(models), allow_unicode=True, sort_keys=False), encoding="utf-8")
