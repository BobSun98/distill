"""真实 LM1B/OWT 文本在 GPT-2-Large 下的 token 加权 PPL。"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from . import distributed, progress
from .backbones.langflow.text import lm1b_detokenizer
from .config import create_run_dir, load_yaml_config, project_path, write_json


def load_config(path, overrides=()):
    config = load_yaml_config(path, overrides)
    settings = config["evaluation"]
    if min(settings["num_samples"], settings["batch_size"], settings["max_length"] - 1) < 1:
        raise ValueError("样本数、batch 必须为正，评分长度至少为 2")
    if config["model"]["dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("model.dtype 只支持 float32、float16、bfloat16")
    return config


def resolve_scorer(settings):
    local = project_path(settings["name_or_path"])
    if local.is_dir():
        return str(local)
    cache = settings["cache_dir"]
    # 单机共享缓存只由 rank 0 下载，其他卡等待 CPU/Gloo 广播路径。
    return snapshot_download(settings["name_or_path"],
        allow_patterns=["config.json", "generation_config.json", "model.safetensors",
                        "tokenizer*.json", "special_tokens_map.json", "vocab.json", "merges.txt"],
        cache_dir=str(project_path(cache)) if cache else None)


def select_samples(config, corpus, tokenizer):
    from datasets import DatasetDict, load_dataset, load_from_disk

    settings = config["data"][corpus]
    split, tail = settings["split"], settings["tail_documents"]
    split_description = f"{split}[-{tail}:]" if tail else split
    if settings["local_dataset"]:
        dataset = load_from_disk(str(project_path(settings["local_dataset"])))
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]
        if tail:
            if len(dataset) < tail:
                raise ValueError(f"{corpus} 本地缓存不足 {tail} 条，请提供完整原始 split")
            dataset = dataset.select(range(len(dataset) - tail, len(dataset)))
    else:
        cache = config["data"]["hf_cache_dir"] or os.environ.get("DATA_CACHE")
        dataset = load_dataset(settings["dataset"], settings["dataset_config"],
            revision=settings["revision"], split=split_description,
            cache_dir=str(project_path(cache)) if cache else None)
    if "text" not in dataset.column_names:
        raise ValueError(f"{corpus} 需要含 text 列的原始文本缓存，不能使用 token-only 缓存")

    # 先确定 held-out 范围，再固定种子抽取；每条文档/句子是一个独立样本，不 packing。
    indices = list(range(len(dataset)))
    random.Random(config["run"]["seed"]).shuffle(indices)
    samples, skipped = [], 0
    evaluation = config["evaluation"]
    for index in indices:
        text = dataset[index]["text"]
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        if settings["detokenize"]:
            text = lm1b_detokenizer(text)
        ids = tokenizer(text, add_special_tokens=False, truncation=True,
                        max_length=evaluation["max_length"], return_attention_mask=False)["input_ids"]
        # GPT 的首 token 没有前文，不计 NLL；保证恰好选出所需数量的可评分样本。
        if len(ids) < 2:
            skipped += 1
            continue
        samples.append({"sample_id": len(samples), "source_index": index,
                        "text": text, "input_ids": ids})
        if len(samples) == evaluation["num_samples"]:
            break
    if len(samples) != evaluation["num_samples"]:
        raise ValueError(f"{corpus} 只有 {len(samples)} 条可评分文本，少于请求的样本数")
    source = {**settings, "effective_split": split_description,
              "source_index_scope": "effective_split", "seed": config["run"]["seed"],
              "candidate_documents": len(dataset), "skipped_documents": skipped}
    return {"source": source, "samples": samples}


def token_nll(logits, input_ids, attention_mask):
    # logits[t] 预测 token[t+1]；按 attention mask 排除 padding，真实 EOS 仍正常计分。
    labels = input_ids[:, 1:].masked_fill(attention_mask[:, 1:] == 0, -100)
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels,
                             ignore_index=-100, reduction="none")
    return losses.double().sum(dim=1), attention_mask[:, 1:].sum(dim=1)


@torch.inference_mode()
def score_samples(model, samples, pad_token_id, batch_size, device):
    scores = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        length = max(len(sample["input_ids"]) for sample in batch)
        ids = torch.full((len(batch), length), pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for row, sample in enumerate(batch):
            count = len(sample["input_ids"])
            ids[row, :count] = torch.tensor(sample["input_ids"], dtype=torch.long, device=device)
            mask[row, :count] = 1
        with progress.phase("corpus.score.batch", batch=start // batch_size, samples=len(batch)):
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            sums, counts = token_nll(logits, ids, mask)
        for sample, nll, count in zip(batch, sums.cpu().tolist(), counts.cpu().tolist()):
            if not math.isfinite(nll):
                raise FloatingPointError(f"样本 {sample['sample_id']} 的 GPT NLL 非有限值")
            scores.append({"sample_id": sample["sample_id"], "nll_sum": nll,
                           "num_scored_tokens": count})
    return scores


def aggregate_scores(shards):
    scores = sorted((score for shard in shards for score in shard), key=lambda row: row["sample_id"])
    total_nll = sum(score["nll_sum"] for score in scores)
    total_tokens = sum(score["num_scored_tokens"] for score in scores)
    if not total_tokens:
        raise ValueError("没有可评分 token")
    mean_nll = total_nll / total_tokens
    return {"ppl": math.exp(mean_nll), "nll_per_token": mean_nll,
            "num_samples": len(scores), "num_scored_tokens": total_tokens,
            "per_sample": [{**score, "ppl": math.exp(score["nll_sum"] / score["num_scored_tokens"])}
                           for score in scores]}


def run_corpus_ppl(config, command="all", allow_distributed=True):
    owns_group = distributed.setup(config["model"]["device"], allow_distributed,
                                    config["distributed"]["control_timeout_seconds"])
    run_dir = Path(distributed.broadcast_main(str(create_run_dir(config)) if distributed.is_main() else None))
    progress.configure(run_dir, distributed.rank())
    device = torch.device(config["model"]["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前机器没有 CUDA；实际模型评分请在服务器运行")
        device = torch.device("cuda", torch.cuda.current_device())
    try:
        scorer_path = None
        if distributed.is_main():
            with progress.phase("corpus.scorer.resolve", model=config["model"]["name_or_path"]):
                scorer_path = resolve_scorer(config["model"])
                tokenizer = AutoTokenizer.from_pretrained(scorer_path)
                architecture = AutoConfig.from_pretrained(scorer_path)
                if config["evaluation"]["max_length"] > architecture.max_position_embeddings:
                    raise ValueError("评分长度超过 GPT 的预训练 context 长度")
            for corpus in ("lm1b", "owt"):
                with progress.phase("corpus.data.prepare", corpus=corpus):
                    payload = select_samples(config, corpus, tokenizer)
                    write_json(run_dir / corpus / "samples.json", payload)
        scorer_path = distributed.broadcast_main(scorer_path)
        if command == "all":
            torch.set_float32_matmul_precision("highest")
            dtype = getattr(torch, config["model"]["dtype"]) if device.type == "cuda" else torch.float32
            with progress.phase("corpus.scorer.load", path=scorer_path, dtype=str(dtype)):
                scorer = AutoModelForCausalLM.from_pretrained(scorer_path, torch_dtype=dtype)
                scorer = scorer.to(device).eval().requires_grad_(False)
                scorer.config.use_cache = False
            # 模型全就绪后再评分；结果只通过 Gloo 汇总，无 DDP/梯度/NCCL 通信。
            distributed.barrier()
            results = {"scorer": {**config["model"], "resolved_path": scorer_path,
                                  "effective_dtype": str(dtype)},
                       "protocol": {**config["evaluation"], "world_size": distributed.world_size(),
                         "aggregation": "exp(sum_nll / sum_scored_tokens)",
                         "padding": "excluded_by_attention_mask", "add_special_tokens": False,
                         "first_token": "not_scored",
                         "truncation": "first_max_length_tokens_per_document", "packing": False},
                       "results": {}}
            pad_id = scorer.config.eos_token_id
            for corpus in ("lm1b", "owt"):
                payload = json.loads((run_dir / corpus / "samples.json").read_text(encoding="utf-8"))
                local = payload["samples"][distributed.rank()::distributed.world_size()]
                with progress.phase("corpus.score", corpus=corpus, local_samples=len(local)):
                    scores = score_samples(scorer, local, pad_id, config["evaluation"]["batch_size"], device)
                shards = distributed.gather_main(scores)
                if distributed.is_main():
                    metrics = aggregate_scores(shards)
                    if metrics["num_samples"] != config["evaluation"]["num_samples"]:
                        raise ValueError("跨卡汇总后的样本数与请求数量不一致")
                    metrics.update({"source": payload["source"],
                                    "samples_file": str(run_dir / corpus / "samples.json")})
                    results["results"][corpus] = metrics
                    write_json(run_dir / "metrics.json", results)
                    print(f"[GPT-2-Large] {corpus.upper()}: PPL={metrics['ppl']:.6f}, "
                          f"samples={metrics['num_samples']}, tokens={metrics['num_scored_tokens']}", flush=True)
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "completed", "command": command})
            print(f"[corpus] results: {run_dir}", flush=True)
        distributed.barrier()
    except Exception as error:
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        if owns_group:
            distributed.close()
    return run_dir


def main(default_config="configs/gpt2_corpus_ppl.yaml", allow_distributed=True):
    parser = argparse.ArgumentParser(description="真实 LM1B/OWT 文本的 GPT-2-Large PPL")
    parser.add_argument("command", nargs="?", choices=["all", "prepare"], default="all")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    run_corpus_ppl(load_config(args.config, args.overrides), args.command, allow_distributed)


if __name__ == "__main__":
    main()
