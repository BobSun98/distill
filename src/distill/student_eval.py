"""独立 student benchmark：输入 safetensors/config，按原版 LangFlow 协议评测。"""

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from . import distributed, progress
from .backbones.langflow.flow_nll import LangFlowForFlowNLL
from .backbones.langflow.reference_data import (
    prepare_validation, reference_tokenizer, validation_loader, validation_cache_path,
)
from .backbones.langflow.reference_gen import generate_texts, compute_generated_ppl, sample_entropy
from .config import create_run_dir, load_yaml_config, project_path, write_json
from .models import load_model, load_tokenizer, parameter_counts
from .models import resolve_model_files
from .train import seed_everything


def load_eval_config(path, overrides=()):
    config = load_yaml_config(path, overrides)
    if not config["metrics"] or not set(config["metrics"]) <= {"flow_ppl", "gen_ppl"}:
        raise ValueError("metrics 只能包含 flow_ppl 和 gen_ppl")
    flow, gen = config["flow_ppl"], config["gen_ppl"]
    if min(flow["num_steps"], flow["batch_size"], flow["sequence_length"] - 2) < 1 or flow["first_n"] < 0:
        raise ValueError("Flow 积分步数/batch/长度必须为正，first_n 不能为负")
    if min(gen["num_samples"], gen["num_steps"] - 1, gen["batch_size"], gen["scorer_batch_size"]) < 1:
        raise ValueError("生成/评分 batch 和样本数必须为正，采样至少 2 次前向")
    return config


def flow_ppl(model, settings, device):
    # 与原版 L.seed_everything(config.seed + rank) 的 CUDA 噪声种子一致。
    torch.set_float32_matmul_precision("high")
    seed_everything(settings["seed"] + distributed.rank())
    loader = validation_loader(settings, distributed.rank(), distributed.world_size(), device.type == "cuda")
    sums = {"nll": 0.0, "reconst_nll": 0.0, "flow_nll": 0.0, "prior_nll": 0.0,
            "tokens": 0, "samples": 0}
    for index, batch in enumerate(loader):
        with progress.phase("flow.batch", batch=index):
            # 散度估计需要对 z 求梯度，不能用 inference_mode/no_grad 包住这一段。
            losses = model.compute_flow_nll(
                batch["input_ids"].to(device), n_steps=settings["num_steps"],
                ode_method=settings["ode_method"], use_self_cond=settings["use_self_cond"],
                attention_mask=batch["attention_mask"].to(device))
        for key in ("nll", "reconst_nll", "flow_nll", "prior_nll"):
            sums[key] += losses[key].detach().sum().item()
        sums["tokens"] += losses["num_valid_tokens"].sum().item()
        sums["samples"] += len(batch["input_ids"])
    shards = distributed.gather_main(sums)
    if not distributed.is_main():
        return None
    # 原版每卡 scalar 先转换为 FP32，再在 rank 0 求和；保留该数值聚合口径。
    totals = {key: torch.tensor([shard[key] for shard in shards], dtype=torch.float32,
                                device=device).sum().item() for key in sums}
    if not totals["tokens"]:
        raise ValueError("验证集中没有有效 token")
    mean_nll = totals["nll"] / totals["tokens"]
    samples = int(totals["samples"])
    return {"ppl": torch.exp(torch.tensor(mean_nll)).item(), "nll_per_token": mean_nll,
            "nll_per_seq": totals["nll"] / samples,
            "reconst_nll": totals["reconst_nll"] / samples,
            "flow_nll": totals["flow_nll"] / samples, "prior_nll": totals["prior_nll"] / samples,
            "num_samples": samples, "num_scored_tokens": int(totals["tokens"]),
            "gumbel_loc": model.proposal.loc.item(), "gumbel_scale": model.proposal.scale.item()}


def generation_ppl(model, tokenizer, settings, config, device, run_dir):
    # 原版 OWT gen_ppl 入口没有启用 TF32；独立评测不继承 flow 的 high 设置。
    torch.set_float32_matmul_precision("highest")
    seed_everything(settings["seed"] + distributed.rank())
    base, remainder = divmod(settings["num_samples"], distributed.world_size())
    count = base + (distributed.rank() < remainder)
    if count:
        with progress.phase("gen.generate", samples=count):
            with torch.inference_mode():
                ids, texts = generate_texts(model, tokenizer, SimpleNamespace(
                    num_samples=count, batch_size=settings["batch_size"],
                    num_steps=settings["num_steps"], seq_length=settings["sequence_length"]), device)
        entropy = sample_entropy(ids)
    else:
        ids = torch.empty((0, settings["sequence_length"]), dtype=torch.long)
        texts, entropy = [], 0.0
    samples = distributed.gather_main((ids, texts))
    if distributed.is_main():
        torch.save(torch.cat([shard[0] for shard in samples]), run_dir / "generated_tokens.pt")
        write_json(run_dir / "generated_texts.json", [text for shard in samples for text in shard[1]])
    # 空 rank 不加载 scorer，但所有 rank 都参与最终的 CPU 结果汇总。
    with progress.phase("gen.score", samples=count):
        cache_dir = config["model"]["cache_dir"]
        cache_dir = str(project_path(cache_dir)) if cache_dir else None
        scores = compute_generated_ppl(texts, settings["scorer"], settings["scorer_batch_size"],
                                       settings["scorer_max_length"], device, cache_dir) if texts else {
            "per_sample_nll": [], "per_sample_tokens": []}
    scores["sample_entropy"], scores["num_samples"] = entropy, count
    shards = distributed.gather_main(scores)
    if not distributed.is_main():
        return None
    # 与 LangFlow/merge_gen_ppl.py 相同：汇总逐样本 NLL×token 数，不平均各卡 PPL。
    nlls = [value for shard in shards for value in shard["per_sample_nll"]]
    tokens = [value for shard in shards for value in shard["per_sample_tokens"]]
    valid = [(nll, count) for nll, count in zip(nlls, tokens) if count > 0 and nll == nll]
    total_nll = sum(nll * count for nll, count in valid)
    total_tokens = sum(count for _, count in valid)
    if not total_tokens:
        raise ValueError("生成文本没有可评分 token")
    mean_nll = total_nll / total_tokens
    return {"gen_ppl": math.exp(mean_nll), "nll_per_token": mean_nll,
            "num_scored_tokens": total_tokens, "num_samples": sum(shard["num_samples"] for shard in shards),
            "sample_entropy": sum(shard["sample_entropy"] * shard["num_samples"] for shard in shards) / settings["num_samples"],
            "per_sample_nll": [value if count > 0 else None for value, count in zip(nlls, tokens)],
            "per_sample_tokens": tokens}


def run_student_eval(config, checkpoint, model_config=None, allow_distributed=True):
    owns_group = distributed.setup(config["model"]["device"], allow_distributed,
                                    config["distributed"]["control_timeout_seconds"])
    run_dir = Path(distributed.broadcast_main(str(create_run_dir(config)) if distributed.is_main() else None))
    progress.configure(run_dir, distributed.rank())
    if distributed.is_main():
        print(f"[eval] {run_dir} (world_size={distributed.world_size()}, metrics={config['metrics']})", flush=True)
    device = torch.device(config["model"]["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前机器没有 CUDA；本地调试请设置 model.device=cpu")
        device = torch.device("cuda", torch.cuda.current_device())
    try:
        # 先检查权重/架构，随后按原版协议准备数据。此入口只评测输入模型，不构造 teacher。
        architecture, weights = resolve_model_files(checkpoint, config["model"]["cache_dir"], model_config)
        cache_dir = config["model"]["cache_dir"]
        cache_dir = str(project_path(cache_dir)) if cache_dir else None
        if "flow_ppl" in config["metrics"]:
            if distributed.is_main():
                tokenizer = reference_tokenizer(config["model"]["tokenizer"], cache_dir)
                with progress.phase("flow.data.prepare"):
                    prepare_validation(config["flow_ppl"], tokenizer)
            distributed.barrier()
        model = load_model(weights, device, config_path=architecture, model_class=LangFlowForFlowNLL)
        # 保留原版 flow evaluator 的梯度设置；只对 z 求导，不执行参数优化。
        tokenizer = load_tokenizer(config)
        if tokenizer.vocab_size != model.config.vocab_size:
            raise ValueError("该 benchmark 使用 OWT GPT-2 词表，请提供对应的 LangFlow student")
        results = {"checkpoint": str(weights), "model_config": str(architecture),
                   "architecture": model.config.to_dict(), "parameters": parameter_counts(model),
                   "tokenizer": config["model"]["tokenizer"],
                   "world_size": distributed.world_size(), "results": {}, "protocol": {}}
        if "flow_ppl" in config["metrics"]:
            metrics = flow_ppl(model, config["flow_ppl"], device)
            if distributed.is_main():
                results["results"]["flow_ppl"] = metrics
                results["protocol"]["flow_ppl"] = {**config["flow_ppl"],
                    "validation_cache": str(validation_cache_path(config["flow_ppl"])),
                    "reference": "run_all_eval_8gpu.sh: owt_ppl / LangFlow/eval_ppl.py",
                    "rank_seed": "seed + rank", "matmul_precision": "high"}
                write_json(run_dir / "metrics.json", results)
                print(f'[eval] OWT Flow PPL: {metrics["ppl"]:.4f}', flush=True)
        if "gen_ppl" in config["metrics"]:
            metrics = generation_ppl(model, tokenizer, config["gen_ppl"], config, device, run_dir)
            if distributed.is_main():
                results["results"]["gen_ppl"] = metrics
                results["protocol"]["gen_ppl"] = {**config["gen_ppl"],
                    "reference": "LangFlow/run_gen_ppl_8gpu.sh / gen_ppl.py / merge_gen_ppl.py",
                    "rank_seed": "seed + rank", "scorer_dtype": "float16" if device.type == "cuda" else "float32",
                    "skip_special_tokens": False, "matmul_precision": "highest"}
                write_json(run_dir / "metrics.json", results)
                print(f'[eval] OWT Gen. PPL: {metrics["gen_ppl"]:.4f}', flush=True)
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "completed", "checkpoint": str(weights)})
            print(f"[eval] results: {run_dir / 'metrics.json'}", flush=True)
        distributed.barrier()
    except Exception as error:
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        if owns_group:
            distributed.close()
    return run_dir


def main(default_config="configs/langflow_student_eval.yaml", allow_distributed=True):
    parser = argparse.ArgumentParser(description="与原版 LangFlow 对齐的 student 评测")
    parser.add_argument("--checkpoint", required=True, help="student safetensors 文件、checkpoint 目录或 HF model id")
    parser.add_argument("--model-config", help="架构 config.json；省略时读取权重同目录文件")
    parser.add_argument("--config", default=default_config, help="评测协议 YAML，与模型架构 JSON 分开")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    run_student_eval(load_eval_config(args.config, args.overrides), args.checkpoint,
                     args.model_config, allow_distributed)


if __name__ == "__main__":
    main()
