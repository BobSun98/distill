"""训练中的 PPL 预算、推理状态恢复及结束后的静态曲线。"""

import argparse
import copy
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import torch

from . import distributed, progress
from .config import project_path


def validate_monitor_config(config):
    settings = config["training"].get("ppl_evaluation", {})
    if settings.get("enabled", False):
        if settings["interval"] == -1:
            return
        if min(settings["interval"], settings["num_samples_per_rank"], settings["batch_size"]) < 1:
            raise ValueError("定期 PPL 的 interval 必须为 -1 或正数，样本数与 batch 必须为正")
        if not config["evaluation"]["scorer"]:
            raise ValueError("定期 PPL 需要 evaluation.scorer（默认 gpt2-large）")


def periodic_config(config):
    result = copy.deepcopy(config)
    settings = config["training"]["ppl_evaluation"]
    result["evaluation"]["num_samples"] = settings["num_samples_per_rank"] * distributed.world_size()
    result["evaluation"]["batch_size"] = settings["batch_size"]
    return result


def ppl_due(settings, step, max_steps, stopping=False):
    # -1 关闭训练循环内的全部 PPL，包括最后一步；外部仍可显式评测 checkpoint。
    return settings.get("enabled", False) and settings["interval"] > 0 and (
        step % settings["interval"] == 0 or step == max_steps or stopping)


@contextmanager
def evaluation_state(model):
    device = next(model.parameters()).device
    training, python_rng = model.training, random.getstate()
    precision = torch.get_float32_matmul_precision()
    devices = [device.index] if device.type == "cuda" else []
    try:
        # 采样会重新设种子；退出后恢复 RNG，不能改变接下来的训练噪声/dropout。
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            yield
    finally:
        model.train(training)
        random.setstate(python_rng)
        torch.set_float32_matmul_precision(precision)


def scorer_files(config, cache):
    if "path" not in cache:
        from .corpus_ppl import resolve_scorer
        path = None
        if distributed.is_main():
            with progress.phase("training.ppl.scorer.resolve", model=config["evaluation"]["scorer"]):
                path = resolve_scorer({"name_or_path": config["evaluation"]["scorer"],
                                       "cache_dir": config["model"]["cache_dir"]})
        cache["path"] = distributed.broadcast_main(path)
    return cache["path"]


def scorer_on_device(cache, device, dtype):
    from transformers import AutoModelForCausalLM
    if "model" not in cache:
        with progress.phase("training.ppl.scorer.load", path=cache["path"], dtype=str(dtype)):
            cache["model"] = AutoModelForCausalLM.from_pretrained(cache["path"], torch_dtype=dtype)
            cache["model"].eval().requires_grad_(False)
            cache["model"].config.use_cache = False
    return cache["model"].to(device)


def release_scorer(cache, device):
    # 保留 CPU 权重供下次使用；训练期间不让 GPT scorer 占据 GPU 显存。
    if "model" in cache:
        cache["model"].cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def plot_training_curves(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    paths = []
    fig, ax = plt.subplots(figsize=(9, 5))
    for phase, label in (("train", "Train loss"), ("valid", "Validation loss")):
        rows = [row for row in events if row["phase"] == phase and math.isfinite(row["loss"])]
        if rows:
            ax.plot([row["step"] for row in rows], [row["loss"] for row in rows], label=label,
                    marker="o" if phase == "valid" else None, markersize=3)
    ax.set(xlabel="Optimizer step", ylabel="Loss", title="Distillation loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = run_dir / "loss_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    evaluations = [row for row in events if row["phase"] == "gen_ppl"]
    if evaluations:
        fig, ax = plt.subplots(figsize=(9, 5))
        # None 表示空文本/无有效 token；用 NaN 留出断点，不能画成零 PPL。
        values = [row["gen_ppl"] if row.get("gen_ppl") is not None else math.nan for row in evaluations]
        ax.plot([row["step"] for row in evaluations], values, marker="o", markersize=3)
        ax.set(xlabel="Optimizer step", ylabel="GPT-2-Large Gen. PPL", title="Current student generation")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = run_dir / "ppl_curve.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="从已有 metrics.jsonl 重绘训练/PPL 曲线")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    for path in plot_training_curves(project_path(args.run_dir)):
        print(path)


if __name__ == "__main__":
    main()
