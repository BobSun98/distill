"""固定 12→6 层 LangFlow-OWT，独立扫描五个初始 LR 并比较训练轨迹。"""

import argparse
import copy
import csv
import gc
import json
import math
import os
from functools import partial
from pathlib import Path

import torch
import yaml

from . import distributed, progress
from .config import create_run_dir, load_config as load_training_config, write_json
from .data import prepare_data
from .depth_sweep import uniform_keep_layers
from .evaluate import training_ppl
from .models import build_student, load_teacher, load_tokenizer
from .train import seed_everything, train
from .gen_ppl_protocol import write_manifest


def load_config(path, overrides=()):
    config = load_training_config(path, overrides)
    settings, training = config["lr_sweep"], config["training"]
    rates = settings["learning_rates"]
    if len(rates) < 2 or len(set(rates)) != len(rates) or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("LR sweep 需要至少两个互不重复的有限正学习率")
    if config["model"]["keep_layers"] != uniform_keep_layers(settings["teacher_depth"], 6):
        raise ValueError("本实验使用从原始 teacher 均匀保留的 6 层 student")
    if training["early_stopping"]["enabled"]:
        raise ValueError("LR sweep 需关闭早停，保证每组训练预算一致")
    if not training["ppl_evaluation"]["enabled"] or training["ppl_evaluation"]["interval"] < 1:
        raise ValueError("LR sweep 需开启正数 interval 的训练期间 PPL 评测")
    if training["lr_scheduler"]["name"] != "cosine" or training["lr_scheduler"]["final_ratio"] != 0.1:
        raise ValueError("本实验使用 cosine scheduler，从初始 LR 衰减到 lr/10")
    return config


def trial_config(config, rate):
    result = copy.deepcopy(config)
    result["run"]["name"] = f"lr_{float(rate)}"
    result["training"]["learning_rate"] = float(rate)
    return result


def read_events(directory):
    return [json.loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def trial_row(rate, directory, checkpoint=None):
    """从一个已完成的 trial 目录重建汇总行；checkpoint 为空时取 training.json 记录的路径。"""
    training = json.loads((directory / "training.json").read_text(encoding="utf-8"))
    events = read_events(directory)
    final_ppl = [event for event in events if event["phase"] == "gen_ppl"][-1]
    return {"learning_rate": float(rate), "final_learning_rate": training["final_learning_rate"],
            "steps": training["steps"], "gen_ppl": final_ppl["gen_ppl"],
            "sample_entropy": final_ppl["sample_entropy"],
            "checkpoint": str(checkpoint) if checkpoint is not None else training["checkpoint"],
            "directory": str(directory)}


def plot_sweep_curves(run_dir, config, rows, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histories = [(row, read_events(Path(row["directory"]))) for row in rows]
    colors = plt.get_cmap("tab10").colors
    paths = []
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, (row, events) in enumerate(histories):
        points = [event for event in events if event["phase"] == "gen_ppl"]
        ax.plot([event["step"] for event in points],
                [event["gen_ppl"] if event.get("gen_ppl") is not None else math.nan for event in points],
                color=colors[index], marker="o", markersize=3, label=f"LR={row['learning_rate']:g}")
    ax.set(xlabel="Optimizer step", ylabel="GPT-2-Large Gen. PPL", title="LangFlow-OWT 12→6: LR sweep")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = run_dir / f"ppl_comparison{suffix}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 6))
    for index, (row, events) in enumerate(histories):
        for phase, linestyle, label in (("train", "-", "train"), ("valid", "--", "val")):
            # 首步诊断日志仍保留；全局对照图按配置的记录间隔取 train loss。
            points = [event for event in events if event["phase"] == phase
                      and (phase == "valid" or event["step"] % config["training"]["log_every"] == 0
                           or event["step"] == config["training"]["max_steps"])]
            ax.plot([event["step"] for event in points], [event["loss"] for event in points],
                    color=colors[index], linestyle=linestyle, label=f"LR={row['learning_rate']:g} {label}")
    ax.set(xlabel="Optimizer step", ylabel="Distillation loss", title="LangFlow-OWT 12→6: train / validation loss")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    path = run_dir / f"loss_comparison{suffix}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def save_results(run_dir, config, rows):
    write_manifest(run_dir / "gen_ppl_models.yaml", [{"name": Path(row["directory"]).name,
        "architecture": str(Path(row["checkpoint"]) / "config.json"),
        "weights": str(Path(row["checkpoint"]) / "model.safetensors"),
        "generation": {"num_steps": config["evaluation"]["num_steps"],
                       "sequence_length": config["evaluation"]["sequence_length"]}} for row in rows])
    write_json(run_dir / "summary.json", {"teacher": config["model"]["teacher"],
        "student_keep_layers": config["model"]["keep_layers"], "initialization": "independent_from_teacher",
        "learning_rates": config["lr_sweep"]["learning_rates"], "training": config["training"],
        "evaluation": config["evaluation"], "world_size": distributed.world_size(), "results": rows})
    fields = ["learning_rate", "final_learning_rate", "steps", "gen_ppl", "sample_entropy", "checkpoint"]
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plot_sweep_curves(run_dir, config, rows)


def run_sweep(config, command="all", allow_distributed=True, resume_dir=None):
    communication = config["distributed"]
    owns_group = distributed.setup(config["model"]["device"], allow_distributed,
                                    communication["control_timeout_seconds"])
    if resume_dir is not None:
        # 复用已有 run 目录：已完成的 trial（存在 training.json）直接跳过并重建汇总行，
        # 仅训练缺失的学习率，最后用全部 trial 重绘对照图。
        run_dir = Path(distributed.broadcast_main(
            str(Path(resume_dir).resolve()) if distributed.is_main() else None))
        if distributed.is_main():
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        run_dir = Path(distributed.broadcast_main(str(create_run_dir(config)) if distributed.is_main() else None))
    progress.configure(run_dir, distributed.rank())
    rows, scorer_cache = [], {}
    try:
        if distributed.is_main():
            write_json(run_dir / "distributed.json", {"world_size": distributed.world_size(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE")})
        tokenizer = load_tokenizer(config) if distributed.is_main() else None
        if distributed.is_main():
            with progress.phase("lr_sweep.data.prepare"):
                prepare_data(config, tokenizer)
        distributed.barrier()
        if command != "prepare":
            if not distributed.is_main():
                tokenizer = load_tokenizer(config)
            distributed.setup_tensor_group(config["model"]["device"], communication["tensor_timeout_seconds"])
            teacher = load_teacher(config)
            device = next(teacher.parameters()).device
            if teacher.config.n_blocks != config["lr_sweep"]["teacher_depth"]:
                raise ValueError("teacher 实际层数与 lr_sweep.teacher_depth 不一致")
            if (len(tokenizer) != teacher.config.vocab_size
                    or max(config["data"]["sequence_length"], config["evaluation"]["sequence_length"]) > teacher.config.model_length):
                raise ValueError("tokenizer/序列长度与真实 teacher 不匹配")
            distributed.barrier()
            for rate in config["lr_sweep"]["learning_rates"]:
                current = trial_config(config, rate)
                directory = run_dir / current["run"]["name"]
                # 所有 rank 共享文件系统，对同一条件判定一致，可统一 continue 而不破坏 barrier 配对。
                if (directory / "training.json").exists():
                    progress.emit("lr_sweep.trial.skip", learning_rate=rate)
                    if distributed.is_main():
                        rows.append(trial_row(rate, directory))
                    continue
                if distributed.is_main():
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "config.yaml").write_text(
                        yaml.safe_dump(current, allow_unicode=True, sort_keys=False), encoding="utf-8")
                distributed.barrier()
                progress.configure(directory, distributed.rank())
                progress.emit("lr_sweep.trial.begin", learning_rate=rate, keep_layers=current["model"]["keep_layers"])
                # 五组均重新复制 teacher，重新创建 optimizer/scheduler，并重置训练随机序列。
                student = build_student(teacher, current["model"]["keep_layers"],
                                        current["model"].get("block_initialization", "teacher"))
                seed_everything(config["run"]["seed"] + distributed.rank())
                distributed.barrier()
                checkpoint = train(teacher, student, tokenizer, current, directory,
                                   ppl_fn=partial(training_ppl, scorer_cache=scorer_cache))
                del student
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if distributed.is_main():
                    row = trial_row(rate, directory, checkpoint)
                    rows.append(row)
                    save_results(run_dir, config, rows)
                    print(f"[lr_sweep] initial_lr={rate:g}, final_lr={row['final_learning_rate']:g}, "
                          f"PPL={row['gen_ppl']:.6f}", flush=True)
                distributed.barrier()
                progress.configure(run_dir, distributed.rank())
        if distributed.is_main() and rows:
            # 跳过全部训练（纯 resume 补图）时这里确保对照图按配置的 LR 顺序重绘一次。
            rows.sort(key=lambda row: config["lr_sweep"]["learning_rates"].index(row["learning_rate"]))
            save_results(run_dir, config, rows)
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "completed", "command": command,
                "completed_learning_rates": [row["learning_rate"] for row in rows]})
            print(f"[lr_sweep] results: {run_dir}", flush=True)
        distributed.barrier()
    except Exception as error:
        progress.emit("lr_sweep.failed", error=str(error))
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "failed", "error": str(error),
                "completed_learning_rates": [row["learning_rate"] for row in rows]})
        raise
    finally:
        if owns_group:
            distributed.close()
    return run_dir


def main(default_config="configs/owt_lr_sweep.yaml", allow_distributed=True):
    parser = argparse.ArgumentParser(description="LangFlow-OWT 12→6 层五组 cosine LR sweep")
    parser.add_argument("command", nargs="?", choices=["all", "prepare"], default="all")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-dir", dest="run_dir", default=None,
                        help="复用已有 sweep 目录：跳过已完成的 LR，仅补训缺失项并重绘对照图")
    args = parser.parse_args()
    run_sweep(load_config(args.config, args.overrides), args.command, allow_distributed, args.run_dir)


if __name__ == "__main__":
    main()
