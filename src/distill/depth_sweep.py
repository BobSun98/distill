"""同一个 LangFlow-OWT teacher 的独立、均匀删层实验。"""

import argparse
import copy
import csv
import gc
import math
import os
from pathlib import Path

import torch
import yaml

from . import distributed, progress
from .config import create_run_dir, load_config as load_training_config, write_json
from .data import prepare_data
from .evaluate import evaluate_generation
from .models import build_student, load_model, load_teacher, load_tokenizer, parameter_counts
from .train import seed_everything, train
from .training_monitor import evaluation_state
from .gen_ppl_protocol import write_manifest


def uniform_keep_layers(teacher_depth, student_depth):
    if not 2 <= student_depth < teacher_depth:
        raise ValueError("均匀删层要求 2 <= student_depth < teacher_depth")
    # 在原始层编号上等间距取点并取最近整数，保留首尾层；各深度独立计算。
    return [round(index * (teacher_depth - 1) / (student_depth - 1))
            for index in range(student_depth)]


def load_config(path, overrides=()):
    config = load_training_config(path, overrides)
    sweep, training = config["sweep"], config["training"]
    depths = sweep["student_depths"]
    if sweep["layer_selection"] != "uniform":
        raise ValueError("本实验使用 uniform 等间距保留层")
    if not depths or depths != sorted(set(depths), reverse=True):
        raise ValueError("student_depths 必须非空、递减且不重复")
    for depth in depths:
        uniform_keep_layers(sweep["teacher_depth"], depth)
    if training["early_stopping"]["enabled"] or training["ppl_evaluation"]["interval"] != -1:
        raise ValueError("独立层数比较要求关闭早停，并设置 PPL interval=-1；每阶段训练完成后统一评测")
    if not config["evaluation"]["scorer"]:
        raise ValueError("层数实验需要 evaluation.scorer，例如 gpt2-large")
    evaluation = config["evaluation"]
    if min(evaluation["num_samples"], evaluation["batch_size"], evaluation["num_steps"] - 1,
           evaluation["sequence_length"], evaluation["scorer_batch_size"], evaluation["scorer_max_length"] - 1) < 1:
        raise ValueError("生成样本数、batch、长度必须为正，采样前向与 GPT 评分长度至少为 2")
    return config


def stage_config(config, depth):
    result = copy.deepcopy(config)
    result["run"]["name"] = f"student_{depth:02d}_layers"
    result["model"]["keep_layers"] = uniform_keep_layers(config["sweep"]["teacher_depth"], depth)
    return result


def write_summary(run_dir, config, rows):
    write_manifest(run_dir / "gen_ppl_models.yaml", [{"name": f"student_{row['student_depth']}_layers",
        "architecture": str(Path(row["checkpoint"]) / "config.json"),
        "weights": str(Path(row["checkpoint"]) / "model.safetensors"),
        "generation": {"num_steps": config["evaluation"]["num_steps"],
                       "sequence_length": config["evaluation"]["sequence_length"]}} for row in rows])
    write_json(run_dir / "summary.json", {"teacher": config["model"]["teacher"],
        "teacher_depth": config["sweep"]["teacher_depth"], "initialization": "independent_from_teacher",
        "layer_selection": "uniform_linspace_round_keep_endpoints", "steps_per_student": config["training"]["max_steps"],
        "world_size": distributed.world_size(), "evaluation": config["evaluation"], "results": rows})
    fields = ["student_depth", "steps", "gen_ppl", "sample_entropy", "num_samples", "checkpoint"]
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_depth_curve(run_dir, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = [row["student_depth"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, label in ((axes[0], "gen_ppl", "GPT-2-Large Gen. PPL"),
                           (axes[1], "sample_entropy", "Sample entropy")):
        values = [row[key] if row[key] is not None else math.nan for row in rows]
        ax.plot(depths, values, marker="o")
        ax.set(xlabel="Student layers", ylabel=label)
        ax.set_xticks(depths)
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
    fig.suptitle(f"Independent uniform layer pruning — {rows[0]['steps']} steps per student")
    fig.tight_layout()
    path = run_dir / "depth_ppl_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_sweep(config, command="all", allow_distributed=True):
    communication = config["distributed"]
    owns_group = distributed.setup(config["model"]["device"], allow_distributed,
                                    communication["control_timeout_seconds"])
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
            with progress.phase("sweep.data.prepare"):
                prepare_data(config, tokenizer)
        distributed.barrier()
        if command != "prepare":
            if not distributed.is_main():
                tokenizer = load_tokenizer(config)
            distributed.setup_tensor_group(config["model"]["device"], communication["tensor_timeout_seconds"])
            teacher = load_teacher(config)
            device = next(teacher.parameters()).device
            if teacher.config.n_blocks != config["sweep"]["teacher_depth"]:
                raise ValueError("teacher 实际层数与 sweep.teacher_depth 不一致")
            if (len(tokenizer) != teacher.config.vocab_size
                    or max(config["data"]["sequence_length"], config["evaluation"]["sequence_length"]) > teacher.config.model_length):
                raise ValueError("tokenizer/序列长度与真实 teacher 不匹配")
            distributed.barrier()
            for depth in config["sweep"]["student_depths"]:
                current = stage_config(config, depth)
                directory = run_dir / current["run"]["name"]
                if distributed.is_main():
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "config.yaml").write_text(
                        yaml.safe_dump(current, allow_unicode=True, sort_keys=False), encoding="utf-8")
                distributed.barrier()
                progress.configure(directory, distributed.rank())
                keep = current["model"]["keep_layers"]
                progress.emit("sweep.student.begin", depth=depth, keep_layers=keep)
                # 每次都从同一个冻结 teacher 复制；不使用前一阶段 student 的权重或 optimizer。
                student = build_student(teacher, keep)
                counts = parameter_counts(student)
                seed_everything(config["run"]["seed"] + distributed.rank())
                distributed.barrier()
                checkpoint = train(teacher, student, tokenizer, current, directory)
                del student
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                # 早停已关闭，train 返回最后一步；重载真实 safetensors 验证该 checkpoint。
                student = load_model(str(checkpoint), device).eval().requires_grad_(False)
                with progress.phase("sweep.final_ppl", depth=depth, step=current["training"]["max_steps"]):
                    with evaluation_state(student):
                        metrics = evaluate_generation(student, tokenizer, current, directory / "evaluation",
                            step=current["training"]["max_steps"], scorer_cache=scorer_cache, checkpoint=checkpoint)
                metrics = distributed.broadcast_main(metrics)
                if distributed.is_main():
                    row = {"student_depth": depth, "keep_layers": keep,
                        "removed_layers": [i for i in range(teacher.config.n_blocks) if i not in keep],
                        "steps": current["training"]["max_steps"], "parameters": counts,
                        "checkpoint": str(checkpoint), "evaluation_dir": str(directory / "evaluation"),
                        "gen_ppl": metrics["gen_ppl"], "sample_entropy": metrics["sample_entropy"],
                        "num_samples": metrics["num_samples"]}
                    rows.append(row)
                    write_summary(run_dir, config, rows)
                    plot_depth_curve(run_dir, rows)
                    print(f"[sweep] layers={depth}, PPL={row['gen_ppl']:.6f}, checkpoint={checkpoint}", flush=True)
                del student
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                distributed.barrier()
                progress.configure(run_dir, distributed.rank())
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "completed", "command": command,
                                                "completed_depths": [row["student_depth"] for row in rows]})
            print(f"[sweep] results: {run_dir}", flush=True)
        distributed.barrier()
    except Exception as error:
        progress.emit("sweep.failed", error=str(error))
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "failed", "error": str(error),
                                                "completed_depths": [row["student_depth"] for row in rows]})
        raise
    finally:
        if owns_group:
            distributed.close()
    return run_dir


def main(default_config="configs/owt_depth_sweep.yaml", allow_distributed=True):
    parser = argparse.ArgumentParser(description="LangFlow-OWT 11→6 层独立均匀删层与最终 PPL 曲线")
    parser.add_argument("command", nargs="?", choices=["all", "prepare"], default="all")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    run_sweep(load_config(args.config, args.overrides), args.command, allow_distributed)


if __name__ == "__main__":
    main()
