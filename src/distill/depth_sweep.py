"""同一个 LangFlow-OWT teacher 的独立、均匀删层实验。"""

import argparse
import copy
import csv
import gc
import json
import math
import os
from pathlib import Path

import torch
import yaml

from . import distributed, progress
from .config import create_run_dir, load_config as load_training_config, write_json
from .data import prepare_data
from .evaluate import evaluate_generation
from .models import (build_student, load_model, load_teacher, load_tokenizer,
                     parameter_counts, resolve_model_files)
from .train import seed_everything, train
from .training_monitor import evaluation_state
from .gen_ppl import run_gen_ppl, validate_protocol
from .gen_ppl_protocol import VERSION, MAX_LENGTH, manifest, write_manifest


def uniform_keep_layers(teacher_depth, student_depth):
    if not 2 <= student_depth < teacher_depth:
        raise ValueError("均匀删层要求 2 <= student_depth < teacher_depth")
    # 在原始层编号上等间距取点并取最近整数，保留首尾层；各深度独立计算。
    return [round(index * (teacher_depth - 1) / (student_depth - 1))
            for index in range(student_depth)]


def unified_protocol(config):
    evaluation = config["evaluation"]
    return {"version": VERSION, "seed": evaluation["seed"], "num_samples": evaluation["num_samples"],
            "scorer": evaluation["scorer"], "scorer_max_length": evaluation["scorer_max_length"]}


def load_config(path, overrides=(), debug=False):
    config = load_training_config(path, overrides)
    sweep, training = config["sweep"], config["training"]
    depths = sweep["student_depths"]
    if sweep["layer_selection"] != "uniform":
        raise ValueError("本实验使用 uniform 等间距保留层")
    if not depths or len(depths) != len(set(depths)):
        raise ValueError("student_depths 必须非空且不重复；按配置顺序执行")
    protocol = sweep.get("evaluation_protocol", "legacy")
    if protocol not in ("legacy", VERSION):
        raise ValueError("sweep.evaluation_protocol 只支持 legacy 或 gen_ppl_v1")
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
    if protocol == VERSION:
        # 在长训练开始前复用统一入口的检查，不能训完才发现评分协议配错。
        validate_protocol(unified_protocol(config), debug)
        if evaluation["sequence_length"] > MAX_LENGTH:
            raise ValueError("统一评测生成长度最多 1024")
    return config


def stage_config(config, depth):
    result = copy.deepcopy(config)
    result["run"]["name"] = f"student_{depth:02d}_layers"
    result["model"]["keep_layers"] = uniform_keep_layers(config["sweep"]["teacher_depth"], depth)
    return result


def write_summary(run_dir, config, rows, evaluation=None):
    if config["sweep"].get("evaluation_protocol", "legacy") == "legacy":
        write_manifest(run_dir / "gen_ppl_models.yaml", [{"name": f"student_{row['student_depth']}_layers",
            "architecture": str(Path(row["checkpoint"]) / "config.json"),
            "weights": str(Path(row["checkpoint"]) / "model.safetensors"),
            "generation": {"num_steps": config["evaluation"]["num_steps"],
                           "sequence_length": config["evaluation"]["sequence_length"]}} for row in rows])
    write_json(run_dir / "summary.json", {"teacher": config["model"]["teacher"],
        "teacher_depth": config["sweep"]["teacher_depth"], "initialization": "independent_from_teacher",
        "layer_selection": "uniform_linspace_round_keep_endpoints", "steps_per_student": config["training"]["max_steps"],
        "world_size": distributed.world_size(), "evaluation": config["evaluation"],
        "unified_evaluation": evaluation, "results": rows})
    fields = ["student_depth", "steps", "gen_ppl", "sample_entropy", "num_samples", "checkpoint"]
    if config["sweep"].get("evaluation_protocol") == VERSION:
        fields = ["name", "role", "depth", *fields, "repetition_4gram", "num_scored_samples",
                  "num_empty_samples", "score_status", "evaluation_dir"]
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_depth_curve(run_dir, rows, protocol=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    teacher = next((row for row in rows if row.get("role") == "teacher"), None)
    students = [row for row in rows if row.get("role") != "teacher"]
    if teacher:
        students = sorted(students, key=lambda row: row["student_depth"])
    depths = [row["student_depth"] for row in students]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, label in ((axes[0], "gen_ppl", "GPT-2-Large Gen. PPL"),
                           (axes[1], "sample_entropy", "Sample entropy")):
        values = [row[key] if row[key] is not None else math.nan for row in students]
        ax.plot(depths, values, marker="o", label=f"Student ({students[0]['steps']} steps)")
        ax.set(xlabel="Layers" if teacher else "Student layers", ylabel=label)
        ax.set_xticks(depths + [teacher["depth"]] if teacher else depths)
        if teacher:
            value = teacher[key] if teacher[key] is not None else math.nan
            ax.scatter([teacher["depth"]], [value], marker="*", s=140, color="tab:orange",
                       label=f"Teacher ({teacher['depth']} layers, no training)", zorder=3)
            ax.axhline(value, color="tab:orange", linestyle="--", alpha=0.7)
            # 空文本导致边缘层数的 PPL 为 None 时，也保留该层刻度和曲线缺口。
            ax.set_xlim(min(depths) - 0.4, teacher["depth"] + 0.4)
            ax.legend(fontsize=8)
        else:
            ax.invert_xaxis()
        ax.grid(alpha=0.25)
    title = f"Independent uniform layer pruning — {students[0]['steps']} steps per student"
    fig.suptitle(f"{title}\n{protocol['version']} | seed={protocol['seed']} | {protocol['num_samples']} samples/model"
                 if protocol else title)
    fig.tight_layout()
    path = run_dir / "depth_ppl_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def unified_evaluation_config(run_dir, config, rows, teacher_files):
    generation = {"num_steps": config["evaluation"]["num_steps"],
                  "sequence_length": config["evaluation"]["sequence_length"]}
    models = [{"name": f"teacher_{config['sweep']['teacher_depth']}_layers",
               "architecture": teacher_files[0], "weights": teacher_files[1], "generation": dict(generation)}]
    models.extend({"name": row["name"], "architecture": str(Path(row["checkpoint"]) / "config.json"),
                   "weights": str(Path(row["checkpoint"]) / "model.safetensors"), "generation": dict(generation)}
                  for row in rows)
    result = manifest(models)
    result["run"]["root"] = str(run_dir)
    result["protocol"] = unified_protocol(config)
    result["runtime"].update(device=config["model"]["device"], cache_dir=config["model"]["cache_dir"],
        generation_batch_size=config["evaluation"]["batch_size"],
        control_timeout_seconds=config["distributed"]["control_timeout_seconds"])
    return result


def evaluate_unified_sweep(run_dir, config, rows, teacher_files, allow_distributed, debug):
    evaluation = unified_evaluation_config(run_dir, config, rows, teacher_files) if distributed.is_main() else None
    if distributed.is_main():
        (run_dir / "gen_ppl_models.yaml").write_text(
            yaml.safe_dump(evaluation, allow_unicode=True, sort_keys=False), encoding="utf-8")
    evaluation = distributed.broadcast_main(evaluation)
    try:
        # 直接调用统一工程函数，复用当前八卡通信组；debug 断点仍在同一 Python 进程。
        result_dir = run_gen_ppl(evaluation, allow_distributed=allow_distributed, debug=debug)
    finally:
        progress.configure(run_dir, distributed.rank())
    if distributed.is_main():
        report = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
        teacher = {"name": evaluation["models"][0]["name"], "role": "teacher",
                   "depth": config["sweep"]["teacher_depth"], "steps": 0,
                   "checkpoint": teacher_files[1]}
        rows = [teacher, *rows]
        scores = {result["name"]: result for result in report["models"]}
        for row in rows:
            row.update(scores[row["name"]])
            row["evaluation_dir"] = str(result_dir / row["name"])
        write_summary(run_dir, config, rows, {"directory": str(result_dir), "protocol": report["protocol"]})
        plot_depth_curve(run_dir, rows, report["protocol"])
    distributed.barrier()
    return rows


def run_sweep(config, command="all", allow_distributed=True, debug=False):
    unified = config["sweep"].get("evaluation_protocol") == VERSION
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
                student = build_student(teacher, keep, current["model"].get("block_initialization", "teacher"))
                counts = parameter_counts(student)
                seed_everything(config["run"]["seed"] + distributed.rank())
                distributed.barrier()
                checkpoint = train(teacher, student, tokenizer, current, directory)
                del student
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                metrics = None
                if not unified:
                    # 旧配置继续使用旧评测口径；新配置在全部训练后统一加载七套真实权重。
                    student = load_model(str(checkpoint), device).eval().requires_grad_(False)
                    with progress.phase("sweep.final_ppl", depth=depth, step=current["training"]["max_steps"]):
                        with evaluation_state(student):
                            metrics = evaluate_generation(student, tokenizer, current, directory / "evaluation",
                                step=current["training"]["max_steps"], scorer_cache=scorer_cache, checkpoint=checkpoint)
                    metrics = distributed.broadcast_main(metrics)
                    del student
                if distributed.is_main():
                    row = {"student_depth": depth, "keep_layers": keep,
                        "removed_layers": [i for i in range(teacher.config.n_blocks) if i not in keep],
                        "steps": current["training"]["max_steps"], "parameters": counts,
                        "checkpoint": str(checkpoint)}
                    if unified:
                        row.update(name=current["run"]["name"], role="student", depth=depth)
                    else:
                        row.update(evaluation_dir=str(directory / "evaluation"), gen_ppl=metrics["gen_ppl"],
                                   sample_entropy=metrics["sample_entropy"], num_samples=metrics["num_samples"])
                    rows.append(row)
                    write_summary(run_dir, config, rows)
                    if not unified:
                        plot_depth_curve(run_dir, rows)
                    print(f"[sweep] layers={depth}, checkpoint={checkpoint}, "
                          f"PPL={row.get('gen_ppl', 'pending unified evaluation')}", flush=True)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                distributed.barrier()
                progress.configure(run_dir, distributed.rank())
            if unified:
                # 训练 teacher 与 student 均释放后再推理，GPT-2-Large 不与训练权重争抢显存。
                del teacher
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                settings = config["model"]
                teacher_files = resolve_model_files(settings["teacher"], settings["cache_dir"], settings["teacher_config"])
                with progress.phase("sweep.unified_ppl", student_depths=config["sweep"]["student_depths"]):
                    rows = evaluate_unified_sweep(run_dir, config, rows, teacher_files, allow_distributed, debug)
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "completed", "command": command,
                                                "completed_depths": [row["student_depth"] for row in rows
                                                                     if row.get("role") != "teacher"]})
            print(f"[sweep] results: {run_dir}", flush=True)
        distributed.barrier()
    except Exception as error:
        progress.emit("sweep.failed", error=str(error))
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "failed", "error": str(error),
                                                "completed_depths": [row["student_depth"] for row in rows
                                                                     if row.get("role") != "teacher"]})
        raise
    finally:
        if owns_group:
            distributed.close()
    return run_dir


def main(default_config="configs/owt_depth_sweep.yaml", allow_distributed=True, debug=False):
    parser = argparse.ArgumentParser(description="LangFlow-OWT 独立均匀删层与最终 PPL 曲线；支持统一接口评测 teacher")
    parser.add_argument("command", nargs="?", choices=["all", "prepare"], default="all")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    run_sweep(load_config(args.config, args.overrides, debug), args.command, allow_distributed, debug)


if __name__ == "__main__":
    main()
