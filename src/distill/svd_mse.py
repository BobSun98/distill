"""共享原始 Teacher SVD 起点的全层40%/60%学生，独立比较纯 embedding/velocity MSE。"""

import argparse
import copy
import csv
import gc
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import yaml

from . import distributed, progress
from .config import create_run_dir, load_config as load_training_config, load_yaml_config, project_path, write_json
from .execution import execution_options
from .gen_ppl import completed_model_result
from .low_rank import TARGETS, retained_rank
from .models import load_student_checkpoint, load_teacher, load_tokenizer, resolve_model_files
from .rank_sweep import (checkpoint_entry, conditioning_rank, evaluate_stage, evaluation_config,
                         run_sweep, save_rank_checkpoint, seed_everything, student_specs)
from .train import train


def pure_loss(objective):
    if objective not in ("embedding", "velocity"):
        raise ValueError("纯 MSE 目标只支持 embedding / velocity")
    return {"kd_weight": 0.0, "ce_weight": 0.0,
            "mse_weight": float(objective == "velocity"), "embedding_weight": float(objective == "embedding")}


def load_config(path, overrides=(), debug=False):
    config = load_training_config(path, overrides)
    sweep, objectives = config["rank_sweep"], config["svd_mse"]["objectives"]
    if (not objectives or len(set(objectives)) != len(objectives)
            or not set(objectives).issubset({"embedding", "velocity"})):
        raise ValueError("svd_mse.objectives 必须为不重复的 embedding/velocity 列表")
    if (not sweep["ratios"] or len(set(sweep["ratios"])) != len(sweep["ratios"])
            or not set(sweep["ratios"]).issubset({0.4, 0.6})):
        raise ValueError("本实验只运行40%和60% rank，可选择其中一组")
    if (config["model"]["family"] != "langflow" or config["data"].get("corpus", "owt") != "owt"
            or sweep["teacher_depth"] != 12 or sweep["hidden_size"] != 768 or sweep["cond_dim"] != 128
            or config["model"]["keep_layers"] != list(range(12))
            or len(sweep["targets"]) != 5 or set(sweep["targets"]) != set(TARGETS)
            or sweep.get("tail_layers") is not None or sweep.get("reference_run") or sweep.get("max_steps_by_rank")):
        raise ValueError("本实验使用全12层 LangFlow-OWT、全部五个投影、统一训练预算")
    if config["model"].get("student_weights") or config["model"].get("block_initialization", "teacher") != "teacher":
        raise ValueError("本实验必须从原始 Teacher 新做 SVD，不能使用训练后的 student 或随机 blocks")
    if config["loss"] not in (pure_loss("embedding"), pure_loss("velocity")):
        raise ValueError("本实验只使用系数1的纯 MSE，KL/CE系数必须为0；用 --objective 选择目标")
    config["loss"] = pure_loss(objectives[0])
    if config["training"]["ppl_evaluation"]["enabled"] or config["training"]["early_stopping"]["metric"] != "loss":
        raise ValueError("关闭训练中间 PPL，并按本组纯 MSE loss 早停")
    deltas = config["svd_mse"].get("min_delta", {})
    if not isinstance(deltas, dict) or any(not math.isfinite(deltas.get(kind, config["training"]["early_stopping"]["min_delta"]))
            or deltas.get(kind, config["training"]["early_stopping"]["min_delta"]) < 0 for kind in objectives):
        raise ValueError("各目标的 min_delta 必须为有限非负数")
    if not debug and config["evaluation"]["num_samples"] != 1024:
        raise ValueError("正式统一评测固定1024条；小预算请使用单卡debug入口")
    evaluation = config["evaluation"]
    if (evaluation["num_steps"] < 2 or not 1 <= evaluation["sequence_length"] <= 1024
            or evaluation["batch_size"] < 1 or evaluation["timing_warmup_forwards"] < 0):
        raise ValueError("生成长度/步数/batch/预热设置无效")
    config["execution"] = execution_options(config.get("execution"), debug=debug)
    config["rank_sweep"]["debug"] = debug
    evaluation_config(config, [{"name": "teacher", "corpus": "owt"}], "run", debug)
    return config


def jobs(config):
    return [{**spec, "name": f"{objective}_{spec['name']}", "initial_name": spec["name"],
             "objective": objective, "loss": pure_loss(objective)}
            for objective in config["svd_mse"]["objectives"] for spec in student_specs(config)]


def job_config(config, job, root):
    current = copy.deepcopy(config)
    current["run"]["name"] = job["name"]
    current["loss"] = dict(job["loss"])
    current["model"]["student_weights"] = str(root / job["initial_name"] / "checkpoints/initial/model.safetensors")
    current["svd_mse"].update(current_objective=job["objective"], current_ratio=job["ratio"])
    current["training"]["early_stopping"]["min_delta"] = config["svd_mse"].get("min_delta", {}).get(
        job["objective"], config["training"]["early_stopping"]["min_delta"])
    return current


def training_record(directory):
    try:
        record = json.loads((directory / "training.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) and all(key in record for key in ("checkpoint", "steps")) else None


def evaluation_directory(root, stage):
    receipt = root / f"{stage}_evaluation.json"
    if not receipt.is_file():
        return None
    path = Path(json.loads(receipt.read_text())["directory"])
    local = root / path.name
    return local if local.is_dir() else path


def save_results(root, config, reports, entries):
    specs = {spec["name"] + "_initial": spec for spec in student_specs(config)}
    specs.update({job["name"] + "_trained": job for job in jobs(config)})
    rows = []
    for phase, report in reports.items():
        for raw in report["models"]:
            teacher = raw["name"] == "teacher"
            spec = None if teacher else specs[raw["name"]]
            trained = phase != "before"
            record = training_record(root / spec["name"]) if trained else {}
            ratio = 1.0 if teacher else spec["ratio"]
            rows.append({**raw, "phase": "teacher" if teacher else "after" if trained else "before",
                "objective": spec["objective"] if trained else None, "rank_ratio": ratio,
                "depth": config["rank_sweep"]["teacher_depth"],
                "rank": None if teacher else retained_rank((config["rank_sweep"]["hidden_size"],) * 2, ratio),
                "conditioning_rank": None if teacher else conditioning_rank(config, ratio),
                "initialization": "teacher" if teacher else "fresh_teacher_svd", "initial_training_steps": 0,
                **{key: spec["loss"][key] if trained else None for key in pure_loss("embedding")},
                "training_steps": record.get("steps"), "best_step": record.get("best_step"),
                "training_max_steps": record.get("max_steps") if trained else None,
                "best_value": record.get("best_value"), "stop_reason": record.get("stop_reason"),
                "training_backend": record.get("execution_backend") if trained else None,
                "early_stopping_min_delta": record.get("early_stopping", {}).get("min_delta"),
                "tokens_seen": record.get("tokens_seen"), "checkpoint_tokens_seen": record.get("checkpoint_tokens_seen")})
    write_json(root / "summary.json", {"config": config, "results": rows,
        "initialization_note": "Both objectives start independently from the same per-rank Teacher SVD step-0 checkpoint; no KL warmup.",
        "loss_note": "KL/CE diagnostic values may be logged, but their coefficients are zero; best is selected by pure MSE loss."})
    fields = ["name", "phase", "objective", "rank_ratio", "depth", "rank", "conditioning_rank", "initialization",
        "initial_training_steps", "kd_weight", "ce_weight", "mse_weight", "embedding_weight",
        "gen_ppl", "sample_entropy", "repetition_4gram", "gm", "gm_standard_error", "num_parameters", "mean_step_ms",
        "execution_backend", "compile_mode", "compile_target", "cuda_graphs_requested", "cuda_graphs_verified",
        "num_samples", "num_scored_samples", "num_empty_samples", "gm_num_pairs", "gm_num_skipped_pairs", "gm_status",
        "training_steps", "best_step", "training_max_steps", "best_value", "stop_reason", "training_backend",
        "early_stopping_min_delta", "tokens_seen", "checkpoint_tokens_seen", "weights"]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (root / "gen_ppl_models.yaml").write_text(yaml.safe_dump(evaluation_config(
        config, entries, root, config["rank_sweep"]["debug"]), sort_keys=False), encoding="utf-8")


def release_models():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_experiment(config, command="all", allow_distributed=True, debug=False, resume_dir=None):
    if command == "all" and config["training"]["plot_loss"]:
        from importlib.util import find_spec
        if find_spec("matplotlib") is None:
            raise RuntimeError("训练绘图需要 matplotlib；请先在服务器安装项目依赖")
    owns_group = distributed.setup(config["model"]["device"], allow_distributed,
                                    config["distributed"]["control_timeout_seconds"])
    root = Path(distributed.broadcast_main(str(project_path(resume_dir) if resume_dir else create_run_dir(config))
                                          if distributed.is_main() else None))
    progress.configure(root, distributed.rank())
    reports, entries = {}, []
    try:
        if not debug and command != "prepare" and distributed.world_size() != 8:
            raise RuntimeError("正式纯MSE实验使用八卡，请通过 scripts/run_owt_svd_pure_mse.sh 启动")
        # 复用真实SVD工程构造共享初始权重；外层持有通信组，后续不重复创建NCCL组。
        # root已保存同一份配置，build使用已有目录，完全不执行KL或其他训练。
        run_sweep(config, "prepare" if command == "prepare" else "build", allow_distributed, debug, resume_dir=root)
        if command in ("prepare", "build"):
            return root
        if distributed.is_main():
            write_json(root / "status.json", {"status": "running", "stage": "initial_evaluation", "resumed": bool(resume_dir)})
        device = torch.device("cuda", torch.cuda.current_device())
        tokenizer = load_tokenizer(config)
        files = resolve_model_files(config["model"]["teacher"], config["model"]["cache_dir"], config["model"]["teacher_config"])
        teacher_entry = {"name": "teacher", "corpus": "owt", "architecture": files[0], "weights": files[1],
            "generation": {key: config["evaluation"][key] for key in ("num_steps", "sequence_length")}}
        entries = [teacher_entry, *[checkpoint_entry(config, root / spec["name"] / "checkpoints/initial",
                                                   spec["name"] + "_initial") for spec in student_specs(config)]]
        reports["before"] = evaluate_stage(config, entries, root, "before", allow_distributed, debug)
        before_directory = evaluation_directory(root, "before")
        gm_reference = before_directory / "gradient_moment/reference_owt.json" if config["gradient_moment"]["enabled"] else None
        if distributed.is_main():
            save_results(root, config, reports, entries)
        for objective in config["svd_mse"]["objectives"]:
            teacher, final_entries = None, []
            for job in (job for job in jobs(config) if job["objective"] == objective):
                directory = root / job["name"]
                current = job_config(config, job, root)
                if distributed.is_main():
                    directory.mkdir(parents=True, exist_ok=True)
                    saved = directory / "config.yaml"
                    if saved.is_file() and load_yaml_config(saved) != current:
                        raise ValueError(f"训练记录设置不同，不能混用已有结果: {directory}")
                    saved.write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
                    write_json(root / "status.json", {"status": "running", "stage": "training", "job": job["name"]})
                distributed.barrier()
                record = training_record(directory)
                if record is None:
                    if teacher is None:
                        teacher = load_teacher(config)
                    progress.configure(directory, distributed.rank())
                    student = load_student_checkpoint(teacher, current)
                    if distributed.is_main():
                        write_json(directory / "initialization.json", {"teacher": config["model"]["teacher"],
                            "source": current["model"]["student_weights"], "source_step": 0,
                            "initialization": "fresh_teacher_svd", "objective": objective, "loss": current["loss"]})
                    seed_everything(config["run"]["seed"] + distributed.rank())
                    train(teacher, student, tokenizer, current, directory, checkpoint_fn=save_rank_checkpoint)
                    del student
                    release_models()
                    record = training_record(directory)
                checkpoint = directory / "checkpoints" / Path(record["checkpoint"]).name
                final_entries.append(checkpoint_entry(config, checkpoint, job["name"] + "_trained"))
            del teacher
            release_models()
            progress.configure(root, distributed.rank())
            if distributed.is_main():
                write_json(root / "status.json", {"status": "running", "stage": objective + "_evaluation"})
            reports[objective] = evaluate_stage(config, final_entries, root, objective, allow_distributed, debug,
                                                 gm_reference=gm_reference)
            entries.extend(final_entries)
            if distributed.is_main():
                save_results(root, config, reports, entries)
            distributed.barrier()
        if distributed.is_main():
            write_json(root / "status.json", {"status": "completed", "jobs": [job["name"] for job in jobs(config)],
                "initialization": "fresh_teacher_svd", "teacher": config["model"]["teacher"],
                "initial_training_steps": 0, "world_size": distributed.world_size()})
        distributed.barrier()
    except Exception as error:
        try:
            if distributed.is_main():
                write_json(root / "status.json", {"status": "failed", "error": str(error)})
        except OSError as logging_error:
            print(f"[svd-mse] 保存失败记录时出错: {logging_error}", flush=True)
        distributed.abort_worker(error, root, reason="oom" if isinstance(error, torch.OutOfMemoryError) else "execution")
        raise
    finally:
        if owns_group:
            distributed.close()
    return root


def dry_run_plan(config, resume_dir=None):
    root = project_path(resume_dir) if resume_dir else project_path(config["run"]["root"]) / "<new_svd_pure_mse_run>"
    planned = []
    for job in jobs(config):
        current = job_config(config, job, root)
        planned.append({**job, "student_weights": current["model"]["student_weights"], "source_step": 0,
            "rank": retained_rank((config["rank_sweep"]["hidden_size"],) * 2, job["ratio"]),
            "conditioning_rank": conditioning_rank(config, job["ratio"]),
            "early_stopping": current["training"]["early_stopping"],
            "training_completed": training_record(root / job["name"]) is not None})
    completed = []
    if resume_dir:
        for stage in ("before", *config["svd_mse"]["objectives"]):
            directory = evaluation_directory(root, stage)
            if directory is not None:
                names = (["teacher", *[spec["name"] + "_initial" for spec in student_specs(config)]] if stage == "before"
                         else [job["name"] + "_trained" for job in jobs(config) if job["objective"] == stage])
                completed.extend(name for name in names if completed_model_result(directory / name, config["gradient_moment"]["enabled"]) is not None)
    total = 1 + len(config["rank_sweep"]["ratios"]) + len(planned)
    return {"experiment": "fresh_teacher_svd_pure_mse", "teacher": config["model"]["teacher"],
        "initialization": "Teacher -> FP32 SVD -> shared per-rank step-0 checkpoint", "student_depth": 12,
        "num_svd_matrices": 60, "jobs": planned, "training": config["training"], "evaluation": config["evaluation"],
        "gradient_moment": config["gradient_moment"], "execution": config["execution"], "formal_world_size": 8,
        "global_batch_size_8gpu": 8 * config["training"]["batch_size"] * config["training"]["grad_accumulation"],
        "evaluation_models_total": total, "completed_evaluation_models": completed,
        "remaining_evaluation_models": total - len(completed),
        "scorer_execution": "float32/eager unchanged", "diagnostic_kl": "logged only; coefficient zero; no KL warmup"}


def main(default_config="configs/owt_svd_pure_mse.yaml", allow_distributed=True, debug=False):
    parser = argparse.ArgumentParser(description="原始Teacher SVD 40%/60% -> pure embedding / velocity MSE ->统一评测")
    parser.add_argument("command", nargs="?", choices=("all", "prepare", "build"), default="all")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--objective", choices=("embedding", "velocity"))
    parser.add_argument("--rank", choices=(40, 60), type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.run_dir and (args.overrides or args.objective or args.rank):
        parser.error("恢复使用保存的四组配置；修改目标/网格/预算请另开实验")
    overrides = list(args.overrides)
    if args.objective:
        overrides.append("svd_mse.objectives=" + json.dumps([args.objective]))
    if args.rank:
        overrides.append("rank_sweep.ratios=" + json.dumps([args.rank / 100]))
    path = project_path(args.run_dir) / "config.yaml" if args.run_dir else args.config
    if args.run_dir and load_yaml_config(path)["rank_sweep"].get("debug", False) != debug:
        parser.error("debug与正式目录不能混用")
    config = load_config(path, overrides, debug)
    if args.dry_run:
        plan = dry_run_plan(config, args.run_dir)
        output = project_path("run/launch") / f"svd_pure_mse_plan_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        write_json(output, plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        print(f"[svd-mse] plan: {output}", flush=True)
        return
    root = run_experiment(config, args.command, allow_distributed, debug, args.run_dir)
    print(f"[svd-mse] results: {root}", flush=True)
    return root


if __name__ == "__main__":
    main()
