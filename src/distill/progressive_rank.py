"""LangFlow-OWT：逐级截断 student、固定原始满秩 Teacher；兼容旧实验元数据。"""

import argparse
import copy
import csv
import gc
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml

from . import distributed, progress
from .backbones.langflow import LangFlow, LangFlowConfig
from .config import create_run_dir, load_yaml_config, project_path, write_json
from .data import prepare_data
from .execution import execution_options, tensor_group_timeout
from .gen_ppl import completed_model_result
from .low_rank import TARGETS, retained_rank, truncate_factors
from .models import load_model, load_teacher, load_tokenizer, resolve_model_files
from .rank_sweep import (checkpoint_entry, evaluation_config, evaluate_stage, find_evaluation_directory,
                         load_config as load_rank_config, rank_name, save_rank_checkpoint)
from .train import seed_everything, train


def rank_schedule(settings):
    start, end, step = (settings[key] for key in ("start_percent", "end_percent", "step_percent"))
    if (any(type(value) is not int for value in (start, end, step))
            or not 0 < end < start <= 100 or step < 1 or (start - end) % step):
        raise ValueError("逐级 rank 必须使用整数百分比，start > end > 0，且步长能整除区间")
    return [percent / 100 for percent in range(start - step, end - 1, -step)]


def teacher_mode(config):
    # 旧 run 没有此字段，必须按旧含义读取，不能静默重解释其 Teacher 链。
    return config["progressive_rank"].get("teacher_mode", "previous_student")


def validate_dense_teacher_config(architecture, config):
    shape = config["rank_sweep"]
    if (architecture.low_rank or architecture.n_blocks != shape["teacher_depth"]
            or architecture.hidden_size != shape["hidden_size"] or architecture.cond_dim != shape["cond_dim"]):
        raise ValueError("fixed_original 模式需要原始完整层数、宽度且未低秩分解的 Teacher")


def validate_supervision_path(teacher, student, config):
    validate_dense_teacher_config(teacher.config, config)
    # 蒸馏两侧查询同一加噪状态，必须共享冻结的 embedding 坐标和噪声 proposal。
    if not torch.equal(teacher._get_embedding_matrix(), student._get_embedding_matrix()):
        raise ValueError("固定 Teacher 与 student 的 embedding 坐标不同")
    proposal = teacher.proposal.state_dict()
    if any(not torch.equal(value, proposal[key]) for key, value in student.proposal.state_dict().items()):
        raise ValueError("固定 Teacher 与 student 的噪声 proposal 不同")


def validate_architecture(architecture, config, ratio):
    shape = config["rank_sweep"]
    if (architecture.n_blocks != shape["teacher_depth"] or architecture.hidden_size != shape["hidden_size"]
            or architecture.cond_dim != shape["cond_dim"]):
        raise ValueError("逐级蒸馏起点必须保持配置中的完整层数、宽度和条件维度")
    expected = {f"backbone.blocks.{index}.{target}": retained_rank(
        (6 * shape["hidden_size"], shape["cond_dim"]) if target == "adaLN_modulation"
        else (shape["hidden_size"], shape["hidden_size"]), ratio)
        for index in range(shape["teacher_depth"]) for target in TARGETS}
    if not architecture.low_rank or architecture.low_rank["ranks"] != expected:
        raise ValueError(f"起点必须对全部层的五个投影按 {ratio:.0%} 原始维度 rank 分解")


def load_config(path, overrides=(), debug=False):
    raw = load_yaml_config(path, overrides)
    ratios = rank_schedule(raw["progressive_rank"])
    config = load_rank_config(path, [*overrides, "rank_sweep.ratios=" + json.dumps(ratios)], debug)
    if teacher_mode(config) not in ("fixed_original", "previous_student"):
        raise ValueError("teacher_mode 只支持 fixed_original / previous_student")
    if (config["model"]["family"] != "langflow" or config["data"].get("corpus", "owt") != "owt"
            or config["rank_sweep"].get("tail_layers") is not None
            or set(config["rank_sweep"]["targets"]) != set(TARGETS)
            or config["rank_sweep"].get("reference_run") or config["rank_sweep"].get("max_steps_by_rank")):
        raise ValueError("逐级实验使用全层 LangFlow-OWT、全部五个投影、统一每级预算")
    if not debug and not config["training"]["early_stopping"]["enabled"]:
        raise ValueError("正式逐级实验需要验证 KL 平台期早停")
    config["execution"] = execution_options(config.get("execution"), debug=debug)
    config["rank_sweep"]["debug"] = debug
    # dry-run 只读真实架构文件，不检查 Git 忽略的权重文件、不下载模型。
    source = project_path(config["progressive_rank"]["source_checkpoint"])
    validate_architecture(LangFlowConfig.from_json_file(source / "config.json"), config,
                          config["progressive_rank"]["start_percent"] / 100)
    return config


def completed_record(path, required):
    # 断电/配额错误可能只写下半个 JSON；不能把“文件存在”当成阶段完成。
    try:
        saved = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return saved if isinstance(saved, dict) and all(key in saved for key in required) else None


def update_step_config(directory, config, *, write=False):
    previous = load_yaml_config(directory / "config.yaml")
    if teacher_mode(previous) != teacher_mode(config):
        raise ValueError("已有实验的 Teacher 模式不能改变，请为固定原始 Teacher 新建实验")
    old_step, new_step = previous["progressive_rank"]["step_percent"], config["progressive_rank"]["step_percent"]
    if old_step == new_step:
        return False
    # 改步长会改变每级 Teacher；只允许在尚未构造任何学生时复用起点评测。
    if any(path.is_dir() for path in directory.glob("rank_*")):
        raise ValueError("已有学生阶段，不能改变其 Teacher 链；请新建使用新步长的实验")
    if write:
        backup = directory / "extensions" / f"step_{new_step:03d}"
        backup.mkdir(parents=True, exist_ok=True)
        if not (backup / "config.yaml").is_file():
            shutil.copyfile(directory / "config.yaml", backup / "config.yaml")
        write_json(backup / "change.json", {"previous_step_percent": old_step, "step_percent": new_step,
            "ratios": config["rank_sweep"]["ratios"], "source_evaluation": "retained; unfinished model is recomputed"})
        temporary = directory / "config.yaml.tmp"
        temporary.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        temporary.replace(directory / "config.yaml")
    return True


def best_checkpoint(directory, config):
    saved = completed_record(directory / "training.json", ("checkpoint", "steps"))
    if saved is not None:
        # 不依赖另一台机器写下的绝对项目路径，按当前 stage 目录定位权重。
        return directory / "checkpoints" / Path(saved["checkpoint"]).name
    name = "best" if config["training"]["early_stopping"]["enabled"] else f"step_{config['training']['max_steps']:06d}"
    return directory / "checkpoints" / name


def stage_config(config, ratio, parent, parent_ratio):
    result = copy.deepcopy(config)
    result["run"]["name"] = rank_name(ratio)
    if teacher_mode(config) == "fixed_original":
        # model.teacher 保留原始模型；上一阶 best 只决定 student 的截断起点。
        result["progressive_rank"].update(parent_checkpoint=str(parent), source_ratio=parent_ratio,
                                         teacher_ratio=1.0, current_ratio=ratio)
    else:
        result["model"].update(teacher=str(parent), teacher_config=None)
        result["progressive_rank"].update(parent_checkpoint=str(parent), teacher_ratio=parent_ratio, current_ratio=ratio)
    return result


@torch.no_grad()
def build_truncated_student(source, ratio, config):
    settings = config["progressive_rank"]
    validate_architecture(source.config, config, settings.get("source_ratio", settings["teacher_ratio"]))
    architecture = copy.deepcopy(source.config)
    previous = architecture.low_rank["ranks"]
    paths = sorted(previous)
    architecture.low_rank = {"ratio": ratio, "method": "svd_sqrt", "rounding": "floor",
        "layer_indices": list(range(architecture.n_blocks)), "ranks": {
            path: retained_rank((source.get_submodule(path).out_features, source.get_submodule(path).in_features), ratio)
            for path in paths}}
    device = next(source.parameters()).device
    student = LangFlow(architecture).to(device)
    weights = student.state_dict()
    replaced = {path + suffix for path in paths for suffix in (".down.weight", ".up.weight")}
    # bias、embedding、时间编码、输出头等全部沿用上一级训练后的值。
    weights.update({key: value for key, value in source.state_dict().items() if key not in replaced})
    student.load_state_dict(weights, strict=True)
    local = []
    # 先让八卡并行完成各自的 SVD，再广播新因子；不把大矩阵塞进 Python object collective。
    for index, path in enumerate(paths):
        if index % distributed.world_size() != distributed.rank():
            continue
        with progress.phase("progressive.svd", module=path, ratio=ratio):
            down, up, report = truncate_factors(source.get_submodule(path), ratio)
            target = student.get_submodule(path)
            target.down.weight.copy_(down)
            target.up.weight.copy_(up)
            local.append({"module": path, **report})
            del down, up
    distributed.barrier()
    if distributed.active():
        for index, path in enumerate(paths):
            target = student.get_submodule(path)
            for weight in (target.down.weight, target.up.weight):
                torch.distributed.broadcast(weight, src=index % distributed.world_size(), group=distributed.tensor_group())
    shards = distributed.gather_main(local)
    student.backbone.vocab_embed.requires_grad_(False)
    student.proposal.requires_grad_(False)
    return student, sorted((row for shard in shards for row in shard), key=lambda row: row["module"]) if shards else None


def save_progressive_checkpoint(student, tokenizer, run_dir, step, config, name=None):
    checkpoint = save_rank_checkpoint(student, tokenizer, run_dir, step, config, name)
    if distributed.is_main():
        path = checkpoint / "distillation.json"
        metadata = json.loads(path.read_text())
        metadata.update(initialization="progressive_svd", progressive_rank=config["progressive_rank"])
        write_json(path, metadata)
    distributed.barrier()
    return checkpoint


def evaluation_complete(config, entry, directory, stage, debug):
    evaluation = evaluation_config(config, [entry], directory, debug)
    child = find_evaluation_directory(directory, stage, evaluation)
    return child is not None and completed_model_result(child / entry["name"], config["gradient_moment"]["enabled"]) is not None


def save_results(run_dir, config, rows, entries):
    mode = teacher_mode(config)
    write_json(run_dir / "summary.json", {"config": config, "results": rows,
        "teacher_mode": mode,
        "method": ("Fixed original dense Teacher; each student is SVD-truncated from the preceding best student and re-distilled off-policy with KL."
                   if mode == "fixed_original" else "Each teacher is the frozen best checkpoint of the previous rank; truncated student is re-distilled off-policy with KL."),
        "rank_definition": "Absolute floor(ratio * min(original matrix dimensions)); all blocks and five projections.",
        "timing_note": "Per-batch generator forward latency; compilation/warmup, GPT scoring and GM excluded."})
    fields = ["name", "phase", "rank_ratio", "teacher_mode", "teacher_ratio", "teacher_checkpoint", "source_ratio", "parent_checkpoint", "depth", "rank", "conditioning_rank",
        "gen_ppl", "sample_entropy", "repetition_4gram", "gm", "gm_standard_error", "num_parameters", "mean_step_ms",
        "effective_generation_batch_size", "execution_backend", "compile_mode", "compile_target", "cuda_graphs_requested",
        "cuda_graphs_verified", "compile_warmup_seconds", "training_backend", "training_compile_mode",
        "num_samples", "num_scored_samples", "num_empty_samples", "gm_num_pairs", "gm_num_skipped_pairs", "gm_status",
        "training_max_steps", "training_steps", "best_step", "tokens_seen", "checkpoint_tokens_seen", "stop_reason"]
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "gen_ppl_models.yaml").write_text(yaml.safe_dump(evaluation_config(config, entries, run_dir,
        config["rank_sweep"]["debug"]), sort_keys=False), encoding="utf-8")
    if config["rank_sweep"]["plot_results"]:
        plot_results(run_dir, rows, mode=mode)


def result_row(row, config, phase, ratio, parent=None, parent_ratio=None, directory=None):
    training = json.loads((directory / "training.json").read_text()) if phase == "after" else {}
    fixed = teacher_mode(config) == "fixed_original"
    return {**row, "phase": phase, "rank_ratio": ratio, "teacher_mode": teacher_mode(config),
        "teacher_ratio": (1.0 if fixed else parent_ratio) if phase != "source" else None,
        "teacher_checkpoint": (str(config["model"]["teacher"]) if fixed else str(parent)) if phase != "source" else None,
        "source_ratio": parent_ratio,
        "parent_checkpoint": str(parent) if parent is not None else None, "depth": config["rank_sweep"]["teacher_depth"],
        "rank": retained_rank((config["rank_sweep"]["hidden_size"],) * 2, ratio),
        "conditioning_rank": retained_rank((config["rank_sweep"]["cond_dim"],) * 2, ratio),
        "training_max_steps": training.get("max_steps", config["training"]["max_steps"]) if phase != "source" else None,
        "training_backend": training.get("execution_backend", "eager") if phase == "after" else None,
        "training_compile_mode": training.get("compile_mode"),
        **{key: training.get(key) for key in ("best_step", "stop_reason", "tokens_seen", "checkpoint_tokens_seen")},
        "training_steps": training.get("steps")}


def plot_results(run_dir, rows, *, mode="previous_student"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, label in zip(axes, ("gen_ppl", "sample_entropy", "gm"), ("Gen PPL", "Unigram entropy (nats)", "GM")):
        for phase, title in (("before", "SVD truncated"), ("after", "after KL")):
            selected = sorted((row for row in rows if row["phase"] == phase), key=lambda row: -row["rank_ratio"])
            if selected:
                ax.plot([100 * row["rank_ratio"] for row in selected],
                    [row.get(key) if row.get(key) is not None else float("nan") for row in selected], marker="o", label=title)
        source = next((row for row in rows if row["phase"] == "source"), None)
        if source and source.get(key) is not None:
            ax.scatter([100 * source["rank_ratio"]], [source[key]], marker="*", color="black", s=80, label="source student")
        if key == "gen_ppl":
            ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set(xlabel="Retained rank (%)", ylabel=label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("LangFlow-OWT: progressive rank, " + ("fixed original dense Teacher"
                 if mode == "fixed_original" else "previous best student as teacher"))
    fig.tight_layout()
    fig.savefig(run_dir / "progressive_rank_quality_curve.png", dpi=160)
    plt.close(fig)


def release_models():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_progressive(config, command="all", allow_distributed=True, debug=False, resume_dir=None):
    mode = teacher_mode(config)
    if config["rank_sweep"]["plot_results"] or config["training"]["plot_loss"]:
        from importlib.util import find_spec
        if find_spec("matplotlib") is None:
            raise RuntimeError("绘图需要 matplotlib；请先在服务器安装项目依赖")
    owns_group = distributed.setup(config["model"]["device"], allow_distributed, config["distributed"]["control_timeout_seconds"])
    run_dir = Path(distributed.broadcast_main(str(project_path(resume_dir) if resume_dir else create_run_dir(config))
                                             if distributed.is_main() else None))
    progress.configure(run_dir, distributed.rank())
    rows, entries = [], []
    try:
        if not debug and command == "all" and distributed.world_size() != 8:
            raise RuntimeError("正式逐级实验使用八卡，请通过 scripts/run_owt_progressive_rank.sh 启动")
        if distributed.is_main():
            if resume_dir:
                update_step_config(run_dir, config, write=True)
            write_json(run_dir / "status.json", {"status": "running", "command": command,
                                                "resumed": bool(resume_dir), "teacher_mode": mode})
        tokenizer = load_tokenizer(config) if distributed.is_main() else None
        if distributed.is_main():
            prepare_data(config, tokenizer)
        distributed.barrier()
        if command == "prepare":
            if distributed.is_main():
                write_json(run_dir / "status.json", {"status": "completed", "command": command})
            return run_dir
        if not distributed.is_main():
            tokenizer = load_tokenizer(config)
        distributed.setup_tensor_group(config["model"]["device"], tensor_group_timeout(config))
        torch.set_float32_matmul_precision("highest")
        device = torch.device("cuda", torch.cuda.current_device())
        if mode == "fixed_original":
            # 长评测之前核对原始 Teacher 架构；SVD 起点仍由独立 source_checkpoint 指定。
            architecture, _ = resolve_model_files(config["model"]["teacher"], config["model"]["cache_dir"],
                                                   config["model"]["teacher_config"])
            validate_dense_teacher_config(LangFlowConfig.from_json_file(architecture), config)
        parent = project_path(config["progressive_rank"]["source_checkpoint"])
        parent_ratio = config["progressive_rank"]["start_percent"] / 100
        entry = checkpoint_entry(config, parent, rank_name(parent_ratio) + "_source")
        report = evaluate_stage(config, [entry], run_dir, "source", allow_distributed, debug)
        rows.append(result_row(report["models"][0], config, "source", parent_ratio))
        entries.append(entry)
        source_eval = find_evaluation_directory(run_dir, "source", evaluation_config(config, [entry], run_dir, debug))
        gm_reference = source_eval / "gradient_moment/reference_owt.json" if config["gradient_moment"]["enabled"] else None
        if distributed.is_main():
            save_results(run_dir, config, rows, entries)
        for ratio in config["rank_sweep"]["ratios"]:
            directory = run_dir / rank_name(ratio)
            current = stage_config(config, ratio, parent, parent_ratio)
            if distributed.is_main():
                directory.mkdir(parents=True, exist_ok=True)
                saved_config = directory / "config.yaml"
                if saved_config.is_file() and load_yaml_config(saved_config) != current:
                    raise ValueError(f"本级设置或上游 Teacher 与保存记录不同，不能混用已有结果: {directory}")
                saved_config.write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
                write_json(run_dir / "status.json", {"status": "running", "current_rank": ratio,
                    "source_rank": parent_ratio, "teacher_rank": current["progressive_rank"]["teacher_ratio"],
                    "teacher_mode": mode})
            distributed.barrier()
            progress.configure(directory, distributed.rank())
            progress.emit("progressive.stage", student_rank=ratio, svd_source=str(parent),
                          teacher=current["model"]["teacher"], teacher_mode=mode)
            initial = directory / "checkpoints/initial"
            ready = distributed.broadcast_main(completed_record(directory / "truncation.json",
                ("source_checkpoint", "ratio", "matrices")) is not None and (initial / "model.safetensors").is_file()
                and (initial / "config.json").is_file() if distributed.is_main() else None)
            if not ready:
                if distributed.is_main():
                    (directory / "truncation.json").unlink(missing_ok=True)
                distributed.barrier()
                seed_everything(config["run"]["seed"] + distributed.rank())
                # 上一阶 student 只提供初始化；蒸馏 Teacher 在训练分支单独加载。
                source = load_model(str(parent), device).requires_grad_(False)
                student, matrices = build_truncated_student(source, ratio, current)
                save_progressive_checkpoint(student, tokenizer, directory, 0, current, name="initial")
                if distributed.is_main():
                    write_json(directory / "truncation.json", {"source_checkpoint": str(parent), "source_ratio": parent_ratio,
                        "ratio": ratio, "method": "SVD of learned up @ down", "matrices": matrices})
                del student, source
                release_models()
                distributed.barrier()
            entry = checkpoint_entry(config, initial, rank_name(ratio) + "_initial")
            report = evaluate_stage(config, [entry], directory, "before", allow_distributed, debug, gm_reference=gm_reference)
            rows.append(result_row(report["models"][0], config, "before", ratio, parent, parent_ratio))
            entries.append(entry)
            if distributed.is_main():
                save_results(run_dir, config, rows, entries)
            done = distributed.broadcast_main(completed_record(directory / "training.json", ("checkpoint", "steps"))
                                             is not None if distributed.is_main() else None)
            if not done:
                progress.configure(directory, distributed.rank())
                teacher = load_teacher(current)
                student = load_model(str(initial), device)
                student.backbone.vocab_embed.requires_grad_(False)
                student.proposal.requires_grad_(False)
                if mode == "fixed_original":
                    validate_supervision_path(teacher, student, current)
                seed_everything(config["run"]["seed"] + distributed.rank())
                train(teacher, student, tokenizer, current, directory, checkpoint_fn=save_progressive_checkpoint)
                del student, teacher
                release_models()
            checkpoint = best_checkpoint(directory, config)
            entry = checkpoint_entry(config, checkpoint, rank_name(ratio) + "_trained")
            report = evaluate_stage(config, [entry], directory, "after", allow_distributed, debug, gm_reference=gm_reference)
            rows.append(result_row(report["models"][0], config, "after", ratio, parent, parent_ratio, directory))
            entries.append(entry)
            if distributed.is_main():
                save_results(run_dir, config, rows, entries)
            distributed.barrier()
            # 逐级推进 student 的 SVD 起点；固定模式下 model.teacher 始终保持原始模型。
            parent, parent_ratio = checkpoint, ratio
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "completed", "ratios": config["rank_sweep"]["ratios"],
                "source_checkpoint": config["progressive_rank"]["source_checkpoint"], "final_checkpoint": str(parent),
                "teacher_mode": mode})
        distributed.barrier()
    except Exception as error:
        try:
            if distributed.is_main():
                write_json(run_dir / "status.json", {"status": "failed", "error": str(error)})
        except OSError as logging_error:
            print(f"[progressive-rank] 保存失败记录时出错: {logging_error}", flush=True)
        # 截断/写盘失败也可能让其他 rank 等待 barrier；交由 torchrun 结束本作业。
        distributed.abort_worker(error, run_dir, reason="oom" if isinstance(error, torch.OutOfMemoryError) else "execution")
        raise
    finally:
        if owns_group:
            distributed.close()
    return run_dir


def dry_run_plan(config, run_dir=None):
    fixed = teacher_mode(config) == "fixed_original"
    root = project_path(run_dir) if run_dir else project_path(config["run"]["root"]) / "<new_progressive_run>"
    parent = project_path(config["progressive_rank"]["source_checkpoint"])
    previous_ratio = config["progressive_rank"]["start_percent"] / 100
    stages = []
    for ratio in config["rank_sweep"]["ratios"]:
        directory = root / rank_name(ratio)
        initial = checkpoint_entry(config, directory / "checkpoints/initial", rank_name(ratio) + "_initial")
        trained = checkpoint_entry(config, best_checkpoint(directory, config), rank_name(ratio) + "_trained")
        stages.append({"name": rank_name(ratio), "ratio": ratio, "teacher_ratio": 1.0 if fixed else previous_ratio,
            "teacher_checkpoint": str(config["model"]["teacher"]) if fixed else str(parent),
            "svd_source": str(parent), "source_ratio": previous_ratio,
            "rank": retained_rank((config["rank_sweep"]["hidden_size"],) * 2, ratio),
            "conditioning_rank": retained_rank((config["rank_sweep"]["cond_dim"],) * 2, ratio),
            "training_completed": completed_record(directory / "training.json", ("checkpoint", "steps")) is not None,
            "before_evaluation_completed": evaluation_complete(config, initial, directory, "before", config["rank_sweep"]["debug"]) if run_dir else False,
            "after_evaluation_completed": evaluation_complete(config, trained, directory, "after", config["rank_sweep"]["debug"]) if run_dir else False})
        parent, previous_ratio = best_checkpoint(directory, config), ratio
    source = checkpoint_entry(config, project_path(config["progressive_rank"]["source_checkpoint"]),
                              rank_name(config["progressive_rank"]["start_percent"] / 100) + "_source")
    source_complete = evaluation_complete(config, source, root, "source", config["rank_sweep"]["debug"]) if run_dir else False
    return {"experiment": "progressive_svd_rank", "run_name": config["run"]["name"],
        "teacher_mode": teacher_mode(config), "teacher": "fixed original dense Teacher" if fixed else "previous rank best student",
        "source_checkpoint": config["progressive_rank"]["source_checkpoint"], "step_percent": config["progressive_rank"]["step_percent"],
        "student_depth": config["rank_sweep"]["teacher_depth"],
        "targets": config["rank_sweep"]["targets"], "matrices_per_stage": 5 * config["rank_sweep"]["teacher_depth"],
        "stages": stages, "loss": config["loss"], "training": config["training"], "evaluation": config["evaluation"], "execution": config["execution"],
        "gradient_moment": config["gradient_moment"], "total_evaluation_models": 1 + 2 * len(stages),
        "source_evaluation_completed": source_complete,
        "remaining_training_stages": sum(not stage["training_completed"] for stage in stages),
        "remaining_evaluation_models": int(not source_complete) + sum(
            int(not stage[phase + "_evaluation_completed"]) for stage in stages for phase in ("before", "after")),
        "formal_world_size": 8, "global_batch_size_8gpu": 8 * config["training"]["batch_size"] * config["training"]["grad_accumulation"],
        "scorer_execution": "float32/eager unchanged", "resume_granularity": "completed SVD/model evaluation/training stages; incomplete training restarts initial"}


def main(default_config="configs/owt_progressive_rank.yaml", allow_distributed=True, debug=False):
    parser = argparse.ArgumentParser(description="LangFlow-OWT：逐级初始化 student，固定原始满秩 Teacher 蒸馏")
    parser.add_argument("command", nargs="?", choices=("all", "prepare"), default="all")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--run-dir", help="复用已完成级别/评测；未完成训练从本级 SVD initial 重训")
    parser.add_argument("--step-percent", type=int, help="rank 百分比步长；恢复时仅允许在尚无学生阶段的目录改变")
    parser.add_argument("--teacher-mode", choices=("fixed_original", "previous_student"),
                        help="要求配置/续跑目录使用此 Teacher 模式；不转换旧实验")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.run_dir and args.overrides:
        parser.error("逐级链恢复使用保存的完整配置；更改实验条件请另开目录")
    if args.run_dir and load_yaml_config(project_path(args.run_dir) / "config.yaml")["rank_sweep"].get("debug", False) != debug:
        parser.error("debug 与正式目录不能混用，请使用创建该实验时的入口恢复")
    overrides = list(args.overrides)
    if args.step_percent is not None:
        overrides.append(f"progressive_rank.step_percent={args.step_percent}")
    config = load_config(project_path(args.run_dir) / "config.yaml" if args.run_dir else args.config, overrides, debug)
    if args.teacher_mode is not None and teacher_mode(config) != args.teacher_mode:
        parser.error("当前目录的 Teacher 模式与入口要求不同；固定原始 Teacher 需要新建实验，不能续用旧的学生 Teacher 链")
    if args.run_dir:
        update_step_config(project_path(args.run_dir), config)
    if args.dry_run:
        plan = dry_run_plan(config, args.run_dir)
        output = project_path("run/launch") / f"progressive_rank_plan_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        write_json(output, plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        print(f"[progressive-rank] plan: {output}", flush=True)
        return
    directory = run_progressive(config, args.command, allow_distributed, debug, args.run_dir)
    print(f"[progressive-rank] results: {directory}", flush=True)
    return directory


if __name__ == "__main__":
    main()
