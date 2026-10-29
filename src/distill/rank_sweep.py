"""同深度 LangFlow-OWT：Teacher SVD → 统一初评 → KL 蒸馏 → 统一复评。"""

import argparse
import copy
import csv
import gc
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml

from . import distributed, gradient_moment, progress
from .backbones.langflow import LangFlow
from .config import create_run_dir, load_config as load_training_config, load_yaml_config, project_path, write_json
from .data import load_tokens, prepare_data
from .gen_ppl import completed_model_result, evaluation_matches, run_gen_ppl, validate_protocol
from .gen_ppl_protocol import MAX_LENGTH, NUM_SAMPLES, manifest
from .low_rank import TARGETS, balanced_factors, retained_rank, target_linears
from .losses import distillation_loss
from .models import load_model, load_teacher, load_tokenizer, resolve_model_files
from .states import off_policy_state, posterior_logits
from .train import save_checkpoint, seed_everything, train


def rank_name(ratio):
    return f"rank_{round(ratio * 100):03d}"


def conditioning_rank(config, ratio):
    sweep = config["rank_sweep"]
    if "adaLN_modulation" not in sweep["targets"]:
        return None
    # AdaLN 权重是 [6 * hidden_size, cond_dim]，不能误用主干的 768 维 rank。
    return retained_rank((6 * sweep["hidden_size"], sweep.get("cond_dim", 128)), ratio)


def evaluation_config(config, entries, root, debug=False):
    result = manifest(entries)
    result["run"]["root"] = str(root)
    result["gradient_moment"].update(config["gradient_moment"])
    result["runtime"].update(device=config["model"]["device"], cache_dir=config["model"]["cache_dir"],
        generation_batch_size=config["evaluation"]["batch_size"],
        timing_warmup_forwards=config["evaluation"]["timing_warmup_forwards"],
        control_timeout_seconds=config["distributed"]["control_timeout_seconds"])
    if debug:
        result["protocol"]["num_samples"] = config["evaluation"]["num_samples"]
    validate_protocol(result["protocol"], debug)
    gradient_moment.validate_config(result)
    return result


def load_config(path, overrides=(), debug=False):
    config = load_training_config(path, overrides)
    sweep, training, evaluation = config["rank_sweep"], config["training"], config["evaluation"]
    ratios = sweep["ratios"]
    if (not ratios or len(set(ratios)) != len(ratios)
            or len({rank_name(r) for r in ratios}) != len(ratios)):
        raise ValueError("rank ratios 必须非空、不重复，且百分比目录名不能冲突")
    for ratio in ratios:
        retained_rank((sweep["hidden_size"], sweep["hidden_size"]), ratio)
    if (not sweep["targets"] or len(set(sweep["targets"])) != len(sweep["targets"])
            or not set(sweep["targets"]).issubset(TARGETS)):
        raise ValueError("SVD targets 必须是不重复的 block 投影：QKV、attention 输出、FFN 或 AdaLN")
    if config["model"]["keep_layers"] != list(range(sweep["teacher_depth"])):
        raise ValueError("rank sweep 的所有学生必须保留 teacher 全部层")
    if training["ppl_evaluation"]["enabled"]:
        raise ValueError("rank 对照关闭旧版训练中间 PPL，统一在初始与训练后评测")
    if training["early_stopping"]["enabled"] and training["early_stopping"]["metric"] not in ("kl", "loss"):
        raise ValueError("低秩学生按验证 KL（或等价的纯 KL loss）早停")
    if config["loss"] != {"kd_weight": 1.0, "ce_weight": 0.0}:
        raise ValueError("本实验沿用 LangFlow off-policy 的纯 KL 蒸馏")
    if (evaluation["num_steps"] < 2 or not 1 <= evaluation["sequence_length"] <= MAX_LENGTH
            or evaluation["batch_size"] < 1 or evaluation["timing_warmup_forwards"] < 0):
        raise ValueError("生成步数至少为 2，长度在 1–1024，batch 为正，预热次数非负")
    if not debug and evaluation["num_samples"] != NUM_SAMPLES:
        raise ValueError("正式 rank sweep 固定评测 1024 条；小样本配置请使用单卡 debug 入口")
    evaluation_config(config, [{"name": "teacher", "corpus": "owt"}], "run", debug)
    return config


@torch.no_grad()
def prepare_svd(teacher, config, run_dir):
    directory = run_dir / "svd"
    if (directory / "summary.json").is_file():
        return
    directory.mkdir(parents=True, exist_ok=True)
    local = []
    # 新版 60 个真实矩阵按序分给八卡，每个矩阵只分解一次，四组截取同一 SVD 的不同前缀。
    for index, (path, layer) in enumerate(target_linears(teacher, config["rank_sweep"]["targets"])):
        if index % distributed.world_size() != distributed.rank():
            continue
        with progress.phase("rank_sweep.svd", module=path):
            matrix = layer.weight.detach().float()
            u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
            relative_error = (torch.linalg.vector_norm(matrix - (u * singular) @ vh)
                              / torch.linalg.vector_norm(matrix).clamp_min(1e-12)).item()
            if not torch.isfinite(singular).all().item() or relative_error > 1e-2:
                raise ValueError(f"SVD 满秩重构检查未通过: {path}, relative_error={relative_error}")
            torch.save({"u": u.cpu(), "singular": singular.cpu(), "vh": vh.cpu()}, directory / f"{path}.pt")
            energy = singular.square()
            local.append({"module": path, "shape": list(matrix.shape), "full_rank_relative_error": relative_error,
                "retained": [{"ratio": ratio, "rank": retained_rank(matrix.shape, ratio),
                    "energy_fraction": (energy[:retained_rank(matrix.shape, ratio)].sum()
                                         / energy.sum().clamp_min(1e-12)).item()}
                    for ratio in config["rank_sweep"]["ratios"]]})
            del matrix, u, singular, vh
    shards = distributed.gather_main(local)
    if distributed.is_main():
        write_json(directory / "summary.json", {"teacher": config["model"]["teacher"],
            "targets": config["rank_sweep"]["targets"],
            "factorization": "U_sqrtS__sqrtS_Vh", "rounding": "floor_ratio_times_min_dimension",
            "matrices": sorted((row for shard in shards for row in shard), key=lambda row: row["module"])})
    distributed.barrier()


@torch.no_grad()
def build_svd_student(teacher, ratio, config, run_dir):
    targets = list(target_linears(teacher, config["rank_sweep"]["targets"]))
    architecture = copy.deepcopy(teacher.config)
    architecture.low_rank = {"ratio": ratio, "method": "svd_sqrt", "rounding": "floor",
        "ranks": {path: retained_rank(layer.weight.shape, ratio) for path, layer in targets}}
    student = LangFlow(architecture).to(next(teacher.parameters()).device)
    replaced = {path + suffix for path, _ in targets for suffix in (".weight", ".bias")}
    weights = {key: value for key, value in teacher.state_dict().items() if key not in replaced}
    for path, layer in targets:
        values = torch.load(run_dir / "svd" / f"{path}.pt", map_location="cpu", weights_only=True)
        down, up = balanced_factors(values["u"], values["singular"], values["vh"], architecture.low_rank["ranks"][path])
        weights[path + ".down.weight"], weights[path + ".up.weight"] = down, up
        if layer.bias is not None:
            weights[path + ".up.bias"] = layer.bias.detach()
    student.load_state_dict(weights, strict=True)
    student.backbone.vocab_embed.requires_grad_(False)
    student.proposal.requires_grad_(False)
    return student


@torch.no_grad()
def check_full_rank(teacher, config, run_dir):
    """真实 teacher 的满秩因子化单步对照；不额外训练或生成一整套样本。"""
    with progress.phase("rank_sweep.full_rank_check"):
        student = build_svd_student(teacher, 1.0, config, run_dir).eval()
        device = next(teacher.parameters()).device
        # 只缩短校验输入，结构仍是全部真实预训练层；正式训练/生成长度保持 1024。
        token_ids = load_tokens(config, "valid")[:1, :128].to(device)
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            torch.manual_seed(config["run"]["seed"])
            state = off_policy_state(teacher, student, token_ids, 0.0)
            expected, actual = posterior_logits(teacher, state), posterior_logits(student, state)
        loss, _ = distillation_loss(actual, expected, None, config["loss"])
        difference = actual.float() - expected.float()
        if not torch.isfinite(loss).item() or not torch.isfinite(difference).all().item():
            raise FloatingPointError("满秩 SVD 单步对照产生非有限结果")
        report = {"rank_ratio": 1.0, "num_samples": len(token_ids), "sequence_length": token_ids.shape[1],
            "num_factorized_matrices": len(student.config.low_rank["ranks"]),
            "posterior_kl": float(loss), "max_abs_logit_error": difference.abs().max().item(),
            "logits_relative_l2_error": (torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)).item(),
            "note": "Single-step numerical comparison; BF16 dense/factorized matmuls may differ. Not generation evaluation."}
        if distributed.is_main():
            write_json(run_dir / "full_rank_check.json", report)
    distributed.barrier()


def stage_config(config, ratio):
    result = copy.deepcopy(config)
    result["run"]["name"] = rank_name(ratio)
    result["rank_sweep"]["current_ratio"] = ratio
    return result


def checkpoint_entry(config, checkpoint, name):
    checkpoint = Path(checkpoint)
    return {"name": name, "corpus": "owt", "architecture": str(checkpoint / "config.json"),
        "weights": str(checkpoint / "model.safetensors"),
        "generation": {key: config["evaluation"][key] for key in ("num_steps", "sequence_length")}}


def save_rank_checkpoint(student, tokenizer, run_dir, step, config, name=None):
    checkpoint = save_checkpoint(student, tokenizer, run_dir, step, config, name)
    if distributed.is_main():
        metadata = json.loads((checkpoint / "distillation.json").read_text())
        metadata.update(initialization="teacher_svd", low_rank=distributed.unwrap(student).config.low_rank)
        write_json(checkpoint / "distillation.json", metadata)
        evaluation = evaluation_config(config, [checkpoint_entry(config, checkpoint, config["run"]["name"])],
                                       run_dir, debug=config["rank_sweep"].get("debug", False))
        (checkpoint / "gen_ppl_eval.yaml").write_text(yaml.safe_dump(evaluation, sort_keys=False))
    distributed.barrier()
    return checkpoint


def find_evaluation_directory(run_dir, stage, evaluation):
    receipt = run_dir / f"{stage}_evaluation.json"
    if receipt.is_file():
        return Path(json.loads(receipt.read_text())["directory"])
    # 兼容旧运行：失败前还没写 stage receipt，但已保存部分模型的 metrics.json。
    for directory in sorted(run_dir.glob("*_unified_gen_ppl"), reverse=True):
        saved = directory / "config.yaml"
        if saved.is_file() and evaluation_matches(evaluation, load_yaml_config(saved)):
            return directory
    return None


def evaluate_stage(config, entries, run_dir, stage, allow_distributed, debug):
    receipt = run_dir / f"{stage}_evaluation.json"
    evaluation = evaluation_config(config, entries, run_dir, debug)
    path = run_dir / f"gen_ppl_{stage}.yaml"
    if distributed.is_main():
        path.write_text(yaml.safe_dump(evaluation, sort_keys=False), encoding="utf-8")
    directory = None
    if distributed.is_main():
        directory = find_evaluation_directory(run_dir, stage, evaluation)
        if directory is None:
            directory = create_run_dir(evaluation)
        # 在昂贵的评测之前就保存位置，OOM 后也能按模型恢复。
        write_json(receipt, {"directory": str(directory), "status": "running"})
    directory = Path(distributed.broadcast_main(str(directory) if distributed.is_main() else None))
    # 与独立 eval_gen_ppl.sh 共用逐模型恢复；未完成模型重跑，已完成结果原样保留。
    run_gen_ppl(evaluation, allow_distributed=allow_distributed, debug=debug, resume_dir=directory)
    if distributed.is_main():
        write_json(receipt, {"directory": str(directory), "status": "completed"})
    progress.configure(run_dir, distributed.rank())
    distributed.barrier()
    return json.loads((directory / "metrics.json").read_text())


def save_results(run_dir, config, reports):
    rows = []
    for phase, report in reports.items():
        for row in report["models"]:
            teacher = row["name"] == "teacher"
            ratio = 1.0 if teacher else next(r for r in config["rank_sweep"]["ratios"] if rank_name(r) in row["name"])
            training = {}
            if phase == "after" and not teacher:
                training = json.loads((run_dir / rank_name(ratio) / "training.json").read_text())
            rows.append({**row, "phase": "teacher" if teacher else phase, "rank_ratio": ratio,
                "rank": None if teacher else retained_rank((config["rank_sweep"]["hidden_size"],) * 2, ratio),
                "conditioning_rank": None if teacher else conditioning_rank(config, ratio),
                "depth": config["rank_sweep"]["teacher_depth"],
                "training_steps": training.get("steps"), "best_step": training.get("best_step"),
                "tokens_seen": training.get("tokens_seen"),
                "checkpoint_tokens_seen": training.get("checkpoint_tokens_seen"),
                "stop_reason": training.get("stop_reason")})
    write_json(run_dir / "summary.json", {"config": config, "results": rows,
        "metric_definition": "rank is the main hidden-projection rank; conditioning_rank is the AdaLN rank; mean_step_ms is per-batch generator forward latency after warmup; num_parameters includes embeddings"})
    fields = ["name", "phase", "rank_ratio", "rank", "conditioning_rank", "depth", "gen_ppl", "sample_entropy", "gm", "gm_standard_error",
        "num_parameters", "mean_step_ms", "effective_generation_batch_size", "timed_forward_calls",
        "num_samples", "num_scored_samples", "num_empty_samples", "score_status", "gm_num_pairs", "gm_num_skipped_pairs", "gm_status",
        "training_steps", "best_step", "tokens_seen", "checkpoint_tokens_seen", "stop_reason"]
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if config["rank_sweep"]["plot_results"]:
        plot_results(run_dir, rows)


def plot_results(run_dir, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, label in zip(axes, ("gen_ppl", "sample_entropy", "gm"), ("Gen PPL", "Unigram entropy (nats)", "GM")):
        for phase in ("before", "after"):
            selected = sorted((r for r in rows if r["phase"] == phase), key=lambda r: r["rank_ratio"])
            if selected:
                ax.plot([100 * r["rank_ratio"] for r in selected],
                    [r.get(metric) if r.get(metric) is not None else float("nan") for r in selected],
                    marker="o", label=phase + " KL distillation")
        teacher = next((r for r in rows if r["phase"] == "teacher"), None)
        if teacher and teacher.get(metric) is not None:
            ax.axhline(teacher[metric], linestyle="--", color="gray", label="Teacher")
        if metric == "gen_ppl":
            ax.set_yscale("log")
        ax.set(xlabel="Retained rank (%)", ylabel=label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("LangFlow-OWT: same depth, SVD rank bottleneck")
    fig.tight_layout()
    fig.savefig(run_dir / "rank_quality_curve.png", dpi=160)
    plt.close(fig)


def run_sweep(config, command="all", allow_distributed=True, debug=False, resume_dir=None):
    if command == "all" and (config["training"]["plot_loss"] or config["rank_sweep"]["plot_results"]):
        from importlib.util import find_spec
        if find_spec("matplotlib") is None:
            raise RuntimeError("绘图需要 matplotlib；请先在服务器安装项目依赖")
    communication = config["distributed"]
    owns_group = distributed.setup(config["model"]["device"], allow_distributed, communication["control_timeout_seconds"])
    run_dir = Path(distributed.broadcast_main(str(project_path(resume_dir) if resume_dir else create_run_dir(config))
                                             if distributed.is_main() else None))
    progress.configure(run_dir, distributed.rank())
    reports = {}
    try:
        if distributed.is_main():
            saved_config = run_dir / "config.yaml"
            if resume_dir and config["training"]["early_stopping"] != load_yaml_config(saved_config)["training"]["early_stopping"]:
                # 仅在学生尚未开始训练时更新规则；初始权重和已有初评不受影响。
                write_json(run_dir / "early_stopping_update.json", {
                    "previous": load_yaml_config(saved_config)["training"]["early_stopping"],
                    "current": config["training"]["early_stopping"]})
                saved_config.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            write_json(run_dir / "status.json", {"status": "running", "command": command, "resumed": bool(resume_dir)})
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
        distributed.setup_tensor_group(config["model"]["device"], communication["tensor_timeout_seconds"])
        torch.set_float32_matmul_precision("highest")
        device = torch.device(config["model"]["device"])
        if device.type == "cuda":
            device = torch.device("cuda", torch.cuda.current_device())
        teacher_files = resolve_model_files(config["model"]["teacher"], config["model"]["cache_dir"], config["model"]["teacher_config"])
        initial_entries = []
        needs_build = any(not (run_dir / rank_name(r) / "checkpoints/initial/model.safetensors").is_file()
                          for r in config["rank_sweep"]["ratios"])
        if needs_build:
            seed_everything(config["run"]["seed"] + distributed.rank())
            teacher = load_teacher(config)
            if (teacher.config.n_blocks != config["rank_sweep"]["teacher_depth"]
                    or teacher.config.hidden_size != config["rank_sweep"]["hidden_size"]
                    or teacher.config.cond_dim != config["rank_sweep"].get("cond_dim", 128) or teacher.config.low_rank):
                raise ValueError("SVD 起点必须是配置指定深度/宽度的原始 dense teacher")
            if len(tokenizer) != teacher.config.vocab_size or max(config["data"]["sequence_length"], config["evaluation"]["sequence_length"]) > teacher.config.model_length:
                raise ValueError("真实 tokenizer/序列长度与 teacher 不匹配")
            prepare_svd(teacher, config, run_dir)
            if config["rank_sweep"].get("full_rank_check", False):
                check_full_rank(teacher, config, run_dir)
        for ratio in config["rank_sweep"]["ratios"]:
            current = stage_config(config, ratio)
            directory = run_dir / rank_name(ratio)
            checkpoint = directory / "checkpoints/initial"
            if not (checkpoint / "model.safetensors").is_file():
                if distributed.is_main():
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "config.yaml").write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
                distributed.barrier()
                student = build_svd_student(teacher, ratio, config, run_dir)
                save_rank_checkpoint(student, tokenizer, directory, 0, current, name="initial")
                del student
            initial_entries.append(checkpoint_entry(config, checkpoint, rank_name(ratio) + "_initial"))
        if needs_build:
            del teacher
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if command == "build":
            if distributed.is_main():
                write_json(run_dir / "status.json", {"status": "completed", "command": command})
            return run_dir
        teacher_entry = {"name": "teacher", "corpus": "owt", "architecture": teacher_files[0], "weights": teacher_files[1],
            "generation": {key: config["evaluation"][key] for key in ("num_steps", "sequence_length")}}
        reports["before"] = evaluate_stage(config, [teacher_entry, *initial_entries], run_dir, "before", allow_distributed, debug)
        if distributed.is_main():
            save_results(run_dir, config, reports)
        distributed.barrier()
        # 各组重新加载自己的 SVD 起点，并重置训练 RNG/optimizer；始终保留全部层。
        teacher = None
        final_entries = []
        for ratio in config["rank_sweep"]["ratios"]:
            current = stage_config(config, ratio)
            directory = run_dir / rank_name(ratio)
            if (directory / "training.json").is_file():
                checkpoint = Path(json.loads((directory / "training.json").read_text())["checkpoint"])
            else:
                if distributed.is_main():
                    # 恢复旧初评时，分组记录也保存本次实际采用的早停设置。
                    (directory / "config.yaml").write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
                if teacher is None:
                    teacher = load_teacher(config)
                    distributed.barrier()
                progress.configure(directory, distributed.rank())
                student = load_model(str(directory / "checkpoints/initial"), device)
                student.backbone.vocab_embed.requires_grad_(False)
                student.proposal.requires_grad_(False)
                seed_everything(config["run"]["seed"] + distributed.rank())
                checkpoint = train(teacher, student, tokenizer, current, directory, checkpoint_fn=save_rank_checkpoint)
                del student
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            final_entries.append(checkpoint_entry(config, checkpoint, rank_name(ratio) + "_trained"))
        del teacher
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        progress.configure(run_dir, distributed.rank())
        reports["after"] = evaluate_stage(config, final_entries, run_dir, "after", allow_distributed, debug)
        if distributed.is_main():
            combined = evaluation_config(config, [teacher_entry, *initial_entries, *final_entries], run_dir, debug)
            (run_dir / "gen_ppl_models.yaml").write_text(yaml.safe_dump(combined, sort_keys=False), encoding="utf-8")
            save_results(run_dir, config, reports)
            write_json(run_dir / "status.json", {"status": "completed", "command": command,
                "student_depth": config["rank_sweep"]["teacher_depth"], "ratios": config["rank_sweep"]["ratios"]})
        distributed.barrier()
    except Exception as error:
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "failed", "error": str(error)})
        distributed.abort_on_cuda_oom(error, run_dir)
        raise
    finally:
        if owns_group:
            distributed.close()
    return run_dir


def main(default_config="configs/owt_rank_sweep.yaml", allow_distributed=True, debug=False):
    parser = argparse.ArgumentParser(description="同层数 LangFlow-OWT 的 SVD rank sweep 与统一评测")
    parser.add_argument("command", nargs="?", choices=("all", "prepare", "build"), default="all")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-dir", help="复用实验保存的配置及已完成阶段；未完成训练组从 SVD 起点重跑")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置并保存计划，不加载模型或数据")
    args = parser.parse_args()
    if args.run_dir and args.overrides:
        allowed = {f"training.early_stopping.{key}" for key in ("enabled", "metric", "min_steps", "patience", "min_delta")}
        if any(override.split("=", 1)[0] not in allowed for override in args.overrides):
            parser.error("恢复时 --set 仅允许修改尚未开始训练的早停设置；其他新条件请创建新实验")
    path = project_path(args.run_dir) / "config.yaml" if args.run_dir else args.config
    config = load_config(path, args.overrides, debug)
    if args.run_dir and args.overrides:
        saved = load_yaml_config(path)
        directory = project_path(args.run_dir)
        if config["training"]["early_stopping"] != saved["training"]["early_stopping"]:
            if any(directory.glob("rank_*/metrics.jsonl")) or any(directory.glob("rank_*/training.json")):
                parser.error("学生已有训练记录，不能在原实验中改变早停规则；请创建新实验")
    config["rank_sweep"]["debug"] = debug
    if args.dry_run:
        plan = {"teacher": config["model"]["teacher"], "student_depth": config["rank_sweep"]["teacher_depth"],
            "students": [{"ratio": ratio, "rank": retained_rank((config["rank_sweep"]["hidden_size"],) * 2, ratio),
                          "conditioning_rank": conditioning_rank(config, ratio)}
                         for ratio in config["rank_sweep"]["ratios"]],
            "targets": config["rank_sweep"]["targets"],
            "num_svd_matrices": config["rank_sweep"]["teacher_depth"] * len(config["rank_sweep"]["targets"]),
            "full_rank_check": config["rank_sweep"].get("full_rank_check", False),
            "training": config["training"], "evaluation": config["evaluation"], "gradient_moment": config["gradient_moment"],
            "global_batch_size_8gpu": 8 * config["training"]["batch_size"] * config["training"]["grad_accumulation"],
            "stages": ["teacher_svd", "teacher_and_initial_evaluation", "off_policy_kl", "trained_evaluation"], "debug": debug}
        if args.run_dir:
            manifest_path = project_path(args.run_dir) / "gen_ppl_before.yaml"
            if manifest_path.is_file():
                before = load_yaml_config(manifest_path)
                directory = find_evaluation_directory(project_path(args.run_dir), "before", before)
                if directory is not None:
                    completed = [entry["name"] for entry in before["models"]
                                 if completed_model_result(directory / entry["name"], before["gradient_moment"]["enabled"]) is not None]
                    plan["resume_before"] = {"directory": str(directory), "completed": completed,
                        "remaining": [entry["name"] for entry in before["models"] if entry["name"] not in completed]}
        output = project_path("run/launch") / f"rank_sweep_plan_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        write_json(output, plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        print(f"[rank-sweep] plan: {output}", flush=True)
        return
    directory = run_sweep(config, args.command, allow_distributed, debug, args.run_dir)
    print(f"[rank-sweep] results: {directory}", flush=True)


if __name__ == "__main__":
    main()
