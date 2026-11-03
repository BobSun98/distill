"""同深度 LangFlow-OWT：全层/尾部层数 × rank 的 SVD、KL 蒸馏与统一评测。"""

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


def compressed_layers(config, tail_layers=None):
    depth = config["rank_sweep"]["teacher_depth"]
    # block 下标从 0 开始；K=2 只选中 10/11，不删除其余 dense 层。
    return list(range(depth - tail_layers, depth)) if tail_layers is not None else list(range(depth))


def rank_max_steps(config, ratio):
    # 新增 rank 可以单独增加预算，不能把旧学生的 5000 步记录改成 10000 步。
    return config["rank_sweep"].get("max_steps_by_rank", {}).get(rank_name(ratio), config["training"]["max_steps"])


def student_specs(config):
    sweep = config["rank_sweep"]
    counts = sweep.get("tail_layers")
    return [{"name": (f"tail_{count:02d}_" if count is not None else "") + rank_name(ratio),
             "ratio": ratio, "tail_layers": count if count is not None else sweep["teacher_depth"],
             "max_steps": rank_max_steps(config, ratio),
             "layer_indices": compressed_layers(config, count)}
            for count in (counts if counts is not None else [None]) for ratio in sweep["ratios"]]


def reference_rows(config):
    """只读引用已完成的全层结果，不复制 checkpoint，不重新训练/评测。"""
    source = config["rank_sweep"].get("reference_run")
    if not source:
        return []
    report = json.loads((project_path(source) / "summary.json").read_text(encoding="utf-8"))
    previous = report["config"]
    # 比较实验条件即可，不做权重哈希；改了预算/协议时应关闭旧结果引用。
    for key in ("model", "data", "loss", "training", "evaluation", "gradient_moment"):
        if config[key] != previous[key]:
            raise ValueError(f"全层参考实验的 {key} 与当前配置不同；请设置 rank_sweep.reference_run=null 或选择匹配实验")
    if config["run"]["seed"] != previous["run"]["seed"] or any(
        config["rank_sweep"].get(key) != previous["rank_sweep"].get(key)
        for key in ("teacher_depth", "hidden_size", "cond_dim", "targets")):
        raise ValueError("全层参考实验的 seed/架构/SVD 目标与当前配置不一致")
    depth = config["rank_sweep"]["teacher_depth"]
    rows = [{**row, "tail_layers": depth, "layer_indices": list(range(depth)),
             "training_max_steps": row.get("training_max_steps", rank_max_steps(previous, row["rank_ratio"])),
             "is_reference": True, "reference_run": str(source)}
            for row in report["results"] if row["phase"] in ("before", "after")
            and row["rank_ratio"] in config["rank_sweep"]["ratios"]
            and row.get("tail_layers", row["depth"]) == depth and not row.get("is_reference", False)]
    expected = {(phase, ratio) for phase in ("before", "after") for ratio in config["rank_sweep"]["ratios"]}
    if {(row["phase"], row["rank_ratio"]) for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("参考实验缺少所选 rank 的完整全层初评/复评结果")
    return rows


def reference_gm_path(config):
    source = config["rank_sweep"].get("reference_run")
    if not source or not config["gradient_moment"]["enabled"]:
        return None
    root = project_path(source)
    receipt = json.loads((root / "before_evaluation.json").read_text(encoding="utf-8"))
    # 使用子目录名定位，服务器绝对路径的记录在本地同步后也可读取。
    path = root / Path(receipt["directory"]).name / "gradient_moment/reference_owt.json"
    if not path.is_file():
        raise FileNotFoundError(f"全层参考实验缺少固定 GM 参考语料: {path}")
    return path


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
    limits = sweep.get("max_steps_by_rank", {})
    if not isinstance(limits, dict) or any(
        name not in {rank_name(ratio) for ratio in ratios} or type(steps) is not int or steps < 1
        for name, steps in limits.items()):
        raise ValueError("max_steps_by_rank 必须按 rank_XXX 配置已选 rank 的正整数训练上限")
    counts = sweep.get("tail_layers")
    if counts is not None and (not counts or len(set(counts)) != len(counts)
            or any(type(count) is not int or not 1 <= count <= sweep["teacher_depth"] for count in counts)):
        raise ValueError("tail_layers 必须是非空、不重复且介于 1 和 teacher_depth 的整数列表")
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
    reference_rows(config)
    reference_gm_path(config)
    return config


@torch.no_grad()
def prepare_svd(teacher, config, run_dir):
    directory = run_dir / "svd"
    if (directory / "summary.json").is_file():
        return
    directory.mkdir(parents=True, exist_ok=True)
    local = []
    layers = sorted({index for spec in student_specs(config) for index in spec["layer_indices"]})
    # 只分解本轮网格覆盖的层；各格共享同一份 teacher SVD，不逐个学生重复计算。
    for index, (path, layer) in enumerate(target_linears(teacher, config["rank_sweep"]["targets"], layers)):
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
            "layer_indices": layers,
            "factorization": "U_sqrtS__sqrtS_Vh", "rounding": "floor_ratio_times_min_dimension",
            "matrices": sorted((row for shard in shards for row in shard), key=lambda row: row["module"])})
    distributed.barrier()


@torch.no_grad()
def build_svd_student(teacher, ratio, config, run_dir, tail_layers=None):
    layers = compressed_layers(config, tail_layers)
    targets = list(target_linears(teacher, config["rank_sweep"]["targets"], layers))
    architecture = copy.deepcopy(teacher.config)
    architecture.low_rank = {"ratio": ratio, "method": "svd_sqrt", "rounding": "floor",
        "layer_indices": layers,
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
def check_full_rank(teacher, config, run_dir, tail_layers=None):
    """真实 teacher 的满秩因子化单步对照；不额外训练或生成一整套样本。"""
    filename = "full_rank_check.json" if tail_layers is None else f"full_rank_check_tail_{tail_layers:02d}.json"
    path = run_dir / filename
    cached = distributed.broadcast_main(path.is_file() if distributed.is_main() else None)
    if cached:
        return
    with progress.phase("rank_sweep.full_rank_check"):
        student = build_svd_student(teacher, 1.0, config, run_dir, tail_layers).eval()
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
            "tail_layers": len(compressed_layers(config, tail_layers)),
            "layer_indices": compressed_layers(config, tail_layers),
            "num_factorized_matrices": len(student.config.low_rank["ranks"]),
            "posterior_kl": float(loss), "max_abs_logit_error": difference.abs().max().item(),
            "logits_relative_l2_error": (torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)).item(),
            "note": "Single-step numerical comparison; BF16 dense/factorized matmuls may differ. Not generation evaluation."}
        if distributed.is_main():
            write_json(path, report)
    distributed.barrier()


def stage_config(config, ratio, tail_layers=None):
    result = copy.deepcopy(config)
    prefix = f"tail_{tail_layers:02d}_" if config["rank_sweep"].get("tail_layers") is not None else ""
    result["run"]["name"] = prefix + rank_name(ratio)
    result["rank_sweep"]["current_ratio"] = ratio
    result["rank_sweep"]["current_tail_layers"] = tail_layers if tail_layers is not None else config["rank_sweep"]["teacher_depth"]
    result["training"]["max_steps"] = rank_max_steps(config, ratio)
    return result


def check_step_limit_change(run_dir, previous, config):
    changes = {rank_name(ratio): {"previous": rank_max_steps(previous, ratio), "current": rank_max_steps(config, ratio)}
               for ratio in config["rank_sweep"]["ratios"]
               if rank_max_steps(previous, ratio) != rank_max_steps(config, ratio)}
    for spec in student_specs(config):
        if rank_name(spec["ratio"]) not in changes:
            continue
        directory = run_dir / spec["name"]
        # 训练入口先写 step=0 验证记录；保守地将任何训练日志视为已开始，初始生成评测不受影响。
        if any((directory / name).is_file() for name in ("metrics.jsonl", "training.json")):
            raise ValueError(f"{spec['name']} 已有训练记录，不能在原组中改变 max_steps；该组预算保持不变")
    return changes


def save_rank_extension(run_dir, config):
    """同目录增加 rank/调整未训练 rank 的预算，旧学生与逐模型指标不覆盖。"""
    path = run_dir / "config.yaml"
    previous = load_yaml_config(path)
    original = previous["rank_sweep"]["ratios"]
    added = [ratio for ratio in config["rank_sweep"]["ratios"] if ratio not in original]
    changes = check_step_limit_change(run_dir, previous, config)
    unchanged = copy.deepcopy(config)
    unchanged["rank_sweep"]["ratios"] = original
    if "max_steps_by_rank" in previous["rank_sweep"]:
        unchanged["rank_sweep"]["max_steps_by_rank"] = previous["rank_sweep"]["max_steps_by_rank"]
    else:
        unchanged["rank_sweep"].pop("max_steps_by_rank", None)
    if not set(original).issubset(config["rank_sweep"]["ratios"]) or unchanged != previous:
        raise ValueError("必须保留所有原 rank；除未训练 rank 的步数上限外，训练、数据、评测等其他配置必须完全一致")
    if not added and not changes:
        return
    name = "_".join(rank_name(ratio) for ratio in added) if added else "_".join(
        f"{name}_max_steps_{change['current']}" for name, change in changes.items())
    backup = run_dir / "extensions" / name
    backup.mkdir(parents=True, exist_ok=True)
    for filename in ("config.yaml", "status.json", "summary.json", "summary.csv", "tail_rank_quality_curve.png",
                     "gen_ppl_before.yaml", "gen_ppl_after.yaml", "gen_ppl_models.yaml"):
        source, target = run_dir / filename, backup / filename
        if source.is_file() and not target.is_file():
            shutil.copyfile(source, target)
    write_json(backup / "extension.json", {"original_ratios": original,
        "added_ratios": added, "ratios": config["rank_sweep"]["ratios"],
        "step_limit_changes": changes,
        "new_students": [spec["name"] for spec in student_specs(config) if spec["ratio"] in added]})
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


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
        directory = Path(json.loads(receipt.read_text())["directory"])
        local = run_dir / directory.name
        return local if local.is_dir() else directory
    # 兼容旧运行：失败前还没写 stage receipt，但已保存部分模型的 metrics.json。
    for directory in sorted(run_dir.glob("*_unified_gen_ppl"), reverse=True):
        saved = directory / "config.yaml"
        if saved.is_file() and evaluation_matches(evaluation, load_yaml_config(saved), allow_new_models=True):
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
        reference = reference_gm_path(config)
        if reference is not None:
            target = directory / "gradient_moment/reference_owt.json"
            # 与旧 K=12 结果共用同一批真实文本；续评时保留本轮已固定的文件。
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(reference, target)
        # 在昂贵的评测之前就保存位置，OOM 后也能按模型恢复。
        write_json(receipt, {"directory": str(directory), "status": "running"})
    directory = Path(distributed.broadcast_main(str(directory) if distributed.is_main() else None))
    # 与独立 eval_gen_ppl.sh 共用逐模型恢复；未完成模型重跑，已完成结果原样保留。
    run_gen_ppl(evaluation, allow_distributed=allow_distributed, debug=debug, resume_dir=directory, allow_new_models=True)
    if distributed.is_main():
        write_json(receipt, {"directory": str(directory), "status": "completed"})
    progress.configure(run_dir, distributed.rank())
    distributed.barrier()
    return json.loads((directory / "metrics.json").read_text())


def save_results(run_dir, config, reports):
    rows = []
    specs = {spec["name"]: spec for spec in student_specs(config)}
    for phase, report in reports.items():
        for row in report["models"]:
            teacher = row["name"] == "teacher"
            name = row["name"].removesuffix("_initial").removesuffix("_trained")
            spec = None if teacher else specs[name]
            ratio = 1.0 if teacher else spec["ratio"]
            training = {}
            if phase == "after" and not teacher:
                training = json.loads((run_dir / spec["name"] / "training.json").read_text())
            rows.append({**row, "phase": "teacher" if teacher else phase, "rank_ratio": ratio,
                "rank": None if teacher else retained_rank((config["rank_sweep"]["hidden_size"],) * 2, ratio),
                "conditioning_rank": None if teacher else conditioning_rank(config, ratio),
                "depth": config["rank_sweep"]["teacher_depth"],
                "tail_layers": 0 if teacher else spec["tail_layers"],
                "layer_indices": [] if teacher else spec["layer_indices"],
                "is_reference": False, "reference_run": None,
                "training_max_steps": None if teacher else training.get("max_steps", spec["max_steps"]),
                "training_steps": training.get("steps"), "best_step": training.get("best_step"),
                "tokens_seen": training.get("tokens_seen"),
                "checkpoint_tokens_seen": training.get("checkpoint_tokens_seen"),
                "stop_reason": training.get("stop_reason")})
    rows.extend(row for row in reference_rows(config) if row["phase"] in reports)
    write_json(run_dir / "summary.json", {"config": config, "results": rows,
        "metric_definition": "tail_layers counts compressed final blocks; depth always counts all blocks; training_max_steps is the per-student cap (trained rows use training.json); is_reference keeps historical timing/training budgets; rank is the main projection rank; conditioning_rank is the AdaLN rank; mean_step_ms is per-batch forward latency; num_parameters includes embeddings"})
    fields = ["name", "phase", "rank_ratio", "rank", "conditioning_rank", "depth", "tail_layers", "is_reference", "reference_run",
        "gen_ppl", "sample_entropy", "repetition_4gram", "gm", "gm_standard_error",
        "num_parameters", "mean_step_ms", "effective_generation_batch_size", "timed_forward_calls",
        "num_samples", "num_scored_samples", "num_empty_samples", "score_status", "gm_num_pairs", "gm_num_skipped_pairs", "gm_status",
        "training_max_steps", "training_steps", "best_step", "tokens_seen", "checkpoint_tokens_seen", "stop_reason"]
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if config["rank_sweep"]["plot_results"]:
        if config["rank_sweep"].get("tail_layers") is not None:
            plot_tail_grid(run_dir, rows)
        else:
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


def plot_tail_grid(run_dir, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey="col")
    teacher = next((row for row in rows if row["phase"] == "teacher"), None)
    ratios = sorted({row["rank_ratio"] for row in rows if row["phase"] != "teacher"})
    for line, phase in zip(axes, ("before", "after")):
        for ax, metric, label in zip(line, ("gen_ppl", "sample_entropy", "gm"), ("Gen PPL", "Unigram entropy (nats)", "GM")):
            for ratio in ratios:
                selected = sorted((row for row in rows if row["phase"] == phase and row["rank_ratio"] == ratio),
                                  key=lambda row: row["tail_layers"])
                x = [row["tail_layers"] for row in selected]
                y = [row.get(metric) if row.get(metric) is not None else float("nan") for row in selected]
                # 追加 20% 后，40%/60% 仍保持原来的蓝色/橙色，便于对照旧图。
                color = {0.4: "C0", 0.6: "C1", 0.2: "C2"}.get(ratio)
                curve, = ax.plot(x, y, marker="o", color=color, label=f"{ratio:.0%} rank")
                if metric == "gm":
                    ax.errorbar(x, y, yerr=[row.get("gm_standard_error") or 0 for row in selected],
                                fmt="none", ecolor=curve.get_color(), capsize=3)
                # K=12 是旧实验的只读参考，空心点明确区分，不再次训练或评测。
                references = [row for row in selected if row.get("is_reference") and row.get(metric) is not None]
                if references:
                    ax.scatter([row["tail_layers"] for row in references], [row[metric] for row in references],
                               facecolors="white", edgecolors=curve.get_color(), zorder=4)
            if teacher and teacher.get(metric) is not None:
                ax.axhline(teacher[metric], linestyle="--", color="gray", label="Teacher")
            if metric == "gen_ppl":
                ax.set_yscale("log")
            ax.set(ylabel=label, title=phase + " KL distillation")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    counts = sorted({row["tail_layers"] for row in rows if row["phase"] != "teacher"})
    for ax in axes[-1]:
        ax.set(xlabel="Compressed tail blocks K", xticks=counts)
    fig.suptitle("LangFlow-OWT: tail blocks x rank; hollow markers = existing full-layer results")
    caps = sorted({(row["rank_ratio"], row["training_max_steps"]) for row in rows
                   if not row.get("is_reference") and row.get("training_max_steps") is not None})
    references = sorted({row["training_max_steps"] for row in rows
                         if row.get("is_reference") and row.get("training_max_steps") is not None})
    notes = []
    if caps:
        notes.append("Tail training caps: " + ", ".join(f"{ratio:.0%}={steps}" for ratio, steps in caps))
    if references:
        notes.append("K=12 reference caps: " + ", ".join(str(steps) for steps in references))
    if notes:
        fig.text(.5, .01, "; ".join(notes), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035 if notes else 0, 1, 1))
    fig.savefig(run_dir / "tail_rank_quality_curve.png", dpi=160)
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
            previous = load_yaml_config(saved_config) if resume_dir else None
            if resume_dir and (config["rank_sweep"]["ratios"] != previous["rank_sweep"]["ratios"]
                    or config["rank_sweep"].get("max_steps_by_rank", {}) != previous["rank_sweep"].get("max_steps_by_rank", {})):
                save_rank_extension(run_dir, config)
            elif resume_dir and config["training"]["early_stopping"] != previous["training"]["early_stopping"]:
                # 仅在学生尚未开始训练时更新规则；初始权重和已有初评不受影响。
                write_json(run_dir / "early_stopping_update.json", {
                    "previous": previous["training"]["early_stopping"],
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
        specs = student_specs(config)
        initial_entries = []
        needs_build = any(not (run_dir / spec["name"] / "checkpoints/initial/model.safetensors").is_file()
                          for spec in specs)
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
                for count in config["rank_sweep"].get("tail_layers") or [None]:
                    check_full_rank(teacher, config, run_dir, count)
        for spec in specs:
            ratio, count = spec["ratio"], spec["tail_layers"]
            current = stage_config(config, ratio, count)
            directory = run_dir / spec["name"]
            checkpoint = directory / "checkpoints/initial"
            if not (checkpoint / "model.safetensors").is_file():
                if distributed.is_main():
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "config.yaml").write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
                distributed.barrier()
                student = build_svd_student(teacher, ratio, config, run_dir, count)
                save_rank_checkpoint(student, tokenizer, directory, 0, current, name="initial")
                del student
            initial_entries.append(checkpoint_entry(config, checkpoint, spec["name"] + "_initial"))
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
        for spec in specs:
            current = stage_config(config, spec["ratio"], spec["tail_layers"])
            directory = run_dir / spec["name"]
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
            final_entries.append(checkpoint_entry(config, checkpoint, spec["name"] + "_trained"))
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
                "student_depth": config["rank_sweep"]["teacher_depth"], "ratios": config["rank_sweep"]["ratios"],
                "tail_layers": config["rank_sweep"].get("tail_layers"), "completed_students": [spec["name"] for spec in specs]})
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
    parser = argparse.ArgumentParser(description="同层数 LangFlow-OWT 的全层/尾部 SVD rank sweep 与统一评测")
    parser.add_argument("command", nargs="?", choices=("all", "prepare", "build"), default="all")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-dir", help="复用实验保存的配置及已完成阶段；未完成训练组从 SVD 起点重跑")
    parser.add_argument("--add-rank", dest="add_ratios", action="append", type=float, default=[],
                        help="向 --run-dir 追加 rank 比例（例如 0.2）；其他实验条件沿用原记录，可重复执行续跑")
    parser.add_argument("--max-steps", type=int, help="只设置 --add-rank 指定 rank 的训练上限；该 rank 所有组均未开始训练时可修改")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置并保存计划，不加载模型或数据")
    args = parser.parse_args()
    if args.add_ratios and not args.run_dir:
        parser.error("--add-rank 必须配合 --run-dir，追加到已有实验")
    if args.max_steps is not None and (not args.add_ratios or args.max_steps < 1):
        parser.error("--max-steps 必须为正整数，并配合 --run-dir / --add-rank 指定待训练的 rank")
    if args.run_dir and args.overrides:
        allowed = {f"training.early_stopping.{key}" for key in ("enabled", "metric", "min_steps", "patience", "min_delta")}
        if any(override.split("=", 1)[0] not in allowed for override in args.overrides):
            parser.error("恢复时 --set 仅允许修改尚未开始训练的早停设置；其他新条件请创建新实验")
    path = project_path(args.run_dir) / "config.yaml" if args.run_dir else args.config
    overrides = list(args.overrides)
    if args.add_ratios:
        ratios = list(load_yaml_config(path)["rank_sweep"]["ratios"])
        for ratio in args.add_ratios:
            if ratio not in ratios:
                ratios.append(ratio)
        overrides.append("rank_sweep.ratios=" + json.dumps(ratios))
    config = load_config(path, overrides, debug)
    if args.max_steps is not None:
        limits = dict(config["rank_sweep"].get("max_steps_by_rank", {}))
        limits.update({rank_name(ratio): args.max_steps for ratio in args.add_ratios})
        config["rank_sweep"]["max_steps_by_rank"] = limits
    if args.run_dir:
        # 在初始化 CUDA/DDP 前检查服务器上的真实记录；已完成的其他 rank 不阻挡新增 rank 改预算。
        check_step_limit_change(project_path(args.run_dir), load_yaml_config(path), config)
    if args.run_dir and args.overrides:
        saved = load_yaml_config(path)
        directory = project_path(args.run_dir)
        if config["training"]["early_stopping"] != saved["training"]["early_stopping"]:
            if any(directory.glob("*/metrics.jsonl")) or any(directory.glob("*/training.json")):
                parser.error("学生已有训练记录，不能在原实验中改变早停规则；请创建新实验")
    config["rank_sweep"]["debug"] = debug
    if args.dry_run:
        specs = student_specs(config)
        layers = sorted({index for spec in specs for index in spec["layer_indices"]})
        plan = {"teacher": config["model"]["teacher"], "student_depth": config["rank_sweep"]["teacher_depth"],
            "students": [{**spec, "rank": retained_rank((config["rank_sweep"]["hidden_size"],) * 2, spec["ratio"]),
                          "conditioning_rank": conditioning_rank(config, spec["ratio"])} for spec in specs],
            "targets": config["rank_sweep"]["targets"],
            "num_svd_matrices": len(layers) * len(config["rank_sweep"]["targets"]),
            "requested_add_ratios": args.add_ratios,
            "added_students": [spec["name"] for spec in specs if spec["ratio"] in args.add_ratios],
            "evaluation_models_total": 1 + 2 * len(specs),
            "new_evaluation_models": 1 + 2 * len(specs),
            "reference_run": config["rank_sweep"].get("reference_run"),
            "reference_models": [{key: row[key] for key in ("name", "phase", "rank_ratio", "tail_layers", "training_max_steps")}
                                 for row in reference_rows(config)],
            "full_rank_check": config["rank_sweep"].get("full_rank_check", False),
            "training": config["training"], "evaluation": config["evaluation"], "gradient_moment": config["gradient_moment"],
            "max_steps_by_rank": config["rank_sweep"].get("max_steps_by_rank", {}),
            "global_batch_size_8gpu": 8 * config["training"]["batch_size"] * config["training"]["grad_accumulation"],
            "stages": ["teacher_svd", "teacher_and_initial_evaluation", "off_policy_kl", "trained_evaluation"], "debug": debug}
        if args.run_dir:
            root = project_path(args.run_dir)
            for stage in ("before", "after"):
                suffix = "initial" if stage == "before" else "trained"
                expected = (["teacher"] if stage == "before" else []) + [spec["name"] + "_" + suffix for spec in specs]
                completed, directory = [], None
                manifest_path = root / f"gen_ppl_{stage}.yaml"
                if manifest_path.is_file():
                    evaluation = load_yaml_config(manifest_path)
                    directory = find_evaluation_directory(root, stage, evaluation)
                    if directory is not None:
                        completed = [name for name in expected
                                     if completed_model_result(directory / name, evaluation["gradient_moment"]["enabled"]) is not None]
                plan[f"resume_{stage}"] = {"directory": str(directory) if directory else None, "completed": completed,
                    "remaining": [name for name in expected if name not in completed]}
            trained = [spec["name"] for spec in specs if (root / spec["name"] / "training.json").is_file()]
            plan["resume_training"] = {"completed": trained, "remaining": [spec["name"] for spec in specs if spec["name"] not in trained]}
            plan["new_evaluation_models"] = sum(len(plan[f"resume_{stage}"]["remaining"]) for stage in ("before", "after"))
        output = project_path("run/launch") / f"rank_sweep_plan_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        write_json(output, plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        print(f"[rank-sweep] plan: {output}", flush=True)
        return
    directory = run_sweep(config, args.command, allow_distributed, debug, args.run_dir)
    print(f"[rank-sweep] results: {directory}", flush=True)


if __name__ == "__main__":
    main()
