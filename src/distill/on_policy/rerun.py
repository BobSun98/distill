"""读取实际 off-policy loss 元数据，默认只生成 ELF-B/L 的对齐重跑配置。"""

import argparse
import json

import yaml

from ..config import project_path, write_json
from .pipeline import load_config, student_files

CONFIGS = ("langflow_owt", "langflow_lm1b", "elf_b", "elf_l")
DEFAULT_MODELS = ("elf_b", "elf_l")


def matched_loss(family, source):
    # rollout 没有原文本标签，不能把第一阶段的 supervised CE/native target 偷换成伪标签。
    if source.get("ce_weight", 0) or source.get("native_flow_weight", 0):
        raise ValueError("该 off-policy loss 含真实 token CE 或 native flow，本重跑入口不能自动等价映射")
    if "kd_weight" not in source:
        raise ValueError("需要第一阶段含 kd_weight 的 loss 元数据，不能用旧 on-policy checkpoint 代替起点")
    if family == "elf" and "flow_weight" not in source:
        raise ValueError("ELF off-policy 元数据缺少 flow_weight")
    return {"ce_weight": 0.0, "kl_weight": source["kd_weight"],
            "mse_weight": source["flow_weight"] if family == "elf" else 0.0, "ce_target": "soft"}


def prepare_plan(output_dir, check_weights=True, models=DEFAULT_MODELS):
    if not models or len(set(models)) != len(models) or any(tag not in CONFIGS for tag in models):
        raise ValueError("重跑模型列表为空、重复或包含未知模型")
    output = project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for tag in models:
        source_yaml = f"configs/on_policy_{tag}.yaml"
        config = load_config(source_yaml)
        value = config["model"]["student_weights"]
        if not value:
            raise ValueError(f"{source_yaml}: 请先填写 model.student_weights")
        path = project_path(value)
        directory = path.parent if path.suffix == ".safetensors" else path
        metadata_path = directory / "distillation.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"需要 off-policy checkpoint 的 loss 元数据: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("stage") == "on_policy":
            raise ValueError(f"{tag} 指向 on-policy checkpoint，请恢复第一阶段权重路径")
        if metadata.get("teacher") != config["model"]["teacher"]:
            raise ValueError(f"{tag}: off-policy 元数据中的 teacher 与当前配置不一致")
        if check_weights:
            # 启动任何 GPU 任务前检查选中模型的权重与架构；不加载模型或计算 SHA。
            student_files(config)
        source_loss = metadata["loss"]
        previous_loss = dict(config["loss"])
        config["loss"] = matched_loss(config["model"]["family"], source_loss)
        source_config = directory.parent.parent / "config.yaml"
        if config["model"]["family"] == "elf" and source_config.is_file():
            original = yaml.safe_load(source_config.read_text(encoding="utf-8"))
            probability = original.get("diffusion", {}).get("decoder_probability")
            if probability is not None:
                config["on_policy"]["decoder_probability"] = probability
        config["run"]["name"] = f"on_policy_{tag}_loss_matched"
        config["loss_alignment"] = {"source_checkpoint": str(directory),
                                     "source_metadata": str(metadata_path), "off_policy_loss": source_loss}
        path = output / f"{tag}.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        # 生成后仍走正式入口的配置校验，确保 KL decoder 的缓存和概率也合法。
        load_config(path)
        job = {"model": tag, "config": str(path), "student_weights": value,
               "previous_on_policy_loss": previous_loss, "off_policy_loss": source_loss,
               "matched_on_policy_loss": config["loss"],
               "decoder_probability": config["on_policy"]["decoder_probability"] if config["model"]["family"] == "elf" else None}
        jobs.append(job)
        print(f"[loss-matched] {tag}: {config['loss']}", flush=True)
        print(f"  off-policy start: {value}", flush=True)
    write_json(output / "plan.json", {"weights_checked": check_weights,
        "evaluation": "scripts/eval_gen_ppl.sh / gen_ppl_v1", "jobs": jobs})
    return jobs


def main():
    parser = argparse.ArgumentParser(description="默认为 ELF-B/L 生成与实际 off-policy loss 对齐的配置")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", help="仅读取元数据生成计划，不要求本地同步 safetensors")
    parser.add_argument("--all-models", action="store_true", help="额外重跑梯度已等价的两个 LangFlow 模型")
    args = parser.parse_args()
    prepare_plan(args.output_dir, check_weights=not args.dry_run,
                 models=CONFIGS if args.all_models else DEFAULT_MODELS)


if __name__ == "__main__":
    main()
