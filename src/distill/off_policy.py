"""两类 off-policy 训练共用的统一评测入口；复用原有采集与训练流程。"""

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml

from . import distributed, gradient_moment, progress
from .config import create_run_dir, load_yaml_config, project_path, write_json
from .gen_ppl import run_gen_ppl, validate_protocol
from .gen_ppl_protocol import MAX_LENGTH, manifest


def pipeline(config):
    family = config["model"]["family"]
    if family == "langflow":
        from . import pipeline as implementation
    elif family == "elf":
        from .elf import pipeline as implementation
    else:
        raise ValueError("off-policy model.family 只支持 langflow 或 elf")
    return implementation


def load_config(path, overrides=(), debug=False):
    config = load_yaml_config(path, overrides)
    config = pipeline(config).load_config(path, overrides)
    # 新入口只在训练完成后用统一接口评测，避免无意触发旧版定期 PPL。
    periodic = config["training"]["ppl_evaluation"]
    if periodic.get("enabled", False) and periodic["interval"] > 0:
        raise ValueError("本入口的训练中间 PPL 请设 enabled=false；all 结束后自动运行统一评测")
    evaluation = config["evaluation"]
    if (evaluation["num_steps"] < 2 or evaluation["batch_size"] < 1
            or not 1 <= evaluation["sequence_length"] <= MAX_LENGTH):
        raise ValueError("统一评测采样至少 2 步，batch 为正，生成长度必须在 1–1024")
    # 长训练开始前就验证最终 GM 配置，避免训练结束后才发现参考语料或路径缺失。
    evaluation_config(config, "<checkpoint>", "run", debug)
    return config


def evaluation_config(config, checkpoint, run_dir, debug=False):
    checkpoint = project_path(checkpoint)
    directory = checkpoint.parent if checkpoint.is_file() else checkpoint
    weights = checkpoint if checkpoint.is_file() else checkpoint / "model.safetensors"
    settings = config["evaluation"]
    generation = {key: settings[key] for key in ("num_steps", "sequence_length")}
    if config["model"]["family"] == "elf":
        generation["sampling"] = dict(settings["sampling"])
    result = manifest([{"name": config["run"]["name"] + "_student",
        "corpus": settings.get("corpus"),
        "architecture": str(directory / "config.json"), "weights": str(weights), "generation": generation}])
    # 透传统一接口的 GM 开关/参考配置；corpus 必须由实验配置显式提供。
    result["gradient_moment"].update(config.get("gradient_moment", {}))
    result["run"]["root"] = str(run_dir)
    result["runtime"].update(device=config["model"]["device"], cache_dir=config["model"]["cache_dir"],
        generation_batch_size=settings["batch_size"],
        control_timeout_seconds=config["distributed"]["control_timeout_seconds"])
    if debug:
        result["protocol"]["num_samples"] = settings["num_samples"]
    validate_protocol(result["protocol"], debug)
    gradient_moment.validate_config(result)
    return result


def run_experiment(config, command="all", checkpoint=None, allow_distributed=True, debug=False):
    owns_group = distributed.setup(config["model"]["device"], allow_distributed,
                                    config["distributed"]["control_timeout_seconds"])
    run_dir = None
    try:
        if command == "evaluate":
            run_dir = Path(distributed.broadcast_main(str(create_run_dir(config)) if distributed.is_main() else None))
        else:
            # 外层持有通信组，原训练入口与统一评测直接复用；debug 不启动子进程。
            run_dir = pipeline(config).run_experiment(config, "train" if command == "all" else command,
                                                     allow_distributed=allow_distributed)
            if command in ("prepare", "train"):
                return run_dir
            training = json.loads((run_dir / "training.json").read_text(encoding="utf-8")) if distributed.is_main() else None
            checkpoint = distributed.broadcast_main(training["checkpoint"] if training else None)
        if checkpoint is None:
            raise ValueError("evaluate 需要 --checkpoint")
        # 原训练函数已经退出，释放其权重/优化器，再加载 generator 和 GPT scorer。
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        evaluation = evaluation_config(config, checkpoint, run_dir, debug)
        if distributed.is_main():
            (run_dir / "gen_ppl_eval.yaml").write_text(
                yaml.safe_dump(evaluation, allow_unicode=True, sort_keys=False), encoding="utf-8")
        result_dir = run_gen_ppl(evaluation, allow_distributed=allow_distributed, debug=debug)
        progress.configure(run_dir, distributed.rank())
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "completed", "command": command,
                "checkpoint": str(checkpoint), "evaluation_dir": str(result_dir)})
        distributed.barrier()
    except Exception as error:
        if run_dir is not None and distributed.is_main():
            write_json(run_dir / "status.json", {"status": "failed", "command": command, "error": str(error)})
        raise
    finally:
        if owns_group:
            distributed.close()
    return run_dir


def main(default_config="configs/off_policy_random_langflow_owt.yaml", allow_distributed=True, debug=False):
    parser = argparse.ArgumentParser(description="Off-policy 蒸馏与统一 Gen PPL；支持只随机初始化 Transformer blocks")
    parser.add_argument("command", nargs="?", choices=["prepare", "train", "evaluate", "all"], default="all")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="仅校验配置并输出计划，不加载模型或数据")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides, debug=debug)
    if args.dry_run:
        evaluation = evaluation_config(config, "<checkpoint>", "run", debug)
        print(json.dumps({"config": args.config, "model": config["model"], "loss": config["loss"],
            "training": config["training"], "protocol": evaluation["protocol"],
            "gradient_moment": evaluation["gradient_moment"], "corpus": evaluation["models"][0]["corpus"],
            "generation": evaluation["models"][0]["generation"]}, ensure_ascii=False, indent=2))
        return
    if args.command == "evaluate" and args.checkpoint is None:
        parser.error("evaluate 必须指定 --checkpoint")
    run_experiment(config, args.command, args.checkpoint, allow_distributed, debug)


if __name__ == "__main__":
    main()
