"""ELF-B/L 同进程蒸馏入口，八卡训练与单卡 debug 复用实际函数。"""

import argparse
import os
from functools import partial
from pathlib import Path

import torch

from .. import distributed, progress
from ..config import load_yaml_config, create_run_dir, project_path, write_json
from ..train import train, seed_everything
from . import models, data
from .evaluate import evaluate
from .losses import batch_loss, METRIC_KEYS


def load_config(path, overrides=()):
    config = load_yaml_config(path, overrides)
    training, loss, diffusion = config["training"], config["loss"], config["diffusion"]
    if min(training["max_steps"], training["batch_size"], training["grad_accumulation"],
           training["log_every"], training["validate_every"], training["validation_batches"]) < 1:
        raise ValueError("ELF 训练步数、batch、累积与验证间隔必须为正")
    if min(loss.values()) < 0 or sum(loss.values()) == 0:
        raise ValueError("ELF loss 权重不能为负，且至少启用一项")
    stopping = training["early_stopping"]
    if stopping["metric"] not in METRIC_KEYS or stopping["patience"] < 1 or stopping["min_steps"] < 0 or stopping["min_delta"] < 0:
        raise ValueError("ELF 早停指标或 patience/min_steps/min_delta 无效")
    decoder_probability = diffusion["decoder_probability"]
    if not 0 <= decoder_probability <= 1 or not 0 <= training["self_condition_probability"] <= 1:
        raise ValueError("ELF decoder/SC 概率必须在 [0,1]")
    if (decoder_probability == 0 and loss["kd_weight"] + loss["ce_weight"] > 0
            or decoder_probability == 1 and loss["flow_weight"] + loss["native_flow_weight"] > 0):
        raise ValueError("ELF loss 启用的分支必须有非零采样概率")
    length = config["model"]["architecture"]["max_length"]
    if not 0 < config["data"]["sequence_length"] <= length or not 0 < config["evaluation"]["sequence_length"] <= length:
        raise ValueError("ELF 实际序列不能超过预训练 RoPE 长度")
    if min(config["evaluation"]["num_samples"], config["evaluation"]["batch_size"], config["evaluation"]["num_steps"]) < 1:
        raise ValueError("ELF 生成样本、batch 和采样步数必须为正")
    return config


def run_experiment(config, command="all", checkpoint=None, allow_distributed=True):
    communication = config["distributed"]
    owns_group = distributed.setup(config["model"]["device"], allow_distributed, communication["control_timeout_seconds"])
    run_dir = Path(distributed.broadcast_main(str(create_run_dir(config)) if distributed.is_main() else None))
    progress.configure(run_dir, distributed.rank())
    seed_everything(config["run"]["seed"] + distributed.rank())
    device = torch.device(config["model"]["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前机器没有 CUDA；实际验证请在服务器运行")
        device = torch.device("cuda", torch.cuda.current_device())
    if distributed.is_main():
        print(f"[run] {run_dir} (ELF, world_size={distributed.world_size()})", flush=True)
        write_json(run_dir / "distributed.json", {"world_size": distributed.world_size(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE")})
    try:
        if command == "check-distributed":
            if distributed.world_size() < 2:
                raise ValueError("通信检查需要 NGPU>=2")
            distributed.setup_tensor_group(device, communication["tensor_timeout_seconds"])
        else:
            encoder_path = models.resolve_encoder(config)
            tokenizer = models.load_tokenizer(encoder_path)
            if tokenizer.vocab_size != config["model"]["architecture"]["vocab_size"]:
                raise ValueError("T5 tokenizer 与 ELF decoder 词表不一致")
            if distributed.is_main():
                with progress.phase("elf.data.prepare"):
                    data.prepare_data(config, tokenizer)
            distributed.barrier()
            if command != "prepare":
                distributed.setup_tensor_group(device, communication["tensor_timeout_seconds"])
                encoder = models.load_encoder(encoder_path, config, device)
                teacher = models.load_teacher(config, device)
                distributed.barrier()
                if distributed.is_main():
                    write_json(run_dir / "encoder.json", {"model": config["model"]["encoder"],
                        "parameters": models.parameter_counts(encoder), "frozen": True,
                        "latent_mean": config["diffusion"]["latent_mean"], "latent_std": config["diffusion"]["latent_std"]})
                if command in ("train", "all"):
                    student = models.build_student(teacher, config)
                    distributed.barrier()
                    checkpoint = train(teacher, student, tokenizer, config, run_dir,
                        loss_fn=partial(batch_loss, encoder=encoder), loader_fn=data.make_loader,
                        checkpoint_fn=models.save_checkpoint, counts_fn=models.parameter_counts,
                        metric_keys=METRIC_KEYS)
                    del student
                if command in ("evaluate", "all"):
                    if checkpoint is None:
                        raise ValueError("ELF evaluate 需要 --checkpoint")
                    evaluate(teacher, encoder, tokenizer, project_path(checkpoint), config, run_dir)
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "completed", "command": command,
                                                 "checkpoint": str(checkpoint) if checkpoint else None})
        distributed.barrier()
    except Exception as error:
        progress.emit("elf.run.failed", error=str(error))
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        if owns_group:
            distributed.close()
    return run_dir


def main(default_config="configs/elf_b_kd.yaml", allow_distributed=True):
    parser = argparse.ArgumentParser(description="真实 ELF-B/L 容量蒸馏")
    parser.add_argument("command", nargs="?", choices=["prepare", "train", "evaluate", "all", "check-distributed"], default="all")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoint", help="本项目 ELF student checkpoint 目录或 safetensors")
    args = parser.parse_args()
    if args.command == "evaluate" and args.checkpoint is None:
        parser.error("evaluate 必须指定 --checkpoint")
    run_experiment(load_config(args.config, args.overrides), args.command, args.checkpoint, allow_distributed)


if __name__ == "__main__":
    main()
