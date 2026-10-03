"""采集、训练、评测在同一进程调用；CLI 与 IDE debug 共用此入口。"""

import argparse
import os
from pathlib import Path

import torch

from .config import create_run_dir, load_config, project_path, write_json
from .data import prepare_data
from .evaluate import evaluate
from .models import build_student, load_teacher, load_tokenizer
from .train import seed_everything, train
from . import distributed, progress


def run_experiment(config, command="all", checkpoint=None, allow_distributed=True):
    communication = config["distributed"]
    owns_group = distributed.setup(config["model"]["device"], allow_distributed,
                                   communication["control_timeout_seconds"])
    run_dir = Path(distributed.broadcast_main(str(create_run_dir(config)) if distributed.is_main() else None))
    seed_everything(config["run"]["seed"] + distributed.rank())
    progress.configure(run_dir, distributed.rank())
    progress.emit("run.start", world_size=distributed.world_size(), torch_version=torch.__version__,
                  local_rank=int(os.environ.get("LOCAL_RANK", "0")), device=config["model"]["device"])
    if torch.device(config["model"]["device"]).type == "cuda" and torch.cuda.is_available():
        progress.emit("cuda.runtime", logical_device=torch.cuda.current_device(),
                      gpu=torch.cuda.get_device_name(), cuda_version=torch.version.cuda,
                      nccl_version=torch.cuda.nccl.version())
    if distributed.is_main():
        print(f"[run] {run_dir} (world_size={distributed.world_size()})", flush=True)
        write_json(run_dir / "distributed.json", {
            "world_size": distributed.world_size(),
            "control_backend": torch.distributed.get_backend() if distributed.active() else None,
            "tensor_backend": ("nccl" if torch.device(config["model"]["device"]).type == "cuda" else "gloo") if distributed.active() else None,
            "timeouts": communication,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
        })
    try:
        if command == "check-distributed":
            if distributed.world_size() < 2:
                raise ValueError("通信诊断需要至少两个进程，请设置 NGPU>=2")
            distributed.setup_tensor_group(config["model"]["device"], communication["tensor_timeout_seconds"])
            if distributed.is_main():
                write_json(run_dir / "status.json", {"status": "completed", "command": command})
            progress.emit("communication.check.completed")
            return run_dir
        # 同一台服务器只有 rank 0 写预处理缓存，其余 rank 等待并直接读取。
        if distributed.is_main():
            with progress.phase("data.prepare"):
                tokenizer = load_tokenizer(config)
                prepare_data(config, tokenizer)
        with progress.phase("data.wait_ready"):
            distributed.barrier()
        if not distributed.is_main():
            tokenizer = load_tokenizer(config)
        if command == "prepare":
            if distributed.is_main():
                write_json(run_dir / "status.json", {"status": "completed", "command": command})
            return run_dir
        distributed.setup_tensor_group(config["model"]["device"], communication["tensor_timeout_seconds"])
        with progress.phase("teacher.load"):
            teacher = load_teacher(config)
        with progress.phase("models.wait_teacher"):
            distributed.barrier()
        if len(tokenizer) != teacher.config.vocab_size:
            raise ValueError("tokenizer 词表大小与模型不一致，请使用 teacher 对应的 tokenizer")
        if config["data"]["sequence_length"] > teacher.config.model_length:
            raise ValueError("训练序列长度超过 teacher 配置的 model_length")
        if command in ("train", "all"):
            with progress.phase("student.build"):
                student = build_student(teacher, config["model"]["keep_layers"])
            with progress.phase("models.wait_student"):
                distributed.barrier()
            checkpoint = train(teacher, student, tokenizer, config, run_dir)
            del student
        if command in ("evaluate", "all"):
            if checkpoint is None:
                raise ValueError("evaluate 需要通过 --checkpoint 指定 student 目录")
            evaluate(teacher, tokenizer, project_path(checkpoint), config, run_dir)
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "completed", "command": command,
                                                 "checkpoint": str(checkpoint) if checkpoint else None})
        distributed.barrier()
        progress.emit("run.completed")
    except Exception as error:
        progress.emit("run.failed", error=f"{type(error).__name__}: {error}")
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "failed", "command": command,
                                                 "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if owns_group:
            distributed.close()
    return run_dir


def main(default_config="configs/owt_kd.yaml", default_command="all", allow_distributed=True):
    parser = argparse.ArgumentParser(description="OWT/LM1B 连续 DLM 容量蒸馏")
    parser.add_argument("command", nargs="?", choices=["prepare", "train", "evaluate", "all", "check-distributed"],
                        default=default_command)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoint", help="训练后的 student checkpoint 目录")
    args = parser.parse_args()
    if args.command == "evaluate" and args.checkpoint is None:
        parser.error("evaluate 必须指定 --checkpoint")
    config = load_config(args.config, args.overrides)
    run_experiment(config, args.command, args.checkpoint, allow_distributed)
