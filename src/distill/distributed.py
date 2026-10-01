"""单机 DDP 的必要通信工具；单进程调用保持原有行为。"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from . import progress

_tensor_group = None


def active():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if active() else 0


def world_size():
    return dist.get_world_size() if active() else 1


def is_main():
    return rank() == 0


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def setup(device_name, allow_distributed=True, control_timeout_seconds=7200):
    processes = max(int(os.environ.get("WORLD_SIZE", "1")), world_size())
    if not allow_distributed and processes > 1:
        raise RuntimeError("debug 入口只允许单进程单卡，请直接运行 debug/debug_pipeline.py")
    if processes == 1 or active():
        return False
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前机器没有 CUDA；本地调试请设置 model.device=cpu")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # CPU/Gloo 处理数据、下载与文件同步；等待期间不启动 NCCL 的 GPU polling kernel。
    with progress.phase("control_group.init", backend="gloo", world_size=processes):
        dist.init_process_group(backend="gloo", timeout=timedelta(seconds=control_timeout_seconds))
    return True


def tensor_group():
    return _tensor_group


def setup_tensor_group(device, timeout_seconds=180):
    global _tensor_group
    if not active():
        return
    device = torch.device(device)
    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    # CPU 回归也使用独立 tensor 组，验证与 CUDA 相同的双组调用流程。
    with progress.phase("tensor_group.init", backend=backend, device=str(device),
                        timeout_seconds=timeout_seconds):
        _tensor_group = dist.new_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
    # 第一次 GPU collective 显式使用当前 rank 的设备，不让 NCCL barrier 猜测 GPU。
    with progress.phase("communication.all_reduce", backend=backend, device=str(device), bytes=1048576):
        probe = torch.full((262144,), float(rank() + 1), dtype=torch.float32, device=device)
        dist.all_reduce(probe, op=dist.ReduceOp.SUM, group=_tensor_group)
        expected = world_size() * (world_size() + 1) / 2
        if not torch.all(probe == expected).item():
            raise RuntimeError("通信检查的 all_reduce 结果不正确")
    with progress.phase("communication.broadcast", backend=backend, device=str(device)):
        probe.fill_(rank())
        dist.broadcast(probe, src=0, group=_tensor_group)
        if not torch.all(probe == 0).item():
            raise RuntimeError("通信检查的 broadcast 结果不正确")


def barrier():
    if active():
        dist.barrier()


def broadcast_main(value):
    values = [value]
    if active():
        dist.broadcast_object_list(values, src=0)
    return values[0]


def gather_main(value):
    if not active():
        return [value]
    values = [None] * world_size() if is_main() else None
    dist.gather_object(value, values, dst=0)
    return values


def sum_tensor(values, device):
    result = torch.as_tensor(values, dtype=torch.float64, device=device)
    if active():
        group = _tensor_group
        if result.device.type == "cuda" and group is None:
            raise RuntimeError("CUDA tensor 通信组尚未初始化")
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
    return result


def mean_metrics(sums, tokens, device):
    keys = sorted(sums)
    combined = sum_tensor([tokens] + [sums[key] for key in keys], device).tolist()
    if combined[0] == 0:
        raise ValueError("没有可用的 token")
    return {key: value / combined[0] for key, value in zip(keys, combined[1:])}


def close():
    global _tensor_group
    if active():
        if _tensor_group is not None:
            dist.destroy_process_group(_tensor_group)
            _tensor_group = None
        dist.destroy_process_group()
