"""单机 DDP 的必要通信工具；单进程调用保持原有行为。"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


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


def setup(device_name, allow_distributed=True):
    processes = max(int(os.environ.get("WORLD_SIZE", "1")), world_size())
    if not allow_distributed and processes > 1:
        raise RuntimeError("debug 入口只允许单进程单卡，请直接运行 debug/debug_pipeline.py")
    if processes == 1 or active():
        return False
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # 数据首次下载可能较久；只在 rank 0 准备，其余进程等待共享缓存。
    dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                            timeout=timedelta(minutes=120))
    return True


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
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def mean_metrics(sums, tokens, device):
    keys = sorted(sums)
    combined = sum_tensor([tokens] + [sums[key] for key in keys], device).tolist()
    if combined[0] == 0:
        raise ValueError("没有可用的 token")
    return {key: value / combined[0] for key, value in zip(keys, combined[1:])}


def close():
    if active():
        dist.destroy_process_group()
