"""单卡/DDP 共用训练流程；debug 在当前进程直接进入各工程函数。"""

import json
import random
import time
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel

from .config import write_json
from .data import make_loader
from .losses import distillation_loss
from .models import parameter_counts
from .states import off_policy_state, posterior_logits
from .early_stopping import PlateauMonitor
from . import distributed, progress


def move_batch(batch, device):
    if isinstance(batch, dict):
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return batch.to(device, non_blocking=True)


def batch_tokens(batch):
    if isinstance(batch, dict):
        return int(batch["attention_mask"].sum().item())
    return batch.numel()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_loss(teacher, student, token_ids, config, query_teacher=None):
    # SC 的无梯度前向不能经过 DDP：各 rank 的 SC 开关不同，通信次数必须一致。
    state = off_policy_state(teacher, distributed.unwrap(student), token_ids,
                             config["training"]["self_condition_probability"])
    if query_teacher is None:
        query_teacher = config["loss"]["kd_weight"] > 0
    teacher_logits = None
    if query_teacher:
        with torch.no_grad():
            teacher_logits = posterior_logits(teacher, state)
    student_logits = posterior_logits(student, state)
    return distillation_loss(student_logits, teacher_logits, token_ids, config["loss"])


@torch.no_grad()
def validate(teacher, student, loader, config, loss_fn=batch_loss, metric_keys=("loss", "ce", "kl")):
    student = distributed.unwrap(student)
    device = next(student.parameters()).device
    was_training = student.training
    student.eval()
    sums, tokens = {key: 0.0 for key in metric_keys}, 0
    # 固定验证噪声，并在退出后恢复 RNG；验证不会改变后续训练状态的随机序列。
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config["evaluation"]["seed"] + distributed.rank())
            for index, batch in enumerate(loader):
                if index >= config["training"]["validation_batches"]:
                    break
                batch = move_batch(batch, device)
                if index == 0:
                    progress.emit("validation.first_batch.begin", tokens=batch_tokens(batch))
                _, metrics = loss_fn(teacher, student, batch, config, query_teacher=True)
                if index == 0:
                    progress.emit("validation.first_batch.end", **metrics)
                count = batch_tokens(batch)
                tokens += count
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + value * count
    finally:
        student.train(was_training)
    # ce 是加噪状态上的 denoising CE，不是完整生成模型的 NLL/PPL。
    with progress.phase("validation.reduce", local_tokens=tokens):
        return distributed.mean_metrics(sums, tokens, device)


def save_checkpoint(student, tokenizer, run_dir, step, config, name=None):
    path = run_dir / "checkpoints" / (name or f"step_{step:06d}")
    if distributed.is_main():
        progress.emit("checkpoint.save.begin", step=step, path=str(path))
        distributed.unwrap(student).save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)
        write_json(path / "distillation.json", {"step": step, "teacher": config["model"]["teacher"],
                                               "keep_layers": config["model"]["keep_layers"],
                                               "loss": config["loss"]})
        progress.emit("checkpoint.save.end", step=step)
    with progress.phase("checkpoint.wait_ready", step=step):
        distributed.barrier()
    return path


def train(teacher, student, tokenizer, config, run_dir, *, loss_fn=batch_loss,
          loader_fn=make_loader, checkpoint_fn=save_checkpoint, counts_fn=parameter_counts,
          metric_keys=("loss", "ce", "kl")):
    settings = config["training"]
    stopping = settings["early_stopping"]
    monitor = PlateauMonitor(stopping) if stopping["enabled"] and distributed.is_main() else None
    best_state = None
    stop_reason = "max_steps"
    with progress.phase("data.loaders"):
        train_loader, valid_loader = loader_fn(config, "train"), loader_fn(config, "valid")
    device = next(student.parameters()).device
    student.train()
    if distributed.world_size() > 1:
        with progress.phase("ddp.init", device=str(device)):
            student = DistributedDataParallel(
                student, device_ids=[device.index] if device.type == "cuda" else None,
                process_group=distributed.tensor_group(), broadcast_buffers=False)
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad),
                                 lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    if distributed.is_main():
        write_json(run_dir / "models.json", {"teacher": counts_fn(teacher),
                    "student": counts_fn(distributed.unwrap(student)),
                    "keep_layers": config["model"]["keep_layers"],
                    "world_size": distributed.world_size(),
                    "per_device_batch_size": settings["batch_size"],
                    "grad_accumulation": settings["grad_accumulation"],
                    "global_batch_size": distributed.world_size() * settings["batch_size"] * settings["grad_accumulation"]})
    epoch = 0
    batches = iter(train_loader)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log_context = ((run_dir / "metrics.jsonl").open("a", encoding="utf-8")
                   if distributed.is_main() else nullcontext(None))
    with log_context as log:
        def record(event):
            if not distributed.is_main():
                return
            event["elapsed_seconds"] = time.perf_counter() - started
            log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            print(f"[train] {event}", flush=True)

        def check_plateau(metrics, step):
            if not stopping["enabled"]:
                return None
            # rank 0 基于全局验证结果决定保存/停止，广播后各 rank 走相同分支。
            decision = monitor.update(metrics[stopping["metric"]], step) if distributed.is_main() else None
            decision = distributed.broadcast_main(decision)
            record({"phase": "early_stopping", "step": step, **decision})
            if decision["new_best"]:
                checkpoint_fn(student, tokenizer, run_dir, step, config, name="best")
            return decision

        with progress.phase("validation", step=0):
            metrics = validate(teacher, student, valid_loader, config, loss_fn, metric_keys)
            record({"phase": "valid", "step": 0, **metrics})
            best_state = check_plateau(metrics, 0)
        for step in range(1, settings["max_steps"] + 1):
            visible_step = step == 1 or step % settings["log_every"] == 0 or step == settings["max_steps"]
            if visible_step:
                progress.emit("train.step.begin", step=step)
            optimizer.zero_grad(set_to_none=True)
            sums, tokens = {}, 0
            # 先取齐一个优化步的 micro-batches；尾批较小时仍按真实 token 数加权。
            micro_batches = []
            for _ in range(settings["grad_accumulation"]):
                try:
                    batch = next(batches)
                except StopIteration:
                    epoch += 1
                    if isinstance(train_loader.sampler, torch.utils.data.DistributedSampler):
                        train_loader.sampler.set_epoch(epoch)
                    batches = iter(train_loader)
                    batch = next(batches)
                micro_batches.append(batch)
            step_tokens = sum(batch_tokens(batch) for batch in micro_batches)
            global_tokens = distributed.sum_tensor(step_tokens, device).item()
            for index, batch in enumerate(micro_batches):
                batch = move_batch(batch, device)
                accumulating = index < len(micro_batches) - 1
                context = (student.no_sync()
                           if isinstance(student, DistributedDataParallel) and accumulating
                           else nullcontext())
                # 累积期间只在最后一次 backward 通信；DDP 平均梯度后仍是全局 token 平均。
                with context:
                    if step == 1:
                        progress.emit("train.micro_batch.begin", step=step, micro_batch=index + 1)
                    loss, metrics = loss_fn(teacher, student, batch, config)
                    if step == 1:
                        progress.emit("train.micro_batch.forward_done", step=step, micro_batch=index + 1)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"step {step}: loss 非有限值")
                    count = batch_tokens(batch)
                    (loss * count * distributed.world_size() / global_tokens).backward()
                    if step == 1:
                        progress.emit("train.micro_batch.backward_done", step=step, micro_batch=index + 1)
                tokens += count
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + value * count
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in student.parameters() if p.requires_grad),
                settings["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            if visible_step:
                event = {"phase": "train", "step": step,
                        **distributed.mean_metrics(sums, tokens, device),
                        "grad_norm": float(grad_norm), "tokens": int(global_tokens), "epoch": epoch}
                if device.type == "cuda" and distributed.is_main():
                    event["rank0_peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
                record(event)
                progress.emit("train.step.end", step=step)
            if step % settings["validate_every"] == 0 or step == settings["max_steps"]:
                with progress.phase("validation", step=step):
                    metrics = validate(teacher, student, valid_loader, config, loss_fn, metric_keys)
                    record({"phase": "valid", "step": step, **metrics})
                    best_state = check_plateau(metrics, step)
                if best_state is not None and best_state["should_stop"] and step < settings["max_steps"]:
                    stop_reason = "plateau"
                    progress.emit("train.early_stop", step=step, metric=stopping["metric"],
                                  best_step=best_state["best_step"], best_value=best_state["best_value"])
                    break
            if settings["save_every"] and step % settings["save_every"] == 0 and step < settings["max_steps"]:
                checkpoint_fn(student, tokenizer, run_dir, step, config)
    # 保留停止时的权重；all 后续评测选择验证指标最佳模型，而不是默认使用最后一步。
    last_checkpoint = checkpoint_fn(student, tokenizer, run_dir, step, config)
    checkpoint = run_dir / "checkpoints" / "best" if stopping["enabled"] else last_checkpoint
    if distributed.is_main():
        write_json(run_dir / "training.json", {"checkpoint": str(checkpoint),
                    "last_checkpoint": str(last_checkpoint), "steps": step,
                    "max_steps": settings["max_steps"], "stop_reason": stop_reason,
                    "early_stopping": stopping,
                    "best_step": best_state["best_step"] if best_state else None,
                    "best_value": best_state["best_value"] if best_state else None})
        print(f"[train] stopped at step {step}: {stop_reason}", flush=True)
        print(f"[train] checkpoint: {checkpoint}", flush=True)
    return checkpoint
