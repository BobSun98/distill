"""单进程训练入口：状态构造、posterior 查询与优化分开，便于 IDE 调试。"""

import json
import random
import time

import torch

from .config import write_json
from .data import make_loader
from .losses import distillation_loss
from .models import parameter_counts
from .states import off_policy_state, posterior_logits


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_loss(teacher, student, token_ids, config, query_teacher=None):
    state = off_policy_state(teacher, student, token_ids,
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
def validate(teacher, student, loader, config):
    device = next(student.parameters()).device
    was_training = student.training
    student.eval()
    sums, tokens = {}, 0
    # 固定验证噪声，并在退出后恢复 RNG；验证不会改变后续训练状态的随机序列。
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config["evaluation"]["seed"])
            for index, batch in enumerate(loader):
                if index >= config["training"]["validation_batches"]:
                    break
                batch = batch.to(device)
                _, metrics = batch_loss(teacher, student, batch, config, query_teacher=True)
                count = batch.numel()
                tokens += count
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + value * count
    finally:
        student.train(was_training)
    if not tokens:
        raise ValueError("没有可用的验证 batch")
    # ce 是加噪状态上的 denoising CE，不是完整生成模型的 NLL/PPL。
    return {key: value / tokens for key, value in sums.items()}


def save_checkpoint(student, tokenizer, run_dir, step, config):
    path = run_dir / "checkpoints" / f"step_{step:06d}"
    student.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    write_json(path / "distillation.json", {"step": step, "teacher": config["model"]["teacher"],
                                           "keep_layers": config["model"]["keep_layers"],
                                           "loss": config["loss"]})
    return path


def train(teacher, student, tokenizer, config, run_dir):
    settings = config["training"]
    train_loader, valid_loader = make_loader(config, "train"), make_loader(config, "valid")
    device = next(student.parameters()).device
    student.train()
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad),
                                 lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    write_json(run_dir / "models.json", {"teacher": parameter_counts(teacher),
                                        "student": parameter_counts(student),
                                        "keep_layers": config["model"]["keep_layers"]})
    batches = iter(train_loader)
    started = time.perf_counter()
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as log:
        def record(event):
            event["elapsed_seconds"] = time.perf_counter() - started
            log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            print(f"[train] {event}", flush=True)

        record({"phase": "valid", "step": 0,
                **validate(teacher, student, valid_loader, config)})
        for step in range(1, settings["max_steps"] + 1):
            optimizer.zero_grad(set_to_none=True)
            sums, tokens = {}, 0
            # 先取齐一个优化步的 micro-batches；尾批较小时仍按真实 token 数加权。
            micro_batches = []
            for _ in range(settings["grad_accumulation"]):
                try:
                    batch = next(batches)
                except StopIteration:
                    batches = iter(train_loader)
                    batch = next(batches)
                micro_batches.append(batch)
            step_tokens = sum(batch.numel() for batch in micro_batches)
            for batch in micro_batches:
                batch = batch.to(device)
                loss, metrics = batch_loss(teacher, student, batch, config)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"step {step}: loss 非有限值")
                count = batch.numel()
                (loss * count / step_tokens).backward()
                tokens += count
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + value * count
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in student.parameters() if p.requires_grad),
                settings["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % settings["log_every"] == 0 or step == settings["max_steps"]:
                record({"phase": "train", "step": step,
                        **{key: value / tokens for key, value in sums.items()},
                        "grad_norm": float(grad_norm), "tokens": tokens})
            if step % settings["validate_every"] == 0 or step == settings["max_steps"]:
                record({"phase": "valid", "step": step,
                        **validate(teacher, student, valid_loader, config)})
            if settings["save_every"] and step % settings["save_every"] == 0 and step < settings["max_steps"]:
                save_checkpoint(student, tokenizer, run_dir, step, config)
    checkpoint = save_checkpoint(student, tokenizer, run_dir, settings["max_steps"], config)
    write_json(run_dir / "training.json", {"checkpoint": str(checkpoint),
                                          "steps": settings["max_steps"]})
    print(f"[train] checkpoint: {checkpoint}", flush=True)
    return checkpoint
