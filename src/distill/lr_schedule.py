"""按 optimizer 更新步调度 LR；未配置时保持原有恒定学习率。"""

import math

import torch


def validate_lr_schedule(settings):
    if not math.isfinite(settings["learning_rate"]) or settings["learning_rate"] <= 0:
        raise ValueError("learning_rate 必须是有限正数")
    schedule = settings.get("lr_scheduler", {"name": "constant"})
    if schedule["name"] not in ("constant", "cosine"):
        raise ValueError("lr_scheduler.name 只支持 constant 或 cosine")
    if schedule["name"] == "cosine" and not 0 < schedule["final_ratio"] <= 1:
        raise ValueError("cosine final_ratio 必须在 (0,1] 内")


def make_lr_scheduler(optimizer, settings):
    validate_lr_schedule(settings)
    schedule = settings.get("lr_scheduler", {"name": "constant"})
    if schedule["name"] == "constant":
        return None
    # 第 1 次更新用初始 LR，第 max_steps 次更新恰好用 initial_lr * final_ratio。
    # 因而只有 max_steps-1 次调度变化；每次 optimizer 更新后再推进一次。
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=max(settings["max_steps"] - 1, 1),
        eta_min=settings["learning_rate"] * schedule["final_ratio"])
