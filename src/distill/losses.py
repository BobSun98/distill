"""Token posterior 蒸馏与真实 token CE，共享有效 token 的归一化规则。"""

import torch
import torch.nn.functional as F


def masked_mean(values, mask=None):
    if mask is None:
        return values.mean()
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


def distillation_loss(student_logits, teacher_logits, token_ids, settings, mask=None):
    log_student = F.log_softmax(student_logits.float(), dim=-1)
    ce = None
    if token_ids is not None:
        ce = masked_mean(-log_student.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1), mask)
    elif settings["ce_weight"]:
        raise ValueError("启用 CE 时需要真实 token；on-policy 无标签状态可只使用 KD")
    kd = None
    if teacher_logits is not None:
        log_teacher = F.log_softmax(teacher_logits.detach().float(), dim=-1)
        # forward KL 按词表求和、按 token 平均，不用会随序列长度放大的 batchmean。
        kd = masked_mean((log_teacher.exp() * (log_teacher - log_student)).sum(-1), mask)
    # KD-only 不依赖干净 token，后续 student rollout 无标签状态可直接复用。
    loss = log_student.new_zeros(())
    if settings["ce_weight"]:
        loss = loss + settings["ce_weight"] * ce
    if settings["kd_weight"]:
        if kd is None:
            raise ValueError("启用 KD 时必须提供 teacher logits")
        loss = loss + settings["kd_weight"] * kd
    metrics = {"loss": float(loss.detach())}
    if ce is not None:
        metrics["ce"] = float(ce.detach())
    if kd is not None:
        metrics["kl"] = float(kd.detach())
    return loss, metrics
