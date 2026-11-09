"""Token posterior 蒸馏与真实 token CE，共享有效 token 的归一化规则。"""

import torch
import torch.nn.functional as F


def masked_mean(values, mask=None):
    if mask is None:
        return values.mean()
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


def langflow_velocity_mse(student_prediction, teacher_prediction, gamma):
    # VP gamma 路径下同一状态的速度差为 -alpha/2 * (xhat_S-xhat_T)。
    # 返回每个 token 的 MSE；先对 latent 维度平均，再由调用方按有效 token 平均。
    return 0.25 * torch.sigmoid(-gamma)[:, None] * (
        student_prediction.float() - teacher_prediction.detach().float()).square().mean(-1)


def distillation_loss(student_logits, teacher_logits, token_ids, settings, mask=None, *, gamma=None, embedding=None):
    log_student = F.log_softmax(student_logits.float(), dim=-1)
    ce = None
    if token_ids is not None:
        ce = masked_mean(-log_student.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1), mask)
    elif settings["ce_weight"]:
        raise ValueError("启用 CE 时需要真实 token；on-policy 无标签状态可只使用 KD")
    kd = mse = None
    teacher_probabilities = None
    if teacher_logits is not None:
        log_teacher = F.log_softmax(teacher_logits.detach().float(), dim=-1)
        teacher_probabilities = log_teacher.exp()
        # forward KL 按词表求和、按 token 平均，不用会随序列长度放大的 batchmean。
        kd = masked_mean((teacher_probabilities * (log_teacher - log_student)).sum(-1), mask)
    if settings.get("mse_weight", 0):
        if teacher_probabilities is None or gamma is None or embedding is None:
            raise ValueError("velocity MSE 需要 teacher logits、同一 gamma 和共享 embedding")
        # KL 与速度项复用同一份 teacher posterior；这里是 denoised embedding 的速度差，
        # 不是 logits MSE，也不是无时间权重的 embedding MSE。所有概率/归约保持 FP32。
        coordinates = embedding.detach().float()
        with torch.no_grad():
            teacher_prediction = teacher_probabilities @ coordinates
        student_prediction = log_student.exp() @ coordinates
        mse = masked_mean(langflow_velocity_mse(student_prediction, teacher_prediction, gamma.detach()), mask)
    # KD-only 不依赖干净 token，后续 student rollout 无标签状态可直接复用。
    loss = log_student.new_zeros(())
    if settings["ce_weight"]:
        loss = loss + settings["ce_weight"] * ce
    if settings["kd_weight"]:
        if kd is None:
            raise ValueError("启用 KD 时必须提供 teacher logits")
        loss = loss + settings["kd_weight"] * kd
    if settings.get("mse_weight", 0):
        loss = loss + settings["mse_weight"] * mse
    metrics = {"loss": float(loss.detach())}
    if ce is not None:
        metrics["ce"] = float(ce.detach())
    if kd is not None:
        metrics["kl"] = float(kd.detach())
    if mse is not None:
        metrics["mse"] = float(mse.detach())
    return loss, metrics
