"""同状态下的 teacher CE / forward KL 与 velocity MSE。"""

import torch

from .. import distributed
from .rollout import StateBuffer

METRIC_KEYS = ("loss", "ce", "kl", "mse", "teacher_entropy", "error")


def posterior_terms(student_logits, teacher_logits, target, *, include_ce=True, include_kl=False):
    log_student = student_logits.float().log_softmax(-1)
    with torch.no_grad():
        log_teacher = teacher_logits.float().log_softmax(-1)
        probabilities = log_teacher.exp()
    zero = log_student[..., 0] * 0
    ce = kl = entropy = zero
    if include_ce:
        if target == "hard":
            ce = -log_student.gather(-1, teacher_logits.argmax(-1, keepdim=True)).squeeze(-1)
        else:
            ce = -(probabilities * log_student).sum(-1)
            entropy = -(probabilities * log_teacher).sum(-1)
    if include_kl:
        # 与第一阶段一致：KL(teacher || student)，词表求和，之后按 token 求平均。
        kl = (probabilities * (log_teacher - log_student)).sum(-1)
    return ce, kl, entropy


def teacher_ce(student_logits, teacher_logits, target):
    ce, _, entropy = posterior_terms(student_logits, teacher_logits, target)
    return ce, entropy


def langflow_velocity_mse(student_prediction, teacher_prediction, time):
    # VP gamma 路径：v_S-v_T = -alpha/2 * (xhat_S-xhat_T)。
    return 0.25 * torch.sigmoid(-time)[:, None] * (student_prediction.float() - teacher_prediction.float()).square().mean(-1)


class OnPolicyLoss:
    def __init__(self, config):
        self.buffer = StateBuffer(config)

    def __call__(self, teacher, student, batch, config, query_teacher=None):
        raw = distributed.unwrap(student)
        state = self.buffer.sample(raw, len(batch["attention_mask"]), raw.training)
        weights = config["loss"]
        kl_weight = weights.get("kl_weight", 0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                enabled=state["z"].is_cuda and config["model"]["use_bf16"]):
            if config["model"]["family"] == "langflow":
                inputs = dict(noisy_embeds=state["z"], timesteps=state["time"],
                              x_self_cond=state["self_cond"], return_dict=False)
                with torch.no_grad():
                    target = teacher(**inputs)
                prediction = student(**inputs)
                zero = prediction.sum() * 0
                ce = kl = entropy = mse = zero
                if weights["ce_weight"] or kl_weight:
                    values, divergences, entropies = posterior_terms(prediction, target, weights["ce_target"],
                        include_ce=bool(weights["ce_weight"]), include_kl=bool(kl_weight))
                    ce, kl, entropy = values.mean(), divergences.mean(), entropies.mean()
                if weights["mse_weight"]:
                    with torch.no_grad():
                        teacher_x = teacher._embed_tokens(target.float().softmax(-1))
                    student_x = raw._embed_tokens(prediction.float().softmax(-1))
                    mse = langflow_velocity_mse(student_x, teacher_x, state["time"]).mean()
            else:
                x = torch.cat([state["z"], state["self_cond"]], -1)
                decoder = state["decoder"] if weights["ce_weight"] or kl_weight else None
                inputs = dict(t=state["time"], self_cond_cfg_scale=state["sc_scale"],
                              decoder_step_active=decoder, deterministic=True)
                with torch.no_grad():
                    target, teacher_logits = teacher(x, **inputs)
                prediction, logits = student(x, **inputs)
                zero = prediction.sum() * 0
                ce = kl = entropy = mse = zero
                if weights["mse_weight"]:
                    difference = (prediction.float() - target.float()) / (1 - state["time"][:, None, None]).clamp_min(config["diffusion"]["t_eps"])
                    mse = (difference.square().mean(-1) * (~state["decoder"])[:, None]).mean()
                if weights["ce_weight"] or kl_weight:
                    values, divergences, entropies = posterior_terms(logits, teacher_logits, weights["ce_target"],
                        include_ce=bool(weights["ce_weight"]), include_kl=bool(kl_weight))
                    ce = (values * state["decoder"][:, None]).mean()
                    kl = (divergences * state["decoder"][:, None]).mean()
                    entropy = (entropies * state["decoder"][:, None]).mean()
        loss = weights["ce_weight"] * ce + kl_weight * kl + weights["mse_weight"] * mse
        # Soft CE 包含会随访问状态变化的 teacher entropy；早停看扣除该常数的拟合误差。
        error = weights["ce_weight"] * (ce - entropy) + kl_weight * kl + weights["mse_weight"] * mse
        return loss, {"loss": float(loss.detach()), "ce": float(ce.detach()), "kl": float(kl.detach()), "mse": float(mse.detach()),
                      "teacher_entropy": float(entropy.detach()), "error": float(error.detach())}
