"""CE 使用 teacher soft/hard 标签；MSE 比较相同状态下的 velocity。"""

import torch
import torch.nn.functional as F

from .. import distributed
from .rollout import StateBuffer

METRIC_KEYS = ("loss", "ce", "mse", "teacher_entropy", "error")


def teacher_ce(student_logits, teacher_logits, target):
    log_student = student_logits.float().log_softmax(-1)
    with torch.no_grad():
        log_teacher = teacher_logits.float().log_softmax(-1)
        probabilities = log_teacher.exp()
        entropy = -(probabilities * log_teacher).sum(-1)
    if target == "hard":
        ce = -log_student.gather(-1, teacher_logits.argmax(-1, keepdim=True)).squeeze(-1)
        entropy = torch.zeros_like(ce)
    else:
        ce = -(probabilities * log_student).sum(-1)
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
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                enabled=state["z"].is_cuda and config["model"]["use_bf16"]):
            if config["model"]["family"] == "langflow":
                inputs = dict(noisy_embeds=state["z"], timesteps=state["time"],
                              x_self_cond=state["self_cond"], return_dict=False)
                with torch.no_grad():
                    target = teacher(**inputs)
                prediction = student(**inputs)
                zero = prediction.sum() * 0
                ce = entropy = mse = zero
                if weights["ce_weight"]:
                    values, entropies = teacher_ce(prediction, target, weights["ce_target"])
                    ce, entropy = values.mean(), entropies.mean()
                if weights["mse_weight"]:
                    with torch.no_grad():
                        teacher_x = teacher._embed_tokens(target.float().softmax(-1))
                    student_x = raw._embed_tokens(prediction.float().softmax(-1))
                    mse = langflow_velocity_mse(student_x, teacher_x, state["time"]).mean()
            else:
                x = torch.cat([state["z"], state["self_cond"]], -1)
                decoder = state["decoder"] if weights["ce_weight"] else None
                inputs = dict(t=state["time"], self_cond_cfg_scale=state["sc_scale"],
                              decoder_step_active=decoder, deterministic=True)
                with torch.no_grad():
                    target, teacher_logits = teacher(x, **inputs)
                prediction, logits = student(x, **inputs)
                zero = prediction.sum() * 0
                ce = entropy = mse = zero
                if weights["mse_weight"]:
                    difference = (prediction.float() - target.float()) / (1 - state["time"][:, None, None]).clamp_min(config["diffusion"]["t_eps"])
                    mse = (difference.square().mean(-1) * (~state["decoder"])[:, None]).mean()
                if weights["ce_weight"]:
                    values, entropies = teacher_ce(logits, teacher_logits, weights["ce_target"])
                    ce = (values * state["decoder"][:, None]).mean()
                    entropy = (entropies * state["decoder"][:, None]).mean()
        loss = weights["ce_weight"] * ce + weights["mse_weight"] * mse
        # Soft CE 包含会随访问状态变化的 teacher entropy；早停看扣除该常数的拟合误差。
        error = weights["ce_weight"] * (ce - entropy) + weights["mse_weight"] * mse
        return loss, {"loss": float(loss.detach()), "ce": float(ce.detach()), "mse": float(mse.detach()),
                      "teacher_entropy": float(entropy.detach()), "error": float(error.detach())}
