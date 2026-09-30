"""蒸馏条件状态。后续 rollout 可以直接产出相同状态，复用 posterior 与 loss。"""

from dataclasses import dataclass

import torch


@dataclass
class DistillationState:
    z: torch.Tensor
    gamma: torch.Tensor
    self_cond: torch.Tensor


def posterior_logits(model, state):
    return model(noisy_embeds=state.z, timesteps=state.gamma,
                 x_self_cond=state.self_cond, return_dict=False)


def add_noise(clean_embeddings, gamma, noise):
    alpha = torch.sigmoid(-gamma).sqrt()[:, None, None]
    sigma = torch.sigmoid(gamma).sqrt()[:, None, None]
    return alpha * clean_embeddings + sigma * noise


@torch.no_grad()
def off_policy_state(teacher, student, token_ids, self_condition_probability):
    clean = teacher._embed_tokens(token_ids)
    q = torch.rand(len(token_ids), device=clean.device)
    q = q.clamp(teacher.proposal.cutoff, 1 - teacher.proposal.cutoff)
    gamma = teacher.proposal(q).detach()
    # 同一份 z 同时给 teacher/student；不分别调用 input_ids 路径生成两份独立噪声。
    z = add_noise(clean, gamma, torch.randn_like(clean))
    self_cond = torch.zeros_like(z)
    selected = torch.rand(len(token_ids), device=z.device) < self_condition_probability
    if selected.any():
        was_training = student.training
        student.eval()
        try:
            logits = posterior_logits(student, DistillationState(z, gamma, self_cond))
            prediction = teacher._embed_tokens(logits.float().softmax(dim=-1))
            self_cond = torch.where(selected[:, None, None], prediction, self_cond)
        finally:
            student.train(was_training)
    # SC 来自 student 的无梯度预估，teacher 查询时使用完全相同的条件。
    return DistillationState(z.detach(), gamma, self_cond.detach())
