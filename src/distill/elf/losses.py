"""ELF 两种训练状态：去噪速度 MSE、decoder KL/CE，同一输入查询双方。"""

from types import SimpleNamespace

import torch

from .. import distributed
from ..backbones.elf.sampling_utils import add_noise, sample_timesteps, sample_cfg_scale

METRIC_KEYS = ("loss", "flow_mse", "native_flow_mse", "decoder_kl", "decoder_ce")


def forward(model, inputs, t, sc_scale, decoder_active, config, deterministic):
    cuda = inputs.is_cuda
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cuda and config["model"]["use_bf16"]):
        # 与原版无条件训练一致：ELF 不额外屏蔽 padded latent；损失只计有效 token。
        return model(inputs, t, deterministic=deterministic,
                     self_cond_cfg_scale=sc_scale, decoder_step_active=decoder_active)


def batch_loss(teacher, student, batch, config, query_teacher=None, *, encoder):
    diffusion, weights = config["diffusion"], config["loss"]
    token_ids, valid = batch["input_ids"], batch["attention_mask"].float()
    model = distributed.unwrap(student)
    count, length = token_ids.shape
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=token_ids.is_cuda and config["model"]["use_bf16"]):
            # 冻结真实 T5，共享 latent 坐标；attention mask 与原版无条件 T5 key mask 等价。
            clean = encoder(input_ids=token_ids, attention_mask=valid).last_hidden_state.float()
        clean = (clean - diffusion["latent_mean"]) / diffusion["latent_std"]
        t = sample_timesteps(count, diffusion["denoiser_p_mean"], diffusion["denoiser_p_std"],
                             diffusion["time_schedule"], clean.device, clean.dtype)
        z_denoiser = add_noise(clean, torch.randn_like(clean), t, SimpleNamespace(**diffusion))
        decoder = torch.rand(count, device=clean.device) < diffusion["decoder_probability"]
        # decoder 原生状态：逐 token logit-normal 混合，t=1，SC 为零。
        ratio = torch.sigmoid(torch.randn(count, length, 1, device=clean.device)
                              * diffusion["decoder_p_std"] + diffusion["decoder_p_mean"])
        z_decoder = ratio * clean + (1 - ratio) * torch.randn_like(clean) * diffusion["decoder_noise_scale"]
        sc_scale = sample_cfg_scale(count, diffusion["self_cond_cfg_min"], diffusion["self_cond_cfg_max"],
                                   clean.dtype, clean.device)
        sc_selected = torch.rand(count, device=clean.device) < config["training"]["self_condition_probability"]
        zeros = torch.zeros_like(clean)
        # 无梯度 SC 只调用原始 student，不经过 DDP，防止各卡随机分支触发不同通信。
        self_conditioning = config["training"]["self_condition_probability"] > 0
        if self_conditioning:
            initial, _ = forward(model, torch.cat([z_denoiser, zeros], -1), t, sc_scale, None, config, True)
            sc = initial.float() * (sc_selected & ~decoder)[:, None, None]
        else:
            sc = zeros
        native_target = (clean - z_denoiser) / (1 - t[:, None, None]).clamp_min(diffusion["t_eps"])
        if weights["native_flow_weight"] and self_conditioning:
            # 复现原版 SC-CFG 的原生速度目标修正，目标本身停止梯度。
            conditional, _ = forward(model, torch.cat([z_denoiser, initial.float()], -1),
                                      t, sc_scale, None, config, True)
            guidance = (conditional.float() - initial.float()) / (1 - t[:, None, None]).clamp_min(diffusion["t_eps"])
            native_target += (1 - 1 / sc_scale[:, None, None]) * guidance * sc_selected[:, None, None]
        z = torch.where(decoder[:, None, None], z_decoder, z_denoiser)
        times = torch.where(decoder, torch.ones_like(t), t)
        inputs = (torch.cat([z, sc], -1) if self_conditioning else z).detach()
        if query_teacher is None:
            query_teacher = weights["flow_weight"] > 0 or weights["kd_weight"] > 0
        teacher_outputs = forward(teacher, inputs, times, sc_scale, decoder, config, True) if query_teacher else None
    prediction, logits = forward(student, inputs, times, sc_scale, decoder, config, not model.training)
    denom = valid.sum().clamp_min(1)
    flow_mask, decoder_mask = valid * (~decoder)[:, None], valid * decoder[:, None]
    zero = logits.float().sum() * 0
    flow_mse = zero
    kl = zero
    log_student = logits.float().log_softmax(-1)
    if teacher_outputs is not None:
        teacher_prediction, teacher_logits = teacher_outputs
        # 同一 z/t 下速度差 = (xhat_S-xhat_T)/(1-t)，沿用原版 t_eps 截断。
        difference = (prediction.float() - teacher_prediction.float()) / (1 - times[:, None, None]).clamp_min(diffusion["t_eps"])
        flow_mse = (difference.square().mean(-1) * flow_mask).sum() / denom
        log_teacher = teacher_logits.float().log_softmax(-1)
        kl = ((log_teacher.exp() * (log_teacher - log_student)).sum(-1) * decoder_mask).sum() / denom
    ce = (-log_student.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1) * decoder_mask).sum() / denom
    velocity = (prediction.float() - z) / (1 - times[:, None, None]).clamp_min(diffusion["t_eps"])
    native_mse = ((velocity - native_target).square().mean(-1) * flow_mask).sum() / denom
    # 与原版混合分支同一分母；零权重仍保留计算图连接，避免偶发全单分支的 DDP unused。
    loss = (weights["flow_weight"] * flow_mse + weights["kd_weight"] * kl
            + weights["ce_weight"] * ce + weights["native_flow_weight"] * native_mse)
    metrics = {"loss": float(loss.detach()), "flow_mse": float(flow_mse.detach()),
               "native_flow_mse": float(native_mse.detach()), "decoder_kl": float(kl.detach()),
               "decoder_ce": float(ce.detach())}
    return loss, metrics
