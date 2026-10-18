"""真实 LangFlow/ELF 的统一噪声采样适配，保持模型各自的 transport 公式。"""

from types import SimpleNamespace

import torch

from .backbones.elf.generation_utils import _dlm_decode_batch
from .backbones.elf.sampling_utils import _ode_step, _sde_step
from .gen_ppl_protocol import SEED, gaussian_noise


def elf_time_grid(steps, settings, device):
    if settings["time_schedule"] == "uniform":
        return torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float32)
    if settings["time_schedule"] != "logit_normal":
        raise ValueError("ELF time_schedule 只支持 uniform / logit_normal")
    generator = torch.Generator(device="cpu").manual_seed(SEED + 2_000_000_000_011)
    values = torch.sigmoid(torch.randn(steps - 1, generator=generator, dtype=torch.float32)
                           * settings["denoiser_p_std"] + settings["denoiser_p_mean"])
    # 所有 batch/rank 对同一步数使用同一个固定网格，避免时间表跟随 batch RNG 改变。
    return torch.cat([torch.zeros(1, dtype=torch.float32), values.sort().values,
                      torch.ones(1, dtype=torch.float32)]).to(device)


@torch.inference_mode()
def generate_batch(model, family, ids, generation, metadata):
    device = next(model.parameters()).device
    length, steps = generation["sequence_length"], generation["num_steps"]
    dimension = model.config.hidden_size if family == "langflow" else model.text_encoder_dim
    initial = gaussian_noise(ids, length, dimension).to(device)
    if family == "langflow":
        metadata["effective_sampling"] = {"sampling_method": "euler_edm", "noise_scale": 1.0,
            "self_conditioning": model.config.self_conditioning, "nfe": steps}
        return model.generate_samples(len(ids), length, steps, device, initial_noise=initial)

    diffusion = metadata.get("diffusion", {})
    native = SimpleNamespace(
        denoiser_noise_scale=diffusion.get("denoiser_noise_scale", 2.0),
        t_eps=diffusion.get("t_eps", 0.05),
        self_cond_prob=diffusion.get("self_cond_prob", 0.5),
        num_self_cond_cfg_tokens=model.num_self_cond_cfg_tokens,
        use_bf16=device.type == "cuda")
    sampling = {"sampling_method": "sde", "sde_gamma": 1.5,
        "time_schedule": "logit_normal", "cfg_scale": 1.0, "self_cond_cfg_scale": 3.0,
        **metadata.get("sampling", {}), **generation.get("sampling", {})}
    metadata["effective_sampling"] = {**sampling, "noise_scale": native.denoiser_noise_scale,
        "t_eps": native.t_eps, "self_cond_prob": native.self_cond_prob,
        "decoder_extra_forward": 1,
        "nfe": steps + 1 if model.num_self_cond_cfg_tokens > 0 and sampling["cfg_scale"] == 1 else None}
    grid = elf_time_grid(steps, {**sampling,
        "denoiser_p_mean": diffusion.get("denoiser_p_mean", -1.5),
        "denoiser_p_std": diffusion.get("denoiser_p_std", 0.8)}, device)
    z = initial * native.denoiser_noise_scale
    previous = torch.zeros_like(z)
    cond = torch.zeros_like(z)
    cond_mask = torch.zeros(z.shape[:2], device=device)
    kwargs = dict(model=model, config=native, cfg_scale=sampling["cfg_scale"],
        self_cond_cfg_scale=sampling["self_cond_cfg_scale"], cond_seq=cond, cond_seq_mask=cond_mask)
    if sampling["sampling_method"] not in ("sde", "ode"):
        raise ValueError("ELF sampling_method 只支持 sde / ode")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for index in range(steps):
            times = {"t": grid[index].item(), "t_next": grid[index + 1].item(), "x_pred_prev": previous}
            if sampling["sampling_method"] == "sde" and index < steps - 1:
                noise = gaussian_noise(ids, length, dimension, stream=index + 1).to(device)
                z, previous = _sde_step(z=z, gamma=sampling["sde_gamma"], generator=None,
                                        noise=noise, **times, **kwargs)
            else:
                z, previous = _ode_step(z=z, **times, **kwargs)
    return _dlm_decode_batch(z, model, grid[-1], native, sampling["self_cond_cfg_scale"])
