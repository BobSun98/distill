"""复用实际 transport 采集 student 查询状态；只缓存 latent，不缓存词表 logits。"""

from collections import deque
from types import SimpleNamespace

import torch

from .. import progress
from ..backbones.elf.generation_utils import _dlm_decode_batch
from ..backbones.elf.sampling_utils import get_sampling_steps, _ode_step, _sde_step


def cpu_states(z, time, self_cond, decoder=False, sc_scale=None):
    result = []
    for index in range(len(z)):
        # 必须保存真实查询时的 SC，不能事后用 teacher 或当前 student 重建。
        result.append({"z": z[index].detach().to("cpu", copy=True),
                       "time": time[index].detach().to("cpu", copy=True),
                       "self_cond": self_cond[index].detach().to("cpu", copy=True),
                       "decoder": torch.tensor(decoder),
                       "sc_scale": (sc_scale[index].detach().cpu().clone()
                                    if sc_scale is not None else torch.tensor(1.0))})
    return result


@torch.no_grad()
def collect_states(model, config):
    rollout = config["on_policy"]
    count, length = rollout["rollout_batch_size"], config["evaluation"]["sequence_length"]
    steps = config["evaluation"]["num_steps"]
    device = next(model.parameters()).device
    # 分段抽取覆盖早/中/晚阶段，不只监督接近干净文本的末端。
    selected = set(torch.linspace(0, steps - 1, rollout["states_per_rollout"]).long().tolist())
    was_training = model.training
    model.eval()
    states = []
    try:
        with progress.phase("on_policy.rollout", family=config["model"]["family"],
                            trajectories=count, steps=steps):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                    enabled=device.type == "cuda" and config["model"]["use_bf16"]):
                if config["model"]["family"] == "langflow":
                    z = torch.randn(count, length, model.config.hidden_size, device=device)
                    grid = model.proposal(torch.linspace(1 - 1e-5, 1e-5, steps, device=device))
                    previous = torch.zeros_like(z)
                    for index, time in enumerate(grid):
                        times = time.expand(count)
                        if index in selected:
                            states.extend(cpu_states(z, times, previous))
                        logits = model(noisy_embeds=z, timesteps=times,
                                       x_self_cond=previous, return_dict=False)
                        if index < steps - 1:
                            prediction = model._embed_tokens(logits.float().softmax(-1))
                            z = model._euler_edm_step(z, prediction, time, grid[index + 1])
                            if model.config.self_conditioning:
                                previous = prediction
                        del logits
                else:
                    diffusion, sampling = config["diffusion"], config["evaluation"]["sampling"]
                    native = SimpleNamespace(**diffusion,
                        self_cond_prob=config["training"]["self_condition_probability"],
                        num_self_cond_cfg_tokens=model.num_self_cond_cfg_tokens,
                        use_bf16=config["model"]["use_bf16"])
                    grid = get_sampling_steps(steps, sampling["time_schedule"],
                        diffusion["denoiser_p_mean"], diffusion["denoiser_p_std"], device, torch.float32)
                    z = torch.randn(count, length, model.text_encoder_dim, device=device) * native.denoiser_noise_scale
                    previous = torch.zeros_like(z)
                    index = 0

                    def query(inputs, time, **kwargs):
                        decoder = bool(kwargs.get("decoder_step_active", False))
                        if index in selected or decoder:
                            dimension = model.text_encoder_dim
                            sc = inputs[..., dimension:] if inputs.shape[-1] == 2 * dimension else torch.zeros_like(inputs)
                            # SDE 会先回退/加噪：这里记录 z_back/t_back，而非错误地记录外层网格状态。
                            states.extend(cpu_states(inputs[..., :dimension], time, sc, decoder,
                                                     kwargs.get("self_cond_cfg_scale")))
                        return model(inputs, time, **kwargs)

                    kwargs = dict(model=query, config=native, cfg_scale=1.0,
                        self_cond_cfg_scale=sampling["self_cond_cfg_scale"],
                        cond_seq=torch.zeros_like(z), cond_seq_mask=torch.zeros(z.shape[:2], device=device))
                    for index in range(steps):
                        times = dict(t=grid[index].item(), t_next=grid[index + 1].item(), x_pred_prev=previous)
                        if sampling["sampling_method"] == "sde" and index < steps - 1:
                            z, previous = _sde_step(z=z, gamma=sampling["sde_gamma"], generator=None, **times, **kwargs)
                        else:
                            z, previous = _ode_step(z=z, **times, **kwargs)
                    if config["loss"]["ce_weight"] or config["loss"].get("kl_weight", 0):
                        # Decoder CE/KL 只在实际 rollout 的终点、t=1、零 SC、decoder mode 上监督。
                        _dlm_decode_batch(z, query, grid[-1], native, sampling["self_cond_cfg_scale"])
    finally:
        model.train(was_training)
    return states


def choose_states(entries, count, config):
    selected = []
    loss = config["loss"]
    for _ in range(count):
        decoder = False
        if config["model"]["family"] == "elf" and (loss["ce_weight"] or loss.get("kl_weight", 0)):
            decoder = loss["mse_weight"] == 0 or bool(torch.rand(()) < config["on_policy"]["decoder_probability"])
        candidates = [state for state in entries if bool(state["decoder"]) == decoder]
        selected.append(candidates[int(torch.randint(len(candidates), ()))])
    return {key: torch.stack([state[key] for state in selected]) for key in selected[0]}


class StateBuffer:
    def __init__(self, config):
        self.config = config
        self.entries = deque(maxlen=config["on_policy"]["buffer_capacity"])
        self.calls = 0

    def sample(self, model, count, training):
        if training:
            if not self.entries or self.calls % self.config["on_policy"]["refresh_every"] == 0:
                self.entries.extend(collect_states(model, self.config))
                progress.emit("on_policy.buffer", states=len(self.entries), micro_batch=self.calls)
            self.calls += 1
            entries = list(self.entries)
        else:
            # 验证使用固定 RNG 的新 rollout，不读取/污染训练缓存与刷新计数。
            entries = collect_states(model, self.config)
        selected = choose_states(entries, count, self.config)
        device = next(model.parameters()).device
        return {key: value.to(device) for key, value in selected.items()}
