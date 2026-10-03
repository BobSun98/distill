"""复制自 LangFlow/eval_ppl.py 的 NLL 计算，保持原始积分、SC 与 Hutchinson 估计。"""

import torch
import torch.nn.functional as F
import torchdiffeq

from .model import LangFlow


class LangFlowForFlowNLL(LangFlow):
    """LangFlow with flow-based NLL computation."""

    def _pred_x_and_div(self, z_t, gamma, use_self_cond=False, attention_mask=None):
        B, L, D = z_t.shape
        u = torch.randn_like(z_t)
        if attention_mask is not None:
            # Zero out padding positions so they don't contribute to the
            # Hutchinson divergence estimate or the ODE z-trajectory.
            token_mask = attention_mask[:, :, None].to(u.dtype)
            u = u * token_mask

        z = z_t.clone().detach().requires_grad_(True)
        gamma_batch = gamma.expand(B)

        x_self_cond = None
        if use_self_cond and self.config.self_conditioning:
            logits_sc = self.forward(noisy_embeds=z, timesteps=gamma_batch, x_self_cond=None, return_dict=False)
            probs_sc = F.softmax(logits_sc, dim=-1)
            x_self_cond = self._embed_tokens(probs_sc).detach()

        logits = self.forward(noisy_embeds=z, timesteps=gamma_batch, x_self_cond=x_self_cond, return_dict=False)
        probs = F.softmax(logits, dim=-1)
        x_reconst = self._embed_tokens(probs)
        if attention_mask is not None:
            x_reconst = x_reconst * token_mask

        grad = torch.autograd.grad(
            (u * x_reconst).sum(), z, create_graph=False, retain_graph=False
        )[0]

        div = (u * grad).sum(dim=[1, 2])
        return x_reconst, div

    def compute_flow_nll(self, x0, n_steps=128, ode_method="euler", use_self_cond=True,
                         attention_mask=None):
        B, L = x0.shape
        device = x0.device
        dtype = torch.float32

        x_embed = self._embed_tokens(x0).to(dtype)
        D = x_embed.shape[2]
        N = L * D

        gamma_0 = torch.tensor(self.proposal.gamma_min, device=device, dtype=dtype)
        alpha_0 = torch.sigmoid(-gamma_0).sqrt()
        sigma_0 = torch.sigmoid(gamma_0).sqrt()

        eps = torch.randn_like(x_embed)
        z_0 = alpha_0 * x_embed + sigma_0 * eps
        z_0 = z_0.detach()

        # === 1. Reconstruction loss at gamma_0 ===
        gamma_batch = gamma_0.expand(B)

        x_self_cond = None
        if use_self_cond and self.config.self_conditioning:
            with torch.no_grad():
                logits_sc = self.forward(noisy_embeds=z_0, timesteps=gamma_batch, x_self_cond=None, return_dict=False)
                probs_sc = F.softmax(logits_sc, dim=-1)
                x_self_cond = self._embed_tokens(probs_sc).detach()

        logits = self.forward(noisy_embeds=z_0, timesteps=gamma_batch, x_self_cond=x_self_cond, return_dict=False)
        reconst_nll_per_token = F.cross_entropy(
            logits.flatten(0, 1), x0.flatten(), reduction="none"
        ).reshape(B, L).detach()
        if attention_mask is not None:
            reconst_nll = (reconst_nll_per_token * attention_mask.to(dtype)).sum(dim=1)
        else:
            reconst_nll = reconst_nll_per_token.sum(dim=1)

        # === 2. Flow integral via ODE ===
        def ode_func(sqrt_snr, y):
            gamma = -2.0 * torch.log(sqrt_snr)
            sigma = torch.sigmoid(gamma).sqrt()
            z_scaled = y[:, 1:].reshape(B, L, D)
            z_t = z_scaled * sigma
            x_reconst, x_div = self._pred_x_and_div(
                z_t, gamma, use_self_cond=use_self_cond, attention_mask=attention_mask)
            div_scaled = sigma * x_div
            v = x_reconst.reshape(B, -1)
            return torch.cat([div_scaled.view(B, 1), v], dim=1).detach()

        y0 = torch.cat([
            torch.zeros(B, 1, device=device, dtype=dtype),
            (z_0 / sigma_0).reshape(B, -1)
        ], dim=1)

        t_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        t_grid = t_grid.clamp(min=1e-5, max=1.0 - 1e-5)
        gamma_grid = self.proposal(t_grid).to(dtype)
        sqrt_snr_grid = torch.exp(-0.5 * gamma_grid)

        y1 = torchdiffeq.odeint(ode_func, y0, sqrt_snr_grid, method=ode_method)[-1]

        flow_nll = -y1[:, 0].detach()

        # === 3. Prior loss at gamma_1 ===
        z_scaled_final = y1[:, 1:].reshape(B, L, D)
        if attention_mask is not None:
            token_mask = attention_mask[:, :, None].to(dtype)
            num_valid_per_seq = attention_mask.sum(dim=1).to(dtype)   # (B,)
            prior_nll = (0.5 * (z_scaled_final ** 2) * token_mask).sum(dim=[1, 2]) \
                        - num_valid_per_seq * D / 2
        else:
            prior_nll = 0.5 * (z_scaled_final ** 2).sum(dim=[1, 2]) - N / 2

        num_valid_tokens = (
            attention_mask.sum(dim=1).long()
            if attention_mask is not None
            else torch.full((B,), L, dtype=torch.long, device=device)
        )

        return {
            "nll": reconst_nll + flow_nll + prior_nll,
            "reconst_nll": reconst_nll,
            "flow_nll": flow_nll,
            "prior_nll": prior_nll,
            "num_valid_tokens": num_valid_tokens,
        }
