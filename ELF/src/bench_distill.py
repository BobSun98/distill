#!/usr/bin/env python
"""Empirical single-GPU benchmark for ELF training / distillation cost.

Measures per-step wall-clock and peak GPU memory for:
  - a full-depth model training step (train_step)      [--mode train]
  - a half-depth student training step                  [--mode train --depth N]
  - a teacher forward-only pass (no grad, 1 fwd)         [--mode fwd]

Uses synthetic latents/batch so no dataset download is needed; step time and
memory depend only on tensor shapes, not data content. Numbers here feed the
GPU-hour / RAM estimate for distillation.

Examples:
  python src/bench_distill.py --config src/configs/training_configs/train_owt_ELF-B.yml \
      --mode train --batch 8 --steps 30
  python src/bench_distill.py --config src/configs/training_configs/train_owt_ELF-B.yml \
      --mode train --batch 8 --depth 6 --steps 30      # half-depth student
  python src/bench_distill.py --config src/configs/training_configs/train_owt_ELF-B.yml \
      --mode fwd  --batch 8 --steps 30                  # teacher fwd-only
"""

import argparse
import os
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from transformers import AutoTokenizer

from modules.t5_encoder import get_encoder
from modules.model import ELF, ELF_models
from utils.train_utils import TrainState, get_optimizer
from train_step import train_step
from configs.config import load_config_from_yaml, apply_config_overrides


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--config_override", action="append", default=[])
    p.add_argument("--mode", choices=["train", "fwd"], default="train")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--depth", type=int, default=None,
                   help="Override model depth (e.g. half of teacher for a student).")
    p.add_argument("--hidden", type=int, default=None, help="Override hidden size.")
    p.add_argument("--heads", type=int, default=None, help="Override num heads.")
    p.add_argument("--steps", type=int, default=30, help="Timed steps (after warmup).")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--optimizer", type=str, default=None, help="Override optimizer (adamw/muon).")
    return p.parse_args()


# Default (depth, hidden, heads) per model name — mirrors modules/model.py factory.
_ARCH = {"ELF-B": (12, 768, 12), "ELF-M": (24, 1056, 16), "ELF-L": (32, 1280, 16)}


def main():
    args = parse_args()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    config = load_config_from_yaml(args.config)
    if args.optimizer:
        config.optimizer = args.optimizer
    config = apply_config_overrides(config, args.config_override)

    tok = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    vocab_size = tok.vocab_size

    enc_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    d_model = enc_config.d_model

    depth, hidden, heads = _ARCH[config.model]
    depth = args.depth or depth
    hidden = args.hidden or hidden
    heads = args.heads or heads

    model = ELF(
        text_encoder_dim=d_model, max_length=config.max_length,
        depth=depth, hidden_size=hidden, num_heads=heads,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
        gradient_checkpointing=config.gradient_checkpointing,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[cfg] model={config.model} depth={depth} hidden={hidden} heads={heads} "
          f"params={n_params/1e6:.1f}M batch={args.batch} max_len={config.max_length} "
          f"mode={args.mode} bf16={config.use_bf16} optim={config.optimizer}")

    B, S = args.batch, config.max_length
    # Synthetic uncond batch: no conditioning tokens, all positions valid.
    batch = {
        "input_ids": torch.randint(0, vocab_size, (B, S), device=device),
        "encoder_attention_mask": torch.ones((B, S), device=device),
        "cond_seq_mask": torch.zeros((B, S), device=device),
        "attention_mask": torch.ones((B, S), device=device),
        "label_drop_mask": torch.zeros((B,), dtype=torch.bool, device=device),
    }

    if args.mode == "train":
        optimizer = get_optimizer(model, config, lr=1e-4)
        g = torch.Generator(device="cpu").manual_seed(0)
        state = TrainState(
            model=model, optimizer=optimizer, lr_scheduler=None,
            ema_params1=TrainState.init_ema(model), step=0, epoch=0,
            dropout_generator=g,
        )

        def one_step():
            train_step(state, encoder, batch, config)
    else:
        # Teacher forward-only (no grad): the off-policy target-generation cost.
        from utils.encoder_utils import encode_text
        model.eval()
        input_ids = batch["input_ids"].long()
        with torch.no_grad():
            x0 = encode_text(input_ids=input_ids, attention_mask=batch["encoder_attention_mask"],
                             encoder=encoder, latent_mean=config.latent_mean,
                             latent_std=config.latent_std, use_bf16=config.use_bf16)
        z = x0.clone()
        z_in = torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
        t = torch.rand((B,), device=device)
        sc = (torch.ones((B,), device=device)
              if config.num_self_cond_cfg_tokens > 0 else None)

        def one_step():
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=config.use_bf16):
                model(z_in, t, deterministic=True, self_cond_cfg_scale=sc)

    # Warmup
    for _ in range(args.warmup):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    med = statistics.median(times)
    print(f"[result] median_step={med*1000:.1f} ms  "
          f"mean={statistics.mean(times)*1000:.1f} ms  "
          f"throughput={B/med:.1f} seq/s  peak_mem={peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
