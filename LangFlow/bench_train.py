"""Benchmark a synthetic LangFlow training / distillation step (fwd+bwd) and
teacher forward-only. Training code isn't released, so we reconstruct the
diffusion training step: embed+noise input_ids at sampled gamma -> forward ->
CE loss -> backward. Cost is dominated by the DiT fwd/bwd, which this measures
directly (not by analogy)."""
import argparse, os, statistics, time
import torch
import torch.nn.functional as F
from langflow import LangFlow, LangFlowConfig

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["train", "fwd"], default="train")
ap.add_argument("--n_blocks", type=int, default=None, help="Override depth (student=half).")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--warmup", type=int, default=5)
args = ap.parse_args()

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")
cfg = LangFlowConfig.from_pretrained(os.path.join(os.path.dirname(__file__), "langflow"))
if args.n_blocks:
    cfg.n_blocks = args.n_blocks

model = LangFlow(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
B, L, V = args.batch, args.seq, cfg.vocab_size
print(f"[cfg] n_blocks={cfg.n_blocks} hidden={cfg.hidden_size} params={n_params/1e6:.1f}M "
      f"batch={B} seq={L} mode={args.mode}")

input_ids = torch.randint(0, V, (B, L), device=device)
gmin, gmax = model.proposal.gamma_min, model.proposal.gamma_max

if args.mode == "train":
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        gamma = torch.empty(B, device=device).uniform_(gmin, gmax)
        logits = model(input_ids=input_ids, timesteps=gamma, return_dict=False)
        loss = F.cross_entropy(logits.float().reshape(-1, V), input_ids.reshape(-1))
        loss.backward()
        opt.step()
else:
    model.eval()
    with torch.no_grad():
        z = model._embed_tokens(input_ids)
    gamma = torch.empty(B, device=device).uniform_(gmin, gmax)

    def step():
        with torch.no_grad():
            model(noisy_embeds=z, timesteps=gamma, return_dict=False)

for _ in range(args.warmup):
    step()
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
times = []
for _ in range(args.steps):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    step()
    torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
med = statistics.median(times)
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"[result] median_step={med*1000:.1f} ms  throughput={B/med:.1f} seq/s  peak_mem={peak:.2f} GB")
