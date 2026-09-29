"""Sweep batch size and report peak GPU memory + time for LangFlow inference."""
import argparse, os, time
import torch
from safetensors.torch import load_file
from langflow import LangFlow, LangFlowConfig

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="./checkpoints/model.safetensors")
ap.add_argument("--batches", default="1,4,8,16")
ap.add_argument("--num_steps", type=int, default=128)
ap.add_argument("--seq_length", type=int, default=1024)
args = ap.parse_args()

device = torch.device("cuda")
config = LangFlowConfig.from_pretrained(os.path.join(os.path.dirname(__file__), "langflow"))
model = LangFlow(config)
model.load_state_dict(load_file(args.checkpoint, device=str(device)))
model = model.to(device).eval()

n_params = sum(p.numel() for p in model.parameters())
weights_gb = torch.cuda.memory_allocated() / 1e9
print(f"[model] params={n_params/1e6:.1f}M  weights_resident={weights_gb:.2f} GB  "
      f"steps={args.num_steps} seq_len={args.seq_length}")

for b in [int(x) for x in args.batches.split(",")]:
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    try:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            model.generate_samples(num_samples=b, seq_length=args.seq_length,
                                   num_steps=args.num_steps, device=device)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[batch={b:>3}] peak_mem={peak:6.2f} GB  time={dt:6.1f} s  "
              f"per_sample={dt/b:5.2f} s  ({args.num_steps} steps)")
    except RuntimeError as e:
        print(f"[batch={b:>3}] OOM / error: {str(e)[:80]}")
        torch.cuda.empty_cache()
