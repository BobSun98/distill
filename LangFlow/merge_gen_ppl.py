#!/usr/bin/env python
"""Merge per-GPU gen_ppl.py shard outputs into one token-weighted Gen. PPL.

Token-weighted pooling of per-sample NLL is exactly equivalent to running all
samples in a single process:
    nll_per_token = sum_i(nll_i * tokens_i) / sum_i(tokens_i)
    gen_ppl       = exp(nll_per_token)
"""
import json
import math
import sys


def main(paths):
    tot_nll = 0.0
    tot_tokens = 0
    n_samples = 0
    ent_weighted = 0.0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
        m = payload["metrics"]
        nlls = m["per_sample_nll"]
        toks = m["per_sample_tokens"]
        for nll, t in zip(nlls, toks):
            if t <= 0 or nll != nll:  # skip empty / NaN samples
                continue
            tot_nll += nll * t
            tot_tokens += t
        shard_n = len(nlls)
        n_samples += shard_n
        ent_weighted += m["sample_entropy"] * shard_n

    if tot_tokens == 0:
        raise ValueError("No scoreable tokens across shards.")

    nll_per_token = tot_nll / tot_tokens
    print("=" * 60)
    print("Merged Gen. PPL (token-weighted across shards)")
    print("=" * 60)
    print(f"Shards:         {len(paths)}")
    print(f"Samples:        {n_samples}")
    print(f"Scored tokens:  {tot_tokens}")
    print(f"NLL/token:      {nll_per_token:.4f}")
    print(f"Gen. PPL:       {math.exp(nll_per_token):.4f}")
    print(f"Sample entropy: {ent_weighted / n_samples:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
