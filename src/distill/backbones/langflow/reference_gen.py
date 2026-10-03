"""复制自 LangFlow/gen_ppl.py 的生成/评分函数，保持原版 CUDA FP16 与逐样本重评分。"""

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_texts(model, tokenizer, args, device):
    samples = []
    remaining = args.num_samples

    print(
        f"Generating {args.num_samples} sample(s) "
        f"(batch_size={args.batch_size}, steps={args.num_steps}, length={args.seq_length})")
    with torch.inference_mode():
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)
            token_ids = model.generate_samples(
                num_samples=batch_size,
                seq_length=args.seq_length,
                num_steps=args.num_steps,
                device=device,
            )
            samples.append(token_ids.cpu())
            remaining -= batch_size

    sample_ids = torch.cat(samples, dim=0)
    # Remark: Following the baselines' evaluation protocol, we don't skip special tokens like [CLS].
    # When evaluating Gen. PPL for LM1B, [CLS] will be verbatim re-tokenized as [ CL S ].
    # While this seems weird, we preserve the original behavior for consistency.
    texts = tokenizer.batch_decode(sample_ids, skip_special_tokens=False)
    return sample_ids, texts


def sample_entropy(sample_ids):
    entropies = []
    for ids in sample_ids:
        _, counts = torch.unique(ids, return_counts=True)
        probs = counts.float() / counts.sum()
        entropies.append(-(probs * torch.log(probs)).sum())
    return torch.stack(entropies).mean().item()


def compute_generated_ppl(texts, scorer_name_or_path, batch_size, max_length, device, cache_dir=None):
    print(f"Loading PPL scorer: {scorer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(scorer_name_or_path, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    scorer = AutoModelForCausalLM.from_pretrained(
        scorer_name_or_path,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    scorer.to(device)
    scorer.eval()

    total_nll = 0.0
    total_tokens = 0
    per_sample_nll = []
    per_sample_tokens = []

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            labels = input_ids.masked_fill(attention_mask == 0, -100)

            outputs = scorer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            valid_tokens = (attention_mask.sum(dim=1) - 1).clamp(min=0)
            batch_tokens = int(valid_tokens.sum().item())
            if batch_tokens == 0:
                continue

            batch_nll = outputs.loss.item() * batch_tokens
            total_nll += batch_nll
            total_tokens += batch_tokens

            # Re-score each text to keep per-sample values exact despite padding.
            for text in batch_texts:
                single = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                single_input_ids = single["input_ids"].to(device)
                single_attention_mask = single["attention_mask"].to(device)
                single_tokens = int((single_attention_mask.sum() - 1).clamp(min=0).item())
                if single_tokens == 0:
                    per_sample_nll.append(float("nan"))
                    per_sample_tokens.append(0)
                    continue
                single_out = scorer(
                    input_ids=single_input_ids,
                    attention_mask=single_attention_mask,
                    labels=single_input_ids,
                )
                per_sample_nll.append(single_out.loss.item())
                per_sample_tokens.append(single_tokens)

    if total_tokens == 0:
        raise ValueError("No scoreable tokens were generated.")

    nll_per_token = total_nll / total_tokens
    return {
        "gen_ppl": math.exp(nll_per_token),
        "nll_per_token": nll_per_token,
        "num_scored_tokens": total_tokens,
        "per_sample_nll": per_sample_nll,
        "per_sample_tokens": per_sample_tokens,
    }
