"""在相同采样协议下比较 teacher、删层初始 student 与训练后 student。"""

import math
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import project_path, write_json
from .data import make_loader
from .models import build_student, load_model, parameter_counts
from .train import seed_everything, validate
from . import distributed


@torch.no_grad()
def generate(model, tokenizer, settings):
    # 每个 rank 负责连续的一段样本；同一 rank 的三种模型使用相同初始噪声。
    seed_everything(settings["seed"] + distributed.rank())
    model.eval()
    device = next(model.parameters()).device
    ids = []
    base, remainder = divmod(settings["num_samples"], distributed.world_size())
    local_samples = base + (distributed.rank() < remainder)
    for start in range(0, local_samples, settings["batch_size"]):
        count = min(settings["batch_size"], local_samples - start)
        ids.append(model.generate_samples(num_samples=count, seq_length=settings["sequence_length"],
                                          num_steps=settings["num_steps"], device=device).cpu())
    tokens = torch.cat(ids) if ids else torch.empty((0, settings["sequence_length"]), dtype=torch.long)
    # 与现有 gen_ppl 约定相同，特殊 token 也保留在解码文本中。
    texts = tokenizer.batch_decode(tokens, skip_special_tokens=False)
    return tokens, texts


def diversity_metrics(tokens):
    bigrams, total_bigrams, entropies = set(), 0, []
    for row in tokens.tolist():
        pairs = list(zip(row, row[1:]))
        bigrams.update(pairs)
        total_bigrams += len(pairs)
        counts = Counter(row)
        probabilities = [count / len(row) for count in counts.values()]
        entropies.append(-sum(p * math.log(p) for p in probabilities))
    return {"distinct_2": len(bigrams) / max(total_bigrams, 1),
            "sample_entropy": sum(entropies) / len(entropies)}


@torch.no_grad()
def score_texts(texts, scorer, tokenizer, settings, device):
    total_nll, total_tokens, per_sample = 0.0, 0, []
    for start in range(0, len(texts), settings["scorer_batch_size"]):
        batch = tokenizer(texts[start:start + settings["scorer_batch_size"]], padding=True,
                          truncation=True, max_length=settings["scorer_max_length"],
                          return_tensors="pt").to(device)
        logits = scorer(**batch).logits
        labels = batch["input_ids"][:, 1:]
        mask = batch["attention_mask"][:, 1:].bool()
        losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels, reduction="none")
        nlls = (losses * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        total_nll += nlls.sum().item()
        total_tokens += counts.sum().item()
        for nll, count in zip(nlls.tolist(), counts.tolist()):
            per_sample.append({"nll_per_token": nll / count if count else None, "tokens": count})
    # 汇总的是 NLL 和 token 数，不能直接平均各卡的 PPL。
    total_nll, total_tokens = distributed.sum_tensor([total_nll, total_tokens], device).tolist()
    sample_shards = distributed.gather_main(per_sample)
    if not total_tokens:
        raise ValueError("生成文本中没有可评分的 token")
    mean_nll = total_nll / total_tokens
    return {"gen_ppl": math.exp(mean_nll), "nll_per_token": mean_nll,
            "scored_tokens": int(total_tokens),
            "per_sample": [item for shard in sample_shards for item in shard] if sample_shards else []}


def evaluate(teacher, tokenizer, checkpoint, config, run_dir):
    settings = config["evaluation"]
    if min(settings["num_samples"], settings["batch_size"], settings["num_steps"] - 1,
           settings["sequence_length"]) < 1:
        raise ValueError("评测需要正样本数、batch size、长度和至少 2 次模型前向")
    device = next(teacher.parameters()).device
    loader = make_loader(config, "valid")
    results = {"protocol": {"seed": settings["seed"], "nfe": settings["num_steps"],
                             "sequence_length": settings["sequence_length"],
                             "num_samples": settings["num_samples"],
                             "scorer": settings["scorer"],
                             "sampler": "LangFlow Euler-EDM",
                             "world_size": distributed.world_size(),
                             "generation_batch_size_per_device": settings["batch_size"],
                             "rank_seed": "seed + rank"}, "models": {}}
    generated = {}
    output = run_dir / "evaluation"
    if distributed.is_main():
        output.mkdir(parents=True, exist_ok=True)
    for label in ("teacher", "pruned", "trained"):
        if distributed.is_main():
            print(f"[eval] {label}", flush=True)
        if label == "teacher":
            model = teacher
        elif label == "pruned":
            model = build_student(teacher, config["model"]["keep_layers"]).eval()
        else:
            # 从保存目录重新加载，而非直接评测内存里的模型，同时验证 checkpoint 可用。
            model = load_model(str(checkpoint), device)
            model.backbone.vocab_embed.requires_grad_(False)
            model.proposal.requires_grad_(False)
            if (model.config.n_blocks != len(config["model"]["keep_layers"])
                    or model.config.hidden_size != teacher.config.hidden_size
                    or model.config.vocab_size != teacher.config.vocab_size):
                raise ValueError("checkpoint 架构与当前 teacher/keep_layers 配置不一致")
        metrics = {"parameters": parameter_counts(model),
                   "denoising": validate(teacher, model, loader, config)}
        tokens, texts = generate(model, tokenizer, settings)
        shards = distributed.gather_main((tokens, texts))
        if distributed.is_main():
            merged_tokens = torch.cat([shard[0] for shard in shards])
            merged_texts = [text for shard in shards for text in shard[1]]
            metrics.update(diversity_metrics(merged_tokens))
            torch.save(merged_tokens, output / f"{label}_tokens.pt")
            write_json(output / f"{label}_texts.json", merged_texts)
            results["models"][label] = metrics
        generated[label] = texts
        del model
    # scorer 与生成器错峰占用 GPU；teacher 后续无需再查询。
    if settings["scorer"]:
        teacher.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        cache_dir = config["model"]["cache_dir"]
        cache_dir = str(project_path(cache_dir)) if cache_dir else None
        scorer_path = project_path(settings["scorer"])
        scorer_name = str(scorer_path) if scorer_path.is_dir() else settings["scorer"]
        scorer_tokenizer = AutoTokenizer.from_pretrained(scorer_name, cache_dir=cache_dir)
        scorer_tokenizer.padding_side = "right"
        if scorer_tokenizer.pad_token_id is None:
            scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
        scorer = AutoModelForCausalLM.from_pretrained(scorer_name, cache_dir=cache_dir)
        scorer = scorer.to(device).eval()
        for label, texts in generated.items():
            metrics = score_texts(texts, scorer, scorer_tokenizer, settings, device)
            if distributed.is_main():
                results["models"][label].update(metrics)
        del scorer
    if distributed.is_main():
        write_json(output / "metrics.json", results)
        print(f"[eval] results: {output / 'metrics.json'}", flush=True)
    return results
