"""ELF teacher/删层/训练后 student 的同协议生成与原版 Gen. PPL。"""

import math
from functools import partial
from types import SimpleNamespace

import torch

from .. import distributed, progress
from ..config import write_json
from ..train import seed_everything, validate
from ..backbones.elf.sampling_utils import get_sampling_steps
from ..backbones.elf.generation_utils import (
    _generate_samples_single_batch, _dlm_decode_batch, mask_after_eos,
)
from ..backbones.elf.metrics import Metrics
from .data import make_loader
from .losses import batch_loss, METRIC_KEYS
from .models import build_student, load_student, parameter_counts


def native_config(config):
    return SimpleNamespace(**config["diffusion"],
        self_cond_prob=config["training"]["self_condition_probability"],
        num_self_cond_cfg_tokens=config["model"]["architecture"]["num_self_cond_cfg_tokens"],
        use_bf16=config["model"]["use_bf16"])


@torch.no_grad()
def generate(model, tokenizer, config):
    settings = config["evaluation"]
    sampling = SimpleNamespace(**settings["sampling"])
    native = native_config(config)
    # 与 ELF/src/eval.py 一致，各 rank 使用大间隔种子，三组模型分别重置同一噪声序列。
    seed = settings["seed"] + distributed.rank() * 1_000_003
    seed_everything(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = next(model.parameters()).device
    model.eval()
    local_samples = (settings["num_samples"] + distributed.world_size() - 1) // distributed.world_size()
    texts, ids = [], []
    for start in range(0, local_samples, settings["batch_size"]):
        count = min(settings["batch_size"], local_samples - start)
        times = get_sampling_steps(settings["num_steps"], sampling.time_schedule,
                                    native.denoiser_p_mean, native.denoiser_p_std, device, torch.float32)
        shape = (count, settings["sequence_length"], model.text_encoder_dim)
        noise = torch.randn(shape, device=device) if device.type == "cuda" else torch.randn(shape, generator=generator)
        z = noise * native.denoiser_noise_scale
        with progress.phase("elf.generation.batch", samples=count, steps=settings["num_steps"]):
            latent = _generate_samples_single_batch(model, generator, z, times, None, None,
                         native, sampling, sampling.cfg_scale, sampling.self_cond_cfg_scale)
            tokens = _dlm_decode_batch(latent, model, times[-1], native, sampling.self_cond_cfg_scale)
        tokens = mask_after_eos(tokens, tokenizer.eos_token_id, tokenizer.pad_token_id).cpu()
        ids.append(tokens)
        texts.extend(tokenizer.batch_decode(tokens, skip_special_tokens=True))
    tokens = torch.cat(ids) if ids else torch.empty((0, settings["sequence_length"]), dtype=torch.long)
    return tokens, texts


def evaluate(teacher, encoder, tokenizer, checkpoint, config, run_dir):
    loss_fn = partial(batch_loss, encoder=encoder)
    loader = make_loader(config, "valid")
    settings = config["evaluation"]
    output = run_dir / "evaluation"
    if distributed.is_main():
        output.mkdir(parents=True, exist_ok=True)
    results = {"protocol": {**settings, "world_size": distributed.world_size(),
               "rank_seed": "seed + rank * 1000003", "score_rank": 0,
               "nfe_including_decoder": settings["num_steps"] + 1,
               "reference": "ELF/src/generation.py and utils/metrics_utils.py"}, "models": {}}
    all_texts = {}
    device = next(teacher.parameters()).device
    torch.set_float32_matmul_precision("high")
    for label in ("teacher", "pruned", "trained"):
        progress.emit("elf.evaluation.model", model=label)
        model = teacher if label == "teacher" else (
            build_student(teacher, config).eval() if label == "pruned" else load_student(checkpoint, device))
        denoising = validate(teacher, model, loader, config, loss_fn, METRIC_KEYS)
        tokens, texts = generate(model, tokenizer, config)
        shards = distributed.gather_main((tokens, texts))
        if distributed.is_main():
            merged_tokens = torch.cat([shard[0] for shard in shards])[:settings["num_samples"]]
            merged_texts = [text for shard in shards for text in shard[1]][:settings["num_samples"]]
            torch.save(merged_tokens, output / f"{label}_tokens.pt")
            write_json(output / f"{label}_texts.json", merged_texts)
            all_texts[label] = merged_texts
            results["models"][label] = {"parameters": {"total": parameter_counts(model)["total"]},
                                        "denoising": denoising, "num_generated_samples": len(merged_texts)}
        del model
    # 原版先对完整文本集合统一 padding，再在 rank 0 评分；保留该 EOS-mask 口径。
    teacher.cpu()
    encoder.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if distributed.is_main():
        if settings["scorer"]:
            scorer = Metrics(settings["scorer"], settings["scorer_batch_size"], settings["scorer_max_length"])
            for label, texts in all_texts.items():
                nonempty = [text for text in texts if text.strip()]
                if not nonempty:
                    # 与原版一样跳过全空文本，但保留失败的生成表现，继续比较其他模型。
                    results["models"][label].update({"gen_ppl": None, "sample_entropy": None,
                        "per_sample_ppl": [], "num_scored_samples": 0, "score_status": "empty_generation"})
                    continue
                scorer.reset()
                with progress.phase("elf.scorer", model=label, samples=len(nonempty)):
                    metrics = scorer.record_generative_perplexity(nonempty, settings["scorer_max_length"])
                metrics["per_sample_ppl"] = [value if math.isfinite(value) else None for value in metrics["per_sample_ppl"]]
                finite_ppl = math.isfinite(metrics["ppl"])
                results["models"][label].update({"gen_ppl": metrics["ppl"] if finite_ppl else None,
                    "sample_entropy": metrics["mean_entropy"], "per_sample_ppl": metrics["per_sample_ppl"],
                    "num_scored_samples": len(nonempty),
                    "score_status": "completed" if finite_ppl else "nonfinite_ppl"})
        write_json(output / "metrics.json", results)
        print(f"[eval] ELF results: {output / 'metrics.json'}", flush=True)
    distributed.barrier()
    return results
