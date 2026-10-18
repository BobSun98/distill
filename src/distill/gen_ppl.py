"""统一、独立的多权重 Gen PPL 入口；只加载待测 generator 与 GPT-2-Large。"""

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import distributed, progress
from .backbones.elf import ELF
from .backbones.langflow import LangFlow, LangFlowConfig
from .config import create_run_dir, load_yaml_config, project_path, write_json
from .corpus_ppl import resolve_scorer, score_samples
from .gen_ppl_protocol import (SEED, NUM_SAMPLES, SCORER, MAX_LENGTH, VERSION,
    sample_ids, gaussian_noise, decode_text, text_statistics, aggregate_scores)
from .gen_ppl_sampling import generate_batch


def load_config(path, overrides=(), debug=False):
    config = load_yaml_config(path, overrides)
    protocol, runtime = config["protocol"], config["runtime"]
    if (protocol["version"] != VERSION or protocol["seed"] != SEED or protocol["scorer"] != SCORER
            or protocol["scorer_max_length"] != MAX_LENGTH):
        raise ValueError("统一协议固定 gen_ppl_v1、seed=42、gpt2-large、评分上限 1024")
    if (not isinstance(protocol["num_samples"], int)
            or not debug and protocol["num_samples"] != NUM_SAMPLES
            or debug and not 1 <= protocol["num_samples"] <= NUM_SAMPLES):
        raise ValueError("正式统一评测固定 1024 条；只有单卡 debug 入口可减少样本数")
    if runtime["generation_batch_size"] < 1 or not config["models"]:
        raise ValueError("生成 batch 必须为正，models 不能为空")
    names = []
    for entry in config["models"]:
        name = entry["name"]
        if not name or Path(name).name != name or name in (".", "..") or name in names:
            raise ValueError("模型 name 必须是非空、不重复的单级目录名")
        names.append(name)
        generation = entry["generation"]
        if generation["num_steps"] < 2 or not 1 <= generation["sequence_length"] <= MAX_LENGTH:
            raise ValueError("采样至少 2 步，生成长度必须在 1–1024")
        if generation.get("batch_size", runtime["generation_batch_size"]) < 1:
            raise ValueError("模型生成 batch 必须为正")
        if not entry["architecture"] or not entry["weights"]:
            raise ValueError("每套模型必须提供 architecture 与 weights")
    return config


def resolve_reference(reference, cache_dir):
    if isinstance(reference, dict):
        return hf_hub_download(reference["repo_id"], reference["filename"], cache_dir=cache_dir)
    path = project_path(reference)
    if not path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {path}；HF 文件请使用 repo_id/filename 显式引用")
    return str(path)


def prepare_files(entry, output, cache_dir):
    files = None
    if distributed.is_main():
        output.mkdir(parents=True, exist_ok=True)
        architecture = resolve_reference(entry["architecture"], cache_dir)
        weights = resolve_reference(entry["weights"], cache_dir)
        metadata = json.loads(Path(architecture).read_text(encoding="utf-8"))
        family = metadata["model_type"].lower()
        if family not in ("langflow", "elf"):
            raise ValueError("统一入口当前支持 LangFlow / ELF 架构")
        sibling = Path(weights).parent
        vocabulary = metadata["vocab_size"] if family == "langflow" else metadata["architecture"]["vocab_size"]
        if (sibling / "tokenizer_config.json").is_file():
            tokenizer_ref = str(sibling)
        elif metadata.get("tokenizer") or metadata.get("encoder"):
            tokenizer_ref = metadata.get("tokenizer") or metadata["encoder"]
        elif vocabulary == 50257:
            tokenizer_ref = "gpt2"
        elif vocabulary == 30522:
            tokenizer_ref = "bert-base-uncased"
        else:
            raise ValueError("无法从架构确定真实 tokenizer；请在架构 JSON 中保存 tokenizer 引用")
        local = project_path(tokenizer_ref)
        tokenizer = AutoTokenizer.from_pretrained(str(local) if local.is_dir() else tokenizer_ref,
            cache_dir=cache_dir, use_fast=vocabulary != 30522)
        if len(tokenizer) != vocabulary:
            raise ValueError("架构词表大小与真实 tokenizer 不一致")
        if family == "langflow" and tokenizer.bos_token_id is None and tokenizer.cls_token_id is not None:
            tokenizer.bos_token = tokenizer.cls_token
            tokenizer.eos_token = tokenizer.sep_token
        tokenizer_path = output / "tokenizer"
        tokenizer.save_pretrained(tokenizer_path)
        files = {"architecture": architecture, "weights": weights, "family": family,
                 "tokenizer": str(tokenizer_path), "use_fast": vocabulary != 30522}
    return distributed.broadcast_main(files)


def load_generator(files, entry, device):
    metadata = json.loads(Path(files["architecture"]).read_text(encoding="utf-8"))
    if files["family"] == "langflow":
        model = LangFlow(LangFlowConfig.from_dict(metadata))
        max_length = model.config.model_length
    else:
        model = ELF(**metadata["architecture"])
        max_length = model.max_length
    if entry["generation"]["sequence_length"] > max_length:
        raise ValueError("生成长度超过模型架构的位置编码上限")
    key = None
    if files["weights"].endswith(".safetensors"):
        if entry.get("weights_key") is not None:
            raise ValueError("safetensors 是直接模型权重，不使用 weights_key；该字段只用于原生 PyTorch checkpoint")
        weights = load_file(files["weights"])
    else:
        payload = torch.load(files["weights"], map_location="cpu", weights_only=True, mmap=True)
        key = entry.get("weights_key")
        if key is None:
            key = "ema_params1" if payload.get("ema_params1") else "params" if "params" in payload else None
        weights = payload[key] if key is not None else payload
    model.load_state_dict(weights, strict=True)
    metadata["selected_weights_key"] = key
    return model.float().to(device).eval().requires_grad_(False), metadata


@torch.inference_mode()
def generate_records(model, tokenizer, files, entry, config, metadata, output):
    ids = sample_ids(distributed.rank(), distributed.world_size(), config["protocol"]["num_samples"])
    generation = entry["generation"]
    batch_size = generation.get("batch_size", config["runtime"]["generation_batch_size"])
    records = []
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        with progress.phase("unified.generate", model=entry["name"], sample_ids=batch_ids):
            tokens = generate_batch(model, files["family"], batch_ids, generation, metadata).cpu()
        for sample_id, row in zip(batch_ids, tokens.tolist()):
            records.append({"sample_id": sample_id, "token_ids": row, "text": decode_text(row, tokenizer)})
    if 0 in ids:
        dimension = model.config.hidden_size if files["family"] == "langflow" else model.text_encoder_dim
        torch.save(gaussian_noise([0], generation["sequence_length"], dimension)[0], output / "initial_noise_sample_0.pt")
    return records


def scorer_on_device(cache, cache_dir, device):
    if "path" not in cache:
        with progress.phase("unified.scorer.files", model=SCORER):
            path = resolve_scorer({"name_or_path": SCORER, "cache_dir": cache_dir}) if distributed.is_main() else None
            cache["path"] = distributed.broadcast_main(path)
    if "model" not in cache:
        cache["tokenizer"] = AutoTokenizer.from_pretrained(cache["path"])
        cache["tokenizer"].pad_token = cache["tokenizer"].eos_token
        cache["tokenizer"].padding_side = "right"
        cache["tokenizer"].truncation_side = "right"
        # 所有模型共用 FP32/eager scorer，单条评分消除 batch padding 长度的数值差异。
        with progress.phase("unified.scorer.load", path=cache["path"]):
            cache["model"] = AutoModelForCausalLM.from_pretrained(cache["path"], torch_dtype=torch.float32,
                attn_implementation="eager").eval().requires_grad_(False)
        cache["model"].config.use_cache = False
    return cache["model"].to(device), cache["tokenizer"]


def score_records(records, scorer, tokenizer, device):
    valid, rows = [], {}
    for record in records:
        empty = not record["text"].strip()
        ids = tokenizer(record["text"], add_special_tokens=False, truncation=True,
                        max_length=MAX_LENGTH, return_attention_mask=False)["input_ids"] if not empty else []
        sample_id = record["sample_id"]
        rows[sample_id] = {"sample_id": sample_id, "nll_sum": 0.0, "num_scored_tokens": 0,
                          "empty": empty, **text_statistics(ids)}
        if len(ids) >= 2:
            valid.append({"sample_id": sample_id, "input_ids": ids})
    # 复用标准 shifted CE；首 token 不评分，正文 EOS 正常评分，padding 不计入分母。
    for score in score_samples(scorer, valid, tokenizer.eos_token_id, 1, device):
        rows[score["sample_id"]].update(score)
    return list(rows.values())


def save_summary(run_dir, protocol, results):
    write_json(run_dir / "metrics.json", {"protocol": protocol, "models": results})
    fields = ["name", "family", "num_steps", "sequence_length", "gen_ppl", "sample_entropy",
              "repetition_4gram", "num_samples", "num_scored_samples", "num_empty_samples", "score_status"]
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def run_gen_ppl(config, allow_distributed=True, debug=False):
    runtime = config["runtime"]
    owns_group = distributed.setup(runtime["device"], allow_distributed, runtime["control_timeout_seconds"])
    run_dir = Path(distributed.broadcast_main(str(create_run_dir(config)) if distributed.is_main() else None))
    progress.configure(run_dir, distributed.rank())
    results, cache = [], {}
    try:
        device = torch.device(runtime["device"])
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("当前机器没有 CUDA；统一模型评测请在服务器运行")
            device = torch.device("cuda", torch.cuda.current_device())
        torch.set_float32_matmul_precision("highest")
        cache_dir = str(project_path(runtime["cache_dir"])) if runtime["cache_dir"] else None
        protocol = {**config["protocol"], "version": VERSION + "_debug" if debug else VERSION,
            "debug": debug, "world_size": distributed.world_size(), "torch_version": torch.__version__,
            "noise": "sample_id_stream_cpu_fp32_dim_major_stride_1024_v1",
            "scorer_dtype": "float32", "scorer_attention": "eager", "scorer_batch_size": 1,
            "generation_precision": "float32_parameters_bfloat16_cuda_backbone",
            "decode": "strip_one_leading_bos_stop_at_first_eos_skip_special_tokens_no_cleanup",
            "aggregation": "exp(sum_nll / sum_valid_target_tokens)", "padding": "excluded_by_attention_mask",
            "add_special_tokens": False, "first_token": "not_scored",
            "empty_policy": "no_resampling_report_coverage", "scoring_truncation": "first_1024_gpt_tokens"}
        if distributed.is_main():
            write_json(run_dir / "protocol.json", protocol)
        for entry in config["models"]:
            output = run_dir / entry["name"]
            files = prepare_files(entry, output, cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(files["tokenizer"], use_fast=files["use_fast"], local_files_only=True)
            with progress.phase("unified.model.load", name=entry["name"]):
                model, metadata = load_generator(files, entry, device)
            records = generate_records(model, tokenizer, files, entry, config, metadata, output)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            try:
                scorer, scorer_tokenizer = scorer_on_device(cache, cache_dir, device)
                with progress.phase("unified.score", model=entry["name"], local_samples=len(records)):
                    scores = score_records(records, scorer, scorer_tokenizer, device)
            finally:
                if "model" in cache:
                    cache["model"].cpu()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            shards = distributed.gather_main((records, scores))
            if distributed.is_main():
                metrics = aggregate_scores([shard[1] for shard in shards], config["protocol"]["num_samples"])
                samples = sorted((row for shard in shards for row in shard[0]), key=lambda row: row["sample_id"])
                with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
                    for sample in samples:
                        handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                result = {"name": entry["name"], "family": files["family"], **entry["generation"], **metrics,
                    "architecture": files["architecture"], "weights": files["weights"],
                    "selected_weights_key": metadata["selected_weights_key"], "model_metadata": metadata,
                    "effective_sampling": metadata["effective_sampling"],
                    "effective_generation_batch_size": entry["generation"].get("batch_size", runtime["generation_batch_size"])}
                write_json(output / "metrics.json", {"protocol": protocol, **result})
                results.append({key: value for key, value in result.items() if key not in ("per_sample", "model_metadata")})
                save_summary(run_dir, protocol, results)
                print(f"[unified] {entry['name']}: Gen. PPL={metrics['gen_ppl']}, "
                      f"scored={metrics['num_scored_samples']}/{metrics['num_samples']}", flush=True)
            distributed.barrier()
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "completed", "protocol": protocol["version"], "debug": debug})
        distributed.barrier()
    except Exception as error:
        progress.emit("unified.failed", error=str(error))
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        if owns_group:
            distributed.close()
    return run_dir


def main(default_config="configs/gen_ppl.yaml", allow_distributed=True, debug=False):
    parser = argparse.ArgumentParser(description="固定 seed=42 / 1024 samples / GPT-2-Large 的多模型 Gen PPL")
    parser.add_argument("command", nargs="?", choices=["all"], default="all")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    run_gen_ppl(load_config(args.config, args.overrides, debug), allow_distributed, debug)


if __name__ == "__main__":
    main()
