"""训练期间评测当前内存权重，直接复用 eval_gen_ppl.sh 对应的统一实现。"""

import json

import torch

from .. import distributed
from ..config import project_path, write_json
from ..gen_ppl import (generate_records, score_records, scorer_on_device, save_summary,
                       protocol_metadata, validate_protocol)
from ..gen_ppl_protocol import manifest, aggregate_scores


def training_ppl(student, tokenizer, config, run_dir, step, *, scorer_cache, debug=False):
    device = next(student.parameters()).device
    output = run_dir / "ppl" / f"step_{step:06d}"
    output.mkdir(parents=True, exist_ok=True)
    generation = {key: config["evaluation"][key] for key in ("num_steps", "sequence_length")}
    if config["model"]["family"] == "elf":
        generation["sampling"] = dict(config["evaluation"]["sampling"])
    entry = {"name": "current_student", "generation": generation}
    current = manifest([entry])
    current["runtime"]["generation_batch_size"] = config["training"]["ppl_evaluation"]["batch_size"]
    if debug:
        current["protocol"]["num_samples"] = config["evaluation"]["num_samples"]
    validate_protocol(current["protocol"], debug)
    protocol = protocol_metadata(current, debug)
    metadata = {"diffusion": {**config.get("diffusion", {}),
                    "self_cond_prob": config["training"].get("self_condition_probability", 0.5)},
                "sampling": config["evaluation"].get("sampling", {}), "selected_weights_key": None}
    torch.set_float32_matmul_precision("highest")
    # 与 CLI 同一组 sample_id、噪声、生成/清理/评分函数；不另写一套 PPL 口径。
    records = generate_records(student, tokenizer, {"family": config["model"]["family"]},
                               entry, current, metadata, output)
    cache_dir = config["model"]["cache_dir"]
    try:
        scorer, scorer_tokenizer = scorer_on_device(scorer_cache,
            str(project_path(cache_dir)) if cache_dir else None, device)
        scores = score_records(records, scorer, scorer_tokenizer, device)
    finally:
        if "model" in scorer_cache:
            scorer_cache["model"].cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    shards = distributed.gather_main((records, scores))
    if not distributed.is_main():
        return None
    metrics = aggregate_scores([shard[1] for shard in shards], current["protocol"]["num_samples"])
    samples = sorted((row for shard in shards for row in shard[0]), key=lambda row: row["sample_id"])
    with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    result = {"name": "current_student", "family": config["model"]["family"], **generation, **metrics,
              "step": step, "weight_source": "in_memory_student_after_optimizer_step",
              "effective_sampling": metadata["effective_sampling"]}
    save_summary(output, protocol, [result])
    write_json(output / "protocol.json", protocol)
    return {"model": "current_student", "gen_ppl": metrics["gen_ppl"],
            "sample_entropy": metrics["sample_entropy"], "repetition_4gram": metrics["repetition_4gram"],
            "num_samples": metrics["num_samples"], "score_status": metrics["score_status"], "directory": str(output)}
