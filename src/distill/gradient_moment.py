"""GPT-2-Large Gradient Moment：论文 Eq.14 的独立梯度差内积估计。"""

import json
import math

import torch

from . import distributed, progress
from .config import load_yaml_config, project_path, write_json
from .corpus_ppl import select_samples, token_nll
from .gen_ppl_protocol import MAX_LENGTH, SCORER

VERSION = "gpt2_large_gm_v1"


def validate_config(config):
    settings = config.get("gradient_moment", {})
    if not settings.get("enabled", False):
        return
    count = config["protocol"]["num_samples"]
    if count < 2 or count % 2:
        raise ValueError("GM 需要至少 2 条且为偶数的样本，组成不重复的独立梯度对")
    reference = project_path(settings["reference_config"])
    if not reference.is_file():
        raise FileNotFoundError(f"GM 真实数据配置不存在: {reference}")
    for entry in config["models"]:
        if entry.get("corpus") not in ("owt", "lm1b"):
            raise ValueError(f"启用 GM 时，模型 {entry['name']} 必须显式填写 corpus: owt 或 lm1b")


def protocol_metadata(config):
    settings = config.get("gradient_moment", {})
    if not settings.get("enabled", False):
        return {"enabled": False}
    return {"enabled": True, "version": VERSION, "scorer": SCORER,
        "reference_config": settings["reference_config"], "seed": config["protocol"]["seed"],
        "num_reference_samples_per_corpus": config["protocol"]["num_samples"], "max_length": MAX_LENGTH,
        "loss": "per_sequence_mean_shifted_token_nll_excluding_padding",
        "parameters": "all_unique_reference_parameters_including_tied_embedding_once",
        "estimator": "mean_of_independent_gradient_difference_inner_products_eq14",
        "pairing": "consecutive_sample_ids_pair_id_mod_world_size",
        "gradient_dtype": "float32", "inner_product_dtype": "float64",
        "invalid_samples": "skip_pair_without_resampling_and_report_coverage",
        "negative_estimates": "retained"}


def reference_config(config):
    result = load_yaml_config(config["gradient_moment"]["reference_config"])
    result["run"]["seed"] = config["protocol"]["seed"]
    result["evaluation"].update(num_samples=config["protocol"]["num_samples"], max_length=MAX_LENGTH)
    return result


def prepare_reference(config, corpus, tokenizer, run_dir, cache):
    references = cache.setdefault("gm_references", {})
    if corpus not in references:
        path = run_dir / "gradient_moment" / f"reference_{corpus}.json"
        if distributed.is_main():
            with progress.phase("unified.gm.reference", corpus=corpus):
                payload = select_samples(reference_config(config), corpus, tokenizer)
                payload["protocol"] = protocol_metadata(config)
                write_json(path, payload)
        # 只由 rank 0 选择/编码真实语料，其余卡从本次运行的同一份记录读取。
        distributed.barrier()
        references[corpus] = json.loads(path.read_text(encoding="utf-8"))
    return references[corpus]


def pair_ids(count, rank, world_size):
    if count < 2 or count % 2:
        raise ValueError("GM 样本数必须为至少 2 的偶数")
    return range(rank, count // 2, world_size)


def gradient_inner_product(first, second):
    # 逐参数计算，不拼接数 GB 的完整梯度向量；乘积和累加使用 FP64。
    if len(first) != len(second):
        raise ValueError("GM 的两组参数梯度长度不同")
    total = torch.zeros((), dtype=torch.float64, device=first[0].device)
    for left, right in zip(first, second):
        total += torch.dot(left.reshape(-1).double(), right.reshape(-1).double())
    return float(total.item())


def summarize_pairs(shards, expected_pairs):
    pairs = sorted((row for shard in shards for row in shard), key=lambda row: row["pair_id"])
    if [row["pair_id"] for row in pairs] != list(range(expected_pairs)):
        raise ValueError("GM pair_id 必须恰好覆盖所有梯度对，不重复、不缺失")
    values = [row["gm"] for row in pairs if row["gm"] is not None]
    if any(not math.isfinite(value) for value in values):
        raise FloatingPointError("GM 出现非有限梯度内积")
    mean = math.fsum(values) / len(values) if values else None
    standard_error = (math.sqrt(math.fsum((value - mean) ** 2 for value in values)
                        / (len(values) - 1) / len(values)) if len(values) > 1 else None)
    return {"gm": mean, "gm_standard_error": standard_error, "gm_num_pairs": len(values),
        "gm_num_skipped_pairs": len(pairs) - len(values), "gm_num_samples_used": 2 * len(values),
        "gm_status": ("completed" if len(values) == len(pairs) else "partial_coverage") if values else "no_valid_pairs",
        "per_pair": pairs}


def sample_gradient(scorer, parameters, ids, device):
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(tokens)
    logits = scorer(input_ids=tokens, attention_mask=mask, use_cache=False).logits
    sums, counts = token_nll(logits, tokens, mask)
    # 与统一 PPL 共用 shifted CE 和 padding 规则；每个独立样本先按有效 token 平均。
    loss = (sums / counts).mean()
    return list(torch.autograd.grad(loss, parameters))


def gradient_difference(scorer, parameters, generated_ids, real_ids, device):
    difference = sample_gradient(scorer, parameters, generated_ids, device)
    real = sample_gradient(scorer, parameters, real_ids, device)
    for sample, data in zip(difference, real):
        sample.sub_(data)
    return difference


@torch.enable_grad()
def score_gradient_moment(records, scorer, tokenizer, reference, device, config, corpus, output):
    # 文本很小，通过 Gloo 汇合；梯度始终留在各卡，只汇总每对的标量结果。
    shards = distributed.gather_main(records)
    all_records = sorted((row for shard in shards for row in shard), key=lambda row: row["sample_id"]) if distributed.is_main() else None
    all_records = distributed.broadcast_main(all_records)
    count = config["protocol"]["num_samples"]
    if [row["sample_id"] for row in all_records] != list(range(count)) or len(reference["samples"]) != count:
        raise ValueError("GM 的生成样本/真实参考数量与统一协议不一致")
    if distributed.is_main():
        # 在昂贵的反向传播前保留生成文本，GM 异常也不会丢掉已完成的采样结果。
        with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    generated = [tokenizer(row["text"], add_special_tokens=False, truncation=True,
        max_length=MAX_LENGTH, return_attention_mask=False)["input_ids"] if row["text"].strip() else []
        for row in all_records]
    real = [row["input_ids"] for row in reference["samples"]]
    parameters = list(scorer.parameters())
    flags, was_training = [parameter.requires_grad for parameter in parameters], scorer.training
    local = []
    try:
        scorer.eval().requires_grad_(True)
        for pair in pair_ids(count, distributed.rank(), distributed.world_size()):
            first, second = 2 * pair, 2 * pair + 1
            if min(len(generated[first]), len(generated[second])) < 2:
                local.append({"pair_id": pair, "gm": None, "sample_ids": [first, second]})
                continue
            with progress.phase("unified.gm.pair", pair_id=pair):
                left = gradient_difference(scorer, parameters, generated[first], real[first], device)
                right = gradient_difference(scorer, parameters, generated[second], real[second], device)
                value = gradient_inner_product(left, right)
                del left, right
            local.append({"pair_id": pair, "gm": value, "sample_ids": [first, second]})
    finally:
        # autograd.grad 不写入 .grad，不创建 optimizer；恢复共享 scorer 的所有参数开关和模式。
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
        scorer.train(was_training)
    shards = distributed.gather_main(local)
    summary = None
    if distributed.is_main():
        summary = summarize_pairs(shards, count // 2)
        summary["gm_reference_corpus"] = corpus
        write_json(output / "gradient_moment.json", {"protocol": protocol_metadata(config),
            "corpus": corpus, "reference_source": reference["source"], **summary})
        summary.pop("per_pair")
    return distributed.broadcast_main(summary)
