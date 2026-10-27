"""四种真实模型共用的 on-policy 入口，沿用现有训练、checkpoint 和统一评测。"""

import argparse
import gc
import json
import math
from functools import partial
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .. import distributed, progress, models
from ..backbones.elf import ELF
from ..backbones.langflow import LangFlow, LangFlowConfig
from ..config import create_run_dir, load_yaml_config, project_path, write_json
from ..elf import models as elf_models
from ..gen_ppl import run_gen_ppl
from ..gen_ppl_protocol import write_manifest
from ..lr_schedule import validate_lr_schedule
from ..train import train, seed_everything, save_checkpoint as save_langflow
from .evaluate import training_ppl
from .losses import OnPolicyLoss, METRIC_KEYS


def load_config(path, overrides=()):
    config = load_yaml_config(path, overrides)
    family, rollout, loss, training = config["model"]["family"], config["on_policy"], config["loss"], config["training"]
    loss.setdefault("kl_weight", 0.0)
    if family not in ("langflow", "elf"):
        raise ValueError("on-policy model.family 只支持 langflow / elf")
    coefficients = [loss[key] for key in ("ce_weight", "kl_weight", "mse_weight")]
    if any(not math.isfinite(value) or value < 0 for value in coefficients) or sum(coefficients) == 0:
        raise ValueError("CE/KL/MSE 权重必须为有限非负数，且至少启用一项")
    if loss["ce_target"] not in ("soft", "hard"):
        raise ValueError("ce_target 只支持 teacher soft distribution 或 hard argmax")
    if min(training["max_steps"], training["batch_size"], training["grad_accumulation"], training["log_every"],
           training["validate_every"], training["validation_batches"], rollout["rollout_batch_size"], rollout["refresh_every"]) < 1:
        raise ValueError("训练与 rollout 的 batch、步数、刷新间隔必须为正")
    steps = config["evaluation"]["num_steps"]
    if steps < 2 or not 1 <= rollout["states_per_rollout"] <= steps:
        raise ValueError("采样至少 2 次前向，抽取状态数不能超过 rollout 步数")
    posterior_weight = loss["ce_weight"] + loss["kl_weight"]
    minimum = rollout["rollout_batch_size"] * (rollout["states_per_rollout"] + int(family == "elf" and posterior_weight > 0))
    if rollout["buffer_capacity"] < minimum:
        raise ValueError("buffer_capacity 至少要容纳一轮采集的所有状态")
    stopping = training["early_stopping"]
    if stopping["metric"] not in METRIC_KEYS or stopping["patience"] < 1 or stopping["min_steps"] < 0 or stopping["min_delta"] < 0:
        raise ValueError("on-policy early_stopping 指标或参数无效")
    if family == "elf":
        native = json.loads(project_path(config["model"]["architecture_config"]).read_text(encoding="utf-8"))
        config["model"]["architecture"] = native["architecture"]
        config["model"]["encoder"] = native["encoder"]
        config["diffusion"] = dict(native["diffusion"])
        training["self_condition_probability"] = config["diffusion"].pop("self_cond_prob", 0.5)
        config["evaluation"]["sampling"] = {**native["sampling"], **config["evaluation"].get("sampling", {})}
        sampling = config["evaluation"]["sampling"]
        if sampling["sampling_method"] not in ("ode", "sde") or sampling["cfg_scale"] != 1:
            raise ValueError("首版 ELF 支持 ODE/SDE 和原版无条件 CFG=1")
        if config["model"]["architecture"]["num_self_cond_cfg_tokens"] <= 0 or training["self_condition_probability"] <= 0:
            raise ValueError("当前 ELF rollout 使用预训练模型的 SC-CFG prefix 结构")
        if posterior_weight and loss["mse_weight"] and not 0 < rollout["decoder_probability"] < 1:
            raise ValueError("同时启用 ELF posterior/MSE 时 decoder_probability 必须在 (0,1)")
    monitoring = training.get("ppl_evaluation", {})
    if monitoring.get("enabled", False) and monitoring["interval"] != -1:
        if min(monitoring["interval"], monitoring["batch_size"]) < 1:
            raise ValueError("统一训练评测的 interval、generation batch 必须为正，或用 interval=-1 关闭")
    validate_lr_schedule(training)
    return config


def student_files(config):
    value = config["model"]["student_weights"]
    if not value:
        raise ValueError("请先填写 model.student_weights：off-policy student 目录或 model.safetensors 文件")
    path = project_path(value)
    directory = path if path.is_dir() else path.parent
    weights = directory / "model.safetensors" if path.is_dir() else path
    architecture = (project_path(config["model"]["student_config"])
                    if config["model"]["student_config"] else directory / "config.json")
    if not weights.is_file() or not architecture.is_file():
        raise FileNotFoundError("off-policy 权重和架构 JSON 必须存在；safetensors 本身不包含完整架构")
    return weights, architecture, directory


def load_pair(config, weights, architecture, directory, device):
    settings = json.loads(architecture.read_text(encoding="utf-8"))
    if config["model"]["family"] == "langflow":
        if settings.get("model_type") not in ("LangFlow", "langflow"):
            raise ValueError("off-policy checkpoint 不是 LangFlow")
        teacher = models.load_teacher(config)
        student = LangFlow(LangFlowConfig.from_json_file(architecture))
        student.load_state_dict(load_file(str(weights)), strict=True)
        student.to(device)
        if (student.config.hidden_size, student.config.vocab_size) != (teacher.config.hidden_size, teacher.config.vocab_size):
            raise ValueError("LangFlow teacher/student 必须共享 latent 维度与词表")
        # 共享坐标和噪声 schedule 是本项目容量蒸馏的必要条件；不静默覆盖已有权重。
        if not torch.equal(student._get_embedding_matrix(), teacher._get_embedding_matrix()):
            raise ValueError("off-policy student 的 embedding 与 teacher 不一致")
        if any(not torch.equal(value, teacher.proposal.state_dict()[key]) for key, value in student.proposal.state_dict().items()):
            raise ValueError("off-policy student 的 Gumbel schedule 与 teacher 不一致")
        student.backbone.vocab_embed.requires_grad_(False)
        student.proposal.requires_grad_(False)
        limit = min(student.config.model_length, teacher.config.model_length)
    else:
        if settings.get("model_type") != "ELF":
            raise ValueError("off-policy checkpoint 不是 ELF")
        teacher = elf_models.load_teacher(config, device)
        student = ELF(**settings["architecture"])
        student.load_state_dict(load_file(str(weights)), strict=True)
        student.to(device)
        student.architecture = dict(settings["architecture"])
        student.gradient_checkpointing = config["model"]["gradient_checkpointing"]
        student.architecture["gradient_checkpointing"] = student.gradient_checkpointing
        if (student.text_encoder_dim, student.vocab_size) != (teacher.text_encoder_dim, teacher.vocab_size):
            raise ValueError("ELF teacher/student 的 latent 维度和词表不匹配")
        if settings["encoder"] != config["model"]["encoder"] or any(
                settings["diffusion"][key] != config["diffusion"][key]
                for key in ("latent_mean", "latent_std", "denoiser_noise_scale", "t_eps")):
            raise ValueError("ELF off-policy checkpoint 的 T5/latent/noise 定义与配置不一致")
        if config["loss"]["mse_weight"] == 0:
            student.final_layer.requires_grad_(False)
        if config["loss"]["ce_weight"] + config["loss"]["kl_weight"] == 0:
            for key in ("proj_kernel", "proj_bias", "unembed_kernel", "unembed_bias"):
                getattr(student, key).requires_grad_(False)
            # decoder mode token 在纯去噪分支恒为零门控，也应避免 AdamW 对它做无监督衰减。
            if hasattr(student, "mode_tokens"):
                student.mode_tokens.requires_grad_(False)
        limit = min(student.max_length, teacher.max_length)
    if not 0 < config["evaluation"]["sequence_length"] <= limit:
        raise ValueError("rollout 长度超过真实预训练模型 context")
    source = directory / "distillation.json"
    original = json.loads(source.read_text(encoding="utf-8")) if source.exists() else {}
    config["model"]["keep_layers"] = original.get("keep_layers")
    return teacher, student, original


def collate_slots(indices, length):
    # 无条件训练 slot 只决定 batch/有效 token 数；真正输入由真实 student rollout 提供。
    return {"sample_id": torch.tensor(indices), "attention_mask": torch.ones(len(indices), length, dtype=torch.bool)}


def make_loader(config, split):
    training = config["training"]
    batches = training["max_steps"] * training["grad_accumulation"] if split == "train" else training["validation_batches"]
    total = batches * training["batch_size"] * distributed.world_size()
    return DataLoader(range(distributed.rank(), total, distributed.world_size()),
        batch_size=training["batch_size"], num_workers=0,
        collate_fn=partial(collate_slots, length=config["evaluation"]["sequence_length"]),
        generator=torch.Generator().manual_seed(config["run"]["seed"]))


def save_checkpoint(student, tokenizer, run_dir, step, config, name=None):
    save = save_langflow if config["model"]["family"] == "langflow" else elf_models.save_checkpoint
    path = save(student, tokenizer, run_dir, step, config, name)
    if distributed.is_main():
        metadata = json.loads((path / "distillation.json").read_text(encoding="utf-8"))
        write_json(path / "distillation.json", {**metadata, "stage": "on_policy",
            "off_policy_student": config["model"]["student_weights"], "on_policy": config["on_policy"]})
    distributed.barrier()
    return path


def evaluation_entries(config, initial_weights, initial_architecture, final_weights, final_architecture):
    generation = {key: config["evaluation"][key] for key in ("num_steps", "sequence_length")}
    if config["model"]["family"] == "elf":
        # 两组都覆盖 sampling，避免起点沿用旧 SDE、新 checkpoint 却使用 ODE 的比较混淆。
        generation["sampling"] = dict(config["evaluation"]["sampling"])
    return [{"name": "off_policy_initial", "architecture": str(initial_architecture),
             "weights": str(initial_weights), "generation": generation},
            {"name": "on_policy_student", "architecture": str(final_architecture),
             "weights": str(final_weights), "generation": generation}]


def run_experiment(config, command="all", checkpoint=None, allow_distributed=True, debug=False):
    if command != "check-distributed":
        weights, architecture, directory = student_files(config)
    communication = config["distributed"]
    owns_group = distributed.setup(config["model"]["device"], allow_distributed, communication["control_timeout_seconds"])
    run_dir = Path(distributed.broadcast_main(str(create_run_dir(config)) if distributed.is_main() else None))
    progress.configure(run_dir, distributed.rank())
    seed_everything(config["run"]["seed"] + distributed.rank())
    device = torch.device(config["model"]["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("真实 rollout 训练请在 CUDA 服务器运行")
        device = torch.device("cuda", torch.cuda.current_device())
    try:
        if command == "check-distributed":
            distributed.setup_tensor_group(device, communication["tensor_timeout_seconds"])
        else:
            if command in ("train", "all"):
                if distributed.is_main():
                    source = str(directory) if (directory / "tokenizer_config.json").exists() else config["model"]["tokenizer"]
                    tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=config["model"]["cache_dir"])
                    tokenizer.save_pretrained(run_dir / "tokenizer")
                distributed.barrier()
                tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer", local_files_only=True)
                distributed.setup_tensor_group(device, communication["tensor_timeout_seconds"])
                teacher, student, original = load_pair(config, weights, architecture, directory, device)
                vocab_size = student.config.vocab_size if config["model"]["family"] == "langflow" else student.vocab_size
                if len(tokenizer) != vocab_size:
                    raise ValueError("off-policy tokenizer 的词表与实际 student 架构不一致")
                distributed.barrier()
                if distributed.is_main():
                    write_json(run_dir / "initialization.json", {"stage": "on_policy", "weights": str(weights),
                        "architecture": str(architecture), "source_distillation": original,
                        "state_source": "student_real_rollout", "teacher_inputs": "same_z_time_student_previous_self_cond"})
                    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
                if config["model"]["family"] == "langflow":
                    counts = models.parameter_counts
                else:
                    counts = elf_models.parameter_counts
                checkpoint = train(teacher, student, tokenizer, config, run_dir, loss_fn=OnPolicyLoss(config),
                    loader_fn=make_loader, checkpoint_fn=save_checkpoint, counts_fn=counts,
                    metric_keys=METRIC_KEYS, ppl_fn=partial(training_ppl, scorer_cache={}, debug=debug))
                del teacher, student
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if checkpoint is None:
                raise ValueError("evaluate 需要 --checkpoint")
            final = project_path(checkpoint)
            final_dir = final if final.is_dir() else final.parent
            entries = evaluation_entries(config, weights, architecture,
                final if final.is_file() else final / "model.safetensors", final_dir / "config.json")
            manifest = run_dir / "gen_ppl_eval.yaml"
            if distributed.is_main():
                write_manifest(manifest, entries)
            distributed.barrier()
            if command in ("all", "evaluate"):
                evaluation = load_yaml_config(manifest)
                evaluation["run"].update(name=config["run"]["name"] + "_gen_ppl", root=config["run"]["root"])
                evaluation["runtime"].update(device=str(device), cache_dir=config["model"]["cache_dir"],
                                               generation_batch_size=config["evaluation"]["batch_size"])
                if debug:
                    evaluation["protocol"]["num_samples"] = config["evaluation"]["num_samples"]
                evaluated = run_gen_ppl(evaluation, allow_distributed=allow_distributed, debug=debug)
                if distributed.is_main():
                    write_json(run_dir / "evaluation.json", {"run_dir": str(evaluated)})
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "completed", "command": command,
                                                 "checkpoint": str(checkpoint) if checkpoint else None})
            print(f"[on-policy] {run_dir}", flush=True)
        distributed.barrier()
    except Exception as error:
        if distributed.is_main():
            write_json(run_dir / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        if owns_group:
            distributed.close()
    return run_dir


def main(default_config="configs/on_policy_langflow_owt.yaml", allow_distributed=True, debug=False):
    parser = argparse.ArgumentParser(description="真实 student rollout 的 CE/KL/MSE on-policy 蒸馏")
    parser.add_argument("command", nargs="?", choices=["train", "all", "evaluate", "check-distributed"], default="all")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoint", help="单独评测时提供第二阶段 student checkpoint")
    args = parser.parse_args()
    run_experiment(load_config(args.config, args.overrides), args.command, args.checkpoint, allow_distributed, debug)


if __name__ == "__main__":
    main()
