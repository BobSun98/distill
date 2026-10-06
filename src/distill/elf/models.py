"""真实 ELF/T5 权重加载、EMA teacher 与删层 student 初始化。"""

import copy
import json
import re
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, T5EncoderModel

from .. import distributed, progress
from ..backbones.elf import ELF
from ..config import project_path, write_json


def resolve_teacher(config):
    settings = config["model"]
    result = None
    if distributed.is_main():
        name, selected = settings["teacher"], settings["checkpoint_file"]
        local = project_path(name)
        if local.is_file():
            result = str(local)
        elif local.is_dir():
            if selected:
                result = str(local / selected)
            elif (local / "model.safetensors").exists():
                result = str(local / "model.safetensors")
            else:
                candidates = [p for p in local.glob("checkpoint_*") if p.is_file()]
                if not candidates:
                    raise FileNotFoundError(f"ELF checkpoint 目录没有权重: {local}")
                result = str(max(candidates, key=lambda p: int(re.search(r"\d+", p.name).group())))
        else:
            if selected is None:
                files = HfApi().list_repo_files(name)
                candidates = [p for p in files if re.fullmatch(r"checkpoint_\d+(?:\.pt)?", Path(p).name)]
                if candidates:
                    selected = max(candidates, key=lambda p: int(re.search(r"checkpoint_(\d+)", p).group(1)))
                elif "model.safetensors" in files:
                    selected = "model.safetensors"
                else:
                    raise ValueError("请在 model.checkpoint_file 指定原生 ELF 权重文件名")
            cache = settings["cache_dir"]
            result = hf_hub_download(name, selected, cache_dir=str(project_path(cache)) if cache else None)
    return distributed.broadcast_main(result)


def load_teacher(config, device):
    path = resolve_teacher(config)
    architecture = copy.deepcopy(config["model"]["architecture"])
    with progress.phase("elf.teacher.load", path=path, use_ema=config["model"]["use_ema"]):
        model = ELF(**architecture)
        if path.endswith(".safetensors"):
            weights = load_file(path)
            ema_used = False
        else:
            # 原生 ELF checkpoint 含 params/EMA/optimizer；只选模型，不恢复原训练状态。
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            ema = payload.get("ema_params1")
            ema_used = bool(config["model"]["use_ema"] and ema)
            weights = ema if ema_used else payload["params"]
        model.load_state_dict(weights, strict=True)
        model.architecture = architecture
        model.to(device).eval().requires_grad_(False)
    if distributed.is_main():
        progress.emit("elf.teacher.loaded", ema_used=ema_used, depth=model.depth)
    return model


def resolve_encoder(config):
    settings = config["model"]
    result = None
    if distributed.is_main():
        local = project_path(settings["encoder"])
        if local.is_dir():
            result = str(local)
        else:
            files = HfApi().list_repo_files(settings["encoder"])
            weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
            cache = settings["cache_dir"]
            result = snapshot_download(settings["encoder"],
                allow_patterns=["config.json", weights, "tokenizer*", "spiece.model", "special_tokens_map.json"],
                cache_dir=str(project_path(cache)) if cache else None)
    return distributed.broadcast_main(result)


def load_encoder(path, config, device):
    with progress.phase("elf.encoder.load", model=config["model"]["encoder"]):
        encoder = T5EncoderModel.from_pretrained(path).to(device).eval().requires_grad_(False)
    if encoder.config.d_model != config["model"]["architecture"]["text_encoder_dim"]:
        raise ValueError("冻结 T5 encoder 的 latent 维度与 ELF 架构不匹配")
    return encoder


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("ELF 需要真实 T5 tokenizer 的 PAD/EOS")
    return tokenizer


def build_student(teacher, config):
    layers = config["model"]["keep_layers"]
    if not layers or layers != sorted(set(layers)) or layers[0] < 0 or layers[-1] >= teacher.depth:
        raise ValueError("keep_layers 必须是 teacher 范围内非空、递增且不重复的层编号")
    architecture = copy.deepcopy(teacher.architecture)
    architecture["depth"] = len(layers)
    architecture["gradient_checkpointing"] = config["model"]["gradient_checkpointing"]
    student = ELF(**architecture)
    original = teacher.state_dict()
    weights = {name: tensor for name, tensor in original.items() if not name.startswith("blocks.")}
    for new_index, old_index in enumerate(layers):
        prefix = f"blocks.{old_index}."
        weights.update({f"blocks.{new_index}." + name[len(prefix):]: tensor
                        for name, tensor in original.items() if name.startswith(prefix)})
    student.load_state_dict(weights, strict=True)
    student.architecture = architecture
    # 单头消融冻结未使用的输出头，DDP 不需要启用 find_unused_parameters。
    loss = config["loss"]
    if loss["flow_weight"] + loss["native_flow_weight"] == 0:
        student.final_layer.requires_grad_(False)
    if loss["kd_weight"] + loss["ce_weight"] == 0:
        for name in ("proj_kernel", "proj_bias", "unembed_kernel", "unembed_bias"):
            getattr(student, name).requires_grad_(False)
    if config["training"]["self_condition_probability"] == 0:
        student.self_cond_proj.requires_grad_(False)
    return student.to(next(teacher.parameters()).device)


def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def save_checkpoint(student, tokenizer, run_dir, step, config, name=None):
    path = run_dir / "checkpoints" / (name or f"step_{step:06d}")
    if distributed.is_main():
        model = distributed.unwrap(student)
        path.mkdir(parents=True, exist_ok=True)
        with progress.phase("elf.checkpoint.save", step=step, path=str(path)):
            save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()},
                       str(path / "model.safetensors"), metadata={"format": "pt"})
            write_json(path / "config.json", {"model_type": "ELF", "architecture": model.architecture,
                                              "encoder": config["model"]["encoder"],
                                              "diffusion": config["diffusion"],
                                              "sampling": config["evaluation"]["sampling"]})
            tokenizer.save_pretrained(path)
            write_json(path / "distillation.json", {"step": step, "teacher": config["model"]["teacher"],
                       "keep_layers": config["model"]["keep_layers"], "loss": config["loss"]})
    distributed.barrier()
    return path


def load_student(path, device):
    path = project_path(path)
    directory = path.parent if path.is_file() else path
    weights = path if path.is_file() else path / "model.safetensors"
    settings = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if settings["model_type"] != "ELF":
        raise ValueError("checkpoint 不是本项目保存的 ELF student")
    model = ELF(**settings["architecture"])
    model.load_state_dict(load_file(str(weights)), strict=True)
    model.architecture = settings["architecture"]
    return model.to(device).eval()
