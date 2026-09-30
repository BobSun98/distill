"""加载真实 teacher，并从其权重构造删层 student。"""

import copy
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .backbones.langflow import LangFlow, LangFlowConfig
from .config import project_path


def load_model(model_name, device, cache_dir=None, config_path=None):
    cache_dir = str(project_path(cache_dir)) if cache_dir else None
    local = project_path(model_name)
    if local.is_file():
        if config_path is None:
            raise ValueError("单独加载 safetensors 时必须指定 model.teacher_config")
        weights, architecture = local, project_path(config_path)
    elif local.is_dir():
        weights, architecture = local / "model.safetensors", local / "config.json"
    else:
        # 只下载配置和权重，始终使用本项目的模型实现，不执行远端 Python 代码。
        architecture = hf_hub_download(model_name, "config.json", cache_dir=cache_dir)
        weights = hf_hub_download(model_name, "model.safetensors", cache_dir=cache_dir)
    config = LangFlowConfig.from_json_file(str(architecture))
    model = LangFlow(config)
    model.load_state_dict(load_file(str(weights)), strict=True)
    return model.to(device).eval()


def load_teacher(config):
    settings = config["model"]
    device = torch.device(settings["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前机器没有 CUDA；本地调试请设置 model.device=cpu")
    teacher = load_model(settings["teacher"], device, settings["cache_dir"],
                         settings["teacher_config"])
    teacher.requires_grad_(False)
    return teacher


def load_tokenizer(config):
    settings = config["model"]
    name = settings["tokenizer"]
    local = project_path(name)
    tokenizer = AutoTokenizer.from_pretrained(
        str(local) if local.is_dir() else name,
        cache_dir=str(project_path(settings["cache_dir"])) if settings["cache_dir"] else None)
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("OWT packing 需要预训练 tokenizer 的 BOS/EOS token")
    return tokenizer


def build_student(teacher, keep_layers):
    depth = teacher.config.n_blocks
    if not keep_layers or keep_layers != sorted(set(keep_layers)):
        raise ValueError("keep_layers 必须是非空、递增且不重复的层编号")
    if keep_layers[0] < 0 or keep_layers[-1] >= depth:
        raise ValueError(f"teacher 只有 {depth} 层，keep_layers 越界")
    config = copy.deepcopy(teacher.config)
    config.n_blocks = len(keep_layers)
    student = LangFlow(config)
    teacher_weights = teacher.state_dict()
    weights = {key: value for key, value in teacher_weights.items()
               if not key.startswith("backbone.blocks.")}
    # 只重编号保留的 block；embedding、输出头、时间编码、SC 投影与 schedule 全部复制。
    for student_index, teacher_index in enumerate(keep_layers):
        prefix = f"backbone.blocks.{teacher_index}."
        for key, value in teacher_weights.items():
            if key.startswith(prefix):
                weights[f"backbone.blocks.{student_index}." + key[len(prefix):]] = value
    student.load_state_dict(weights, strict=True)
    student.backbone.vocab_embed.requires_grad_(False)
    student.proposal.requires_grad_(False)
    return student.to(next(teacher.parameters()).device)


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    embedding = sum(p.numel() for p in model.backbone.vocab_embed.parameters())
    return {"total": total, "non_embedding": total - embedding,
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
