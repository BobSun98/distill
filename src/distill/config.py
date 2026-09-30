"""YAML 配置与少量实验运行工具。"""

import copy
import json
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _merge(base, changes):
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path, overrides=()):
    path = project_path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base = config.pop("extends", None)
    if base:
        config = _merge(load_config(path.parent / base), config)
    for override in overrides:
        key, value = override.split("=", 1)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(f"未知配置项: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    training, loss, data = config["training"], config["loss"], config["data"]
    if min(training["max_steps"], training["batch_size"], training["grad_accumulation"],
           training["log_every"], training["validate_every"]) < 1:
        raise ValueError("训练步数、batch size、梯度累积与日志间隔必须为正数")
    if min(loss["kd_weight"], loss["ce_weight"]) < 0 or sum(loss.values()) <= 0:
        raise ValueError("KD/CE 权重不能为负，且至少启用一个 loss")
    if not 0 <= training["self_condition_probability"] <= 1:
        raise ValueError("self_condition_probability 必须在 [0, 1] 内")
    if data["sequence_length"] < 3:
        raise ValueError("sequence_length 至少为 3，包含 BOS 和 EOS")
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def create_run_dir(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = project_path(config["run"]["root"]) / f'{timestamp}_{config["run"]["name"]}'
    path.mkdir(parents=True)
    (path / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path
