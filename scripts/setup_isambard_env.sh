#!/usr/bin/env bash
# 从 ARM64 二进制包创建独立环境；环境、下载缓存和安装记录都存入 project。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DISTILL_STORAGE_ROOT=${DISTILL_STORAGE_ROOT:-/projects/u6xs/bobsun.u6xs/distill}
DISTILL_ENV_PREFIX=${DISTILL_ENV_PREFIX:-$DISTILL_STORAGE_ROOT/envs/distill}
CONDA_BIN=${CONDA_BIN:-$HOME/miniforge3/bin/conda}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$DISTILL_STORAGE_ROOT/cache/conda}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$DISTILL_STORAGE_ROOT/cache/pip}

[[ $(uname -m) == aarch64 ]] || { echo '此环境配置用于 Isambard ARM64 节点。' >&2; exit 2; }
[[ -x "$CONDA_BIN" ]] || { echo "找不到 Conda: $CONDA_BIN" >&2; exit 2; }
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$DISTILL_STORAGE_ROOT/environment" \
    "$DISTILL_STORAGE_ROOT/run/launch" "$DISTILL_STORAGE_ROOT/assets"
cd "$PROJECT_ROOT"

if [[ ! -x "$DISTILL_ENV_PREFIX/bin/python" ]]; then
    "$CONDA_BIN" env create --yes --prefix "$DISTILL_ENV_PREFIX" \
        --file configs/environment_isambard.yaml
fi
# 禁止回退到源码构建，避免在登录节点意外编译大型依赖。
"$DISTILL_ENV_PREFIX/bin/python" -m pip install --only-binary=:all: \
    --requirement configs/requirements_isambard.txt
# NVIDIA 0.7.1 的 ARM wheel 文件名为 aarch64，内部却误写 sbsa；核实真实库后修正标签。
# https://github.com/pytorch/pytorch/issues/202632 描述了同类上游打包问题。
"$DISTILL_ENV_PREFIX/bin/python" - <<'PY'
import base64
import csv
import ctypes
import hashlib
import platform
from importlib.metadata import distribution

dist = distribution("nvidia-cusparselt-cu12")
wheel = next(dist.locate_file(path) for path in dist.files if str(path).endswith(".dist-info/WHEEL"))
metadata = wheel.read_text()
old_tag = "Tag: py3-none-manylinux2014_sbsa"
if platform.machine() == "aarch64" and dist.version == "0.7.1" and old_tag in metadata:
    library = dist.locate_file("nvidia/cusparselt/lib/libcusparseLt.so.0")
    with library.open("rb") as handle:
        header = handle.read(20)
    assert header[:4] == b"\x7fELF" and header[18:20] == b"\xb7\x00", "cuSPARSELt 不是 AArch64 ELF"
    ctypes.CDLL(str(library))
    wheel.write_text(metadata.replace(old_tag, "Tag: py3-none-manylinux2014_aarch64"))
    record = wheel.with_name("RECORD")
    rows = list(csv.reader(record.read_text().splitlines()))
    digest = base64.urlsafe_b64encode(hashlib.sha256(wheel.read_bytes()).digest()).rstrip(b"=").decode()
    for row in rows:
        if row[0].endswith(".dist-info/WHEEL"):
            row[1:] = ["sha256=" + digest, str(wheel.stat().st_size)]
    with record.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    print("已核对 cuSPARSELt AArch64 库并修正 wheel 的 sbsa 平台标签。")
PY
"$DISTILL_ENV_PREFIX/bin/python" -m pip check
"$DISTILL_ENV_PREFIX/bin/python" -m pip freeze > "$DISTILL_STORAGE_ROOT/environment/pip-freeze.txt"
"$CONDA_BIN" list --prefix "$DISTILL_ENV_PREFIX" --explicit > "$DISTILL_STORAGE_ROOT/environment/conda-explicit.txt"
"$DISTILL_ENV_PREFIX/bin/python" - <<'PY'
import platform
import torch
from distill.progressive_rank import load_config

print(f"Python {platform.python_version()} ({platform.machine()})")
print(f"Torch {torch.__version__}, CUDA runtime {torch.version.cuda}")
print("工程导入通过；GPU、NCCL、compile 和 CUDA Graph 仍需计算节点验证。")
PY
printf '激活环境: source %s/miniforge3/bin/activate %s\n' "$HOME" "$DISTILL_ENV_PREFIX"
