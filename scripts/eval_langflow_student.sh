#!/usr/bin/env bash
# 用法: bash scripts/eval_langflow_student.sh /path/to/student/model.safetensors
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-/proj/gpu_mtk53742/.conda/envs/distill/bin/python}
CHECKPOINT=${1:?请传入 student safetensors 文件或 checkpoint 目录}
shift
NGPU=${NGPU:-8}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
[[ "$NGPU" =~ ^[1-9][0-9]*$ ]] || { echo 'NGPU 必须是正整数' >&2; exit 1; }
IFS=',' read -r -a GPU_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
(( ${#GPU_DEVICES[@]} >= NGPU )) || { echo '可见 GPU 少于 NGPU' >&2; exit 1; }
if [[ "$(uname -s)" == Linux ]]; then export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"; fi
cd "$PROJECT_ROOT"
mkdir -p run/launch
LAUNCH_LOG="$PROJECT_ROOT/run/launch/$(date +%Y%m%d_%H%M%S)_student_eval_$$.log"
printf '[launcher] GPUs: %s; log: %s\n' "$NGPU" "$LAUNCH_LOG"

# 每卡本地计算 NLL/生成；路径、缓存与最终统计用 CPU/Gloo 汇总，无 NCCL busy-wait。
if [[ "$NGPU" == 1 ]]; then
    "$PY" -m distill.student_eval --checkpoint "$CHECKPOINT" "$@" 2>&1 | tee "$LAUNCH_LOG"
else
    "$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node="$NGPU" \
        --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29500}" \
        -m distill.student_eval --checkpoint "$CHECKPOINT" "$@" 2>&1 | tee "$LAUNCH_LOG"
fi
