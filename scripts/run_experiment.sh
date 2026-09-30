#!/usr/bin/env bash
# 用法: bash scripts/run_experiment.sh configs/owt_debug.yaml debug
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-/proj/gpu_mtk53742/.conda/envs/distill/bin/python}
CONFIG_FILE=${1:-configs/owt_kd.yaml}
COMMAND=${2:-all}
if (( $# >= 2 )); then shift 2; elif (( $# == 1 )); then shift; fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

if [[ "$COMMAND" == debug ]]; then
    # debug 只暴露用户选择的第一张卡，并直接启动一个 Python 进程。
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
    "$PY" "$PROJECT_ROOT/debug/debug_pipeline.py" --config "$CONFIG_FILE" "$@"
elif [[ "$COMMAND" == prepare ]]; then
    "$PY" -m distill "$COMMAND" --config "$CONFIG_FILE" "$@"
else
    NGPU=${NGPU:-8}
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    [[ "$NGPU" =~ ^[1-9][0-9]*$ ]] || { echo 'NGPU 必须是正整数' >&2; exit 1; }
    IFS=',' read -r -a GPU_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
    (( ${#GPU_DEVICES[@]} >= NGPU )) || { echo '可见 GPU 少于 NGPU；单卡运行请设置 NGPU=1' >&2; exit 1; }
    if [[ "$NGPU" == 1 ]]; then
        "$PY" -m distill "$COMMAND" --config "$CONFIG_FILE" "$@"
    else
        "$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node="$NGPU" \
            --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29500}" \
            -m distill "$COMMAND" --config "$CONFIG_FILE" "$@"
    fi
fi
