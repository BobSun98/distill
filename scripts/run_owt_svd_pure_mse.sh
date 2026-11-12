#!/usr/bin/env bash
# 八卡：Teacher共享SVD起点，40%/60% × pure embedding/velocity MSE。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=configs/owt_svd_pure_mse.yaml
COMMAND=all
DRY_RUN=0
COMPILE_DEBUG=0
ARGS=()
while (( $# )); do
    case "$1" in
        all|prepare|build) COMMAND="$1"; shift ;;
        debug|compile-debug)
            [[ "$1" == compile-debug ]] && COMPILE_DEBUG=1
            COMMAND=debug; CONFIG=configs/owt_svd_pure_mse_debug.yaml; shift ;;
        --config)
            [[ $# -ge 2 ]] || { echo '--config 需要路径' >&2; exit 2; }
            CONFIG="$2"; shift 2 ;;
        --set|--rank|--objective|--run-dir)
            [[ $# -ge 2 ]] || { echo "$1 需要参数" >&2; exit 2; }
            ARGS+=("$1" "$2"); shift 2 ;;
        --dry-run) DRY_RUN=1; ARGS+=(--dry-run); shift ;;
        --help|-h) DRY_RUN=1; ARGS+=(--help); shift ;;
        *) echo '用法: bash scripts/run_owt_svd_pure_mse.sh [all|prepare|build|debug|compile-debug] [--rank 40|60] [--objective embedding|velocity] [--dry-run] [--run-dir DIR] [--set KEY=VALUE]' >&2; exit 2 ;;
    esac
done
if (( DRY_RUN )) && [[ "$COMMAND" != debug ]]; then COMMAND=prepare; fi
if (( COMPILE_DEBUG )); then
    ARGS=(--set execution.backend=compile --set execution.debug_compile=true ${ARGS[@]+"${ARGS[@]}"})
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NGPU=8
export MASTER_PORT=${MASTER_PORT:-29543}
export DISTILL_MODULE=distill.svd_mse
export DISTILL_DEBUG_ENTRY=debug/debug_svd_mse.py
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export TORCH_LOGS=${TORCH_LOGS:-perf_hints}
exec bash "$PROJECT_ROOT/scripts/run_experiment.sh" "$CONFIG" "$COMMAND" ${ARGS[@]+"${ARGS[@]}"}
