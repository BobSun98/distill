#!/usr/bin/env bash
# 本次仅用物理 GPU 4/5/6/7；原始 LangFlow Teacher + 已训全层 40% student。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=configs/owt_rank040_velocity.yaml
COMMAND=all
DRY_RUN=0
COMPILE_DEBUG=0
ARGS=()
while (( $# )); do
    case "$1" in
        all|train|prepare|evaluate) COMMAND="$1"; shift ;;
        debug|compile-debug)
            [[ "$1" == compile-debug ]] && COMPILE_DEBUG=1
            COMMAND=debug; CONFIG=configs/owt_rank040_velocity_debug.yaml; shift ;;
        --config|--set|--checkpoint)
            [[ $# -ge 2 ]] || { echo "$1 需要参数" >&2; exit 2; }
            if [[ "$1" == --config ]]; then CONFIG="$2";
            else ARGS+=("$1" "$2"); fi
            shift 2 ;;
        --dry-run) DRY_RUN=1; ARGS+=(--dry-run); shift ;;
        --help|-h) DRY_RUN=1; ARGS+=(--help); shift ;;
        *) echo '用法: bash scripts/run_owt_rank040_velocity.sh [all|train|prepare|evaluate|debug|compile-debug] [--dry-run] [--set KEY=VALUE] [--checkpoint PATH] [--config CONFIG.yaml]' >&2; exit 2 ;;
    esac
done
if (( DRY_RUN )) && [[ "$COMMAND" != debug ]]; then COMMAND=prepare; fi
if (( COMPILE_DEBUG )); then
    ARGS=(--set execution.backend=compile --set execution.debug_compile=true ${ARGS[@]+"${ARGS[@]}"})
fi
# 固定本次授权的四张卡；debug 启动器只保留其中第一张，即物理 GPU 4。
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NGPU=4
export MASTER_PORT=${MASTER_PORT:-29541}
export DISTILL_MODULE=distill.off_policy
export DISTILL_DEBUG_ENTRY=debug/debug_rank040_velocity.py
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export TORCH_LOGS=${TORCH_LOGS:-perf_hints}
exec bash "$PROJECT_ROOT/scripts/run_experiment.sh" "$CONFIG" "$COMMAND" ${ARGS[@]+"${ARGS[@]}"}
