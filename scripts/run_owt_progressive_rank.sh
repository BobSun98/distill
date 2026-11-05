#!/usr/bin/env bash
# 八卡：从已训练的 80% 学生逐级截断到 20%，每一级先统一评测、KL 蒸馏、再统一评测。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=configs/owt_progressive_rank.yaml
COMMAND=all
DRY_RUN=0
COMPILE_DEBUG=0
ARGS=()
while (( $# )); do
    case "$1" in
        all|prepare) COMMAND="$1"; shift ;;
        debug|compile-debug)
            [[ "$1" == compile-debug ]] && COMPILE_DEBUG=1
            COMMAND=debug; CONFIG=configs/owt_progressive_rank_debug.yaml; shift ;;
        --config)
            [[ $# -ge 2 ]] || { echo '--config 需要路径' >&2; exit 2; }
            CONFIG="$2"; shift 2 ;;
        *.yaml|*.yml) CONFIG="$1"; shift ;;
        --run-dir|--set)
            [[ $# -ge 2 ]] || { echo "$1 需要参数" >&2; exit 2; }
            ARGS+=("$1" "$2"); shift 2 ;;
        --dry-run) DRY_RUN=1; ARGS+=(--dry-run); shift ;;
        --help|-h) DRY_RUN=1; ARGS+=(--help); shift ;;
        *) echo '用法: bash scripts/run_owt_progressive_rank.sh [all|prepare|debug|compile-debug] [--dry-run] [--run-dir DIR] [--set KEY=VALUE] [--config CONFIG.yaml]' >&2; exit 2 ;;
    esac
done
if (( DRY_RUN )) && [[ "$COMMAND" != debug ]]; then COMMAND=prepare; fi
if (( COMPILE_DEBUG )); then
    ARGS=(--set execution.backend=compile --set execution.debug_compile=true ${ARGS[@]+"${ARGS[@]}"})
fi
export DISTILL_MODULE=distill.progressive_rank
export DISTILL_DEBUG_ENTRY=debug/debug_progressive_rank.py
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export TORCH_LOGS=${TORCH_LOGS:-perf_hints}
exec bash "$PROJECT_ROOT/scripts/run_experiment.sh" "$CONFIG" "$COMMAND" ${ARGS[@]+"${ARGS[@]}"}
