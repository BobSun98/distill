#!/usr/bin/env bash
# 默认八卡：七个 12 层 SVD 学生，统一初评、KL 蒸馏、统一复评。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=""
COMMAND=all
DRY_RUN=0
ARGS=()
while (( $# )); do
    case "$1" in
        all|prepare|build|debug) COMMAND="$1"; shift ;;
        --config)
            [[ $# -ge 2 ]] || { echo '--config 需要路径' >&2; exit 2; }
            CONFIG="$2"; shift 2 ;;
        *.yaml|*.yml) CONFIG="$1"; shift ;;
        --set|--run-dir)
            [[ $# -ge 2 ]] || { echo "$1 需要参数" >&2; exit 2; }
            ARGS+=("$1" "$2"); shift 2 ;;
        --dry-run) DRY_RUN=1; ARGS+=("$1"); shift ;;
        --help|-h) ARGS+=(--help); shift ;;
        *) echo '用法: bash scripts/run_owt_rank_sweep.sh [CONFIG.yaml] [all|prepare|build|debug] [--dry-run] [--run-dir DIR] [--set KEY=VALUE]' >&2; exit 2 ;;
    esac
done
if [[ -z "$CONFIG" ]]; then
    if [[ "$COMMAND" == debug ]]; then CONFIG=configs/owt_rank_sweep_debug.yaml;
    else CONFIG=configs/owt_rank_sweep.yaml; fi
fi
# dry-run 只启动一个配置检查进程，不初始化 CUDA 或八卡通信。
if (( DRY_RUN )) && [[ "$COMMAND" != debug ]]; then COMMAND=prepare; fi
export DISTILL_MODULE=distill.rank_sweep
export DISTILL_DEBUG_ENTRY=debug/debug_rank_sweep.py
exec bash "$PROJECT_ROOT/scripts/run_experiment.sh" "$CONFIG" "$COMMAND" ${ARGS[@]+"${ARGS[@]}"}
