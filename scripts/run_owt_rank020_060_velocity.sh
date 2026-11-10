#!/usr/bin/env bash
# 八卡 GPU 0-7：顺序补跑20%与60%的 KL+velocity 续训及统一评测。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COMMAND=all
DRY_RUN=0
RANKS=(020 060)
ARGS=()
while (( $# )); do
    case "$1" in
        all|train|prepare|debug|compile-debug) COMMAND="$1"; shift ;;
        --rank)
            case "${2:-}" in
                20|020) RANKS=(020) ;;
                60|060) RANKS=(060) ;;
                *) echo '--rank 只支持20或60' >&2; exit 2 ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --set)
            [[ $# -ge 2 ]] || { echo '--set 需要 KEY=VALUE' >&2; exit 2; }
            ARGS+=(--set "$2"); shift 2 ;;
        --help|-h)
            echo '用法: bash scripts/run_owt_rank020_060_velocity.sh [all|train|prepare|debug|compile-debug] [--rank 20|60] [--dry-run] [--set KEY=VALUE]'
            echo '默认顺序运行20%和60%，每组使用 GPU 0-7 八卡；debug单卡使用GPU 0。'
            exit 0 ;;
        *) echo "不支持的参数: $1；可用 --help 查看用法" >&2; exit 2 ;;
    esac
done
cd "$PROJECT_ROOT"
export DISTILL_VELOCITY_GPUS=0,1,2,3,4,5,6,7
LOG_DIR="run/launch/rank020_060_velocity_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOG_DIR"
SUFFIX=""
if [[ "$COMMAND" == debug || "$COMMAND" == compile-debug ]]; then SUFFIX=_debug; fi

# 先用真实工程的 dry-run 检查两份配置，避免第一组跑完后才发现第二组配置错误。
for RATIO in "${RANKS[@]}"; do
    CONFIG="configs/owt_rank${RATIO}_velocity${SUFFIX}.yaml"
    bash "$PROJECT_ROOT/scripts/run_owt_rank040_velocity.sh" "$COMMAND" --config "$CONFIG" \
        --dry-run ${ARGS[@]+"${ARGS[@]}"} > "$LOG_DIR/rank_${RATIO}.plan.log" 2>&1
    printf '[velocity] %s plan: %s/rank_%s.plan.log\n' "$RATIO" "$LOG_DIR" "$RATIO"
done
if (( DRY_RUN )); then exit 0; fi

printf 'rank\tconfig\texit_code\tlog\n' > "$LOG_DIR/status.tsv"
for RATIO in "${RANKS[@]}"; do
    CONFIG="configs/owt_rank${RATIO}_velocity${SUFFIX}.yaml"
    printf '[velocity] starting %s%% (command=%s, GPUs=0,1,2,3,4,5,6,7)\n' "$((10#$RATIO))" "$COMMAND"
    if bash "$PROJECT_ROOT/scripts/run_owt_rank040_velocity.sh" "$COMMAND" --config "$CONFIG" \
        ${ARGS[@]+"${ARGS[@]}"} 2>&1 | tee "$LOG_DIR/rank_${RATIO}.log"; then
        printf '%s\t%s\t0\t%s/rank_%s.log\n' "$RATIO" "$CONFIG" "$LOG_DIR" "$RATIO" >> "$LOG_DIR/status.tsv"
    else
        RESULT=$?
        printf '%s\t%s\t%s\t%s/rank_%s.log\n' "$RATIO" "$CONFIG" "$RESULT" "$LOG_DIR" "$RATIO" >> "$LOG_DIR/status.tsv"
        printf '[velocity] %s failed; records: %s\n' "$RATIO" "$LOG_DIR" >&2
        exit "$RESULT"
    fi
done
printf '[velocity] finished; records: %s/status.tsv\n' "$LOG_DIR"
