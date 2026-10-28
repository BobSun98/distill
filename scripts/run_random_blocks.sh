#!/usr/bin/env bash
# 四组半层学生只随机初始化 Transformer blocks；默认八卡，训后统一计算 PPL/entropy/GM。
# 用法: bash scripts/run_random_blocks.sh [all|train|debug] [--elf-only|--model MODEL] [--dry-run] [--set KEY=VALUE]
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-/proj/gpu_mtk53742/.conda/envs/distill/bin/python}
COMMAND=all
DRY_RUN=0
MODELS=(langflow_owt langflow_lm1b elf_b elf_l)
OVERRIDES=()
while (( $# )); do
    case "$1" in
        all|train|debug) COMMAND="$1"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --elf-only) MODELS=(elf_b elf_l); shift ;;
        --model)
            case "${2:-}" in
                langflow_owt|langflow_lm1b|elf_b|elf_l) MODELS=("$2"); shift 2 ;;
                *) echo '--model 支持 langflow_owt / langflow_lm1b / elf_b / elf_l' >&2; exit 2 ;;
            esac ;;
        --set)
            [[ $# -ge 2 ]] || { echo '--set 需要 KEY=VALUE' >&2; exit 2; }
            OVERRIDES+=(--set "$2"); shift 2 ;;
        *) echo '用法: bash scripts/run_random_blocks.sh [all|train|debug] [--elf-only|--model MODEL] [--dry-run] [--set KEY=VALUE]' >&2; exit 2 ;;
    esac
done
export NGPU="${NGPU:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"
LOG_DIR="$PROJECT_ROOT/run/launch/random_blocks_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOG_DIR"

# 在任何 GPU 任务前仅加载配置，记录 loss、训练预算、统一协议和 GM 参考语料。
SUFFIX=""
PLAN_COMMAND=("$PY" -m distill.off_policy)
if [[ "$COMMAND" == debug ]]; then
    SUFFIX=_debug
    PLAN_COMMAND=("$PY" debug/debug_random_blocks.py)
fi
for MODEL in "${MODELS[@]}"; do
    CONFIG="configs/off_policy_random_${MODEL}${SUFFIX}.yaml"
    "${PLAN_COMMAND[@]}" --config "$CONFIG" --dry-run ${OVERRIDES[@]+"${OVERRIDES[@]}"} > "$LOG_DIR/$MODEL.plan.json"
    printf '[random-blocks] %s: %s\n' "$MODEL" "$LOG_DIR/$MODEL.plan.json"
done
if (( DRY_RUN )); then exit 0; fi

printf 'model\texit_code\tlog\n' > "$LOG_DIR/status.tsv"
for MODEL in "${MODELS[@]}"; do
    CONFIG="configs/off_policy_random_${MODEL}${SUFFIX}.yaml"
    printf '[random-blocks] start %s (command=%s)\n' "$MODEL" "$COMMAND"
    if DISTILL_MODULE=distill.off_policy DISTILL_DEBUG_ENTRY=debug/debug_random_blocks.py \
        bash scripts/run_experiment.sh "$CONFIG" "$COMMAND" ${OVERRIDES[@]+"${OVERRIDES[@]}"} 2>&1 | tee "$LOG_DIR/$MODEL.log"; then
        printf '%s\t0\t%s\n' "$MODEL" "$LOG_DIR/$MODEL.log" >> "$LOG_DIR/status.tsv"
    else
        RESULT=$?
        printf '%s\t%s\t%s\n' "$MODEL" "$RESULT" "$LOG_DIR/$MODEL.log" >> "$LOG_DIR/status.tsv"
        # 出现真实训练/评测错误即停止队列，保留现场，避免后续实验掩盖问题。
        printf '[random-blocks] failed; log: %s/%s.log\n' "$LOG_DIR" "$MODEL" >&2
        exit "$RESULT"
    fi
done
printf '[random-blocks] finished; status: %s/status.tsv\n' "$LOG_DIR"
