#!/usr/bin/env bash
# 默认仅重跑 ELF-B/L，从原 off-policy 权重重新训练；all 沿用统一 Gen PPL 评测。
# 用法: bash scripts/rerun_on_policy_loss_matched.sh [all|train] [--dry-run] [--all-models]
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-/proj/gpu_mtk53742/.conda/envs/distill/bin/python}
COMMAND=all
DRY_RUN=0
ALL_MODELS=0
for arg in "$@"; do
    case "$arg" in
        all|train) COMMAND="$arg" ;;
        --dry-run) DRY_RUN=1 ;;
        --all-models) ALL_MODELS=1 ;;
        *) echo '用法: bash scripts/rerun_on_policy_loss_matched.sh [all|train] [--dry-run] [--all-models]' >&2; exit 2 ;;
    esac
done
export NGPU="${NGPU:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"
PLAN_DIR="$PROJECT_ROOT/run/launch/on_policy_loss_matched_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$PLAN_DIR"
PREPARE_ARGS=(--output-dir "$PLAN_DIR")
MODELS=(elf_b elf_l)
if (( ALL_MODELS )); then
    PREPARE_ARGS+=(--all-models)
    MODELS=(langflow_owt langflow_lm1b elf_b elf_l)
fi
if (( DRY_RUN )); then PREPARE_ARGS+=(--dry-run); fi
"$PY" -m distill.on_policy.rerun "${PREPARE_ARGS[@]}" 2>&1 | tee "$PLAN_DIR/prepare.log"
printf '[loss-matched] plan: %s/plan.json\n' "$PLAN_DIR"
if (( DRY_RUN )); then exit 0; fi

printf 'model\texit_code\tlog\n' > "$PLAN_DIR/status.tsv"
FAILED=0
for MODEL in "${MODELS[@]}"; do
    printf '[loss-matched] start %s (%s GPUs, command=%s)\n' "$MODEL" "$NGPU" "$COMMAND"
    if bash scripts/run_on_policy.sh "$PLAN_DIR/$MODEL.yaml" "$COMMAND" 2>&1 | tee "$PLAN_DIR/$MODEL.log"; then
        RESULT=0
    else
        RESULT=$?
        FAILED=1
    fi
    printf '%s\t%s\t%s\n' "$MODEL" "$RESULT" "$PLAN_DIR/$MODEL.log" >> "$PLAN_DIR/status.tsv"
done
printf '[loss-matched] finished; status: %s/status.tsv\n' "$PLAN_DIR"
exit "$FAILED"
