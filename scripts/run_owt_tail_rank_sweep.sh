#!/usr/bin/env bash
# 八卡：40%/60% rank × 尾部 2/4/6/8/10 层，沿用统一评测和恢复逻辑。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DISTILL_RANK_SWEEP_CONFIG=configs/owt_tail_rank_sweep.yaml
export DISTILL_RANK_SWEEP_DEBUG_CONFIG=configs/owt_tail_rank_sweep_debug.yaml
export DISTILL_DEBUG_ENTRY=debug/debug_tail_rank_sweep.py
exec bash "$PROJECT_ROOT/scripts/run_owt_rank_sweep.sh" "$@"
