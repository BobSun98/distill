#!/usr/bin/env bash
# 默认八卡运行五组独立的 12→6 层 LangFlow-OWT 学习率实验。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DISTILL_MODULE=distill.lr_sweep
export DISTILL_DEBUG_ENTRY=debug/debug_lr_sweep.py
if (( $# == 0 )); then set -- configs/owt_lr_sweep.yaml all; fi
exec bash "$PROJECT_ROOT/scripts/run_experiment.sh" "$@"
