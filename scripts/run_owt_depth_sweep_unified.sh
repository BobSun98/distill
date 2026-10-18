#!/usr/bin/env bash
# 新的 6→11 层 sweep：默认八卡，统一评测六个 student 与原始 teacher。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DISTILL_MODULE=distill.depth_sweep
export DISTILL_DEBUG_ENTRY=debug/debug_depth_sweep_unified.py
if (( $# == 0 )); then set -- configs/owt_depth_sweep_unified.yaml all; fi
exec bash "$PROJECT_ROOT/scripts/run_experiment.sh" "$@"
