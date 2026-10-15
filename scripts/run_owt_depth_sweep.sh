#!/usr/bin/env bash
# 新实验入口，不改变既有 LangFlow/ELF 启动脚本及配置。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DISTILL_MODULE=distill.depth_sweep
export DISTILL_DEBUG_ENTRY=debug/debug_depth_sweep.py
if (( $# == 0 )); then set -- configs/owt_depth_sweep.yaml all; fi
exec bash "$PROJECT_ROOT/scripts/run_experiment.sh" "$@"
