#!/usr/bin/env bash
# 与 LangFlow 共用八卡资源、端口、P2P 与启动日志设置；debug 仍只启动一个 Python。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DISTILL_MODULE=distill.elf.pipeline
export DISTILL_DEBUG_ENTRY=debug/debug_elf_pipeline.py
if (( $# == 0 )); then set -- configs/elf_b_kd.yaml all; fi
exec bash "$PROJECT_ROOT/scripts/run_experiment.sh" "$@"
