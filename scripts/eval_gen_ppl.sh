#!/usr/bin/env bash
# 默认八卡；正式 Gen PPL 固定 42 / 1024 / GPT-2-Large。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DISTILL_MODULE=distill.gen_ppl
export DISTILL_DEBUG_ENTRY=debug/debug_gen_ppl.py
if (( $# == 0 )); then set -- configs/gen_ppl.yaml all; fi
exec bash "$PROJECT_ROOT/scripts/run_experiment.sh" "$@"
