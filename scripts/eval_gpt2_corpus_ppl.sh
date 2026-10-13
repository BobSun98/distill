#!/usr/bin/env bash
# 默认八卡分别评分 LM1B/OWT 的 512 条真实文本，debug 使用单卡当前进程。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DISTILL_MODULE=distill.corpus_ppl
export DISTILL_DEBUG_ENTRY=debug/debug_corpus_ppl.py
if (( $# == 0 )); then set -- configs/gpt2_corpus_ppl.yaml all; fi
exec bash "$PROJECT_ROOT/scripts/run_experiment.sh" "$@"
