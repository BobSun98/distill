#!/usr/bin/env bash
# 用法: bash scripts/run_experiment.sh configs/owt_debug.yaml debug
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-/proj/gpu_mtk53742/.conda/envs/distill/bin/python}
CONFIG_FILE=${1:-configs/owt_kd.yaml}
COMMAND=${2:-all}
if (( $# >= 2 )); then shift 2; elif (( $# == 1 )); then shift; fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

if [[ "$COMMAND" == debug ]]; then
    "$PY" "$PROJECT_ROOT/debug/debug_pipeline.py" --config "$CONFIG_FILE" "$@"
else
    "$PY" -m distill "$COMMAND" --config "$CONFIG_FILE" "$@"
fi
