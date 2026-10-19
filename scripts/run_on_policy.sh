#!/usr/bin/env bash
# 用法: bash scripts/run_on_policy.sh configs/on_policy_langflow_owt.yaml all
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DISTILL_MODULE=distill.on_policy.pipeline
export DISTILL_DEBUG_ENTRY=debug/debug_on_policy.py
if (( $# == 0 )); then set -- configs/on_policy_langflow_owt.yaml all; fi
exec bash "$PROJECT_ROOT/scripts/run_experiment.sh" "$@"
