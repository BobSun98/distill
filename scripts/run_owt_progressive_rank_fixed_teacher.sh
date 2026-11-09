#!/usr/bin/env bash
# 新实验入口：SVD 起点逐级推进，KL Teacher 始终是原始满秩 LangFlow-OWT。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MASTER_PORT=${MASTER_PORT:-29551}
# 约束恢复模式，防止 --run-dir 指向旧的“上一阶学生当 Teacher”实验。
exec bash "$PROJECT_ROOT/scripts/run_owt_progressive_rank.sh" --teacher-mode fixed_original "$@"
