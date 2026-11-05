#!/usr/bin/env bash
# 默认八卡：20% 五个尾部学生最多 10000 步，结果并入已有 40%/60% 实验。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ "${1:-}" == compile-debug ]]; then
    shift
    exec bash "$PROJECT_ROOT/scripts/verify_compile_cuda_graph.sh" "$@"
fi
if [[ "${1:-}" == debug ]]; then
    shift
    # debug 新建短预算目录，同进程单卡调用真实工程，不改动正式实验。
    exec bash "$PROJECT_ROOT/scripts/run_owt_tail_rank_sweep.sh" debug --set 'rank_sweep.ratios=[0.2]' "$@"
fi
RUN_DIR=run/20261101_075912_158102_langflow_owt_svd_tail_rank_sweep
exec bash "$PROJECT_ROOT/scripts/run_owt_tail_rank_sweep.sh" --run-dir "$RUN_DIR" --add-rank 0.2 --max-steps 10000 --backend compile "$@"
