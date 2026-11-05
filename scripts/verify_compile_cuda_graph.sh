#!/usr/bin/env bash
# 单卡真实流程验证，日志和结果保存在 run；不接入正式 sweep 的恢复目录。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
用法: bash scripts/verify_compile_cuda_graph.sh [--dry-run] [--set KEY=VALUE]

默认使用 CUDA 0，验证真实 12 层 LangFlow Teacher、20% 尾部两层 SVD 学生、
3 步 KL 训练和统一生成/PPL/entropy/GM；检查热态 CUDA Graph replay 与 checkpoint 参数名。
每次新建 run/*_langflow_owt_compile_cuda_graph_debug/，完整日志在 run/launch/。

选择其他卡: CUDA_VISIBLE_DEVICES=3 bash scripts/verify_compile_cuda_graph.sh
只检查配置: bash scripts/verify_compile_cuda_graph.sh --dry-run
解释器覆盖: PY=/path/to/python bash scripts/verify_compile_cuda_graph.sh
HELP
    exit 0
fi

# 复用同进程 debug 入口和公共日志启动器，保留完整编译异常堆栈。
export DISTILL_DEBUG_ENTRY=debug/debug_compile_backend.py
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export TORCH_LOGS=${TORCH_LOGS:-perf_hints}
exec bash "$PROJECT_ROOT/scripts/run_experiment.sh" configs/owt_tail_rank_sweep_compile_debug.yaml debug "$@"
