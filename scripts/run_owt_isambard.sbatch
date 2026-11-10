#!/usr/bin/env bash
#SBATCH --job-name=distill-owt
#SBATCH --account=brics.u6xs
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/u6xs/bobsun.u6xs/distill/run/launch/slurm_%j.log
# 单个 srun task 启动 torchrun，后者负责本节点的四个 worker。
set -euo pipefail
PROJECT_ROOT=${SLURM_SUBMIT_DIR:?请从项目根目录通过 sbatch 提交}
DISTILL_STORAGE_ROOT=/projects/u6xs/bobsun.u6xs/distill
export PY=${PY:-$DISTILL_STORAGE_ROOT/envs/distill/bin/python}
export DISTILL_LAUNCH_LOG_DIR=$DISTILL_STORAGE_ROOT/run/launch
export HF_HOME=$DISTILL_STORAGE_ROOT/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCHINDUCTOR_CACHE_DIR=$DISTILL_STORAGE_ROOT/cache/torchinductor
export TRITON_CACHE_DIR=$DISTILL_STORAGE_ROOT/cache/triton
export MPLCONFIGDIR=$DISTILL_STORAGE_ROOT/cache/matplotlib
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCH_LOGS=${TORCH_LOGS:-perf_hints}
# 登录节点的 /local 临时目录在计算节点上可能不存在，必须在启动 Python 前创建共享目录。
export TMPDIR=$DISTILL_STORAGE_ROOT/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
printf '[runtime] job=%s TMPDIR=%s\n' "$SLURM_JOB_ID" "$TMPDIR"
# 旧服务器 A6000 的 P2P 禁用设置不适用于 GH200，恢复 NCCL P2P。
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export NGPU=4
cd "$PROJECT_ROOT"
module load cuda/12.6 gcc-native/12.3
export CC=$(command -v gcc)
export CXX=$(command -v g++)
mkdir -p "$DISTILL_LAUNCH_LOG_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"

MODE=${1:-progressive}
if (( $# )); then shift; fi
case "$MODE" in
    prepare)
        srun --ntasks=1 bash scripts/run_owt_rank_sweep.sh configs/owt_rank080_isambard.yaml prepare "$@"
        ;;
    compile-debug)
        srun --ntasks=1 --gpu-bind=none bash scripts/run_owt_rank_sweep.sh configs/owt_rank080_isambard_debug.yaml debug "$@"
        ;;
    compile-debug-ddp)
        export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
        srun --ntasks=1 --gpu-bind=none "$PY" -m torch.distributed.run \
            --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
            debug/debug_rank_sweep_distributed.py --config configs/owt_rank080_isambard_ddp_debug.yaml "$@"
        ;;
    source80)
        SOURCE_ARGS=()
        SOURCE_RESUME=0
        for argument in "$@"; do
            if [[ "$argument" == --run-dir ]]; then SOURCE_RESUME=1; fi
        done
        # 恢复使用已保存的run.root，不能再传--set覆盖实验条件。
        if [[ -n "${DISTILL_SOURCE_RUN_ROOT:-}" ]] && (( ! SOURCE_RESUME )); then
            SOURCE_ARGS+=(--set "run.root=$DISTILL_SOURCE_RUN_ROOT")
        fi
        srun --ntasks=1 --gpu-bind=none bash scripts/run_owt_rank_sweep.sh configs/owt_rank080_isambard.yaml all "${SOURCE_ARGS[@]}" "$@"
        if [[ -n "${DISTILL_SOURCE_POINTER:-}" ]]; then
            # 依赖链使用本轮独立目录，发布训练成功的 best 后才放行逐级作业。
            "$PY" - "${DISTILL_SOURCE_RUN_ROOT:?需要本轮80%输出目录}" "$DISTILL_SOURCE_POINTER" <<'PY'
import json
import sys
from pathlib import Path

root, pointer = map(Path, sys.argv[1:])
runs = list(root.glob('*_langflow_owt_rank080_isambard'))
if len(runs) != 1:
    raise RuntimeError(f'本轮80%实验目录应有且仅有一个，实际为 {runs}')
run = runs[0]
status = json.loads((run / 'status.json').read_text())
training = json.loads((run / 'rank_080/training.json').read_text())
checkpoint = run / 'rank_080/checkpoints/best'
if status['status'] != 'completed' or not (checkpoint / 'model.safetensors').is_file():
    raise RuntimeError('80%训练/评测未完整成功，不能作为逐级起点')
pointer.parent.mkdir(parents=True, exist_ok=True)
if pointer.is_symlink():
    if pointer.resolve() != checkpoint.resolve():
        raise RuntimeError(f'起点指针已经指向另一套权重: {pointer}')
elif pointer.exists():
    raise RuntimeError(f'起点指针位置已有文件: {pointer}')
else:
    pointer.symlink_to(checkpoint)
receipt = {'run_dir':str(run), 'checkpoint':str(checkpoint), 'pointer':str(pointer),
           'best_step':training.get('best_step'), 'best_value':training.get('best_value')}
pointer.with_suffix('.json').write_text(json.dumps(receipt, indent=2))
print('[source80] ' + json.dumps(receipt), flush=True)
PY
        fi
        ;;
    progressive)
        srun --ntasks=1 --gpu-bind=none bash scripts/run_owt_progressive_rank_fixed_teacher.sh --config configs/owt_progressive_rank_isambard.yaml "$@"
        ;;
    *) echo '用法: sbatch scripts/run_owt_isambard.sbatch [prepare|compile-debug|compile-debug-ddp|source80|progressive] [原入口参数]' >&2; exit 2 ;;
esac
