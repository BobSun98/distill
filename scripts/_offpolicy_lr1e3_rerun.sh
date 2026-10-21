#!/usr/bin/env bash
# 临时：用 lr=1e-3 重跑 lm1b / elf-b / elf-l 的 off-policy(stage-1) half-student 蒸馏。
# 三个 8 卡任务顺序执行；某个失败也继续后面的，最后汇总退出码。
set -uo pipefail
cd /proj/gpu_mtk53742/CDLM-distillation
LR=0.001
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="run/launch/offpolicy_lr1e3_$TS"
mkdir -p "$LOGDIR"
declare -A RC

run_job() {
  local tag=$1 script=$2 config=$3 name=$4
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] START $tag  ($config, lr=$LR, name=$name)"
  echo "=================================================================="
  bash "scripts/$script" "configs/$config" all \
      --set training.learning_rate=$LR \
      --set run.name="$name" \
      > "$LOGDIR/$tag.log" 2>&1
  RC[$tag]=$?
  echo "[$(date +%H:%M:%S)] DONE  $tag  exit=${RC[$tag]}  log=$LOGDIR/$tag.log"
}

run_job lm1b  run_experiment.sh     lm1b_kd.yaml   lm1b_kd_12to6_lr1e-3
run_job elf_b run_elf_experiment.sh elf_b_kd.yaml  elf_b_owt_kd_12to6_lr1e-3
run_job elf_l run_elf_experiment.sh elf_l_kd.yaml  elf_l_owt_kd_32to16_lr1e-3

echo "=================================================================="
echo "ALL DONE. exit codes:"
for k in lm1b elf_b elf_l; do echo "  $k = ${RC[$k]:-?}"; done
echo "logs in $LOGDIR"
