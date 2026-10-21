#!/usr/bin/env bash
# 临时：给 on-policy 测 per-device batch size。每张卡跑一个单卡进程、各测一档，6 步即停。
# 用法: _bs_probe.sh <tag> <config.yaml> <key> "<v1 v2 ... v8>" [extra --set ...]
# key 例: training.batch_size 或 on_policy.rollout_batch_size
set -uo pipefail
cd /proj/gpu_mtk53742/CDLM-distillation
TAG=$1; CFG=$2; KEY=$3; VALUES=($4); shift 4
OUT=/tmp/onp_bs_test/logs; mkdir -p "$OUT"
i=0
for V in "${VALUES[@]}"; do
  CUDA_VISIBLE_DEVICES=$i NGPU=1 timeout 1500 bash scripts/run_on_policy.sh "configs/$CFG" train \
    --set "$KEY=$V" \
    --set training.max_steps=6 --set training.log_every=1 --set training.grad_accumulation=1 \
    --set training.validation_batches=1 --set training.validate_every=1000 --set training.save_every=0 \
    --set training.early_stopping.enabled=false --set training.ppl_evaluation.enabled=false \
    --set training.plot_loss=false --set run.root=/tmp/onp_bs_test --set "run.name=probe_${TAG}_${V}" \
    "$@" > "$OUT/${TAG}_${KEY##*.}_${V}.log" 2>&1 &
  i=$((i+1))
done
wait
echo "round done: $TAG $KEY = ${VALUES[*]}"
