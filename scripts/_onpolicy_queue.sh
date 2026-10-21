#!/usr/bin/env bash
# 排队：等当前 ELF-L(off-policy lr1e-3)进程结束后，顺序启动 4 个 on-policy 训练。
# 每个 on-policy run 用对应 lr1e-3 off-policy half-student 作为起点(已写进 configs)。
# 为避免再次撑满 200G 配额：ELF-L 结束后、以及每个 on-policy run 结束后，
# 都剪掉该 run 的中间 step_* 检查点(保留 best/ 和最高 step)。
set -uo pipefail
cd /proj/gpu_mtk53742/CDLM-distillation
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="run/launch/onpolicy_queue_$TS"
mkdir -p "$LOGDIR"
MAIN="$LOGDIR/queue.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

# 剪枝：某个 run 目录下所有 checkpoints/*，每个保留 best/ 和最高 step_*。
prune_steps(){ # $1 = run dir
  find "$1" -type d -name checkpoints 2>/dev/null | while read -r ck; do
    local final; final=$(ls -d "$ck"/step_* 2>/dev/null | sort | tail -1)
    for s in "$ck"/step_*; do [ -d "$s" ] && [ "$s" != "$final" ] && rm -rf "$s"; done
  done
}

run_onpolicy(){ # $1 tag  $2 config
  local tag=$1 cfg=$2
  log "START on-policy $tag ($cfg)"
  bash scripts/run_on_policy.sh "configs/$cfg" all > "$LOGDIR/$tag.log" 2>&1
  local rc=$?
  log "DONE  on-policy $tag exit=$rc"
  # 剪枝该 run 的中间检查点
  local d; d=$(ls -dt run/2026*_on_policy_$tag 2>/dev/null | head -1)
  [ -n "$d" ] && { prune_steps "$d"; log "pruned step_* in $d"; }
  log "quota: $(quota -s 2>/dev/null | awk 'NR==3{print $1}')"
}

# ---- 1) 等 ELF-L off-policy 结束 ----
log "waiting for ELF-L off-policy (configs/elf_l_kd.yaml) to finish ..."
while pgrep -f "distill.elf.pipeline all --config configs/elf_l_kd.yaml" >/dev/null 2>&1; do sleep 60; done
ELFL_DIR=$(ls -dt run/2026*_elf_l_owt_kd_32to16_lr1e-3 2>/dev/null | head -1)
log "ELF-L process gone. run dir=$ELFL_DIR status=$(cat "$ELFL_DIR/status.json" 2>/dev/null | tr -d '\n ')"
prune_steps "$ELFL_DIR"; log "pruned ELF-L step_*; quota: $(quota -s 2>/dev/null | awk 'NR==3{print $1}')"

# ---- 2) 顺序跑 4 个 on-policy（run.name 前缀见 config：on_policy_<tag>）----
run_onpolicy langflow_owt  on_policy_langflow_owt.yaml
run_onpolicy langflow_lm1b on_policy_langflow_lm1b.yaml
run_onpolicy elf_b         on_policy_elf_b.yaml
# elf_l 需要 ELF-L 的 best；若缺失则跳过
if [ -f "$ELFL_DIR/checkpoints/best/model.safetensors" ]; then
  run_onpolicy elf_l on_policy_elf_l.yaml
else
  log "SKIP on-policy elf_l: ELF-L best 缺失 ($ELFL_DIR/checkpoints/best)"
fi

log "ALL ON-POLICY QUEUE DONE. logs in $LOGDIR"
