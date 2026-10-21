#!/usr/bin/env bash
# 顺序跑 4 个 on-policy 训练(ELF-L off-policy 已完成,best 就绪,无需等待)。
# 每个 run 用 lr=1e-3 off-policy half-student 作起点(已写进 configs)。
# 每个 run 结束后剪掉其中间 step_* 检查点(保留 best/ + 最高 step),防止撑满 200G 配额。
set -uo pipefail
cd /proj/gpu_mtk53742/CDLM-distillation
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="run/launch/onpolicy_run4_$TS"
mkdir -p "$LOGDIR"
MAIN="$LOGDIR/run4.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

prune_steps(){
  find "$1" -type d -name checkpoints 2>/dev/null | while read -r ck; do
    local final; final=$(ls -d "$ck"/step_* 2>/dev/null | sort | tail -1)
    for s in "$ck"/step_*; do [ -d "$s" ] && [ "$s" != "$final" ] && rm -rf "$s"; done
  done
}

run_onpolicy(){ # $1 tag(=config run.name 的 on_policy_ 后缀)  $2 config
  local tag=$1 cfg=$2
  log "START on-policy $tag"
  bash scripts/run_on_policy.sh "configs/$cfg" all > "$LOGDIR/$tag.log" 2>&1
  log "DONE  on-policy $tag exit=$?"
  local d; d=$(ls -dt run/2026*_on_policy_$tag 2>/dev/null | head -1)
  [ -n "$d" ] && { prune_steps "$d"; log "pruned $d; quota=$(quota -s 2>/dev/null | awk 'NR==3{print $1}')"; }
}

run_onpolicy langflow_owt  on_policy_langflow_owt.yaml
run_onpolicy langflow_lm1b on_policy_langflow_lm1b.yaml
run_onpolicy elf_b         on_policy_elf_b.yaml
run_onpolicy elf_l         on_policy_elf_l.yaml
log "ALL 4 ON-POLICY DONE. logs in $LOGDIR"
