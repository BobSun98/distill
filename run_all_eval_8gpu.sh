#!/usr/bin/env bash
# Run five evaluations, one after another, using eight GPUs for each model run.
# From the repository root: bash run_all_eval_8gpu.sh
# TASKS=lm1b_gen,owt_ppl,lm1b_ppl,elf_b,elf_l selects a subset.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-/proj/gpu_mtk53742/.conda/envs/distill/bin/python}
NGPU=${NGPU:-8}
TASKS=${TASKS:-lm1b_gen,owt_ppl,lm1b_ppl,elf_b,elf_l}
OUTDIR=${OUTDIR:-"$ROOT/eval_runs/$(date +%Y%m%d_%H%M%S)"}

OWT_MODEL=${OWT_MODEL:-"$ROOT/LangFlow/checkpoints"}
LM1B_MODEL=${LM1B_MODEL:-Continuous-Rivals-Discrete/langflow-lm1b}
ELF_B_MODEL=${ELF_B_MODEL:-embedded-language-flows/ELF-B-owt-torch}
ELF_L_MODEL=${ELF_L_MODEL:-embedded-language-flows/ELF-L-owt-torch}
PPL_MODEL=${PPL_MODEL:-gpt2-large}

LM1B_GEN_TOTAL=${LM1B_GEN_TOTAL:-512}
LM1B_GEN_STEPS=${LM1B_GEN_STEPS:-128}
LM1B_GEN_LENGTH=${LM1B_GEN_LENGTH:-128}
LM1B_GEN_BS=${LM1B_GEN_BS:-8}
LM1B_SCORE_BS=${LM1B_SCORE_BS:-8}

FLOW_NFE=${FLOW_NFE:-128}
FLOW_ODE=${FLOW_ODE:-heun2}
FLOW_BS=${FLOW_BS:-1}                 # per GPU; flow NLL is memory intensive
FLOW_WORKERS=${FLOW_WORKERS:-2}
export LANGFLOW_DIST_TIMEOUT_MIN=${LANGFLOW_DIST_TIMEOUT_MIN:-120}
OWT_FIRST_N=${OWT_FIRST_N:-0}         # 0 = entire validation set
LM1B_FIRST_N=${LM1B_FIRST_N:-0}       # 0 = entire test set
export DATA_CACHE=${DATA_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/datasets}

ELF_SAMPLES=${ELF_SAMPLES:-1000}
ELF_BS=${ELF_BS:-1}                   # per GPU
ELF_SCORE_BS=${ELF_SCORE_BS:-8}       # scoring runs on rank 0
ELF_COMPILE=${ELF_COMPILE:-true}

die() { printf '[error] %s\n' "$*" >&2; exit 1; }
wants() { [[ ",$TASKS," == *",$1,"* ]]; }
run_logged() {
    local label=$1
    shift
    printf '[run] %s\n' "$label"
    "$@" 2>&1 | tee "$OUTDIR/$label.log"
}

[[ -x "$PY" ]] || die "Python is not executable: $PY"
[[ "$NGPU" =~ ^[1-9][0-9]*$ ]] || die "NGPU must be a positive integer"
IFS=',' read -r -a selected_tasks <<< "$TASKS"
(( ${#selected_tasks[@]} > 0 )) || die "TASKS cannot be empty"
for task in "${selected_tasks[@]}"; do
    case "$task" in
        lm1b_gen|owt_ppl|lm1b_ppl|elf_b|elf_l) ;;
        *) die "Unknown task: $task" ;;
    esac
done
for value in "$LM1B_GEN_TOTAL" "$FLOW_NFE" "$FLOW_BS" "$ELF_SAMPLES" "$ELF_BS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "Sample counts, NFE, and batch sizes must be positive integers"
done
(( LM1B_GEN_TOTAL >= NGPU )) || die "LM1B_GEN_TOTAL must be at least NGPU"
[[ ! -e "$OUTDIR" ]] || die "OUTDIR already exists; use a new directory: $OUTDIR"

gpu_ids=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
    (( ${#gpu_ids[@]} >= NGPU )) || die "CUDA_VISIBLE_DEVICES exposes fewer than $NGPU GPUs"
else
    for ((i=0; i<NGPU; i++)); do gpu_ids+=("$i"); done
fi

mkdir -p "$OUTDIR" "$DATA_CACHE"
OUTDIR=$(cd "$OUTDIR" && pwd)
DATA_CACHE=$(cd "$DATA_CACHE" && pwd)
export DATA_CACHE
printf '[run] Python: %s\n[run] Results: %s\n[run] GPUs: %s\n[run] Tasks: %s\n' \
    "$PY" "$OUTDIR" "$NGPU" "$TASKS"

if wants lm1b_gen; then
    shard_dir="$OUTDIR/langflow_lm1b_gen_shards"
    mkdir -p "$shard_dir"
    base=$((LM1B_GEN_TOTAL / NGPU))
    extra=$((LM1B_GEN_TOTAL % NGPU))
    pids=()
    shard_files=()
    printf '[run] LangFlow LM1B Gen PPL: %s samples, %s steps\n' "$LM1B_GEN_TOTAL" "$LM1B_GEN_STEPS"
    for ((i=0; i<NGPU; i++)); do
        n=$base
        (( i < extra )) && n=$((n + 1))
        shard_files+=("$shard_dir/shard_$i.json")
        (
            cd "$ROOT/LangFlow"
            CUDA_VISIBLE_DEVICES="${gpu_ids[$i]}" "$PY" gen_ppl.py \
                --model "$LM1B_MODEL" --tokenizer bert-base-uncased \
                --ppl-model "$PPL_MODEL" --num-samples "$n" \
                --num-steps "$LM1B_GEN_STEPS" --seq-length "$LM1B_GEN_LENGTH" \
                --batch-size "$LM1B_GEN_BS" --ppl-batch-size "$LM1B_SCORE_BS" \
                --seed "$((42 + i))" --print-samples 0 --device cuda \
                --output "${shard_files[$i]}" \
                > "$shard_dir/shard_$i.log" 2>&1
        ) &
        pids+=("$!")
        printf '[run]   GPU %s: %s samples -> %s\n' "${gpu_ids[$i]}" "$n" "${shard_files[$i]}"
    done
    failed=0
    for ((i=0; i<NGPU; i++)); do
        if ! wait "${pids[$i]}"; then
            printf '[error] LM1B shard %s failed: %s\n' "$i" "$shard_dir/shard_$i.log" >&2
            failed=1
        fi
    done
    (( failed == 0 )) || die "LangFlow LM1B Gen PPL failed"
    (
        cd "$ROOT/LangFlow"
        run_logged langflow_lm1b_gen_ppl "$PY" merge_gen_ppl.py "${shard_files[@]}"
    )
fi

run_flow_ppl() {
    local label=$1 config=$2 model=$3 first_n=$4
    local overrides=("eval.checkpoint_path=$model" "eval.n_steps=$FLOW_NFE"
                     "eval.ode_method=$FLOW_ODE" "loader.eval_batch_size=$FLOW_BS"
                     "loader.num_workers=$FLOW_WORKERS")
    (( first_n == 0 )) || overrides+=("+eval.first_n=$first_n")
    mkdir -p "$OUTDIR/$label"
    (
        cd "$ROOT/LangFlow"
        LANGFLOW_EVAL_OUTPUT_DIR="$OUTDIR/$label" \
            run_logged "$label" "$PY" -m torch.distributed.run \
                --standalone --nproc_per_node="$NGPU" eval_ppl.py \
                -cn "$config" "${overrides[@]}"
    )
}

if wants owt_ppl; then
    run_flow_ppl langflow_owt_ppl eval-owt "$OWT_MODEL" "$OWT_FIRST_N"
fi
if wants lm1b_ppl; then
    run_flow_ppl langflow_lm1b_ppl eval-lm1b-wrap "$LM1B_MODEL" "$LM1B_FIRST_N"
fi

cat > "$OUTDIR/elf_b_sampling.yml" <<'YAML'
- sampling_method: sde
  num_sampling_steps: [32]
  cfgs: [1]
  sde_gamma: 1.5
  self_cond_cfg_scales: [3]
  time_schedule: logit_normal
YAML
cat > "$OUTDIR/elf_l_sampling.yml" <<'YAML'
- sampling_method: sde
  num_sampling_steps: [64]
  cfgs: [1]
  sde_gamma: 1.0
  self_cond_cfg_scales: [3]
  time_schedule: logit_normal
YAML

run_elf() {
    local label=$1 size=$2 model=$3 sampling=$4
    local model_out="$OUTDIR/$label"
    mkdir -p "$model_out"
    (
        cd "$ROOT/ELF"
        PYTHONPATH="$ROOT/ELF/src${PYTHONPATH:+:$PYTHONPATH}" \
            run_logged "$label" "$PY" -m torch.distributed.run \
                --standalone --nproc_per_node="$NGPU" src/eval.py \
                --config "src/configs/training_configs/train_owt_ELF-$size.yml" \
                --checkpoint_path "$model" \
                --config_override "sampling_configs_path=$sampling" \
                --config_override "output_dir=$model_out" \
                --config_override "num_samples=$ELF_SAMPLES" \
                --config_override "global_batch_size=$((NGPU * ELF_BS))" \
                --config_override "eval_ppl_batch_size=$ELF_SCORE_BS" \
                --config_override "eval_ppl_model=$PPL_MODEL" \
                --config_override online_eval=true \
                --config_override use_bf16=true \
                --config_override "use_compile=$ELF_COMPILE" \
                --config_override use_wandb=false
    )
}

if wants elf_b; then
    run_elf elf_b_gen_ppl B "$ELF_B_MODEL" "$OUTDIR/elf_b_sampling.yml"
fi
if wants elf_l; then
    run_elf elf_l_gen_ppl L "$ELF_L_MODEL" "$OUTDIR/elf_l_sampling.yml"
fi

"$PY" - "$OUTDIR" <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
summary = {}

merge_log = root / 'langflow_lm1b_gen_ppl.log'
if merge_log.exists():
    match = re.search(r'Gen\. PPL:\s*([0-9.eE+-]+)', merge_log.read_text())
    if not match:
        raise RuntimeError(f'No Gen PPL in {merge_log}')
    summary['langflow_lm1b_gen_ppl'] = float(match.group(1))

for name in ('langflow_owt_ppl', 'langflow_lm1b_ppl'):
    directory = root / name
    if directory.exists():
        paths = list(directory.glob('flow_ppl_*.json'))
        if len(paths) != 1:
            raise RuntimeError(f'Expected one PPL result in {directory}, found {len(paths)}')
        summary[name] = json.loads(paths[0].read_text())['results']['ppl']

for name in ('elf_b_gen_ppl', 'elf_l_gen_ppl'):
    directory = root / name
    if directory.exists():
        paths = list(directory.rglob('metrics.jsonl'))
        if len(paths) != 1:
            raise RuntimeError(f'Expected one ELF metrics file in {directory}, found {len(paths)}')
        rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
        if len(rows) != 1:
            raise RuntimeError(f'Expected one ELF metrics row in {paths[0]}, found {len(rows)}')
        summary[name] = rows[0]['ppl']

(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('\nEvaluation summary:')
for name, value in summary.items():
    print(f'  {name}: {value:.4f}')
print(f'Results saved to: {root}')
PY
