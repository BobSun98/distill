#!/usr/bin/env bash
# Shard LangFlow Gen. PPL generation across 8 GPUs, then merge.
set -euo pipefail

PY=/proj/gpu_mtk53742/.conda/envs/distill/bin/python
NGPU=${NGPU:-8}
TOTAL=${TOTAL:-512}
STEPS=${STEPS:-1024}
SEQLEN=${SEQLEN:-1024}
BS=${BS:-8}
PPL_BS=${PPL_BS:-8}
OUTDIR=${OUTDIR:-gen_ppl_shards}

mkdir -p "$OUTDIR"
PER=$(( (TOTAL + NGPU - 1) / NGPU ))   # ceil-divide; last shard may overshoot slightly
echo "[run] $NGPU GPUs, $PER samples/GPU (total target $TOTAL), steps=$STEPS seqlen=$SEQLEN bs=$BS"

pids=()
for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i "$PY" gen_ppl.py \
        --model ./checkpoints \
        --tokenizer gpt2 \
        --ppl-model gpt2-large \
        --num-samples "$PER" \
        --num-steps "$STEPS" \
        --seq-length "$SEQLEN" \
        --batch-size "$BS" \
        --ppl-batch-size "$PPL_BS" \
        --seed $((42 + i)) \
        --print-samples 0 \
        --device cuda \
        --output "$OUTDIR/shard_$i.json" \
        > "$OUTDIR/shard_$i.log" 2>&1 &
    pids+=($!)
    echo "[run]   launched GPU $i -> $OUTDIR/shard_$i.json (pid ${pids[-1]})"
done

fail=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        echo "[run] GPU $idx FAILED — see $OUTDIR/shard_$idx.log" >&2
        fail=1
    fi
done
[[ $fail -eq 0 ]] || { echo "[run] some shards failed, aborting merge" >&2; exit 1; }

echo "[run] all shards done, merging..."
"$PY" merge_gen_ppl.py "$OUTDIR"/shard_*.json
