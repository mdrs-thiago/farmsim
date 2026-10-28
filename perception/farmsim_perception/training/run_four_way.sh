#!/usr/bin/env bash
# Compare architecture and encoder initialization on the same dataset split.
set -euo pipefail

# Run from anywhere with the farmsim venv active.
cd "$(dirname "$0")/../../.."
PY="${PY:-python}"
DATA="$(pwd)/dataset/row_perception_v1"
EPOCHS="${EPOCHS:-12}"
SEED="${SEED:-0}"
OUT="${OUT:-$(pwd)/models/row_net/four_way_seed${SEED}_epochs${EPOCHS}}"

if [[ "${ALLOW_CPU:-0}" != 1 ]]; then
    "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "CUDA unavailable; set ALLOW_CPU=1 to run on CPU")'
fi

mkdir -p "$OUT"
for init in scratch imagenet; do
    for model in cnn cnn_lstm; do
        run="${model}_${init}"
        if [[ -f "$OUT/$run/done" ]]; then
            echo "[skip] $run already complete"
            continue
        fi
        if [[ -e "$OUT/$run/last.pt" ]]; then
            echo "[error] $run has an incomplete run; use a fresh OUT to avoid overwriting it" >&2
            exit 1
        fi
        "$PY" -m farmsim_perception.training.train --data "$DATA" --out "$OUT" --model "$model" \
            --encoder-init "$init" --run-name "$run" --epochs "$EPOCHS" \
            --seed "$SEED" --clip-len 8 --frames-per-batch 32 \
            --lr 6e-4 --encoder-lr-scale 0.3 --warmup-steps 500 \
            > "$OUT/${run}.log" 2>&1
        touch "$OUT/$run/done"
    done
done

"$PY" -m farmsim_perception.training.evaluate --data "$DATA" --runs "$OUT" \
    --models cnn_scratch cnn_lstm_scratch cnn_imagenet cnn_lstm_imagenet \
    > "$OUT/evaluate.log" 2>&1
for init in scratch imagenet; do
    "$PY" -m farmsim_perception.training.analyze --data "$DATA" --runs "$OUT" \
        --models "cnn_${init}" "cnn_lstm_${init}" \
        --conditions clean dropout_0.1 dropout_0.3 \
        > "$OUT/analyze_${init}.log" 2>&1
done
echo "Results: $OUT/comparison_test.md"
