#!/usr/bin/env bash
# NVIDIA Audio Flamingo 3 over the full benchmark (n = 10,800). GPU, a few hours.
source "$(dirname "$0")/_common.sh"
R=$OUT/af3

step "Main necessity/sufficiency grid (Table 4)"
$PY scripts/run_experiment.py --model af3 \
    --audio-root "$AUDIO" --n 10800 --out "$R" --seed "$SEED"

step "Adaptive necessity search (Sec. 5.2)"
$PY scripts/run_adaptive_necessity.py --model af3 \
    --audio-root "$AUDIO" --main-out "$R" --out "$R/adaptive.jsonl" --seed "$SEED"

step "Language prior"
$PY scripts/run_language_prior.py --model af3 --out "$OUT/language_prior_af3.json"
