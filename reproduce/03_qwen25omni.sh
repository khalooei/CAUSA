#!/usr/bin/env bash
# Qwen2.5-Omni-7B, the third model (Sec. 5.6). GPU.
source "$(dirname "$0")/_common.sh"
R=$OUT/qwen25omni

step "Main necessity/sufficiency grid (Table 4)"
$PY scripts/run_experiment.py --model qwen25omni \
    --audio-root "$AUDIO" --n 2000 --out "$R" --seed "$SEED"

step "Language prior"
$PY scripts/run_language_prior.py --model qwen25omni --out "$OUT/language_prior_qwen25omni.json"
