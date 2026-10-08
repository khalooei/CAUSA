#!/usr/bin/env bash
# Qwen2-Audio-7B-Instruct: main grid, graded necessity, oracle control,
# language priors and substitution control. GPU, ~12 h on one RTX 5090.
# Every runner is resumable: re-running continues from the last saved row.
source "$(dirname "$0")/_common.sh"
R=$OUT/qwen2audio

step "Main necessity/sufficiency grid (Table 1, Table 4)"
$PY scripts/run_experiment.py --model qwen2audio \
    --audio-root "$AUDIO" --n 5000 --out "$R" --seed "$SEED"

step "Adaptive necessity search (Sec. 5.2)"
$PY scripts/run_adaptive_necessity.py --model qwen2audio \
    --audio-root "$AUDIO" --main-out "$R" --out "$R/adaptive.jsonl" --seed "$SEED"

step "Oracle-localization control (Table 2)"
$PY scripts/run_oracle_localization.py --model qwen2audio \
    --audio-root "$AUDIO" --n 800 --out "$OUT/oracle_qwen2audio" --seed "$SEED"

step "Language priors: single prompt and 5-paraphrase ensemble (Sec. 5.4-5.5)"
$PY scripts/run_language_prior.py --model qwen2audio --out "$OUT/language_prior_qwen2audio.json"
$PY scripts/run_language_prior_ensemble.py --model qwen2audio \
    --out "$OUT/language_prior_ensemble_qwen2audio.json"

step "Real-unrelated-audio substitution (Sec. 5.4)"
$PY scripts/run_substitution_test.py \
    --audio-root "$AUDIO" --main-out "$R" --out "$R/substitution.jsonl" --n 5000 --seed "$SEED"
