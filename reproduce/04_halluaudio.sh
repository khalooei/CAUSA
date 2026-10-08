#!/usr/bin/env bash
# Generalization check on HalluAudio (Sec. 6, Limitations). GPU.
source "$(dirname "$0")/_common.sh"

for M in qwen2audio af3; do
    R=$OUT/halluaudio_$M
    step "HalluAudio grid: $M"
    $PY scripts/run_experiment.py --model "$M" --dataset halluaudio \
        --audio-root "$HALLU_AUDIO" --n 517 --out "$R" --seed "$SEED"
    $PY scripts/run_language_prior_halluaudio.py --model "$M" \
        --necessity "$R/necessity.jsonl" --out "$OUT/language_prior_halluaudio_$M.json"
done

step "Matched-duration effect and combined decoding on HalluAudio"
$PY scripts/analyze_duration_confound.py --dataset halluaudio \
    --necessity "$OUT/halluaudio_qwen2audio/necessity.jsonl" --audio-root "$HALLU_AUDIO" \
    --out "$OUT/halluaudio_qwen2audio/duration_confound.json"
$PY scripts/analyze_decoding_strengths.py \
    --necessity "$OUT/halluaudio_qwen2audio/necessity.jsonl" \
    --language-prior "$OUT/language_prior_halluaudio_qwen2audio.json" \
    --model-name qwen2audio_halluaudio --out "$OUT/halluaudio_qwen2audio/decoding_strengths.json"
