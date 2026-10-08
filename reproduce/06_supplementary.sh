#!/usr/bin/env bash
# Additional analyses not reported in the paper. The wrong-label and
# temporal-concentration controls need a GPU (CLAP and, for the former,
# the LALM); the rest are CPU only.
source "$(dirname "$0")/_common.sh"
Q=$OUT/qwen2audio; O=$OUT/qwen25omni

for M in qwen2audio af3; do
    step "Matched-removal-budget curve: $M"
    $PY scripts/analyze_matched_budget_curve.py --adaptive "$OUT/$M/adaptive.jsonl" \
        --model-name "$M" --out "$OUT/$M/matched_budget_curve.json"
    step "Temporal concentration of evidence: $M"
    $PY scripts/analyze_temporal_concentration.py --necessity "$OUT/$M/necessity.jsonl" \
        --adaptive "$OUT/$M/adaptive.jsonl" --audio-root "$AUDIO" \
        --model-name "$M" --out "$OUT/$M/temporal_concentration.json"
done

step "Wrong-label CLAP control"
$PY scripts/run_wronglabel_control.py --necessity "$Q/necessity.jsonl" \
    --audio-root "$AUDIO" --n 1500 --out "$Q/wronglabel_control.jsonl"
$PY scripts/analyze_wronglabel_control.py --wronglabel "$Q/wronglabel_control.jsonl" \
    --out "$Q/wronglabel_analysis.json"

step "Learned multi-signal decoding"
$PY scripts/analyze_learned_decoding.py --features "$Q/causal_grounding_features.npz" \
    --model-name qwen2audio --out "$Q/learned_decoding.json"

step "Gated CAUSA+LangPrior decoding on Qwen2.5-Omni"
$PY scripts/analyze_gated_decoding_causa.py --necessity "$O/necessity.jsonl" \
    --language-prior "$OUT/language_prior_qwen25omni.json" \
    --model-name qwen25omni --out "$O/gated_decoding_causa.json"
