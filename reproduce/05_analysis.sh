#!/usr/bin/env bash
# Every paper number derived from the stored model outputs. CPU only,
# minutes. Requires stages 01-03.
source "$(dirname "$0")/_common.sh"
Q=$OUT/qwen2audio; A=$OUT/af3; O=$OUT/qwen25omni

step "Sec. 5.1: duration confound"
for M in qwen2audio af3; do
    $PY scripts/analyze_duration_confound.py --necessity "$OUT/$M/necessity.jsonl" \
        --audio-root "$AUDIO" --out "$OUT/$M/duration_confound.json"
done

step "Sec. 5.2: transient vs. continuous classes"
$PY scripts/analyze_object_class_breakdown.py --necessity "$Q/necessity.jsonl" \
    --adaptive "$Q/adaptive.jsonl" --out "$Q/object_class_breakdown.json"

step "Sec. 3.5: monotonicity check"
for M in qwen2audio af3; do
    $PY scripts/analyze_monotonicity.py --adaptive "$OUT/$M/adaptive.jsonl" \
        --model-name "$M" --out "$OUT/$M/monotonicity.json"
done

step "Table 2: oracle-localization control"
$PY scripts/analyze_oracle_localization.py \
    --jsonl "$OUT/oracle_qwen2audio/oracle_localization.jsonl" \
    --out "$OUT/oracle_qwen2audio/report.json"

step "Sec. 5.4: language-prior correction"
$PY scripts/analyze_language_prior_correction.py --necessity "$Q/necessity.jsonl" \
    --language-prior "$OUT/language_prior_qwen2audio.json" \
    --substitution "$Q/substitution.jsonl" --out "$Q/language_prior_correction.json"

step "Table 3 / Fig. 1: contrastive decoding"
for M in qwen2audio af3 qwen25omni; do
    $PY scripts/run_contrastive_decoding.py --necessity "$OUT/$M/necessity.jsonl" \
        --language-prior "$OUT/language_prior_$M.json" --out "$OUT/$M/contrastive_decoding.json"
    $PY scripts/analyze_decoding_strengths.py --necessity "$OUT/$M/necessity.jsonl" \
        --language-prior "$OUT/language_prior_$M.json" --model-name "$M" \
        --out "$OUT/$M/decoding_strengths.json"
done
$PY scripts/analyze_gated_decoding.py --necessity "$Q/necessity.jsonl" \
    --language-prior "$OUT/language_prior_ensemble_qwen2audio.json" \
    --language-prior-single "$OUT/language_prior_qwen2audio.json" \
    --model-name qwen2audio --out "$Q/gated_decoding.json"

step "Sec. 3.6 / 5.7: causal grounding profile and cross-model transfer"
for M in qwen2audio af3; do
    $PY scripts/analyze_causal_grounding_profile.py --necessity "$OUT/$M/necessity.jsonl" \
        --adaptive "$OUT/$M/adaptive.jsonl" --language-prior "$OUT/language_prior_$M.json" \
        --out "$OUT/$M/causal_grounding_profile.json" \
        --features-out "$OUT/$M/causal_grounding_features.npz"
done
$PY scripts/analyze_cross_model_transfer.py --qwen "$Q/causal_grounding_features.npz" \
    --af3 "$A/causal_grounding_features.npz" --out "$OUT/cross_model_transfer.json"
