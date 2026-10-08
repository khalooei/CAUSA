"""Contrastive-decoding correction (paper Sec. 3.4, Sec. 5.5, Table 3, Fig. 1).

Reuses logits stored by the main run (no new model calls) and applies

    d_adj = (1 + alpha) * logit(p_yes_real) - alpha * logit(p_yes_contrast),

answering "yes" iff d_adj > 0, for three contrasts: AAD-CD (blank audio),
LangPrior-CD (no audio block) and CAUSA-CD (strongest localized cell).
Sweeps alpha in [0, 3] and reports alpha = 1.

Usage:
    python scripts/run_contrastive_decoding.py \
        --necessity outputs/qwen2audio/necessity.jsonl \
        --language-prior outputs/language_prior_qwen2audio.json \
        --out outputs/qwen2audio/contrastive_decoding.json
"""
import argparse
import json

import numpy as np


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def cd_accuracy(p_real, p_contrast, ground_truth_is_yes, alpha):
    """ground_truth_is_yes: bool array, True where the object truly is present."""
    d = (1 + alpha) * logit(p_real) - alpha * logit(p_contrast)
    pred_is_yes = d > 0
    correct = pred_is_yes == ground_truth_is_yes
    return float(correct.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--necessity", required=True)
    p.add_argument("--language-prior", default=None,
                   help="language_prior.json; omit to skip the LangPrior-CD condition")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.necessity) if not json.loads(l).get("skipped")]
    p_real = np.array([r["p_yes_before"] for r in rows])
    p_blank = np.array([r["p_yes_after_blank"] for r in rows])
    p_causa = np.array([r["grid"]["loc_w4.0_d-100"]["p_yes_after"] for r in rows])
    gt_is_yes = np.array([not r["is_hallucination"] for r in rows])  # is_hallucination == claimed yes, gt no
    n = len(rows)

    raw_accuracy = float(gt_is_yes.mean())  # every row here has raw claim == "yes"

    conditions = {"AAD-CD (blank)": p_blank, "CAUSA-CD (localized ablation)": p_causa}
    if args.language_prior:
        lang_prior = json.load(open(args.language_prior))
        p_lang = np.array([lang_prior[r["object_name"]]["p_yes_text_only"] for r in rows])
        conditions["LangPrior-CD (M3ID-style)"] = p_lang

    alphas = [round(a, 2) for a in np.arange(0.0, 3.01, 0.25)]
    result = {"n": n, "raw_accuracy_all_yes": raw_accuracy, "sweep": {}, "alpha_1_summary": {}}

    for name, p_contrast in conditions.items():
        sweep = {a: cd_accuracy(p_real, p_contrast, gt_is_yes, a) for a in alphas}
        result["sweep"][name] = sweep
        result["alpha_1_summary"][name] = {
            "accuracy": sweep[1.0],
            "delta_vs_raw": sweep[1.0] - raw_accuracy,
            "best_alpha_in_sweep": max(sweep, key=sweep.get),
            "best_alpha_accuracy": max(sweep.values()),
        }
        print(f"{name:35s} raw={raw_accuracy:.4f} alpha=1: {sweep[1.0]:.4f} "
              f"(delta {sweep[1.0]-raw_accuracy:+.4f})  best-in-sweep: "
              f"alpha={max(sweep, key=sweep.get)} -> {max(sweep.values()):.4f}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
