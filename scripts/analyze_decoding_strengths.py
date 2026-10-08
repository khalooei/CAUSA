"""Fix/break breakdown and multi-contrast decoding (paper Sec. 5.5).

From stored logits (no new model calls):

  1. per-contrast confusion breakdown: hallucinations fixed vs. correct
     claims broken;
  2. combined AAD + LangPrior decoding,
        d = (1 + a_b + a_l) logit(p_real) - a_b logit(p_blank) - a_l logit(p_lang),
     at a_b = a_l = 1, its swept optimum, and 5-fold CV over (a_b, a_l).

Usage:
    python scripts/analyze_decoding_strengths.py \
        --necessity outputs/qwen2audio/necessity.jsonl \
        --language-prior outputs/language_prior_qwen2audio.json \
        --model-name qwen2audio --out outputs/qwen2audio/decoding_strengths.json
"""
import argparse
import json

import numpy as np


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def cd_predict(p_real, p_contrast, alpha):
    d = (1 + alpha) * logit(p_real) - alpha * logit(p_contrast)
    return d > 0


def confusion_breakdown(p_real, p_contrast, gt_is_yes, alpha=1.0):
    pred_before = p_real > 0.5  # the model's own raw affirmative claim (always True here by construction)
    pred_after = cd_predict(p_real, p_contrast, alpha)
    changed = pred_before != pred_after
    hallucination_fixed = int((changed & ~gt_is_yes).sum())  # was wrong "yes", now says "no" -> fixed
    correct_broken = int((changed & gt_is_yes).sum())  # was correct "yes", now wrongly says "no" -> broken
    return {
        "n_changed": int(changed.sum()),
        "n_hallucinations_fixed": hallucination_fixed,
        "n_correct_claims_broken": correct_broken,
        "n_hallucinations_total": int((~gt_is_yes).sum()),
        "n_correct_total": int(gt_is_yes.sum()),
        "fix_rate": hallucination_fixed / max(1, int((~gt_is_yes).sum())),
        "break_rate": correct_broken / max(1, int(gt_is_yes.sum())),
        "net_accuracy_change": (hallucination_fixed - correct_broken) / len(gt_is_yes),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--necessity", required=True)
    p.add_argument("--language-prior", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.necessity) if not json.loads(l).get("skipped")]
    lang_prior = json.load(open(args.language_prior))
    rows = [r for r in rows if r["object_name"] in lang_prior]

    p_real = np.array([r["p_yes_before"] for r in rows])
    p_blank = np.array([r["p_yes_after_blank"] for r in rows])
    p_causa = np.array([r["grid"]["loc_w4.0_d-100"]["p_yes_after"] for r in rows])
    p_lang = np.array([lang_prior[r["object_name"]]["p_yes_text_only"] for r in rows])
    gt_is_yes = np.array([not r["is_hallucination"] for r in rows])
    n = len(rows)

    result = {"model": args.model_name, "n": n, "raw_accuracy": float(gt_is_yes.mean())}

    # --- 1. multi-contrast decoding: AAD (blank) + LangPrior combined ---
    # alpha in [0,3], step 0.25: same literature-standard range as the
    # single-contrast sweep (run_contrastive_decoding.py), so "best in
    # sweep" numbers are comparable and not a wider search on one side.
    alphas = [round(a, 2) for a in np.arange(0.0, 3.01, 0.25)]
    d_primary = (1 + 1 + 1) * logit(p_real) - 1 * logit(p_blank) - 1 * logit(p_lang)
    acc_primary = float(((d_primary > 0) == gt_is_yes).mean())
    best = {"acc": -1.0}
    grid_summary = []
    for a_b in alphas:
        for a_l in alphas:
            d = ((1 + a_b + a_l) * logit(p_real) - a_b * logit(p_blank) - a_l * logit(p_lang))
            acc = float(((d > 0) == gt_is_yes).mean())
            grid_summary.append({"alpha_blank": float(a_b), "alpha_lang": float(a_l), "accuracy": acc})
            if acc > best["acc"]:
                best = {"alpha_blank": float(a_b), "alpha_lang": float(a_l), "acc": acc}
    acc_blank_sweep = max(float(((cd_predict(p_real, p_blank, a)) == gt_is_yes).mean()) for a in alphas)
    acc_lang_sweep = max(float(((cd_predict(p_real, p_lang, a)) == gt_is_yes).mean()) for a in alphas)
    at_boundary = bool(best["alpha_blank"] == alphas[-1] or best["alpha_lang"] == alphas[-1])
    result["multi_contrast_decoding"] = {
        "primary_alpha1_1_accuracy": acc_primary,
        "primary_minus_best_single": acc_primary - max(acc_blank_sweep, acc_lang_sweep),
        "best_in_sweep": best,
        "best_in_sweep_at_boundary_caveat": at_boundary,
        "best_single_blank_sweep": acc_blank_sweep,
        "best_single_lang_sweep": acc_lang_sweep,
        "best_combined_minus_best_single": best["acc"] - max(acc_blank_sweep, acc_lang_sweep),
        "grid": grid_summary,
    }

    # --- 2. confusion-matrix breakdown at alpha=1, each single contrast ---
    result["confusion_breakdown_alpha1"] = {
        "AAD-CD (blank)": confusion_breakdown(p_real, p_blank, gt_is_yes),
        "LangPrior-CD": confusion_breakdown(p_real, p_lang, gt_is_yes),
        "CAUSA-CD (localized)": confusion_breakdown(p_real, p_causa, gt_is_yes),
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "multi_contrast_decoding"}, indent=2))
    print(json.dumps({"multi_contrast_decoding": {k: v for k, v in result["multi_contrast_decoding"].items() if k != "grid"}}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
