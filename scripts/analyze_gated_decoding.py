"""Confidence-gated multi-contrast decoding (paper Sec. 5.5, Table 3).

    d = (1 + a_b + a_l * p_lang) logit(p_real)
        - a_b logit(p_blank) - a_l * p_lang * logit(p_lang)

The LangPrior weight is scaled per example by the prior's own confidence
p_lang. (a_b, a_l) are grid-searched on each training fold and applied to
the held-out fold (5-fold CV, 20 repeats); gated vs. flat weighting is
compared with a paired t-test over matched splits. Also evaluates a
CLAP-similarity gate on the blank contrast.

Pass the paraphrase-averaged prior from `run_language_prior_ensemble.py`
as --language-prior (Table 3, "+ ensemble prior") and, optionally, the
single-prompt prior as --language-prior-single to also compute the flat
and gated single-prompt rows, the gated-vs-flat test, and the
ensemble-vs-single paired comparison.

Usage:
    python scripts/analyze_gated_decoding.py \
        --necessity outputs/qwen2audio/necessity.jsonl \
        --language-prior outputs/language_prior_ensemble_qwen2audio.json \
        --language-prior-single outputs/language_prior_qwen2audio.json \
        --model-name qwen2audio --out outputs/qwen2audio/gated_decoding.json
"""
import argparse
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def clap_normalized(clap_sim):
    """min-max normalize CLAP similarity into [0,1] so (1-clap_sim) is a
    usable per-example gate on the same scale as p_lang."""
    lo, hi = clap_sim.min(), clap_sim.max()
    return (clap_sim - lo) / max(1e-9, hi - lo)


def decode_flat(p_real, p_blank, p_lang, gt_is_yes, alpha_blank, alpha_lang):
    d = (1 + alpha_blank + alpha_lang) * logit(p_real) - alpha_blank * logit(p_blank) - alpha_lang * logit(p_lang)
    return (d > 0)


def decode_baseline(p_real, p_blank, p_lang, gt_is_yes, alpha_blank, alpha_lang):
    w_lang = alpha_lang * p_lang
    d = (1 + alpha_blank + w_lang) * logit(p_real) - alpha_blank * logit(p_blank) - w_lang * logit(p_lang)
    return (d > 0)


def decode_clap_gated(p_real, p_blank, p_lang, clap_gate, gt_is_yes, alpha_blank, alpha_lang):
    w_blank = alpha_blank * clap_gate
    w_lang = alpha_lang * p_lang
    d = (1 + w_blank + w_lang) * logit(p_real) - w_blank * logit(p_blank) - w_lang * logit(p_lang)
    return (d > 0)


def grid_search_cv(decode_fn, arrays, gt_is_yes, alphas, n_repeats=20, n_splits=5):
    """arrays: tuple of per-example arrays (excluding gt) passed positionally
    to decode_fn(*arrays_subset, gt_is_yes, alpha_blank, alpha_lang)."""
    fold_accs = []
    chosen_alphas = []
    for seed in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(np.zeros(len(gt_is_yes)), gt_is_yes.astype(int)):
            best = {"acc": -1.0, "ab": None, "al": None}
            train_arrays = [a[train_idx] for a in arrays]
            for ab in alphas:
                for al in alphas:
                    pred = decode_fn(*train_arrays, gt_is_yes[train_idx], ab, al)
                    acc = (pred == gt_is_yes[train_idx]).mean()
                    if acc > best["acc"]:
                        best = {"acc": float(acc), "ab": ab, "al": al}
            test_arrays = [a[test_idx] for a in arrays]
            pred_test = decode_fn(*test_arrays, gt_is_yes[test_idx], best["ab"], best["al"])
            fold_accs.append(float((pred_test == gt_is_yes[test_idx]).mean()))
            chosen_alphas.append((best["ab"], best["al"]))
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs, chosen_alphas


def paired_ttest_win_rate(accs_a, accs_b):
    """accs_a, accs_b: same-length lists from identically-seeded repeated
    CV (same fold splits), so this is a legitimate paired comparison."""
    from scipy import stats
    diffs = np.array(accs_a) - np.array(accs_b)
    t, p = stats.ttest_rel(accs_a, accs_b)
    win_rate = float((diffs > 0).mean())
    return float(t), float(p), win_rate


def paired_summary(accs_a, accs_b, n_splits=5):
    """Paired comparison over identical splits, per fold and per repeat
    (fold accuracies averaged within each CV repeat)."""
    t, pval, win_fold = paired_ttest_win_rate(accs_a, accs_b)
    rep_a = np.array(accs_a).reshape(-1, n_splits).mean(axis=1)
    rep_b = np.array(accs_b).reshape(-1, n_splits).mean(axis=1)
    return {
        "t_stat": t, "p_value": pval,
        "win_rate_per_fold": win_fold,
        "win_rate_per_repeat": float((rep_a > rep_b).mean()),
        "delta_mean": float(np.mean(accs_a) - np.mean(accs_b)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--necessity", required=True)
    p.add_argument("--language-prior", required=True)
    p.add_argument("--language-prior-single", default=None,
                   help="optional single-prompt prior (run_language_prior.py) to compare against "
                        "the paraphrase-averaged prior passed as --language-prior")
    p.add_argument("--model-name", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.necessity) if not json.loads(l).get("skipped")]
    lang_prior = json.load(open(args.language_prior))
    single_prior = json.load(open(args.language_prior_single)) if args.language_prior_single else None
    rows = [r for r in rows if r["object_name"] in lang_prior
            and (single_prior is None or r["object_name"] in single_prior)]

    p_real = np.array([r["p_yes_before"] for r in rows])
    p_blank = np.array([r["p_yes_after_blank"] for r in rows])
    p_lang = np.array([lang_prior[r["object_name"]]["p_yes_text_only"] for r in rows])
    clap_sim = np.array([r["clap_similarity"] for r in rows])
    # trust blank-ablation necessity MORE when CLAP does NOT confirm
    # alignment (low clap_sim), so gate = (1 - normalized clap_sim)
    clap_gate = 1.0 - clap_normalized(clap_sim)
    gt_is_yes = np.array([not r["is_hallucination"] for r in rows])
    n = len(rows)

    alphas = [round(a, 2) for a in np.arange(0.0, 3.01, 0.25)]

    raw_accuracy = float(gt_is_yes.mean())

    base_mean, base_std, base_accs, base_alphas = grid_search_cv(
        decode_baseline, (p_real, p_blank, p_lang), gt_is_yes, alphas)

    clap_mean, clap_std, clap_accs, clap_alphas = grid_search_cv(
        decode_clap_gated, (p_real, p_blank, p_lang, clap_gate), gt_is_yes, alphas)

    t, pval, win_rate = paired_ttest_win_rate(clap_accs, base_accs)

    flat_mean, flat_std, flat_accs, _ = grid_search_cv(
        decode_flat, (p_real, p_blank, p_lang), gt_is_yes, alphas)

    result = {
        "model": args.model_name,
        "n": n,
        "raw_accuracy": raw_accuracy,
        "baseline_langprior_gated_decoding": {
            "description": "paper's 71.9% setting (blank flat alpha, "
                            "lang gated by p_lang), 5-fold CV x 20 repeats, alpha grid-searched per fold",
            "cv_accuracy_mean": base_mean,
            "cv_accuracy_std": base_std,
        },
        "clap_gated_blank_plus_langprior_gated_decoding": {
            "description": "new: blank contrast also gated, by (1 - normalized CLAP similarity) "
                            "instead of a flat alpha_blank",
            "cv_accuracy_mean": clap_mean,
            "cv_accuracy_std": clap_std,
        },
        "clap_gated_vs_baseline_paired": {
            "t_stat": t, "p_value": pval, "win_rate_clap_gated": win_rate,
            "delta_mean": clap_mean - base_mean,
        },
        "flat_aad_langprior_decoding": {
            "description": "flat alpha on both contrasts, 5-fold CV x 20 repeats",
            "cv_accuracy_mean": flat_mean,
            "cv_accuracy_std": flat_std,
        },
        "gated_vs_flat_paired": paired_summary(base_accs, flat_accs),
    }

    if single_prior is not None:
        # paper Sec. 5.5: flat vs. gated with the single-prompt prior, then
        # single-prompt vs. paraphrase-averaged prior under the gated rule
        p_single = np.array([single_prior[r["object_name"]]["p_yes_text_only"] for r in rows])
        sflat_mean, sflat_std, sflat_accs, _ = grid_search_cv(
            decode_flat, (p_real, p_blank, p_single), gt_is_yes, alphas)
        sgate_mean, sgate_std, sgate_accs, _ = grid_search_cv(
            decode_baseline, (p_real, p_blank, p_single), gt_is_yes, alphas)
        result["single_prompt_prior"] = {
            "flat_cv_accuracy_mean": sflat_mean, "flat_cv_accuracy_std": sflat_std,
            "gated_cv_accuracy_mean": sgate_mean, "gated_cv_accuracy_std": sgate_std,
            "gated_vs_flat_paired": paired_summary(sgate_accs, sflat_accs),
        }
        result["ensemble_vs_single_prior_gated_paired"] = paired_summary(base_accs, sgate_accs)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
