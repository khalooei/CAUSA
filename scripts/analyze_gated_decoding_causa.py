"""Gated decoding with the localized (CAUSA) contrast (supplementary).

Same recipe as `analyze_gated_decoding.py`, but combining CAUSA-CD with
a p_lang-gated LangPrior-CD instead of AAD-CD, for models where the
localized contrast is the strongest single contrast (Qwen2.5-Omni).
Also tests adding the blank contrast as a third term.

Usage:
    python scripts/analyze_gated_decoding_causa.py \
        --necessity outputs/qwen25omni/necessity.jsonl \
        --language-prior outputs/language_prior_qwen25omni.json \
        --model-name qwen25omni --out outputs/qwen25omni/gated_decoding_causa.json
"""
import argparse
import json

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def decode_causa_lang(p_real, p_causa, p_lang, gt_is_yes, alpha_c, alpha_l):
    w_lang = alpha_l * p_lang
    d = (1 + alpha_c + w_lang) * logit(p_real) - alpha_c * logit(p_causa) - w_lang * logit(p_lang)
    return (d > 0)


def decode_flat_causa_lang(p_real, p_causa, p_lang, gt_is_yes, alpha_c, alpha_l):
    d = (1 + alpha_c + alpha_l) * logit(p_real) - alpha_c * logit(p_causa) - alpha_l * logit(p_lang)
    return (d > 0)


def decode_flat_3way(p_real, p_blank, p_causa, p_lang, gt_is_yes, alpha_b, alpha_c, alpha_l):
    d = ((1 + alpha_b + alpha_c + alpha_l) * logit(p_real) - alpha_b * logit(p_blank)
         - alpha_c * logit(p_causa) - alpha_l * logit(p_lang))
    return (d > 0)


def grid_search_cv(decode_fn, arrays, gt_is_yes, alphas, n_repeats=20, n_splits=5):
    fold_accs = []
    for seed in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(np.zeros(len(gt_is_yes)), gt_is_yes.astype(int)):
            best = {"acc": -1.0, "ac": None, "al": None}
            train_arrays = [a[train_idx] for a in arrays]
            for ac in alphas:
                for al in alphas:
                    pred = decode_fn(*train_arrays, gt_is_yes[train_idx], ac, al)
                    acc = (pred == gt_is_yes[train_idx]).mean()
                    if acc > best["acc"]:
                        best = {"acc": float(acc), "ac": ac, "al": al}
            test_arrays = [a[test_idx] for a in arrays]
            pred_test = decode_fn(*test_arrays, gt_is_yes[test_idx], best["ac"], best["al"])
            fold_accs.append(float((pred_test == gt_is_yes[test_idx]).mean()))
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


def grid_search_cv_3way(decode_fn, arrays, gt_is_yes, alphas, n_repeats=10, n_splits=5):
    """Coarser alpha grid (step 0.5 instead of 0.25) and fewer repeats
    (10 instead of 20) than the 2-way search -- a 3D grid at the 2-way
    resolution is 5x more (alpha,fold,repeat) evaluations than the
    paper's own 2-way CV protocol was designed for; halving repeats and
    doubling the alpha step keeps this a same-day CPU-only run while
    still giving a fair, non-cherry-picked per-fold-selected estimate."""
    fold_accs = []
    for seed in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(np.zeros(len(gt_is_yes)), gt_is_yes.astype(int)):
            best = {"acc": -1.0, "ab": None, "ac": None, "al": None}
            train_arrays = [a[train_idx] for a in arrays]
            for ab in alphas:
                for ac in alphas:
                    for al in alphas:
                        pred = decode_fn(*train_arrays, gt_is_yes[train_idx], ab, ac, al)
                        acc = (pred == gt_is_yes[train_idx]).mean()
                        if acc > best["acc"]:
                            best = {"acc": float(acc), "ab": ab, "ac": ac, "al": al}
            test_arrays = [a[test_idx] for a in arrays]
            pred_test = decode_fn(*test_arrays, gt_is_yes[test_idx], best["ab"], best["ac"], best["al"])
            fold_accs.append(float((pred_test == gt_is_yes[test_idx]).mean()))
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


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

    raw_accuracy = float(gt_is_yes.mean())
    alphas = [round(a, 2) for a in np.arange(0.0, 3.01, 0.25)]
    alphas_coarse = [round(a, 2) for a in np.arange(0.0, 3.01, 0.5)]

    flat_mean, flat_std, flat_accs = grid_search_cv(
        decode_flat_causa_lang, (p_real, p_causa, p_lang), gt_is_yes, alphas)

    gated_mean, gated_std, gated_accs = grid_search_cv(
        decode_causa_lang, (p_real, p_causa, p_lang), gt_is_yes, alphas)

    t, pval = stats.ttest_rel(gated_accs, flat_accs)
    win_rate = float((np.array(gated_accs) > np.array(flat_accs)).mean())

    threeway_mean, threeway_std, threeway_accs = grid_search_cv_3way(
        decode_flat_3way, (p_real, p_blank, p_causa, p_lang), gt_is_yes, alphas_coarse)

    # grid_search_cv's first 10 repeats (seeds 0-9) use the identical
    # StratifiedKFold splits as grid_search_cv_3way's 10 repeats (same
    # seeds, same n_splits, same gt_is_yes array) -- a legitimate paired
    # comparison despite the different alpha-grid resolution used to
    # select each fold's winning alpha.
    flat_accs_matched = flat_accs[: len(threeway_accs)]
    t3, pval3 = stats.ttest_rel(threeway_accs, flat_accs_matched)
    win_rate3 = float((np.array(threeway_accs) > np.array(flat_accs_matched)).mean())

    result = {
        "model": args.model_name,
        "n": n,
        "raw_accuracy": raw_accuracy,
        "flat_causa_plus_langprior_cv": {
            "description": "CAUSA-CD (loc_w4.0_d-100) + LangPrior-CD combined, flat alpha, "
                            "5-fold CV x 20 repeats, alpha grid-searched per fold",
            "cv_accuracy_mean": flat_mean, "cv_accuracy_std": flat_std,
        },
        "langprior_gated_causa_plus_langprior_cv": {
            "description": "same, but LangPrior's weight gated by p_lang itself "
                            "(the recipe that took Qwen2-Audio 70.0%->71.4%)",
            "cv_accuracy_mean": gated_mean, "cv_accuracy_std": gated_std,
        },
        "gated_vs_flat_paired": {
            "t_stat": float(t), "p_value": float(pval), "win_rate_gated": win_rate,
            "delta_mean": gated_mean - flat_mean,
        },
        "flat_3way_blank_causa_lang_cv": {
            "description": "AAD(blank)+CAUSA+LangPrior, all flat alpha, 5-fold CV x 10 repeats, "
                            "coarser alpha step (0.5) -- does adding the (weak-alone) blank "
                            "contrast as a third signal push the 2-way CAUSA+LangPrior "
                            "combination higher?",
            "cv_accuracy_mean": threeway_mean, "cv_accuracy_std": threeway_std,
            "delta_vs_2way_causa_lang": threeway_mean - flat_mean,
        },
        "threeway_vs_2way_paired_matched_folds": {
            "description": "paired t-test, first 10 repeats only (identical fold splits "
                            "between the 2-way and 3-way searches)",
            "t_stat": float(t3), "p_value": float(pval3), "win_rate_3way": win_rate3,
        },
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
