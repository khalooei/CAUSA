"""Language-prior correction analysis (paper Sec. 5.4).

Computes the AUROC of the language-only prior and of
Delta = P(yes | real) - P(yes | no audio), a cross-validated 3-feature
ensemble (blank-drop, CLAP agreement, Delta), optionally a 4th feature
(real-unrelated-audio substitution), and per-example / per-class
bootstrap 95% CIs on each gap to the blank-audio baseline.

Usage:
    python scripts/analyze_language_prior_correction.py \
        --necessity outputs/qwen2audio/necessity.jsonl \
        --language-prior outputs/language_prior_qwen2audio.json \
        --substitution outputs/qwen2audio/substitution.jsonl \
        --out outputs/qwen2audio/language_prior_correction.json
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if not json.loads(l).get("skipped")]


def cv_auroc(X, y, n_splits=5, seed=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        pred[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return roc_auc_score(y, pred), pred


def bootstrap_auroc_diff_by_example(y, score_a, score_b, n_resamples=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0 or y[idx].sum() == n:
            continue
        diffs.append(roc_auc_score(y[idx], score_a[idx]) - roc_auc_score(y[idx], score_b[idx]))
    diffs = np.array(diffs)
    point = roc_auc_score(y, score_a) - roc_auc_score(y, score_b)
    return point, (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def bootstrap_auroc_diff_by_class(y, score_a, score_b, classes, n_resamples=2000, seed=0):
    rng = np.random.default_rng(seed)
    unique_classes = np.array(sorted(set(classes)))
    classes = np.array(classes)
    diffs = []
    favor_b_count = 0
    for _ in range(n_resamples):
        sampled_classes = rng.choice(unique_classes, size=len(unique_classes), replace=True)
        idx = np.concatenate([np.where(classes == c)[0] for c in sampled_classes])
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        d = roc_auc_score(y[idx], score_a[idx]) - roc_auc_score(y[idx], score_b[idx])
        diffs.append(d)
        if d < 0:
            favor_b_count += 1
    diffs = np.array(diffs)
    point = roc_auc_score(y, score_a) - roc_auc_score(y, score_b)
    return point, (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))), favor_b_count / len(diffs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--necessity", required=True)
    p.add_argument("--language-prior", required=True)
    p.add_argument("--substitution", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = load_jsonl(args.necessity)
    lang_prior = json.load(open(args.language_prior))

    y = np.array([r["is_hallucination"] for r in rows], dtype=int)
    p_before = np.array([r["p_yes_before"] for r in rows])
    p_lang = np.array([lang_prior[r["object_name"]]["p_yes_text_only"] for r in rows])
    clap_sim = np.array([r["clap_similarity"] for r in rows])
    blank_drop = p_before - np.array([r["p_yes_after_blank"] for r in rows])
    delta = p_before - p_lang  # audio_added_confidence
    object_names = [r["object_name"] for r in rows]

    tp_mask = y == 0
    result = {
        "n": len(rows),
        "mean_p_yes_lang_true_positive": float(p_lang[tp_mask].mean()),
        "mean_p_yes_lang_hallucination": float(p_lang[~tp_mask].mean()),
        "auroc_language_prior_alone": float(roc_auc_score(y, p_lang)),
        "mean_delta_true_positive": float(delta[tp_mask].mean()),
        "mean_delta_hallucination": float(delta[~tp_mask].mean()),
        "auroc_delta_alone": float(roc_auc_score(y, -delta)),
        "auprc_delta_alone": float(average_precision_score(y, -delta)),
        "auroc_blank_drop": float(roc_auc_score(y, -blank_drop)),
        "auprc_blank_drop": float(average_precision_score(y, -blank_drop)),
        "auprc_clap_agreement": float(average_precision_score(y, -clap_sim)),
    }

    X3 = np.stack([-blank_drop, -clap_sim, -delta], axis=1)
    ens3_auroc, ens3_pred = cv_auroc(X3, y)
    result["auroc_ensemble_3feature"] = float(ens3_auroc)
    result["auprc_ensemble_3feature"] = float(average_precision_score(y, ens3_pred))

    point_d, ci_d = bootstrap_auroc_diff_by_example(y, -delta, -blank_drop)
    result["delta_vs_blank_by_example"] = {"point": point_d, "ci95": list(ci_d)}
    # ens3_pred is already P(hallucination) from predict_proba - higher = more
    # likely hallucination, same orientation as y, so used directly (not negated)
    point_e, ci_e = bootstrap_auroc_diff_by_example(y, ens3_pred, -blank_drop)
    result["ensemble_vs_blank_by_example"] = {"point": point_e, "ci95": list(ci_e)}

    point_dc, ci_dc, frac_favor_blank = bootstrap_auroc_diff_by_class(y, -delta, -blank_drop, object_names)
    result["delta_vs_blank_by_class"] = {
        "point": point_dc, "ci95": list(ci_dc), "frac_resamples_favoring_blank": frac_favor_blank,
    }

    if args.substitution:
        sub_rows = {r["entry_id"]: r for r in load_jsonl(args.substitution)}
        keep = [i for i, r in enumerate(rows) if r["entry_id"] in sub_rows]
        sub_drop = np.array([sub_rows[rows[i]["entry_id"]]["p_yes_before"] -
                              sub_rows[rows[i]["entry_id"]]["p_yes_after_substitution"] for i in keep])
        y_sub = y[keep]
        result["auroc_substitution_alone"] = float(roc_auc_score(y_sub, -sub_drop))
        result["auprc_substitution_alone"] = float(average_precision_score(y_sub, -sub_drop))
        X4 = np.stack([-blank_drop[keep], -clap_sim[keep], -delta[keep], -sub_drop], axis=1)
        ens4_auroc, ens4_pred = cv_auroc(X4, y_sub)
        result["auroc_ensemble_4feature"] = float(ens4_auroc)
        result["auprc_ensemble_4feature"] = float(average_precision_score(y_sub, ens4_pred))
        result["n_substitution_matched"] = len(keep)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
