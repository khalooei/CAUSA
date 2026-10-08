"""Causal grounding profile z (paper Sec. 3.6, Sec. 5.7).

Joins five per-claim signals, all computed by earlier steps (no new model
calls): blank-drop, whole-clip CLAP similarity, language-prior gap Delta,
matched-control specificity CS and localization efficiency Delta_loc.
Reports each signal's AUROC, a 5-fold cross-validated logistic ensemble
over z, bootstrap CIs, and saves the feature matrix for
`analyze_cross_model_transfer.py`.

Usage:
    python scripts/analyze_causal_grounding_profile.py \
        --necessity outputs/qwen2audio/necessity.jsonl \
        --adaptive outputs/qwen2audio/adaptive.jsonl \
        --language-prior outputs/language_prior_qwen2audio.json \
        --out outputs/qwen2audio/causal_grounding_profile.json \
        --features-out outputs/qwen2audio/causal_grounding_features.npz
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

GRID_CELLS = ["w2.0_d-40", "w2.0_d-100", "w4.0_d-40", "w4.0_d-100"]


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
    classes = np.array(classes)
    unique_classes = np.array(sorted(set(classes)))
    diffs = []
    for _ in range(n_resamples):
        sampled = rng.choice(unique_classes, size=len(unique_classes), replace=True)
        idx = np.concatenate([np.where(classes == c)[0] for c in sampled])
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        diffs.append(roc_auc_score(y[idx], score_a[idx]) - roc_auc_score(y[idx], score_b[idx]))
    diffs = np.array(diffs)
    point = roc_auc_score(y, score_a) - roc_auc_score(y, score_b)
    return point, (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--necessity", required=True)
    p.add_argument("--adaptive", required=True)
    p.add_argument("--language-prior", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--features-out", required=True)
    args = p.parse_args()

    necessity_rows = {r["entry_id"]: r for r in load_jsonl(args.necessity)}
    adaptive_rows = {r["entry_id"]: r for r in load_jsonl(args.adaptive)}
    lang_prior = json.load(open(args.language_prior))

    ids = sorted(set(necessity_rows) & set(adaptive_rows))
    dropped = len(necessity_rows) - len(ids)

    y, blank_drop, clap_sim, delta_lang, cs_grid, delta_loc, object_names, kept_ids = [], [], [], [], [], [], [], []
    for eid in ids:
        r = necessity_rows[eid]
        a = adaptive_rows[eid]
        obj = r["object_name"]
        if obj not in lang_prior:
            continue
        p_before = r["p_yes_before"]
        y.append(r["is_hallucination"])
        blank_drop.append(p_before - r["p_yes_after_blank"])
        clap_sim.append(r["clap_similarity"])
        delta_lang.append(p_before - lang_prior[obj]["p_yes_text_only"])
        cs = np.mean([r["grid"][f"rand_{c}"]["p_yes_after"] - r["grid"][f"loc_{c}"]["p_yes_after"]
                      for c in GRID_CELLS])
        cs_grid.append(cs)
        delta_loc.append(a["random_order"]["necessary_fraction"] - a["clap_order"]["necessary_fraction"])
        object_names.append(obj)
        kept_ids.append(eid)

    y = np.array(y, dtype=int)
    blank_drop = np.array(blank_drop)
    clap_sim = np.array(clap_sim)
    delta_lang = np.array(delta_lang)
    cs_grid = np.array(cs_grid)
    delta_loc = np.array(delta_loc)

    result = {
        "n": len(y),
        "n_dropped_no_adaptive_or_lang_match": dropped,
        "n_hallucinated": int(y.sum()),
        "n_correct": int(len(y) - y.sum()),
        "auroc_blank_drop": float(roc_auc_score(y, -blank_drop)),
        "auroc_clap_agreement": float(roc_auc_score(y, -clap_sim)),
        "auroc_delta_lang": float(roc_auc_score(y, -delta_lang)),
        "auroc_cs_grid_specificity": float(roc_auc_score(y, -cs_grid)),
        "auroc_delta_loc_graded_specificity": float(roc_auc_score(y, -delta_loc)),
    }

    X5 = np.stack([-blank_drop, -clap_sim, -delta_lang, -cs_grid, -delta_loc], axis=1)
    ens5_auroc, ens5_pred = cv_auroc(X5, y)
    result["auroc_causal_grounding_ensemble_5feature"] = float(ens5_auroc)
    result["auprc_causal_grounding_ensemble_5feature"] = float(average_precision_score(y, ens5_pred))

    best_single_name = max(
        ["auroc_blank_drop", "auroc_clap_agreement", "auroc_delta_lang",
         "auroc_cs_grid_specificity", "auroc_delta_loc_graded_specificity"],
        key=lambda k: result[k],
    )
    result["best_single_fixed_contrast"] = best_single_name
    best_single_score = {
        "auroc_blank_drop": -blank_drop, "auroc_clap_agreement": -clap_sim,
        "auroc_delta_lang": -delta_lang, "auroc_cs_grid_specificity": -cs_grid,
        "auroc_delta_loc_graded_specificity": -delta_loc,
    }[best_single_name]

    point_e, ci_e = bootstrap_auroc_diff_by_example(y, ens5_pred, best_single_score)
    result["ensemble_vs_best_single_by_example"] = {"point": point_e, "ci95": list(ci_e)}
    point_ec, ci_ec = bootstrap_auroc_diff_by_class(y, ens5_pred, best_single_score, object_names)
    result["ensemble_vs_best_single_by_class"] = {"point": point_ec, "ci95": list(ci_ec)}

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")

    np.savez(args.features_out, y=y, blank_drop=blank_drop, clap_sim=clap_sim,
             delta_lang=delta_lang, cs_grid=cs_grid, delta_loc=delta_loc,
             object_names=np.array(object_names), entry_ids=np.array(kept_ids))
    print(f"wrote {args.features_out}")


if __name__ == "__main__":
    main()
