"""Transient vs. continuous sound classes (paper Sec. 5.2).

Splits localization efficiency (Delta_loc) and detection AUROC by
acoustic event type (transient/impulsive vs. continuous/ambient classes),
with a Mann-Whitney test and a per-class table.

Usage:
    python scripts/analyze_object_class_breakdown.py \
        --necessity outputs/qwen2audio/necessity.jsonl \
        --adaptive outputs/qwen2audio/adaptive.jsonl \
        --out outputs/qwen2audio/object_class_breakdown.json
"""
import argparse
import json

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

TRANSIENT = {"clapping", "frog", "laughter", "siren", "cat", "crow", "glass breaking"}
CONTINUOUS = {"airplane", "vacuum cleaner", "car horn", "dog", "toilet flush",
              "drinking, sipping", "thunderstorm", "hen", "snoring", "door, wood creaks"}


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if not json.loads(l).get("skipped")]


def bootstrap_auroc_diff(y, drop_a, drop_b, n_resamples=3000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0 or y[idx].sum() == n:
            continue
        diffs.append(roc_auc_score(y[idx], -drop_a[idx]) - roc_auc_score(y[idx], -drop_b[idx]))
    diffs = np.array(diffs)
    point = roc_auc_score(y, -drop_a) - roc_auc_score(y, -drop_b)
    return point, (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--necessity", required=True)
    p.add_argument("--adaptive", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    necessity_rows = load_jsonl(args.necessity)
    adaptive_rows = load_jsonl(args.adaptive)

    # per-class Delta_loc from the adaptive (graded necessity) search
    per_class = {}
    for r in adaptive_rows:
        obj = r["object_name"]
        d = r["random_order"]["necessary_fraction"] - r["clap_order"]["necessary_fraction"]
        per_class.setdefault(obj, []).append(d)

    class_table = []
    for obj, diffs in per_class.items():
        if len(diffs) >= 8:
            class_table.append({
                "object": obj, "n": len(diffs),
                "mean_delta_loc": float(np.mean(diffs)),
                "se": float(np.std(diffs) / np.sqrt(len(diffs))),
            })
    class_table.sort(key=lambda r: -r["mean_delta_loc"])

    t_diffs = np.array([d for r in adaptive_rows if r["object_name"] in TRANSIENT
                         for d in [r["random_order"]["necessary_fraction"] - r["clap_order"]["necessary_fraction"]]])
    c_diffs = np.array([d for r in adaptive_rows if r["object_name"] in CONTINUOUS
                         for d in [r["random_order"]["necessary_fraction"] - r["clap_order"]["necessary_fraction"]]])
    u_stat, mw_p = stats.mannwhitneyu(t_diffs, c_diffs, alternative="greater")

    # per-class-type detection AUROC (blank vs. localized) from the main grid
    def auroc_gap(rows_subset):
        y = np.array([r["is_hallucination"] for r in rows_subset], dtype=int)
        p_before = np.array([r["p_yes_before"] for r in rows_subset])
        drop_blank = p_before - np.array([r["p_yes_after_blank"] for r in rows_subset])
        drop_loc = p_before - np.array([r["grid"]["loc_w4.0_d-100"]["p_yes_after"] for r in rows_subset])
        return bootstrap_auroc_diff(y, drop_blank, drop_loc)

    transient_rows = [r for r in necessity_rows if r["object_name"] in TRANSIENT]
    continuous_rows = [r for r in necessity_rows if r["object_name"] in CONTINUOUS]
    gap_t, ci_t = auroc_gap(transient_rows)
    gap_c, ci_c = auroc_gap(continuous_rows)

    result = {
        "delta_loc_by_class_type": {
            "transient": {"n": len(t_diffs), "mean": float(t_diffs.mean())},
            "continuous": {"n": len(c_diffs), "mean": float(c_diffs.mean())},
            "mann_whitney_p_transient_greater": float(mw_p),
        },
        "blank_minus_localized_auroc_by_class_type": {
            "transient": {"n": len(transient_rows), "point": gap_t, "ci95": list(ci_t)},
            "continuous": {"n": len(continuous_rows), "point": gap_c, "ci95": list(ci_c)},
        },
        "per_class_delta_loc": class_table,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
