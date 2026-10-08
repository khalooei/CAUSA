"""Aggregates an oracle-localization run (paper Table 2).

Reports CLAP-vs-true-window temporal IoU and paired flip-rate /
P(yes)-drop comparisons across the oracle, CLAP and random arms, with
bootstrap 95% confidence intervals on each pairwise difference.

Usage:
    python scripts/analyze_oracle_localization.py \
        --jsonl outputs/oracle_qwen2audio/oracle_localization.jsonl \
        --out outputs/oracle_qwen2audio/report.json
"""
import argparse
import json

import numpy as np


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("skipped"):
                rows.append(row)
    return rows


def paired_bootstrap_diff(a, b, n_resamples=2000, seed=0):
    """95% CI on mean(a) - mean(b) via paired resampling."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    return float(np.mean(a) - np.mean(b)), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = load_rows(args.jsonl)
    n = len(rows)
    if n == 0:
        raise SystemExit(f"no usable rows in {args.jsonl}")

    iou = np.array([r["clap_oracle_iou"] for r in rows])
    flip_oracle = np.array([r["flip_oracle"] for r in rows], dtype=float)
    flip_clap = np.array([r["flip_clap"] for r in rows], dtype=float)
    flip_random = np.array([r["flip_random"] for r in rows], dtype=float)
    drop_oracle = np.array([r["p_yes_mixed"] - r["p_yes_after_oracle"] for r in rows])
    drop_clap = np.array([r["p_yes_mixed"] - r["p_yes_after_clap"] for r in rows])
    drop_random = np.array([r["p_yes_mixed"] - r["p_yes_after_random"] for r in rows])

    def pair(name_a, a, name_b, b):
        diff, ci = paired_bootstrap_diff(a, b)
        return {"mean_diff": diff, "ci95": list(ci), "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}

    report = {
        "n_usable_synthetic_positives": n,
        "localization_accuracy": {
            "mean_clap_oracle_iou": float(iou.mean()),
            "median_clap_oracle_iou": float(np.median(iou)),
            "frac_zero_overlap": float((iou == 0).mean()),
            "frac_iou_over_0.5": float((iou > 0.5).mean()),
        },
        "flip_rate": {
            "oracle": float(flip_oracle.mean()),
            "clap": float(flip_clap.mean()),
            "random": float(flip_random.mean()),
        },
        "mean_p_yes_drop": {
            "oracle": float(drop_oracle.mean()),
            "clap": float(drop_clap.mean()),
            "random": float(drop_random.mean()),
        },
        "paired_flip_rate_diff": {
            "oracle_minus_clap": pair("oracle", flip_oracle, "clap", flip_clap),
            "oracle_minus_random": pair("oracle", flip_oracle, "random", flip_random),
            "clap_minus_random": pair("clap", flip_clap, "random", flip_random),
        },
        "paired_drop_diff": {
            "oracle_minus_clap": pair("oracle", drop_oracle, "clap", drop_clap),
            "oracle_minus_random": pair("oracle", drop_oracle, "random", drop_random),
            "clap_minus_random": pair("clap", drop_clap, "random", drop_random),
        },
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
