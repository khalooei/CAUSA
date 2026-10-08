"""Analysis of the wrong-label CLAP control (supplementary).

Compares P(yes)-drop and flip rate for target-label CLAP, wrong-label
CLAP and random windows (paired tests, bootstrap CIs). Expected ordering
if localization is semantic: target > wrong-label >= random.

Usage:
    python scripts/analyze_wronglabel_control.py \
        --wronglabel outputs/qwen2audio/wronglabel_control.jsonl \
        --out outputs/qwen2audio/wronglabel_analysis.json
"""
import argparse
import json

import numpy as np
from scipy import stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wronglabel", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.wronglabel) if not json.loads(l).get("skipped")]
    n = len(rows)
    print(f"{n} usable rows")

    p_before = np.array([r["p_yes_before"] for r in rows])
    d_target = p_before - np.array([r["p_yes_after_target_clap"] for r in rows])
    d_wrong = p_before - np.array([r["p_yes_after_wronglabel"] for r in rows])
    d_random = p_before - np.array([r["p_yes_after_random"] for r in rows])

    D_target, D_wrong, D_random = d_target.mean(), d_wrong.mean(), d_random.mean()

    def paired_test(a, b, name):
        diff = a - b
        t, p_t = stats.ttest_rel(a, b)
        w, p_w = stats.wilcoxon(diff)
        return {
            "mean_diff": float(diff.mean()),
            "ttest_p": float(p_t),
            "wilcoxon_p": float(p_w),
            "win_rate": float((diff > 0).mean()),
        }

    target_vs_wrong = paired_test(d_target, d_wrong, "target_vs_wrong")
    wrong_vs_random = paired_test(d_wrong, d_random, "wrong_vs_random")
    target_vs_random = paired_test(d_target, d_random, "target_vs_random")

    # bootstrap CI on the three means
    rng = np.random.default_rng(0)
    n_boot = 5000
    boot_target, boot_wrong, boot_random = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_target.append(d_target[idx].mean())
        boot_wrong.append(d_wrong[idx].mean())
        boot_random.append(d_random[idx].mean())
    boot_target, boot_wrong, boot_random = map(np.array, (boot_target, boot_wrong, boot_random))

    # where does wrong-label sit relative to target and random, normalized
    # 0 = matches random, 1 = matches target (same idea as rho_loc)
    denom = D_target - D_random
    rho_wrong = (D_wrong - D_random) / denom if abs(denom) > 1e-9 else None

    result = {
        "n": n,
        "D_target_clap": {"mean": float(D_target), "ci95": [float(np.percentile(boot_target, 2.5)),
                                                              float(np.percentile(boot_target, 97.5))]},
        "D_wrong_label": {"mean": float(D_wrong), "ci95": [float(np.percentile(boot_wrong, 2.5)),
                                                             float(np.percentile(boot_wrong, 97.5))]},
        "D_random": {"mean": float(D_random), "ci95": [float(np.percentile(boot_random, 2.5)),
                                                          float(np.percentile(boot_random, 97.5))]},
        "target_vs_wrong": target_vs_wrong,
        "wrong_vs_random": wrong_vs_random,
        "target_vs_random": target_vs_random,
        "rho_wrong_label": rho_wrong,
        "hypothesis_check": {
            "description": "D_target > D_wrong >= D_random",
            "target_beats_wrong": bool(D_target > D_wrong),
            "wrong_beats_or_matches_random": bool(D_wrong >= D_random or wrong_vs_random["wilcoxon_p"] > 0.05),
        },
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
