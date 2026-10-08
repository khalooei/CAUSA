"""Matched-removal-budget localization curve (supplementary).

Interpolates each claim's adaptive-search trajectory onto a common grid
of removed fraction r, separately for CLAP order and random order, and
reports G(r) = D_loc(r) - D_rand(r) and its integral (integrated
localization advantage), with bootstrap CIs. No new model calls.

Usage:
    python scripts/analyze_matched_budget_curve.py \
        --adaptive outputs/qwen2audio/adaptive.jsonl \
        --model-name qwen2audio --out outputs/qwen2audio/matched_budget_curve.json
"""
import argparse
import json

import numpy as np
from scipy import stats

R_GRID = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def drop_curve_on_grid(row, order_key, r_grid):
    """Interpolate this example's (fraction, drop) trajectory onto r_grid.
    fraction=0 -> drop=0 (no removal, no change) is the implicit anchor."""
    order = row[order_key]
    fracs = [0.0] + [s["fraction"] for s in order["steps"]]
    p_before = row["p_yes_before"]
    drops = [0.0] + [p_before - s["p_yes"] for s in order["steps"]]
    # fracs is non-decreasing by construction (cumulative removal);
    # np.interp requires it, and clips r beyond the last checkpoint to
    # the final (full-coverage) drop rather than extrapolating
    return np.interp(r_grid, fracs, drops)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adaptive", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = load_jsonl(args.adaptive)
    n = len(rows)

    loc_curves = np.array([drop_curve_on_grid(r, "clap_order", R_GRID) for r in rows])
    rand_curves = np.array([drop_curve_on_grid(r, "random_order", R_GRID) for r in rows])

    D_loc = loc_curves.mean(axis=0)
    D_rand = rand_curves.mean(axis=0)
    G = D_loc - D_rand

    # paired bootstrap CI on G(r) at each grid point, resampling examples
    rng = np.random.default_rng(0)
    n_boot = 2000
    boot_G = np.zeros((n_boot, len(R_GRID)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_G[b] = loc_curves[idx].mean(axis=0) - rand_curves[idx].mean(axis=0)
    ci_lo = np.percentile(boot_G, 2.5, axis=0)
    ci_hi = np.percentile(boot_G, 97.5, axis=0)

    # paired significance test at each r (Wilcoxon on per-example loc-rand difference)
    pvals = []
    for j in range(len(R_GRID)):
        diffs = loc_curves[:, j] - rand_curves[:, j]
        if np.allclose(diffs, 0):
            pvals.append(1.0)
        else:
            _, pv = stats.wilcoxon(diffs)
            pvals.append(float(pv))

    # Integrated Localization Advantage: trapezoidal integral of G(r) over [0,1]
    # (r=0 anchor is G(0)=0 by construction, included for the integral)
    r_full = np.concatenate([[0.0], R_GRID])
    G_full = np.concatenate([[0.0], G])
    ila = float(np.trapezoid(G_full, r_full))

    boot_ila = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        gb = loc_curves[idx].mean(axis=0) - rand_curves[idx].mean(axis=0)
        gb_full = np.concatenate([[0.0], gb])
        boot_ila.append(float(np.trapezoid(gb_full, r_full)))
    boot_ila = np.array(boot_ila)

    result = {
        "model": args.model_name,
        "n": n,
        "r_grid": R_GRID.tolist(),
        "D_loc": D_loc.tolist(),
        "D_rand": D_rand.tolist(),
        "G": G.tolist(),
        "G_ci95_lo": ci_lo.tolist(),
        "G_ci95_hi": ci_hi.tolist(),
        "G_wilcoxon_p": pvals,
        "integrated_localization_advantage": {
            "point": ila,
            "ci95": [float(np.percentile(boot_ila, 2.5)), float(np.percentile(boot_ila, 97.5))],
        },
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ("D_loc", "D_rand")}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
