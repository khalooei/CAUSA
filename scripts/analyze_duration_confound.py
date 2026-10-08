"""Duration-confound analysis (paper Sec. 5.1).

Blank audio removes the whole clip (~10 s) while the localized grid
removes 2-4 s, so their raw AUROCs are not a like-for-like comparison.
This script computes, from an existing `necessity.jsonl` plus each clip's
duration:

  1. a regression of P(yes)-drop on the fraction of the clip removed,
     fitted on random-window cells only (no localization signal), and
     its prediction for blank audio;
  2. localized vs. random-window removal at matched duration (same clip,
     same window size), paired Wilcoxon signed-rank test per grid cell;
  3. residuals of localized and random cells above the duration-only
     line, pooled over all cells (paired t-test).

Usage:
    python scripts/analyze_duration_confound.py \
        --necessity outputs/qwen2audio/necessity.jsonl \
        --audio-root data/audio/beaf_audio \
        --out outputs/qwen2audio/duration_confound.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf
from scipy import stats

from causa.data import load_examples  # noqa: E402
from causa.halluaudio_data import download_and_load as load_halluaudio  # noqa: E402

GRID_CELLS = ["w2.0_d-40", "w2.0_d-100", "w4.0_d-40", "w4.0_d-100"]
WINDOW_SEC = {"w2.0": 2.0, "w4.0": 4.0}


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if not json.loads(l).get("skipped")]


def get_duration(path):
    info = sf.info(path)
    return info.frames / info.samplerate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--necessity", required=True)
    p.add_argument("--audio-root", required=True)
    p.add_argument("--dataset", choices=["object_existence", "halluaudio"], default="object_existence")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = load_jsonl(args.necessity)
    print(f"{len(rows)} necessity rows")

    print("loading benchmark pool to map entry_id -> audio path...")
    if args.dataset == "halluaudio":
        pool = load_halluaudio(args.audio_root, limit=None)
    else:
        pool = load_examples(args.audio_root, limit=None)
    path_by_id = {e.entry_id: e.audio_path for e in pool}

    print("reading clip durations (header-only, no decode)...")
    durations = {}
    missing = 0
    for r in rows:
        eid = r["entry_id"]
        if eid not in path_by_id:
            continue
        path = path_by_id[eid]
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            durations[eid] = get_duration(path)
        except Exception:
            missing += 1
    print(f"got durations for {len(durations)}/{len(rows)} rows ({missing} missing)")

    # --- pooled regression: duration_removed_fraction vs p_yes drop ---
    frac_all, drop_all, cond_all = [], [], []
    loc_vs_rand_matched = {c: {"loc": [], "rand": []} for c in GRID_CELLS}

    for r in rows:
        eid = r["entry_id"]
        if eid not in durations:
            continue
        dur = durations[eid]
        if dur <= 0:
            continue
        p_before = r["p_yes_before"]

        # blank: 100% of clip removed
        frac_all.append(1.0)
        drop_all.append(p_before - r["p_yes_after_blank"])
        cond_all.append("blank")

        for cell in GRID_CELLS:
            wkey = cell.split("_")[0]
            wsec = WINDOW_SEC[wkey]
            frac = min(1.0, wsec / dur)

            loc_p = r["grid"][f"loc_{cell}"]["p_yes_after"]
            rand_p = r["grid"][f"rand_{cell}"]["p_yes_after"]

            frac_all.append(frac); drop_all.append(p_before - loc_p); cond_all.append("loc")
            frac_all.append(frac); drop_all.append(p_before - rand_p); cond_all.append("rand")

            loc_vs_rand_matched[cell]["loc"].append(p_before - loc_p)
            loc_vs_rand_matched[cell]["rand"].append(p_before - rand_p)

    frac_all = np.array(frac_all); drop_all = np.array(drop_all)
    cond_all = np.array(cond_all)

    # pooled OLS: drop ~ a*frac + b, fit on rand+blank only (the
    # "duration alone" model with zero localization signal -- rand cells
    # carry no location signal by construction, blank carries none either)
    mask_fit = (cond_all == "rand") | (cond_all == "blank")
    slope, intercept, r_value, p_value, std_err = stats.linregress(frac_all[mask_fit], drop_all[mask_fit])

    predicted_at_1 = slope * 1.0 + intercept
    actual_blank_mean = float(drop_all[cond_all == "blank"].mean())

    # where do loc points sit relative to the duration-alone fitted line?
    mask_loc = cond_all == "loc"
    loc_residuals = drop_all[mask_loc] - (slope * frac_all[mask_loc] + intercept)
    mask_rand = cond_all == "rand"
    rand_residuals = drop_all[mask_rand] - (slope * frac_all[mask_rand] + intercept)

    result = {
        "n_examples_with_duration": len(durations),
        "n_missing_duration": missing,
        "clip_duration_stats": {
            "mean": float(np.mean(list(durations.values()))),
            "median": float(np.median(list(durations.values()))),
            "min": float(np.min(list(durations.values()))),
            "max": float(np.max(list(durations.values()))),
        },
        "duration_alone_regression_fit_on_rand_and_blank": {
            "description": "p_yes_drop ~ slope*duration_fraction_removed + intercept, "
                            "fit only on rand_* cells + blank (zero localization signal by construction)",
            "slope": float(slope), "intercept": float(intercept),
            "r_value": float(r_value), "p_value": float(p_value),
            "predicted_drop_at_100pct_removed": float(predicted_at_1),
            "actual_blank_mean_drop": actual_blank_mean,
            "blank_minus_predicted": actual_blank_mean - float(predicted_at_1),
        },
        "localization_residual_beyond_duration": {
            "description": "does loc_* sit above the duration-alone fitted line more than rand_* does? "
                            "(mean residual = actual drop - line's predicted drop at that duration)",
            "mean_residual_loc": float(loc_residuals.mean()),
            "mean_residual_rand": float(rand_residuals.mean()),
            "loc_minus_rand_residual": float(loc_residuals.mean() - rand_residuals.mean()),
            "paired_ttest": None,
        },
        "loc_vs_rand_matched_duration_per_cell": {},
    }

    # paired test on residuals (same length only if 1:1 aligned per example;
    # loc_residuals/rand_residuals are per (example,cell) so already paired
    # in construction order -- confirm same length before pairing)
    if len(loc_residuals) == len(rand_residuals):
        t, pval = stats.ttest_rel(loc_residuals, rand_residuals)
        w, pval_w = stats.wilcoxon(loc_residuals, rand_residuals)
        result["localization_residual_beyond_duration"]["paired_ttest"] = {
            "t_stat": float(t), "p_value": float(pval)
        }
        result["localization_residual_beyond_duration"]["paired_wilcoxon"] = {
            "w_stat": float(w), "p_value": float(pval_w)
        }

    for cell in GRID_CELLS:
        loc_vals = np.array(loc_vs_rand_matched[cell]["loc"])
        rand_vals = np.array(loc_vs_rand_matched[cell]["rand"])
        t, pval = stats.ttest_rel(loc_vals, rand_vals)
        w, pval_w = stats.wilcoxon(loc_vals, rand_vals)
        result["loc_vs_rand_matched_duration_per_cell"][cell] = {
            "n": len(loc_vals),
            "mean_drop_loc": float(loc_vals.mean()),
            "mean_drop_rand": float(rand_vals.mean()),
            "loc_minus_rand": float(loc_vals.mean() - rand_vals.mean()),
            "paired_ttest_p": float(pval),
            "paired_wilcoxon_p": float(pval_w),
        }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
