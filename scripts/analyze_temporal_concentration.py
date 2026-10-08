"""Temporal concentration of evidence vs. localization advantage (supplementary).

Recomputes CLAP's per-window relevance curve for every claim (CLAP only,
no LALM calls), summarizes how concentrated it is (normalized entropy,
Gini coefficient), and regresses each claim's Delta_loc on it with
class-clustered standard errors. A continuous version of the
transient-vs-continuous split in Sec. 5.2; descriptive, not causal.

Usage:
    python scripts/analyze_temporal_concentration.py \
        --necessity outputs/qwen2audio/necessity.jsonl \
        --adaptive outputs/qwen2audio/adaptive.jsonl \
        --audio-root data/audio/beaf_audio \
        --model-name qwen2audio --out outputs/qwen2audio/temporal_concentration.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from causa.clap_localizer import ClapLocalizer, SAMPLE_RATE  # noqa: E402
from causa.data import load_examples  # noqa: E402
from causa.halluaudio_data import download_and_load as load_halluaudio  # noqa: E402


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if not json.loads(l).get("skipped")]


def normalized_entropy(scores):
    s = np.array(scores, dtype=np.float64)
    s = s - s.min() + 1e-6  # shift non-negative; relative shape is what matters for concentration
    q = s / s.sum()
    H = -np.sum(q * np.log(q + 1e-12))
    H_max = np.log(len(s))
    return float(H / H_max) if H_max > 0 else 0.0


def gini(scores):
    s = np.array(scores, dtype=np.float64)
    s = s - s.min() + 1e-6
    s = np.sort(s)
    n = len(s)
    cum = np.cumsum(s)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--necessity", required=True)
    p.add_argument("--adaptive", required=True)
    p.add_argument("--audio-root", required=True)
    p.add_argument("--dataset", choices=["object_existence", "halluaudio"], default="object_existence")
    p.add_argument("--model-name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="cap examples for a fast smoke test")
    args = p.parse_args()

    necessity_rows = load_jsonl(args.necessity)
    adaptive_rows = {r["entry_id"]: r for r in load_jsonl(args.adaptive)}
    if args.limit:
        necessity_rows = necessity_rows[: args.limit]
    print(f"{len(necessity_rows)} necessity rows, {len(adaptive_rows)} adaptive rows")

    print("loading benchmark pool to map entry_id -> audio path...")
    if args.dataset == "halluaudio":
        pool = load_halluaudio(args.audio_root, limit=None)
    else:
        pool = load_examples(args.audio_root, limit=None)
    path_by_id = {e.entry_id: e.audio_path for e in pool}

    print("loading CLAP...")
    clap = ClapLocalizer()

    import librosa

    results = []
    n_missing_audio = 0
    n_missing_adaptive = 0
    for i, r in enumerate(necessity_rows):
        eid = r["entry_id"]
        if eid not in adaptive_rows:
            n_missing_adaptive += 1
            continue
        path = path_by_id.get(eid)
        if path is None or not os.path.exists(path):
            n_missing_audio += 1
            continue
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            n_missing_audio += 1
            continue

        loc = clap.localize(audio, SAMPLE_RATE, r["object_name"])
        if len(loc.window_scores) < 2:
            continue  # can't compute concentration on a single window

        H = normalized_entropy(loc.window_scores)
        G = gini(loc.window_scores)

        a = adaptive_rows[eid]
        delta_loc = a["random_order"]["necessary_fraction"] - a["clap_order"]["necessary_fraction"]

        results.append({
            "entry_id": eid,
            "object_name": r["object_name"],
            "is_hallucination": r["is_hallucination"],
            "n_windows": len(loc.window_scores),
            "entropy": H,
            "gini": G,
            "delta_loc": delta_loc,
        })

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(necessity_rows)} processed, {len(results)} usable")

    print(f"usable examples: {len(results)} (missing_audio={n_missing_audio}, "
          f"missing_adaptive={n_missing_adaptive})")

    entropy = np.array([r["entropy"] for r in results])
    delta = np.array([r["delta_loc"] for r in results])
    gini_arr = np.array([r["gini"] for r in results])
    objects = np.array([r["object_name"] for r in results])

    # OLS with class-clustered robust SEs (cluster-robust sandwich estimator)
    def ols_clustered(x, y, clusters):
        X = np.column_stack([np.ones_like(x), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        XtX_inv = np.linalg.inv(X.T @ X)
        meat = np.zeros((2, 2))
        for c in np.unique(clusters):
            idx = clusters == c
            Xc = X[idx]
            uc = resid[idx]
            score = Xc.T @ uc
            meat += np.outer(score, score)
        n_clusters = len(np.unique(clusters))
        correction = n_clusters / (n_clusters - 1) if n_clusters > 1 else 1.0
        vcov = correction * XtX_inv @ meat @ XtX_inv
        se = np.sqrt(np.diag(vcov))
        return beta, se

    beta, se = ols_clustered(entropy, delta, objects)
    beta_g, se_g = ols_clustered(gini_arr, delta, objects)

    from scipy import stats as scipy_stats
    t_entropy = beta[1] / se[1]
    p_entropy = 2 * (1 - scipy_stats.norm.cdf(abs(t_entropy)))
    t_gini = beta_g[1] / se_g[1]
    p_gini = 2 * (1 - scipy_stats.norm.cdf(abs(t_gini)))

    result = {
        "model": args.model_name,
        "n": len(results),
        "regression_delta_loc_on_entropy": {
            "description": "delta_loc ~ beta0 + beta1*normalized_entropy, "
                            "class(object_name)-clustered robust SE. beta1<0 supported "
                            "hypothesis: more concentrated evidence (lower entropy) -> "
                            "larger localization advantage.",
            "beta0": float(beta[0]), "beta1": float(beta[1]),
            "se_beta1_clustered": float(se[1]),
            "t_stat": float(t_entropy), "p_value": float(p_entropy),
        },
        "regression_delta_loc_on_gini": {
            "description": "same, using Gini concentration instead of entropy "
                            "(higher Gini = more concentrated); beta1>0 supports the hypothesis",
            "beta0": float(beta_g[0]), "beta1": float(beta_g[1]),
            "se_beta1_clustered": float(se_g[1]),
            "t_stat": float(t_gini), "p_value": float(p_gini),
        },
        "descriptive": {
            "mean_entropy": float(entropy.mean()), "std_entropy": float(entropy.std()),
            "mean_delta_loc": float(delta.mean()),
        },
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    with open(args.out.replace(".json", "_raw.json"), "w") as f:
        json.dump(results, f)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
