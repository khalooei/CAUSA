"""Learned multi-signal decoding (supplementary).

Uses the held-out prediction of the cross-validated grounding-profile
model (`analyze_causal_grounding_profile.py --features-out`) to overrule
the model's least-trusted "yes" answers, and compares the resulting
accuracy against the best hand-designed contrastive-decoding rule.

Usage:
    python scripts/analyze_learned_decoding.py \
        --features outputs/qwen2audio/causal_grounding_features.npz \
        --model-name qwen2audio --out outputs/qwen2audio/learned_decoding.json
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

FEATURE_NAMES = ["blank_drop", "clap_sim", "delta_lang", "cs_grid", "delta_loc"]


def cv_predict_proba(X, y, n_splits=5, seed=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros(len(y))
    coefs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        pred[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        coefs.append(clf.coef_[0])
    return pred, np.array(coefs)


def repeated_cv_accuracy(X, y, n_repeats=20, n_splits=5, threshold=0.5):
    """Matches the paper's existing repeated-5-fold-CV convention
    (analyze_decoding_strengths.py's confidence-gated result) for a
    fair, non-single-split accuracy estimate."""
    accs = []
    for seed in range(n_repeats):
        pred, _ = cv_predict_proba(X, y, n_splits=n_splits, seed=seed)
        corrected_is_hallu = (pred > threshold).astype(int)
        accs.append(accuracy_score(y, corrected_is_hallu))
    return float(np.mean(accs)), float(np.std(accs)), accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = np.load(args.features)
    y = d["y"]  # 1 = hallucination (ground truth "no"), 0 = correct (ground truth "yes")
    raw_accuracy = float(1.0 - y.mean())  # the model's own "yes" is correct on this fraction

    # orient every raw signal so higher = more hallucination-like, matching
    # analyze_causal_grounding_profile.py's convention exactly
    X5 = np.stack([-d["blank_drop"], -d["clap_sim"], -d["delta_lang"],
                    -d["cs_grid"], -d["delta_loc"]], axis=1)

    mean_acc, std_acc, accs = repeated_cv_accuracy(X5, y, n_repeats=20)

    # one seed-0 CV fit, reused for both the per-fold coefficients
    # (interpretability: which signals the learned rule leans on, vs.
    # the two hand-picked by alpha search) and the threshold sweep below
    # (predicted probabilities don't depend on the threshold, so there's
    # no need to re-fit the 5-fold CV once per threshold)
    pred_seed0, coefs = cv_predict_proba(X5, y, seed=0)
    mean_coefs = coefs.mean(axis=0)

    # threshold-sweep robustness check (0.5 is the pre-registered primary,
    # not cherry-picked - report the sweep so that's verifiable, not assumed)
    thresholds = [round(t, 2) for t in np.arange(0.3, 0.71, 0.05)]
    sweep = {t: float(accuracy_score(y, (pred_seed0 > t).astype(int))) for t in thresholds}

    result = {
        "model": args.model_name,
        "n": int(len(y)),
        "raw_accuracy": raw_accuracy,
        "learned_5feature_decoding": {
            "cv_accuracy_mean_20x5fold": mean_acc,
            "cv_accuracy_std_20x5fold": std_acc,
            "delta_vs_raw": mean_acc - raw_accuracy,
            "mean_logistic_coefficients": dict(zip(FEATURE_NAMES, mean_coefs.tolist())),
        },
        "threshold_sweep_seed0": sweep,
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
