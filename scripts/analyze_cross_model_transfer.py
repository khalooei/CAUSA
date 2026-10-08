"""Cross-model transfer of the grounding profile (paper Sec. 5.7).

Standardizes each model's profile features within that model, fits the
logistic router on one model and scores the other without retraining,
and compares against the target model's own best fixed contrast.

Usage:
    python scripts/analyze_cross_model_transfer.py \
        --qwen outputs/qwen2audio/causal_grounding_features.npz \
        --af3 outputs/af3/causal_grounding_features.npz \
        --out outputs/cross_model_transfer.json
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURE_KEYS = ["blank_drop", "clap_sim", "delta_lang", "cs_grid", "delta_loc"]


def load_features(path):
    d = np.load(path)
    y = d["y"]
    # orient every raw signal so larger = more hallucination-like, matching
    # analyze_causal_grounding_profile.py's convention, before standardizing
    X = np.stack([-d[k] for k in FEATURE_KEYS], axis=1)
    return X, y


def bootstrap_auroc_ci(y, score, n_resamples=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0 or y[idx].sum() == n:
            continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    point = roc_auc_score(y, score)
    return point, (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def transfer(name_train, X_train, y_train, name_test, X_test, y_test, best_single_test):
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
    pred_test = clf.predict_proba(scaler.transform(X_test))[:, 1]
    auroc_transfer, ci_transfer = bootstrap_auroc_ci(y_test, pred_test)
    auroc_best_single, ci_best_single = bootstrap_auroc_ci(y_test, best_single_test)
    return {
        "train_on": name_train, "test_on": name_test,
        "n_train": len(y_train), "n_test": len(y_test),
        "transferred_ensemble_auroc": {"point": auroc_transfer, "ci95": list(ci_transfer)},
        "best_single_fixed_contrast_on_test_model_auroc": {"point": auroc_best_single, "ci95": list(ci_best_single)},
        "transfer_minus_best_single": auroc_transfer - auroc_best_single,
        "coefficients": dict(zip(FEATURE_KEYS, clf.coef_[0].tolist())),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen", required=True)
    p.add_argument("--af3", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    Xq, yq = load_features(args.qwen)
    Xa, ya = load_features(args.af3)

    # best single fixed contrast, per model, in its own raw orientation
    best_single_q = Xq[:, FEATURE_KEYS.index("delta_lang")]  # from causal_grounding_profile.json (Qwen)
    best_single_a = Xa[:, FEATURE_KEYS.index("blank_drop")]  # from causal_grounding_profile.json (AF3)

    result = {
        "qwen_to_af3": transfer("qwen2audio", Xq, yq, "af3", Xa, ya, best_single_a),
        "af3_to_qwen": transfer("af3", Xa, ya, "qwen2audio", Xq, yq, best_single_q),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
