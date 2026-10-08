"""Empirical check of the monotonicity assumption (paper Sec. 3.5).

R* is well defined if the thresholded decision, once flipped to "no" by
a nested removal, never reverts to "yes" under a larger removal. This
checks that condition, and separately whether raw P(yes) is monotone,
on the adaptive-search trajectories (CLAP order and random order).

Usage:
    python scripts/analyze_monotonicity.py \
        --adaptive outputs/qwen2audio/adaptive.jsonl \
        --model-name qwen2audio --out outputs/qwen2audio/monotonicity.json
"""
import argparse
import json

import numpy as np


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def sequence_for(row, order_key):
    order = row[order_key]
    seq = [row["p_yes_before"]] + [s["p_yes"] for s in order["steps"]]
    return seq


def binary_reversal(seq, threshold=0.5):
    """True if the thresholded yes/no decision ever flips back to
    'yes' after having flipped to 'no' earlier in the sequence."""
    decisions = [p > threshold for p in seq]  # True = "yes"
    seen_no = False
    for d in decisions:
        if not d:
            seen_no = True
        elif seen_no and d:
            return True
    return False


TRANSIENT = {"clapping", "laughter", "siren", "glass_breaking", "glass breaking",
             "cat", "crow", "frog"}
CONTINUOUS = {"thunderstorm", "snoring", "car_horn", "car horn", "dog",
              "toilet_flush", "toilet flush"}

EPSILONS = [0.0, 0.01, 0.05]


def analyze_order(rows, order_key, subset_name="all"):
    n = len(rows)
    if n == 0:
        return {"n_examples": 0}
    all_violating_increases = []  # every individual increase, any epsilon=0 step
    max_increases = []
    n_binary_reversal = 0
    step_violation_by_eps = {eps: {"n_increase_steps": 0, "n_total_steps": 0} for eps in EPSILONS}
    example_violation_by_eps = {eps: 0 for eps in EPSILONS}

    for r in rows:
        seq = sequence_for(r, order_key)
        diffs = np.diff(seq)
        for eps in EPSILONS:
            inc = diffs[diffs > eps]
            step_violation_by_eps[eps]["n_increase_steps"] += len(inc)
            step_violation_by_eps[eps]["n_total_steps"] += len(diffs)
            if len(inc) > 0:
                example_violation_by_eps[eps] += 1
        increases = diffs[diffs > 0]
        all_violating_increases.extend(increases.tolist())
        max_increases.append(float(increases.max()) if len(increases) else 0.0)
        if binary_reversal(seq):
            n_binary_reversal += 1

    max_increases = np.array(max_increases)
    all_inc = np.array(all_violating_increases) if all_violating_increases else np.array([0.0])

    result = {
        "n_examples": n,
        "subset": subset_name,
        "continuous_score_check": {
            "description": "stricter than the theory requires -- reported for transparency",
            "by_epsilon": {
                str(eps): {
                    "frac_examples_with_any_increase": example_violation_by_eps[eps] / n,
                    "frac_steps_that_increase": (step_violation_by_eps[eps]["n_increase_steps"]
                                                  / step_violation_by_eps[eps]["n_total_steps"]),
                } for eps in EPSILONS
            },
            "violation_magnitude_epsilon0": {
                "mean": float(all_inc.mean()),
                "median": float(np.median(all_inc)),
                "p95": float(np.percentile(all_inc, 95)),
                "max": float(all_inc.max()),
            },
            "mean_max_increase_per_example": float(max_increases.mean()),
        },
        "binary_decision_reversal_check": {
            "description": "the actual requirement for R* to be well-defined: does the "
                            "yes/no decision ever flip back to yes after flipping to no?",
            "frac_examples_with_reversal": n_binary_reversal / n,
        },
    }
    return result


def class_bucket(object_name):
    name = object_name.lower().replace(" ", "_")
    if name in {t.replace(" ", "_") for t in TRANSIENT}:
        return "transient"
    if name in {c.replace(" ", "_") for c in CONTINUOUS}:
        return "continuous"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adaptive", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = load_jsonl(args.adaptive)

    tp_rows = [r for r in rows if not r["is_hallucination"]]
    hallu_rows = [r for r in rows if r["is_hallucination"]]
    transient_rows = [r for r in rows if class_bucket(r["object_name"]) == "transient"]
    continuous_rows = [r for r in rows if class_bucket(r["object_name"]) == "continuous"]

    result = {
        "model": args.model_name,
        "n": len(rows),
        "clap_order": analyze_order(rows, "clap_order"),
        "random_order": analyze_order(rows, "random_order"),
        "clap_order_true_positive": analyze_order(tp_rows, "clap_order", "true_positive"),
        "clap_order_hallucination": analyze_order(hallu_rows, "clap_order", "hallucination"),
        "clap_order_transient": analyze_order(transient_rows, "clap_order", "transient")
                                  if transient_rows else {"n_examples": 0},
        "clap_order_continuous": analyze_order(continuous_rows, "clap_order", "continuous")
                                    if continuous_rows else {"n_examples": 0},
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
