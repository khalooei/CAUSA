"""Aggregates pipeline output into the paper's reportable numbers.

Two families of numbers:
  - Hallucination-detection performance (AUROC / AUPRC): can our causal
    necessity score separate real "yes" claims from hallucinated ("yes"
    but ground truth is "no") ones better than the AAD-style blank-audio
    contrast, or than raw CLAP agreement with no intervention at all?
  - PN/PS estimates: the empirical Probability-of-Necessity and
    Probability-of-Sufficiency quantities themselves, as a descriptive/
    calibration statistic (PS in particular should be high, or the whole
    ablation-based signal is suspect - it would mean the pipeline can't
    even detect evidence it inserted itself).
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclasses.dataclass
class DetectionScore:
    method: str
    auroc: float
    auprc: float
    n_positive: int  # n hallucinated ("yes" but ground_truth == "no")
    n_negative: int  # n correct ("yes" and ground_truth == "yes")


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("skipped"):
                continue
            rows.append(row)
    return rows


def hallucination_detection_scores(necessity_rows: list[dict]) -> list[DetectionScore]:
    y = np.array([r["is_hallucination"] for r in necessity_rows], dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("need both hallucinated and non-hallucinated 'yes' claims to score AUROC")

    p_before = np.array([r["p_yes_before"] for r in necessity_rows])
    blank_drop = p_before - np.array([r["p_yes_after_blank"] for r in necessity_rows])
    clap_sim = np.array([r["clap_similarity"] for r in necessity_rows])

    # in every case, a LOWER score under intervention/agreement signals a
    # more likely hallucination, so the hallucination-likelihood score is
    # the negation of the raw grounding evidence.
    candidates = {
        "AAD-style baseline (blank-audio ablation)": -blank_drop,
        "CLAP agreement (correlational, no intervention)": -clap_sim,
    }

    grid_keys = sorted(necessity_rows[0].get("grid", {}).keys()) if necessity_rows else []
    for key in grid_keys:
        drop = p_before - np.array([r["grid"][key]["p_yes_after"] for r in necessity_rows])
        placement = "localized" if key.startswith("loc_") else "random-window control"
        candidates[f"causa grid[{key}] ({placement})"] = -drop

    results = []
    for name, score in candidates.items():
        results.append(DetectionScore(
            method=name,
            auroc=float(roc_auc_score(y, score)),
            auprc=float(average_precision_score(y, score)),
            n_positive=int(y.sum()),
            n_negative=int(len(y) - y.sum()),
        ))
    return results


def localization_quality_check(necessity_rows: list[dict]) -> dict:
    """Direct test of whether CLAP's chosen window matters at all: for
    matched (window_sec, attenuation_db) settings, compares the necessity
    flip rate (on true-positive claims) between the CLAP-localized window
    and a random same-size control window. If localized doesn't clearly
    beat random, the bottleneck is not localization quality - it's that
    the model's claim doesn't depend much on any single small window.
    """
    tp_rows = [r for r in necessity_rows if not r["is_hallucination"] and "grid" in r]
    if not tp_rows:
        return {}
    grid_keys = tp_rows[0]["grid"].keys()
    settings = sorted({k[len("loc_"):] for k in grid_keys if k.startswith("loc_")})

    out = {}
    for setting in settings:
        loc_key, rand_key = f"loc_{setting}", f"rand_{setting}"
        loc_flip = float(np.mean([r["grid"][loc_key]["flip"] for r in tp_rows]))
        rand_flip = float(np.mean([r["grid"][rand_key]["flip"] for r in tp_rows]))
        out[setting] = {
            "localized_flip_rate": loc_flip,
            "random_flip_rate": rand_flip,
            "localized_minus_random": loc_flip - rand_flip,
            "n": len(tp_rows),
        }
    return out


def pn_ps_estimates(necessity_rows: list[dict], sufficiency_rows: list[dict]) -> dict:
    true_positive_rows = [r for r in necessity_rows if not r["is_hallucination"]]
    pn_hat = float(np.mean([r["flip_ablation"] for r in true_positive_rows])) if true_positive_rows else float("nan")
    pn_blank_hat = float(np.mean([r["flip_blank"] for r in true_positive_rows])) if true_positive_rows else float("nan")

    valid_suff = [r for r in sufficiency_rows if r.get("exemplar_available")]
    ps_hat = float(np.mean([r["flip_insertion"] for r in valid_suff])) if valid_suff else float("nan")

    return {
        "PN_hat_localized_ablation": pn_hat,
        "PN_hat_blank_ablation_baseline": pn_blank_hat,
        "PS_hat": ps_hat,
        "n_true_positive_for_PN": len(true_positive_rows),
        "n_valid_for_PS": len(valid_suff),
        "n_sufficiency_no_exemplar": len(sufficiency_rows) - len(valid_suff),
    }


def summarize(necessity_path: str, sufficiency_path: str) -> dict:
    necessity_rows = load_jsonl(necessity_path)
    sufficiency_rows = load_jsonl(sufficiency_path)

    n_hallucinated = sum(1 for r in necessity_rows if r["is_hallucination"])
    n_correct_yes = len(necessity_rows) - n_hallucinated

    out = {
        "n_necessity_examples": len(necessity_rows),
        "n_sufficiency_examples": len(sufficiency_rows),
        "n_hallucinated_yes_claims": n_hallucinated,
        "n_correct_yes_claims": n_correct_yes,
    }
    out["pn_ps"] = pn_ps_estimates(necessity_rows, sufficiency_rows)
    out["localization_quality_check"] = localization_quality_check(necessity_rows)
    try:
        out["detection"] = [dataclasses.asdict(d) for d in hallucination_detection_scores(necessity_rows)]
    except ValueError as e:
        out["detection_error"] = str(e)
    return out
