"""Adaptive (graded) necessity search (paper Sec. 3.5, Sec. 5.2).

For every affirmative claim of a completed main run, removes the top-k
CLAP-ranked windows at k = 1, 2, 4, 7, 11, 17 (and the same number of
windows in random order) and records P(yes) after each step, giving the
minimal necessary removal fraction R* and the localization efficiency
Delta_loc. Eligibility and the unablated P(yes) are read from the main
run's `necessity.jsonl`, so only the new interventions cost GPU time.

Usage:
    python scripts/run_adaptive_necessity.py --model qwen2audio \
        --audio-root data/audio/beaf_audio --main-out outputs/qwen2audio \
        --out outputs/qwen2audio/adaptive.jsonl --seed 42
"""
import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causa.adaptive_necessity import get_windows, search  # noqa: E402
from causa.clap_localizer import ClapLocalizer  # noqa: E402
from causa.clap_localizer import SAMPLE_RATE as CLAP_SR  # noqa: E402
from causa.data import load_audio  # noqa: E402
from causa.lalm import SAMPLE_RATE as LALM_SR  # noqa: E402

import numpy as np  # noqa: E402

LALM_CHOICES = {
    "qwen2audio": ("causa.lalm", "Qwen2AudioLALM"),
    "af3": ("causa.af3_lalm", "AudioFlamingo3LALM"),
}


def _load_lalm_class(model_key: str):
    import importlib
    module_name, class_name = LALM_CHOICES[model_key]
    return getattr(importlib.import_module(module_name), class_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(LALM_CHOICES.keys()), required=True)
    p.add_argument("--audio-root", required=True)
    p.add_argument("--main-out", required=True, help="dir with the main run's necessity.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    necessity_path = os.path.join(args.main_out, "necessity.jsonl")
    rows = []
    with open(necessity_path) as f:
        for line in f:
            row = json.loads(line)
            if not row.get("skipped"):
                rows.append(row)
    print(f"{len(rows)} eligible examples from {necessity_path}")

    # need the original audio_path per entry_id - reconstruct via the same
    # dataset loader/seed the main run used, matching entry_id to Example.
    # Must load the FULL pool (limit=None): entry_id is a dataset-native
    # column, not a position in the shuffled order, so any limit smaller
    # than the pool truncates a random subset of it. A limit heuristic
    # keyed off len(rows) silently dropped 560/1192 AF3 examples as "not
    # found in reloaded pool" because AF3's eligible set (1192) is a small
    # fraction of its much larger full pool (10800) - found by comparing
    # against causal_grounding_profile.json's n_dropped field.
    from causa.data import load_examples
    pool = load_examples(args.audio_root, limit=None, shuffle_seed=args.seed)
    by_id = {ex.entry_id: ex for ex in pool}

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                done.add(json.loads(line)["entry_id"])

    clap = ClapLocalizer()
    LALMClass = _load_lalm_class(args.model)
    print(f"loading {args.model}...")
    lalm = LALMClass()
    rng = np.random.default_rng(args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_done_this_run = 0
    with open(args.out, "a") as fout:
        for i, row in enumerate(rows):
            entry_id = row["entry_id"]
            if entry_id in done:
                continue
            ex = by_id.get(entry_id)
            if ex is None:
                fout.write(json.dumps({"entry_id": entry_id, "skipped": True,
                                        "reason": "entry not found in reloaded pool"}) + "\n")
                continue

            audio_lalm = load_audio(ex.audio_path, sr=LALM_SR)
            audio_clap = load_audio(ex.audio_path, sr=CLAP_SR)
            duration = len(audio_lalm) / LALM_SR
            window_starts, window_scores = get_windows(clap, audio_clap, ex.object_name)

            clap_result = search(lalm, audio_lalm, ex.object_name, duration,
                                  window_starts, window_scores, "clap", rng)
            rand_result = search(lalm, audio_lalm, ex.object_name, duration,
                                  window_starts, window_scores, "random", rng)

            out_row = {
                "entry_id": entry_id,
                "object_name": ex.object_name,
                "is_hallucination": row["is_hallucination"],
                "p_yes_before": row["p_yes_before"],
                "clap_order": dataclasses.asdict(clap_result),
                "random_order": dataclasses.asdict(rand_result),
            }
            fout.write(json.dumps(out_row) + "\n")
            fout.flush()
            n_done_this_run += 1
            if n_done_this_run % 25 == 0:
                print(f"{i + 1}/{len(rows)} (this run: {n_done_this_run})")

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
