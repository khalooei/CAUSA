"""Real-unrelated-audio substitution control (paper Sec. 3.2, Sec. 5.4).

Replaces each affirmed clip with a different real recording that is a
verified negative for the queried object and re-scores the claim: a
black-box version of the substitution test of Cho et al., "Tracing Audio
Grounding and Answer Selection in Audio LLMs" (arXiv:2609.04637).
Eligible entries are read from the main run's `necessity.jsonl`.

Usage:
    python scripts/run_substitution_test.py \
        --audio-root data/audio/beaf_audio --main-out outputs/qwen2audio \
        --out outputs/qwen2audio/substitution.jsonl --n 5000 --seed 42
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causa.ablation import substitute_unrelated  # noqa: E402
from causa.data import load_examples, load_audio  # noqa: E402
from causa.exemplars import UnrelatedAudioBank  # noqa: E402
from causa.lalm import SAMPLE_RATE as LALM_SR  # noqa: E402
from causa.lalm import Qwen2AudioLALM  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-root", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--exemplar-pool-n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--main-out", required=True, help="dir with the main run's necessity.jsonl")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    necessity_path = os.path.join(args.main_out, "necessity.jsonl")
    eligible_ids = set()
    by_id = {}
    with open(necessity_path) as f:
        for line in f:
            row = json.loads(line)
            if not row.get("skipped"):
                eligible_ids.add(row["entry_id"])
                by_id[row["entry_id"]] = row
    print(f"{len(eligible_ids)} examples eligible (had a 'yes' claim in the main run)")

    pool_n = max(args.n, args.exemplar_pool_n)
    pool = load_examples(args.audio_root, limit=pool_n, shuffle_seed=args.seed)
    examples = [e for e in pool[: args.n] if e.entry_id in eligible_ids]
    print(f"{len(examples)} examples to run substitution test on")

    unrelated_bank = UnrelatedAudioBank(pool, seed=args.seed)
    lalm = Qwen2AudioLALM()

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                done.add(json.loads(line)["entry_id"])

    # sequential, not batched: score_yes_no_batch shows measurable (up to
    # ~0.03 p_yes) discrepancy vs sequential scoring for a minority of
    # items, likely bf16 batched-matmul numerical instability - not worth
    # the risk for a number that goes straight into the paper's results.
    todo = [ex for ex in examples if ex.entry_id not in done]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as fout:
        for i, ex in enumerate(todo):
            unrelated_ex = unrelated_bank.get(ex.object_name, exclude_audio_path=ex.audio_path)
            if unrelated_ex is None:
                row = {"entry_id": ex.entry_id, "skipped": True, "reason": "no unrelated clip available"}
            else:
                audio_lalm = load_audio(ex.audio_path, sr=LALM_SR)
                unrelated_audio = load_audio(unrelated_ex.audio_path, sr=LALM_SR)
                substituted = substitute_unrelated(audio_lalm, unrelated_audio)
                score = lalm.score_yes_no(substituted, ex.object_name)
                row = {
                    "entry_id": ex.entry_id,
                    "object_name": ex.object_name,
                    "is_hallucination": by_id[ex.entry_id]["is_hallucination"],
                    "p_yes_before": by_id[ex.entry_id]["p_yes_before"],
                    "p_yes_after_substitution": score.p_yes,
                    "flip_substitution": score.claim == "no",
                }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(todo)}")

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
