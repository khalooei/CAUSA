"""Wrong-label CLAP control (supplementary).

Tests whether CLAP localization finds evidence for the *claimed* object
or merely salient audio: localizes with a different, reproducibly sampled
object label, ablates that window (default 4 s / -100 dB) and re-asks the
original question about the true object. Target-label and random-window
results are reused from `necessity.jsonl`.

Usage:
    python scripts/run_wronglabel_control.py \
        --necessity outputs/qwen2audio/necessity.jsonl \
        --audio-root data/audio/beaf_audio --n 1500 \
        --out outputs/qwen2audio/wronglabel_control.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from causa.ablation import spectral_gate_remove  # noqa: E402
from causa.clap_localizer import ClapLocalizer  # noqa: E402
from causa.clap_localizer import SAMPLE_RATE as CLAP_SR  # noqa: E402
from causa.data import load_examples  # noqa: E402
from causa.lalm import SAMPLE_RATE as LALM_SR  # noqa: E402
from causa.lalm import Qwen2AudioLALM  # noqa: E402


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--necessity", required=True)
    p.add_argument("--audio-root", required=True)
    p.add_argument("--n", type=int, default=1500, help="number of examples to run (first N by "
                    "necessity.jsonl order, a fixed deterministic subset)")
    p.add_argument("--window-sec", type=float, default=4.0)
    p.add_argument("--atten-db", type=float, default=-100.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    necessity_rows = [json.loads(l) for l in open(args.necessity) if not json.loads(l).get("skipped")]
    necessity_rows = necessity_rows[: args.n]
    print(f"{len(necessity_rows)} target examples")

    object_vocab = sorted({r["object_name"] for r in necessity_rows})
    print(f"{len(object_vocab)} object classes in vocabulary")

    done_ids = {r["entry_id"] for r in load_jsonl(args.out)}
    todo = [r for r in necessity_rows if r["entry_id"] not in done_ids]
    print(f"{len(done_ids)} already done, {len(todo)} remaining")
    if not todo:
        print("nothing to do")
        return

    print("loading benchmark pool for audio paths...")
    pool = load_examples(args.audio_root, limit=None)
    path_by_id = {e.entry_id: e.audio_path for e in pool}

    print("loading CLAP...")
    clap = ClapLocalizer()
    print("loading Qwen2-Audio...")
    lalm = Qwen2AudioLALM()

    import librosa

    with open(args.out, "a") as fout:
        for i, r in enumerate(todo):
            eid = r["entry_id"]
            true_object = r["object_name"]
            path = path_by_id.get(eid)
            if path is None or not os.path.exists(path):
                fout.write(json.dumps({"entry_id": eid, "skipped": True, "reason": "missing_audio"}) + "\n")
                fout.flush()
                continue

            # deterministic wrong-label sample: seed on entry_id so reruns
            # (e.g. after a resume) pick the identical wrong label every time
            rng = np.random.default_rng(eid)
            candidates = [o for o in object_vocab if o != true_object]
            wrong_object = candidates[rng.integers(0, len(candidates))]

            try:
                audio_clap = librosa.load(path, sr=CLAP_SR, mono=True)[0]
                audio_lalm = librosa.load(path, sr=LALM_SR, mono=True)[0]
            except Exception as e:
                fout.write(json.dumps({"entry_id": eid, "skipped": True, "reason": f"load_error:{e}"}) + "\n")
                fout.flush()
                continue

            wrong_loc = clap.localize(audio_clap, CLAP_SR, wrong_object, window_sec=args.window_sec)
            ablated = spectral_gate_remove(audio_lalm, LALM_SR, wrong_loc.t_start, wrong_loc.t_end,
                                            attenuation_db=args.atten_db)
            # re-query about the TRUE object, not the wrong one -- the
            # question the model originally affirmed is unchanged
            score_after = lalm.score_yes_no(ablated, true_object)

            row = {
                "entry_id": eid,
                "true_object": true_object,
                "wrong_object": wrong_object,
                "wrong_label_t_start": wrong_loc.t_start,
                "wrong_label_t_end": wrong_loc.t_end,
                "p_yes_before": r["p_yes_before"],
                "p_yes_after_wronglabel": score_after.p_yes,
                "flip_wronglabel": score_after.claim == "no",
                "p_yes_after_target_clap": r["grid"]["loc_w4.0_d-100"]["p_yes_after"],
                "p_yes_after_random": r["grid"]["rand_w4.0_d-100"]["p_yes_after"],
                "is_hallucination": r["is_hallucination"],
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(todo)} done")

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
