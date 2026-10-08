"""Oracle-localization control (paper Sec. 3.3, Sec. 5.3, Table 2).

Inserts real exemplars at a known, recorded offset into negative-control
clips, keeps the mixtures the model affirms, and ablates the true
insertion window, CLAP's top window and a random window of the same size.
See `causa/oracle_localization.py`.

Usage:
    python scripts/run_oracle_localization.py \
        --audio-root data/audio/beaf_audio --n 800 \
        --out outputs/oracle_qwen2audio --seed 42
"""
import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causa.clap_localizer import ClapLocalizer  # noqa: E402
from causa.data import load_examples  # noqa: E402
from causa.exemplars import ExemplarBank  # noqa: E402
from causa.oracle_localization import OracleLocalizationPipeline  # noqa: E402

LALM_CHOICES = {
    "qwen2audio": ("causa.lalm", "Qwen2AudioLALM", "Qwen2-Audio-7B-Instruct"),
    "af3": ("causa.af3_lalm", "AudioFlamingo3LALM", "NVIDIA Audio Flamingo 3"),
}


def _load_lalm_class(model_key: str):
    import importlib
    module_name, class_name, _ = LALM_CHOICES[model_key]
    return getattr(importlib.import_module(module_name), class_name)


def _load_lalm_sr(model_key: str) -> int:
    import importlib
    module_name, _, _ = LALM_CHOICES[model_key]
    return getattr(importlib.import_module(module_name), "SAMPLE_RATE")


def _load_done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["entry_id"])
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-root", required=True)
    p.add_argument("--n", type=int, default=400,
                   help="number of negative-control (ground_truth=='no') rows to scan for candidates")
    p.add_argument("--out", required=True)
    p.add_argument("--exemplar-pool-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=list(LALM_CHOICES.keys()), default="qwen2audio")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "oracle_localization.jsonl")

    print("loading benchmark examples...")
    pool_n = max(args.n * 3, args.exemplar_pool_n)  # oversample: only ground_truth=='no' rows are candidates
    pool = load_examples(args.audio_root, limit=pool_n, shuffle_seed=args.seed)
    negatives = [ex for ex in pool if ex.ground_truth == "no"][: args.n]
    print(f"{len(negatives)} negative-control candidates (from {len(pool)} scanned rows)")

    print("loading CLAP...")
    clap = ClapLocalizer()
    LALMClass = _load_lalm_class(args.model)
    print(f"loading {LALM_CHOICES[args.model][2]}...")
    lalm = LALMClass()

    print(f"building exemplar bank from {len(pool)} rows...")
    bank = ExemplarBank(pool, clap, seed=args.seed)

    lalm_sr = _load_lalm_sr(args.model)
    pipeline = OracleLocalizationPipeline(lalm, clap, bank, lalm_sr=lalm_sr, seed=args.seed)

    done = _load_done_ids(out_path)
    n_done_this_run = 0
    n_usable = len(done)
    with open(out_path, "a") as fout:
        for i, ex in enumerate(negatives):
            if ex.entry_id in done:
                continue
            r = pipeline.run(ex)
            row = dataclasses.asdict(r) if r is not None else {"entry_id": ex.entry_id, "skipped": True}
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n_done_this_run += 1
            if r is not None:
                n_usable += 1
            if n_done_this_run % 25 == 0:
                print(f"{i + 1}/{len(negatives)} scanned this run, {n_usable} usable synthetic positives so far")

    print(f"wrote {out_path} ({n_usable} usable synthetic positives)")


if __name__ == "__main__":
    main()
