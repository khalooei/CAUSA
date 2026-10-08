"""Language-only "yes" prior per object class (paper Sec. 3.2, Sec. 5.4).

Asks "Is there a sound of X in the audio?" with no audio content block
at all (the audio encoder is not invoked), once per object class. The
result is the object-conditioned prior used by LangPrior-CD and by the
Delta = P(yes | real) - P(yes | no audio) detector.

Usage:
    python scripts/run_language_prior.py --model qwen2audio --out outputs/language_prior_qwen2audio.json
    python scripts/run_language_prior.py --model af3 --out outputs/language_prior_af3.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causa.data import DATASET_ID, parse_object  # noqa: E402
from datasets import load_dataset  # noqa: E402

LALM_CHOICES = {
    "qwen2audio": ("causa.lalm", "Qwen2AudioLALM", "Qwen2-Audio-7B-Instruct"),
    "af3": ("causa.af3_lalm", "AudioFlamingo3LALM", "NVIDIA Audio Flamingo 3"),
    "qwen25omni": ("causa.qwen25omni_lalm", "Qwen25OmniLALM", "Qwen2.5-Omni-7B"),
}


def _load_lalm_class(model_key: str):
    import importlib
    module_name, class_name, _ = LALM_CHOICES[model_key]
    return getattr(importlib.import_module(module_name), class_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "outputs", "language_prior.json"))
    p.add_argument("--model", choices=list(LALM_CHOICES.keys()), default="qwen2audio")
    args = p.parse_args()

    ds = load_dataset(DATASET_ID, split="test")
    objects = sorted({parse_object(row["query"]) for row in ds})
    print(f"{len(objects)} distinct object classes")

    LALMClass = _load_lalm_class(args.model)
    print(f"loading {LALM_CHOICES[args.model][2]}...")
    lalm = LALMClass()
    # sequential, not batched: score_yes_no_batch shows measurable (up to
    # ~0.03 p_yes) discrepancy vs sequential scoring for a minority of
    # items, likely bf16 batched-matmul numerical instability - not worth
    # the risk for only 53 calls, where sequential is already <1 min.
    results = {}
    for obj in objects:
        score = lalm.score_yes_no(None, obj)
        results[obj] = {"p_yes_text_only": score.p_yes, "p_no_text_only": score.p_no}
        print(f"{obj:25s} p_yes_text_only={score.p_yes:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
