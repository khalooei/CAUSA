"""Paraphrase-averaged language prior (paper Sec. 5.5).

Queries five paraphrases of the presence question per object class
(no audio block) and averages P(yes), giving a lower-variance prior for
the confidence-gated decoding rule.

Usage:
    python scripts/run_language_prior_ensemble.py --model qwen2audio \
        --out outputs/language_prior_ensemble_qwen2audio.json
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
}

QUESTION_TEMPLATES = [
    "Is there a sound of {object} in the audio?",
    "Does the audio contain a sound of {object}?",
    "Can you hear {object} in the recording?",
    "Is {object} present in the audio?",
    "Does the recording include the sound of {object}?",
]


def _load_lalm_class(model_key: str):
    import importlib
    module_name, class_name, _ = LALM_CHOICES[model_key]
    return getattr(importlib.import_module(module_name), class_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model", choices=list(LALM_CHOICES.keys()), default="qwen2audio")
    args = p.parse_args()

    ds = load_dataset(DATASET_ID, split="test")
    objects = sorted({parse_object(row["query"]) for row in ds})
    print(f"{len(objects)} distinct object classes x {len(QUESTION_TEMPLATES)} phrasings")

    LALMClass = _load_lalm_class(args.model)
    print(f"loading {LALM_CHOICES[args.model][2]}...")
    lalm = LALMClass()

    results = {}
    for i, obj in enumerate(objects):
        p_yes_list = []
        for template in QUESTION_TEMPLATES:
            q = template.format(object=obj)
            score = lalm.score_yes_no(None, obj, question=q)
            p_yes_list.append(score.p_yes)
        results[obj] = {
            "p_yes_text_only": sum(p_yes_list) / len(p_yes_list),
            "p_yes_per_phrasing": p_yes_list,
        }
        print(f"{obj:25s} mean p_yes={results[obj]['p_yes_text_only']:.3f}  "
              f"range=[{min(p_yes_list):.3f},{max(p_yes_list):.3f}]")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
