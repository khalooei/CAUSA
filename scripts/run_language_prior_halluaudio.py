"""Language-only prior for the HalluAudio object classes (paper Sec. 6).

Same procedure as `run_language_prior.py`, over the object classes that
appear in a HalluAudio run's `necessity.jsonl`.

Usage:
    python scripts/run_language_prior_halluaudio.py --model qwen2audio \
        --necessity outputs/halluaudio_qwen2audio/necessity.jsonl \
        --out outputs/language_prior_halluaudio_qwen2audio.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

LALM_CHOICES = {
    "qwen2audio": ("causa.lalm", "Qwen2AudioLALM", "Qwen2-Audio-7B-Instruct"),
    "af3": ("causa.af3_lalm", "AudioFlamingo3LALM", "NVIDIA Audio Flamingo 3"),
}


def _load_lalm_class(model_key: str):
    import importlib
    module_name, class_name, _ = LALM_CHOICES[model_key]
    return getattr(importlib.import_module(module_name), class_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--necessity", required=True, help="halluaudio necessity.jsonl, to enumerate object classes")
    p.add_argument("--out", required=True)
    p.add_argument("--model", choices=list(LALM_CHOICES.keys()), default="qwen2audio")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.necessity) if not json.loads(l).get("skipped")]
    objects = sorted({r["object_name"] for r in rows})
    print(f"{len(objects)} distinct HalluAudio object classes")

    LALMClass = _load_lalm_class(args.model)
    print(f"loading {LALM_CHOICES[args.model][2]}...")
    lalm = LALMClass()

    results = {}
    for i, obj in enumerate(objects):
        score = lalm.score_yes_no(None, obj)
        results[obj] = {"p_yes_text_only": score.p_yes, "p_no_text_only": score.p_no}
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(objects)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
