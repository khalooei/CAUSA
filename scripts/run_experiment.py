"""Main causal-grounding experiment (paper Sec. 3.1-3.2, Table 1, Table 4).

For every benchmark row the LALM answers "yes" to, runs the necessity
grid (CLAP-localized vs. random-window spectral gating at 2 s / 4 s
windows and -40 / -100 dB), the AAD-style blank-audio contrast and the
whole-clip CLAP-agreement baseline. For rows the model correctly answers
"no", runs the sufficiency test (exemplar insertion). Writes resumable
`necessity.jsonl` / `sufficiency.jsonl` and an aggregate `report.json`
(AUROC / AUPRC, PN / PS estimates).

Usage:
    python scripts/run_experiment.py --model qwen2audio \
        --audio-root data/audio/beaf_audio --n 5000 --out outputs/qwen2audio --seed 42
    python scripts/run_experiment.py --model af3 \
        --audio-root data/audio/beaf_audio --n 10800 --out outputs/af3 --seed 42
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causa.clap_localizer import ClapLocalizer  # noqa: E402
from causa.data import load_examples  # noqa: E402
from causa.exemplars import ExemplarBank  # noqa: E402
from causa.halluaudio_data import download_and_load as load_halluaudio  # noqa: E402
from causa.metrics import summarize  # noqa: E402
from causa.pipeline import GroundingPipeline, run_and_dump  # noqa: E402

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
    p.add_argument("--audio-root", required=True, help="dir holding extracted BEAF wav files")
    p.add_argument("--n", type=int, default=200, help="number of benchmark rows to run")
    p.add_argument("--out", required=True, help="output directory for jsonl + report")
    p.add_argument("--exemplar-pool-n", type=int, default=1000,
                   help="rows scanned to build the per-object exemplar bank")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=list(LALM_CHOICES.keys()), default="qwen2audio",
                   help="which LALM to run the pipeline against")
    p.add_argument("--dataset", choices=["object_existence", "halluaudio"], default="object_existence",
                   help="object_existence: kuanhuggingface benchmark (default). "
                        "halluaudio: newer, more adversarial benchmark (arXiv:2604.19300) - "
                        "run scripts/download_halluaudio.py first.")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    necessity_path = os.path.join(args.out, "necessity.jsonl")
    sufficiency_path = os.path.join(args.out, "sufficiency.jsonl")
    report_path = os.path.join(args.out, "report.json")

    print("loading benchmark examples...")
    pool_n = max(args.n, args.exemplar_pool_n)
    if args.dataset == "halluaudio":
        exemplar_pool = load_halluaudio(args.audio_root, limit=pool_n, shuffle_seed=args.seed)
    else:
        exemplar_pool = load_examples(args.audio_root, limit=pool_n, shuffle_seed=args.seed)
    examples = exemplar_pool[: args.n]

    print("loading CLAP...")
    clap = ClapLocalizer()
    LALMClass = _load_lalm_class(args.model)
    print(f"loading {LALM_CHOICES[args.model][2]}...")
    lalm = LALMClass()

    print(f"building exemplar bank from {len(exemplar_pool)} rows...")
    bank = ExemplarBank(exemplar_pool, clap, seed=args.seed)

    pipeline = GroundingPipeline(lalm, clap, bank, seed=args.seed)

    print(f"running pipeline over {len(examples)} examples "
          f"(resumable at {necessity_path} / {sufficiency_path})...")
    run_and_dump(pipeline, examples, necessity_path, sufficiency_path)

    print("computing report...")
    report = summarize(necessity_path, sufficiency_path)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
