"""Black-box causal audit of a single claim.

Asks an LALM "Is there a sound of <object> in the audio?" about one clip,
then re-asks under each intervention used in the paper:

  localized   CLAP-localized window removed by spectral gating
  random      same-size window at a random position (location control)
  blank       all-zero waveform (the Audio-Aware Decoding contrast)
  no audio    text-only prompt, no audio block (language prior)

and applies the contrastive-decoding corrections at alpha = 1.

Usage:
    python examples/necessity_demo.py --audio path/to/clip.wav --object laughter
    python examples/necessity_demo.py --audio clip.wav --object "dog barking" \\
        --model af3 --window-sec 4 --atten-db -100 --save-dir demo_out
"""
import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from causa.ablation import make_blank_audio, random_window, spectral_gate_remove  # noqa: E402
from causa.clap_localizer import SAMPLE_RATE as CLAP_SR, ClapLocalizer  # noqa: E402
from causa.data import load_audio  # noqa: E402

LALM_SR = 16000
MODELS = {
    "qwen2audio": ("causa.lalm", "Qwen2AudioLALM"),
    "af3": ("causa.af3_lalm", "AudioFlamingo3LALM"),
    "qwen25omni": ("causa.qwen25omni_lalm", "Qwen25OmniLALM"),
}


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", required=True, help="path to an audio file (any format librosa reads)")
    p.add_argument("--object", required=True, help='sound to ask about, e.g. "laughter"')
    p.add_argument("--model", choices=list(MODELS), default="qwen2audio")
    p.add_argument("--window-sec", type=float, default=4.0)
    p.add_argument("--atten-db", type=float, default=-100.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-dir", default=None, help="optionally write the ablated waveforms here")
    args = p.parse_args()

    module, cls = MODELS[args.model]
    lalm = getattr(importlib.import_module(module), cls)(device=args.device)
    clap = ClapLocalizer(device=args.device)

    audio = load_audio(args.audio, sr=LALM_SR)
    audio_clap = load_audio(args.audio, sr=CLAP_SR)
    duration = len(audio) / LALM_SR

    loc = clap.localize(audio_clap, CLAP_SR, args.object, window_sec=args.window_sec)
    rand_t = random_window(duration, args.window_sec, np.random.default_rng(args.seed))

    variants = {
        "original": audio,
        f"localized [{loc.t_start:.1f}-{loc.t_end:.1f}s]": spectral_gate_remove(
            audio, LALM_SR, loc.t_start, loc.t_end, attenuation_db=args.atten_db),
        f"random    [{rand_t[0]:.1f}-{rand_t[1]:.1f}s]": spectral_gate_remove(
            audio, LALM_SR, rand_t[0], rand_t[1], attenuation_db=args.atten_db),
        "blank": make_blank_audio(audio),
        "no audio": None,
    }
    scores = {name: lalm.score_yes_no(x, args.object) for name, x in variants.items()}

    print(f'\nQ: "Is there a sound of {args.object} in the audio?"  ({args.model}, '
          f"{duration:.1f}s clip, {args.window_sec:g}s / {args.atten_db:g} dB)\n")
    print(f"{'condition':<26}{'P(yes)':>8}  answer")
    for name, s in scores.items():
        print(f"{name:<26}{s.p_yes:>8.3f}  {s.claim}")

    names = list(scores)
    p_real = scores["original"].p_yes
    p_loc, p_rand = scores[names[1]].p_yes, scores[names[2]].p_yes
    p_blank, p_lang = scores["blank"].p_yes, scores["no audio"].p_yes
    print(f"\nlocalized - random drop : {(p_real - p_loc) - (p_real - p_rand):+.3f}"
          "   (> 0: the located region matters more than an arbitrary one)")
    print(f"audio beyond lang. prior: {p_real - p_lang:+.3f}")

    def decide(d: float) -> str:
        return "yes" if d > 0 else "no"

    lr, lb, ll = logit(p_real), logit(p_blank), logit(p_lang)
    print("\ncontrastive decoding (alpha = 1):")
    print(f"  AAD-CD        -> {decide(2 * lr - lb)}")
    print(f"  LangPrior-CD  -> {decide(2 * lr - ll)}")
    print(f"  AAD+LangPrior -> {decide(3 * lr - lb - ll)}")

    if args.save_dir:
        import soundfile as sf
        os.makedirs(args.save_dir, exist_ok=True)
        for tag, x in zip(["original", "localized", "random"], list(variants.values())[:3]):
            sf.write(os.path.join(args.save_dir, f"{tag}.wav"), x, LALM_SR)
        print(f"\nwrote original/localized/random .wav files to {args.save_dir}/")


if __name__ == "__main__":
    main()
