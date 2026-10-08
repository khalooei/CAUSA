"""Materializes HalluAudio's object-presence subset to disk.

Writes `present_sound` and `mismatch_query` clips to
`data/audio/halluaudio/*.wav`. Idempotent: existing files are skipped.

Usage:
    python scripts/download_halluaudio.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causa.halluaudio_data import download_and_load  # noqa: E402


def main():
    audio_root = os.path.join(os.path.dirname(__file__), "..", "data", "audio", "halluaudio")
    examples = download_and_load(audio_root)
    n_yes = sum(1 for e in examples if e.ground_truth == "yes")
    n_no = sum(1 for e in examples if e.ground_truth == "no")
    print(f"{len(examples)} examples materialized to {audio_root} ({n_yes} yes, {n_no} no)")


if __name__ == "__main__":
    main()
