"""Loader for HalluAudio's object-presence subset (Zhao et al., "HalluAudio:
A Comprehensive Benchmark for Hallucination Detection in Large
Audio-Language Models," arXiv:2604.19300) - a newer, more adversarial
benchmark used as a generalization check.

HalluAudio spans many task types across three parquet-per-subset files
on the Hub (`zhaozhao09/HalluAudio`); a single `load_dataset(..., split="train")`
call fails outright, because different subsets use incompatible schemas
(`Question`/`Answer` vs. `prompt`/`reference`) that the `datasets` library
can't unify automatically.

Two subsets give exactly this project's object-existence format:
`sound_subset/present_sound` (260 rows, all `reference="Yes"`) and
`sound_subset/mismatch_query` (257 rows, all `reference="No"`) - together
a clean, balanced yes/no object-presence set, phrased identically to the
kuanhuggingface benchmark this project otherwise uses
(`Does the recording contain a "X" sound?`). Audio is embedded as raw
WAV bytes per row rather than referenced by path, so this module
materializes each clip to disk once (cached, skipped on rerun) and
returns the same `causa.data.Example` objects the rest of the pipeline
already consumes unchanged.
"""
from __future__ import annotations

import io
import os
import re

import soundfile as sf
from datasets import load_dataset

from causa.data import Example

DATASET_ID = "zhaozhao09/HalluAudio"
SUBSET_FILES = {
    "yes": "sound_subset/present_sound/data.parquet",
    "no": "sound_subset/mismatch_query/data.parquet",
}
_OBJECT_RE = re.compile(r'"(.+?)"')


def _clean_object(raw: str) -> str:
    return raw.replace("_", " ").replace(".", "").strip().lower()


def download_and_load(audio_root: str, limit: int | None = None,
                       shuffle_seed: int | None = None) -> list[Example]:
    """Materializes HalluAudio's present_sound + mismatch_query rows as
    .wav files under `audio_root` (idempotent - skips files that already
    exist) and returns them as `Example` objects, entry_id offset by
    +900000 so they can never collide with the kuanhuggingface benchmark's
    own entry_ids if both are ever loaded in the same process."""
    os.makedirs(audio_root, exist_ok=True)
    examples: list[Example] = []
    entry_id = 900000

    for ground_truth, rel_path in SUBSET_FILES.items():
        ds = load_dataset(DATASET_ID, data_files=rel_path, split="train")
        for row in ds:
            m = _OBJECT_RE.search(row["prompt"])
            if not m:
                continue
            object_name = _clean_object(m.group(1))
            wav_path = os.path.join(audio_root, f"{ground_truth}_{entry_id}.wav")
            if not os.path.exists(wav_path):
                data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
                sf.write(wav_path, data, sr)
            examples.append(Example(
                entry_id=entry_id,
                audio_path=wav_path,
                object_name=object_name,
                query=row["prompt"],
                ground_truth=ground_truth,
            ))
            entry_id += 1

    if shuffle_seed is not None:
        import random
        random.Random(shuffle_seed).shuffle(examples)
    if limit is not None:
        examples = examples[:limit]
    return examples
