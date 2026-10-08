"""Loads the object-existence hallucination benchmark and its audio.

Dataset: kuanhuggingface/Audio-Hallucination_Object-Existence_AudioCaps-ESC50-VocalSound
(Kuan, Lee. "Can Large Audio-Language Models Truly Hear?", ICASSP 2025),
audio bundled separately as kuanhuggingface/BEAF-Audio (BEAF_Audio.tar).

Each row already asks "Is there a sound of X in the audio?" with a
yes/no ground truth. `object` isn't a separate column in this release,
so it's parsed out of `query`.
"""
from __future__ import annotations

import dataclasses
import re

import librosa
import numpy as np
from datasets import load_dataset

DATASET_ID = "kuanhuggingface/Audio-Hallucination_Object-Existence_AudioCaps-ESC50-VocalSound"
_OBJECT_RE = re.compile(r"sound of (.+?)(?: in the audio)?\?", re.IGNORECASE)


@dataclasses.dataclass
class Example:
    entry_id: int
    audio_path: str
    object_name: str
    query: str
    ground_truth: str  # "yes" | "no"


def parse_object(query: str) -> str:
    m = _OBJECT_RE.search(query)
    if not m:
        raise ValueError(f"could not parse object from query: {query!r}")
    return m.group(1).strip()


def load_examples(audio_root: str, split: str = "test", limit: int | None = None,
                   shuffle_seed: int | None = None) -> list[Example]:
    ds = load_dataset(DATASET_ID, split=split)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    examples = []
    for row in ds:
        obj = parse_object(row["query"])
        examples.append(
            Example(
                entry_id=row["entry_id"],
                audio_path=f"{audio_root}/{row['audio']}",
                object_name=obj,
                query=row["query"],
                ground_truth=row["ground_truth"].strip().lower(),
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def load_audio(path: str, sr: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio
