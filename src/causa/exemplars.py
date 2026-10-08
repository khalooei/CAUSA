"""Builds a small per-object exemplar bank for the sufficiency intervention.

For each object class, picks a handful of clips the benchmark labels as
truly containing that object (ground_truth == "yes"), CLAP-localizes the
evidence region within each, and caches the extracted snippet (at the
LALM's sample rate) so it can be inserted into negative-control clips
during the sufficiency test. Snippets are extracted once per object and
reused, since re-running CLAP localization for every negative instance
would be wasteful and the exemplar itself doesn't depend on the target
clip.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from causa.clap_localizer import ClapLocalizer
from causa.clap_localizer import SAMPLE_RATE as CLAP_SR
from causa.data import Example, load_audio
from causa.lalm import SAMPLE_RATE as LALM_SR


class ExemplarBank:
    def __init__(self, examples: list[Example], clap: ClapLocalizer,
                 per_object: int = 3, seed: int = 0):
        self.clap = clap
        self.per_object = per_object
        self._rng = random.Random(seed)
        self._by_object: dict[str, list[Example]] = defaultdict(list)
        for ex in examples:
            if ex.ground_truth == "yes":
                self._by_object[ex.object_name].append(ex)
        self._cache: dict[str, list[np.ndarray]] = {}

    def _extract_snippet(self, ex: Example) -> np.ndarray | None:
        try:
            audio_clap = load_audio(ex.audio_path, sr=CLAP_SR)
            loc = self.clap.localize(audio_clap, CLAP_SR, ex.object_name)
            audio_lalm = load_audio(ex.audio_path, sr=LALM_SR)
            s = int(loc.t_start * LALM_SR)
            e = int(loc.t_end * LALM_SR)
            snippet = audio_lalm[s:e]
            if len(snippet) < LALM_SR * 0.2:  # too short to be a usable exemplar
                return None
            return snippet
        except Exception:
            return None

    def get(self, object_name: str) -> np.ndarray | None:
        if object_name not in self._cache:
            pool = [e for e in self._by_object.get(object_name, [])]
            self._rng.shuffle(pool)
            snippets = []
            for ex in pool:
                if len(snippets) >= self.per_object:
                    break
                snip = self._extract_snippet(ex)
                if snip is not None:
                    snippets.append(snip)
            self._cache[object_name] = snippets
        candidates = self._cache[object_name]
        if not candidates:
            return None
        return self._rng.choice(candidates)


class UnrelatedAudioBank:
    """Provides real, unrelated full clips for the substitution test.

    "Unrelated" means: a genuine negative for the object in question
    (ground_truth == "no"), drawn from a different underlying audio file
    than the clip under test - not just a different row (many rows in
    this benchmark share one audio file across several objects).
    """

    def __init__(self, examples: list[Example], seed: int = 0):
        self._rng = random.Random(seed)
        self._by_object: dict[str, list[Example]] = defaultdict(list)
        for ex in examples:
            if ex.ground_truth == "no":
                self._by_object[ex.object_name].append(ex)

    def get(self, object_name: str, exclude_audio_path: str) -> Example | None:
        pool = [e for e in self._by_object.get(object_name, []) if e.audio_path != exclude_audio_path]
        if not pool:
            return None
        return self._rng.choice(pool)
