"""Adaptive necessity search: a continuous, per-example causal
explanation size, instead of the fixed 2x2 grid's binary flip test.

Adapts the "deletion metric" / minimal-sufficient-region paradigm from
black-box vision interpretability (RISE, Petsiuk et al., BMVC 2018,
arXiv:1806.07421; extremal perturbation, Fong et al., ICCV 2019,
arXiv:1910.08485) to audio hallucination detection: instead of asking
"does removing this one fixed window flip the claim," ask "what is the
smallest fraction of the clip - added in CLAP-priority order - that
must be removed before the claim flips." That fraction is a continuous,
per-example score: small for a claim genuinely grounded in a compact,
correctly-localized piece of evidence, large (or 1.0, never flipping)
for a claim not really tied to any specific part of the clip.

Run against a random-window-priority control identically, to keep the
"does the ordering matter, independent of how much is removed" question
answerable exactly as in the main grid study.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from causa.ablation import spectral_gate_remove
from causa.clap_localizer import ClapLocalizer
from causa.clap_localizer import SAMPLE_RATE as CLAP_SR
from causa.lalm import SAMPLE_RATE as LALM_SR

ATTENUATION_DB = -100.0  # the grid study's strongest, most decisive setting
WINDOW_SEC = 2.0
HOP_SEC = 0.5
CHECKPOINTS_K = [1, 2, 4, 7, 11, 17]  # cumulative window counts; 17 ~= full 10s clip coverage


def _merged_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ivs = sorted(intervals)
    merged = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged)


@dataclasses.dataclass
class SearchResult:
    flipped: bool
    necessary_fraction: float  # 1.0 (censored) if never flipped within the budget
    n_steps: int
    steps: list[dict]


def search(lalm, audio_lalm: np.ndarray, object_name: str, duration: float,
           window_starts: list[float], window_scores: list[float],
           order: str, rng: np.random.Generator) -> SearchResult:
    """`order`: "clap" (rank by descending CLAP similarity) or "random"."""
    n_windows = len(window_starts)
    if order == "clap":
        ranked = list(np.argsort(window_scores)[::-1])
    elif order == "random":
        ranked = list(rng.permutation(n_windows))
    else:
        raise ValueError(order)

    steps = []
    for k in CHECKPOINTS_K:
        k = min(k, n_windows)
        idx = ranked[:k]
        intervals = [(window_starts[i], min(duration, window_starts[i] + WINDOW_SEC)) for i in idx]
        frac = _merged_duration(intervals) / duration
        ablated = spectral_gate_remove(audio_lalm, LALM_SR, intervals, attenuation_db=ATTENUATION_DB)
        score = lalm.score_yes_no(ablated, object_name)
        flipped = score.claim == "no"
        steps.append({"k": k, "fraction": frac, "p_yes": score.p_yes, "flip": flipped})
        if flipped:
            return SearchResult(flipped=True, necessary_fraction=frac, n_steps=len(steps), steps=steps)
        if k >= n_windows:
            break
    return SearchResult(flipped=False, necessary_fraction=1.0, n_steps=len(steps), steps=steps)


def get_windows(clap: ClapLocalizer, audio_clap: np.ndarray, object_name: str):
    """One CLAP localize call gives every sliding window's score at once -
    reused for both the "clap" and "random" search orderings."""
    loc = clap.localize(audio_clap, CLAP_SR, object_name, window_sec=WINDOW_SEC)
    return loc.window_starts, loc.window_scores
