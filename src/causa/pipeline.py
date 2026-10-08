"""Orchestrates the causal necessity/sufficiency grounding test.

For every example the LALM claims "yes" to (a presence claim - whether
that claim is actually correct or a hallucination is exactly what we're
trying to detect), we run:

  necessity test  - CLAP-localize the claimed evidence, spectrally gate
                     it out, re-ask. A claim that survives removal of its
                     own best-matching evidence is flagged as ungrounded.
  AAD baseline    - the same before/after comparison, but using an
                     all-zero blank clip instead of a localized ablation
                     (the contrast condition of Audio-Aware Decoding,
                     Hsu et al., arXiv:2506.07233).
  CLAP agreement  - a purely correlational baseline: does whole-clip CLAP
                     similarity alone predict whether the claim is a
                     hallucination, with no intervention at all.

For every example the LALM claims "no" to on a genuine negative-control
clip, we separately run the sufficiency test (insert a real exemplar,
see if the claim flips to "yes"). This validates the pipeline's
sensitivity - it is not itself a hallucination-detection signal, since
we're intervening on a clip the model already (correctly) says "no" to.

Each result row is written as it completes (JSONL) so a run can be
interrupted and resumed without losing completed LALM queries, which
are the expensive part.
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np

from causa.ablation import insert_exemplar, make_blank_audio, random_window, spectral_gate_remove
from causa.clap_localizer import ClapLocalizer
from causa.clap_localizer import SAMPLE_RATE as CLAP_SR
from causa.data import Example, load_audio
from causa.exemplars import ExemplarBank
from causa.lalm import SAMPLE_RATE as LALM_SR
from causa.lalm import Qwen2AudioLALM

# documented ablation-study grid: window length x attenuation depth x
# whether the window is CLAP-localized or a random same-size control.
# the random condition isolates whether CLAP's specific window choice
# matters at all, separate from whether the intervention is strong enough.
GRID_WINDOW_SEC = [2.0, 4.0]
GRID_ATTENUATION_DB = [-40.0, -100.0]
DEFAULT_GRID_KEY = "loc_w2.0_d-40"  # matches the original (pre-grid) single-condition run


def _grid_key(placement: str, window_sec: float, atten_db: float) -> str:
    return f"{placement}_w{window_sec}_d{atten_db:g}"


@dataclasses.dataclass
class NecessityResult:
    entry_id: int
    object_name: str
    ground_truth: str
    p_yes_before: float
    p_yes_after_ablation: float  # = grid[DEFAULT_GRID_KEY], kept for backward-compat metrics
    p_yes_after_blank: float
    clap_similarity: float
    localized_t_start: float
    localized_t_end: float
    flip_ablation: bool
    flip_blank: bool
    is_hallucination: bool  # claimed "yes" but ground_truth == "no"
    grid: dict  # {grid_key: {"p_yes_after": float, "flip": bool}} for every (placement, window, depth) cell


@dataclasses.dataclass
class SufficiencyResult:
    entry_id: int
    object_name: str
    p_no_before: float
    p_yes_after_insertion: float
    flip_insertion: bool
    exemplar_available: bool


class GroundingPipeline:
    def __init__(self, lalm: Qwen2AudioLALM, clap: ClapLocalizer, exemplar_bank: ExemplarBank,
                 seed: int = 0):
        self.lalm = lalm
        self.clap = clap
        self.exemplar_bank = exemplar_bank
        self._rng = np.random.default_rng(seed)

    def run_necessity(self, ex: Example) -> NecessityResult | None:
        audio_lalm = load_audio(ex.audio_path, sr=LALM_SR)
        before = self.lalm.score_yes_no(audio_lalm, ex.object_name)
        if before.claim != "yes":
            return None  # necessity test only applies to affirmative claims
        duration = len(audio_lalm) / LALM_SR

        audio_clap = load_audio(ex.audio_path, sr=CLAP_SR)
        whole_clip_sim = self.clap.whole_clip_similarity(audio_clap, CLAP_SR, ex.object_name)

        # one CLAP localization per window size in the grid
        loc_by_window = {
            w: self.clap.localize(audio_clap, CLAP_SR, ex.object_name, window_sec=w)
            for w in GRID_WINDOW_SEC
        }
        base_loc = loc_by_window[GRID_WINDOW_SEC[0]]

        # every ablation variant (8 grid cells + blank), scored sequentially:
        # score_yes_no_batch showed measurable (up to ~0.03 p_yes)
        # discrepancy vs sequential scoring for a minority of items in
        # testing, likely bf16 batched-matmul numerical instability - not
        # an acceptable risk for numbers that go into the paper's results,
        # so this stays sequential despite the GPU-utilization cost.
        grid: dict[str, dict] = {}
        for window_sec in GRID_WINDOW_SEC:
            loc = loc_by_window[window_sec]
            rand_t_start, rand_t_end = random_window(duration, window_sec, self._rng)
            for atten_db in GRID_ATTENUATION_DB:
                loc_ablated = spectral_gate_remove(audio_lalm, LALM_SR, loc.t_start, loc.t_end,
                                                    attenuation_db=atten_db)
                loc_score = self.lalm.score_yes_no(loc_ablated, ex.object_name)
                grid[_grid_key("loc", window_sec, atten_db)] = {
                    "p_yes_after": loc_score.p_yes, "flip": loc_score.claim == "no",
                }
                rand_ablated = spectral_gate_remove(audio_lalm, LALM_SR, rand_t_start, rand_t_end,
                                                     attenuation_db=atten_db)
                rand_score = self.lalm.score_yes_no(rand_ablated, ex.object_name)
                grid[_grid_key("rand", window_sec, atten_db)] = {
                    "p_yes_after": rand_score.p_yes, "flip": rand_score.claim == "no",
                }

        blank = make_blank_audio(audio_lalm)
        after_blank = self.lalm.score_yes_no(blank, ex.object_name)

        default = grid[DEFAULT_GRID_KEY]
        return NecessityResult(
            entry_id=ex.entry_id,
            object_name=ex.object_name,
            ground_truth=ex.ground_truth,
            p_yes_before=before.p_yes,
            p_yes_after_ablation=default["p_yes_after"],
            p_yes_after_blank=after_blank.p_yes,
            clap_similarity=whole_clip_sim,
            localized_t_start=base_loc.t_start,
            localized_t_end=base_loc.t_end,
            flip_ablation=default["flip"],
            flip_blank=after_blank.claim == "no",
            is_hallucination=ex.ground_truth == "no",
            grid=grid,
        )

    def run_sufficiency(self, ex: Example) -> SufficiencyResult | None:
        audio_lalm = load_audio(ex.audio_path, sr=LALM_SR)
        before = self.lalm.score_yes_no(audio_lalm, ex.object_name)
        if before.claim != "no":
            return None  # sufficiency test only applies to (correct) negative claims

        exemplar = self.exemplar_bank.get(ex.object_name)
        if exemplar is None:
            return SufficiencyResult(
                entry_id=ex.entry_id, object_name=ex.object_name,
                p_no_before=before.p_no, p_yes_after_insertion=float("nan"),
                flip_insertion=False, exemplar_available=False,
            )

        mixed = insert_exemplar(audio_lalm, LALM_SR, exemplar, LALM_SR)
        after = self.lalm.score_yes_no(mixed, ex.object_name)
        return SufficiencyResult(
            entry_id=ex.entry_id, object_name=ex.object_name,
            p_no_before=before.p_no, p_yes_after_insertion=after.p_yes,
            flip_insertion=after.claim == "yes", exemplar_available=True,
        )


def run_and_dump(pipeline: GroundingPipeline, examples: list[Example],
                  necessity_path: str, sufficiency_path: str) -> None:
    done_necessity = _load_done_ids(necessity_path)
    done_sufficiency = _load_done_ids(sufficiency_path)

    os.makedirs(os.path.dirname(necessity_path) or ".", exist_ok=True)
    with open(necessity_path, "a") as fn, open(sufficiency_path, "a") as fs:
        for ex in examples:
            if ex.entry_id not in done_necessity:
                r = pipeline.run_necessity(ex)
                row = dataclasses.asdict(r) if r is not None else {"entry_id": ex.entry_id, "skipped": True}
                fn.write(json.dumps(row) + "\n")
                fn.flush()
                done_necessity.add(ex.entry_id)

            if ex.entry_id not in done_sufficiency:
                r = pipeline.run_sufficiency(ex)
                row = dataclasses.asdict(r) if r is not None else {"entry_id": ex.entry_id, "skipped": True}
                fs.write(json.dumps(row) + "\n")
                fs.flush()
                done_sufficiency.add(ex.entry_id)


def _load_done_ids(path: str) -> set[int]:
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["entry_id"])
    return ids
