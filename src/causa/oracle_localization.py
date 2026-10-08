"""Oracle-localization control: does CLAP find the right place, and does
that matter?

The main necessity grid only ever ablates CLAP's *guess* at where the
evidence is, never the *true* location - so a null result ("location
doesn't matter") is consistent with two mechanistically different stories
that look identical from outside:

  (a) the model's claim genuinely doesn't depend on any specific region
      of the clip, or
  (b) CLAP's guess is often wrong, so "ablate CLAP's window" ends up close
      to "ablate a random window" even though the model *would* be
      sensitive to the true window, if only it were ablated.

This experiment builds synthetic affirmative claims with a KNOWN evidence
location: insert a real per-object exemplar (reusing `exemplars.py`, the
same machinery the sufficiency test already uses) into a genuine
negative-control clip at a random - but recorded - offset, keep only the
synthetic mixtures the model actually affirms, then ablate three
same-size windows on the *same* mixture: the true insertion window
("oracle"), CLAP's own top pick on that mixture, and a uniform random
window. Because ground truth is now known by construction, this also
yields a pure localization-accuracy number (temporal IoU between CLAP's
window and the true one) that the original benchmark can never give,
since real clips have no ground-truth event timestamp.

Three possible outcomes, each diagnostic:
  - oracle >> clap ~= random: CLAP's imprecision is the bottleneck - a
    sharper localizer (e.g. frame-level FLAM) should recover a real
    localization effect the main grid study currently misses.
  - oracle ~= clap > random: CLAP is reasonably accurate; location has a
    small but genuine effect that CLAP mostly captures already.
  - oracle ~= clap ~= random: the strongest possible confirmation of the
    main finding - even ablating the *true* evidence location barely
    matters for this model's claims, independent of localizer quality.
"""
from __future__ import annotations

import dataclasses

import librosa
import numpy as np

from causa.ablation import insert_exemplar, random_window, spectral_gate_remove
from causa.clap_localizer import ClapLocalizer
from causa.clap_localizer import SAMPLE_RATE as CLAP_SR
from causa.data import Example, load_audio
from causa.exemplars import ExemplarBank

ATTENUATION_DB = -100.0  # the grid study's strongest, most decisive setting
MIN_WINDOW_SEC = 2.0  # matches the main grid's smaller window; exemplars are ~this long already


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


@dataclasses.dataclass
class OracleResult:
    entry_id: int
    object_name: str
    exemplar_duration: float
    window_sec: float
    true_t_start: float  # exact exemplar insertion point (unpadded)
    true_t_end: float
    oracle_t_start: float  # true interval, padded to window_sec for a fair size-matched ablation
    oracle_t_end: float
    clap_t_start: float
    clap_t_end: float
    clap_oracle_iou: float
    p_yes_mixed: float  # after insertion, before any ablation
    p_yes_after_oracle: float
    p_yes_after_clap: float
    p_yes_after_random: float
    flip_oracle: bool
    flip_clap: bool
    flip_random: bool


class OracleLocalizationPipeline:
    """`lalm_sr` must match the *given* `lalm`'s own native sample rate
    (each LALM wrapper module exports its own `SAMPLE_RATE`) - it is
    passed explicitly rather than imported from `causa.lalm` so this
    pipeline works correctly for any LALM, not just Qwen2-Audio."""

    def __init__(self, lalm, clap: ClapLocalizer, exemplar_bank: ExemplarBank,
                 lalm_sr: int, seed: int = 0):
        self.lalm = lalm
        self.clap = clap
        self.exemplar_bank = exemplar_bank
        self.lalm_sr = lalm_sr
        self._rng = np.random.default_rng(seed)

    def run(self, ex: Example) -> OracleResult | None:
        exemplar = self.exemplar_bank.get(ex.object_name)
        if exemplar is None:
            return None

        LALM_SR = self.lalm_sr
        audio_lalm = load_audio(ex.audio_path, sr=LALM_SR)
        duration = len(audio_lalm) / LALM_SR
        exemplar_duration = len(exemplar) / LALM_SR
        window_sec = max(MIN_WINDOW_SEC, exemplar_duration)
        if duration <= window_sec:
            return None  # too short to place a randomly-offset, known-location window

        # random but KNOWN offset - unlike the sufficiency test (always
        # centered), so ground-truth location actually varies across examples
        max_start = duration - exemplar_duration
        true_t_start = float(self._rng.uniform(0.0, max_start))
        true_t_end = min(duration, true_t_start + exemplar_duration)
        mixed = insert_exemplar(audio_lalm, LALM_SR, exemplar, LALM_SR, t_start=true_t_start)

        mixed_score = self.lalm.score_yes_no(mixed, ex.object_name)
        if mixed_score.claim != "yes":
            return None  # only usable if the model actually affirms the synthetic mixture

        # pad the true interval up to window_sec (symmetric, then re-clamped
        # to the clip) so all three ablation conditions remove an equal amount
        pad = (window_sec - (true_t_end - true_t_start)) / 2.0
        oracle_t_start = max(0.0, true_t_start - pad)
        oracle_t_end = min(duration, oracle_t_start + window_sec)
        oracle_t_start = max(0.0, oracle_t_end - window_sec)

        mixed_clap = librosa.resample(mixed.astype(np.float32), orig_sr=LALM_SR, target_sr=CLAP_SR)
        clap_loc = self.clap.localize(mixed_clap, CLAP_SR, ex.object_name, window_sec=window_sec)
        clap_t_start, clap_t_end = clap_loc.t_start, clap_loc.t_end

        rand_t_start, rand_t_end = random_window(duration, window_sec, self._rng)

        iou = _iou((oracle_t_start, oracle_t_end), (clap_t_start, clap_t_end))

        def ablate_and_score(t0: float, t1: float):
            ablated = spectral_gate_remove(mixed, LALM_SR, t0, t1, attenuation_db=ATTENUATION_DB)
            return self.lalm.score_yes_no(ablated, ex.object_name)

        oracle_score = ablate_and_score(oracle_t_start, oracle_t_end)
        clap_score = ablate_and_score(clap_t_start, clap_t_end)
        random_score = ablate_and_score(rand_t_start, rand_t_end)

        return OracleResult(
            entry_id=ex.entry_id, object_name=ex.object_name,
            exemplar_duration=exemplar_duration, window_sec=window_sec,
            true_t_start=true_t_start, true_t_end=true_t_end,
            oracle_t_start=oracle_t_start, oracle_t_end=oracle_t_end,
            clap_t_start=clap_t_start, clap_t_end=clap_t_end,
            clap_oracle_iou=iou,
            p_yes_mixed=mixed_score.p_yes,
            p_yes_after_oracle=oracle_score.p_yes,
            p_yes_after_clap=clap_score.p_yes,
            p_yes_after_random=random_score.p_yes,
            flip_oracle=oracle_score.claim == "no",
            flip_clap=clap_score.claim == "no",
            flip_random=random_score.claim == "no",
        )
