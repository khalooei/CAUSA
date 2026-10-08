"""Physical interventions on the audio signal.

Two interventions, both applied directly to the waveform (never to model
internals), so they stay valid for any black-box LALM:

  - `spectral_gate_remove`: the necessity intervention do(R(o) := empty).
    Attenuates energy inside a localized time window (optionally a
    frequency band within it too) while leaving the rest of the clip -
    background noise, other events, silence - completely untouched. This
    is deliberately unlike AAD's all-zero "blank audio" contrast: the
    surrounding context stays real and in-distribution for the audio
    encoder, only the claimed evidence is removed.

  - `insert_exemplar`: the sufficiency intervention do(R(o) := r). Mixes a
    real exemplar of the target class into a genuine negative-control
    clip (one that truly lacks the object), producing an in-distribution
    positive.

  - `make_blank_audio`: reproduces AAD's negative/contrast condition
    (all-zero waveform of matching length) so it can be run as a baseline
    through the exact same downstream scoring code.

  - `substitute_unrelated`: full-clip replacement with a genuine, real,
    unrelated recording (one that also does not contain the target
    object). Concurrent work on white-box audio grounding (Cho et al.,
    "Tracing Audio Grounding and Answer Selection in Audio LLMs,"
    arXiv:2609.04637) finds that substituting real unrelated audio
    degrades a grounded model's answer confidence *more* than silence
    does - silence is a degenerate, easily-detected "no input" signal,
    whereas real unrelated content still activates the encoder normally
    while definitely not containing the evidence. Unlike that paper's
    method, this stays black-box: it only ever swaps waveforms in and
    reads text/logits out.
"""
from __future__ import annotations

import numpy as np


def make_blank_audio(audio: np.ndarray) -> np.ndarray:
    return np.zeros_like(audio)


def substitute_unrelated(audio: np.ndarray, unrelated_audio: np.ndarray) -> np.ndarray:
    """Replace `audio` entirely with a real, unrelated recording, matched
    to the same length (tiled if shorter, trimmed if longer) so duration
    itself carries no information about which condition is which."""
    n = len(audio)
    if len(unrelated_audio) >= n:
        return unrelated_audio[:n].astype(audio.dtype)
    reps = int(np.ceil(n / len(unrelated_audio)))
    return np.tile(unrelated_audio, reps)[:n].astype(audio.dtype)


def random_window(duration: float, window_sec: float, rng: np.random.Generator) -> tuple[float, float]:
    """A same-size control window placed uniformly at random, independent
    of CLAP's localization. Isolates whether CLAP's specific window choice
    matters, or whether ablating *any* window of that size has the same
    (lack of) effect on the model's claim."""
    window_sec = min(window_sec, duration)
    t_start = float(rng.uniform(0, max(0.0, duration - window_sec)))
    return t_start, t_start + window_sec


def spectral_gate_remove(
    audio: np.ndarray,
    sr: int,
    t_start: float | list[tuple[float, float]],
    t_end: float | None = None,
    attenuation_db: float = -40.0,
    fade_sec: float = 0.05,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Attenuate energy within one or more [t_start, t_end) intervals, in
    one STFT pass. Either call as `(t_start, t_end)` for a single interval
    (the original single-window necessity test), or pass a list of
    `(t_start, t_end)` tuples as `t_start` with `t_end=None` for several
    disjoint regions gated together - used by the adaptive necessity
    search (`adaptive_necessity.py`), which grows a merged CLAP-ranked
    region one window at a time.

    A short raised-cosine fade at each interval's edges avoids the
    clicking artifact a hard gate would introduce, without touching audio
    outside the padded regions.
    """
    import librosa

    intervals = t_start if t_end is None else [(t_start, t_end)]

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    n_frames = stft.shape[1]
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    gain = np.ones(n_frames, dtype=np.float32)
    target_gain = 10 ** (attenuation_db / 20.0)
    fade_frames = max(1, int(fade_sec * sr / hop_length))

    for lo_t, hi_t in intervals:
        inside = (frame_times >= lo_t) & (frame_times < hi_t)
        gain[inside] = target_gain
        idx = np.where(inside)[0]
        if len(idx) == 0:
            continue
        ramp = np.linspace(1.0, target_gain, fade_frames)
        for i, g in enumerate(ramp):
            lo = idx[0] - fade_frames + i
            if 0 <= lo < n_frames and not inside[lo]:
                gain[lo] = min(gain[lo], g)
        ramp_out = np.linspace(target_gain, 1.0, fade_frames)
        for i, g in enumerate(ramp_out):
            hi = idx[-1] + 1 + i
            if 0 <= hi < n_frames and not inside[hi]:
                gain[hi] = min(gain[hi], g)

    gated_stft = stft * gain[np.newaxis, :]
    ablated = librosa.istft(gated_stft, hop_length=hop_length, length=len(audio))
    return ablated.astype(audio.dtype)


def insert_exemplar(
    negative_audio: np.ndarray,
    sr: int,
    exemplar_audio: np.ndarray,
    exemplar_sr: int,
    t_start: float | None = None,
    target_snr_db: float = 6.0,
) -> np.ndarray:
    """Overlay a real exemplar of the target class into a genuine negative clip.

    The exemplar is resampled to `sr`, RMS-normalized relative to the
    local background so it is audible but not clipping, and additively
    mixed at `t_start` (defaults to the clip's temporal center).
    """
    import librosa

    if exemplar_sr != sr:
        exemplar_audio = librosa.resample(exemplar_audio, orig_sr=exemplar_sr, target_sr=sr)

    mixed = negative_audio.copy()
    n = len(mixed)
    m = len(exemplar_audio)

    if t_start is None:
        start_sample = max(0, (n - m) // 2)
    else:
        start_sample = int(t_start * sr)

    end_sample = min(n, start_sample + m)
    seg_len = end_sample - start_sample
    if seg_len <= 0:
        return mixed
    exemplar_seg = exemplar_audio[:seg_len]

    bg_rms = np.sqrt(np.mean(mixed[start_sample:end_sample] ** 2)) + 1e-8
    ex_rms = np.sqrt(np.mean(exemplar_seg ** 2)) + 1e-8
    target_rms = bg_rms * (10 ** (target_snr_db / 20.0))
    exemplar_seg = exemplar_seg * (target_rms / ex_rms)

    mixed[start_sample:end_sample] = mixed[start_sample:end_sample] + exemplar_seg
    peak = np.abs(mixed).max()
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(negative_audio.dtype)
