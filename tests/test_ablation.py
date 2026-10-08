"""CPU-only tests for the waveform interventions (no models required)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causa.ablation import (  # noqa: E402
    insert_exemplar,
    make_blank_audio,
    random_window,
    spectral_gate_remove,
    substitute_unrelated,
)

SR = 16000


def _tone(seconds: float, freq: float = 440.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def test_spectral_gate_removes_only_the_window():
    audio = _tone(10.0)
    out = spectral_gate_remove(audio, SR, 4.0, 6.0, attenuation_db=-100.0)
    assert out.shape == audio.shape and out.dtype == audio.dtype
    inside = slice(int(4.2 * SR), int(5.8 * SR))
    assert _rms(out[inside]) < 1e-3 * _rms(audio[inside])
    for outside in (slice(0, int(3.5 * SR)), slice(int(6.5 * SR), len(audio))):
        np.testing.assert_allclose(out[outside], audio[outside], atol=1e-3)


@pytest.mark.parametrize("db", [-40.0, -100.0])
def test_spectral_gate_depth(db):
    audio = _tone(6.0)
    out = spectral_gate_remove(audio, SR, 2.0, 4.0, attenuation_db=db)
    inside = slice(int(2.3 * SR), int(3.7 * SR))
    measured = 20 * np.log10(_rms(out[inside]) / _rms(audio[inside]) + 1e-12)
    assert measured < db + 3.0


def test_spectral_gate_multiple_intervals():
    audio = _tone(10.0)
    out = spectral_gate_remove(audio, SR, [(1.0, 2.0), (7.0, 8.0)], attenuation_db=-100.0)
    for lo, hi in [(1.2, 1.8), (7.2, 7.8)]:
        assert _rms(out[int(lo * SR):int(hi * SR)]) < 1e-3
    np.testing.assert_allclose(out[int(3 * SR):int(6 * SR)], audio[int(3 * SR):int(6 * SR)], atol=1e-3)


def test_random_window_bounds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        lo, hi = random_window(10.0, 4.0, rng)
        assert 0.0 <= lo and hi <= 10.0 + 1e-9
        assert hi - lo == pytest.approx(4.0)
    lo, hi = random_window(1.5, 4.0, rng)
    assert (lo, hi) == (0.0, 1.5)


def test_blank_and_substitution_preserve_length():
    audio = _tone(5.0)
    blank = make_blank_audio(audio)
    assert blank.shape == audio.shape and not blank.any()
    short = _tone(2.0, freq=1000.0)
    sub = substitute_unrelated(audio, short)
    assert sub.shape == audio.shape
    assert substitute_unrelated(audio, _tone(8.0)).shape == audio.shape


def test_insert_exemplar_mixes_only_the_target_span():
    rng = np.random.default_rng(0)
    negative = (0.01 * rng.standard_normal(10 * SR)).astype(np.float32)
    exemplar = _tone(2.0, freq=880.0, sr=48000)
    mixed = insert_exemplar(negative, SR, exemplar, 48000, t_start=3.0)
    assert mixed.shape == negative.shape and np.abs(mixed).max() <= 1.0
    np.testing.assert_array_equal(mixed[: 3 * SR], negative[: 3 * SR])
    np.testing.assert_array_equal(mixed[5 * SR + 10:], negative[5 * SR + 10:])
    assert _rms(mixed[3 * SR:5 * SR]) > 1.5 * _rms(negative[3 * SR:5 * SR])

