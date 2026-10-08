"""CLAP-based sliding-window localization of acoustic evidence.

Given a full audio clip and a target object/event phrase, slides a fixed
window over the waveform and scores each window against the text prompt
with LAION-CLAP's joint audio-text embedding space. The window with the
highest cosine similarity is taken as R(o): the spectro-temporal region a
causal ablation should target.

Also exposes a whole-clip similarity score, which is the purely
correlational "does CLAP agree with the claim" baseline (no
intervention, just a second opinion).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

SAMPLE_RATE = 48000  # LAION-CLAP's native rate


def _as_tensor(out) -> torch.Tensor:
    """transformers versions differ on whether get_*_features returns a
    raw embedding tensor or a BaseModelOutputWithPooling; normalize here."""
    return out.pooler_output if hasattr(out, "pooler_output") else out


@dataclasses.dataclass
class Localization:
    t_start: float
    t_end: float
    score: float
    window_scores: list[float]
    window_starts: list[float]


class ClapLocalizer:
    def __init__(self, model_id: str = "laion/clap-htsat-unfused", device: str = "cuda",
                 window_sec: float = 2.0, hop_sec: float = 0.5):
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device
        self.window_sec = window_sec
        self.hop_sec = hop_sec

    def _text_embed(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            out = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(_as_tensor(out), dim=-1)

    def _audio_embed_batch(self, windows: list[np.ndarray]) -> torch.Tensor:
        inputs = self.processor(audio=windows, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.get_audio_features(**inputs)
        return torch.nn.functional.normalize(_as_tensor(out), dim=-1)

    def localize(self, audio: np.ndarray, sr: int, object_name: str,
                 prompt_template: str = "a sound of {object}",
                 window_sec: float | None = None) -> Localization:
        assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz audio, got {sr}"
        window_sec = window_sec if window_sec is not None else self.window_sec
        duration = len(audio) / sr
        win_n = int(window_sec * sr)
        hop_n = int(self.hop_sec * sr)

        if duration <= window_sec:
            windows = [audio]
            starts = [0.0]
        else:
            starts_samples = list(range(0, max(1, len(audio) - win_n + 1), hop_n))
            if not starts_samples:
                starts_samples = [0]
            windows = [_pad_or_trim(audio[s:s + win_n], win_n) for s in starts_samples]
            starts = [s / sr for s in starts_samples]

        text_emb = self._text_embed(prompt_template.format(object=object_name))
        audio_emb = self._audio_embed_batch(windows)
        sims = (audio_emb @ text_emb.T).squeeze(-1).float().cpu().numpy().tolist()

        best_idx = int(np.argmax(sims))
        t_start = starts[best_idx]
        t_end = min(duration, t_start + window_sec)
        return Localization(
            t_start=t_start, t_end=t_end, score=sims[best_idx],
            window_scores=sims, window_starts=starts,
        )

    def whole_clip_similarity(self, audio: np.ndarray, sr: int, object_name: str,
                               prompt_template: str = "a sound of {object}") -> float:
        assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz audio, got {sr}"
        text_emb = self._text_embed(prompt_template.format(object=object_name))
        audio_emb = self._audio_embed_batch([audio])
        return float((audio_emb @ text_emb.T).squeeze().item())


def _pad_or_trim(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))
