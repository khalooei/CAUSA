"""Generates paper/fig_illustration.pdf: what the localized spectral-gate
necessity ablation does to a real clip.

Example: entry_id 9726, a true-positive "laughter" claim from
Qwen2-Audio-7B-Instruct. CLAP localizes 6.0-8.0 s as the evidence
region; gating it at -100 dB flips the claim (P(yes) 0.905 -> 0.133),
while an equal-size random window at 0.5-2.5 s does not.

Usage:
    python paper/make_fig_illustration.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from causa.ablation import spectral_gate_remove
from causa.data import load_audio

matplotlib.rcParams.update({"font.size": 8, "pdf.fonttype": 42})

AUDIO_PATH = os.path.join(HERE, "..", "data", "audio", "beaf_audio",
                          "Y31WGUPOYS5g@2-37806-A-40@1-19501-A-7@m2428_0_laughter.wav")
SR = 16000
LOC_T = (6.0, 8.0)
RAND_T = (0.5, 2.5)

audio = load_audio(AUDIO_PATH, sr=SR)
duration = len(audio) / SR
gated_loc = spectral_gate_remove(audio, SR, LOC_T[0], LOC_T[1], attenuation_db=-100.0)

fig, axes = plt.subplots(2, 1, figsize=(3.3, 1.9), sharex=True)

for ax, sig, title in [
    (axes[0], audio, "original clip"),
    (axes[1], gated_loc, r"after localized gate ($-100$ dB)"),
]:
    D = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=1024, hop_length=256)), ref=np.max)
    librosa.display.specshow(D, sr=SR, hop_length=256, x_axis="time", y_axis="linear", ax=ax, cmap="magma")
    ax.set_ylim(0, 4000)
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("Hz", fontsize=7)
    ax.set_xlabel("")

for ax in axes:
    ax.axvspan(LOC_T[0], LOC_T[1], edgecolor="cyan", facecolor="none", linewidth=1.3, linestyle="-")
    ax.axvspan(RAND_T[0], RAND_T[1], edgecolor="white", facecolor="none", linewidth=1.0, linestyle="--")

axes[0].text(LOC_T[0], 3600, "CLAP-localized\n(evidence region)", color="cyan", fontsize=6, va="top")
axes[0].text(RAND_T[0], 3600, "random control", color="white", fontsize=6, va="top")
axes[1].set_xlabel("time (s)", fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_illustration.pdf"), bbox_inches="tight")
print("wrote fig_illustration.pdf")
