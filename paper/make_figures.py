"""Generates paper/fig_cd_sweep.pdf (paper Fig. 1).

Qwen2-Audio contrastive-decoding accuracy vs. alpha over the full sweep
for each single contrast and the combined AAD+LangPrior rule, showing
that the reported alpha = 1 point is not cherry-picked. Audio Flamingo 3
sits at a flat ceiling for every condition and is reported in the text.

Requires the outputs of `run_contrastive_decoding.py` and
`run_language_prior.py` (see REPRODUCE.md).

Usage:
    python paper/make_figures.py
"""
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

matplotlib.rcParams.update({
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
})

STYLES = {
    "LangPrior-CD (M3ID-style)": dict(color="#0072B2", marker="o", linestyle="-", label="LangPrior-CD (ours)"),
    "AAD-CD (blank)": dict(color="#D55E00", marker="s", linestyle="--", label="AAD-CD (blank, reproduced)"),
    "CAUSA-CD (localized ablation)": dict(color="#009E73", marker="^", linestyle=":", label="CAUSA-CD"),
}

fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.3))

data = json.load(open(os.path.join(OUT, "qwen2audio", "contrastive_decoding.json")))
raw = data["raw_accuracy_all_yes"] * 100
for name, style in STYLES.items():
    sweep = data["sweep"][name]  # keys are alpha values as strings, e.g. "0.0", "0.25", ...
    pairs = sorted(((float(a), acc) for a, acc in sweep.items()), key=lambda p: p[0])
    alphas_sorted = [a for a, _ in pairs]
    accs_sorted = [acc * 100 for _, acc in pairs]
    ax.plot(alphas_sorted, accs_sorted, markersize=3, linewidth=1.2, **style)

# combined (AAD+LangPrior) decoding: shared alpha on both contrasts at once,
# same x-axis as the single-contrast sweeps above - shows our final method
# sitting above every single-contrast curve across most of the range, not
# just at the alpha=1 point reported in Table III.
necessity_rows = [json.loads(l) for l in open(os.path.join(OUT, "qwen2audio", "necessity.jsonl"))
                   if not json.loads(l).get("skipped")]
lang_prior = json.load(open(os.path.join(OUT, "language_prior_qwen2audio.json")))
necessity_rows = [r for r in necessity_rows if r["object_name"] in lang_prior]
p_real = np.array([r["p_yes_before"] for r in necessity_rows])
p_blank = np.array([r["p_yes_after_blank"] for r in necessity_rows])
p_lang = np.array([lang_prior[r["object_name"]]["p_yes_text_only"] for r in necessity_rows])
gt_is_yes = np.array([not r["is_hallucination"] for r in necessity_rows])

combined_alphas = sorted({float(a) for a in data["sweep"]["AAD-CD (blank)"].keys()})
combined_accs = []
for a in combined_alphas:
    d = (1 + 2 * a) * logit(p_real) - a * logit(p_blank) - a * logit(p_lang)
    combined_accs.append(((d > 0) == gt_is_yes).mean() * 100)
ax.plot(combined_alphas, combined_accs, color="#CC79A7", marker="D", markersize=3,
        linewidth=1.6, linestyle="-", label="Combined AAD+LangPrior (ours)", zorder=5)
ax.axhline(71.9, color="#CC79A7", linewidth=1.0, linestyle=":", alpha=0.9)

ax.axhline(raw, color="gray", linewidth=0.9, linestyle="-.", alpha=0.8)
ax.axvline(1.0, color="black", linewidth=0.6, alpha=0.4)
ymin, ymax = ax.get_ylim()
# "raw accuracy" sits far from x=0 (clear of the "0.0" tick label) and
# above the line (clear of the x-axis itself) - the line is flat and
# every curve has risen away from it by alpha=1.5, so this region stays
# empty regardless of which run's numbers are plotted
ax.text(1.55, raw + 0.02 * (ymax - ymin), "raw accuracy", fontsize=6, color="gray", va="bottom")
# alpha=1 sits at the very bottom of its own vertical line instead of
# the top, where it collided with the "gated + ensemble prior" label;
# the line itself still marks alpha=1 at every height, so the label
# doesn't need to share space with text already occupying the top strip
ax.text(1.05, ymin + 0.04 * (ymax - ymin), r"$\alpha{=}1$", fontsize=6, color="black", va="bottom")
ax.text(0.05, 71.9 + 0.015 * (ymax - ymin), "gated + ensemble prior (CV, best)", fontsize=6, color="#CC79A7")
ax.set_xlabel(r"contrastive-decoding strength $\alpha$", fontsize=8)
ax.set_ylabel("accuracy on affirmative\nclaims (%)", fontsize=8)
# legend placed entirely outside the axes (below the x-axis label) so it
# can never overlap a data curve regardless of which region of the plot
# they occupy - in-plot placements all collided with CAUSA-CD's curve,
# which sweeps across most of the lower half of the axes
ax.legend(fontsize=6, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, -0.32), ncol=2, columnspacing=1.0, handletextpad=0.5)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_cd_sweep.pdf"), bbox_inches="tight")
print("wrote fig_cd_sweep.pdf")
