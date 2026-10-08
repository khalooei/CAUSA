# Reproducing the paper

Every table, figure and reported statistic in
[`paper/main.pdf`](paper/main.pdf) is produced by one of the stage scripts
in [`reproduce/`](reproduce). Stages must run in order: later stages read
the JSONL/JSON files written by earlier ones.

```bash
bash reproduce/run_all.sh        # everything, in order
bash reproduce/05_analysis.sh    # or any single stage
```

All runners are **resumable**. Results are appended row by row to JSONL
files keyed by `entry_id`, so an interrupted run continues where it
stopped when re-launched with the same arguments, and re-running with a
larger `--n` extends an existing run.

Settings shared by all stages can be overridden through environment
variables: `PY` (interpreter), `SEED` (default `42`), `AUDIO`,
`HALLU_AUDIO` and `OUT` (default `outputs/`).

## Compute

| Stage | What | Hardware | Wall-clock (measured) |
|---|---|---|---|
| `00_data.sh` | BEAF audio (~2 GB) + HalluAudio (517 clips) | network | minutes |
| `01_qwen2audio.sh` | main grid (n = 5000), adaptive search, oracle, priors, substitution | 1 GPU, 32 GB | ~12 h |
| `02_af3.sh` | main grid over the full benchmark (n = 10,800), adaptive search, prior | 1 GPU, 32 GB | a few hours |
| `03_qwen25omni.sh` | main grid (n = 2000), prior | 1 GPU, 32 GB | — |
| `04_halluaudio.sh` | Qwen2-Audio and AF3 on HalluAudio (n = 517) | 1 GPU, 32 GB | — |
| `05_analysis.sh` | all statistics, tables and decoding results | CPU | minutes |
| `06_supplementary.sh` | analyses not in the paper | CPU + GPU | — |
| `07_paper.sh` | Fig. 1 and `paper/main.pdf` | CPU + LaTeX | seconds |

Only one model is loaded at a time. Model weights are downloaded from the
Hugging Face Hub on first use: `Qwen/Qwen2-Audio-7B-Instruct` (~16 GB),
`nvidia/audio-flamingo-3-hf` (~33 GB), `Qwen/Qwen2.5-Omni-7B` and
`laion/clap-htsat-unfused` (~1.2 GB). Set `HF_HOME` to choose the cache
location.

Qwen2-Audio affirms ~87% of queries, so most rows go through the full
nine-condition grid (~6 s per row once the model is warm). AF3 affirms
only ~11%, so most rows need a single screening pass, which is why its
full-benchmark run is the faster of the two.

## Where each result comes from

`Q` = `outputs/qwen2audio`, `A` = `outputs/af3`, `O` = `outputs/qwen25omni`.

| Paper | Stage | Script | Output |
|---|---|---|---|
| Table 1: single-cell, random-window and blank AUROCs; PN̂, PŜ | 01 | `run_experiment.py` | `Q/report.json` |
| Table 1: 5-feature ensemble (0.743) | 05 | `analyze_causal_grounding_profile.py` | `Q/causal_grounding_profile.json` |
| Sec. 5.1: duration regression, matched-duration Wilcoxon tests, residuals | 05 | `analyze_duration_confound.py` | `Q/duration_confound.json`, `A/duration_confound.json` |
| Sec. 5.2: Δ_loc, R̂\*, continuous-drop AUROC | 01, 02 | `run_adaptive_necessity.py` | `Q/adaptive.jsonl`, `A/adaptive.jsonl` |
| Sec. 5.2: transient vs. continuous classes | 05 | `analyze_object_class_breakdown.py` | `Q/object_class_breakdown.json` |
| Sec. 3.5: monotonicity of the thresholded decision | 05 | `analyze_monotonicity.py` | `Q/monotonicity.json`, `A/monotonicity.json` |
| Table 2: oracle / CLAP / random flip rates, IoU | 01, 05 | `run_oracle_localization.py`, `analyze_oracle_localization.py` | `outputs/oracle_qwen2audio/report.json` |
| Sec. 5.4: language prior, Δ detector, 3/4-feature ensembles, CIs | 01, 05 | `run_language_prior.py`, `run_substitution_test.py`, `analyze_language_prior_correction.py` | `Q/language_prior_correction.json` |
| Table 3 (single contrasts), Fig. 1 | 05, 07 | `run_contrastive_decoding.py`, `paper/make_figures.py` | `*/contrastive_decoding.json`, `paper/fig_cd_sweep.pdf` |
| Table 3 (AAD+LangPrior), fix/break rates, CV | 05 | `analyze_decoding_strengths.py` | `*/decoding_strengths.json` |
| Table 3 (+ gated, + ensemble prior) | 01, 05 | `run_language_prior_ensemble.py`, `analyze_gated_decoding.py` | `Q/gated_decoding.json` |
| Table 4: localized vs. random flip rates, three models | 01–03 | `run_experiment.py` | `{Q,O,A}/report.json` |
| Sec. 5.6: Qwen2.5-Omni AUROC and decoding gains | 03, 05 | `run_experiment.py`, `analyze_decoding_strengths.py` | `O/report.json`, `O/decoding_strengths.json` |
| Sec. 5.7: in-distribution profile and cross-model transfer | 05 | `analyze_causal_grounding_profile.py`, `analyze_cross_model_transfer.py` | `outputs/cross_model_transfer.json` |
| Sec. 6: HalluAudio matched-duration effect and combined decoding | 04 | `analyze_duration_confound.py`, `analyze_decoding_strengths.py` | `outputs/halluaudio_qwen2audio/*.json` |
| Method illustration (README) | 07 | `paper/make_fig_illustration.py` | `paper/fig_illustration.pdf` |

## Building the paper

`07_paper.sh` runs the standard `pdflatex` → `bibtex` → `pdflatex` ×2
cycle with the official ICASSP 2027 template (`spconf.sty`,
`IEEEbib.bst`, included). The result must be five pages, with the fifth
holding only references.

## Determinism

Every model query is a single forward pass that reads the next-token
probabilities of "yes" and "no" (no sampling), and all random choices
(benchmark shuffling, random-window placement, exemplar selection) are
seeded by `--seed`. Ablation variants are scored one at a time rather
than batched, because batched bf16 inference changed P(yes) by up to
~0.03 for a small fraction of inputs.
