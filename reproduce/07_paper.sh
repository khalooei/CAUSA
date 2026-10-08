#!/usr/bin/env bash
# Regenerate Fig. 1 from the outputs and build paper/main.pdf.
source "$(dirname "$0")/_common.sh"

step "Figures"
$PY paper/make_figures.py
[ -f "$AUDIO/Y31WGUPOYS5g@2-37806-A-40@1-19501-A-7@m2428_0_laughter.wav" ] \
    && $PY paper/make_fig_illustration.py

step "LaTeX"
cd paper
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex | grep -E "Output written|Error" || true
