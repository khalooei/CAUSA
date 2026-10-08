#!/usr/bin/env bash
# Full reproduction, in order. Multi-day on a single GPU; each stage can
# also be run on its own and every runner resumes where it stopped.
set -euo pipefail
D="$(dirname "$0")"
for s in 00_data 01_qwen2audio 02_af3 03_qwen25omni 04_halluaudio 05_analysis 07_paper; do
    bash "$D/$s.sh"
done
