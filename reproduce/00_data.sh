#!/usr/bin/env bash
# Download benchmark audio (~2 GB BEAF archive + 517 HalluAudio clips).
source "$(dirname "$0")/_common.sh"

step "BEAF object-existence audio"
$PY scripts/download_data.py

step "HalluAudio object-presence subset"
$PY scripts/download_halluaudio.py
