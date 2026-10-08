# Shared settings for the reproduction stages. Source, don't execute.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=${PY:-python}
SEED=${SEED:-42}
AUDIO=${AUDIO:-data/audio/beaf_audio}
HALLU_AUDIO=${HALLU_AUDIO:-data/audio/halluaudio}
OUT=${OUT:-outputs}
mkdir -p "$OUT"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
