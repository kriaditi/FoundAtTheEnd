#!/usr/bin/env bash
# Stage 8, step 0: inspect the M2KR E-VQA TRAIN files (text only, no images, no GPU; login node is fine).
#   bash scripts/02c_inspect_evqa.sh
set -euo pipefail
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
source "$PROJ/scripts/config.sh"
source "$PROJ/scripts/lib.sh"
mkdirs
start_log 02c_inspect_evqa

is_ok setup_env || die "run setup first: bash scripts/00_setup.sh"
cd "$AUTHORS_DIR"
"$PY" "$PROJ/lae/inspect_evqa_train.py" \
  --authors-dir "$AUTHORS_DIR" \
  --out "$RUNS_DIR/inspect_evqa_train/report.txt"
