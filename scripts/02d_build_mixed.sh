#!/usr/bin/env bash
# Stage 8: build the MIXED (InfoSeek + E-VQA) training data: 10,000 + 10,000 train, 250 + 250 dev,
# conditions mixed_standard and mixed_balanced_s<S> for S in MIXED_SEEDS.
#
#   bash scripts/02d_build_mixed.sh            # Slurm: one GPU job
#   bash scripts/02d_build_mixed.sh --status   # status + data summary
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
source "$PROJ/scripts/config.sh"
source "$PROJ/scripts/lib.sh"
mkdirs

case "${1:-}" in
  --status)
    status_summary
    f="$MIXED_DATA_DIR/final/build_stats.json"
    if [[ -f "$f" ]]; then
      "$PY" -c "import json; s=json.load(open('$f')); [print(f'   {k}: {v}') for k,v in s.items() if k not in ('design','infoseek_source_manifest')]"
    fi
    exit 0 ;;
  --yes) export ASSUME_YES=1 ;;
  "") ;;
  *) die "unknown option: $1" ;;
esac

start_log 02d_build_mixed
is_ok build_data || die "InfoSeek data not built yet"
is_ok evqa_frozen || die "the frozen E-VQA evaluation must have run (it builds the E-VQA index): bash scripts/06_evqa.sh"
if is_ok build_mixed; then
  log "build_mixed already done: $(cat "$STATUS_DIR/build_mixed.ok")"; exit 0
fi
job_active build_mixed && { log "job $(cat "$STATUS_DIR/build_mixed.jobid") already queued or running"; exit 0; }

cat <<EOT
About to build the mixed training data (one GPU job):
  1. leakage filter + random pool of $MIXED_POOL E-VQA train questions      (CPU, minutes)
  2. extract the pool's images from inat.zip (8.9 GB) and google-landmark.tar  (disk, one pass each)
  3. PreFLMR retrieval (evaluation index) until $MIXED_PER_SOURCE_TRAIN + $MIXED_PER_SOURCE_DEV E-VQA questions qualify  (GPU)
  4. combine with the first $MIXED_PER_SOURCE_TRAIN + $MIXED_PER_SOURCE_DEV InfoSeek standard rows; write mixed_standard
     and mixed_balanced_s{${MIXED_SEEDS// /,}}; hard checks
  output: $MIXED_DATA_DIR/final/    resumable: rerun the same command
EOT
confirm "Start?" || { log "not started"; exit 0; }
submit_job build_mixed "$PROJ/scripts/slurm/build_mixed.sub" gpu "$BUILD_TIME"
