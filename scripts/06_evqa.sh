#!/usr/bin/env bash
# Stage 6: E-VQA transfer (authors' P20 protocol) for the frozen reader and every trained adapter.
#
#   bash scripts/06_evqa.sh            # Slurm: frozen first, then one job per adapter (dependent on frozen)
#   bash scripts/06_evqa.sh --compare  # only rebuild results/COMPARISON_EVQA.md from finished runs
#   bash scripts/06_evqa.sh --status
#
# The frozen run builds the E-VQA index and retrieval cache (first time: downloads the two
# image archives, ~1 to 2 h); each adapter run then takes about an hour on this GPU.
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
source "$PROJ/scripts/config.sh"
source "$PROJ/scripts/lib.sh"
mkdirs

ONLY_COMPARE=0
case "${1:-}" in
  --status) status_summary; exit 0 ;;
  --compare) ONLY_COMPARE=1 ;;
  --yes) export ASSUME_YES=1 ;;
  "") ;;
  *) die "unknown option: $1" ;;
esac

start_log 06_evqa
python3 "$PROJ/lae/patch_authors.py" --authors-dir "$AUTHORS_DIR"   # LoRA hook (idempotent)

compare() {
  log "building E-VQA comparison tables"
  (cd "$AUTHORS_DIR" && "$PY" "$PROJ/lae/evqa_transfer.py" --authors-dir "$AUTHORS_DIR" \
     --root "$RUNS_DIR/evqa" --compare --out "$RESULTS_DIR" --planned-seeds "$(wc -w <<< "$TRAIN_SEEDS")")
}
if [[ $ONLY_COMPARE == 1 ]]; then compare; exit 0; fi

# models: frozen + every finished (non-smoke) training run not yet evaluated on E-VQA
todo=()
is_ok evqa_frozen || todo+=("frozen")
for ok in "$STATUS_DIR"/train_*.ok; do
  [[ -e "$ok" ]] || continue
  name="$(basename "$ok" .ok)"; name="${name#train_}"
  [[ "$name" == smoke_* ]] && continue
  is_ok "evqa_$name" || todo+=("$name")
done
if (( ${#todo[@]} == 0 )); then log "nothing new to evaluate"; compare; exit 0; fi

cat <<EOT
About to run E-VQA transfer (authors' P20 protocol, 3 cells x 500 questions), one GPU job each:
$(for n in "${todo[@]}"; do [[ $n == frozen ]] && echo "   frozen   (builds the retrieval cache; adapter jobs wait for it)" || echo "   $n   adapter: $RUNS_DIR/train/$n/adapter"; done)
EOT
confirm "Start?" || { log "not started"; exit 0; }

dep=()
if [[ " ${todo[*]} " == *" frozen "* ]]; then
  if job_active evqa_frozen; then log "frozen already queued"
  else EVQA_NAME=frozen ADAPTER="" submit_job evqa_frozen "$PROJ/scripts/slurm/evqa.sub" gpu "$REPRO_TIME"; fi
  dep=(--dependency="afterok:$(cat "$STATUS_DIR/evqa_frozen.jobid")")
fi
for n in "${todo[@]}"; do
  [[ $n == frozen ]] && continue
  job_active "evqa_$n" && { log "$n already queued"; continue; }
  EVQA_NAME="$n" ADAPTER="$RUNS_DIR/train/$n/adapter" \
    submit_job "evqa_$n" "$PROJ/scripts/slurm/evqa.sub" gpu "$REPRO_TIME" ${dep[@]+"${dep[@]}"}
done
status_summary
