#!/usr/bin/env bash
# Stage 7: counterfactual (no passages, gold removed) and natural-order evaluation,
# frozen reader + every trained adapter (lae/counterfactual.py).
#
#   bash scripts/07_counterfactual.sh            # Slurm: one GPU job per model
#   bash scripts/07_counterfactual.sh --compare  # only rebuild results/COUNTERFACTUAL.md
#   bash scripts/07_counterfactual.sh --status
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

start_log 07_counterfactual
is_ok reproduce_p17 || die "the frozen reproduction must exist first (scripts/01_reproduce.sh)"
python3 "$PROJ/lae/patch_authors.py" --authors-dir "$AUTHORS_DIR"   # LoRA hook (idempotent)

compare() {
  log "building counterfactual tables"
  (cd "$AUTHORS_DIR" && "$PY" "$PROJ/lae/counterfactual.py" --compare --frozen "$RUNS_DIR/p17_repro" \
     --root "$RUNS_DIR/counterfactual" --out "$RESULTS_DIR" --planned-seeds "$(wc -w <<< "$TRAIN_SEEDS")")
}
if [[ $ONLY_COMPARE == 1 ]]; then compare; exit 0; fi

todo=()
is_ok cf_frozen || todo+=("frozen")
for ok in "$STATUS_DIR"/train_*.ok; do
  [[ -e "$ok" ]] || continue
  name="$(basename "$ok" .ok)"; name="${name#train_}"
  [[ "$name" == smoke_* ]] && continue
  is_ok "cf_$name" || todo+=("$name")
done
if (( ${#todo[@]} == 0 )); then log "nothing new to evaluate"; compare; exit 0; fi

cat <<EOT
About to run the counterfactual evaluation (3 conditions x $INFOSEEK_N_KEPT questions), one GPU job each:
$(for n in "${todo[@]}"; do [[ $n == frozen ]] && echo "   frozen" || echo "   $n   adapter: $RUNS_DIR/train/$n/adapter"; done)
  about 1,500 reader calls per model; expect roughly 20 to 30 min each on this GPU
EOT
confirm "Start?" || { log "not started"; exit 0; }

for n in "${todo[@]}"; do
  job_active "cf_$n" && { log "$n already queued"; continue; }
  if [[ $n == frozen ]]; then adapter=""; else adapter="$RUNS_DIR/train/$n/adapter"; fi
  CF_NAME="$n" ADAPTER="$adapter" submit_job "cf_$n" "$PROJ/scripts/slurm/counterfactual.sub" gpu "$REPRO_TIME"
done
status_summary
