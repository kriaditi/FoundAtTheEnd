#!/usr/bin/env bash
# Stage 2b: derive extra training conditions from the finished standard data (CPU, seconds).
#   * goldfirst      gold-always-first control (same passages, gold at slot 1)
#   * balanced_s<S>  extra balanced seeds, for every seed in TRAIN_SEEDS without files yet
#   * last<P>_s<S>   late-exposure sweep (LATE_RATES x LATE_SEEDS in config.sh), for files not yet present
#
#   bash scripts/02b_extend_data.sh            # runs on the login node, no GPU needed
#   bash scripts/02b_extend_data.sh --status
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
source "$PROJ/scripts/config.sh"
source "$PROJ/scripts/lib.sh"
mkdirs

if [[ "${1:-}" == --status ]]; then
  status_summary
  for c in "$TRAIN_DATA_DIR"/final/*/; do
    c=${c%/}; [[ -f "$c/train.jsonl" ]] || continue
    printf '   %-14s train %6d  dev %4d\n' "$(basename "$c")" "$(wc -l < "$c/train.jsonl")" "$(wc -l < "$c/dev.jsonl")"
  done
  exit 0
fi

start_log 02b_extend_data
is_ok build_data || die "training data not built yet: bash scripts/02_build_data.sh"
need_file "$PY" "(venv missing)"

# balanced seeds that still need files (the build made BALANCED_SEEDS; TRAIN_SEEDS may add more)
new_seeds=()
for s in $TRAIN_SEEDS; do
  [[ -f "$TRAIN_DATA_DIR/final/balanced_s$s/train.jsonl" ]] || new_seeds+=("$s")
done
log "goldfirst: $([[ -f "$TRAIN_DATA_DIR/final/goldfirst/train.jsonl" ]] && echo exists || echo 'to create')"
log "extra balanced seeds to create: ${new_seeds[*]:-none}"
# late-exposure conditions that still need files
late_seeds=()
for s in $LATE_SEEDS; do
  for p in $LATE_RATES; do
    [[ -f "$TRAIN_DATA_DIR/final/last${p}_s$s/train.jsonl" ]] || { late_seeds+=("$s"); break; }
  done
done
log "late-exposure rates: ${LATE_RATES:-none}; seeds to create: ${late_seeds[*]:-none}"

cd "$AUTHORS_DIR"
"$PY" "$PROJ/lae/extend_train_data.py" \
  --data-dir "$TRAIN_DATA_DIR" --authors-dir "$AUTHORS_DIR" \
  --balanced-seeds ${new_seeds[@]+"${new_seeds[@]}"} \
  --late-rates ${LATE_RATES} --late-seeds ${late_seeds[@]+"${late_seeds[@]}"}

mark_ok extend_data "-> $TRAIN_DATA_DIR/final (goldfirst, balanced seeds: $TRAIN_SEEDS, late rates: $LATE_RATES seeds: $LATE_SEEDS)"
log "extend_data OK"
