#!/usr/bin/env bash
# Best-effort dataset.txt backfill for pre-CAC-159 result dirs.
#
# Walks every results/*/ dir that has a scenario.yaml referencing a
# dataset_path, and writes dataset.txt using the CURRENT on-disk dataset's
# fingerprint. Because the dataset may have been regenerated since the run,
# every backfilled record carries a NOTE flagging that the hash reflects the
# dataset as it sits today, not as it was at run time.
#
# When the dir name matches a known Phase-2 sizing bucket (see CAC-159), the
# NOTE includes the historical generator config so a future reader knows what
# to re-run to reproduce.
#
# Usage:
#   tools/backfill_dataset_meta.sh             # dry-run, prints what it would do
#   tools/backfill_dataset_meta.sh --apply     # actually write the files
#
# Idempotent: skips any result dir that already has a dataset.txt.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$ROOT/results"
LIB_DIR="$ROOT/lib"

apply=0
[[ "${1:-}" == "--apply" ]] && apply=1

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "no results dir at $RESULTS_DIR" >&2
  exit 1
fi

skipped=0
written=0
no_dataset=0
already=0

for d in "$RESULTS_DIR"/*/; do
  [[ -d "$d" ]] || continue
  label=$(basename "$d")
  cfg="${d}scenario.yaml"
  if [[ ! -f "$cfg" ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  if [[ -f "${d}dataset.txt" ]]; then
    already=$((already + 1))
    continue
  fi
  dp_rel=$(yq -r '.genai_bench.dataset_path // ""' "$cfg" 2>/dev/null)
  if [[ -z "$dp_rel" ]]; then
    no_dataset=$((no_dataset + 1))
    continue
  fi
  dp_abs="$dp_rel"
  [[ "$dp_abs" != /* ]] && dp_abs="$ROOT/$dp_abs"
  if [[ ! -f "$dp_abs" ]]; then
    echo "skip $label: scenario.yaml references missing $dp_rel" >&2
    skipped=$((skipped + 1))
    continue
  fi

  # Map label prefix → historical generator config (per CAC-159 timeline).
  note="backfilled from current on-disk dataset; pre-CAC-159 run, original dataset bytes may differ"
  case "$label" in
    cs2x-*)
      note="$note. Label suggests CAC-139 forced-eviction sizing: 600 prefixes x 5 questions x 5000 words = 3000 prompts"
      ;;
    stress-*|smoke-postfix-*|postfix-*|rerun-*|cac150-*)
      note="$note. Label era used the reverted 200 x 5 x 5000 = 1000-prompt sizing"
      ;;
  esac

  if (( apply )); then
    python3 "$LIB_DIR/write_dataset_meta.py" \
      --dataset "$dp_abs" \
      --dataset-rel "$dp_rel" \
      --scenario "$(yq -r '.name // ""' "$cfg")" \
      --outdir "$d" \
      --note "$note" \
      || { echo "FAILED: $label" >&2; continue; }
    written=$((written + 1))
  else
    echo "would write: ${d}dataset.txt   (note: $note)"
    written=$((written + 1))
  fi
done

if (( apply )); then
  echo
  echo "backfill complete: wrote=$written, already-had-dataset.txt=$already, no-dataset-scenario=$no_dataset, skipped=$skipped"
else
  echo
  echo "dry run: would write=$written, already-had-dataset.txt=$already, no-dataset-scenario=$no_dataset, skipped=$skipped"
  echo "re-run with --apply to actually write the files."
fi
