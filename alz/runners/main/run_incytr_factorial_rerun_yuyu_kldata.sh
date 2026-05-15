#!/usr/bin/env bash
# Phase 0c: snapshot pre-fix factorial outputs and re-run with the corrected
# Yuyu-derived kldata (replacing the 5xFAD demo kldata).
#
# See docs/incytr_pair_mode_benchmark_plan.md Phase 0 for context.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="outputs/reports/incytr_factorial"
SNAP_DIR="outputs/reports/incytr_factorial_5xfad_kldata"
KLDATA_NEW="data/datasets/song/kinase/kldata_pspy.csv"

echo "=== Pre-flight ==="
test -s "$KLDATA_NEW" || { echo "missing Yuyu kldata: $KLDATA_NEW"; exit 1; }
test -d "$OUT_DIR" || { echo "no prior factorial outputs at $OUT_DIR"; exit 1; }
test ! -e "$SNAP_DIR" || { echo "snapshot dir $SNAP_DIR already exists; refusing to overwrite"; exit 1; }

echo "  Yuyu kldata: $(wc -l < "$KLDATA_NEW") rows"
echo "  factorial outputs: $(du -sh "$OUT_DIR" | cut -f1)"

echo
echo "=== Snapshot pre-fix outputs ==="
mv "$OUT_DIR" "$SNAP_DIR"
echo "  $OUT_DIR -> $SNAP_DIR"

echo
echo "=== install-incytr ==="
pixi run install-incytr

echo
echo "=== export-factorial-inputs ==="
pixi run export-factorial-inputs

echo
echo "=== incytr-factorial (long-running) ==="
pixi run incytr-factorial

echo
echo "=== viewer rebuild ==="
pixi run viewer

echo
echo "=== Done ==="
echo "Pre-fix snapshot at: $SNAP_DIR"
echo "Corrected outputs at: $OUT_DIR"
