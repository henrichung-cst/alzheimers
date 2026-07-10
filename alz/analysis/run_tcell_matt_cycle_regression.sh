#!/usr/bin/env bash
# Reproduce Matt's per-cell AUROC report and add donor-wise cell-cycle regression.
# This workflow intentionally does not assign labels, rerun Incytr, or update viewers.
set -euo pipefail
cd "$(dirname "$0")/../.."

PIXI_BIN="${PIXI_BIN:-$HOME/.pixi/bin/pixi}"
QUARTO_BIN="${QUARTO_BIN:-$HOME/.local/bin/quarto}"
MEMMAX="${MEMMAX:-26G}"
MEMSWAPMAX="${MEMSWAPMAX:-2G}"
REEXTRACT="${REEXTRACT:-0}"
OUT="outputs/reports/tcell_labeling"
MARKERS="$OUT/auroc/marker_genes.txt"

if [[ ! -x "$PIXI_BIN" ]]; then
  echo "pixi executable not found: $PIXI_BIN" >&2
  exit 1
fi
if [[ ! -x "$QUARTO_BIN" ]]; then
  echo "quarto executable not found: $QUARTO_BIN" >&2
  exit 1
fi

mkdir -p "$OUT/auroc"
"$PIXI_BIN" run python alz/analysis/tcell_percell_auroc.py \
  --write-markers "$MARKERS"

for donor in donor1 donor2; do
  expression="$OUT/auroc/${donor}_marker_cell_expr.csv"
  if [[ "$REEXTRACT" == "1" || ! -s "$expression" ]]; then
    echo "=== extracting $donor marker RNA and cycle covariates (MemoryMax=$MEMMAX) ==="
    systemd-run --user --scope -p MemoryMax="$MEMMAX" -p MemorySwapMax="$MEMSWAPMAX" \
      "$PIXI_BIN" run Rscript alz/analysis/tcell_export_marker_cells.R "$donor"
  else
    echo "=== reusing $expression (set REEXTRACT=1 for a clean RDS extraction) ==="
  fi
done

"$PIXI_BIN" run python alz/analysis/tcell_percell_auroc.py

(
  cd "$OUT"
  "$PIXI_BIN" run --manifest-path ../../../pixi.toml \
    "$QUARTO_BIN" render tcell_state_labeling_evidence_cycle_regressed.qmd
)

ARCHIVE="$OUT/archive/matt_original_snapshot_2026-07-10/tcell_state_labeling_evidence_matt.html"
test -s "$ARCHIVE"
test -s "$OUT/tcell_state_labeling_evidence_cycle_regressed.html"
echo "Historical Matt snapshot: $ARCHIVE"
echo "Revised report: $OUT/tcell_state_labeling_evidence_cycle_regressed.html"
