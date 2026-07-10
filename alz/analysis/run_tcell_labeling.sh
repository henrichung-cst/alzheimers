#!/usr/bin/env bash
# Assign cycle-independent per-cell T-cell states from RNA/ADT marker evidence.
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
"$PIXI_BIN" run python alz/analysis/tcell_state_labels.py --write-markers "$MARKERS"

marker_coverage_ok() {
  local expression="$1"
  "$PIXI_BIN" run python - "$MARKERS" "$expression" <<'PY'
import csv
import sys
from pathlib import Path

marker_path, expression_path = map(Path, sys.argv[1:])
if not expression_path.is_file():
    raise SystemExit(1)
required = {line.strip() for line in marker_path.read_text().splitlines() if line.strip()}
with expression_path.open(newline="") as handle:
    columns = set(next(csv.reader(handle)))
raise SystemExit(0 if required <= columns else 1)
PY
}

for donor in donor1 donor2; do
  expression="$OUT/auroc/${donor}_marker_cell_expr.csv"
  if [[ "$REEXTRACT" == "1" || ! -s "$expression" ]] || \
     ! marker_coverage_ok "$expression"; then
    echo "=== extracting $donor non-cycle marker RNA/ADT (MemoryMax=$MEMMAX) ==="
    systemd-run --user --scope -p MemoryMax="$MEMMAX" -p MemorySwapMax="$MEMSWAPMAX" \
      "$PIXI_BIN" run Rscript alz/analysis/tcell_export_marker_cells.R "$donor"
  else
    echo "=== reusing $expression ==="
  fi
done

"$PIXI_BIN" run python alz/analysis/tcell_state_labels.py
"$PIXI_BIN" run python alz/analysis/tcell_state_evidence.py
"$PIXI_BIN" run python alz/analysis/tcell_native_umap_plots.py
"$QUARTO_BIN" render \
  "$OUT/tcell_state_labeling_evidence_percell.qmd" \
  --to html \
  --execute-dir .

echo "Per-cell label report: $OUT/tcell_state_labeling_evidence_percell.html"
