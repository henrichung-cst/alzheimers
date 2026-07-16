#!/usr/bin/env bash
# Reproduce the per-cell marker AUROCs displayed in the original report.
# The original HTML and its native UMAP remain the canonical presentation.
set -euo pipefail
cd "$(dirname "$0")/../.."

PIXI_BIN="${PIXI_BIN:-$HOME/.pixi/bin/pixi}"
MEMMAX="${MEMMAX:-26G}"
MEMSWAPMAX="${MEMSWAPMAX:-2G}"
REEXTRACT="${REEXTRACT:-0}"
OUT="outputs/reports/tcell_labeling"
MARKERS="$OUT/auroc/marker_genes.txt"
REPORT="$OUT/tcell_state_labeling_evidence_original.html"
UMAP="$OUT/umap/umap_label_comparison.png"

if [[ ! -x "$PIXI_BIN" ]]; then
  echo "pixi executable not found: $PIXI_BIN" >&2
  exit 1
fi

mkdir -p "$OUT/auroc"
"$PIXI_BIN" run python alz/analysis/tcell_percell_auroc.py \
  --write-markers "$MARKERS"

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
    echo "=== extracting $donor marker RNA (MemoryMax=$MEMMAX) ==="
    systemd-run --user --scope -p MemoryMax="$MEMMAX" -p MemorySwapMax="$MEMSWAPMAX" \
      "$PIXI_BIN" run Rscript alz/analysis/tcell_export_marker_cells.R "$donor"
  else
    echo "=== reusing $expression (set REEXTRACT=1 for a clean RDS extraction) ==="
  fi
done

"$PIXI_BIN" run python alz/analysis/tcell_percell_auroc.py

test -s "$REPORT"
test -s "$UMAP"
echo "Canonical original report: $REPORT"
echo "Native UMAP: $UMAP"
echo "Reproduced AUROCs: $OUT/reproduced_unadjusted/reproduction_check.csv"
