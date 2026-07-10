#!/usr/bin/env bash
# Rebuild CITE-seq/RNA evidence, definitive per-cell states, figures, and report.
set -euo pipefail
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$PWD"

PIXI_BIN="${PIXI_BIN:-$(command -v pixi 2>/dev/null || true)}"
if [[ -z "$PIXI_BIN" && -x "$HOME/.pixi/bin/pixi" ]]; then
  PIXI_BIN="$HOME/.pixi/bin/pixi"
fi
if [[ -z "$PIXI_BIN" ]]; then
  echo "pixi executable not found" >&2
  exit 1
fi

MEMMAX="${MEMMAX:-26G}"
OUT="outputs/reports/tcell_labeling"
mkdir -p "$OUT/auroc" "$OUT/adt"

"$PIXI_BIN" run python alz/analysis/tcell_percell_auroc.py \
  --write-markers "$OUT/auroc/marker_genes.txt"

for donor in donor1 donor2; do
  systemd-run --user --scope -p MemoryMax="$MEMMAX" -p MemorySwapMax=2G \
    "$PIXI_BIN" run Rscript alz/analysis/tcell_export_marker_cells.R "$donor"
done

"$PIXI_BIN" run python alz/analysis/tcell_state_labels.py
"$PIXI_BIN" run python alz/analysis/tcell_native_umap_plots.py
(
  cd "$OUT"
  QUARTO_BIN="${QUARTO_BIN:-$(command -v quarto 2>/dev/null || true)}"
  if [[ -z "$QUARTO_BIN" && -x "$HOME/.local/bin/quarto" ]]; then
    QUARTO_BIN="$HOME/.local/bin/quarto"
  fi
  if [[ -z "$QUARTO_BIN" ]]; then
    echo "quarto executable not found" >&2
    exit 1
  fi
  "$PIXI_BIN" run --manifest-path "$PROJECT_ROOT/pixi.toml" \
    "$QUARTO_BIN" render tcell_state_labeling_evidence.qmd
)
