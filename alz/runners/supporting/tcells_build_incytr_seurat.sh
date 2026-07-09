#!/usr/bin/env bash
# Build the per-donor Incytr Seurat objects sequentially under the repository
# memory cap. Pass a donor to run just one.
set -euo pipefail
cd "$(dirname "$0")/../../.."

if command -v pixi >/dev/null 2>&1; then
  PIXI_BIN="$(command -v pixi)"
elif [[ -x "$HOME/.pixi/bin/pixi" ]]; then
  PIXI_BIN="$HOME/.pixi/bin/pixi"
else
  echo "pixi executable not found" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then donors=("$@"); else donors=(donor1 donor2); fi

for donor in "${donors[@]}"; do
  echo "=== [tcells-build-incytr-seurat] $donor ==="
  systemd-run --user --scope -p MemoryMax=26G -p MemorySwapMax=2G \
    "$PIXI_BIN" run Rscript alz/incytr_pair/build_tcells_seurat.R "$donor"
done
echo "[tcells-build-incytr-seurat] done -> data/derived/tcells_incytr_inputs/<donor>/incytr_obj.rds"
