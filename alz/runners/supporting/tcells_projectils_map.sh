#!/usr/bin/env bash
# Sequential ProjecTILs mapping per donor — standalone from the extract step so
# this can be re-run without redoing aggexp/markers/cell_counts work that is
# already on disk. Sequential to avoid two large Seurat objects + the
# CD4/CD8 references held in memory simultaneously.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PIXI_BIN="${PIXI_BIN:-$(command -v pixi || true)}"
if [[ -z "$PIXI_BIN" && -x "$HOME/.pixi/bin/pixi" ]]; then
  PIXI_BIN="$HOME/.pixi/bin/pixi"
fi
if [[ -z "$PIXI_BIN" ]]; then
  echo "pixi executable not found; set PIXI_BIN=/path/to/pixi" >&2
  exit 127
fi

if [[ $# -gt 0 ]]; then donors=("$@"); else donors=(donor1 donor2); fi

for d in "${donors[@]}"; do
  echo "=== [tcells-projectils-map] $d ==="
  "$PIXI_BIN" run Rscript alz/ingest/tcells_projectils_map.R "$d"
done

echo "[tcells-projectils-map] all donors done."
