#!/usr/bin/env bash
# D4 scRNA extraction for both T-cell donors. Each donor loads its ~5 GB Seurat
# object exactly once; run sequentially so peak RAM only ever holds one object
# (the box is 30 GB / ~19 GB free). Pass a donor to run just one.
set -euo pipefail
cd "$(dirname "$0")/../../.."

if [[ $# -gt 0 ]]; then donors=("$@"); else donors=(donor1 donor2); fi

for d in "${donors[@]}"; do
  echo "=== [tcells-scrna-extract] $d ==="
  Rscript alz/ingest/tcells_scrna_extract.R "$d"
done
echo "[tcells-scrna-extract] done -> data/derived/tcells_incytr_inputs/<donor>/scrna/"
