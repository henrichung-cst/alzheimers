#!/usr/bin/env bash
# CR03 Phase-2: Allen HBCA (WHB-10Xv3) log2 expression download + aggregation.
#
# Downloads the two log2 expression h5ads (Neurons ~33 GB, Nonneurons ~17 GB)
# and the WHB-taxonomy / cell_metadata, then streams them through h5py to
# produce data/derived/aggregates/hbca/expression_by_class.csv (genes × 31
# superclusters mean log2 expression).
#
# Runs inside a systemd --user scope with MemoryMax so an OOM is confined
# to this scope instead of triggering the kernel OOM-killer in the calling
# terminal/Claude session.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LOG_DIR="outputs/reports/atlas_reference"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/hbca_download_$(date -u +%Y%m%dT%H%M%SZ).log"
SCOPE="hbca_download_$(date -u +%s).scope"

echo "  scope=$SCOPE  log=$LOG"
echo "  starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

systemd-run --user --scope --unit "$SCOPE" \
  -p MemoryMax=24G -p MemorySwapMax=0 \
  bash -c "pixi run python alz/reference/atlas.py --hbca-download 2>&1 | tee '$LOG'"
