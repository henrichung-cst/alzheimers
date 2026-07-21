#!/usr/bin/env bash
set -euo pipefail

# Regenerate the floor-99 kinase sidechain artifacts and both viewer slices.
# This is intentionally explicit: the bridge must finish before edge reduction,
# and both edge reductions must finish before either viewer packages its slices.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Rebuilding Song bridge ==="
pixi run python -m alz.cross_reference.kinase_incytr_bridge --cohort song

echo "=== Rebuilding 5xFAD bridge (cortex + hippocampus) ==="
pixi run python -m alz.cross_reference.kinase_incytr_bridge \
  --cohort fivexfad --tissue both

echo "=== Rebuilding T-cell donor1 bridge ==="
pixi run python -m alz.cross_reference.kinase_incytr_bridge \
  --cohort tcells --donor donor1

echo "=== Rebuilding Song edges ==="
pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort song

echo "=== Rebuilding 5xFAD edges ==="
pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort fivexfad

echo "=== Rebuilding T-cell edges ==="
pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort tcells

echo "=== Rebuilding unified 5xFAD/Song viewer slices ==="
pixi run viewer

echo "=== Rebuilding T-cell viewer slices ==="
pixi run tcell-viewer

echo "=== Verifying rebuilt sidechain payload contracts ==="
pixi run python -m alz.viewer.verify_payload_contract \
  outputs/reports/unified_viewer/unified_viewer.payload.json \
  outputs/reports/tcell_viewer/tcell_viewer.payload.json

echo "=== Floor-99 sidechain reconciliation complete ==="
