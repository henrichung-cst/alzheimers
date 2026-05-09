#!/usr/bin/env bash
# Song snRNA-seq integration: pseudobulk, specificity, concordance
# Prerequisite: 170_gex_celltypes_00.h5ad must exist
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR/.."

H5AD="data/datasets/song/transcriptomics/170_gex_celltypes_00.h5ad"
if [[ ! -f "$H5AD" ]]; then
    echo "ERROR: h5ad file not found: $H5AD"
    echo "This file is required for snRNA integration."
    exit 1
fi

echo "=== Song snRNA Integration ==="
python alz/snrna_integration.py --run
echo "=== Done ==="
