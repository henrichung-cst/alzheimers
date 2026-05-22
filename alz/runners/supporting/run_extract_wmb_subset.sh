#!/usr/bin/env bash
# Extract gene-subset h5ad files from the full WMB atlas.
#
# This is a one-time setup step that creates small h5ad files containing only
# the genes needed by the pipeline (~6,800 of 32,285). After extraction, the
# full h5ad files can stay zstd-compressed permanently, saving ~70 GB of disk.
#
# Prerequisites:
#   - Full WMB h5ad files must be decompressed (done automatically if .zst exist)
#   - data_ingest.py --phospho-match must have run (produces total_proteome_genes.txt)
#
# Usage:
#   bash alz/runners/supporting/run_extract_wmb_subset.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

CACHE_DIR="data/external/allen_abc"
SUBSET_DIR="$CACHE_DIR/expression_matrices/WMB-10Xv3-subset"

echo "============================================================"
echo "WMB Gene-Subset Extraction Pipeline"
echo "============================================================"

# Check if subsets already exist
if [[ -d "$SUBSET_DIR" ]] && [[ -f "$SUBSET_DIR/MANIFEST.json" ]]; then
    N_SUBSETS=$(find "$SUBSET_DIR" -name "*.h5ad" | wc -l)
    if [[ "$N_SUBSETS" -eq 13 ]]; then
        echo "Subset files already exist ($N_SUBSETS regions). Skipping."
        echo "Delete $SUBSET_DIR to force re-extraction."
        du -sh "$SUBSET_DIR"
        exit 0
    fi
fi

# Check if gene list exists
GENE_LIST="outputs/reports/data_ingest/total_proteome_genes.txt"
if [[ ! -f "$GENE_LIST" ]]; then
    echo "ERROR: Proteome gene list not found at $GENE_LIST"
    echo "Run first: python alz/ingest/song.py --phospho-match"
    exit 1
fi

# Step 1: Decompress WMB h5ad files if needed
N_ZST=$(find "$CACHE_DIR/expression_matrices/WMB-10Xv3" -name "*.zst" -type f 2>/dev/null | wc -l)
N_H5AD=$(find "$CACHE_DIR/expression_matrices/WMB-10Xv3" -name "*.h5ad" ! -name "*.zst" -type f 2>/dev/null | wc -l)

echo ""
echo "Current state: $N_H5AD decompressed, $N_ZST compressed WMB files"

NEED_RECOMPRESS=false
if [[ "$N_H5AD" -lt 13 ]] && [[ "$N_ZST" -gt 0 ]]; then
    echo ""
    echo "Step 1: Decompressing WMB files..."
    bash alz/runners/supporting/decompress_atlas_cache.sh WMB
    NEED_RECOMPRESS=true
elif [[ "$N_H5AD" -lt 13 ]]; then
    echo "ERROR: Not enough h5ad files found and no .zst files to decompress."
    echo "Expected 13 WMB regional files."
    exit 1
fi

# Step 2: Extract subsets
echo ""
echo "Step 2: Extracting gene subsets..."
python alz/runners/supporting/extract_wmb_gene_subset.py

# Step 3: Re-compress full files if we decompressed them
if [[ "$NEED_RECOMPRESS" == true ]]; then
    echo ""
    echo "Step 3: Re-compressing full WMB files..."
    if [[ -f "alz/runners/supporting/compress_atlas_cache.sh" ]]; then
        bash alz/runners/supporting/compress_atlas_cache.sh WMB
    else
        echo "  Compress script not found. You can manually compress with:"
        echo "  find $CACHE_DIR/expression_matrices/WMB-10Xv3 -name '*.h5ad' -exec zstd -3 --rm {} \\;"
    fi
fi

echo ""
echo "============================================================"
echo "Done. Disk usage:"
echo "  Subsets:    $(du -sh "$SUBSET_DIR" | cut -f1)"
echo "  Full cache: $(du -sh "$CACHE_DIR" | cut -f1)"
echo "============================================================"
