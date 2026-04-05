#!/usr/bin/env bash
# Decompress zstd-compressed files in the Allen Brain Cell Atlas cache.
#
# Use this if you need to re-run wmb_expression.py against the raw h5ad data
# (e.g., to regenerate expression CSVs with different parameters).
#
# Usage:
#   bash code/runners/supporting/decompress_atlas_cache.sh          # all files
#   bash code/runners/supporting/decompress_atlas_cache.sh WMB      # only WMB files
#   bash code/runners/supporting/decompress_atlas_cache.sh Aging    # only Aging files
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

CACHE_DIR="data/external/allen_abc"
FILTER="${1:-}"

if [[ ! -d "$CACHE_DIR" ]]; then
    echo "ERROR: Atlas cache not found at $CACHE_DIR"
    exit 1
fi

if ! command -v zstd &>/dev/null; then
    echo "ERROR: zstd not found. Install with: sudo dnf install zstd"
    exit 1
fi

# Apply optional filter
FIND_ARGS=(-name "*.zst" -type f)
if [[ "$FILTER" == "WMB" ]]; then
    FIND_ARGS=(-path "*/WMB-*" -name "*.zst" -type f)
    echo "Decompressing WMB files only..."
elif [[ "$FILTER" == "Aging" ]]; then
    FIND_ARGS=(-path "*/Zeng-Aging*" -name "*.zst" -type f)
    echo "Decompressing Aging files only..."
else
    echo "Decompressing all compressed files..."
fi

N_FILES=$(find "$CACHE_DIR" "${FIND_ARGS[@]}" | wc -l)
if [[ "$N_FILES" -eq 0 ]]; then
    echo "No compressed files found."
    exit 0
fi

echo "Found $N_FILES compressed files."
echo ""

find "$CACHE_DIR" "${FIND_ARGS[@]}" | sort | while read -r f; do
    relpath="${f#$CACHE_DIR/}"
    size=$(du -h "$f" | cut -f1)
    echo -n "  $relpath ($size) ... "
    zstd -d --rm -q "$f"
    orig="${f%.zst}"
    origsize=$(du -h "$orig" | cut -f1)
    echo "done ($origsize)"
done

echo ""
echo "Decompression complete."
du -sh "$CACHE_DIR"
