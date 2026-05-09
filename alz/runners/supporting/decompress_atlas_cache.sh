#!/usr/bin/env bash
# Decompress zstd-compressed files in the external atlas data cache.
#
# Filters:
#   WMB      — WMB full regional h5ad (expression_matrices/WMB-10Xv3/)
#   subset   — WMB gene-subset h5ad (expression_matrices/WMB-10Xv3-subset/)
#   sea_ad   — SEA-AD cell-level h5ad (data/external/sea_ad/)
#   (empty)  — All compressed files in both directories
#
# Usage:
#   bash alz/runners/supporting/decompress_atlas_cache.sh          # all files
#   bash alz/runners/supporting/decompress_atlas_cache.sh WMB      # only WMB fulls
#   bash alz/runners/supporting/decompress_atlas_cache.sh subset   # only WMB subsets
#   bash alz/runners/supporting/decompress_atlas_cache.sh sea_ad   # only SEA-AD
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

CACHE_DIR="data/external/allen_abc"
SEA_AD_DIR="data/external/sea_ad"
FILTER="${1:-}"

if ! command -v zstd &>/dev/null; then
    echo "ERROR: zstd not found. Install with: sudo dnf install zstd"
    exit 1
fi

# ---------------------------------------------------------------------------
# Helper: decompress a list of .zst files found by the caller
# ---------------------------------------------------------------------------
decompress_found() {
    local search_dir="$1"
    shift
    local find_args=("$@")

    local n_files
    n_files=$(find "$search_dir" "${find_args[@]}" 2>/dev/null | wc -l)
    if [[ "$n_files" -eq 0 ]]; then
        echo "No compressed files found."
        return
    fi

    echo "Found $n_files compressed files."
    echo ""

    find "$search_dir" "${find_args[@]}" | sort | while read -r f; do
        relpath="${f#$REPO_ROOT/}"
        size=$(du -h "$f" | cut -f1)
        echo -n "  $relpath ($size) ... "
        zstd -d -f -T0 --rm -q "$f"
        orig="${f%.zst}"
        origsize=$(du -h "$orig" | cut -f1)
        echo "done ($origsize)"
    done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "$FILTER" in
    WMB)
        echo "Decompressing WMB full regional files..."
        decompress_found "$CACHE_DIR" -path "*/WMB-10Xv3/2*" -name "*.zst" -type f
        ;;
    subset)
        echo "Decompressing WMB subset files..."
        decompress_found "$CACHE_DIR" -path "*/WMB-10Xv3-subset/*" -name "*.zst" -type f
        ;;
    sea_ad)
        if [[ ! -d "$SEA_AD_DIR" ]]; then
            echo "ERROR: SEA-AD directory not found at $SEA_AD_DIR"
            exit 1
        fi
        echo "Decompressing SEA-AD files..."
        decompress_found "$SEA_AD_DIR" -name "*.zst" -type f
        ;;
    "")
        echo "Decompressing all compressed files..."
        if [[ -d "$CACHE_DIR" ]]; then
            decompress_found "$CACHE_DIR" -name "*.zst" -type f
        fi
        if [[ -d "$SEA_AD_DIR" ]]; then
            decompress_found "$SEA_AD_DIR" -name "*.zst" -type f
        fi
        ;;
    *)
        echo "ERROR: Unknown filter '$FILTER'"
        echo "Usage: $0 [WMB|subset|sea_ad]"
        exit 1
        ;;
esac

echo ""
echo "Decompression complete."
du -sh "$CACHE_DIR" "$SEA_AD_DIR" 2>/dev/null
