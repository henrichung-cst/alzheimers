#!/usr/bin/env bash
# Compress external atlas reference data (Allen ABC + SEA-AD) with zstd.
#
# Three compression tiers:
#   tier1 / WMB     — WMB full regional h5ad (89 GB → ~24 GB)
#   tier2 / sea_ad  — Unused SEA-AD cell-level h5ad (23.5 GB → ~5 GB)
#   tier3 / subset  — WMB gene-subset h5ad (51 GB → ~14 GB)
#   (no argument)   — All three tiers
#
# Exclusions (kept uncompressed for runtime use):
#   - effect_sizes{,_early,_late}.h5ad (read by alz/bulk_mea/attribute.py)
#   - cell_metadata_with_cluster_annotation.csv (read by ABC cache API)
#
# To decompress: bash alz/runners/supporting/decompress_atlas_cache.sh [filter]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

CACHE_DIR="data/external/allen_abc"
SEA_AD_DIR="data/external/sea_ad"
FILTER="${1:-}"

# SEA-AD files that must stay uncompressed (read at runtime)
SEA_AD_KEEP=(
    "effect_sizes.h5ad"
    "effect_sizes_early.h5ad"
    "effect_sizes_late.h5ad"
)

if ! command -v zstd &>/dev/null; then
    echo "ERROR: zstd not found. Install with: sudo dnf install zstd"
    exit 1
fi

# ---------------------------------------------------------------------------
# Helper: compress a list of files
# ---------------------------------------------------------------------------
compress_files() {
    local count=0
    while IFS= read -r f; do
        relpath="${f#$REPO_ROOT/}"
        size=$(du -h "$f" | cut -f1)
        echo -n "  $relpath ($size) ... "
        zstd -3 -T0 --rm -q "$f"
        newsize=$(du -h "${f}.zst" | cut -f1)
        echo "done ($newsize)"
        ((count++)) || true
    done
    echo "  ($count files compressed)"
}

# ---------------------------------------------------------------------------
# Build manifest before compressing (captures original sizes + checksums)
# ---------------------------------------------------------------------------
build_manifest() {
    echo "Building data manifest..."
    local MANIFEST="$CACHE_DIR/MANIFEST.json"
    python3 -c "
import json, os, hashlib, datetime

dirs_to_scan = ['$CACHE_DIR', '$SEA_AD_DIR']
files = []
for scan_dir in dirs_to_scan:
    if not os.path.isdir(scan_dir):
        continue
    for root, dirs, fnames in os.walk(scan_dir):
        for fname in sorted(fnames):
            if fname in ('MANIFEST.json', '_downloaded_data.json', '_manifest_last_used.txt'):
                continue
            if fname.endswith('.zst'):
                continue
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath)
            if size < 1_000_000:  # skip files < 1 MB
                continue
            relpath = os.path.relpath(fpath, '$REPO_ROOT')
            h = hashlib.sha256()
            with open(fpath, 'rb') as f:
                h.update(f.read(64 * 1024 * 1024))
            files.append({
                'path': relpath,
                'size_bytes': size,
                'size_human': f'{size / 1073741824:.1f} GB' if size > 1073741824 else f'{size / 1048576:.0f} MB',
                'sha256_first_64mb': h.hexdigest(),
            })

manifest = {
    'source': 'Allen Institute Brain Cell Atlas + SEA-AD',
    'datasets': {
        'WMB-10Xv3': {
            'expression_version': '20230630',
            'metadata_version': '20241115',
            'regions': 13,
            'description': 'Whole Mouse Brain 10X v3 single-cell transcriptomics',
        },
        'WMB-10Xv3-subset': {
            'description': 'Gene-subset extraction (~7,100 genes) of WMB-10Xv3',
        },
        'SEA-AD': {
            'description': 'Seattle Alzheimer Disease Brain Cell Atlas (MTG region)',
            'runtime_files': ['effect_sizes.h5ad', 'effect_sizes_early.h5ad', 'effect_sizes_late.h5ad'],
        },
    },
    'compression': {
        'method': 'zstd',
        'level': 3,
        'tiers': {
            'tier1': 'WMB full regional h5ad (expression_matrices/WMB-10Xv3/)',
            'tier2': 'Unused SEA-AD cell-level h5ad (sea_ad/ large files)',
            'tier3': 'WMB gene-subset h5ad (expression_matrices/WMB-10Xv3-subset/)',
        },
        'note': 'Originals removed after compression. Decompress with: bash alz/runners/supporting/decompress_atlas_cache.sh [filter]',
    },
    're_acquisition': {
        'install': 'pip install git+https://github.com/alleninstitute/abc_atlas_access.git',
        'command': 'python alz/atlas_reference.py --run',
    },
    'archived_date': datetime.date.today().isoformat(),
    'files': files,
}

with open('$MANIFEST', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'Manifest written: {len(files)} files catalogued')
"
}

# ---------------------------------------------------------------------------
# Tier 1: WMB full regional h5ad files
# ---------------------------------------------------------------------------
compress_tier1() {
    local target="$CACHE_DIR/expression_matrices/WMB-10Xv3"
    if [[ ! -d "$target" ]]; then
        echo "Tier 1: WMB full regionals — directory not found, skipping"
        return
    fi
    local -a files
    mapfile -t files < <(find "$target" -name "*.h5ad" -type f | sort)
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "Tier 1: WMB full regionals — already compressed (0 h5ad files)"
        return
    fi
    echo ""
    echo "Tier 1: Compressing WMB full regional h5ad (${#files[@]} files)..."
    printf '%s\n' "${files[@]}" | compress_files
}

# ---------------------------------------------------------------------------
# Tier 2: Unused SEA-AD cell-level h5ad files
# ---------------------------------------------------------------------------
compress_tier2() {
    if [[ ! -d "$SEA_AD_DIR" ]]; then
        echo "Tier 2: SEA-AD — directory not found, skipping"
        return
    fi

    local exclude_args=()
    for keep in "${SEA_AD_KEEP[@]}"; do
        exclude_args+=(! -name "$keep")
    done

    local -a files
    mapfile -t files < <(find "$SEA_AD_DIR" -name "*.h5ad" -type f "${exclude_args[@]}" | sort)
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "Tier 2: SEA-AD unused files — already compressed (0 eligible h5ad files)"
        return
    fi
    echo ""
    echo "Tier 2: Compressing unused SEA-AD h5ad (${#files[@]} files, excluding runtime effect_sizes)..."
    printf '%s\n' "${files[@]}" | compress_files
}

# ---------------------------------------------------------------------------
# Tier 3: WMB gene-subset h5ad files
# ---------------------------------------------------------------------------
compress_tier3() {
    local target="$CACHE_DIR/expression_matrices/WMB-10Xv3-subset"
    if [[ ! -d "$target" ]]; then
        echo "Tier 3: WMB subsets — directory not found, skipping"
        return
    fi
    local -a files
    mapfile -t files < <(find "$target" -name "*.h5ad" -type f | sort)
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "Tier 3: WMB subsets — already compressed (0 h5ad files)"
        return
    fi
    echo ""
    echo "Tier 3: Compressing WMB gene-subset h5ad (${#files[@]} files)..."
    printf '%s\n' "${files[@]}" | compress_files
}

# ---------------------------------------------------------------------------
# Large CSV compression (metadata, excluding the runtime metadata CSV)
# ---------------------------------------------------------------------------
compress_csvs() {
    echo ""
    echo "Compressing large CSVs (excluding WMB cell metadata)..."
    find "$CACHE_DIR" -name "*.csv" -type f -size +1M \
        ! -name "cell_metadata_with_cluster_annotation.csv" \
        | sort | compress_files
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "$FILTER" in
    tier1|WMB)
        compress_tier1
        ;;
    tier2|sea_ad)
        compress_tier2
        ;;
    tier3|subset)
        compress_tier3
        ;;
    "")
        build_manifest
        compress_tier1
        compress_tier2
        compress_tier3
        compress_csvs
        ;;
    *)
        echo "ERROR: Unknown filter '$FILTER'"
        echo "Usage: $0 [tier1|WMB|tier2|sea_ad|tier3|subset]"
        exit 1
        ;;
esac

echo ""
echo "Compression complete."
du -sh "$CACHE_DIR" "$SEA_AD_DIR" 2>/dev/null
