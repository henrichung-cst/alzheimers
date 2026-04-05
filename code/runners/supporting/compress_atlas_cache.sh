#!/usr/bin/env bash
# Compress the Allen Brain Cell Atlas cache (h5ad + large CSVs) with zstd.
#
# Reduces ~115 GB of uncompressed data to ~26 GB.  Originals are removed
# after successful compression (zstd --rm).  The live pipeline reads only
# the pre-computed CSV outputs, not the raw h5ad files, so this is safe
# for normal operation.
#
# To decompress, run: bash code/runners/supporting/decompress_atlas_cache.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

CACHE_DIR="data/external/allen_abc"

if [[ ! -d "$CACHE_DIR" ]]; then
    echo "ERROR: Atlas cache not found at $CACHE_DIR"
    exit 1
fi

if ! command -v zstd &>/dev/null; then
    echo "ERROR: zstd not found. Install with: sudo dnf install zstd"
    exit 1
fi

# Build manifest before compressing (captures original sizes + checksums)
echo "Building data manifest..."
MANIFEST="$CACHE_DIR/MANIFEST.json"
python3 -c "
import json, os, hashlib, datetime

cache_dir = '$CACHE_DIR'
files = []
for root, dirs, fnames in os.walk(cache_dir):
    for fname in sorted(fnames):
        if fname in ('MANIFEST.json', '_downloaded_data.json', '_manifest_last_used.txt'):
            continue
        if fname.endswith('.zst'):
            continue
        fpath = os.path.join(root, fname)
        size = os.path.getsize(fpath)
        if size < 1_000_000:  # skip files < 1 MB
            continue
        relpath = os.path.relpath(fpath, cache_dir)
        # sha256 of first 64 MB (full checksum too slow for 10+ GB files)
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
    'source': 'Allen Institute Brain Cell Atlas (abc_atlas_access)',
    'datasets': {
        'WMB-10Xv3': {
            'expression_version': '20230630',
            'metadata_version': '20241115',
            'regions': 13,
            'description': 'Whole Mouse Brain 10X v3 single-cell transcriptomics',
        },
        'Zeng-Aging-Mouse-10Xv3': {
            'expression_version': '20241130',
            'metadata_version': '20250131',
            'description': 'Aging Mouse Brain single-cell transcriptomics',
        },
    },
    'compression': {
        'method': 'zstd',
        'level': 3,
        'note': 'Originals removed after compression. Decompress with: bash code/runners/supporting/decompress_atlas_cache.sh',
    },
    're_acquisition': {
        'install': 'pip install git+https://github.com/alleninstitute/abc_atlas_access.git',
        'command': 'python code/atlas_reference.py --run',
    },
    'archived_date': datetime.date.today().isoformat(),
    'files': files,
}

with open('$MANIFEST', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'Manifest written: {len(files)} files catalogued')
"

# Compress all h5ad files
echo ""
echo "Compressing h5ad files..."
find "$CACHE_DIR" -name "*.h5ad" -type f | sort | while read -r f; do
    relpath="${f#$CACHE_DIR/}"
    size=$(du -h "$f" | cut -f1)
    echo -n "  $relpath ($size) ... "
    zstd -3 --rm -q "$f"
    newsize=$(du -h "${f}.zst" | cut -f1)
    echo "done ($newsize)"
done

# Compress large CSV files (>1 MB)
echo ""
echo "Compressing large CSVs..."
find "$CACHE_DIR" -name "*.csv" -type f -size +1M | sort | while read -r f; do
    relpath="${f#$CACHE_DIR/}"
    size=$(du -h "$f" | cut -f1)
    echo -n "  $relpath ($size) ... "
    zstd -3 --rm -q "$f"
    newsize=$(du -h "${f}.zst" | cut -f1)
    echo "done ($newsize)"
done

echo ""
echo "Compression complete."
du -sh "$CACHE_DIR"
