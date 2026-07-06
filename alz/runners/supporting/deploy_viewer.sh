#!/usr/bin/env bash
# Deploy viewer output directories to S3 using incremental sync.
# Only uploads files that have changed (by ETag/content hash) — skips the rest.
# The uncompressed *.payload.json is excluded: the browser loads the .gz sidecar.
#
# Usage:
#   bash alz/runners/supporting/deploy_viewer.sh             # unified viewer
#   bash alz/runners/supporting/deploy_viewer.sh tcell       # T-cell viewer
#   bash alz/runners/supporting/deploy_viewer.sh both        # both viewers
set -euo pipefail

PROFILE="bioplat"
BUCKET="s3://voila-buc-00-prod/pocs/incytr"
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

VIEWER="${1:-unified}"   # unified | tcell | both

deploy_viewer() {
    local name="$1"        # "unified" or "tcell"
    local local_dir="$2"   # absolute path to the local viewer output dir
    local s3_prefix="$3"   # s3://bucket/path/ (trailing slash)

    if [[ ! -d "$local_dir" ]]; then
        echo "ERROR: $local_dir does not exist — run the viewer build first." >&2
        exit 1
    fi

    echo "=== Deploying $name viewer ==="
    echo "  local : $local_dir"
    echo "  remote: $s3_prefix"
    echo ""

    # --- Pass 1: edge_slices/ and audit_sources/ — long cache (content-stable) ---
    # These shards don't change unless a pipeline stage re-ran.
    echo "  [1/3] edge_slices/ + audit_sources/ (max-age=86400)…"
    aws s3 sync "$local_dir" "$s3_prefix" \
        --profile "$PROFILE" \
        --delete \
        --exclude "*" \
        --include "edge_slices/*" \
        --include "audit_sources/*" \
        --cache-control "max-age=86400, public" \
        --no-progress

    # --- Pass 2: payload sidecar (.json.gz) — short cache ---
    # Payload changes on every full viewer build.
    # NO Content-Encoding: gzip. The viewer fetches these .gz objects as opaque
    # bytes and decompresses them client-side via DecompressionStream("gzip")
    # (01_state.js:_loadPayload, incytr_global_index.js, incytr_pathways.js).
    # Setting Content-Encoding makes the browser/gateway transparently
    # decompress first, so the manual gunzip then fails on plain JSON
    # ("incorrect header check") and _loadPayload silently falls back to a stale
    # uncompressed payload.
    echo "  [2/3] payload sidecar (max-age=300)…"
    aws s3 sync "$local_dir" "$s3_prefix" \
        --profile "$PROFILE" \
        --exclude "*" \
        --include "*.payload.json.gz" \
        --include "*_gene_node_index.json.gz" \
        --cache-control "max-age=300, public" \
        --no-progress

    # --- Pass 3: index.html + pipeline_overview.html — always revalidate ---
    echo "  [3/3] HTML shell (no-cache)…"
    aws s3 sync "$local_dir" "$s3_prefix" \
        --profile "$PROFILE" \
        --exclude "*" \
        --include "*.html" \
        --cache-control "no-cache, no-store, must-revalidate" \
        --no-progress

    echo ""
    echo "  Done. Files not listed above were skipped (already up to date)."
    local _rest="${s3_prefix#s3://}"          # <bucket>/<key...>
    echo "  S3 URL: https://${_rest%%/*}.s3.amazonaws.com/${_rest#*/}"
}

UNIFIED_LOCAL="$REPO_ROOT/outputs/reports/unified_viewer"
TCELL_LOCAL="$REPO_ROOT/outputs/reports/tcell_viewer"
UNIFIED_S3="$BUCKET/unified_viewer_human/"
TCELL_S3="$BUCKET/tcell_viewer/"

case "$VIEWER" in
    unified)
        deploy_viewer "unified" "$UNIFIED_LOCAL" "$UNIFIED_S3"
        ;;
    tcell)
        deploy_viewer "tcell" "$TCELL_LOCAL" "$TCELL_S3"
        ;;
    both)
        deploy_viewer "unified" "$UNIFIED_LOCAL" "$UNIFIED_S3"
        deploy_viewer "tcell" "$TCELL_LOCAL" "$TCELL_S3"
        ;;
    *)
        echo "Usage: $0 [unified|tcell|both]" >&2
        exit 1
        ;;
esac
