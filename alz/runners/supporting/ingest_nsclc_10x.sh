#!/usr/bin/env bash
# TODO #2: download the 10x "Aggregate of 900k human NSCLC and normal-adjacent
# cells" Flex dataset (16-plex, 32 samples, Cell Ranger multi 7.1.0) into
# data/external/nsclc_10x/ as the T-cell cohort's cell-type specificity reference.
#
# Public download (no auth) from cf.10xgenomics.com. We fetch:
#   - count_filtered_feature_bc_matrix.h5  (~1.7 GB) — primary input, CSR matrix
#   - count_analysis.tar.gz                (~239 MB) — clustering / projections
#   - count_summary.json                   (~106 KB) — run metrics
#   - aggregation.csv                      (~22 KB)  — per-sample aggregation map
#
# The .cloupe and per-sample web_summaries are intentionally NOT fetched (the
# cloupe is multi-GB and only useful in Loupe Browser).
#
# Downloads are resumable (curl -C -) and skipped when a complete file already
# exists on disk (HTTP content-length match), so re-running is cheap.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DEST="data/external/nsclc_10x"
BASE="https://cf.10xgenomics.com/samples/cell-exp/7.1.0/16plex_900k_32_NSCLC_multiplex/16plex_900k_32_NSCLC_multiplex"
mkdir -p "$DEST"

# Local filename : remote suffix
declare -A FILES=(
  ["sample_feature_bc_matrix.h5"]="_count_filtered_feature_bc_matrix.h5"
  ["analysis.tar.gz"]="_count_analysis.tar.gz"
  ["count_summary.json"]="_count_summary.json"
  ["aggregation.csv"]="_aggregation.csv"
)

for local in "${!FILES[@]}"; do
  url="${BASE}${FILES[$local]}"
  out="$DEST/$local"
  remote_size="$(curl -sIL --max-time 60 "$url" | grep -i '^content-length:' | tail -1 | tr -dc '0-9' || true)"
  if [[ -f "$out" && -n "$remote_size" && "$(stat -c %s "$out")" == "$remote_size" ]]; then
    echo "  [skip] $local already complete ($remote_size bytes)"
    continue
  fi
  echo "  [get ] $local <- $url"
  curl -fL -C - --retry 3 --retry-delay 5 --max-time 3600 -o "$out" "$url"
  got="$(stat -c %s "$out")"
  if [[ -n "$remote_size" && "$got" != "$remote_size" ]]; then
    echo "  [ERR ] $local size mismatch: got $got, expected $remote_size" >&2
    exit 1
  fi
  echo "  [ok  ] $local ($got bytes)"
done

echo "  done -> $DEST"
ls -lh "$DEST"
