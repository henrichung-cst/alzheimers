#!/usr/bin/env bash
# Build the four pair-mode inputs the Incytr driver requires under
# data/derived/incytr_inputs/.
#
# Order:
#   1. alz/incytr_pair/build_pair_seurat.R         -> incytr_obj.rds
#   2. alz/incytr_pair/export_decomposition_for_pair.py -> {pr,ps,py}_yuyu_deconvoluted.csv
#   3. alz/incytr_pair/build_input_gene_list.R       -> allmarkers.csv, HEG_df.csv,
#                                                  input_gene_list.csv
#
# kldata.csv is symlinked in the inputs dir (pointing at
# data/datasets/song/kinase/kldata_pspy.csv) and does not need rebuilding here.
#
# All paths are relative to repo root. Logs to
# outputs/reports/incytr_pair_mode/build.log (also tee'd to stdout).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

INPUTS_DIR="data/derived/incytr_inputs"
LOG_DIR="outputs/reports/incytr_pair_mode"
LOG="$LOG_DIR/build.log"
DEC_DIR="outputs/reports/decomposition/levy_t5"
SPINE_CSV="data/incytr_frozen/v2_46clusters/spines/levy_t5/cluster_spine.csv"

mkdir -p "$INPUTS_DIR" "$LOG_DIR"

# Pre-flight: confirm source data and decomposition parquets exist.
echo "=== Pre-flight (spine=levy_t5) ==="
for f in \
  "data/incytr_frozen/v2_46clusters/incytr input/incytr_obj.rds" \
  "$SPINE_CSV" \
  "$DEC_DIR/protein_per_cluster.parquet" \
  "$DEC_DIR/phospho_per_cluster.parquet" \
  "$DEC_DIR/phospho_per_cluster_pY.parquet" \
  "$INPUTS_DIR/kldata.csv" \
  ; do
  test -e "$f" || { echo "missing: $f"; exit 1; }
  echo "  ok: $f"
done

# Phase 1c needs presto (vectorized Wilcoxon) and future (parallel idents);
# fail loud here rather than 30 min into FindAllMarkers.
echo "  ... checking R deps (presto, future)"
pixi run Rscript -e 'stopifnot(requireNamespace("presto", quietly=TRUE),
                               requireNamespace("future", quietly=TRUE))' \
  || { echo "missing R deps presto/future — see alz/incytr_pair/build_input_gene_list.R"; exit 1; }

{
  echo
  echo "=== $(date -Is) Phase 1a: build_pair_seurat.R (spine=levy_t5) ==="
  pixi run Rscript alz/incytr_pair/build_pair_seurat.R

  echo
  echo "=== $(date -Is) Phase 1b: export_decomposition_for_pair.py ==="
  pixi run python alz/incytr_pair/export_decomposition_for_pair.py --track all

  echo
  echo "=== $(date -Is) Phase 1c: build_input_gene_list.R (spine=levy_t5) ==="
  pixi run Rscript alz/incytr_pair/build_input_gene_list.R

  echo
  echo "=== $(date -Is) Done. Inputs ready under $INPUTS_DIR ==="
  ls -lh "$INPUTS_DIR"
} 2>&1 | tee "$LOG"
