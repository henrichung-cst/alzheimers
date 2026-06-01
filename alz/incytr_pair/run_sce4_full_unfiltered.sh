#!/usr/bin/env bash
# Controlled sce4 reproduction run.
#
# This intentionally stops after writing the driver's unfiltered contrast-level
# parquets. Do not apply filter_significant_paths.py here: verify_sce4_full.R
# needs pre-cap outputs and applies the sce4 gate itself.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PIXI="${PIXI:-/home/hchung/.pixi/bin/pixi}"
DRIVER="alz/incytr_pair/incytr_commandline.R"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-outputs/reports/incytr_pair_mode/_sce4_full_q0}"
LOG_DIR="outputs/reports/incytr_pair_mode"
LOG="${LOG_OVERRIDE:-${LOG_DIR}/sce4_full_unfiltered_run.log}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export SCE4_GENEUSE_DIR="${SCE4_GENEUSE_DIR:-data/incytr_frozen/sce4_geneuse}"
export OUTPUT_DIR_OVERRIDE="$OUTPUT_DIR"
export NBOOT="${FULL_NBOOT:-0}"
export NPAIR_WORKERS="${NPAIR_WORKERS:-1}"
export NPERM_WORKERS="${NPERM_WORKERS:-1}"
export CHUNK_PARALLEL="${CHUNK_PARALLEL:-1}"
export N_CHUNK_MULT="${N_CHUNK_MULT:-48}"

run_one() {
  local geno="$1"
  local age="$2"
  local c1="ma_${age}_${geno}"
  local c2="ma_${age}_WTyp"
  echo "=== $(date -Is) START ${c1} vs ${c2} ==="
  "$PIXI" run Rscript "$DRIVER" "$c1" "$c2"
  echo "=== $(date -Is) DONE  ${c1} vs ${c2} ==="
}

{
  echo "=== sce4 full unfiltered run ==="
  echo "output_dir=$OUTPUT_DIR"
  echo "SCE4_GENEUSE_DIR=$SCE4_GENEUSE_DIR"
  echo "NBOOT=$NBOOT NPAIR_WORKERS=$NPAIR_WORKERS NPERM_WORKERS=$NPERM_WORKERS CHUNK_PARALLEL=$CHUNK_PARALLEL N_CHUNK_MULT=$N_CHUNK_MULT"

  run_one AppP 2mo
  run_one ApTt 2mo
  run_one Ttau 2mo
  run_one AppP 4mo
  run_one ApTt 4mo
  run_one Ttau 4mo
  run_one AppP 6mo
  run_one ApTt 6mo
  run_one Ttau 6mo

  echo "=== $(date -Is) all contrasts complete ==="
  find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*_incytr_output.parquet' -printf '%f %s\n'
} 2>&1 | tee "$LOG"
