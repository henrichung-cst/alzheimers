#!/usr/bin/env bash
# Supporting runner for WMB expression export (unified attribution dependency).
# Produces outputs/reports/wmb_expression/wmb_kinase_expression.csv which is required
# by the live pipeline before running kinase attribution.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/hchung/.local/share/mamba/envs/alzheimers/bin/python
LOG="outputs/reports/wmb_expression/wmb_expression.log"
mkdir -p outputs/reports/wmb_expression

N_H5AD=$(find data/external/allen_abc/expression_matrices/WMB-10Xv3/ \
    -name "*-log2.h5ad" 2>/dev/null | wc -l)
if [ "$N_H5AD" -lt 13 ]; then
    echo "ERROR: Only $N_H5AD/13 WMB region h5ad files found."
    echo "Run run_wmb_download.sh first to download all regions."
    exit 1
fi

echo "=== WMB expression export started at $(date) ===" | tee "$LOG"
$PYTHON code/wmb_expression.py --run 2>&1 | tee -a "$LOG"
echo "=== WMB expression export finished at $(date) ===" | tee -a "$LOG"
