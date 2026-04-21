#!/bin/bash
# Run factorial permutation tests one contrast at a time as separate processes.
# Each contrast writes to its own file, then all are concatenated at the end.
#
# Usage:
#   bash code/integration/run_factorial_permutations.sh
#   bash code/integration/run_factorial_permutations.sh 1000   # custom N

set -euo pipefail
cd "$(dirname "$0")/../.."

N_PERM="${1:-10000}"
OUT_DIR="code/integration/intermediates/factorial/all_pairs/aggregation"
FINAL_FILE="${OUT_DIR}/backbone_permutation_pvalues_by_contrast.csv"
BB_FILE="${OUT_DIR}/backbone_recurrence_by_contrast.csv"
TMP_DIR="${OUT_DIR}/perm_tmp"

CONTRASTS=(App_2mo App_4mo App_6mo Tau_2mo Tau_4mo Tau_6mo ApTt_2mo ApTt_4mo ApTt_6mo)

echo "=== Factorial Permutation Tests (${N_PERM} iterations x 9 contrasts) ==="
echo "  Output: ${FINAL_FILE}"

mkdir -p "${TMP_DIR}"

for i in "${!CONTRASTS[@]}"; do
    contrast="${CONTRASTS[$i]}"
    tmp_file="${TMP_DIR}/perm_${contrast}.csv"

    # Skip if this contrast's temp file already exists
    if [ -f "${tmp_file}" ]; then
        n_lines=$(wc -l < "${tmp_file}")
        echo "--- Contrast $((i+1))/9: ${contrast} --- CACHED (${n_lines} lines)"
        continue
    fi

    echo ""
    echo "--- Contrast $((i+1))/9: ${contrast} ---"

    pixi run python3 -u - <<PYEOF
import sys, os, gc
sys.path.insert(0, os.path.join(os.getcwd(), 'code/integration'))
sys.path.insert(0, os.path.join(os.getcwd(), 'code/integration/adapters'))

import pandas as pd
import config_integration as icfg
from compute_kinase_support_factorial import load_shared_data
from aggregate_factorial import _run_permutation_one_contrast

contrast = '${contrast}'
n_perm = ${N_PERM}

print(f'Loading shared kinase data for {contrast} only...')
shared = load_shared_data(contrast_filter=contrast)

print(f'Loading backbone recurrence for {contrast}...')
bb = pd.read_csv('${BB_FILE}', dtype={'contrast': str, 'receiver': str,
                  'Receptor': str, 'EM': str, 'Target': str},
                  usecols=['contrast', 'receiver', 'Receptor', 'EM', 'Target'])
bb_c = bb[bb['contrast'] == contrast].copy().reset_index(drop=True)
del bb; gc.collect()
print(f'  {len(bb_c):,} backbones')

# Extract needed data and free shared dict to minimize peak memory
all_mea_nes = shared['contrast_data'][contrast]['all_mea_nes']
sub_raw_edges = shared['contrast_data'][contrast]['sub_raw_edges']
attr_by_ct = shared['attr_by_contrast_ct'][contrast]
del shared; gc.collect()

result = _run_permutation_one_contrast(
    bb_c,
    all_mea_nes=all_mea_nes,
    sub_raw_edges=sub_raw_edges,
    attr_by_celltype=attr_by_ct,
    n_permutations=n_perm,
    contrast=contrast,
)
del bb_c, all_mea_nes, attr_by_ct; gc.collect()

result.to_csv('${tmp_file}', index=False)
n_active = (result['n_edges'] > 0).sum()
n1 = result['significant_null1'].sum()
n2 = result['significant_null2'].sum()
print(f'  Written: {len(result):,} rows to ${tmp_file}')
print(f'  Null1={n1:,} ({100*n1/max(n_active,1):.1f}%), Null2={n2:,} ({100*n2/max(n_active,1):.1f}%)')
PYEOF
    echo "  ${contrast} complete"
done

# Concatenate all per-contrast files into the final output
echo ""
echo "Concatenating results..."
first=true
for contrast in "${CONTRASTS[@]}"; do
    tmp_file="${TMP_DIR}/perm_${contrast}.csv"
    if [ ! -f "${tmp_file}" ]; then
        echo "ERROR: Missing ${tmp_file}"
        exit 1
    fi
    if $first; then
        cat "${tmp_file}" > "${FINAL_FILE}"
        first=false
    else
        tail -n +2 "${tmp_file}" >> "${FINAL_FILE}"
    fi
done

echo "=== All contrasts complete ==="
wc -l "${FINAL_FILE}"

# Derive the filtered significant_both subset (Unit 6.2 pending — to be merged into the recurrence CSV).
SIG_FILE="${OUT_DIR}/backbone_significant_both_nulls.csv"
echo "Writing ${SIG_FILE}..."
pixi run python3 - <<PYEOF
import pandas as pd
df = pd.read_csv("${FINAL_FILE}")
sig = df[df["significant_both"]].copy()
sig.to_csv("${SIG_FILE}", index=False)
print(f"  {len(sig):,} significant backbones of {len(df):,}")
PYEOF

echo "Cleaning up temp files..."
rm -rf "${TMP_DIR}"

# -----------------------------------------------------------------
# Edge index build (Unit 1.3/1.4): concatenate per-pair kinase routes
# into kinase_backbone_edges.parquet for the unified viewer.
# Skips pairs whose kinase_routes.parquet already exists.
# -----------------------------------------------------------------
echo ""
echo "=== Emitting per-pair kinase routes (idempotent) ==="
pixi run python3 code/integration/adapters/compute_kinase_support_factorial.py \
  --emit-kinase-routes

echo ""
echo "=== Building edge index ==="
pixi run python3 code/integration/adapters/build_edge_index.py

echo "Done."
