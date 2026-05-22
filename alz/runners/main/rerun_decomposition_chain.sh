#!/usr/bin/env bash
# Regenerate the per-cluster decomposition chain against the active Levy-t5
# spine. Use after rerun_mouse_kinase_chain.sh and before rebuilding the
# viewer. The hardfail gates in snrna_integration.py / snrna_proportions.py
# prevent silently dropping clusters from a stale pseudobulk; this script
# rebuilds the chain end-to-end so the gates can verify it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

SPINE=levy_t5
T0=$(date +%s)

echo "=== $(date -Is) [1/7] snrna_integration --pseudobulk (Levy-t5 spine) ==="
pixi run python alz/reference/snrna_integration.py --pseudobulk

echo "=== $(date -Is) [2/7] snrna_proportions --run (forward-projection weights) ==="
pixi run python alz/reference/snrna_proportions.py --run --spine "$SPINE"

echo "=== $(date -Is) [3/7] build_celltype_decomposition (st + py) ==="
pixi run python alz/decomposition_mea/build_celltype_decomposition.py --spine "$SPINE" --track both

echo "=== $(date -Is) [4/7] enrich_celltype (st) ==="
pixi run python -m alz.decomposition_mea.enrich_celltype --spine "$SPINE" --track st
echo "=== $(date -Is) [5/7] enrich_celltype (py) ==="
pixi run python -m alz.decomposition_mea.enrich_celltype --spine "$SPINE" --track py

echo "=== $(date -Is) [6/7] build_per_animal_site_ols ==="
pixi run python alz/decomposition_mea/build_per_animal_site_ols.py --spine "$SPINE"

echo "=== $(date -Is) [7/7] verify_decomposition (hardfail) ==="
pixi run python alz/decomposition_mea/verify_decomposition.py --spine "$SPINE"

T1=$(date +%s)
echo
echo "=== done in $((T1 - T0))s ==="

# Success condition: all 31 spine clusters present in mea_per_cluster, no skips.
SPINE="$SPINE" .pixi/envs/default/bin/python - <<'PY'
import json, os, sys
spine = os.environ["SPINE"]
audit = json.load(open(f"outputs/reports/decomposition/{spine}/enrich_audit.json"))
per = audit.get("per_cluster", {})
skipped = [c for c, info in per.items() if info.get("status") == "skipped"]
print(f"clusters: {len(per)}  ok: {sum(1 for v in per.values() if v.get('status')=='ok')}  skipped: {len(skipped)}")
if skipped:
    print("SKIPPED clusters:", skipped, file=sys.stderr)
    sys.exit(1)
verif = json.load(open(f"outputs/reports/decomposition/{spine}/verification.json"))
if verif.get("all_pass") is not True:
    print("verification.json all_pass != true", file=sys.stderr)
    sys.exit(1)
print(f"OK: {len(per)}/{len(per)} clusters processed, verification.json all_pass=true")
PY
