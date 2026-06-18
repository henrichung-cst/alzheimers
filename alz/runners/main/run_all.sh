#!/usr/bin/env bash
# Phase 8 of viewer_input_audit_remediation: single end-to-end runner.
# Builds a working unified viewer from raw data — mouse + per-cluster
# decomposition + Incytr pair-mode + human cohort + viewer rebuild.
#
# Every stage is hardfailing (run_step). Sentinels under
# outputs/reports/.run_all_state/ skip already-completed stages on resume.
#
# Usage:
#   bash alz/runners/main/run_all.sh                # full chain
#   bash alz/runners/main/run_all.sh --force        # clear all sentinels
#   bash alz/runners/main/run_all.sh --rerun K1,K2  # clear specific steps
#   bash alz/runners/main/run_all.sh --skip-atlas   # skip atlas downloads (~95 GB)
#   bash alz/runners/main/run_all.sh --skip-incytr  # skip Incytr R chain
#
# Step keys: K-map, A-wmb, A-snrna, K-norm, K-enrich, K-mech, K-attr, K-recover,
#            D-prop, D-decomp, D-enrich-st, D-enrich-py, D-perout, D-verify,
#            I-inputs, I-pair, I-cache,
#            H-ingest, H-perdonor, H-seaad,
#            V-viewer

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SPINE=levy_t5
WORKERS=3
STATE_DIR="outputs/reports/.run_all_state"
LOG_DIR="outputs/reports/.run_all_logs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$STATE_DIR" "$LOG_DIR"
MAIN_LOG="$LOG_DIR/build.log"

SKIP_ATLAS=0
SKIP_INCYTR=0
FORCE=0
RERUN_LIST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-atlas)  SKIP_ATLAS=1; shift ;;
    --skip-incytr) SKIP_INCYTR=1; shift ;;
    --force)       FORCE=1; shift ;;
    --rerun)       RERUN_LIST="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$RERUN_LIST" ]]; then
  IFS=',' read -ra _keys <<< "$RERUN_LIST"
  for k in "${_keys[@]}"; do rm -f "$STATE_DIR/${k}.done"; done
fi
if [[ $FORCE -eq 1 ]]; then rm -f "$STATE_DIR"/*.done; fi

exec > >(tee -a "$MAIN_LOG") 2>&1
T_START=$(date +%s)
STEP_IDX=0
PLANNED=(K-map A-wmb A-snrna K-norm K-enrich K-attr K-mech K-recover
         D-prop D-decomp D-enrich-st D-enrich-py D-perout D-verify
         H-ingest H-perdonor H-seaad V-viewer)
if [[ $SKIP_INCYTR -eq 0 ]]; then
  PLANNED+=(I-inputs I-pair I-cache)
fi
N=${#PLANNED[@]}

fmt() { printf '%dh%02dm%02ds' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }

run_step() {
  local key="$1"; shift
  local label="$1"; shift
  local sentinel="$STATE_DIR/${key}.done"
  STEP_IDX=$((STEP_IDX+1))
  local t0=$(date +%s)
  echo
  echo "================================================================"
  echo "[$STEP_IDX/$N] ($key) $label   elapsed=$(fmt $((t0-T_START)))"
  if [[ -f "$sentinel" ]]; then
    echo "  SKIP cached: $(cat "$sentinel")"
    echo "================================================================"
    return 0
  fi
  echo "================================================================"
  "$@"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$sentinel"
  echo "  DONE in $(fmt $(($(date +%s)-t0)))"
}

# --- raw-reference prerequisites (migrated from the retired run_live_pipeline.sh) ---
# run_all assumes the WMB h5ads + SEA-AD effect sizes are on disk; the A-wmb/A-snrna
# steps below consume them. Auto-resolve the downloads unless --skip-atlas.
if [[ $SKIP_ATLAS -eq 0 ]]; then
  if [[ ! -d data/external/sea_ad ]] || [[ -z "$(ls -A data/external/sea_ad 2>/dev/null)" ]]; then
    echo "  prereq: SEA-AD effect sizes missing — running atlas reference acquisition"
    bash alz/runners/supporting/run_atlas_reference.sh
  fi
  n_h5ad=$(find data/external/allen_abc/expression_matrices/WMB-10Xv3/ \
      -name "*-log2.h5ad" 2>/dev/null | wc -l)
  if [[ "$n_h5ad" -lt 13 ]]; then
    echo "  prereq: only $n_h5ad/13 WMB region h5ads — running WMB download"
    bash alz/runners/supporting/run_wmb_download.sh
  fi
fi

# --- mouse kinase chain ---
run_step K-map     "kinase→gene mapping cache"        pixi run python alz/shared/map_kinases_to_genes.py
run_step A-wmb     "WMB expression export"            pixi run python alz/reference/wmb_expression.py --run
run_step A-snrna   "Song snRNA integration"           pixi run python alz/reference/snrna_integration.py --run
run_step K-norm    "kinase_normalize (IRS + stoich)"  pixi run python alz/bulk_mea/normalize.py
run_step K-enrich  "kinase_enrich (OLS + MEA)"        pixi run python alz/bulk_mea/enrich.py
# mechanism MUST run after attribute: mechanism.py merges its annotations into
# unified_attribution.csv (which attribute.py writes). Reversed order silently
# drops the merge — attribute then overwrites with no mechanism columns.
run_step K-attr    "kinase_attribute"                 pixi run python alz/bulk_mea/attribute.py
run_step K-mech    "kinase_mechanism (raw MEA)"       pixi run python alz/bulk_mea/mechanism.py
run_step K-recover "attribution_recovery"             pixi run python alz/bulk_mea/recover.py

# --- per-cluster decomposition ---
run_step D-prop      "snrna_proportions"               pixi run python alz/reference/snrna_proportions.py --run --spine "$SPINE"
run_step D-decomp    "build_celltype_decomposition"    pixi run python alz/decomposition_mea/build_celltype_decomposition.py --spine "$SPINE" --track both
run_step D-enrich-st "enrich_celltype (st)"            pixi run python -m alz.decomposition_mea.enrich_celltype --spine "$SPINE" --track st
run_step D-enrich-py "enrich_celltype (py)"            pixi run python -m alz.decomposition_mea.enrich_celltype --spine "$SPINE" --track py
run_step D-perout    "build_per_animal_site_ols"       pixi run python alz/decomposition_mea/build_per_animal_site_ols.py --spine "$SPINE"
run_step D-verify    "verify_decomposition (hardfail)" pixi run python alz/decomposition_mea/verify_decomposition.py --spine "$SPINE"

# --- Incytr pair-mode (optional) ---
if [[ $SKIP_INCYTR -eq 0 ]]; then
  run_step I-inputs "incytr build_pair_inputs.sh"      bash alz/incytr_pair/build_pair_inputs.sh
  run_step I-pair   "incytr pair-mode 9 contrasts"     env CHUNK_PARALLEL="$WORKERS" bash alz/incytr_pair/run_pair_mode.sh
  run_step I-cache  "pair_to_receiver_cache"           pixi run python alz/incytr_pair/pair_to_receiver_cache.py
fi

# --- human cohort ---
run_step H-ingest    "ingest_mukesh --reshape"        pixi run python alz/cohorts/mukesh/ingest.py --reshape
run_step H-perdonor  "ingest_mukesh_perdonor"         pixi run python alz/cohorts/mukesh/mea.py --track both
run_step H-seaad     "seaad_human_agreement"          pixi run python alz/cross_reference/seaad_human_agreement.py

# --- viewer rebuild (asserts all upstream provenance) ---
run_step V-viewer    "build_unified_viewer"           pixi run python alz/build_unified_viewer.py

T_END=$(date +%s)
echo
echo "================================================================"
echo "ALL $N STEPS DONE   total=$(fmt $((T_END-T_START)))"
echo "log:    $MAIN_LOG"
echo "state:  $STATE_DIR/  (--rerun K1,K2 to redo; --force to clear all)"
echo "================================================================"
