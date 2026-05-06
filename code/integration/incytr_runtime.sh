# Shell-side mirror of code/integration/wrappers/incytr_runtime.R.
#
# Single registry for INCYTR_* runtime knobs. Sourced by run_all_pairs.sh and
# run_factorial_all_pairs.sh so the shell-side adapter-skip decisions agree
# with the R-side reads in run_incytr_factorial_all_pairs.R.
#
# Defaults must mirror incytr_runtime.R; see
# docs/integrations/incytr_layer_inventory.md for the rationale.

# Layer flags (1 = on, 0 = off; defaults off — production pipeline runs as
# Section A + Section B + INC-30 PTM only).
: "${INCYTR_LAYER_KINASE_PACK:=0}"
: "${INCYTR_LAYER_BACKBONE_PERMS:=0}"

# DuckDB pre-prune SigProb cutoff (0.0 = native-equivalent; no pre-prune).
: "${INCYTR_CUTOFF_SIGPROB:=0.0}"

export INCYTR_LAYER_KINASE_PACK INCYTR_LAYER_BACKBONE_PERMS INCYTR_CUTOFF_SIGPROB

# Forwarded env-var list — used by both runners when constructing
# `systemd-run --setenv=` arguments.
INCYTR_FORWARDED_VARS=(
  INCYTR_LAYER_KINASE_PACK
  INCYTR_LAYER_BACKBONE_PERMS
  INCYTR_CUTOFF_SIGPROB
)
