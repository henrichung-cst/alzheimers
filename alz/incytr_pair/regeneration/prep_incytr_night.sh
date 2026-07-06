#!/usr/bin/env bash
# Clear 5xFAD + t-cell Incytr production dirs so the overnight runners regenerate
# them KsG-fresh. Run this ONCE before launching launch_incytr_tmux.sh tonight.
#
#   bash alz/incytr_pair/regeneration/prep_incytr_night.sh
#
# Song is DONE — its KsG production output (wide/ + backbone/) is the canonical
# state in outputs/reports/incytr_pair_mode/ and is never swapped; viewer
# development reads it directly. Only 5xFAD and t-cell remain un-regenerated.
#
# This script empties 5xFAD/t-cell production so each runner's skip-on-existing
# guard falls through and regenerates KsG (+ PTM / backbone) full.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

FIVEX=outputs/reports/incytr_pair_mode_5xfad
TCELL=outputs/reports/incytr_pair_mode_tcells

echo "=== 5xFAD: clear production (runner regenerates KsG full) ==="
for tissue in cortex hippocampus; do
  rm -rf "$FIVEX/$tissue/wide"
done
echo "  cortex/hippocampus wide cleared"

echo "=== t-cell: clear production (runner regenerates KsG full) ==="
for donor in donor1 donor2; do
  rm -rf "$TCELL/$donor/wide"
done
echo "  donor1/donor2 wide cleared"

cat <<'EOF'

5xFAD + t-cell production cleared for KsG regeneration (Song untouched).
Launch the combined sequential runner:

  bash alz/incytr_pair/regeneration/launch_incytr_tmux.sh
  tmux attach -t incytr

  all -> Enter   (5xFAD pair-mode/post-processing first, then t-cell; bridge deferred)
EOF
