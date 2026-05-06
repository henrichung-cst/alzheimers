#!/usr/bin/env bash
# Sprint 1 D3: Wrapper-side degenerate two-condition equivalence runner.
#
# Demonstrates that the factorial Incytr extension preserves native semantics
# in a degenerate two-condition collapse for the slots that should match
# (sigprob, sc_FC, p_value, pr_FC, ps_FC, py_FC), and that the only
# discrepancies are the known Sprint-3/5 drifts (PDS, SiK_score) attributed
# in the ledger.
#
# Two checks run in sequence:
#   1. Re-run the Sprint 0 package-level golden comparison (native 93b9881
#      vs current HEAD on the synthetic 2-condition input). This is the
#      "legacy path preserved" claim.
#   2. Run the testthat factorial integration test (run_factorial on a
#      2-condition synthetic with 3 animals/condition). This is the
#      "factorial path produces sane 2-condition output" claim.
#
# Hard match required for slots: sigprob, sc_FC, p_value, pr_FC, ps_FC, py_FC
# Known drifts (deferred): evaluation.PDS (Sprint 3, INC-25), kl_pathways.SiK_*
#                         (Sprint 5, INC-28 + co-suspect INC-13)
#
# Wrapper-side caveat: code/integration/wrappers/run_incytr_factorial_all_pairs.R
# is hard-wired to the alzheimers proteomics input shape (9 contrasts, 22
# subclass pairs) and does not accept a 2-condition synthetic. The
# wrapper-side equivalence claim is therefore tested via the package-level
# entrypoint (run_factorial) here; refactoring the wrapper to accept a
# synthetic harness is recorded as a Sprint 1 finding (see the audit
# pre-diff addendum), not forced.
#
# Usage: bash code/integration/tests/run_degenerate_2cond.sh

set -euo pipefail

INCYTR_DIR="${INCYTR_DIR:-../incytr}"
ALZHEIMERS_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

if [[ ! -d "$INCYTR_DIR" ]]; then
  INCYTR_DIR="$ALZHEIMERS_DIR/../incytr"
fi

if [[ ! -d "$INCYTR_DIR" ]]; then
  echo "ERROR: incytr package directory not found at $INCYTR_DIR" >&2
  exit 2
fi

echo "=== Sprint 1 D3: degenerate 2-condition equivalence run ==="
echo "incytr dir:     $INCYTR_DIR"
echo "alzheimers dir: $ALZHEIMERS_DIR"
echo

cd "$INCYTR_DIR"

NATIVE="tests/testthat/fixtures/golden_native_93b9881.rds"
CURRENT="tests/testthat/fixtures/golden_current_head.rds"

if [[ ! -f "$NATIVE" || ! -f "$CURRENT" ]]; then
  echo "ERROR: Sprint 0 golden fixtures missing. Re-run generate_golden_output.R." >&2
  exit 3
fi

echo "--- Check 1: legacy 2-condition path (native 93b9881 vs current HEAD) ---"
Rscript tests/testthat/compare_golden_outputs.R "$NATIVE" "$CURRENT" \
  > /tmp/sprint1_d3_legacy_diff.md
cat /tmp/sprint1_d3_legacy_diff.md
echo

# Hard-fail if any of the must-match slots drifted. We grep for "match" status
# in the comparison output.
must_match=(sigprob sc_FC p_value pr_FC ps_FC py_FC)
fail=0
for slot in "${must_match[@]}"; do
  # The compare script emits "## slot: <name>" then "- match" or differences.
  awk -v s="$slot" '
    $0 ~ "^## slot: "s {found=1; next}
    found && /^## slot:/ {found=0}
    found {print}
  ' /tmp/sprint1_d3_legacy_diff.md | grep -q '^- match' || {
    echo "FAIL: slot '$slot' drifted on legacy 2-condition path" >&2
    fail=1
  }
done

if [[ $fail -eq 1 ]]; then
  echo "Legacy-path equivalence broken; this contradicts the Sprint 1 verdict." >&2
  exit 4
fi
echo "OK: must-match slots (${must_match[*]}) all clean."
echo

echo "--- Known Sprint-3/5 drifts (expected, not failures) ---"
echo "  evaluation.PDS  → Sprint 3 (INC-25 abde752 EM promiscuity weight)"
echo "  kl_pathways.SiK_* → Sprint 5 (INC-28 6858063 memory permutation pass; co-suspect INC-13 ca6a96e)"
echo

echo "--- Check 2: factorial path 2-condition synthetic (testthat factorial_integration) ---"
Rscript -e '
  options(warn = 1)
  suppressMessages(pkgload::load_all(".", quiet = TRUE))
  source("tests/testthat/helper-factorial.R")
  res <- testthat::test_file("tests/testthat/test-factorial_integration.R", reporter = "summary")
  fails <- sum(vapply(res, function(r) length(r$results), integer(1)) -
               vapply(res, function(r) sum(vapply(r$results, function(x) inherits(x, "expectation_success"), logical(1))), integer(1)))
  if (fails > 0) {
    cat("FAIL: factorial_integration tests failed\n")
    quit(status = 5)
  }
  cat("OK: factorial 2-condition integration tests passed\n")
'

echo
echo "=== D3 PASS: factorial extension preserves legacy semantics for must-match slots; ==="
echo "===          factorial path produces sane 2-condition outputs.                    ==="
