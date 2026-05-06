# backbone_perms — sidecar layer

Backbone-level dual-null permutation tests (within-receiver shuffle "enrichment
null" + "wiring null", Storey q-values) for the factorial pipeline.
Off by default. Activated by `INCYTR_LAYER_BACKBONE_PERMS=1` (see
`code/integration/incytr_runtime.sh`).

## What this pack adds

`run_factorial_permutations.sh` orchestrates one Python subprocess per contrast
(9 contrasts) calling `aggregate_factorial.py::_run_permutation_one_contrast`
and concatenates per-contrast results into

```
code/integration/intermediates/factorial/all_pairs/aggregation/backbone_permutation_pvalues_by_contrast.csv
```

This is a separate analysis stage. It never modifies `recv_*.parquet`; its
q-values never gate native PDS-based selection.

## Activation

```bash
INCYTR_LAYER_BACKBONE_PERMS=1 \
  bash code/integration/sidecar/backbone_perms/run_factorial_permutations.sh

# Or via aggregate_factorial.py --permutations (also gated):
INCYTR_LAYER_BACKBONE_PERMS=1 \
  pixi run python3 code/integration/adapters/aggregate_factorial.py --permutations
```

The helper functions (`_run_permutation_one_contrast`,
`run_factorial_permutations`) remain in `code/integration/adapters/aggregate_factorial.py`;
deletion would require a larger refactor. The flag-gate is enforced at the
call sites: the runner shell script aborts when the flag is unset, and the
`--permutations` branch in `aggregate_factorial.py` checks the flag before
invoking the helper. The helper functions themselves are dead code unless
reached through one of those call sites.

## Why it's parked

The null-distribution machinery itself is sound. The conceptual carving
between "backbone" and "pathway" needs a design revisit before this is
reactivated as part of the production deliverable. See
`docs/integrations/incytr_layer_inventory.md` (rows ALZ-9 / ALZ-11 / ALZ-12)
and the Sprint 5 working note at `docs/integrations/working/sprint5_added_layers_audit.md`.
