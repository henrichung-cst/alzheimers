# kinase_pack — sidecar layer

Optional kinase-aware extension layer for the factorial Incytr pipeline.
Off by default. Activated by `INCYTR_LAYER_KINASE_PACK=1` (see
`code/integration/incytr_runtime.sh` and `code/integration/wrappers/incytr_runtime.R`).

## What this pack adds

When the flag is set, two additions become live:

1. **Kinase-imputed gene expansion** — `code/integration/adapters/export_kinase_imputed_genes_factorial.py`
   runs and the wrapper rescue block in `code/integration/wrappers/run_incytr_factorial_all_pairs.R`
   admits substrate genes for MEA-significant kinases that fail the expression
   threshold, labeled `kinase-imputed` vs `expression-confirmed`. (The exporter
   itself stays in `adapters/` because it produces an intermediate consumed
   directly by the wrapper.)
2. **Substrate-based kinase support scoring** — `compute_kinase_support_factorial.py`
   (this directory) reads baseline `recv_*.parquet`, computes per-pair IDF /
   sender-attribution-discounted median support scores, and writes
   `kinase_support_scores.csv` (+ optional `kinase_routes.parquet`) per pair.
   Reads baseline outputs only — never overwrites native PDS.

Both pieces ship as one coherent pack because the second consumes intermediates
emitted by the first.

## Activation

```bash
INCYTR_LAYER_KINASE_PACK=1 bash code/integration/run_factorial_all_pairs.sh
# Then, for the kinase support sidecar:
INCYTR_LAYER_KINASE_PACK=1 pixi run python3 \
  code/integration/sidecar/kinase_pack/compute_kinase_support_factorial.py
```

## Why it's parked

The pack is observationally inert when the flag is off (no
`kinase_imputed_genes__*.csv` files exist; the wrapper rescue block
short-circuits). It is retained for future Section C re-activation; see
`docs/integrations/incytr_layer_inventory.md` (rows ALZ-19 / ALZ-20) for the
audit verdict and revival pointers.
