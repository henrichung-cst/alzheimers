# Incytr Integration

Connects bulk kinase activity (from `code/kinase_attribution.py`) to cell-cell signaling pathways inferred by Incytr across 462 sender-receiver cell-type pairs. The full specification — scoring model, configuration, runtime modes, outputs, limitations — lives at [`docs/integrations/kinase_incytr_integration.md`](../../docs/integrations/kinase_incytr_integration.md).

## Layout

- `adapters/` — Python data adapters (`alzheimers` env): snRNA-seq / kldata / MEA / phospho exports, kinase-imputed gene selection, substrate-based kinase support scoring, cross-pair aggregation, factorial examination.
- `wrappers/` — R wrappers (`incytr` env): DuckDB pathway enumeration, vectorized receiver-centric scoring, single-pair and all-pairs orchestrators, factorial OLS variant, postprocessing, bootstrap sensitivity.
- `tests/` — integration tests.
- `config_integration.py` — integration-specific configuration (paths, thresholds, contrast definitions).

## Primary runners

```bash
bash run_factorial_all_pairs.sh    # 462 pairs × 9 contrasts (primary)
bash run_factorial_permutations.sh # per-contrast backbone dual-null tests
bash run_all_pairs.sh              # 462 pairs × single contrast (App_4mo)
bash run_phase1.sh                 # single-pair debug reference
```

Run from the repo root or from `code/integration/`. Prerequisites: the bulk pipeline has produced `outputs/reports/kinase_attribution/mea_stoichiometry.csv` and `unified_attribution.csv`, and the `incytr` R environment is installed alongside the `alzheimers` Python environment.

See [`docs/integrations/kinase_incytr_integration.md`](../../docs/integrations/kinase_incytr_integration.md) for the full spec, including configuration reference, output schema, environment variables, and known limitations.
