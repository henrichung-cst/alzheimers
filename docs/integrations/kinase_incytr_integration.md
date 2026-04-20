# Kinase ↔ Incytr Integration

Source of truth for the integration layer that connects the live kinase-attribution pipeline with Incytr's cell-cell signaling inference. The integration produces, per receiver cell type, a Parquet table of scored Ligand → Receptor → Effector/Mediator → Target (L → R → EM → T) pathways with a kinase-support column that reranks pathways by substrate-level evidence from the phospho layer.

## 1. Purpose and scope

Bridge bulk kinase activity (MEA on stoichiometry β values across 9 disease×timepoint contrasts) with Incytr's snRNA-seq–based intercellular pathway inference across 462 sender-receiver pairs (22 SEA-AD subclasses × 21, excluding self-pairs).

- **Cell types:** 22 SEA-AD-mapped subclasses with snRNA-seq coverage.
- **Pairs:** 462 sender-receiver combinations.
- **Contrasts:** 9 (factorial primary mode) — `App_2mo, App_4mo, App_6mo, Tau_2mo, Tau_4mo, Tau_6mo, ApTt_2mo, ApTt_4mo, ApTt_6mo`. Single-contrast mode covers `App_4mo` only.
- **Sample filter:** males-only, matching the primary kinase-attribution analysis.
- **Nature:** hypothesis-generating. Cell-type attribution of kinase activity is correlational (SEA-AD concordance, WMB specificity, within-cohort snRNA-seq concordance); results are convergent functional evidence, not mechanistic pathway validation.

## 2. End-to-end data flow

```
outputs/reports/kinase_attribution/
   unified_attribution.csv      (kinase × cell-type weights)
   mea_stoichiometry.csv        (NES / FDR per kinase per contrast)
   site_level_ols.csv           (per-site LFC / FDR)
   stoichiometry_matrix.csv     (sites × samples)
outputs/reports/attribution_recovery/
   (diagnostics consumed by export_phospho.py)
data/incytr_collections/song/primary/transcriptomics/
   170_gex_celltypes_00.h5ad    (28 animals, 63K nuclei)
data/.../kldata_pspy.csv        (static kinase-substrate reference)

            │  Python adapters (alzheimers env)
            ▼
code/integration/intermediates/[factorial/]
   expression_matrix.mtx  expression_{genes,barcodes,metadata}.csv
   kl_output.csv          kldata.csv
   phospho_{WT,App}.csv   kinase_imputed_genes__{receiver}[__{contrast}].csv
   [factorial/ has expression_matrix.mtx + animal_metadata.csv;
    kl_output_all_contrasts.csv for all 9 contrasts]

            │  R wrappers (incytr env)
            ▼
code/integration/intermediates/[factorial/]all_pairs/
   recv_{receiver}.parquet      (22 files; columns depend on mode)

            │  Python adapters (alzheimers env)
            ▼
code/integration/intermediates/[factorial/]all_pairs/
   {sender}__{receiver}/kinase_support_scores.csv
   {sender}__{receiver}/adjusted_rankings.csv
   {sender}__{receiver}/reranking_summary.json

            │  aggregate_{cross_pair,factorial}.py
            ▼
code/integration/intermediates/[factorial/]all_pairs/aggregation/
   backbone_recurrence[_by_contrast].csv
   hub_matrix[_by_contrast].csv
   target_convergence[_by_contrast].csv
   backbone_permutation_pvalues[_by_contrast].csv
   backbone_significant_both_nulls.csv   (factorial only)
   contrast_comparison.csv / temporal_dynamics.{csv,png}  (factorial only)

            │  examine_factorial.py  (factorial only)
            ▼
.../aggregation/examination/
   additivity, temporal, cell-type, kinase composites and CSVs
```

## 3. Runtime modes

The factorial mode is the primary production path. Single-contrast mode (`App_4mo` only) is a lighter runner kept for single-contrast analyses and debugging. `run_phase1.sh` is a single-pair debug runner whose output schema is not compatible with the all-pairs path.

| Feature | `run_phase1.sh` | `run_all_pairs.sh` | `run_factorial_all_pairs.sh` |
|---|:---:|:---:|:---:|
| Pairs | 1 | 462 | 462 |
| Contrasts | 1 (`App_4mo`) | 1 (`App_4mo`) | 9 |
| Output format | `incytr_object.rds` + CSVs | 22 receiver Parquet files | 22 receiver Parquet files |
| Kinase support scoring | yes | yes | yes (per contrast) |
| Cross-pair aggregation | n/a | yes (`aggregate_cross_pair.py`) | yes (`aggregate_factorial.py`) |
| Backbone permutation tests | n/a | yes (`--permutations`) | yes (per-contrast, `run_factorial_permutations.sh`) |
| Kinase-imputed pathway expansion | yes | yes | yes |
| Biological interpretation layer | no | no | yes (`examine_factorial.py`) |

## 4. Component inventory

### Python adapters (`code/integration/adapters/`)

| File | Role |
|---|---|
| `common.py` | Naming bridge (abbrev → human → mouse gene), `build_substrate_kinase_map`, `sanitize_celltype_name`, `ensure_intermediates_dir` |
| `export_expression.py` | Sparse MTX from h5ad filtered to males / 4mo / WT+App for single-contrast mode |
| `export_expression_factorial.py` | All genotypes × timepoints; writes `animal_metadata.csv` for OLS design |
| `export_kldata.py` | Static kinase-substrate reference, filtered to MEA kinases |
| `export_kl_output.py` | MEA NES/FDR per kinase-substrate pair; `--all-pairs` widens to all attributed cell types |
| `export_kl_output_factorial.py` | MEA results for all 9 contrasts |
| `export_phospho.py` | Attribution-proportional bulk stoichiometry distributed to cell types |
| `export_kinase_imputed_genes.py` | Per-receiver genes below expression threshold with kinase-substrate evidence (single contrast) |
| `export_kinase_imputed_genes_factorial.py` | Same, iterated across 9 contrasts; one file per (receiver, contrast) |
| `compute_kinase_support.py` | Substrate-based external reranking for one pair (single contrast) |
| `compute_kinase_support_all_pairs.py` | Per-pair rerank across 462 pairs with checkpoint/restart |
| `compute_kinase_support_factorial.py` | Per-pair rerank with per-contrast columns |
| `aggregate_cross_pair.py` | Single-contrast cross-pair aggregation + `--permutations` |
| `aggregate_factorial.py` | Factorial cross-pair aggregation (per-contrast) + `--permutations` |
| `examine_factorial.py` | Final interpretive stage: additivity, temporal trajectories, cell-type centrality, kinase concordance, publication figures |

### R wrappers (`code/integration/wrappers/`)

| File | Role |
|---|---|
| `duckdb_enumeration.R` | R-side edge pruning + DuckDB 3-way join + inline SigProb; replaces Incytr's `pathway_inference()` |
| `receiver_scoring.R` | Vectorized Phase C scoring; computes receiver-side expression/FC once, broadcasts across 21 senders |
| `run_incytr.R` | Single-pair debug: two-pass (expression-only + full); produces `incytr_object.rds` + CSVs |
| `run_incytr_all_pairs.R` | 462-pair single-contrast orchestrator; outputs `recv_{receiver}.parquet` |
| `run_incytr_factorial_all_pairs.R` | Factorial variant: per-animal SigProb + OLS contrast estimation; per-contrast Parquet columns |
| `postprocess.R` | Sensitivity: PhPDS_ps redundancy (rho), IDF sensitivity, lambda sweep, rank divergence |
| `bootstrap_sensitivity.R` | Gated by `RUN_BOOTSTRAP=1`: L5 IT bootstrap (500 iter) and 20% detection-threshold comparison |
| `verify_phase2.R` | Manual regression: confirms vectorized scoring matches pair-centric Incytr S4 output. Run after edits to `receiver_scoring.R` |

### Shell runners (`code/integration/`)

| File | Role |
|---|---|
| `run_phase1.sh` | Single-pair reference / debug; Microglia-PVM → L5 IT |
| `run_all_pairs.sh` | 462-pair single-contrast pipeline; systemd-scoped R at 12 GB; `--skip-adapters` flag |
| `run_factorial_all_pairs.sh` | 462-pair, 9-contrast pipeline (primary mode) |
| `run_factorial_permutations.sh` | Per-contrast backbone dual-null permutation tests |

## 5. Scoring model

### Pathway shell (inside Incytr)

Each pathway is a four-gene chain:

```
Ligand (sender) → Receptor (receiver) → EM (receiver) → Target (receiver)
```

Valid chains are drawn from the curated IncytrDB catalog. The **signaling probability** combines per-link Hill functions of joint node expression:

```
SigProb = Hill(L × R) × Hill(R × EM) × Hill(EM × T)
```

**TPDS** (Transcriptomic Pathway Differential Score) is a logistic transform of SigProb to `[-1, +1]` measuring condition-level change in co-expression. Additional omic layers contribute additive terms to the final **PDS** (Pathway Differential Score); the `kinase_boost = PDS − TPDS` column records how much non-expression evidence moved the score.

### Gene filtering

Expression detection threshold: a gene must be detected in ≥ `EXPRESSION_DETECTION_THRESHOLD` (10%) of cells in the relevant cell type. Receiver gene count dominates the combinatorial scale because the receiver supplies three of four pathway nodes (Receptor, EM, Target). DuckDB SigProb pre-filtering at 0.01 keeps the enumerated set tractable.

### Kinase-imputed pathway expansion

The adapter (`export_kinase_imputed_genes[_factorial].py`) identifies receiver genes that are below the expression threshold but are substrates of kinases with strong MEA enrichment, and unions them with the expression-threshold set before enumeration. Gates:

1. MEA FDR < `KINASE_IMPUTATION_FDR` (default 0.10).
2. `unified_attribution.combined_score > KINASE_IMPUTATION_ATTRIBUTION_TAU` for the target receiver (default = median combined_score across attributed (receiver, kinase) pairs). This **per-receiver cell-type gate** prevents a kinase strongly attributed to microglia from imputing its substrates into unrelated receivers.
3. R-side **expression floor** `EXPR_IMPUTATION_FLOOR` (default 0.05) on `det_rates` excludes genes with no RNA in the target receiver.
4. **Soft rescue** for substrates whose weighted-quantile expression is zero: `rescued = imputed_weight × rowMeans`, where `imputed_weight = 1 − best_fdr`. A substrate supported only by an FDR = 0.09 kinase rescues ~10× weaker than one supported by an FDR = 0.001 kinase.

Adapter output: `intermediates/[factorial/]kinase_imputed_genes__{receiver}[__{contrast}].csv` with columns `gene, n_sig_kinases, source_kinases, best_fdr, imputed_weight, receiver`, plus a `kinase_imputation_summary.csv`. Legacy flat behavior is available via `icfg.KINASE_IMPUTATION_LEGACY = True`.

In factorial mode, gene sets are unioned across the 9 contrasts per receiver (a gene survives if gated in any contrast) and `imputed_weight` is aggregated by `max`. Every enumerated pathway is labeled in the `pathway_evidence` column as `expression-confirmed` (all four nodes pass the detection threshold) or `kinase-imputed` (one or more receiver-side nodes admitted via kinase-substrate evidence). The `imputed_nodes` column records which positions were imputed.

### DuckDB pathway enumeration (`duckdb_enumeration.R`)

Replaces Incytr's default data.table cartesian joins, which OOM at 10% threshold.

1. **Edge pre-pruning.** Compute per-condition Hill values per edge layer (L1: L→R, L2: R→EM, L3: EM→T) from weighted-quantile expression. Drop edges where `max(Hill_WT, Hill_App) < 0.01`. Typical reductions: L1 330→30, L2 685→202, L3 20,464→12,144.
2. **DuckDB 3-way join.** Registered pruned edges, single SQL join with `SigProb_WT ≥ 0.01 OR SigProb_App ≥ 0.01`. DuckDB handles combinatorics in-process with disk spilling (temp dir configured via `DUCKDB_TEMP_DIR=~/.cache/duckdb`).
3. **EM promiscuity weighting.** Each pathway's SigProb includes an EM degree weight `1/log2(1+degree)` via LEFT JOIN, penalizing promiscuous effectors.

### Receiver-centric all-pairs architecture (`run_incytr_all_pairs.R` / `_factorial_all_pairs.R`)

Three of four pathway nodes are receiver-determined. The orchestrator exploits this to avoid 462 independent Incytr invocations:

- **Phase A+B (enumeration).** Outer loop over 22 receivers prunes L2/L3 edges once per receiver, then inner loop over 21 senders prunes L1 edges and runs the DuckDB 3-way join. Produces a unified `all_pathways_df` data.table with `sender` as a column.
- **Phase C (vectorized scoring).** All senders for a given receiver are scored in a single pass (`receiver_scoring.R`). Receiver-side computations (expression lookups, fold changes, phospho normalization, SiK/EI, kinase activity) are computed once and shared across all 21 senders. Only sender-side quantities (Ligand expression, Ligand FC, per-sender SigProb threshold) vary per sender.

Output: 22 receiver-indexed Parquet files (`recv_{receiver}.parquet`) with `sender` as a column. Predicate pushdown keeps per-pair queries efficient.

Operational features:
- **Checkpoint:** Receivers with existing `recv_{receiver}.parquet` are skipped. Set `FORCE_RERUN=1` to override.
- **Memory guard:** R memory is checked after each receiver; aborts cleanly if > `MEMORY_LIMIT_GB` (default 10).
- **Pair filter:** `PAIR_FILTER="Microglia-PVM:L5 IT"` for single-pair testing; `PAIR_FILTER="*:L5 IT"` for single receiver.
- **Memory scope:** R is launched under `systemd-run --user --scope -p MemoryMax=12G`.

### Kinase support score (external substrate channel)

Computed by `compute_kinase_support[_all_pairs|_factorial].py` after R produces Parquet. For each pathway with nodes (L, R, EM, T):

1. **Connected kinases.** For each EM and Target gene, look up kinases that phosphorylate it (from kldata via `build_substrate_kinase_map()`). Retain only kinases with MEA FDR < `PHOSPHO_FDR_GATE` (0.25) for the relevant contrast. **Exclude kinases that are already pathway nodes** (deduplication — Incytr's internal channel already scores them).
2. **Edge weight.** For kinase K connected to substrate G:
   ```
   edge_weight        = |NES_K| × IDF_G × attribution_weight_K
   IDF_G              = 1 / log(N), N = #MEA-significant kinases targeting G
                         (pair-independent; returns 1.0 when N ≤ 1)
   attribution_weight = max over relevant cell types of
                         combined_score × cell_type_relevance
   cell_type_relevance = 1.0  if K attributed to receiver
                         0.25 if K attributed only to sender (SENDER_ATTRIBUTION_DISCOUNT)
   ```
3. **Aggregate per pathway.** `kinase_support_score = median(edge_weights)`, unsigned. The median is robust to hub-substrate inflation (promiscuous substrates like Srrm2 with 80+ kinases) and single-outlier dominance. Sum is retained as `kinase_support_score_sum`. Each pathway is annotated with a concordance flag (`concordant` / `discordant` / `mixed` / `none`, based on mean sign of NES vs sign of TPDS), number of distinct contributing kinases, number of node kinases excluded, and identity of top contributing kinases.
4. **Adjusted rankings.** `adjusted_score = TPDS + λ × kinase_support_score` for `λ ∈ LAMBDA_VALUES`.

**Deduplication rule.** Kinases that are already pathway nodes (EM or Target) for a given pathway are excluded from that pathway's external score. Their contribution is already carried by Incytr's internal channel.

### Factorial statistical model

Factorial mode uses per-animal SigProb and OLS contrast estimation rather than condition-level averaging.

Design matrix (10 parameters, matching `kinase_attribution.py`): `const, App, Tau, Int, time_4mo, time_6mo, App×time4, App×time6, Tau×time4, Tau×time6`. Sample counts (15 male animals with snRNA-seq):

| Genotype | 2mo | 4mo | 6mo | Total |
|---|---|---|---|---|
| WT   | 2 | 1 | 1 | 4 |
| App  | 2 | 1 | 1 | 4 |
| Tau  | 2 | 1 | 1 | 4 |
| ApTt | 1 | 1 | 1 | 3 |

`df_resid = 5`. Nine contrasts yield per-pathway `TPDS_{contrast}`, `SE_{contrast}`, `pvalue_{contrast}`.

**Statistical caveats.** The t-distribution has heavy tails with df_resid = 5; p-values are conservative and SE on interaction terms is large. ApTt contrasts draw from one animal per timepoint, giving them the widest variance. The 9 contrasts share animals and expression data — per-contrast p-values are valid marginally but cross-contrast inference requires care.

### Backbone-level permutation tests

Tests whether kinase-substrate evidence concentrates in specific signaling backbones (receiver × Receptor × EM × Target) beyond what chance wiring or chance enrichment would produce.

- **Null 1 (enrichment).** For each backbone with N edges, sample N kinases from the full MEA kinase universe (311 kinases tested for `App_4mo`). Sampled kinases use their actual |NES| but a uniform attribution weight (median of observed). Asks: do backbone scores reflect enrichment for disease-significant kinases?
- **Null 2 (wiring, within-receiver).** Shuffle kinase identity from the full MEA universe and sample attribution weights *within each backbone's receiver cell type*. Asks: does the specific kinase-substrate-to-cell-type wiring matter, conditioned on receiver identity?

Multiple-testing correction: Benjamini–Hochberg for single-contrast; **Storey's q-value** (fixed λ = 0.5), computed on active backbones only, for factorial. With 69K–135K active tests per contrast, π₀ estimation is stable.

Operational: `compute_kinase_support[_all_pairs|_factorial].py` produces per-pair scores; `aggregate_cross_pair.py --permutations` or `run_factorial_permutations.sh` runs the backbone null tests. Permutation batches process pathways grouped by degree; memory budget ~500 MB peak per contrast. Each contrast runs as a separate Python subprocess to cap peak RAM.

### Kinase naming bridge (`adapters/common.py`)

Three naming conventions are bridged by `load_mouse_gene_to_kinase_mapping()`:

| Context | Example | Convention |
|---|---|---|
| MEA / `unified_attribution.kinase` | MNK1, GSK3A | Kinase abbreviation |
| `unified_attribution.gene_symbol` | MKNK1, GSK3A | Human gene symbol |
| kldata `motif.geneName` / Incytr nodes | Mknk1, Gsk3a | Mouse gene symbol |

Kinase abbreviation → human gene symbol uses MyGene.info, cached at `SONG_ANALYSIS_CACHE_DIR/kinase_to_gene_mapping.csv`. Human → mouse is lexical title-case (`GSK3A` → `Gsk3a`, digit-first symbols pass through), relying on the observed 1:1 case-style equivalence for kinases in scope — this is a deliberate simplification, not an orthology join.

## 6. Configuration — `config_integration.py`

| Name | Value | Authoritative source | Notes |
|---|---|---|---|
| `CONTRAST` | `App_4mo` | Python | Single-contrast default |
| `CONDITION_WT` / `CONDITION_DISEASE` | `WTyp` / `AppP` | Python | h5ad `mutant` values |
| `TIMEPOINT` | `4mo` | Python | |
| `SEX_FILTER` | `ma` | Python | Males-only |
| `SENDER` / `RECEIVER` | `Microglia-PVM` / `L5 IT` | Python | Phase 1 reference pair |
| `FACTORIAL_GENOTYPES` | `[WTyp, AppP, Ttau, ApTt]` | Python | |
| `FACTORIAL_TIMEPOINTS` | `[2mo, 4mo, 6mo]` | Python | |
| `DESIGN_COLUMNS` | 10-param OLS | Python | Matches `kinase_attribution.py` |
| `FACTORIAL_CONTRASTS` | 9 contrast vectors | Python | Same as `CONTRAST_COEFS` |
| `PHOSPHO_FDR_GATE` | `0.25` | Python | MEA FDR gate for kinase support |
| `DISCORDANCE_RANK_QUARTILE` | `0.25` | Python | |
| `EXPRESSION_DETECTION_THRESHOLD` | `0.10` | **R env var `EXPR_DETECTION_THRESHOLD`** | Python constant is documentation; R wrappers read from environment with `0.10` default |
| `ENABLE_KINASE_IMPUTATION` | `True` | Shell env var of same name | |
| `KINASE_IMPUTATION_FDR` | `0.10` | Python | Imputation gate, tighter than `PHOSPHO_FDR_GATE` |
| `KINASE_IMPUTATION_ATTRIBUTION_TAU` | median `combined_score` | Python | Per-receiver cell-type gate |
| `EXPR_IMPUTATION_FLOOR` | `0.05` | R | Minimum `det_rates` for imputed survivors |
| `LAMBDA_VALUES` | `[0.1, 0.25, 0.5, 1.0, 2.0]` | Python | Reranking sweep |
| `N_PERMUTATIONS` | `10_000` | Python | Per-pair null |
| `N_PERMUTATIONS_AGGREGATE` | `10_000` | Python | Backbone-level test |
| `SENDER_ATTRIBUTION_DISCOUNT` | `0.25` | Python | |
| `PDS_SIGNIFICANCE_THRESHOLD` | `0.1` | Python | |
| `DETECTION_THRESHOLD_SENSITIVITY` | `0.20` | Python (via `bootstrap_sensitivity.R`) | Gated by `RUN_BOOTSTRAP=1` |
| `N_BOOTSTRAP_ITERATIONS` | `500` | Python | L5 IT bootstrap |

## 7. Running the integration

Prerequisites: the bulk pipeline (`code/runners/main/run_live_pipeline.sh`) has been run; `mea_stoichiometry.csv` and `unified_attribution.csv` exist.

### Factorial all-pairs (primary)

```bash
bash code/integration/run_factorial_all_pairs.sh
bash code/integration/run_factorial_all_pairs.sh --skip-adapters  # checkpoint resume
PAIR_FILTER="Microglia-PVM:L5 IT" bash code/integration/run_factorial_all_pairs.sh
```

Stages executed:

1. **Python adapters** (`alzheimers` env) — `export_expression_factorial.py`, `export_kldata.py`, `export_kl_output_factorial.py`, `export_kinase_imputed_genes_factorial.py`.
2. **R factorial pipeline** (`incytr` env) — `run_incytr_factorial_all_pairs.R` under `systemd-run --user --scope -p MemoryMax=12G`. Per-animal expression, per-animal SigProb, OLS contrast estimation.
3. **Kinase support** (`alzheimers` env) — `compute_kinase_support_factorial.py` across 462 pairs × 9 contrasts.
4. **Cross-pair aggregation** (`alzheimers` env) — `aggregate_factorial.py` (backbone recurrence, hub matrix, target convergence, per contrast).
5. **Permutation tests** — `run_factorial_permutations.sh` (one Python subprocess per contrast).
6. **Examination** — `examine_factorial.py --run`.

### Single-contrast all-pairs (App_4mo only)

```bash
bash code/integration/run_all_pairs.sh
bash code/integration/run_all_pairs.sh --skip-adapters
```

Same stages with `_all_pairs` adapter/wrapper variants; aggregation via `aggregate_cross_pair.py`; permutations via `--permutations` flag.

### Single-pair reference (debug)

```bash
bash code/integration/run_phase1.sh
```

Produces `incytr_object.rds` + CSVs. Output schema is not compatible with the all-pairs Parquet path.

### Environment variables

| Variable | Default | Applies to | Description |
|---|---|---|---|
| `PAIR_FILTER` | (all pairs) | all-pairs runners | Filter pairs, e.g. `"Microglia-PVM:L5 IT"` or `"*:L5 IT"` |
| `FORCE_RERUN` | `0` | all-pairs runners | Set to `1` to reprocess pairs with existing results |
| `ENABLE_KINASE_IMPUTATION` | `1` | all-pairs runners | Set to `0` to disable kinase-imputed pathway expansion |
| `RUN_PERMUTATIONS` | `0` | single-contrast | Set to `1` to run dual-null backbone tests during aggregation |
| `RUN_BOOTSTRAP` | `0` | all | Set to `1` to run L5 IT bootstrap sensitivity |
| `MEMORY_LIMIT_GB` | `10` | R wrappers | In-R memory guard threshold |
| `EXPR_DETECTION_THRESHOLD` | `0.10` | R wrappers | Expression detection threshold |
| `DUCKDB_TEMP_DIR` | `~/.cache/duckdb` | DuckDB | Spill directory (set by `.envrc`) |

## 8. Outputs

### Factorial all-pairs (primary)

Under `code/integration/intermediates/factorial/all_pairs/`:

- `recv_{receiver}.parquet` (22 files) — columns: `sender, Ligand, Receptor, EM, Target, Path, pathway_evidence, imputed_nodes, kinase_boost, n_animals, df_resid`, plus per-contrast `TPDS_{contrast}`, `SE_{contrast}`, `pvalue_{contrast}` for each of 9 contrasts.
- `pair_summary.csv` — 462-row summary (pathway counts, timing, status).
- `{sender}__{receiver}/kinase_support_scores.csv` — per-pair, per-contrast substrate-based evidence.
- `{sender}__{receiver}/adjusted_rankings.csv` — `TPDS + λ × kinase_support` for the λ sweep.
- `{sender}__{receiver}/reranking_summary.json` — per-pair scoring statistics.
- `aggregation/backbone_recurrence_by_contrast.csv` — R-EM-T triples shared across senders, by contrast.
- `aggregation/hub_matrix_by_contrast.csv` — 22 × 22 sender × receiver signaling summary.
- `aggregation/target_convergence_by_contrast.csv`.
- `aggregation/backbone_permutation_pvalues_by_contrast.csv` — dual-null per backbone per contrast.
- `aggregation/backbone_significant_both_nulls.csv` — backbones passing Null 1 and Null 2 (Storey q < 0.25).
- `aggregation/contrast_comparison.csv`, `temporal_dynamics.{csv,png}`, `kinase_tpds_integration.csv`, `kinase_coverage.png`, `hub_heatmap_grid.png`, `pathway_viewer.html`.
- `aggregation/examination/` — `examine_factorial.py` outputs: additivity (ApTt vs App+Tau), temporal trajectory classification, cell-type centrality, kinase concordance.

`examine_factorial.py` CLI flags (combinable): `--summary`, `--additivity`, `--temporal`, `--celltype`, `--kinase`, `--figures`, `--run`.

### Single-contrast all-pairs

Same schema without the `_{contrast}` column suffixes; aggregation at `aggregation/backbone_recurrence.csv`, `hub_matrix.csv`, `target_convergence.csv`, `backbone_permutation_pvalues.csv`, etc.

### Single-pair reference (`run_phase1.sh`)

Different schema: `incytr_object.rds`, `results_expronly.csv`, `results_full.csv`, `results_concordant.csv`, `results_discordant_{A,B}.csv`, `pvalues_seed{1,2,3}.csv`, `permutation_pvalues.csv`, `sensitivity_report.csv`, `ranking_correlation.json`. Not produced by the all-pairs pipeline.

## 9. Sensitivity analyses

Computed alongside main scoring (`postprocess.R`):

1. **PhPDS_ps redundancy.** Spearman correlation between Incytr's internal PhPDS_ps and the external kinase support score.
2. **IDF sensitivity.** Top-20 overlap between scores with and without IDF weighting.
3. **Lambda sensitivity.** Kendall τ-b between rankings at adjacent λ values.
4. **Rank divergence from TPDS.** Kendall τ between TPDS-only ranking and adjusted ranking at each λ.

Additional (R/Incytr, gated by `RUN_BOOTSTRAP=1`, `bootstrap_sensitivity.R`):

5. **L5 IT bootstrap.** Resample L5 IT nuclei with replacement (500 iterations), rerun TPDS with fixed pathway structure, report `cv_rank`, `frac_in_top20`, `frac_in_top50`.
6. **Detection-threshold comparison.** Run Incytr at 20% detection threshold; compare top-50 overlap with 10% results. Wrapped in `tryCatch` for OOM tolerance.

## 10. Known limitations

The integration is explicitly hypothesis-generating. Cell-type attribution of kinase activity is correlational and the upstream phospho/mRNA evidence streams share animals. The following limitations affect biological inference beyond attribution.

- **Shared animals between evidence streams.** The 28 snRNA-seq animals are a subset of the 72 TMT proteomics animals. Song concordance contributes to kinase attribution weights that feed pathway reranking. This is not full circularity (different modalities), but the two streams are not statistically independent. A SEA-AD + WMB-only sensitivity run can be produced by excluding Song concordance from attribution weights; this is not currently part of the default run.
- **SigProb cutoff (0.01) is permissive.** 80% of pathways have SigProb < 0.05; the cutoff passes essentially any pathway with non-zero expression at all four nodes. Downstream scoring operates on a large combinatorial set.
- **Small per-condition cell counts in some receiver types.** Dropout noise at small n propagates into TPDS at every pathway node. `bootstrap_sensitivity.R` quantifies rank stability per receiver; downstream summaries should condition on per-receiver nuclei counts.
- **Unsigned kinase support score.** The primary score is unsigned by design (avoids discarding inhibitory phosphorylation) but conflates concordant and discordant biology. Summaries and top-N lists must stratify by the concordance flag.
- **IncytrDB constrains discoverable biology.** Enumeration is limited to four-gene chains present in IncytrDB. Non-canonical, recently discovered, or non-four-gene architectures are invisible.
- **Motif-based substrate predictions (kldata)** carry non-trivial false-positive rates. Structural mitigations (IDF weighting, median aggregation, dual-null permutation) limit the influence of individual false edges.
- **Sender attribution discount (0.25) is a point estimate** chosen by reviewer judgment (0.2–0.3 range), not data-derived. Most kinase evidence connects through receiver-attributed kinases, limiting the parameter's influence.
- **`run_phase1.sh` output schema diverges** from the all-pairs Parquet path — keep for debugging and single-pair reference only.
- **`verify_phase2.R` is manual.** Not in any runner; invoke after edits to `receiver_scoring.R`.
- **ApTt contrasts have high variance.** Only one animal per timepoint contributes to the App×Tau interaction estimate; SE on those contrasts is the widest.

## 11. Related documentation

- Bulk kinase-attribution pipeline (upstream producer of MEA / unified attribution): [`docs/foundation/analysis_charter.md`](../foundation/analysis_charter.md), [`docs/foundation/live_pipeline_contract.md`](../foundation/live_pipeline_contract.md)
- Concordance model (feeds cell-type attribution weights): [`docs/foundation/concordance.md`](../foundation/concordance.md)
- Upstream collaborator data layout: [`./integrations-structure.md`](./integrations-structure.md)
