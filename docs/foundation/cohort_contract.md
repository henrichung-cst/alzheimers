# Cohort Contract

Canonical input/output schemas for the four shared analysis modes. Anything
listed here is **stable cross-cohort**: a new collaborator dataset becomes
runnable by writing one ingest module that emits artifacts matching this
contract, plus one `conf/<cohort>/parameters.yml`. No code below
`alz/ingest_<cohort>.py` should know which cohort it is operating on.

This document codifies **today's de-facto schemas** (as of the 2026-05-21
factorial vocab unification + repo-organization Phase 3 closeout). It is
intentionally descriptive, not aspirational — when current code emits a
column, this contract names it required; when current code is permissive,
this contract is permissive.

## 1. Scope

The four analysis modes:

| # | Mode | Scope | Entry |
|---|---|---|---|
| 1 | Bulk MEA | mouse + human | `alz/kinase_normalize.py` → `alz/kinase_enrich.py`; human branch: `alz/ingest_mukesh_perdonor.py` |
| 2 | Decomposition MEA | mouse only (requires matched snRNA) | `alz/snrna_proportions.py` → `alz/decomposition_mea/*` |
| 3 | Incytr pair-mode | mouse only (requires snRNA spine) | `alz/incytr_pair/run_pair_mode.sh` |
| 4 | Cross-reference correlation | atlas-based (cohort-agnostic) | `alz/atlas_reference.py`, `alz/wmb_expression.py`, `alz/human_reference_expression.py` |

Mode 4 has no cohort-specific input — it produces reference aggregates that
modes 1–3 join into. Treat it as a one-time prerequisite.

## 2. Cohort identity

### 2.1 `sample_mapping.csv`

Single source of truth for sample → condition assignment. Two shapes are
allowed depending on whether the cohort has a factorial design or a binary
case/control design.

**Factorial cohort** (e.g. Song mouse, 4 genotypes × 3 timepoints × 2 sexes):

| Column | Type | Required | Notes |
|---|---|---|---|
| `column_name` | str | yes | matches a column in the bulk matrices |
| `mouse_id` | str | yes (mouse) | cohort-internal subject id |
| `animal_id` | str | yes (mouse) | upstream collaborator label; parsed by `config.parse_animal_id` |
| `sex` | str | yes | factor level vocabulary in `conf/<cohort>/parameters.yml:sex_levels` |
| `timepoint` | str | yes | factor level vocabulary in `conf/<cohort>/parameters.yml:timepoint_levels` |
| `genotype` | str | yes | factor level vocabulary in `conf/<cohort>/parameters.yml:genotype_levels` (long form, no slashes) |
| `replicate` | int | optional | within-group replicate counter |
| `has_snrna_seq` | bool | optional | true if matched snRNA is available |
| `snrna_sample_id` | str | optional | join key into snRNA pseudobulk |
| `phospho_group_id` | str | optional | group key for stoichiometry shift correction |

**Case/control cohort** (e.g. Mukesh human, per-donor):

| Column | Type | Required | Notes |
|---|---|---|---|
| `sample_id` | str | yes | matches a column in the bulk matrices; pattern `{group}-{N}` |
| `group` | str | yes | factor level vocabulary in `conf/<cohort>/parameters.yml:group_levels` (e.g. `["CTRL","AD"]`) |

Cohort code in `alz/ingest_<cohort>.py` decides which shape to emit; downstream
nodes branch on `parameters.yml:cohort_design` ∈ `{"factorial","case_control"}`.

### 2.2 Animal-ID parsing

`config.parse_animal_id(s)` returns long-form `(sample_num, mouse_id_raw, sex,
timepoint, genotype)`. The canonical genotype vocabulary is
`config.SAP_FACTORIAL.keys()` after applying `config.GENOTYPE_TO_SAP`. New
cohorts use either `config.parse_animal_id` (if they share the SAP regex
format) or supply their own parser exported as `ingest_<cohort>.parse_id`.

### 2.3 Contrast vocabulary

Defined by `config.CONTRAST_COEFS` — a `dict[contrast_name → dict[coef → value]]`.
Each contrast is a linear combination over the OLS design columns. The Song
default has 9 contrasts (`App|Tau|ApTt × 2mo|4mo|6mo`); a new factorial cohort
overrides `CONTRAST_COEFS` via `conf/<cohort>/parameters.yml:contrast_coefs`.

Downstream code derives contrast lists from `config.CONTRAST_COEFS.keys()` —
no module should re-declare the 9-name list.

### 2.4 Analysis mode

`conf/base/parameters.yml:analysis_mode` ∈ `{"males_only","full_cohort"}` selects
sample subset for OLS. Read at module level via `config.ANALYSIS_MODE` today;
Phase 4 task: inject as `params:analysis_mode` to Kedro nodes.

## 3. Bulk MEA contract (mode 1)

### 3.1 Inputs

| Artifact | Schema | Cohort scope |
|---|---|---|
| `sample_mapping.csv` | §2.1 | per-cohort |
| `<cohort>_total_proteome.xlsx` or .csv | rows=proteins (`gene_symbol`, `protein_id`); cols= per-sample quantities matching `sample_mapping:column_name` | per-cohort raw |
| `<cohort>_phospho_st.csv` | rows=sites (`site_id`, `gene_symbol`, `motif`, `residue_type`); cols=per-sample quantities | per-cohort raw |
| `<cohort>_phospho_py.csv` | same schema as `st` | per-cohort raw, optional |

### 3.2 Outputs (per track `{st, py}`)

| Artifact | Schema |
|---|---|
| `total_proteome_normalized.csv` | `gene_symbol`, `protein_id`, `{column_name}…` (IRS-normalized log2) |
| `raw_phospho_normalized{,_pY}.csv` | `site_id`, `gene_symbol`, `motif`, `{column_name}…` (IRS-normalized log2) |
| `stoichiometry_matrix{,_pY}.csv` | adds `matched_protein` (bool); values = phospho log2 − parent log2 |
| `mea_stoichiometry{,_pY}.csv` | `kinase`, `contrast`, `NES`, `pval`, `FDR`, `residue_type`, `track`, `n_sites` |
| `site_level_ols{,_pY}.csv` | `site_id`, `gene_symbol`, `motif`, `contrast`, `lfc`, `se`, `pval`, `fdr`, `n_obs` |
| `mea_global_shift{,_pY}.csv` | `contrast`, `median_shift`, `mean_before`, `pct_pos_before`, `pct_pos_after` |
| `winsorized_sites{,_pY}.csv` | `contrast`, `site_id`, `gene_symbol`, `original_lfc`, `clipped_lfc`, `lower_bound`, `upper_bound` |
| `mea_substrate_sets{,_pY}.csv` | `kinase`, `contrast`, `motif`, `residue_type`, `track`, `kl_percentile` |

`contrast` cardinality = `len(parameters.yml:contrast_coefs)`. Case/control
cohorts emit one contrast per non-reference group (Mukesh: one per donor =
`{sample_id}_vs_CTRLmean`).

### 3.3 Cohort-specific knobs

Live in `conf/<cohort>/parameters.yml`:

```yaml
cohort_design: factorial          # or "case_control"
genotype_levels: [WTyp, AppP, Ttau, ApTt]   # factorial only
timepoint_levels: [2mo, 4mo, 6mo]           # factorial only
sex_levels: [ma, fe]                        # factorial only
group_levels: [CTRL, AD]                    # case_control only
reference_level: WTyp                       # the OLS baseline
contrast_coefs: { ... }                     # overrides config.CONTRAST_COEFS default
animal_id_regex: <pattern>                  # optional; default = config._ANIMAL_ID_FULL_RE
```

## 4. Decomposition MEA contract (mode 2, mouse-only)

Requires bulk MEA mode 1 outputs **plus** matched snRNA pseudobulk.

### 4.1 Inputs

| Artifact | Schema |
|---|---|
| `raw_phospho_normalized{,_pY}.csv`, `total_proteome_normalized.csv` | as in §3.2 |
| `sample_mapping.csv` | §2.1 |
| `pseudobulk_cpm.csv` | `sample` (= snRNA sample id, joins via `sample_mapping:snrna_sample_id`), `cell_type` (cluster from `config.CLUSTER_SPINE`), `{gene_symbol}…` (CPM log2) |
| `pseudobulk_cell_counts.csv` | `sample`, `cell_type`, `n_cells` |
| `cluster_spine.csv` | `cluster_name`, `in_spine` (bool) — single source of truth for the 31-cluster levy_t5 spine |

### 4.2 Outputs

| Artifact | Schema |
|---|---|
| `proportions.parquet` | `animal_id`, `cluster`, `gene_symbol`, `f_percell` (= `(expr_c / Σ_c' expr_c') × (N_total / N_c)`) |
| `protein_per_cluster.parquet` | `gene_symbol`, `animal_id`, `cluster`, `value`, `log2_value` (= `f_percell × bulk`) |
| `phospho_per_cluster{,_pY}.parquet` | adds `site_id` |
| `mea_per_cluster{,_pY}.parquet` | `cluster`, `kinase`, `contrast`, `NES`, `pval`, `FDR`, `residue_type`, `track` |
| `site_level_ols_per_cluster{,_pY}.parquet` | `cluster`, `site_id`, `contrast`, `lfc`, `se`, `pval`, `fdr` |
| `verification.json` | Verification report. Treat mass identity and spine coverage as hard decomposition gates; treat MEA concordance and Incytr pair counts as artifact-specific diagnostics unless the verifier explicitly labels them hard gates. |

### 4.3 Invariants

Verified by `alz/decomposition_mea/verify_decomposition.py`:

- **Mass identity** (per-cell-rate): `Σ_c [P_c × (N_c / N_total)] ≈ bulk`. **Not** literal `Σ_c P_c = bulk` — `f_c` weights are per-cell rates.
- **Spine coverage**: all clusters in `config.CLUSTER_SPINE` present; rank-deficient clusters emit NaN (no silent drops).
- **MEA concordance**: per-cluster vs bulk Spearman ρ and median absolute NES drift are diagnostic
  summaries. They are useful for detecting gross disagreement, but they are not reconstruction
  identities because MEA/GSEA NES values are computed after ranking, centering, winsorization, and
  enrichment normalization.

## 5. Incytr pair-mode contract (mode 3, mouse-only)

### 5.1 Inputs (built by `alz/incytr_pair/export_decomposition_for_pair.py`)

| Artifact | Schema |
|---|---|
| `{pr,ps,py}_yuyu_deconvoluted.csv` | wide; rows = (gene_symbol, site_id for ps/py); columns = `{sex}_{timepoint}_{genotype}_{cluster}` (12 conditions × 31 clusters = 372 value cols) |
| `kldata_pspy.csv` | kinase-substrate library; `kinase`, `substrate_gene`, `substrate_motif`, mouse-mapped via homologene |
| `allmarkers.csv` | spine-side marker table built by `alz/incytr_pair/build_pair_inputs.sh`; the pair-mode driver derives per-contrast DEG from this file |
| `incytr_obj.rds` | R-side spine snapshot |

### 5.2 Outputs (`Incytr::Cal_pairwise_grid` via `alz/incytr_pair/incytr_commandline.R`)

| Artifact | Schema |
|---|---|
| `wide/*_incytr_output.parquet` | Central viewer-ready pair-mode outputs after the configured significance gate/top-N cap. Current AD default is sce4-style: `(SigProb_A > 0.1 OR SigProb_B > 0.1) AND abs(PDS) >= 0.2`, then per sender/receiver Top300 up/down by `PDS`. |
| raw diagnostic directories, e.g. `_sce4_full_q0/` | Unfiltered or minimally filtered scorer outputs used for reproduction and scorer-coverage checks. Do not confuse these with central viewer-ready `wide/`. |
| `receiver_cache/receiver={cluster}/data.parquet` | Hive-partitioned for viewer; contains only active filtered pairs and rows that survived the central `wide/` gate/cap. |
| `pair_metadata.parquet` | Viewer/cache metadata for active filtered sender/receiver pairs, not raw 31×31 scorer coverage. |

### 5.3 Invariants

- **Raw scorer coverage**: when testing raw scorer completeness, check the raw/unfiltered artifact or run log for expected sender/receiver coverage. Do not use filtered `receiver_cache/pair_metadata.parquet` for this.
- **Filtered viewer readiness**: central `wide/`, `receiver_cache/`, and unified-viewer Incytr pathway shards should agree on total row count after the configured gate/cap. Filtered outputs are not expected to contain all 31×31 pairs.
- **Pair `p_value` is untrustworthy** — filter/rank on `|PDS|`. See [`feedback_no_incytr_pair_pvalue`].
- Rank-deficient clusters emit NaN.

## 6. Cross-reference correlation contract (mode 4)

### 6.1 Reference inputs (one-time downloads)

| Artifact | Schema | Source |
|---|---|---|
| `data/external/sea_ad/effect_sizes{,_early,_late}.h5ad` | genes × 139 MTG supertypes; `var["Subclass"]` | Allen SEA-AD S3 |
| `data/external/allen_abc/expression_matrices/WMB-10Xv3-{region}/log2/` | 13 region matrices | Allen ABC Atlas |
| `data/external/allen_hbca/` | HBCA WHB-10Xv3 | Allen ABC Atlas |

### 6.2 Aggregates (built by `atlas_reference.py`, `wmb_expression.py`, `human_reference_expression.py`)

| Artifact | Schema |
|---|---|
| `data/derived/aggregates/seaad/expression_by_supertype.csv` | `gene_symbol` (index) × 139 supertype columns |
| `data/derived/aggregates/hbca/expression_by_class.csv` | `gene_symbol` (index) × HBCA supercluster columns |
| `outputs/reports/wmb_expression/wmb_kinase_expression.csv` | `gene_symbol`, `cell_type` (WMB class), `specificity_score`, `mean_log2_expression`, `fraction_cells_expressing`, `binary_expressed` |
| `outputs/reports/human_reference_expression/{seaad,hbca}_kinase_specificity.csv` | `gene_symbol` × cell-type specificity scores |

### 6.3 Bridges (1-hop only, no chained mappings)

`data/derived/bridges/`:

| File | Schema |
|---|---|
| `cluster_to_wmb_class.csv` | `cluster_name`, `wmb_class_label` |
| `cluster_to_seaad_supertype.csv` | `cluster_name`, `seaad_supertype`, `weight` |
| `cluster_to_hbca_supercluster.csv` | `cluster_name`, `hbca_supercluster`, `weight` |
| `wmb_subclass_to_class.csv` | `wmb_subclass`, `wmb_class` |

All bridges land on `config.CLUSTER_SPINE` (the 31 levy_t5 cluster names) as
their **fan-out vocabulary**. Chained mappings (e.g. SEA-AD → WMB → cluster)
are forbidden — see [memory: direct_levy_t5_mapping].

## 7. Onboarding a new cohort

Checklist for adding cohort `<cohort>`:

1. Write `alz/ingest_<cohort>.py` emitting:
   - `outputs/reports/data_ingest_<cohort>/sample_mapping.csv` (§2.1 shape that matches the design)
   - `outputs/reports/kinase_attribution_<cohort>/{stoichiometry,raw_phospho}_matrix{,_pY}.csv`
   - `outputs/reports/kinase_attribution_<cohort>/total_proteome_normalized.csv`
2. Add `conf/<cohort>/parameters.yml` with cohort knobs (§3.3).
3. Run mode 1 via `KEDRO_ENV=<cohort> pixi run live`.
4. (Optional, if matched snRNA exists) supply `pseudobulk_cpm.csv` +
   `pseudobulk_cell_counts.csv` and run mode 2.
5. (Optional, if Incytr inputs exist) supply `kldata.csv` + spine artifacts and
   run mode 3.

No edits to `alz/kinase_*.py`, `alz/decomposition_mea/*`, `alz/incytr_pair/*`, or
`alz/integration/*` should be necessary. If they are, that's a contract gap
to file under [`docs/plans/repo_organization_2026-05-21.md`].

## 8. Out of scope (deferred)

- **Multi-cohort joint analysis** (combining Song + Mukesh in a single OLS).
  Today, cross-cohort comparison is done at the kinase-NES level (Mukesh →
  SEA-AD agreement → Song attribution).
- **Spine other than levy_t5**. The contract assumes the active spine; WMB-34
  and Levy-19 are closed paths.
- **Atlas drift**. Bridges are hand-curated; refreshing them when Allen
  reissues the WMB taxonomy is a separate prerequisite.

## References

- Master plan: [`docs/plans/repo_organization_2026-05-21.md`](../plans/repo_organization_2026-05-21.md)
- Live pipeline contract (runtime view, complementary): [`docs/foundation/live_pipeline_contract.md`](./live_pipeline_contract.md)
- Analysis charter (scope, closed paths): [`docs/foundation/analysis_charter.md`](./analysis_charter.md)
