# Plan — Point A: MEA-derived candidate-gene expansion for pair-mode Incytr (seed-list step)

## Context

We are adding MEA kinase-activity evidence to the pair-mode Incytr integration as a new input layer (analogous to how DEG and prG already feed the candidate gene pool). MEA enrichment (kinase-library, run on stoichiometry β values via `alz/kinase_enrich.py` and per-cluster via `alz/decomposition/enrich_celltype.py`) detects kinases whose substrate sets shift coherently in a contrast — including kinases that are *active without abundance change*, which DEG/HEG cannot see.

This work is **step 1 of 2** under the agreed sequencing:
- **Point A (this plan):** expand the candidate gene pool that drives path enumeration. Kinases with significant per-cluster MEA NES enter the per-cluster gene set with a new `"mea"` label, surfacing paths that would not otherwise be enumerated. Validates the seed-list expansion in isolation.
- **Point B (deferred to a later plan):** add an additive MEA term to `Cal_PDS` (new `Integr_kinaseactivity` upstream function), with `mea_rescued` path flag and `PDS_no_mea` sidecar. Adds a scoring contribution to paths that would have been enumerated anyway. Will be designed after Point A's outputs are validated.

The pair-mode candidate pool and per-member labels are assembled **entirely AD-side**, in `bench/incytr_pair_levy_t5/incytr_commandline.R` lines 254–402, not in upstream `~/Projects/work/incytr/`. Pair-mode `Export_results` does not track per-node provenance (confirmed by the existing comment in the driver at line 383). Point A therefore requires **zero changes to the upstream R package** — all plumbing happens in the AD-side bench driver and a new MEA-gene-list builder.

## Design decisions (locked)

1. **Plumbing — sidecar file, union in driver.** Keep `bench/incytr_pair_levy_t5/incytr input/input_gene_list.csv` unchanged. Emit a new `mea_gene_list.csv` (same `(gene, cluster)` schema) alongside it. The driver unions both at runtime. Single evidence type per file; provenance is unambiguous.
2. **MEA threshold — FDR < 0.25 (kinase-pipeline default).** Matches the threshold used elsewhere in this repo for declaring a kinase enriched. No new threshold to defend.
3. **Label union — multi-label set.** A kinase that is *both* DEG (or HEG, folded into DEG by `build_input_gene_list.R`) *and* MEA-significant gets a label like `"DEG+mea"`. Preserves the ability to identify (a) genes whose evidence stacked across sources and (b) the operationally useful "MEA-only" case (label is exactly `"mea"`).
4. **Scope — kinase genes only (Type 1).** Only the *kinase gene itself* enters the candidate pool when MEA-significant. Leading-substrate genes (Type 2) are deferred — they require back-mapping motif peptides to substrate gene symbols, which is additional infrastructure better handled after Type 1 is validated.
5. **Per-cluster, not bulk.** Use `outputs/reports/decomposition/levy_t5/mea_per_cluster.parquet` (and `mea_per_cluster_pY.parquet`). Filter per-cluster rows for cluster C → gives the kinases significant in cluster C → those kinases enter cluster C's gene-list rows. Matches the receiver-cluster granularity of the existing `SiK_score` arm.
6. **pS and pY tracks unioned.** A kinase passing FDR < 0.25 on either residue track enters the pool. The label remains `"mea"` (track origin is informational; can be added as a separate column later if needed).

## Actions

### 1. New: `bench/build_mea_gene_list.R`

Mirror the shape of `bench/build_input_gene_list.R`. Inputs:

- `outputs/reports/decomposition/levy_t5/mea_per_cluster.parquet` (pS track)
- `outputs/reports/decomposition/levy_t5/mea_per_cluster_pY.parquet` (pY track)
- `config_integration.py`-equivalent cluster list (the 31 levy_t5 clusters; or read from the cluster spine — pick whichever is cheapest)

Logic:

1. Read both parquets; union row-bind. Columns of interest: `cluster, kinase, NES, FDR, contrast`.
2. Filter `FDR < 0.25`. Across all 9 contrasts, take the union of (cluster, kinase) pairs that pass on *any* contrast. (Rationale: a kinase active in any contrast should be in the candidate pool for that cluster; the contrast-level signal is handled at the scoring stage in Point B.)
3. Map kinase library kinase names to mouse gene symbols. The kinase-library uses canonical kinase names that mostly match gene symbols, but check for mismatches against the existing `kldata.csv`'s `motif.geneName` column — that's already the authoritative mapping used by pair-mode. If `data/datasets/song/analysis_cache/kinase_to_gene_mapping.csv` exists (from `alz/map_kinases_to_genes.py`), use it.
4. Emit `bench/incytr_pair_levy_t5/incytr input/mea_gene_list.csv` with two columns: `gene, cluster`. Same shape as `input_gene_list.csv`.
5. Log counts: kinases passing per cluster, total (gene, cluster) rows, fraction overlapping with `input_gene_list.csv`.

### 2. Edit: `bench/incytr_pair_levy_t5/incytr_commandline.R` (lines 254–402)

Two changes:

**(a) Union the MEA gene list into the candidate pool.** After the existing `DG.Sender` / `DG.Receiver` construction (lines 259–314), read `mea_gene_list.csv` and union the matching per-cluster gene rows into both `gene.use_Sender` and `gene.use_Receiver`. Keep the existing top-500 proteomics union step unchanged.

**(b) Extend the label assignment (lines 385–402).** Currently labels are scalar strings `"DEG"`, `"prG"`, or `NA`. Change to set-valued strings via concatenation:

- Build three named vectors per cell type: `deg_set`, `pr_set`, `mea_set` — each maps `gene → TRUE` if the gene was in the corresponding source for that cell type.
- For each path member (L, R, EM, T), look up which of `{DEG, prG, mea}` contain it, and assign the label as the sorted concatenation: `"DEG"`, `"DEG+mea"`, `"mea"`, `"prG+mea"`, `"DEG+prG+mea"`, etc. Members with no source remain `NA`.
- Keep the four column names: `Ligand.label`, `Receptor.label`, `EM.label`, `Target.label`.

### 3. Edit: `bench/run_pair_mode.sh` (or wherever the input-list builder is invoked)

Add a call to `bench/build_mea_gene_list.R` before pair-mode invocation, gated on the per-cluster MEA parquets existing. Fail loudly if the parquets are missing — do not silently fall through to a DEG-only pool.

### 4. Edit: `alz/integration/pair_to_receiver_cache.py` (if it filters columns)

Confirm that the four `*.label` columns survive the reshape into `receiver_cache/`. They were already passed through pre-change (DEG/prG); the new multi-label string values must also survive. If column-typing or filtering drops them, fix.

### 5. Edit: `docs/integrations/kinase_incytr_integration.md`

Add a short "MEA-derived candidate genes (Point A)" subsection under the file inventory: name the new `build_mea_gene_list.R` script, describe the `mea_gene_list.csv` sidecar, and note the multi-label convention. Do not pre-document Point B; that lands when Point B's plan executes.

## Critical files

**To create:**
- `bench/build_mea_gene_list.R` — new per-cluster MEA→gene-list builder.

**To edit:**
- `bench/incytr_pair_levy_t5/incytr_commandline.R` (lines 259–314, 385–402) — union the new sidecar; extend label assignment to multi-label strings.
- `bench/run_pair_mode.sh` — invoke the new builder before pair-mode.
- `alz/integration/pair_to_receiver_cache.py` — verify label columns survive (edit only if needed).
- `docs/integrations/kinase_incytr_integration.md` — short Point A subsection.

**Reused (read-only references):**
- `bench/build_input_gene_list.R` — shape model for the new MEA builder.
- `outputs/reports/decomposition/levy_t5/mea_per_cluster.parquet` + `mea_per_cluster_pY.parquet` — MEA inputs (columns: `cluster, kinase, ES, NES, p-value, FDR, Subs fraction, Leading substrates, contrast, residue_type, track`).
- `bench/incytr_pair_levy_t5/incytr input/kldata.csv` — authoritative kinase-name→gene-symbol vocabulary for the kinase-name mapping step.
- `data/datasets/song/analysis_cache/kinase_to_gene_mapping.csv` (if present) — cached mapping from `alz/map_kinases_to_genes.py`.

## Verification

1. **Builder smoke test.** Run `Rscript bench/build_mea_gene_list.R` standalone. Confirm `mea_gene_list.csv` is created, has the `(gene, cluster)` schema, and the row counts are sensible (tens of kinases per cluster, not zero and not thousands).
2. **Kinase-name resolution check.** Print any kinase names from the MEA parquets that fail to resolve to gene symbols. Should be near-zero (kinase-library naming closely matches gene symbols); investigate if more than ~5%.
3. **Driver-level label sanity.** Run a single pair-mode invocation (one sender × receiver pair) end-to-end. Inspect the output parquet's four `*.label` columns: confirm new multi-label strings appear (`"DEG+mea"`, `"mea"`, etc.), and that paths previously labeled `"DEG"` still are.
4. **Universe-expansion check.** Compare path counts pre/post: re-run a small slice (one receiver, one contrast) with the new builder enabled vs. with it disabled. The MEA-enabled run should have ≥ the same path count as the disabled run (strict ≥; new paths possible, no path should disappear).
5. **MEA-only rescue check.** Count paths whose at-least-one member has label exactly `"mea"` (no DEG/prG support). These are the operationally-rescued paths. Print the driving kinases and inspect a handful manually for biological sensibility (AD-relevant kinases like GSK3, CDK5, MAPK family, JNK, p38 should appear).
6. **Receiver-cache pass-through.** After `pair_to_receiver_cache.py`, confirm `*.label` columns survive in the long-form output.

End-to-end:

```bash
bash bench/run_pair_mode.sh         # exercises build_mea_gene_list.R + driver
bash alz/runners/main/run_pair_mode_viewer_build.sh   # reshape verification
```

## Out of scope (deferred)

- **Point B** (additive MEA term in `Cal_PDS`, `mea_rescued` flag, `PDS_no_mea` sidecar, `mea_driving_kinases` column). Designed in a separate plan after Point A is validated.
- **Leading-substrate (Type 2) genes.** Adding substrate gene symbols to the candidate pool via MEA leading-substrates requires motif-to-gene back-mapping. Defer until Type 1 (kinase genes only) is validated and we have a sense of whether substrate-side expansion is needed.
- **Bulk MEA fallback.** We use per-cluster MEA only. If per-cluster parquets are missing, the builder fails — do not silently fall back to bulk.
- **Upstream `Cal_pairwise_grid` modifications.** None required for Point A. Upstream changes are entirely under Point B's scope.
- **Existing factorial-era docs cleanup.** The archive cleanup from the previous plan is complete; the residual surgical doc fixes (in `docs/INDEX.md`, `docs/foundation/`, etc., per the prior plan's deferred list) are still deferred and unrelated to this work.
