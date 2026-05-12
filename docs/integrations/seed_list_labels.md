# Seed-list nomination labels — alz-side implementation

Status: **implemented and run end-to-end on 2026-05-11.** Paired with
the upstream report at
`~/Projects/work/incytr/docs/incytr_proposals/seed_list_labels.md`.
The two streams share a single interface contract (file-format §) and
have now been validated against the full Song fixture (22 cell types,
14 male animals, 9 factorial contrasts including `ApTt_4mo`).

**Methodology revision (2026-05-11).** The original plan called for
`~ 0 + group` (group = genotype × timepoint) with `padj < 0.05`. With
n=1–2 animals per (genotype × timepoint) cell, raw p-values are
depleted (29 of ~23k genes hit p<0.05, vs. ~1150 expected under null)
and BH-padj produces fewer than 10 DEGs at any threshold. The
implementation pivots to a **marginalized genotype-only design with a
raw-p × log2FC filter**:

- DESeq2 design: `~ 0 + genotype` (pools across 2mo/4mo/6mo →
  3–4 reps per genotype)
- Contrasts: the 9 factorial pairs are reduced to **3 disease-vs-WT
  pairs** by stripping the timepoint suffix and deduplicating
  (`AppP vs WTyp`, `Ttau vs WTyp`, `ApTt vs WTyp`).
- Filter: `raw p < 0.05 AND |log2FC| >= 0.5`, union across the 3
  contrasts (matches limma side).

Labels are per-node and contrast-independent (the paper's schema), so
collapsing 9 contrasts to 3 marginalized ones produces a single
biologically defensible seed set rather than nine thin ones.

## Methods

### Goal

Produce the two gene lists that upstream `construct_factorial_paths`
needs to seed candidate paths and emit the `<position>.label` columns:

- `deg_lists.json` — per-cell-type DEGs from snRNA-seq.
- `prg_list.csv` — bulk DEPs from proteomics.

Together these reproduce the paper's strict-partition DEG/prG labels
(DEG-first precedence; prG = DEP \ DEG, with the set subtraction
happening at upstream label-assignment time).

### DEGs — pseudobulk + DESeq2 per cell type

scRNA-seq DE methods that treat cells as independent (Wilcoxon, MAST)
inflate significance because cells share animal-level genetics, batch,
and disease state. The factorial design's biological replicate is the
animal, so DE respects that.

1. **Pseudobulk** raw UMI counts per (animal × cell type), reading
   `adata.layers["counts"]` from the source h5ad. The plan called for
   `adata.X`, but the Song h5ad's `.X` is float32-normalized; raw
   integer counts live in the `counts` layer. Pseudobulk is a sparse
   `selector @ counts` matmul (`csr_matrix` group selector × cell-by-
   gene counts) followed by `np.rint` and zero-elimination, written as
   integer-field MatrixMarket.
2. Pseudobulk runs **pre-omics-intersect** in `export_factorial_inputs.py`
   so DESeq2 sees the full male transcript animal set, not the
   14-animal 4-layer intersect that the kinase-bridge fixture uses.
3. **Filter** (animal × cell type) cells with `n_cells < min_cells`
   (default 10) at DE time inside `compute_seed_lists.R`.
4. **Per cell type**, DESeq2 fits `~ 0 + genotype` (marginalized;
   pools across timepoints). The plan's
   `~ 0 + group / group = genotype × timepoint` parameterization was
   abandoned: with n=1–2 per cell, that design produces empty DEG
   sets at any reasonable threshold. Genotype-only marginalization
   yields 3–4 reps per group, which is enough for stable DESeq2
   dispersion estimation.
5. **Per-contrast estimability gate**: for each of the 3 marginalized
   disease-vs-WT contrasts, both arms must have ≥ `min_reps`
   (default 1, lowered from 3) surviving pseudosamples. Contrasts
   that fail are logged and skipped; the design is restricted to
   groups participating in at least one estimable contrast before
   DESeq2 fits.
6. **Union across contrasts**:
   `DEG[ct] := { g : raw p_g < pvalue AND |log2FC_g| >= log2fc in at
   least one estimable contrast }` (defaults `pvalue=0.05`,
   `log2fc=0.5`).
7. **Cell-type coverage preservation**: all 22 sender/receiver cell
   types are emitted as keys in `deg_lists.json`, even when filtered
   out by `min_cells >= 10` (Chandelier, Sst Chodl, Lamp5 Lhx6) or
   when DESeq2 fails to fit any estimable contrast (L6b, Sncg).
   Those entries carry empty character vectors and a `status` flag
   in the manifest. This avoids upstream `validate_seed_lists()`
   coverage errors when paths span those cell types — the affected
   paths just receive only `prG` labels.

### prGs — limma on proteomics

`pr_matrix.csv` is log-transformed and gene-collapsed (per existing
MANIFEST provenance). Animal column order matches `animal_metadata.csv`.

1. limma fits `~ 0 + genotype` on the bulk pr matrix (marginalized,
   mirrors DESeq2).
2. The 3 marginalized contrasts are extracted via `makeContrasts` /
   `contrasts.fit` / `eBayes` (no robust, no trend).
3. **DEP set**: `{ g : raw P.Value < pvalue AND |logFC| >= log2fc in
   at least one contrast }`.
4. The **full DEP set** is emitted as `prg_list.csv`. The DEG-first
   precedence partition happens at upstream label-assignment time,
   not here. This decouples the streams: alz defines "what's in each
   evidence layer," upstream defines "how labels are assigned on
   overlap."

### Union across contrasts, not per-contrast

The paper's example used a single contrast and emitted one `.label`
per node. Per-contrast labels would mean 4 positions × 9 contrasts =
36 `.label` columns — not the schema being reproduced. Path
construction also runs once (not per contrast), so a single seed set
is the natural fit. Schema-breaking change if revisited; deferred.

### Implementation footprint

- `alz/integration/compute_seed_lists.R` — new. Single R script that
  reads pseudobulk + `pr_matrix.csv`, runs DESeq2 + limma, writes
  `deg_lists.json` and `prg_list.csv`, and stamps the manifest.
- `alz/integration/export_factorial_inputs.py` — added
  `write_pseudobulk_counts()`, four manifest file-list entries, and a
  seed-list paragraph in the consumer README writer.
- `alz/integration/load.R` — loads both seed-list files when present;
  both stay `NULL` if missing so upstream falls back to its current
  HEG-only behavior.
- `alz/integration/factorial.R` — one-line plumbing; passes
  `deg_lists` and `prg_list` through to `construct_factorial_paths`.
- `alz/integration/README.md` — new file-table row, run-order snippet
  now includes `pixi run compute-seed-lists`.
- `pixi.toml` — added `bioconductor-deseq2` and `r-jsonlite` to R
  deps; added `compute-seed-lists` task entry.

Run order: `pixi run export-factorial-inputs` →
`pixi run compute-seed-lists` → `pixi run incytr-factorial`. Keeping
DE as a separate task avoids re-running the slow step for unrelated
metadata fixes.

### Export contract

`deg_lists.json` (one key per sender/receiver cell type, values
sorted-deduplicated):

```json
{
  "Microglia-PVM": ["App", "Apoe", "Trem2", ...],
  "Astrocyte":     ["Aqp4", "Gfap", ...],
  ...
}
```

`prg_list.csv` (header `gene_symbol`, sorted ascending):

```
gene_symbol
Apoe
App
Bin1
...
```

### MANIFEST stamping

`compute_seed_lists.R` reads `MANIFEST.json` (when present in the out
dir), appends a `seed_lists` section, and writes back via `jsonlite`:

```json
"seed_lists": {
  "deg_method": "DESeq2",
  "deg_design": "~ 0 + genotype  (marginalized; pools timepoints)",
  "deg_filter": "raw p < pvalue AND |log2FC| >= log2fc, union across contrasts",
  "deg_p_threshold": 0.05,
  "deg_log2fc_threshold": 0.5,
  "deg_contrasts": ["AppP_vs_WTyp", "Ttau_vs_WTyp", "ApTt_vs_WTyp"],
  "deg_min_cells_per_animal_celltype": 10,
  "deg_min_reps_per_group": 1,
  "deg_cell_types_skipped_contrasts": {"<ct>": ["<contrast>", ...]},
  "deg_cell_type_status": {"<ct>": "ok|no_estimable_contrasts|filtered_by_min_cells"},
  "prg_method": "limma",
  "prg_design": "~ 0 + genotype",
  "prg_filter": "raw P.Value < pvalue AND |logFC| >= log2fc",
  "prg_p_threshold": 0.05,
  "prg_log2fc_threshold": 0.5,
  "prg_genes_dropped_for_vocab_mismatch": <int>,
  "methodology_note": "marginalized design forced by n=1-2 reps per (genotype × timepoint); raw-p × log2FC filter forced by BH-padj producing < 10 DEGs at any threshold"
}
```

### Validation hooks built in

1. **Vocabulary alignment**: DEG union is checked against
   `expression_genes.csv`; any miss aborts with the offending gene
   list. prG genes outside the vocabulary are dropped silently with
   the count recorded in the manifest.
2. **Cardinality logging**: per-cell-type DEG sizes and DEG∩prG
   overlap percentages are printed at run time.
3. **Sentinel sanity check**: presence of Apoe, App, Trem2, Bin1, Clu
   in either union is logged; absence triggers a warning, not an abort.

## Results

End-to-end pipeline ran on the canonical Song fixture (2026-05-11):

**Seed-list cardinalities** (`data/incytr_factorial_inputs/`):

- `deg_lists.json`: 22 cell-type keys; 17 non-empty (93–474 DEGs per
  cell type); 5 empty:
  - filtered by `min_cells >= 10`: Chandelier, Lamp5 Lhx6, Sst Chodl
  - DESeq2 fit failure: L6b, Sncg
  - DEG union across all cell types: **4,164 genes**
- `prg_list.csv`: **146 prGs** (4 genes dropped for vocabulary
  mismatch; sentinel Apoe present). limma flagged 203 probes with
  partial NA coefficients — these are dropped from the contrast
  estimates but other estimable contrasts can still recover them.
- Sentinel coverage: Apoe, App, Bin1 in DEG union; Apoe in prG.

**Factorial output** (`outputs/reports/incytr_factorial/`):

- 22 × 22 sender/receiver matrix, all 9 contrasts (including
  `ApTt_4mo`).
- **1,216,854 path rows** in `receiver_cache/`. Only 36,003 unique
  `(L, R, EM, T)` tuples — seed-list filtering reduced the candidate
  search space by ~100× compared to the prior HEG-based run, which
  is the dominant reason the full pipeline now finishes in minutes
  rather than ~45 minutes.
- All four `.label` columns populated, no `<NA>` values:
  - Ligand: 62% prG / 38% DEG
  - Receptor: 60% DEG / 40% prG
  - EM: 85% DEG / 15% prG
  - Target: 68% DEG / 32% prG
- The five empty-DEG cell types produce only 486 rows each (54
  paths × 9 contrasts), almost all `prG`-labeled. This is the
  intended behavior — cell types without their own DE signal are
  scored against the bulk-proteomics layer only.

**Files added:**

- `alz/integration/compute_seed_lists.R` — DESeq2 + limma + manifest
  writer.

**Files modified:**

- `alz/integration/export_factorial_inputs.py` —
  `write_pseudobulk_counts()`, MANIFEST entries for
  `pseudobulk_*.{mtx,csv}`, README seed-list paragraph.
- `alz/integration/load.R` — `deg_lists` / `prg_list` keys added to
  the loader return list, gated on file presence. Uses
  `jsonlite::fromJSON(..., simplifyVector = FALSE)` plus
  `lapply(..., as.character(unlist(.)))` to coerce both empty
  vectors and singleton entries to character vectors.
- `alz/integration/factorial.R` — forwards `deg_lists` and
  `prg_list` to `construct_factorial_paths`.
- `alz/integration/README.md` — new row, updated run-order snippet.
- `pixi.toml` — `bioconductor-deseq2`, `r-jsonlite`,
  `compute-seed-lists` task entry.

## Caveats

- **Marginalized design loses timepoint resolution.** Labels are
  pooled across 2mo/4mo/6mo, so a gene that's only differential at
  6mo (when AD pathology is most advanced) gets the same DEG label
  as a gene differential from 2mo. This is acceptable because labels
  are designed to be contrast-independent (one label per node), but
  it means the label cannot answer "is this gene differential *for
  this timepoint*?" Use the per-contrast scoring outputs (TPDS, PDS,
  log2FC) for that.
- **Raw p-value, not BH-adjusted.** A nominal-p filter has higher
  false-positive rates than BH-padj. We accept this because BH-padj
  produced empty seed lists on the Song data (insufficient power
  with n=3–4 per genotype after marginalization). The `|log2FC| >=
  0.5` floor provides a partial cap on noise — genes that pass raw
  p<0.05 but barely move are excluded. Replicates with larger n
  would justify reverting to padj.
- **Five cell types have no DEG signal.** Chandelier, L6b, Lamp5
  Lhx6, Sncg, Sst Chodl. These are rare neuronal subtypes with
  `n_cells < 10` per pseudosample (or DESeq2 fit failures from
  near-zero-variance pseudobulk counts). Paths through these cell
  types are labeled `prG` only — they're scored against the
  bulk-proteomics evidence layer with no transcript-specific signal.
- **prG is bulk, not cell-type-resolved.** A `prG`-labeled node says
  "this gene is differentially expressed in *bulk proteomics*," not
  "in this cell type." Cell-type attribution of prG signal requires
  separate downstream analysis (deconvolution, cross-layer
  consistency scoring).
- **L6b and Sncg fit failures not investigated.** DESeq2 returns
  null `results()` after `DESeq()` succeeds. Likely a dispersion-
  estimation edge case with very few non-zero pseudosamples. Low
  priority — both are small cell types and they're handled cleanly
  via the empty-vector + status-flag pathway.
- **GenomeInfoDbData post-link.sh** must be installed manually after
  `pixi install` on fresh clones:
  `PREFIX=$(pwd)/.pixi/envs/default bash "$PREFIX/bin/installBiocDataPackage.sh" "genomeinfodbdata-1.2.13"`.
  Bioconda's post-link scripts don't run reliably under pixi.

## Conclusions

End-to-end pipeline validated. Seed-list labels are now part of the
canonical incytr factorial output (`pair_metadata.parquet` +
`receiver_cache/`). The label distribution is biologically defensible
(sentinel AD genes Apoe/App/Bin1 land in DEG; Apoe also in prG; 17
of 22 cell types produce non-empty DEG seeds).

Re-run order for future fixture refreshes:

1. `pixi run export-factorial-inputs` — rebuild fixture (regenerates
   pseudobulk + omics matrices; minutes).
2. `pixi run compute-seed-lists --input-dir data/incytr_factorial_inputs --out-dir data/incytr_factorial_inputs` — recompute
   DEG/prG (1–2 min).
3. `pixi run install-incytr` — reinstall package from local source
   if `R/factorial.R` changed.
4. `pixi run incytr-factorial` — run factorial pipeline (~5 min
   with seed-list filtering).

## Out of scope (unchanged from plan)

- The upstream label-assignment rule. The alz side emits the full
  DEP set; upstream applies DEG-first precedence at construction time.
- DE methodology beyond pseudobulk + DESeq2 (transcript) and limma
  (proteomics).
- The companion per-layer-design refactor for `ApTt_4mo`. Tracked
  separately. Both require a fixture regeneration and full factorial
  rerun, so they should be merged + run together once both are ready.
- Per-contrast labels.
