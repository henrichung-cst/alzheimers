# Per-cluster decomposition → per-cluster MEA → Incytr pivot

Supersedes the prior bulk-deconvolution proposal at this path
(commit history retains the original). The earlier doc raised three
options for crossing the closed-deconvolution-path gotcha; this plan
takes a fourth route that doesn't re-open the inverse problem:
**proportional decomposition** using snRNA-derived per-gene cell-type
proportions, applied to raw bulk values. The closed-path doc remains
authoritative for *statistical* deconvolution.

## Principle

Each stage publishes to `outputs/reports/<stage>/`; each downstream
stage reads from there, never from raw upstream inputs. Sequential
dependencies accepted in exchange for zero parallel ingest paths.

## Pipeline (new stages in **bold**)

```
1. ingest                                         data_ingest.py
2. normalize        (bulk stoichiometry)          kinase_normalize.py
3. enrich           (bulk stoichiometry MEA)      kinase_enrich.py
4. attribute        (per-cluster evidence prior)  kinase_attribute.py
   ───────────────── new ─────────────────
5. snrna_proportions   → f_c(G, A) per animal where snRNA observed,
                         per (genotype × timepoint × sex) pooled otherwise
6. decompose           → per-cluster raw phospho + raw protein, linear-space
7. enrich_celltype     → per-cluster raw-phospho MEA, 19 × 9 contrasts
   ───────────────── consumer ──────────────
8. incytr              → reads steps 6 (matrices) + 7 (MEA)
```

Bulk stoichiometry MEA (step 3) and bulk attribution (step 4) **stay
exactly as they are**. They remain the activity-corrected primary
deliverable. The per-cluster track is a cell-type-localization
supplement, abundance-aware and explicitly not activity-corrected.

## Why per-cluster *raw phospho* and not per-cluster stoichiometry

Same-proportion decomposition collapses:

```
stoich_c(site, A) = log2(f_c × phospho) − log2(f_c × protein)
                  = log2(phospho) − log2(protein) = bulk_stoich
```

Identical for every cluster — zero per-cluster information. The fix is
to run MEA on per-cluster **raw phospho**, accepting that the result
mixes activity and abundance. Activity-correction lives in the bulk
track; the per-cluster track answers "which cell types carry this
phospho signal."

## Stage 5 — `snrna_proportions.py`

**Input:** Song snRNA pseudobulk (already produced by
`snrna_integration.py` Stage 1 on the Levy-19 spine).
**Output:** `outputs/reports/decomposition/levy19/proportions.parquet`,
keyed `(gene, cluster, animal_id)`, column `f`. Per-gene sum across
clusters = 1.

- 28 snRNA-observed animals: per-animal `f_c(G, A)`
- 44 TMT-only animals: pooled `f_c(G | genotype × timepoint × sex)`
  group mean. Preserves AD-driven proportion shifts (gliosis, neuron
  loss); collapses only within-group variation. Pure pooled mean is
  rejected as statistically inappropriate.

Sidecar `proportions_provenance.csv` flags per (animal, gene) whether
`f` is observed or imputed, and from which group.

## Stage 6 — `build_celltype_decomposition.py`

Generalization of `alz/deconvolution/build_wmb_decomposition.py`.
Rename `alz/deconvolution/` → `alz/decomposition/` to break the
historical association with the closed inverse-problem path.

**Inputs:**
- `outputs/reports/kinase_attribution/raw_phospho_normalized.csv`
- `outputs/reports/kinase_attribution/total_proteome_normalized.csv`
  (add to `kinase_normalize.py` if not already emitted)
- `outputs/reports/decomposition/{spine}/proportions.parquet`

**Math (linear space, corrected):**

```
P_c(gene, A)    = f_c(gene, A) × 2^log2_protein(gene, A)
Phos_c(site, A) = f_c(parent_gene(site), A) × 2^log2_phospho(site, A)
```

Then re-log for downstream symmetry. The branch code's log-space
multiplication is incorrect and gets fixed here. Identity check:
`Σ_c P_c(gene, A) ≈ bulk P(gene, A)` in linear space, to floating-point
tolerance.

**Outputs:**
- `decomposition/{spine}/protein_per_cluster.parquet`
- `decomposition/{spine}/phospho_per_cluster.parquet`

CLI: `python -m alz.decomposition.build_celltype_decomposition
--spine levy19`. Spine is a flag, not a hard-coded constant.

## Stage 7 — `enrich_celltype.py`

Per-cluster MEA on raw phospho. Reuse the OLS + GSEApy code from
`kinase_enrich.py` — refactor into a shared helper rather than
duplicate.

For each of the 19 clusters: OLS on per-cluster phospho × 9 contrasts
→ MEA pre-ranked on β (median-centered + winsorized, same protocol as
bulk).

**Output:** `outputs/reports/kinase_attribution_celltype/{spine}/
mea_celltype.parquet`, columns `(kinase, cluster, contrast, NES, FDR,
n_substrates)`.

Interpretation contract (written into output README):
- Reflects **abundance change** of phosphosites in a cell type
- Does **not** distinguish activity from protein-abundance change
- Used to answer "which cell types carry the phospho signal," not
  "is this kinase more active in this cell type"
- Cross-check against step 4 bulk attribution: agreement = robust,
  disagreement = flag

## Stage 8 — Incytr integration

Replaces `omics_loaders.py` entirely. Integration reads from
`outputs/reports/decomposition/{spine}/` and
`outputs/reports/kinase_attribution_celltype/{spine}/`; no raw XLSX
reads anywhere in the integration directory.

`export_factorial_inputs.py`:
- Drop `omics_loaders.py` calls; read parquet from stage 6
- Emit per-cluster parquet bundles in the export
- Switch h5ad cell labels to barcode-keyed Levy-19 (the previously-
  planned Step 8a work folds in here)

`load.R`:
- Replace `read.csv("{pr,ps,py}_matrix.csv")` with `arrow::read_parquet`
  loop over `per_cluster/{layer}/*.parquet`
- Build `list(pr = list(data_wide = list(cluster1 = mat1, ...)))`
- Upstream `resolve_wide` already dispatches on `is.list` — no R
  package changes needed

`config_integration.py`: switch label set to `config.CLUSTER_SPINE`.

## Branch code to promote / retire

| Branch file | Disposition |
|---|---|
| `alz/deconvolution/build_wmb_decomposition.py` | Promote → `alz/decomposition/build_celltype_decomposition.py`; generalize spine; fix log-space math |
| `alz/deconvolution/mea_per_celltype.py` | Promote → `alz/decomposition/enrich_celltype.py`; share OLS+GSEA with `kinase_enrich.py` |
| `alz/deconvolution/per_animal_extension.py` | Salvage proportion-imputation logic; reject pure pooled mean |
| `alz/deconvolution/run_per_animal.py` | Retire — replaced by per-stage CLI |
| `alz/deconvolution/factorial_ols.py` | Audit before retiring; may merge into `kinase_enrich.py` shared helper |
| `alz/deconvolution/{cohort_concordance,variance_audit,confidence}.py` | Audit before retiring — validation logic may be worth keeping |
| `alz/deconvolution/README.md` | Rewrite as `docs/foundation/per_cluster_track.md` |
| `archive/deconvolution/docs/deconvolution_infeasibility.md` | Keep — closes statistical deconvolution, not proportional decomposition |

## Charter implications

`docs/foundation/analysis_charter.md` "closed paths" list and CLAUDE.md
"Do not reopen closed paths" gotcha need a clarifying note: the
closure applies to **statistical deconvolution** (inverse problem,
recovering per-cell-type LFCs as an inferential target). Proportional
decomposition (forward projection of bulk values onto pre-existing
cell-type proportions) is a different operation and is permitted as a
supplement to the bulk track.

This is a clarification, not a re-opening.

## Failure modes to instrument

1. **Gene coverage shrinkage.** snRNA ~15k genes, proteome ~7k. Only
   the intersection gets per-cluster values. Emit a coverage report.
2. **Sites whose parent gene is absent from snRNA.** Drop (cleaner
   than fallback-to-bulk); report count.
3. **Imputation share.** From `proportions_provenance.csv`, what
   fraction of (gene, animal) cells are imputed vs observed? If > 60%,
   warn loudly.
4. **Bulk-vs-per-cluster MEA divergence.** Summary table comparing
   step 3 bulk MEA to step 7 per-cluster MEA aggregated across
   clusters; large divergence may signal decomposition error.

## Verification

1. **Math:** `Σ_c P_c ≈ bulk P` in linear space, per (gene, animal),
   to floating-point tolerance.
2. **Coverage:** all 19 clusters present in stage 6 outputs; no
   silent drops.
3. **Sanity:** per-cluster MEA reproduces bulk MEA when clusters are
   averaged with `f_c`-weights.
4. **Incytr:** `pixi run incytr-factorial` produces 19² = 361 scored
   sender × receiver cluster pairs.

## Revised sequencing (supersedes earlier Step 7–11 plan)

- **Step 7** (current, done) — `kinase_attribute.py` to Levy-19
- **Step 8** — Promote `alz/deconvolution/` → `alz/decomposition/`;
  retire dead code; rename
- **Step 9** — Implement Stage 5 (`snrna_proportions.py`)
- **Step 10** — Implement Stage 6 (`build_celltype_decomposition.py`,
  linear-space, spine-parametrized)
- **Step 11** — Implement Stage 7 (`enrich_celltype.py`) — **done**
- **Step 12** — Rewire Incytr integration (Stage 8) — **done**
- **Step 13** — End-to-end smoke run, verification, docs/charter
  refresh — **done**

  Stage 6 extended to the pY track via `--track {st,py,both}`. Smoke
  runner at `alz/runners/main/run_pivot_smoke.sh` chains
  normalize → Stage 5 → Stage 6 (st+py) → Stage 7 → factorial export
  → Incytr → verification. Verification harness at
  `alz/decomposition/verify_decomposition.py` runs the four §Verification
  contracts and writes
  `outputs/reports/decomposition/levy19/verification.json`.

## Out of scope

- True statistical deconvolution (remains closed)
- Per-cluster stoichiometry (algebraically collapses to bulk under
  same-proportion-for-both)
- Cluster-specific phospho occupancy proportions (no data source)
- Upstream `Incytr` R package changes (already supports per-cluster
  via `resolve_wide` list-dispatch)
