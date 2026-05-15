# Incytr pair-mode benchmark vs factorial (Levy-19)

Side-by-side benchmark of the legacy Yuyu/Song pair-mode Incytr driver
(`data/incytr/incytr_commandline.R`) against the live factorial pipeline
(`alz/integration/factorial.R`), with both runs on the **Levy-19 cluster spine**
and males-only. The goal is to isolate the scoring-method delta from the
input-pipeline delta — same inputs, two scoring backends, comparable contrasts.

This is a **one-off methodological benchmark**, not a reopening of the closed
direct-deconvolution path. The deliverable is a comparison table + figure;
nothing in the live pipeline changes.

---

## Decisions (from intake)

| Question | Decision |
|---|---|
| 46→19 strategy | **Rebuild inputs on 19 first.** Driver sweeps 19² = 361 pairs natively. |
| Contrast definition | **Disease vs WT at same timepoint.** 9 runs total: {App, Tau, ApTt} × {2mo, 4mo, 6mo}. |
| Input source | **Reuse live Levy-19 decomposition outputs.** No re-execution of v2 provenance notebooks. |
| Sex scope | **Males-only.** snRNA pseudobulk uses 14 male animals (subset of 28); proteomics uses 33 males (post-outlier). Different N per omic, all male. |
| kldata source | **Yuyu-derived kldata (mandatory).** 5xFAD fallback retired in both pair-mode and live factorial — see Phase 0. |
| Self-pair handling | **Run all 19² = 361 pairs**; 19 self-pairs drop naturally on inner-join with factorial's 342, kept as a sidecar table for autocrine biology. |
| DEG-set | **Use pair-mode's native top-500-FC + cluster-marker DEG construction**, unmodified. Pair-mode's gene-universe behavior is part of what's being benchmarked. |
| Comparison structure | **Stream per-(sender, receiver, contrast) concordance**; no full path-level dataframe join. |

---

## Background

- **Legacy driver** (`incytr_commandline.R`): one Rscript invocation = one pair-mode comparison `cond1` vs `cond2`. Internally sweeps `groups² ` sender×receiver pairs (line 241–265). Produces one CSV per condition-pair with all pair rows concatenated.
- **Live factorial** (`alz/integration/factorial.R`): factorial OLS with disease × timepoint interactions, 9 contrasts derived from one fit. Outputs `per_cluster/{pr,ps,py}/<cluster>.parquet` and `pair_metadata.parquet`.
- **Levy-19 spine**: 19 named clusters defined under `alz/decomposition/`. Live pipeline already projects bulk phospho/protein onto this spine via `build_celltype_decomposition.py`. snRNA pseudobulk is keyed on the same 19 clusters.

---

## Phase 0 — Yuyu kldata (blocks everything else, fixes a live-pipeline bug)

**Finding**: `alz/integration/export_factorial_inputs.py:58` hardcodes
`KLDATA_SOURCE_REL = data/datasets/5xFAD/kinase/kldata_pspy.csv`. Every
factorial result currently in `outputs/reports/incytr_factorial/` was scored
with the **5xFAD demo** kinase library, not Yuyu's. The legacy pair-mode
driver has the same defect via `shared/kldata_5xad_fallback.csv`.

The kinase library (`kldata.csv` schema: `gene`, `site_pos`, `motif.geneName`,
plus mouse symbols) is the substrate→kinase motif map used by
`Incytr::IPMcontextNetINF` (`incytr/R/kinases.R:158–160`) to score the kinase
arm of PDS. The substrate/site list **must** come from sites actually
phosphoprofiled in the study cohort, not from a different mouse model. Using
5xFAD kldata systematically biases PDS toward kinases active in 5xFAD's
phospho-landscape and silently downweights kinases active in Yuyu's.

### 0a. Generate Yuyu kldata — `bench/build_yuyu_kldata.py`
- Source motif tables (already in repo):
  - `data/incytr/v2_46clusters/provenance/limma_corrected_yuyu.tsv` (IMAC, pS/pT)
  - `data/incytr/v2_46clusters/provenance/limma_corrected_pY_yuyu.tsv` (pY)
- Reference notebook: `data/incytr/shared/kinase_library.ipynb` (the procedure exists; just hasn't been executed).
- Run kinase-library 1.7.0 (already in `pixi.toml`) over each site to predict the top-N kinases per substrate motif.
- Apply `homologene::human2mouse()` (or the Python `mygene` equivalent already used elsewhere in this repo) to map kinase GENE_NAME → mouse symbols.
- Write `data/datasets/song/kinase/kldata_pspy.csv` with the upstream-required columns: `gene`, `site_pos`, `motif.geneName`, `Type`, mouse-symbol columns.
- Stamp `data/datasets/song/kinase/PROVENANCE.json` with kinase-library version, source motif file hashes, generation date, and which contrasts/animals contributed.

### 0b. Re-point the live factorial integration
- Edit `alz/integration/export_factorial_inputs.py:58–59`:
  ```python
  KLDATA_SOURCE_REL = os.path.join(
      "data", "datasets", "song", "kinase", "kldata_pspy.csv"
  )
  ```
- Update the docstring comment at lines 53–57 to remove the "not study-specific" claim — it **is** study-specific, that was the bug.
- Update the description string at lines 507–510 to point at Song.

### 0c. Re-run the live factorial integration
- `pixi run install-incytr && pixi run export-factorial-inputs && pixi run incytr-factorial`
- Outputs at `outputs/reports/incytr_factorial/{per_cluster,receiver_cache,pair_metadata.parquet}` will change. Rebuild the unified viewer (`pixi run viewer`) so its `incytr_pathways` tab is consistent with the new PDS values.
- **Snapshot pre-fix outputs** (move existing `incytr_factorial/` → `incytr_factorial_5xfad_kldata/` and delete after Phase 0d sign-off) so we can quantify the magnitude of the bug.

### 0d. Diff the old vs new factorial PDS
- Quick QC: per-(sender, receiver, contrast) Spearman ρ between `incytr_factorial_5xfad_kldata/` and the new `incytr_factorial/`. If ρ ≈ 1 across the board, the kldata source has minor effect (unlikely — the substrate sets are different). If ρ is low, document the magnitude in `docs/incytr_kldata_correction_note.md` so downstream consumers know to discard pre-fix interpretations.

### 0e. Pair-mode benchmark consumes the same kldata
- Symlink `bench/incytr_pair_19/incytr input/kldata.csv` → `data/datasets/song/kinase/kldata_pspy.csv`.
- Patch the legacy driver locally (`bench/incytr_pair_19/incytr_commandline_local.R`) to read mouse-format kldata directly (skip `homologene::human2mouse()` since the file is already mouse-mapped).

**Phase 0 output**: corrected live factorial outputs + a Yuyu kldata file shared between live and benchmark + a short correction note.

---

## Phase 1 — Map Levy-19 onto pair-mode driver inputs

The legacy driver reads four artifacts from `incytr input/`:

1. `incytr_obj.rds` — Seurat object with `Type` metadata (cluster) + `condition` metadata (`<sex>_<age>_<genotype>` tokens).
2. `pr_yuyu_deconvoluted.csv` — per-(gene, cluster) protein medians, wide on `<cluster>_pr` columns, repeated per sample.
3. `py_yuyu_deconvoluted.csv` — same shape, phospho-pY.
4. `ps_yuyu_deconvoluted.csv` — same shape, phospho-pS/pT (IMAC).
5. `input_gene_list.csv` — DEGs per cluster, columns `gene` and `cluster`.

### 1a. Build the 19-cluster Seurat object — `bench/build_levy19_seurat.R`
- Source: same `170_gex_celltypes_00.h5ad` used by `snrna_integration.py` for pseudobulk; filter to **14 male animals** per `snrna_sample_manifest.csv`.
- Cluster assignment: roll up Allen Cell Type Mapper `class_name` → Levy-19 via the same manifest the live pipeline uses (`alz/decomposition/load_deconvoluted.py` or the Levy-19 mapping under `outputs/reports/decomposition/levy19/`).
- Add `Type` (19-cluster label, sanitized: replace `/` → `-`, ` ` → `_` to match driver gsub at line 253).
- Add `condition` = `ma_<age>_<geno>` matching the live pipeline's condition tokenization.
- Save as `bench/incytr_pair_19/incytr input/incytr_obj.rds`.

### 1b. Project live Levy-19 decomposition into legacy CSV shape — `bench/export_pair_inputs.py`
- Source parquets:
  - `outputs/reports/decomposition/levy19/protein_per_cluster.parquet` → `pr_yuyu_deconvoluted.csv`
  - `outputs/reports/decomposition/levy19/phospho_per_cluster.parquet` → `ps_yuyu_deconvoluted.csv`
  - `outputs/reports/decomposition/levy19/phospho_per_cluster_pY.parquet` → `py_yuyu_deconvoluted.csv`
- Filter to males-only (33 male animals, post-outlier-exclusion via `sample_exclusions.csv`).
- Pivot to legacy wide shape: rows = (gene, sample_id), columns = `<cluster>_pr` / `_ps` / `_py` for each of 19 sanitized cluster labels.
- Stamp a `provenance.json` with source parquet hashes + sample list so the comparison is reproducible.

### 1c. Build `input_gene_list.csv` on the 19 spine — `bench/build_input_gene_list.R`
- Replicate `provenance/run_input_gene_list.R` logic against the 19-cluster Seurat from 1a (top-N DEGs per cluster vs the rest, Wilcoxon, FDR < 0.05).
- Output columns `gene`, `cluster` matching the 19 sanitized `Type` labels.

---

## Phase 2 — Run 9 contrasts in pair mode

Driver wrapper: `bench/run_pair_mode_sweep.sh`. Sets `nboot=100`, then:

```fish
cd bench/incytr_pair_19
for geno in AppP Ttau ApTt
  for age in 2mo 4mo 6mo
    Rscript ../../data/incytr/incytr_commandline.R \
      "ma_${age}_${geno}" "ma_${age}_WTyp" \
      "incytr input/input_gene_list.csv"
  end
end
```

→ 9 CSVs in `bench/incytr_pair_19/output/ma_<age>_<geno>_ma_<age>_WTyp_incytr_output.csv`,
each with 361 sender×receiver pairs ≈ tens of thousands of path rows.

**Compute estimate**: legacy driver at `nboot=100` over 361 pairs is ~hours per contrast on a single node — 9 contrasts ≈ overnight. Worth a smoke run at `nboot=2` end-to-end first (Phase 2a), then a sweep at `nboot=100` (Phase 2b).

---

## Phase 3 — Concordance analysis vs factorial (streaming, no full join)

Output: `bench/incytr_pair_19/comparison/concordance_summary.parquet`
(one row per (sender, receiver, contrast)) + per-contrast plots + report at
`docs/incytr_pair_mode_benchmark_report.md`.

### 3a. Align contrasts (per-file, lazy)
- Tag each of the 9 pair-mode CSVs with `disease` ∈ {App, Tau, ApTt} and `timepoint` ∈ {2mo, 4mo, 6mo} from the filename.
- Factorial source: `outputs/reports/incytr_factorial/receiver_cache/receiver=<sanitized>/part-<sender>.parquet` (342 pairs).
- Pair-mode includes 19 self-pairs (361 total). Inner join semantics drop them naturally; persist them to `comparison/self_pair_pdss.parquet` first so they're recoverable for autocrine analysis.

### 3b. Streaming concordance (per (sender, receiver, contrast) slice)
No full path-level join. For each of the 9 × 342 = 3 078 (contrast × non-self pair) cells:
1. Load the pair-mode slice from its CSV (filter rows where `Sender == s AND Receiver == r`).
2. Load the matching factorial parquet shard (already partitioned by receiver; filter to sender + contrast columns).
3. Inner-join on `ID_1` **in memory for that one slice** (typically ≤ a few thousand rows).
4. Compute and emit one summary row:
   - `n_pair_only`, `n_factorial_only`, `n_both` (path-coverage breakdown — captures the DEG-set divergence, kept as-is per intake decision).
   - Spearman ρ on PDS over `n_both`.
   - Sign-agreement fraction over `n_both`.
   - Jaccard of top-50 |PDS| paths between the two unioned sets.
5. Append summary row to a parquet writer; discard the slice.

This stays well under a GB of RAM regardless of pair-mode CSV size and makes the comparison rerunnable without rebuilding a giant intermediate table.

### 3c. Reporting
- 9-panel facet plot: per-contrast distribution of (Spearman ρ across 342 pairs).
- Heatmap: 19 × 19 sender × receiver, cell color = ρ at a chosen contrast (with a contrast-picker grid).
- Coverage breakdown table: median `n_both`, `n_pair_only`, `n_factorial_only` per contrast.
- The 19-row self-pair sidecar (PDS distribution per autocrine pair, per contrast) gets one paragraph + a small table in the report — flagged as informational since factorial has no counterpart.

---

## Phase 4 — Wire into existing viewer (optional, gated on Phase 3 results)

If concordance is high (Spearman ≥ 0.7 per contrast), no follow-up needed —
benchmark serves as a validation footnote.

If concordance is low (< 0.5) in any contrast, **stop and write up the
divergence** before touching the viewer. The pair-mode method is closed-path;
divergences are informational, not actionable on the live pipeline.

---

## File layout

```
bench/incytr_pair_19/
├── build_levy19_seurat.R           # Phase 1a
├── build_input_gene_list.R         # Phase 1c
├── export_pair_inputs.py           # Phase 1b
├── run_pair_mode_sweep.sh          # Phase 2 driver
├── incytr input/
│   ├── incytr_obj.rds              # from 1a
│   ├── pr_yuyu_deconvoluted.csv    # from 1b
│   ├── ps_yuyu_deconvoluted.csv    # from 1b
│   ├── py_yuyu_deconvoluted.csv    # from 1b
│   ├── input_gene_list.csv         # from 1c
│   ├── provenance.json             # source hashes
│   ├── Database -> ../../data/incytr/Database
│   └── source -> ../../data/incytr/source
├── output/                         # 9 CSVs from Phase 2
└── comparison/
    ├── factorial_vs_pair.parquet
    └── concordance_plots/          # 9-panel scatter, etc.

docs/incytr_pair_mode_benchmark_plan.md   (this file)
docs/incytr_pair_mode_benchmark_report.md (Phase 3 deliverable)
```

`bench/` is a new top-level directory for one-off methodological comparisons.
It is **not** wired into `pixi run live` or the runners under `alz/runners/`.

---

## Resolved risks (from intake)

1. **kldata** — promoted to Phase 0. Yuyu kldata regenerated; live factorial re-pointed and re-run; 5xFAD fallback retired.
2. **Self-pairs** — calculated (361 pairs run), dropped via inner-join semantics at comparison time, persisted as `self_pair_pdss.parquet` sidecar.
3. **DEG-set divergence** — kept as part of what's being benchmarked. Surfaces in the per-cell `n_pair_only` / `n_factorial_only` / `n_both` breakdown.

## Open / unresolved

4. **N=1 at 4mo and 6mo (snRNA)** — caveat the report; no method change.
5. **Bootstrap cost at `nboot=100`** — measure in Phase 2a smoke (single-contrast end-to-end at `nboot=2`, then time one contrast at `nboot=100`). If wall time > 2 hours per contrast, parallelize the 9 invocations via `parallel -j N` or `xargs -P N`.

---

## What I'm not doing (explicit non-goals)

- Not rerunning v2's `provenance/{protein-,py-,ms-}by-cell-type.ipynb` notebooks. Reusing live Levy-19 decomposition outputs instead, per the intake decision.
- Not building a 46-cluster pair-mode pipeline. Going straight to 19.
- Not running the full 16-comparison legacy sweep ({4mo,6mo}_geno vs 2mo_geno per genotype × sex). Only the 9 disease-vs-WT-at-same-timepoint contrasts that mirror the factorial.
- Not modifying the live viewer logic. **Will** modify `alz/integration/export_factorial_inputs.py` (Phase 0b) and re-run the live factorial (Phase 0c) — the kldata correction is a scope expansion forced by the finding, not an extension of the pair-mode benchmark itself.
- Not promoting pair-mode results into `kinase_hypothesis_table.csv` or any downstream table.
- Not modifying pair-mode's DEG-set construction. It is run unmodified; its gene-universe behavior is part of the benchmark.
