# `alz/decomposition_mea/` audit + reorg — 2026-05-29

Goal: audit the package for dead code, classify every file, fix the stale
README, and decide whether a structural split is warranted.

---

## 1. Liveness / dependency map

### 1a. Live — wired into runners and/or consumed by other packages

| File | Entry point | External consumers | Notes |
|---|---|---|---|
| `build_celltype_decomposition.py` | `__main__`; `rerun_decomposition_chain.sh:22`, `run_all.sh:117`, `run_pair_mode_pipeline.sh:124` | `verify_decomposition.py` re-imports `_bulk_to_long`, `_load_sample_mapping` (broken import — see §2); `alz/integration/` reads the output parquets | Stage 6: forward projection `P_c = f_c × bulk` |
| `enrich_celltype.py` | `-m alz.decomposition_mea.enrich_celltype`; same three runners | `alz/build_unified_viewer.py:165,786` reads `mea_per_cluster.parquet`; `alz/integration/` reads the output | Stage 7: per-cluster OLS + MEA |
| `build_per_animal_site_ols.py` | `__main__`; `rerun_decomposition_chain.sh:30`, `run_all.sh:120` | `alz/build_unified_viewer.py:1568,2972`; `alz/viewer/paths.py:29` reads `per_animal/site_level_ols.parquet` | Publisher: unions st+py OLS, renames cluster→cell_type |
| `verify_decomposition.py` | `__main__`; `rerun_decomposition_chain.sh:33`, `run_all.sh:121`, `run_pair_mode_pipeline.sh:191` | Invoked as hardfail gate in all three main runners | Verification harness (4 contracts) |
| `__init__.py` | package init | imported transitively whenever the package is imported | Contains stale doc reference (see §2) |

### 1b. Orphan / dead — no live runner, no external import

| File | What it is | Why dead |
|---|---|---|
| `paths.py` | Config/path constants for the **WMB-class statistical deconvolution** (the closed path) | All live code (build_celltype_decomposition.py, enrich_celltype.py, build_per_animal_site_ols.py) uses `alz.shared.config` directly; seven dead modules all import `paths`; zero imports from outside the dead cluster |
| `load_deconvoluted.py` | Stage 1 loader for `ps/py/pr_wmb_decomposition.csv` | Imports `paths`; its data targets (`wmb_decomposition/`) are on disk but the pipeline that produces them is closed; never imported from outside the dead cluster |
| `factorial_ols.py` | Stage 2 males-only factorial OLS on WMB-class decomposition | Imports `paths` + `load_deconvoluted`; `per_animal_extension` imports it, which is itself dead |
| `per_animal_extension.py` | Per-animal re-estimation on the WMB-class decomposition | Imports `paths`, `load_deconvoluted`, `factorial_ols`; no runner, no external import |
| `cohort_concordance.py` | Stratum-level binomial sign-concordance check for WMB-class MEA | Imports `paths`; only consumer is `cohort_concordance_audit.py` (itself dead) |
| `snrna_concordance.py` | Stage 4: per-row snRNA kinase-gene LFC join for WMB MEA table | Imports `paths`; no runner, no external import |
| `confidence.py` | Stage 5: attaches evidence columns (n_cells_min, cohort_concordant, expressed) to WMB MEA table | Imports `paths`; no runner, no external import |
| `cohort_concordance_audit.py` | Calibration audit for COHORT_FDR_THRESH / EXPR_PRESENCE_FLOOR | Imports `paths` + `cohort_concordance`; no runner, no external import; output dir `outputs/reports/deconvolution/per_animal/` has no calibration doc |

**Dead-cluster summary:** `paths` → `load_deconvoluted` → `factorial_ols` → `per_animal_extension` (all four form a closed dependency chain). `cohort_concordance` → `cohort_concordance_audit` (standalone closed chain). `snrna_concordance` and `confidence` are isolated orphans, each importing only `paths`.

### 1c. Imports FROM outside the package (what decomposition_mea consumes)

- `alz.shared.config` — all five live files
- `alz.bulk_mea.enrich` — `enrich_celltype.py:39–46` (imports `CONTRAST_COEFS`, `_bh_fdr`, `_build_design_matrix`, `_filter_samples`, `_run_mea`, `_run_ols_all_sites`)
- `alz.integration.config_integration` (as `icfg`) — `verify_decomposition.py:29` (imports `load_cluster_spine`)

### 1d. Imports OF decomposition_mea from outside

- `alz/integration/config_integration.py:49–50` — reads output parquet paths (file-level, not code import)
- `alz/integration/build_normalized_substrate.py:19–21` — doc + file path (no code import)
- `alz/integration/build_omics_trace.py:92–94` — file path constants (no code import)
- `alz/build_unified_viewer.py:165,786,1568,2972` — file paths for `mea_per_cluster.parquet`, `per_animal/site_level_ols.parquet`
- `alz/viewer/paths.py:29` — `per_animal/site_level_ols.parquet`

No external module imports Python symbols from `decomposition_mea` except `verify_decomposition.py`'s own broken self-import (see §2).

---

## 2. Dead code / defect findings

### Finding 1 — Broken import in `verify_decomposition.py`

`verify_decomposition.py:24`:
```python
from decomposition_mea.build_celltype_decomposition import (
    _bulk_to_long, _load_sample_mapping,
)
```
This is a bare (non-package-qualified) import that relies on `alz/` being on `sys.path`. The live file uses `alz.` prefix everywhere else. More importantly, both `_bulk_to_long` and `_load_sample_mapping` are private helpers that should be extracted to a shared helper or inlined in `verify_decomposition.py` itself — importing private functions from a sibling module at runtime couples the two scripts unnecessarily. **Classification: fix.**

### Finding 2 — Stale `--all` flag referenced in `CLAUDE.md`

`CLAUDE.md` says: `alz/decomposition_mea/verify_decomposition.py --all`. The flag does not exist; the script has `--checks` and `--spine`. **Classification: fix CLAUDE.md.**

### Finding 3 — Dead `paths.py` carries active-looking constants

`paths.py` exports `CONTRASTS = config.CONTRAST_COEFS` (line 78), `SNRNA_FDR_HIGH`, `SNRNA_LFC_FLAT`, `COHORT_FDR_THRESH`, `EXPR_PRESENCE_FLOOR` — all consumed only by the dead cluster. The live files get contrast definitions directly from `alz.shared.config`. **Classification: archive with the dead cluster.**

### Finding 4 — `paths.py` points to the closed WMB-class deconvolution

`paths.py:38–49` hard-codes `outputs/reports/deconvolution/wmb_decomposition/` paths that correspond to the closed statistical deconvolution path. The `wmb_decomposition/` subdirectory exists on disk with the legacy CSVs but the code that produces them is archived. Anti-shim: these constants should go with the dead code, not remain as a latent re-entry point.

### Finding 5 — Stale spine path `data/incytr_frozen/v2_46clusters/spines/`

`paths.py:19`: `SPINES_ROOT` points to `data/incytr_frozen/v2_46clusters/spines/`. That directory exists and contains only `levy_t5/` (empty). The live spine resolution uses `alz.integration.config_integration.load_cluster_spine()` and `alz.shared.config.CLUSTER_SPINE_NAME`. Redundant and stale.

### Finding 6 — Stale README

`alz/decomposition_mea/README.md` says "Status: mid-pivot" and refers to `docs/incytr_deconvolution_pivot.md`, which does not exist anywhere in the repo. The table lists files as "Awaiting Step N rewrite" — that work is long complete. **Classification: replace entirely** (proposed text in §5).

### Finding 7 — `__init__.py` stale docstring

`__init__.py:8` references `docs/incytr_deconvolution_pivot.md` (does not exist). **Classification: update in place.**

### Finding 8 — `enrich_celltype.py` contains a local OLS implementation (`_run_ols_pinv`)

`enrich_celltype.py:123–163` implements `_run_ols_pinv` (rank-tolerant pseudoinverse OLS). This is separate from the `_run_ols_all_sites` it imports from `alz.bulk_mea.enrich`. The rationale — handling rank-deficient per-cluster designs — is documented in the function docstring and is architecturally justified. Not dead; but worth noting for the Kedro factoring pass (`kedro_argo_reintroduction_2026-05-26.md:64` already flags `build_celltype_decomposition` as "needs factoring"). **Classification: keep, note for future factoring.**

### Finding 9 — Anti-shim: no legacy flags found in live files

`build_celltype_decomposition.py`, `enrich_celltype.py`, `build_per_animal_site_ols.py`, and `verify_decomposition.py` contain no `if name == "old"` branches, fallback paths, or legacy flags. Clean.

---

## 3. Reorg recommendation

**No structural split warranted.** The package is already cohesive: five live files all serving one pipeline stage (Stage 6–7 forward projection + verification), plus a cluster of eight dead files that all belong to the closed WMB-class statistical deconvolution path.

The right action is **archive the dead cluster**, not a package split:

```
alz/decomposition_mea/paths.py               -> archive/decomposition_mea_wmb_2026-05-29/paths.py
alz/decomposition_mea/load_deconvoluted.py   -> archive/decomposition_mea_wmb_2026-05-29/load_deconvoluted.py
alz/decomposition_mea/factorial_ols.py       -> archive/decomposition_mea_wmb_2026-05-29/factorial_ols.py
alz/decomposition_mea/per_animal_extension.py-> archive/decomposition_mea_wmb_2026-05-29/per_animal_extension.py
alz/decomposition_mea/cohort_concordance.py  -> archive/decomposition_mea_wmb_2026-05-29/cohort_concordance.py
alz/decomposition_mea/cohort_concordance_audit.py -> archive/decomposition_mea_wmb_2026-05-29/cohort_concordance_audit.py
alz/decomposition_mea/snrna_concordance.py   -> archive/decomposition_mea_wmb_2026-05-29/snrna_concordance.py
alz/decomposition_mea/confidence.py         -> archive/decomposition_mea_wmb_2026-05-29/confidence.py
```

No external file references any of these by import or by path. Blast radius: zero outside the package.

**What stays:**

```
alz/decomposition_mea/
  __init__.py                        (update docstring — remove stale pivot doc reference)
  build_celltype_decomposition.py    (live)
  enrich_celltype.py                 (live)
  build_per_animal_site_ols.py       (live)
  verify_decomposition.py            (live — fix import bug, Finding 1)
  README.md                          (replace per §5)
```

**Other fixups in the same pass:**

1. `verify_decomposition.py:24` — change bare `from decomposition_mea.build_celltype_decomposition import` to `from alz.decomposition_mea.build_celltype_decomposition import` (or inline the two helpers).
2. `CLAUDE.md` — replace `verify_decomposition.py --all` with `verify_decomposition.py` (no `--all` flag exists; `--checks` is the optional subset argument).
3. `__init__.py:8` — remove the `docs/incytr_deconvolution_pivot.md` pointer (doc does not exist).

---

## 4. Decisions required

**D1 (archive vs delete):** The eight dead files and their data (`wmb_decomposition/` CSVs at ~170 MB) correspond to the closed WMB-class deconvolution. Is the historical record useful enough to keep in `archive/`, or should the code (not the data) be deleted? The `archive/deconvolution/` tree already exists and holds the infeasibility analysis. If delete is acceptable, blast radius is zero.

**D2 (verify_decomposition.py import):** The two imported private helpers (`_bulk_to_long`, `_load_sample_mapping`) are small. Preferred fix: inline them into `verify_decomposition.py` to remove the cross-script coupling, or extract them to a `_helpers.py` in the package if they'll be reused. Decision is stylistic — either approach passes.

---

## 5. Proposed README for `alz/decomposition_mea/`

```markdown
# `alz/decomposition_mea/` — per-cluster proportional decomposition + MEA

Stage 6–7 of the main pipeline. Converts bulk phospho/proteome into per-cluster
substrates via a forward projection onto snRNA-derived proportions, then runs
per-cluster factorial OLS + MEA.

Does NOT perform statistical (inverse-problem) deconvolution — that path is
closed; see `archive/deconvolution/docs/deconvolution_infeasibility.md`.

## File inventory

| File | Role | Entry point |
|---|---|---|
| `build_celltype_decomposition.py` | Stage 6: project bulk phospho (pS/pT, pY) and protein onto `f_c` per-cell-rate weights; writes `protein_per_cluster.parquet`, `phospho_per_cluster{,_pY}.parquet`, `decomposition_audit.json` | `python alz/decomposition_mea/build_celltype_decomposition.py --spine levy_t5 --track both` |
| `enrich_celltype.py` | Stage 7: per-cluster factorial OLS + GSEA MEA on the projected phospho cube; writes `mea_per_cluster{,_pY}.parquet`, `site_level_ols_per_cluster{,_pY}.parquet`, and CSV sidecars | `python -m alz.decomposition_mea.enrich_celltype --spine levy_t5 --track st` |
| `build_per_animal_site_ols.py` | Publisher: unions st + py OLS into `per_animal/site_level_ols.parquet`; renames `cluster` → `cell_type`; consumed by the unified viewer | `python alz/decomposition_mea/build_per_animal_site_ols.py --spine levy_t5` |
| `verify_decomposition.py` | Verification harness (4 contracts: mass identity, coverage, per-cluster vs bulk MEA, Incytr pair count); exits non-zero on failure | `python alz/decomposition_mea/verify_decomposition.py --spine levy_t5` |

## Key invariants

**Mass identity** — `Σ_c [P_c × (N_c / N_total)] ≈ bulk`, not `Σ_c P_c = bulk`.
The `f_c` weight is a per-cell-rate (`share_c × N_total / N_c`), so literal
summation overshoots. Verified by `check_mass_identity` (threshold `max_rel_err < 1e-6`).

**Sign convention** — `+` = up in disease, matching `bulk_mea` NES/β and Incytr PDS/sclog2FC.

**Spine** — `levy_t5` (31 clusters). Spine definition lives in
`alz/integration/config_integration.py` (`load_cluster_spine()`).

**Track vocabulary** — `st` = IMAC pS/pT (suffix `""`), `py` = pY (suffix `"_pY"`).
pY requires `raw_phospho_normalized_pY.csv` from Stage 1; tolerates missing pY gracefully.

## Upstream prerequisites

1. `alz/reference/snrna_proportions.py --run --spine levy_t5` — produces `proportions.parquet`
2. `alz/bulk_mea/normalize.py` — produces `total_proteome_normalized.csv` and `raw_phospho_normalized*.csv`

## Downstream consumers

- `alz/integration/build_normalized_substrate.py` — reads `protein_per_cluster.parquet`, `phospho_per_cluster*.parquet`
- `alz/integration/build_omics_trace.py` — reads same
- `alz/build_unified_viewer.py` — reads `mea_per_cluster.parquet`, `per_animal/site_level_ols.parquet`
- `alz/viewer/paths.py` — path constant for `per_animal/site_level_ols.parquet`

## Runner scripts

| Script | Covers |
|---|---|
| `alz/runners/main/rerun_decomposition_chain.sh` | Full 7-step chain: pseudobulk → proportions → Stage 6 → Stage 7 (st+py) → publisher → verify |
| `alz/runners/main/run_all.sh` (steps D-decomp … D-verify) | Same chain inside the full pipeline |
| `alz/runners/main/run_pair_mode_pipeline.sh` | Runs Stage 6–7 as part of the pair-mode pipeline |
```
