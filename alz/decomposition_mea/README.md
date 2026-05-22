# `alz/decomposition_mea/` — per-cluster proportional decomposition track

**Status:** mid-pivot. Renamed from `alz/deconvolution/` in Step 8 of
the canonical implementation plan. The math, spine, and CLI are being
rewritten in Steps 9–12.

**Authoritative plan:** [`docs/incytr_deconvolution_pivot.md`](../../docs/incytr_deconvolution_pivot.md).

**What this directory will do:**
1. Compute snRNA-derived per-gene cell-type proportions `f_c(G, A)`
   on the Levy-t5 31-cluster spine (Stage 5).
2. Project bulk phospho + protein onto those proportions in linear
   space: `P_c = f_c × P` (Stage 6, `build_celltype_decomposition.py`).
3. Run per-cluster MEA on raw phospho (Stage 7, `enrich_celltype.py`).

**What this directory does NOT do:** statistical (inverse-problem)
deconvolution. That path is closed; see
`archive/deconvolution/docs/deconvolution_infeasibility.md`.

Files marked "audit pending" below may be retired or merged into shared
helpers in later steps.

| File | Status |
|---|---|
| `build_celltype_decomposition.py` | Awaiting Step 10 rewrite (linear-space, spine-parametrized) |
| `enrich_celltype.py` | Awaiting Step 11 rewrite (shared OLS+GSEA with `kinase_enrich.py`) |
| `per_animal_extension.py` | Salvage in Step 9 |
| `factorial_ols.py` | Audit pending (Step 11) |
| `cohort_concordance.py`, `cohort_concordance_audit.py` | Audit pending |
| `variance_audit.py`, `confidence.py`, `snrna_concordance.py` | Audit pending |
| `load_deconvoluted.py`, `paths.py` | Audit pending |
| `_archive/`, `_results/` | Historical |
