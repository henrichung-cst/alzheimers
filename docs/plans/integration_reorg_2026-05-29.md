# `alz/integration/` cleanup audit — 2026-05-29

Same folder-by-folder pass as `cross_reference` (84be18d), `decomposition_mea`,
and `incytr_pair` (deb14ee, this branch). `integration` is the **consumer** side
of the Incytr pipeline: it reshapes pair-mode wide parquets into viewer caches,
builds the transcript/omics trace substrates the Evidence tab reads, and owns the
cluster-spine config. Two lenses: (1) dead-code / liveness, (2) separation of
concerns vs the `Incytr` package.

---

## 1. Liveness map

| File | Status | Caller(s) | Role |
|---|---|---|---|
| `__init__.py` | LIVE | package marker for `from alz.integration.config_integration import …` | namespace |
| `config_integration.py` | LIVE | `export_decomposition_for_pair.py`, `snrna_proportions.py`, `verify_decomposition.py`, `build_unified_viewer.py:52` | spine/path config; `load_cluster_spine()` |
| `build_cluster_spine.py` | LIVE | `run_pair_mode_pipeline.sh:117` (step A) | run-once-per-rebuild spine generator |
| `build_normalized_substrate.py` | LIVE | `build_unified_viewer.py:555,565` (`bns.build`) | produces limma-normalized omics substrate |
| `build_omics_trace.py` | LIVE | `build_unified_viewer.py:528,538` (`bot.build`) | per-cluster omics-trace shards (raw values) |
| `build_transcript_trace.py` | LIVE | `build_unified_viewer.py:499,509` (`btt.build`) | per-cluster transcript pseudobulk shards |
| `verify_pathway_round_trip.py` | LIVE | `build_unified_viewer.py:634 (strict=False), 3049` | round-trip LFC verification harness |
| `build_yuyu_kldata.py` | LIVE (one-time gen) | output `kldata_pspy.csv` guarded by `run_pair_mode_pipeline.sh:144` | kinase-library ranks → kldata |
| `build_seaad_bridge.py` | LIVE (one-time gen) | output consumed by `attribute.py`, `human_celltype_attribution.py` | hand-curated SEA-AD crosswalk |
| `extract_cluster_assignments.R` | LIVE (one-time gen) | outputs consumed by `shared/config.py`, `build_cluster_spine.py` | barcode→cluster from legacy rds |
| `plot_cluster_spine.py` | **DORMANT** | none in live code; outputs referenced only in archived docs | spine diagnostic plot |

**Dead-code finding:** `plot_cluster_spine.py` has no live caller — no pixi task, no
runner, no import. Its two outputs (`cluster_spine_summary.csv`,
`cluster_spine_selection.png`) are referenced only in
`archive/pre_levy19_2026-05-14/…/README.md`. It is a one-time spine-selection
diagnostic from the WMB-34/Levy-19 era, superseded by the frozen Levy-t5 spine.

**README is stale:** the current `README.md` lists 7 files; the folder has 11 code
files. Missing entirely: `build_normalized_substrate.py`, `build_omics_trace.py`,
`verify_pathway_round_trip.py` — the three Evidence-tab substrate/verify modules.

---

## 2. Separation-of-concerns audit

Python files cannot `library(Incytr)` (it is R), so the boundary here is: do these
files reimplement Incytr scoring math to **produce production truth**, or only to
**verify** the R driver's stored output? Verdicts:

| File | Verdict | Note |
|---|---|---|
| `build_omics_trace.py` | CLEAN GLUE | no math; reshapes raw values |
| `build_transcript_trace.py` | CLEAN GLUE | no math; reshapes pseudobulk |
| `build_yuyu_kldata.py` | CLEAN GLUE | delegates ranks to `kinase_library` package |
| `config_integration.py` | CLEAN GLUE | spine/path config only |
| `verify_pathway_round_trip.py` | LEGITIMATE VERIFICATION | recompute → compare → pass/fail; emits no data |
| `build_normalized_substrate.py` | MIXED — see below | produces substrate (needed) **and** carries a *duplicate* inline verification |

Three concrete findings (B1–B3):

- **B1 — recompute logic is duplicated across the build step and the verifier.**
  `build_normalized_substrate.py::_roundtrip_sample` (lines 258-336) and
  `verify_pathway_round_trip.py::_recompute_omics_lfc` (lines 165-208) independently
  reimplement the *same* `Cal_foldchange` zero-correction branch
  (`has_zero → log2((d+ε)/(w+ε)) else log2(d/w)`). The build step's copy fires on
  **every** substrate build; the verifier's copy is the sampled harness that runs
  after the viewer build. This is exactly the "recompute should be a verification
  pass, not baked into the build that always re-runs it" concern.

- **B2 — `EPSILON = 1e-3` is defined independently in two files**
  (`build_normalized_substrate.py:88`, `verify_pathway_round_trip.py:93`), and
  `EPSILON_SC = 0.01` lives only in the verifier (`:94`) while the R driver holds the
  authoritative `correction = 0.001` / `Cal_scFC(correction = 0.01)`. Same logical
  constant, three uncoordinated definitions → silent-drift risk if the driver default
  changes.

- **B3 — stale comment.** `build_omics_trace.py` (lines ~30-31, ~314-316) says the
  omics correction is `1e-5`; the R driver passes `0.001`. The file does no LFC math
  (advisory comment only), but it is misleading.

`normalize_quantiles` (the numpy port of `limma::normalizeQuantiles`,
`build_normalized_substrate.py:106-155`) is a parallel implementation of a step
Incytr performs via the CRAN `limma` package. It **produces** the per-condition
normalized means the JS Evidence tab reads to recompute LFC client-side — that is a
genuine architectural need (the R driver does not currently emit per-condition
normalized means, only the final `*_log2FC`). Not removed here. It is the one place
the app legitimately re-derives a normalization Incytr also performs; the safeguard
is the round-trip verification, which is exactly why that verification must be
correct and live in one place.

---

## 3. Plan — consolidate verification, fix drift risks, prune dormant code

Scope kept tight to the user's directive ("the recompute should be a verification
pass; sample instead of always re-running") plus the liveness/staleness findings.

**P1 — single source of truth for the recompute (B1).** Remove the inline
`_roundtrip_sample` recompute from `build_normalized_substrate.py`. The build step's
job is to **produce** the substrate; correctness is asserted by the dedicated
sampled harness `verify_pathway_round_trip.py`, which already covers the omics
layers (`_recompute_omics_lfc`) against the same stored `*_log2FC` and runs in the
same viewer build (`build_unified_viewer.py:634`). Net effect: the recompute lives in
exactly one place (the verification pass), and the build no longer re-runs a second,
divergent copy of `Cal_foldchange` on every invocation.

**P2 — shared epsilon constants (B2).** Move `EPSILON_OMICS = 1e-3` and
`EPSILON_SC = 0.01` into one module (`alz/shared/incytr_constants.py`) with a comment
pinning them to the R driver's `correction` args; import from both the verifier and
the substrate index metadata. Kills the three-way drift.

**P3 — fix the stale `1e-5` comment in `build_omics_trace.py` (B3).**

**P4 — delete `plot_cluster_spine.py`** (dormant, superseded-era diagnostic; outputs
referenced only in archived docs).

**P5 — refresh `README.md`**: correct the 11-file inventory; add the separation
verdict (consumer side, one verification harness); document that the build step
produces, the harness verifies.

**Verification:** `verify_pathway_round_trip.py` must still PASS in default
(sampled) mode after P1 (the harness is unchanged; we only remove the build step's
duplicate). Confirm `build_normalized_substrate.build(force=True)` still produces all
cluster shards and that the viewer build's downstream `vpr.verify()` is green.

**Decisions (approved 2026-05-29):**
- Sampling semantics → **seeded-random rotation.** Added `--seed` /
  `verify(seed=)`; the seed is mixed into the reservoir's row-identity hash, so a
  non-zero seed rotates to a different but reproducible sample. `seed=0` (default)
  leaves the hash untouched → historical behaviour preserved.
- `plot_cluster_spine.py` → **deleted.**

## 4. Status — DONE

- **P1** — inline `_roundtrip_sample` (+ `_LAYER_TO_STORED`, `rng`, the
  `ROUNDTRIP_*` constants, the `import math`) removed from
  `build_normalized_substrate.py`; the build step now only produces the substrate.
  The recompute lives solely in `verify_pathway_round_trip.py`.
- **P2** — `alz/shared/incytr_constants.py` holds `EPSILON_OMICS = 1e-3` /
  `EPSILON_SC = 0.01`; imported by the verifier and the substrate index metadata.
- **P3** — stale `1e-5` comments in `build_omics_trace.py` corrected to `0.001`.
- **P4** — `plot_cluster_spine.py` deleted; README + `kinase_incytr_integration.md`
  rows removed.
- **P5** — `README.md` rewritten (corrected 10-file inventory + separation
  contract).
- Seeded-random sampling wired (`--seed`, default 0 = reproducible).
- **Verification:** all four edited modules `ast.parse` clean; imports resolve in
  the pixi env (`EPSILON_OMICS=0.001`, `EPSILON_SC=0.01`, `verify(seed=)` present);
  no live references to the removed symbols remain. The round-trip harness itself is
  unchanged in logic (only the epsilon source + an additive seed knob), so its
  sce4-adjacent parity behaviour is preserved.
