# Architecture refactor plan — config import-time I/O + viewer builder seam

Date: 2026-06-23. Addresses items #1 and #2 of the architecture review
(`/tmp/architecture-review-1782262371.html`). Write-then-approve: no source
edits until this plan is signed off. **Decisions below reflect a grilling pass
(2026-06-23) — see the "Decisions locked" recap at the end.**

Both items are pure structural moves — **zero behavior change is the
acceptance bar**. Every phase ends by regenerating the affected artifact and
proving *content-equality modulo volatile fields* (or import-success) against
the pre-refactor state. Raw byte-equality is **not** the bar (the payload
carries a wall-clock `generated_at` and gzip headers embed an mtime).

---

## Item 1 — `config.py`: kill import-time file I/O

### What's actually wrong (narrower than the review implied)

The review called `config.py` a "god module with import-time I/O." The
god-module *size* (728 lines) is real but is a navigability concern, not a
correctness/testability one. The **testability blocker** is precise: exactly
three module-level names do work at import:

| Line | Name | Work done at import | File tracked? |
|---|---|---|---|
| 115 | `ANALYSIS_MODE` | reads `conf/base/parameters.yml` (+ KEDRO_ENV overlay) | **yes** (committed) |
| 181 | `CLUSTER_SPINE` | reads `data/incytr_frozen/.../levy_t5/cluster_spine.csv` | **no — untracked** |
| 205 | `N_CELL_TYPES` | `len(CLUSTER_SPINE)` (transitively the CSV read) | — |

`cluster_spine.csv` is **present on this box but not tracked by git** (verified:
`git ls-files` returns nothing for it). So in a fresh checkout or CI,
`import alz.shared.config` raises `FileNotFoundError` before any test can run.
4 of the 5 files in `tests/` import config transitively, so the whole suite is
hostage to one untracked data artifact. That is the friction worth removing.

`WMB_REGION_SCOPE` (line 435) reads an **env var only** (no file) and validates
fail-fast. Leave it eager — it doesn't block import in a clean tree, and
fail-fast on a bad scope is a feature, not a bug. (Flagged so a reviewer
doesn't ask "why is this one different" — the answer is: no file dependency.)

### Why the fix is zero-churn and shim-free

Every one of the 56 importers uses `from alz.shared import config` followed by
attribute access (`config.CLUSTER_SPINE`) **inside functions** — verified:

- no `from ...config import CLUSTER_SPINE` (eager name binding) anywhere;
- no module-top-level `X = config.CLUSTER_SPINE` in any consumer;
- no `from config import *` anywhere (would defeat PEP 562).

That means **PEP 562 module `__getattr__`** is the exact-fit tool: lazily compute
+ cache `ANALYSIS_MODE` / `CLUSTER_SPINE` / `N_CELL_TYPES` on first attribute
access. The public access pattern `config.CLUSTER_SPINE` is unchanged for all
56 callers — **no importer edits, no re-export shim, no flag.** Import becomes
pure; the CSV is read only when the spine is actually used.

### Changes (all in `alz/shared/config.py`)

1. Delete the three eager assignments (lines 115, 181, 205). Keep
   `_load_analysis_mode()` and `_load_cluster_spine()` as the compute backends.
2. Add a memoized lazy layer:
   ```python
   _LAZY = {
       "ANALYSIS_MODE": _load_analysis_mode,
       "CLUSTER_SPINE": _load_cluster_spine,
       "N_CELL_TYPES":  lambda: len(_lazy("CLUSTER_SPINE")),
   }
   _LAZY_CACHE: dict[str, object] = {}
   def _lazy(name):
       if name not in _LAZY_CACHE:
           _LAZY_CACHE[name] = _LAZY[name]()
       return _LAZY_CACHE[name]
   def __getattr__(name):                 # PEP 562
       if name in _LAZY:
           return _lazy(name)
       raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
   ```
3. Fix the two **internal** references that currently read these as bare module
   globals (they will no longer exist in `__dict__`):
   - `provenance_stamp()` line 151: `"analysis_mode": ANALYSIS_MODE` →
     `"analysis_mode": _lazy("ANALYSIS_MODE")`.
   - `N_CELL_TYPES` consumers inside config — none other than its own
     definition; the comment references at 208/220 are prose, no change.
   (`wmb_region_keys()` reads `WMB_REGION_SCOPE`, which stays eager — no change.)
4. `config_integration.py` references only `main_config.CLUSTER_SPINE_NAME`
   (a pure string constant, line 138 — no I/O) as a default arg. No change
   needed; confirm nothing there touches `CLUSTER_SPINE`/`ANALYSIS_MODE` at
   module scope (verified: it doesn't).

### Memoization semantics

The three lazy names are computed once and **memoized** in a module-level cache
— matching today's read-once-at-import semantics. This is safe under the `dual`
runner because it invokes each mode as a **separate process** with
`export KEDRO_ENV=full_cohort` set *before* the track-2 processes launch
(verified in `run_dual_analysis.sh`), so memoization is per-process and strictly
*more* forgiving than today's import-time read (which also reads once, earlier).

### Mechanism: transparent lazy via PEP 562

Use a module-level `__getattr__` (PEP 562) so `config.CLUSTER_SPINE` stays the
access pattern for all 56 importers — **zero call-site churn, no re-export
shim**. Type-checking is `off` project-wide (`pyrightconfig.json`) and nothing
does `hasattr/getattr(config, …)`, so the usual "dynamic attribute is invisible
to tooling" cost does not apply here. Add `if TYPE_CHECKING:` annotations naming
the three so a human reading the module sees them declared.

### Verification (item 1)

- **Import-without-data proof (one-time, manual, during implementation):**
  temporarily `mv` `cluster_spine.csv` aside, run
  `python -c "import alz.shared.config"` → must succeed (today it raises).
  Restore the file. **Not committed as a test.**
- **Value-unchanged proof:** with the file present,
  `python -c "from alz.shared import config; print(config.CLUSTER_SPINE, config.N_CELL_TYPES, config.ANALYSIS_MODE)"`
  matches pre-refactor output exactly.
- Run the existing checks named in CLAUDE.md:
  `python alz/bulk_mea/summary.py` and
  `python alz/decomposition_mea/verify_decomposition.py`; plus `pytest tests/`.
- **No new committed regression test** — the manual `mv` check plus the existing
  suite is the agreed bar.

### Deliberately descoped: splitting config into `domain`/`paths` submodules

The review's "split the god module" sub-idea is **not** in this plan, on
purpose. The testability win comes entirely from the I/O fix above; the
remaining 728-line size is navigability-only. A real split is either
name-everywhere (touch all 56 importers) or a re-export module (`config` that
re-exports `config.domain.*`) — and the latter is exactly the dual-representation
shim the project's anti-shim rule forbids. If module size becomes a genuine
navigation problem later, it deserves its own name-everywhere epic, not a
half-measure bolted onto this change. Calling it out so the descope is a
decision, not an omission.

---

## Item 2 — viewer builder seam

### Correcting the review's framing first

The review proposed extracting `alz/viewer/cohorts/tcells.py` "following the
exact same adapter shape as `song.py`" so it flows through
`compose_viewer_slices`. **That part is wrong and is dropped.** The
`CohortViewerSlice` + `compose_viewer_slices` machinery exists for one reason:
to **merge multiple cohorts into a single payload** (song + mukesh + 5xFAD all
land in `outputs/reports/unified_viewer/`, sharing `incytr_pathways.by_context`,
unioned kinase names, merged capabilities/audit tables). The T-cell viewer is a
**standalone, single-cohort** deliverable: its own output dir
(`outputs/reports/tcell_viewer/`), its own paths module, its own template tab,
its own `validate()`, and it assembles inline in `build_tcell_payload()`. There
is nothing to merge. Forcing it through the compose contract would adopt an
abstraction whose entire purpose doesn't apply — the "indirection that buys
nothing" the project guards against.

So the real, defensible goal is the *locality + navigability* win that `song.py`
delivers — **a builder file that is orchestration-only, with cohort slice logic
behind an importable seam** — achieved without the cross-cohort merge contract.

### Scope decision: 2b only, skip the shared-shell dedup (2a)

A separate "dedup the byte-identical builder-shell helpers (`_render_template`,
`write_payload`, `_peak_rss_mb`, `_count_csv_rows`, `_copy_audit_source`,
`_json_preview`) into `viewer/shared/`" pass was considered and **dropped by
decision.** The two builders keep their own copies of those helpers; the
continued duplication is accepted. This refactor does **not** touch
`build_unified_viewer.py` or `alz/viewer/`. Only the T-cell package changes.

### Phase 2b — extract T-cell cohort slice logic out of the 2,729-line builder

`build_tcell_viewer.py` holds ~30 cohort-specific slice functions inline. Move
them into **four focused seam modules grouped by concern** under
`alz/tcell_viewer/`, plus one shared leaf, leaving the builder a thin
orchestrator. (One flat `slices.py` was rejected — it would just relocate the
monolith.)

**`alz/tcell_viewer/common.py`** — shared leaf (no intra-package imports), holds
the cross-cutting domain constants *and* the one cross-cutting helper:
`DONORS`, `DONOR_WITH_MEA`, `TIMEPOINT_COLOR_MAP`, `PROJECTILS_LABEL_MAP`,
`TCELL_ATTRIBUTION_CAVEAT`, and `_incytr_sanitize`. Imported by both the builder
(`write_html` needs `TIMEPOINT_COLOR_MAP`; `validate`/loops need `DONORS`) and
all four slice modules.

**`slices_incytr.py`** — `_write_donor_pair_pathways` (902–1427),
`_write_tcell_pair_pathways` (1430–1478), `_read_tcell_incytr_celltype_qc`
(141–222), `_build_tcell_celltype_pathway_qc` (225–276),
`_contrast_from_filename` (301), `_contrast_days` (131), `_timepoint_label`
(306), `_short_contrast` (1561); module-local constants `_INCYTR_*` (117–128),
`_PAIR_FILE_RE` (298), `_TCELL_CONTRAST_RE` (1558).

**`slices_kinase.py`** — `_build_donor_kinases_slice` (602–739),
`_build_celltypes_slice` (761–778), `_build_celltype_assignment` (781–895),
`_build_tcell_attribution_index` (565–599), `_load_donor_kinase_attribution`
(315), `_load_tcell_attribution` (350), `_load_donor_clusters` (746),
`_load_kinase_to_gene_map` (525), `_tcell_attribution_uniform` (540),
`_nsclc_attribution_uniform` (549); plus the projected-state side-channel
(`_projected_state_candidate_dirs` 363, `_read_projected_state_rows` 381,
`_read_projected_state_manifest` 409, `_read_projected_state_mechanism` 439,
`_load_projected_state_mea_payload` 469).

**`slices_traces.py`** — `_build_tcell_measurement_trace` (1838),
`_write_tcell_transcript_trace` (1998), `_write_tcell_omics_trace` (2135),
`_parse_tcell_deconv_col` (2082), `_tcell_evidence_genes_by_cluster` (2091);
module-local constants `_TCELL_DAY_COL_RE` (1802), `_MEASUREMENT_TRACE_*`
(1804–1809), `_TCELL_OMICS_*` (2070–2076).

**`slices_audit.py`** — `build_tcell_audit_manifest` (1941),
`_register_kinase_audit_tables` (1712), `_tcell_audit_specs` (1509),
`_audit_csv_meta` (1573), `_rewrite_contrast_csv` (1594),
`_synthesize_site_level_ols` (1620), `_shim_audit_entry` (1664); the three audit
helpers that are called **only** here move with them (`_count_csv_rows` 292,
`_copy_audit_source` 1485, `_json_preview` 1495); module-level
`AUDIT_TABLE_SPECS = _tcell_audit_specs()` (1540).

**Stays in `build_tcell_viewer.py`** (orchestration only, importing from the
slice modules + `common`): `build_tcell_payload`, `write_html` (cohort-specific
day-palette sentinel logic), `write_payload`, `validate`, `main`, `_peak_rss_mb`,
`_render_template`, and `_TEMPLATE_DIR` / `_SHARED_TEMPLATE_DIR` /
`_VIEWER_SPECIFIC_TAB_INCLUDES` / `HERE`.

**Import direction is strictly one-way:** `common` ← {slices, builder};
`slices_*` ← builder. No slice module imports the builder; no cycles. Each slice
module re-imports what it needs from `tcell_viewer.paths`,
`tcell_viewer.common`, `alz.viewer.shared.*`, and `alz.shared.config`.

Result: builder shrinks from ~2,729 lines to a few hundred; the slice logic
becomes importable (and therefore unit-testable) in isolation. Same shape as
`song.py` holding the Song slice logic — **without** the compose contract that
doesn't fit a standalone viewer.

### Verification (item 2b)

Acceptance = **content-identical modulo timestamps** (NOT raw byte-identical —
the payload carries `meta.generated_at` and gzip headers embed an mtime). The
extraction is a pure code-move; the risk it guards against is a moved function
silently losing a module-global it closed over.

1. **Before** touching anything: `pixi run tcell-viewer`; snapshot
   `outputs/reports/tcell_viewer/*.payload.json` and the
   `edge_slices/incytr_pathways/` shard dir.
2. **After** the (single, full) extraction: rebuild; run a **throwaway compare
   script** (not committed) that asserts:
   - payload JSON identical after deleting `meta.generated_at`
     (canonical `json.dumps(sort_keys=True)` on both sides);
   - every `.json.gz` / `.bin.gz` shard identical after **gunzip** (compare
     decompressed content, never the gzipped bytes);
   - every parquet shard identical by **content via DuckDB** (streamed
     `EXCEPT` / sorted-hash, per the memory-safety rule — never a whole-frame
     pandas read).
   Any diff is a regression to fix, not to accept.
3. Keep `tests/test_tcell_viewer_optional_payload.py` green.

Execution: **one big extraction**, single content-compare at the end, **one
`refactor:` commit.** The build entry point is unchanged: `pixi run tcell-viewer`.
No pixi task, runner, or CLI changes.

---

## Suggested order of execution

1. **Item 1** (smallest, highest-leverage — unblocks importing config).
2. **Item 2b** (the T-cell extraction).

Each lands as its own `refactor:` commit with verification evidence (per the
project workflow rule: run the affected build/check and report pass/fail before
declaring done).

---

## Decisions locked (grilling pass, 2026-06-23)

**Item 1**
1. Lazy-load only; do **not** track `cluster_spine.csv` (it's derived).
2. Mechanism: PEP 562 module `__getattr__` (transparent lazy, zero call-site
   churn) + `if TYPE_CHECKING:` annotations. Type-checking is off project-wide,
   so the dynamic-attribute tooling cost is nil.
3. Lazy+memoized: `ANALYSIS_MODE`, `CLUSTER_SPINE`, `N_CELL_TYPES`.
   `WMB_REGION_SCOPE` stays eager (env-only; fail-fast validation is a feature).
4. Verification: one-time manual `mv`-the-CSV import check + existing
   `summary.py` / `verify_decomposition.py` / `pytest tests/`. **No new
   committed test.**
5. The `domain`/`paths` module split is descoped.

**Item 2**
6. **2b only.** The shared-shell dedup (2a) is dropped; the two builders keep
   their duplicated helper copies. `build_unified_viewer.py` / `alz/viewer/` are
   untouched.
7. Layout: four concern-grouped modules (`slices_incytr`, `slices_kinase`,
   `slices_traces`, `slices_audit`) + `tcell_viewer/common.py` (cross-cutting
   constants + `_incytr_sanitize`). Module-local constants ride with their slice
   module. One-way imports: `common` ← {slices, builder}; `slices` ← builder.
8. Acceptance: content-identical modulo timestamps; payload compared after
   dropping `meta.generated_at`; shards compared by decompressed/parsed content;
   parquet via streamed DuckDB content-comparison.
9. Execution: one big extraction, one content-compare, one commit.
