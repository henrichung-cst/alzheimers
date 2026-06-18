# Phase 5D — Shared-Machinery Dedup Audit

**Branch:** `refactor/cohort-namespaces`
**Scope:** the 5 helpers the 5A inventory (§3C) flagged as duplicated between
`alz/build_unified_viewer.py` and `alz/build_tcell_viewer.py`.
**Prime directive:** zero payload drift on BOTH viewers.

## Headline: the "byte-identical" premise is false — the two builders have drifted

The 5A inventory called four of these "byte-identical"/"identical." They are not.
Diffing the actual source (AST-extracted, post-5C line numbers):

| Helper | unified vs tcell | Output-equivalent? | Verdict |
|---|---|---|---|
| `_sanitize` | 6 lines: `isinstance(x, np.integer)` vs `(np.integer,)` etc. | **Yes** — `isinstance(x, T)` ≡ `isinstance(x, (T,))` | **SAFE to unify** |
| `_configure_duckdb_tempdir` | unified has a 2-line explanatory comment; same SQL (`temp_directory`, `max_temp_directory_size='40GiB'`) | **Yes** — comment-only | **SAFE to unify** |
| `_build_incytr_gene_node_index` | tcell uses a local `clean_float`; unified uses module `_json_clean_value`; `roles` hoisted | **Yes** — `clean_float(v,d)` ≡ `_json_clean_value(v,d)` on the float-only inputs (best_abs_pds/best_pds/best_pvalue). unified is the evolved copy. | **SAFE to unify** (also lift `_json_clean_value` + `_INCYTR_FC_NODES`) |
| `_build_kinase_motifs` | 57 lines: tcell resolves aliases + emits `source_name`; unified does neither | **NO — different output** | **NEEDS A DECISION** (below) |
| incytr pair-shard writer (`_write_incytr_pair_pathways` / `_write_donor_pair_pathways`) | ~70% common; differs by multi-donor loop, 3-part filename template, per-donor context scoping | partial | **DEFER** (high-volume DuckDB streaming; own wave) |

## The `_build_kinase_motifs` reconciliation (the real decision)

Two genuinely different implementations:

- **tcell**: sorts+dedups names; on a failed `kl.get_matrix` lookup, falls back to
  `_KINASE_LIBRARY_MOTIF_ALIASES` (tcell-only: `ALK1→ACVRL1`, `ALK2→ACVR1`,
  `ALK4→ACVR1B`, `ALK7→ACVR1C`); emits an extra `source_name` key per entry.
- **unified**: iterates names as given; **no alias fallback** — a name that fails
  direct lookup is silently `skipped`; no `source_name` key.

Facts:
- `_KINASE_LIBRARY_MOTIF_ALIASES` exists **only** in the tcell builder.
- `source_name` is read by **no frontend JS** (grep: zero `.js` hits) — unread payload.
- Therefore unifying to ONE implementation is **not zero-drift**:
  - Pick unified's → tcell **loses** ALK alias motifs (regression) + loses source_name.
  - Pick tcell's → unified **gains** source_name on every entry, and gains any
    ALK1/2/4/7 motifs it currently skips. A unified payload change — arguably a
    **bug fix** (unified silently drops resolvable activin-receptor motifs today).

There is no zero-drift unification. This is a correctness reconciliation, not a
pure dedup, so it needs sign-off.

> **CORRECTION (verified post-implementation, 2026-06-18).** The premise that
> unified "silently skips ALK1/2/4/7" was **wrong**. Running both the old-unified
> and the shared `_build_kinase_motifs` on the unified viewer's real 389-name
> kinase set produced an **identical** motif set — `kinase_library` resolves
> ALK1/2/4/7 **natively** via direct `kl.get_matrix`, so old-unified already
> emitted them. The tcell alias map is redundant for these names (defensive
> against a different package data table, per its own comment). **Actual option-B
> delta: unified is byte-identical (zero change); tcell loses only the unread
> `source_name` key from its 389 entries.** Cleaner than projected — there was no
> unified bug to fix. The records below are kept for provenance; this note is the
> authoritative outcome.

### Recommendation (option B below)
Shared `build_kinase_motifs` = tcell's alias-aware logic, **drop the unread
`source_name` key**. Result:
- tcell: loses only `source_name` (unread → behavior-safe).
- unified: gains alias resolution for ALK1/2/4/7 **iff** its kinase universe
  contains those names (a motif-coverage bug fix); no `source_name` added.
- I verify the exact `kinase_motifs` delta on BOTH payloads and report it before
  it ships — if unified's kinase set has no ALK shorthands, unified is unchanged.

Options:
- **A.** Shared = tcell verbatim (alias + `source_name`). Unified gains `source_name`
  on every motif entry (structural payload change, unread).
- **B (recommended).** Shared = tcell alias logic, no `source_name`. Minimal drift;
  keeps the bug fix.
- **C.** Don't dedup `_build_kinase_motifs`; lift only the 3 safe helpers. Strict
  zero-drift, incomplete dedup, leaves the unified motif-skip bug in place.

## Proposed 5D execution
1. **Safe lift (this wave):** `_sanitize`, `_configure_duckdb_tempdir`,
   `_build_incytr_gene_node_index` (+ `_json_clean_value`, `_INCYTR_FC_NODES`) →
   `alz/viewer/shared/payload_helpers.py`; both builders import them. Provably
   output-equivalent → both payloads byte-identical. Independent verifier.
2. **`_build_kinase_motifs`:** per the decision above (A/B/C).
3. **incytr pair-shard writer:** deferred to a dedicated wave (5D-2) — extract a
   shared core parameterized by (output dir, filename template, donor/context
   scope); verified against both viewers' incytr shard trees. Not folded into the
   safe lift because of its volume + parity sensitivity.

## Parity gate
Per helper: AST-diff the lifted body against both originals; for the safe three,
output-equivalence is by construction. For `_build_kinase_motifs`, build the
`kinase_motifs` block for both viewers' real kinase-name sets before/after and
diff the resulting dict (key set + values), reporting any delta. No full 104 MB
payload rebuild required — the dedup is function-local.
