# Phase 5 Decisions — Viewer Slice Contract

**Branch:** `refactor/cohort-namespaces`
**Base commit:** `240af6b` (Phase 4 committed)

---

## Packet 5A — Payload field inventory

Read-only survey delivered to
`docs/audits/cohort_abstraction_refactor/phase_5A_payload_inventory.md`: 14
unified top-level keys, 9 lazy shard families, the T-cell payload, the
frontend payload-key→consumer map, the two validators + their blind spots, and 8
risky extraction points (R1–R8). Adapter order confirmed mukesh → tcell →
fivexfad → song.

---

## Packet 5B — Slice schema

New files (schema only; NOT wired into any builder this packet):
- `alz/viewer/shared/__init__.py`
- `alz/viewer/shared/cohort_slice.py` — `CohortViewerSlice`, `EdgeSliceContribution`
- `alz/viewer/shared/compose.py` — `compose_viewer_slices` + merge primitives

### Design: owned sections vs merge contributions
A slice splits its payload contribution into two disjoint parts:
- **owned sections** — top-level keys produced by exactly one cohort, placed
  verbatim by the composer (no re-mapping).
- **merge contributions** — values several cohorts feed into a SHARED key, which
  the composer reduces: `capabilities`→`meta.capabilities`,
  `audit_entries`→`audit_tables.tables`, `edge_slice_ref`→`edge_slice_ref`,
  `kinase_names`→`kinase_motifs` (name union; PSSM matrices built by an injected
  `kinase_motifs_builder`, so the composer never touches the motif data source).

`SHARED_TOP_LEVEL_KEYS = {meta, kinase_motifs, audit_tables, edge_slice_ref}` —
a cohort claiming any of these as `owned_sections` is rejected at construction.

### Payload versioning: NONE
The slice is an INTERNAL representation. Composing slices must reproduce today's
v2 payload structurally identically — no new keys, no renamed keys, no version
bump, no dropped fields. `meta.viewer_payload_schema_version` stays `2`. This is
forced by the Phase-5 parity invariant, not a design choice; there is no
versioned-contract-change here. (Non-goal restated: "do not change viewer payload
semantics without a versioned contract change" — we are changing none.)

### Canonical key order
`compose.TOP_LEVEL_ORDER` pins the exact emission order of
`build_unified_viewer.build_payload()` (meta, kinases, kinase_motifs, celltypes,
kinase_celltype_evidence, attribution_index, decomposition_index,
agreement_index, subclass_breakdown, audit_tables, edge_slice_ref,
incytr_pathways, [human], [supporting_5xfad]). The composer emits keys in this
order; optional cohort blocks appear only when a slice owns them. Any composed
key absent from the order raises (contract-drift guard).

### Merge rules (handle the inventory's risky points)
- **R1 audit_tables** — `merge_audit_tables`: per-cohort `audit_entries` folded
  under `tables`; duplicate table key across cohorts raises (no silent
  last-writer-wins).
- **R3 kinase_motifs** — `union_kinase_names` → sorted union, fed to the injected
  builder; composer responsibility, no per-cohort logic.
- **R4 capabilities** — `merge_capabilities`: monotone `False`→`True` promotion
  (reproduces the builder's two-pass behavior); never demotes a `True`.
- **R8 edge_slice_ref** — `merge_edge_slice_ref`: per-cohort shard pointers
  aggregated by exact payload key name; cross-cohort key collision raises.

### R2 cross-cohort seam (fivexfad ← song MEA) — DESIGNED, implemented in 5E
The `supporting_5xfad` confidence alignment
(`_assign_fivexfad_song_aligned_confidence`) depends on song's bulk MEA. Decision:
this is an INPUT to the **fivexfad adapter** — its `build_viewer_slice(...)` will
take the song MEA reference as an explicit named argument — NOT a step inside the
composer and NOT a hidden cross-cohort read in the slice. The composer's only
obligation is ordering: build the song slice before the fivexfad slice so the
song MEA is available to pass along. The composer itself never reaches across
cohorts. (5B defines the seam; 5E implements the fivexfad adapter.)

### R6 attribution_index schema divergence
The unified `attribution_index` (~30 song/WMB/SEA-AD columns) and the T-cell
`attribution_index` (11 tcell-specific columns) are NOT unified. Each is an
`owned_sections["attribution_index"]` of its own cohort/viewer; the schema models
one per cohort, never a merged schema. (Attempting unification is explicitly
out of scope — high risk per R6.)

### Composer status: defined, not wired
`compose_viewer_slices` is a pure reducer, exercised by an in-process self-check
(key order, all four merges, and three contract invariants — shared-key claim,
duplicate owner, pointer collision — all pass). It is NOT yet called from
`build_unified_viewer.build_payload()`. Phase 5C wires the first real adapter
(mukesh) and verifies payload parity against the current builder.

### 14-key representability self-check
Every current top-level payload key maps to a slice field — none is
unrepresentable:

| Payload key | Carried by |
|---|---|
| `meta` | composer-assembled (+ `capabilities` merge) |
| `kinases` | song `owned_sections` |
| `kinase_motifs` | composer (`kinase_names` union → builder) |
| `celltypes` | song `owned_sections` |
| `kinase_celltype_evidence` | song `owned_sections` |
| `attribution_index` | song `owned_sections` (tcell: tcell `owned_sections`) |
| `decomposition_index` | song `owned_sections` |
| `agreement_index` | song `owned_sections` |
| `subclass_breakdown` | song `owned_sections` |
| `audit_tables` | composer (`audit_entries` merge) |
| `edge_slice_ref` | composer (`edge_slice_ref` merge) |
| `incytr_pathways` | song `owned_sections` (tcell: tcell `owned_sections`) |
| `human` | mukesh `owned_sections` (optional) |
| `supporting_5xfad` | fivexfad `owned_sections` (optional) |

T-cell-only keys (`celltype_assignment`) are `owned_sections` of the tcell slice;
the tcell viewer is a separate deliverable that composes its own slice set (5D).

---

## Packet 5C — mukesh adapter (first parity-bearing extraction)

First real extraction out of the `alz/build_unified_viewer.py` monolith. Mukesh
(human NBB per-donor) payload construction moved into a cohort adapter; the
monolith now consumes a `CohortViewerSlice` instead of inlining the human block.

### What moved
`build_human_slice`, `_human_track_load`, `_write_human_perdonor_substrate_slices`,
and the two constants `HUMAN_PERDONOR_DIR` / `HUMAN_TRACK_SUFFIXES` moved verbatim
from `build_unified_viewer.py` into new `alz/viewer/cohorts/mukesh.py`. Net
monolith change: **+13 / −499 lines** (one changed file). The nested `_gr` helper
stays nested inside `build_human_slice`; no monolith-local helper dependency
existed, so no circular import. The now-dead `EDGE_SLICES_HUMAN_PERDONOR_DIR`
import was removed from the monolith.

### The adapter
`build_mukesh_viewer_slice() -> CohortViewerSlice | None`. Returns `None` when the
perdonor outputs are absent (mouse-only build omits `PAYLOAD.human`, byte-equivalent).
When present: `cohort_id="mukesh"`, `owned_sections={"human": human_slice}`,
`capabilities={"human_reference": True}`, one `EdgeSliceContribution("human_perdonor",
{"present_human_perdonor_kinase_ids": …})`, `kinase_names` = the human kinase names.

### Incremental wiring (NOT the composer yet)
`build_payload()` calls the adapter and unpacks `owned_sections["human"]`, the
capability flag, the `present_human_perdonor_kinase_ids` edge-ref entry, and the
kinase-name union into the **existing inline assembly**. `compose_viewer_slices`
stays unit-tested but is NOT the assembly path this wave — it becomes the path at
5F when all cohorts are adapters, so the old inline path remains intact and
revertible until full parity is accepted (matches the plan's rollback criteria).
This keeps 5C output **byte-identical**, not merely structurally identical.

### Parity gate (byte-identical, memory-safe)
The diff is confined to the human path, so the other 12 payload keys cannot drift
and a full 104 MB rebuild is unnecessary. Gate = a targeted snapshot of the mukesh
contribution captured BEFORE the edit (`build_human_slice()` output + the 390
on-disk `human_perdonor` shard sha256s) and re-checked AFTER: 389 human kinase
rows, 390 shards, **byte-identical**. The pre-edit `if human_perdonor_substrate_index:`
guard was provably always-True (`_write_human_perdonor_substrate_slices` always
returns a non-empty dict), so dropping it is behavior-preserving.

### Independent verification (verifier ≠ implementer)
A separate `audit-pipeline` verifier re-ran the parity check and audited the diff:
parity PASS; clean import / no cycle; complete move with zero residual references
to the five moved names (no stub/alias/commented block); faithful move (function
bodies byte-identical); rewiring semantically byte-identical including the
None/mouse-only path; adapter claims no `SHARED_TOP_LEVEL_KEYS`; no frozen-path
(`_run_mea`) contamination. Verdict **PASS — 5C closes**. The lone finding (a dead
import) was fixed and re-verified.

### Status
5C closed, byte-identical. Next: 5D tcell adapter (independent builder), then 5E
fivexfad (resolves the R2 song-MEA seam), then 5F song + composer cutover.

---

## Packet 5D (wave 1) — shared-machinery dedup

5D was rescoped after reading the tcell builder: it is **already cohort-pure** (one
builder, one cohort, separate output), so a 5C-style cohort extraction buys little.
User chose **shared-machinery dedup** — collapse the helpers duplicated across the
unified and tcell builders into one shared module. Audit:
`phase_5D_dedup_audit.md`.

### Finding: the "byte-identical" premise (5A §3C) was false
The two builders had DRIFTED. Of the five flagged helpers, three are
output-equivalent, one genuinely diverges, one is deferred:

- **Lifted (output-equivalent) → `alz/viewer/shared/payload_helpers.py`:**
  `_sanitize` (single-type vs 1-tuple `isinstance` — equivalent), `_configure_duckdb_tempdir`
  (comment-only diff), `_build_incytr_gene_node_index` (+ `_json_clean_value`,
  `_INCYTR_FC_NODES`; tcell's local `clean_float` ≡ `_json_clean_value` on the
  float-only aggregate inputs). Both builders now import these; names preserved
  (no rename — `_sanitize` alone has 25 call sites). Both payloads byte-identical.
- **`_build_kinase_motifs` — reconciled per approved option B:** shared = tcell's
  alias-aware logic with the **unread `source_name` key removed**. The alias map
  `_KINASE_LIBRARY_MOTIF_ALIASES` moved to the shared module.
- **incytr pair-shard writer — deferred to 5D-2** (user decision): ~70% common,
  high-volume DuckDB streaming, sce4-sensitive; its own wave.

### Real-payload delta (verified, memory-safe — no 104 MB rebuild)
The independent verifier streamed each viewer's actual 389-name kinase set and ran
old-vs-new `_build_kinase_motifs`:
- **unified: ZERO change** — `kinase_library` resolves ALK1/2/4/7 **natively**, so
  old-unified already emitted them. The audit's "unified silently skips ALK" premise
  was wrong; corrected in the audit doc. No bug existed to fix.
- **tcell: `source_name` removed from all 389 entries** (unread by any JS),
  otherwise identical.

Net diff: **−360 lines** across the two builders (+2 import lines), new shared
module. Independent `audit-pipeline` verifier PASS (function gate + AST verbatim
checks on the safe three + real-payload stream + scope containment: incytr writers
and `_run_mea` untouched). Verdict **PASS — 5D wave-1 closes**.

### Status
5D wave-1 closed (unified byte-identical; tcell −unread `source_name`). Next: 5D-2
incytr pair-shard writer dedup, then 5E fivexfad, then 5F song + composer cutover.

---

## Packet 5D-2 — incytr shard writer: index-format layer only

The 5A "70% common, extract a shared core" recommendation for the two incytr
shard writers was **rejected on measurement** (`phase_5D_dedup_audit.md` §5D-2):
they are 53% common (725 vs 537 lines); the 700-line orchestrators differ for real
reasons (AD 9-contrast 31×31 grid + CR-04 trajectory + global `.bin.gz` vs tcell
donor-scoped 3-part filenames). Merging them = wrong-abstraction trap.

**User decision: lift the binary-index FORMAT layer only.** The one
correctness-grade motive is that both viewers must emit the SAME `.bin.gz` layout
that `incytr_global_index.js` decodes — so make that one source of truth.

New `alz/viewer/shared/incytr_index.py` (42 lines): the vocab constants
(`_INCYTR_LABEL_NODES`/`_INCYTR_LABEL_COLS`/`_INCYTR_LABEL_VOCAB`/`_INCYTR_SCORE_COLS`/
`_SIGN_VEC_LABELS`) + the two PURE encoders (`_idx_label_bits`, `_idx_traj_bits`),
lifted verbatim from the unified builder's nested defs. Both builders import them
(names preserved; tcell omits the unused `_SIGN_VEC_LABELS`/`_idx_traj_bits`). The
700-line orchestrators are **untouched** — SQL, streaming, `_flush`,
`_idx_gene_ids`, `_accumulate_index` all stay per-builder (`_idx_gene_ids` /
`_accumulate_index` are tiny + state-entangled; not worth lifting). Net −66 lines.

**Parity (no shard-tree rebuild):** function-level bit + dtype array-equality of
the lifted encoders vs both builders' HEAD copies (synthetic label frames + traj
series) — orchestration untouched ⇒ identical encoder bytes ⇒ identical shard
trees by construction. Independent verifier PASS; the diff's only `+` lines are
the two import statements (critical check: zero stray orchestrator edits).

### Status
5D fully closed (wave-1 dedup + 5D-2 index-format lift). Next: 5E fivexfad
(resolves the R2 song-MEA seam), then 5F song + composer cutover.

---

## Packet 5E — fivexfad adapter

Extract the 5xFAD supporting-cohort payload construction from the unified
monolith into `alz/viewer/cohorts/fivexfad.py`, mirroring the 5C mukesh adapter.

**R2 correction (the "song-MEA seam" is a non-seam).** The 5B design assumed
"5xFAD attribution confidence derives from **song's** bulk MEA → the adapter must
take song MEA as an explicit argument; the composer orders song-before-fivexfad."
Reading the actual code overturns this:
- `_assign_fivexfad_song_aligned_confidence(attribution_rows, rows)` —
  `build_unified_viewer.py:2410` — the second arg is `rows`, the **5xFAD cohort's
  OWN bulk MEA** (built from `FIVEXFAD_KINASE_DIR` at L2334–2377), not song's.
  "Song-aligned" names the shared *semantics/convention* (significant-bulk +
  direction-support gate), not a shared *data* dependency.
- The slice's ONLY cross-cohort input is `data.celltype_evidence` (a shared
  reference table on `UnifiedData`), consumed by `_build_fivexfad_attribution_rows`
  (L1186) — the same `UnifiedData` every slice already receives, NOT a song-MEA
  argument.

So `build_supporting_5xfad_slice(data)` is self-contained given `data`; there is
no song→fivexfad ordering constraint and no extra argument. R2 closes as a
documentation error, not a real seam. The adapter signature is just
`build_fivexfad_viewer_slice(data) -> CohortViewerSlice | None`, exactly parallel
to mukesh.

**Move set (pure relocation, fully contiguous):** every def from
`_subs_fraction_counts` (L1036) through `build_supporting_5xfad_slice` (L2291,
ends L2462) plus `_age_from_contrast_label` (L2465) — all are `_f5_*` /
`_*fivexfad*` / the slice builder; grep confirms zero song/mukesh callers of the
generically-named ones (`_f5_records`, `_subs_fraction_counts`, `_age_from_contrast_label`).
First non-fivexfad def before is `_build_kinases_slice` (L981); first after is
`_build_celltypes_slice` (L2470). No interleaving → byte-identical relocation.

**Constants:** `FIVEXFAD_KINASE_DIR` is ALSO referenced by the shared
`_audit_specs()` composer (L179–182), so it moves to the leaf
`alz/viewer/paths.py` (SSOT) and is imported by both the monolith and the
adapter. The other 8 fivexfad-only constants are defined locally in the adapter
(mirroring how mukesh defines `HUMAN_PERDONOR_DIR` locally).

**Cycle avoidance:** `UnifiedData` is defined in the monolith (L846); the monolith
imports the adapter → the adapter must not import the monolith at runtime. It only
uses `UnifiedData` as a type annotation and duck-typed `.celltype_evidence` access,
so `from __future__ import annotations` + a `TYPE_CHECKING`-guarded import suffices.

**Wiring:** `build_payload` and `refresh_supporting_5xfad_payload` consume the
`CohortViewerSlice` (`.owned_sections["supporting_5xfad"]`, `.capabilities`,
`.kinase_names`) exactly as build_payload consumes the mukesh slice; the capability
merge stays inline (composer cutover is 5F). `audit_entries` stays unused — the two
5xFAD audit specs remain in the shared `_audit_specs()` composer (5C precedent).

**Parity:** fivexfad inputs are present on disk → real old-vs-new gate. (1) per-symbol
byte-identity of moved bodies vs HEAD (relocation proof ⇒ output identity by
construction); (2) live `build_supporting_5xfad_slice(data)` old-vs-new deep dict
compare (numpy.isclose floats) + shard-tree identity; (3) scope containment (diff =
monolith + new adapter + paths.py only). Independent verifier (audit-pipeline).

### Status
5E CLOSED. Implementer relocated the block (`build_unified_viewer.py` −1460/+13,
`paths.py` +6, new `alz/viewer/cohorts/fivexfad.py`). Independent verifier
(audit-pipeline, ≠ implementer) PASS on all 6 checks: all 30 moved function bodies
+ 10 constants byte-identical to HEAD; `FIVEXFAD_KINASE_DIR` RHS identical;
`DECOMP_FDR_AGREEMENT` from `alz.bulk_mea.confidence` (not the monolith — the one
dependency I had not pre-traced, caught by the implementer); `kinase_names`
set-equivalent to HEAD's row-derived motif union; no runtime monolith import
(TYPE_CHECKING only); live smoke returns `CohortViewerSlice(cohort_id='fivexfad')`
with 6,224 rows and all 15 top-level keys; scope contained. **R2 retired as a
documentation error — fivexfad is self-contained given `data`; no song-MEA
argument, no composer ordering constraint.** Next: 5F song + composer cutover →
Phase 5 boundary gate (hard human stop; final phase).
