# Plans — implementation sequencing & adjudication

Goal: implement every plan in `docs/plans/`. This maps the cross-cutting threads, the blocking
decisions, the conflicts, and a parallel execution order.

Plan shorthand used below:
`geneuse` = `ad_geneuse_unpin_from_sce4`
`apriori` = `tcell_apriori_expectations` (blind exhaustion prediction set → `docs/reference/`, unbuilt)
`deploy` = `deployment/todo9_viewer_aws_deployment` · B2/C3/E1/E2/G1/H1/I2 = backlog files.

---

## 1. Cross-cutting threads (shared substrates)

**T1 — kinase→kinase network.** The kinase→kinase edge model ships in
`alz/cross_reference/kinase_kinase_edges.py` (PSP `Kinase_Substrate_Dataset` ∪ cohort motif
co-enrichment → weighted, provenance-tagged interactome; spec in
`foundation/kinase_sidechain_incytr_graph.md`). `E1` — a concordance overlay on those chain edges — is
the one remaining consumer. It **reuses that backend rather than building a parallel network**: the
Johnson Kinase Library is a motif-scoring atlas and cannot emit kinase→kinase edges, so PSP is the
forced source, not a preference (D2).

**T2 — cell-type specificity engine.** `E2` is the only open consumer. It does not read the engine
directly: the bridge already resolves per-kinase attribution per cohort (song via
`kinase_hypothesis_table`, 5xFAD via `celltype_mea`) and writes `owning_cluster` /
`celltype_match_rank` into `kinase_node_hits.parquet`. E2's work is carrying that label past
`load_motif_edges`, which currently collapses it to `BOOL_OR(celltype_match)`. The review-time
cross-species breadth constraint lives in `foundation/specificity_confidence.md` §3a.

**T3 — Incytr recompute + viewer rebuild.** `geneuse` (un-pin → AD re-run), `B2`, `C3`, and `E1/E2`
tabs all end in a viewer payload rebuild. Recomputes are
**not DB-blocked**: the six IncytrDB objects are version-pinned (2026-03-09, MD5s in the audit manifest
at `~/Projects/work/incytr/inst/extdata/incytrdb_audit/`) and species selection is per-dataset correct;
a first-principles DB rebuild is blocked on source artifacts that no longer exist, which the no-contact
rule forbids requesting. Payload writers (`build_unified_viewer.py`, `build_tcell_viewer.py`,
cohort `*.py`, `slices_incytr.py`) cannot be written concurrently — serialize integration, not authoring.

**T4 — T-cell labeling standard (SETTLED, nothing outstanding).** Canonical = per-cell marker
assignment; spec in `foundation/tcell_labeling_standard.md`. The whole t-cell chain is built on it
and consistent with it: Incytr `wide/` (donor1 ×3, donor2 ×4), the viewer (`build_tcell_viewer.py`
asserts `incytr_pair_mode_tcells` is the resolver default), and the report suite. No t-cell item is
open against this standard, and no plan below is gated on it.

---

## 2. Adjudications required (blocking — resolve before the dependent wave)

**D2 — E1 scope. RESOLVED: concordance overlay only.** E1 reuses `kinase_kinase_edges.py` and renders
as an overlay within the existing sidechain sub-tab. Two constraints found when scoping against the
shipped stack:

- **No signed reference is available.** `Kinase_Substrate_Dataset` records that A phosphorylates B
  with reference counts, but carries no activating/inhibiting sign — phosphorylation does either
  depending on the residue. PSP's `Regulatory_sites` file is not shipped in `kinase_library`. So E1
  flags observed **concordance**, and cannot claim an observation contradicts an expectation. The
  Johnson Kinase Library is a motif-scoring atlas and cannot supply the sign either.
- **Direction source is `signed_nes`, not `disease_lfc`.** Chain edges are signaling edges, so the
  propagated quantity is activity, not abundance; and NES already drives every other sidechain
  encoding. Rationale in `kinase-regulation-network.md`.

E1 is gated on a motif-similarity confound check (Gate 0): similar-motif kinases co-enrich by
construction, so concordance may measure substrate overlap rather than regulation.

**D3 — geneuse un-pin. RESOLVED: un-pin.** AD switches to the derived `DEG∪prG` recipe (same path as
t-cells); single cross-cohort recipe, wider enumeration (downstream SigProb/|PDS| filter still applies).
Removes the `use_frozen_geneuse` branch + `extract_sce4_geneuse.R` (anti-shim). Production-output-
changing — requires a full AD re-run + viewer rebuild (Wave 3). Touch points in
`ad_geneuse_unpin_from_sce4.md`.

**D5 — I2 timing. UNBLOCKED.** The kinase-sidechain work that imported the frozen-layer modules `I2`
relocates (`*_decompose.py`, `song.py`, R extractors) has landed, so the gate that held `I2` back is
gone. It remains an **isolated-window** job: relocating those modules breaks every importer at once, so
run it alone, updating all importers in the same pass, and never interleave it with other file-touching
plans.

---

## 3. Dependency graph

```
D3 ──► geneuse re-run ──► AD viewer rebuild ─────────────────────┐
                                                                 ├─► serialized
E1 concordance overlay (viewer-side, Gate 0) ────────────────────┤   viewer
E2 celltype discrimination (owning_cluster + re-run) ────────────┤   integration
B2 (sankey), C3 (early-change) ──────────────────────────────────┘

G1 (diagrams), apriori (docs), deploy-C/D ── ungated leaves, deprioritized to last
I2 ── isolated window (D5)
```

---

## 4. Conflicts

- **I2 ↔ everything file-touching** — relocates frozen-layer modules the R pipeline and the
  cross-reference layer import (D5). Isolate.
- **Viewer payload write contention** — B2, C3, E1/E2, and the geneuse rebuild all write the payload
  builders. Not a logic conflict; serialize the *integration/rebuild* step (or use worktree isolation
  for authoring, single-threaded merge).
- **geneuse re-run ↔ AD-inclusive payloads** — the AD Incytr outputs change; any payload built during
  the re-run window is stale. Sequence the AD rebuild after the re-run completes.

---

## 5. Parallel execution waves

Ordered by analysis value. Docs and infra are ungated but deliberately last — they produce no
result on our own data.

**Wave 1 — Incytr recompute (long pole, start first).**
- `geneuse` un-pin (D3) → AD re-run → AD viewer rebuild. Production-output-changing, and the re-run
  window invalidates any AD payload built during it — so it leads, and the Wave 3 tabs merge after it.

**Wave 2 — kinase overlays on the shipped sidechain (parallel with Wave 1's re-run).**
- `E1` concordance overlay — viewer-side, no backend, gated on the Gate 0 motif-similarity check.
- `E2` cell-type discrimination — needs `owning_cluster` carried past `load_motif_edges`, plus a
  `kinase_kinase_edges` re-run (bridge unaffected).

**Independent of each other** — `E2` does not depend on `E1`; the earlier dependency was spurious.
Both author against the sidechain view and touch no AD payload, so both run concurrently with the
Wave 1 re-run.

**Wave 3 — viewer tabs (serialize the payload/integration step).**
- `B2` sankey, `C3` early-change tab. Author in parallel; merge and rebuild one at a time, after the
  AD rebuild lands. `E1`/`E2` are overlays inside the shipped sidechain sub-tab, not new tabs — they
  land in Wave 2 and only their payload rebuild serializes here.

**Isolated — `I2`** frozen-layer namespace moves, dedicated window (D5).

**Back of queue — no analysis output, no gate, pick up when the above clears.**
- `G1` workflow diagrams (docs).
- `apriori` blind exhaustion prediction set (docs) — feeds the `/check-controls` plausibility skill.
  Blindness caveat: the t-cell Incytr report suite is already built, so ground the predictions in
  literature only, not those outputs.
- `deploy` Options C (parquet + HTTP Range) and D (DuckDB-Wasm) — infra; worth doing only once
  payload size actually hurts load time.

Nothing is blocked on an open decision: D2, D3, and D5 are settled and direct the waves above.
