# Plans — implementation sequencing & adjudication

Goal: implement every plan in `docs/plans/`. This maps the cross-cutting threads, the blocking
decisions, the conflicts, and a parallel execution order. Companion to `plans_state_audit_2026-07-10.md`.

Plan shorthand used below:
`sidechain` = `kinase_sidechain_incytr_graph/` (01–04) · `geneuse` = `ad_geneuse_unpin_from_sce4`
`corr` = `tcell-proteome-transcriptome-correlation` · `matt` = `tcell-matt-report-restoration` (active WIP, user-owned)
`deploy` = `deployment/todo9_viewer_aws_deployment` · A3/A4/B1/B2/C3/C4/E1/E2/G1/G2/H1/I1/I2 = backlog files.

---

## 1. Cross-cutting threads (shared substrates)

**T1 — kinase→kinase network.** `sidechain-01` and `E1` both build a kinase→kinase graph from
PhosphoSite-family data with a viewer tab. `sidechain-01` = PSP `Kinase_Substrate_Dataset` unified with
MEA motif evidence → weighted interactome + per-pathway cytoscape sidechains. `E1` = a reference
regulation hierarchy (A↑⇒B↓) with an observed disease-direction overlay + click-kinase-neighbors tab.
**These overlap at the backend (`kinase_kinase_edges.py`) and at the viewer (two kinase-network tabs).**
Adjudicate before building either (§2, D2).

**T2 — cell-type specificity engine.** `C4`, `E2`, and `G2` all consume the existing per-kinase
cell-type specificity basis. `C4` gates targets on human-vs-mouse breadth; `E2` uses specificity to
split same-family kinases; `G2` validates the specificity metric against known controls (PHKG1
astrocyte, ATP9A endothelial). Same data source, three lenses — build the read path once, share it.

**T3 — Incytr recompute + viewer rebuild.** `geneuse` (un-pin → AD re-run), `sidechain` (consumes
Incytr `wide/` output), `B2`, `C3`, `E1/E2` tabs, and the t-cell relabel all end in a viewer payload
rebuild. `B1` (IncytrDB version audit) is **covered** — the six DB objects are version-pinned
(2026-03-09, MD5s in `current_manifest.json`) and species selection is per-dataset correct, so
recomputes are not DB-blocked. Payload writers (`build_unified_viewer.py`, `build_tcell_viewer.py`,
cohort `*.py`, `slices_incytr.py`) cannot be written concurrently — serialize integration, not authoring.

**T4 — T-cell labeling ontology (RESOLVED).** Canonical = **Matt cluster-relabel** (Seurat clusters →
a small biological label set via `tcell_state_labels.py` `MATT_RELABEL_TABLE`). ProjecTILs is
**permanently retired** (never canonical again; kept as evidence columns only); the marker-panel
per-cell labeler (`a7d53d2`) is abandoned. The labeler is an **active WIP owned by the user** — the
t-cell Incytr input re-key + pair-mode re-run (donor1 3 contrasts, donor2 4), viewer rebuild, report
regen, `sidechain-02`, and `A3` all wait on those labels **freezing**, not on any further decision.
**`corr` is the one t-cell plan independent of labeling** (raw-Seurat pseudobulk, no state idents) — it
can proceed regardless.

---

## 2. Adjudications required (blocking — resolve before the dependent wave)

**D1 — T-cell labeling ontology. RESOLVED** (see T4): canonical = Matt cluster-relabel; ProjecTILs
permanently retired; marker-panel per-cell abandoned. No longer an open adjudication — the only
remaining gate is the user's WIP labels freezing before the t-cell Incytr re-key/re-run.

**D2 — E1 vs sidechain scope.** Decide: (a) does `E1`'s reference hierarchy reuse `sidechain-01`'s
`kinase_kinase_edges.py` backend, or a different source (the Johnson Kinase Library motif atlas is a
distinct product from PSP's `Kinase_Substrate_Dataset`)? (b) one kinase-network tab with a
regulation-direction overlay toggle, or two separate tabs? Recommendation: **shared backend, one tab.**
`E1` becomes a directional/disease-overlay *layer* on the sidechain interactome rather than a parallel
network — otherwise we ship two kinase graphs the user must reconcile. Confirm the data-source question
with domain input before locking.

**D3 — geneuse un-pin: yes/no.** Un-pinning switches AD to the derived `DEG∪prG` recipe (single
cross-cohort recipe, wider enumeration) and requires a full AD re-run + viewer rebuild. Recommendation:
**un-pin** — consistent with "sce4 parity is a closed non-goal"; keeping the frozen sets only makes
sense if judged the better selection on independent merits. Production-output-changing, so it is an
explicit go/no-go.

**D4 — C4 form.** Operational review checklist (doc/skill) vs an automated guard wired into the
specificity/attribution surface. Recommendation: **automated guard** folded into the existing
`check-controls`/attribution path (consumes T2), with the checklist as its human-readable output — a
prose-only checklist will not be enforced.

**D5 — I2 timing.** `sidechain-01/02` import the exact frozen-layer modules `I2` relocates
(`*_decompose.py`, `song.py`, R extractors). Running `I2` mid-stream breaks those imports.
Recommendation: **do `I2` last, in an isolated window**, updating all importers in the same pass; or
skip until the sidechain work lands. Never interleave it with other file-touching plans.

---

## 3. Dependency graph

```
D3 ──► geneuse re-run ──► AD viewer rebuild ─────────────────────┐
                                                                 ├─► serialized
t-cell labels FREEZE (Matt cluster-relabel WIP)                  │   viewer
   └─► Incytr re-key + re-run ──► t-cell viewer ──┬─► sidechain-02│   integration
                                                  └─► A3          │
                                                                 │
sidechain-01 (PSP backend) ──┬─► sidechain-03 ──► sidechain-04 ──┤
sidechain-02 ────────────────┘                                   │
                                                                 │
D2 ──► E1 (reuse sidechain-01 backend) ──► E2 ───────────────────┤
T2 specificity ──► C4, G2, E2                                    │
B2 (sankey), C3 (early-change) ─────────────────────────────────┘

corr, tmt (H1), G1 (diagrams), deploy-B  ── independent leaves, no gate
B1 (covered), A4 (verified) ── done
I1 ── independent (low conflict) · I2 ── isolated window, last (D5)
```

---

## 4. Conflicts

- **E1 ↔ sidechain** — feature/backend overlap (T1). Hard conflict until D2 resolves; do not build E1
  before sidechain-01's edge model exists and the shared-backend decision is made.
- **I2 ↔ everything file-touching** — relocates modules that sidechain and the R pipeline import (D5).
  Isolate.
- **Viewer payload write contention** — sidechain-03/04, B2, C3, E1/E2, geneuse rebuild all write the
  payload builders. Not a logic conflict; serialize the *integration/rebuild* step (or use worktree
  isolation for authoring, single-threaded merge).
- **geneuse re-run ↔ AD-inclusive payloads** — the AD Incytr outputs change; any payload built during
  the re-run window is stale. Sequence the AD rebuild after the re-run completes.

---

## 5. Parallel execution waves

**Wave 0 — DONE / resolved.**
- `B1` IncytrDB — covered by the existing `incytrdb_audit/` (version-pinned, species-correct; rebuild
  blocked on unobtainable source artifacts, which the no-contact rule forbids requesting).
- `A4` T-cell data structure — verified on disk (donor1 has IMAC, donor2 does not; both total+pY+scRNA).
- `matt` labeling — RESOLVED to Matt cluster-relabel (D1/T4); the labeler is user-owned active WIP.
- Human decisions still open: D2 (E1/sidechain scope), D3 (geneuse), D4 (C4 form).

**Wave 1 — independent builds (parallel, disjoint outputs, no gate).**
- `corr` (labeling-independent standalone script + CSV) — the one unblocked t-cell task.
- `H1` TMT paper IMAC fetch + enrichment.
- `G1` workflow diagrams (docs).
- `deploy` Option B (shard-hash manifest) — infra layer, separate from analysis.
- `I1` sidecar cleanup — low file-conflict, any time.
- `A3` t-cell Incytr trend coverage — deferred: it verifies the t-cell Incytr re-run, which waits on the
  labels freezing. Not Wave 1.

**Wave 2 — kinase backends + specificity thread (after D2).**
- `sidechain-01` (PSP interactome) ∥ `sidechain-02` (t-cell motif — needs D1).
- `E1` on the shared backend, then `E2` (needs T2 specificity).
- `C4`, `G2` (T2 specificity consumers) — parallel with the above, disjoint outputs.

**Wave 3 — Incytr recompute (after B1 + D3).**
- `geneuse` un-pin → AD re-run → AD viewer rebuild.

**Wave 4 — viewer tabs (serialize the payload/integration step).**
- `sidechain-03 → 04`, `B2` sankey, `C3` early-change tab, `E1/E2` tab(s). Author in parallel; merge
  and rebuild one at a time.

**Isolated — `I2`** frozen-layer namespace moves, dedicated window, last (D5).

---

## 6. Open questions to route to domain input
- D2 data source: is E1's reference hierarchy the Johnson Kinase Library atlas or PSP's substrate
  dataset (the one sidechain-01 already loads)?
- D3: un-pin AD gene.use (recommended) or keep frozen sets on independent merit?
- D4: C4 as an automated guard (recommended) or a prose checklist?
