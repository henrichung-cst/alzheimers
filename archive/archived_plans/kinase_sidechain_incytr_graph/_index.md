# kinase_sidechain_incytr_graph — subplan dependency index

Aligned + grilled, then decomposed by `/orchestrate` into four **file-disjoint** subplans — no two touch the same file, so
none can merge-conflict. The ordering below is a **data-flow dependency** (backend artifact → payload
→ viewer), not a file collision: a downstream subplan can be *authored* in parallel against the
contract, but its acceptance check only passes once its upstream lands.

## Subplans

| # | File | Unit | Risk | Files touched |
|---|------|------|------|---------------|
| 01 | `01_backend-edge-model.md` | PSP loader + kinase→kinase interactome + terminal-edge map (song, 5xFAD) | **high** | **new** `alz/cross_reference/kinase_kinase_edges.py` |
| 02 | `02_tcell-motif-bridge.md` | T-cell single-hop motif bridge, per-donor + celltype attribution | **medium** | `alz/cross_reference/kinase_incytr_bridge.py` |
| 03 | `03_payload-slice.md` | Per-cohort edge_slice_ref for the sidechain panel | **medium** | `alz/viewer/shared/compose.py`, `alz/viewer/shared/payload_helpers.py`, `alz/viewer/cohorts/song.py`, `alz/viewer/cohorts/fivexfad.py`, `alz/tcell_viewer/slices_incytr.py`, `alz/tcell_viewer/build_tcell_viewer.py` |
| 04 | `04_viewer-tab.md` | Cytoscape one-pathway sidechain sub-tab in the pathways detail panel | **low** | **new** `alz/viewer_shared/template/js/tabs/incytr_sidechains.js`, `alz/viewer_shared/template/js/tabs/incytr_pathways.js`, `alz/viewer_shared/template/index.html.j2` |
| 05 | `05_nes-edge-weighting.md` | Carry per-kinase NES through backend→payload→viewer; weight/prune sidechain edges by real per-kinase signal | **medium** | `alz/cross_reference/kinase_kinase_edges.py`, `alz/viewer/shared/payload_helpers.py`, `alz/viewer_shared/template/js/tabs/incytr_sidechains.js` |
| 06 | `06_edge-semantics-radial-spine.md` | Static legend; signed NES as edge hue + click-for-value; arc/per-node-wedge spine emphasizing core path over kinase sidechains | **medium** | `alz/cross_reference/kinase_kinase_edges.py`, `alz/viewer/shared/payload_helpers.py`, `alz/viewer_shared/template/js/tabs/incytr_sidechains.js` |
| 07 | `07_substrate-multiplicity-edge-weight.md` | Recover per-gene substrate count (discarded in the bridge) and make terminal-edge strength = \|NES\| × #substrates, so a kinase's fan is weighted by which node it hits hardest instead of flat | **medium** | `alz/cross_reference/kinase_incytr_bridge.py`, `alz/cross_reference/kinase_kinase_edges.py`, `alz/viewer/shared/payload_helpers.py`, `alz/viewer_shared/template/js/tabs/incytr_sidechains.js` |

## Model-mapping policy

Risk tier = **silent-failure risk**, not complexity — `(correctness-critical logic) × (how
undetectable a wrong answer is)`. Route accordingly (retune the mapping here; no subplan hard-codes a
model):

- **high → strongest model** (Opus). Wrong output looks plausible and ships silently.
- **medium → mid model** (Sonnet). Mostly loud failures, with a named silent spot to watch.
- **low → mid/cheapest** (Sonnet/Haiku). Fails loudly; the acceptance check catches it.

Note 04 is the largest subplan but **low** risk — a wrong render is visible on inspection. Complexity
would have mis-routed it to the strongest model and 01 (small, silent) to the cheapest, backwards.

## Collisions

01–03 are file-disjoint. **04 is not:** it wires the sidechain panel into
`alz/viewer_shared/template/js/tabs/incytr_pathways.js` (4 sites) rather than registering a top-level
tab, because no path-selection event exists to drive a separate tab from — see 04's "Navigation".
Do not run 04 concurrently with other work in that file.

## Declared data dependencies (prerequisite order, not file locks)

- **02 feeds 01.** 02's `kinase_node_hits.parquet` is the motif source 01 reads, for every cohort — 01
  reduces it into the interactome + terminal map. They are not independent producers.
- **03 depends on 01 only** — the payload slice reads 01's per-cohort backend artifact for all four
  cohort dirs. 02 is transitive: 03 never opens `kinase_node_hits.parquet`.
- **04 depends on 03** — the panel renders the real `incytr_sidechains` payload keys 03 emits.

Effective order: **02 → 01 → 03 → 04 → 05 → 06 → 07.**

- **05 depends on 01, 03, 04 (all shipped).** It changes the motif weight source
  from |PDS| to per-kinase |NES| in 01's backend, extends 03's shard schema, and
  rewires 04's edge selection. It touches `incytr_sidechains.js` (04's file) — do
  not run concurrently with other work in that file.
- **06 depends on 05.** It adds a signed-NES column to 01's terminal map, extends
  03's shard tuple again, and replaces 05's fan geometry with an arc/wedge layout
  plus legend and edge-click. Same three files as 05 — serialize after it, do not
  run concurrently in `incytr_sidechains.js`.
- **07 depends on 06.** NES is a kinase-level score broadcast identically to all a
  kinase's leading-substrate genes, so every terminal edge from one kinase renders
  at the same weight. 07 recovers the per-gene substrate count discarded at the
  bridge's `seen_genes` dedup (`kinase_incytr_bridge.py:164-172`) and makes
  terminal-edge strength `|NES| × #substrates`, differentiating a kinase's fan.
  Adds `kinase_incytr_bridge.py` to the touched set (new to this series — same file
  02 touched) plus the same three files as 05/06. Serialize after 06, do not run
  concurrently in `incytr_sidechains.js` or the bridge. Motif similarity is
  explicitly out — the gene match is exact-string, no graded PSSM score exists in
  the artifact.

## Before merge

These subplans may be handed to different tools (Claude agent, Codex, Copilot, a human) — each carries
its operating core inline. When each returns a diff, running `/code-review` on it before merge is
**suggested** (this is a data pipeline; the backend edge-weight and provenance math is the part most
worth an adversarial pass). Suggested, not wired — the review call is yours.
