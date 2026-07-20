# Kinase regulation concordance overlay

Add observed disease direction to the kinase→kinase (chain) edges of the sidechain graph, so a
selected kinase's regulatory neighbourhood reads as *concordant* or *discordant* rather than only
*connected*.

Scope note: this is an overlay on the shipped sidechain view
(`foundation/kinase_sidechain_incytr_graph.md`), not a new network or a new tab.

## What already ships

- Directed kinase→kinase edges — `interactome.csv`, PSP literature ∪ cohort motif co-enrichment,
  weighted and provenance-tagged.
- Click-a-kinase → isolate its closed neighbourhood, plus `_isNodeRelationTable` listing terminal
  targets (role, signed NES) then chain kinases (provenance, weight).
- Cell-type gating — the bridge's default filter is `celltype_match == True`, so drawn terminal
  edges are already restricted to kinases attributed to the node's cell type. No filter to build.

The only missing encoding is direction on the chain edges.

## Direction source: `signed_nes`

Chain edges are signaling edges — what propagates from A to B is a change in B's **activity**, not in
its abundance. Kinase abundance is a weak proxy for kinase activity (post-translational control:
phosphorylation, localisation, binding partners), which is why this project measures phosphosites at
all. `disease_lfc` (present in `kinase_node_hits.parquet` `FINAL_COLS`, dropped at the edge model) is
therefore the wrong quantity for this edge class and is not used.

`signed_nes` is also the basis of every existing sidechain encoding (node size, edge width, hue), so
a second differently-derived direction would let one kinase read enriched-red by NES and "down" by
abundance simultaneously.

NES is a **kinase-level** score (foundation spec), so direction is a property of the kinase node, not
of the chain edge. Per contrast, each kinase carries one sign; a chain edge's concordance is the
comparison of its two endpoints' signs.

Kinases present only via PSP (no motif arm, so no NES) have no direction. Render them as
indeterminate — never as concordant by default.

## Gate 0 — motif-similarity confound (go/no-go, do this first)

Kinases with similar motifs enrich together because they score against overlapping site sets, and
PSP kinase→kinase pairs are frequently close relatives. Concordance may therefore measure motif
overlap rather than regulation.

Check: stratify chain edges by motif similarity between endpoints and compare concordance rates
between the similar and dissimilar strata. If the rates do not separate, the overlay is measuring
substrate-preference overlap and **is not worth shipping** — stop here and record the result.

Report the stratified rates whatever the outcome; do not relax the comparison to reach a positive.

## Implementation (only if Gate 0 passes)

1. Derive per-kinase signed direction per contrast from the terminal edges already in the payload
   (`signed_nes`). No backend column, no re-run.
2. Classify each drawn chain edge: both endpoints positive or both negative → concordant; opposite →
   discordant; either endpoint without NES → indeterminate.
3. Encode on the chain edge, keeping the existing provenance colour intact — provenance and
   concordance are independent axes, as node size and edge width already are.
4. Add a concordance column to the chain-edge rows of `_isNodeRelationTable`, and the new encoding to
   the static legend.
5. Contrast selection follows whatever the surrounding pathways view already uses; the overlay does
   not introduce its own contrast control.

## Not in scope — signed reference hierarchy

A reference expectation ("A up *should* drive B down") needs per-site activating/inhibiting
annotation. `Kinase_Substrate_Dataset_count_07_2021.txt` records only that A phosphorylates B, with
reference counts — it carries no sign, because phosphorylation activates or inhibits depending on
the residue. PSP's `Regulatory_sites` file is not shipped in the `kinase_library` package and its
kinase→kinase coverage is thin regardless.

So this plan flags concordance; it cannot state that an observation contradicts an expectation. Do
not reintroduce the expectation layer without a signed source on disk.

## Acceptance

- Gate 0 stratified concordance rates recorded.
- Chain edges carry a concordance state; indeterminate is visually distinct from concordant.
- Provenance encoding unchanged.
- Legend covers the new state.
- Test extends `tests/test_kinase_sidechain_weighting.py` — sign-pair → concordance classification,
  including the missing-NES case.
