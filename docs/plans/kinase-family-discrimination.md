# Discriminating kinases within the same family

Separate same-family kinases by the cell type they are attributed to. When MEA substrate overlap
makes two family members inseparable on activity, cell-type attribution is the discriminator: family
members frequently act in different cell types even when their motifs are near-identical.

Independent of `kinase-regulation-network.md` — that plan's concordance overlay is a sibling
consumer of the same sidechain view, not a prerequisite.

## What already exists

Per-kinase cell-type attribution is computed per cohort and reaches
`kinase_node_hits.parquet` — `FINAL_COLS` carries `owning_cluster`, `celltype_match`, and
`celltype_match_rank` (rank 1–3, `<NA>` when unmatched):

- song → `annotate_celltype_match_song`, from `kinase_hypothesis_table`
- 5xFAD → `annotate_celltype_match_fivexfad`, from `celltype_mea`

## The gap

`load_motif_edges` aggregates to `BOOL_OR(celltype_match)`. The boolean survives all the way to the
payload (`_INCYTR_SIDECHAIN_TERMINAL_COLUMNS`) — **the cell-type identity does not.** `owning_cluster`
and `celltype_match_rank` are dropped at the edge model.

A boolean cannot discriminate family members; "KIN A is microglial, KIN B is astrocytic" is the whole
claim. So the fix is to stop discarding the label.

`celltype_match` currently reaches the payload and no viewer code reads it — it is a dead column
today. This plan activates it alongside the label.

## Implementation

1. **Carry the label through `load_motif_edges`.** Add a second window mirroring the existing
   `nes_rank` idiom — partition by the same key, order by `celltype_match_rank NULLS LAST`, then
   `MAX(CASE WHEN ct_rank = 1 THEN owning_cluster END)`. Rank-1 wins; unmatched rows have a NULL
   `owning_cluster` by construction, so `NULLS LAST` picks a matched row whenever one exists.
   Deterministic on ties, consistent with how `signed_nes`/`n_sites` are already aligned.
2. Thread `owning_cluster` (and `celltype_match_rank`, which gives attribution confidence) through
   `build_terminal_map`'s output columns.
3. **Append both to `_INCYTR_SIDECHAIN_TERMINAL_COLUMNS` in the same pass** — the exact-tuple schema
   guard fails the shard build otherwise. That is the intended behaviour, not an obstacle.
4. Re-run `kinase_kinase_edges` per cohort. **The bridge does not need to re-run** — the columns are
   already in the parquet, so none of the multi-GB bridge cost applies.
5. Viewer: surface cell type on kinase nodes and in `_isNodeRelationTable`, so two same-family
   kinases in one fan are visibly attributed to different cell types.

Cell type is a kinase property carried on terminal edges. Kinases drawn only via PSP chain edges
have no motif hit and therefore no attribution — render as unattributed, never as a guess.

## Acceptance

- `owning_cluster` and `celltype_match_rank` present on terminal edges through to the payload.
- Schema guard updated in the same commit as the backend column.
- A same-family kinase pair with differing attribution is visibly distinguishable in the viewer.
- Kinases without attribution are visually distinct from attributed ones.
- Test extends `tests/test_kinase_sidechain_weighting.py` — rank-1 selection, including the
  all-unmatched case.
