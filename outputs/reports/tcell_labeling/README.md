# T-cell per-cell labeling workspace

The current analysis is [the per-cell labeling report](tcell_state_labeling_evidence_percell.html).
Each cell is labeled from non-cycle RNA marker modules and raw CD4/CD8 CITE-seq
counts. Native clusters provide lineage fallback and UMAP context, not state.

The analysis deliberately excludes cell-cycle genes, phase, cycle scores, and
`% dividing`. Positive and biologically justified negative markers are both used.
Cells without sufficient subtype evidence remain `CD4` or `CD8`; they are not
forced into a named state.

Run the complete labeling workflow with:

```bash
pixi run tcells-label
```

This command refreshes compact RNA/ADT inputs when needed, writes per-cell labels
and evidence summaries, generates native-UMAP figures, and renders the HTML report.

Matt's original report remains preserved historical context at
`tcell_state_labeling_evidence_matt.html`, but its reference-label audit is not part
of the current labeling runner.

The matching downstream Incytr root is
`outputs/reports/incytr_pair_mode_tcells_percell_posneg`. The viewer must remain on
the last complete run until this root finishes.
