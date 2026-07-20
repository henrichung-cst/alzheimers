# T-cell per-cell labeling workspace

The current analysis is [the per-cell labeling report](tcell_state_labeling_evidence_percell.html).
Each cell is labeled from non-cycle RNA marker modules and raw CD4/CD8 CITE-seq
counts. Native clusters provide lineage fallback and UMAP context, not state.

The analysis deliberately excludes cell-cycle genes, phase, cycle scores, and
`% dividing`. Every named state requires direct positive marker-module detection;
biologically justified negative markers resolve exact ties.
Cells without sufficient subtype evidence remain `CD4` or `CD8`; they are not
forced into a named state.

Run the complete labeling workflow with:

```bash
pixi run tcells-label
```

This command refreshes compact RNA/ADT inputs when needed, writes per-cell labels
and evidence summaries, generates native-UMAP figures, and renders the HTML report.

The original report remains preserved historical context at
`tcell_state_labeling_evidence_original.html`, but its reference-label audit is not part
of the current labeling runner.

The matching downstream Incytr root is `outputs/reports/incytr_pair_mode_tcells`.
`build_tcell_viewer.py` asserts this is the resolver default and fails the build
otherwise, so the viewer cannot drift off it.
