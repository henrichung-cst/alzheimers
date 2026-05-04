# Why our verdict might differ from the Allen ctx+HPF Transcriptomics Explorer

Reviewers commonly cross-check our cell-type attributions in the Allen
*Mouse Whole Cortex + Hippocampus 10x* tool (ctx+HPF). When that tool's UMAP
suggests a different pattern than our verdict, the difference is almost
always one of the three reasons below — and **both views can be correct
simultaneously**.

## 1. ctx+HPF is a partial slice of the relevant tissue

ctx+HPF covers exactly two regions (cortex + hippocampal formation, ~1.1M
cells). The Song bulk proteomics this pipeline attributes is dissected from
a broader forebrain territory. The paired snRNA-seq from the **same animals**
tells us exactly what the dissection contained:

| Top subclasses in Song snRNA-seq | n nuclei | Region |
|---|---:|---|
| OB-in Frmd7 Gaba         | 7,403 | **Olfactory bulb** |
| Oligo NN                 | 6,394 | pan-CNS |
| Astro-TE NN              | 5,553 | telencephalon |
| L2/3 IT CTX Glut         | 3,379 | Cortex |
| L4/5 IT CTX Glut         | 3,179 | Cortex |
| STR D1 Gaba              | 2,263 | **Striatum** |
| L6 CT CTX Glut           | 2,064 | Cortex |
| STR D2 Gaba              | 1,782 | **Striatum** |
| DG Glut                  | 1,607 | Hippocampus |
| L2/3 IT PIR-ENTl Glut    | 1,131 | **Piriform/entorhinal** |
| L6 IT CTX Glut           | 1,091 | Cortex |

Striatum (D1/D2 medium spiny), olfactory bulb interneurons, and
piriform/entorhinal cells appear in significant numbers — none of which
exist in ctx+HPF. A kinase concentrated in any of those cell types is real
and attributable in our data, but cannot appear in ctx+HPF. To verify
against the same data the score is computed on, use the **ABC Atlas** link
in the viewer (whole brain, identical Allen WMB 10Xv3 source).

## 2. ctx+HPF and our reference use different cell-type taxonomies

ctx+HPF uses its own hierarchy with hundreds of fine clusters; we collapse
to **34 WMB classes** (the published Allen Whole Mouse Brain class taxonomy,
the level above the 338 subclasses ctx+HPF labels with). Hover-info on a
ctx+HPF cell shows that dataset's fine cluster name, not our class — so
direct visual comparison requires aggregating ctx+HPF clusters up to the
class level.

## 3. ctx+HPF shows per-cell intensity; our score shows relative concentration

ctx+HPF colors each cell by raw expression. A gene can be detected at
moderate intensity in many cells (broadly expressed) **and** concentrated in
one cell type relative to others (because the rest have even less).

Worked example — *Prkch*: our pipeline scores it as endothelial
(specificity 0.52, microglia second at 0.24). In ctx+HPF, Prkch lights up
scattered cells across the UMAP — and on hover those cells are
predominantly endothelial and microglial. Both views agree on the cell-type
call; ctx+HPF presents it as scatter because vasculature threads through
every cortical region anatomically, while our score reports it as
concentration.

## What to do when they disagree

- If our verdict says cell type X and ctx+HPF shows scatter throughout,
  hover on the lit cells. If they're predominantly cell type X (or a fine
  cluster within it), the views agree and ctx+HPF is just rendering
  "broadly distributed cell type" as "spread across the UMAP."
- If our verdict names a cell type that ctx+HPF does not contain (striatum,
  olfactory bulb, thalamus, cerebellum), ctx+HPF cannot validate or refute
  that call — verify in ABC Atlas instead.
- If **both** ABC Atlas and ctx+HPF disagree with our call, that is a real
  disagreement worth flagging.
