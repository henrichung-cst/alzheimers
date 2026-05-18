# Plan: pivot the attribution spine from levy19 → levy_t5

**Goal.** Rebuild the mouse Attribution pipeline so the row spine is the 31-cluster
levy_t5 spine end-to-end. Reference annotations (WMB-34 specificity, SEA-AD
supertype LFC, Song concordance) attach to each levy_t5 row via **direct**
crosswalks — no chained mappings through intermediate vocabularies.

The viewer's path-swaps are already done in `build_unified_viewer.py` and the
pair-mode runner; the upstream `unified_attribution.csv` is still keyed on
levy19, so the viewer's mouse Attribution tab would show 19 rows after the
swap (and the decomp join would partially miss). Fixing that is what this
plan covers.

## Scope

### In-scope (this plan executes)

1. **Repoint `config.CLUSTER_SPINE_FILE`** at the levy_t5 spine file.
   Update `CLUSTER_SPINE_DIR` paths and the `_load_cluster_spine` reader so
   `N_CELL_TYPES == 31`, derived thresholds (`SPECIFICITY_HIGH`,
   `SPECIFICITY_LOW`) recompute, and any "19" comments are corrected.
2. **Extend the WMB crosswalk** `cluster_to_wmb_class.csv` to cover all 31
   levy_t5 clusters (currently has the 19 levy19 entries). The 12 new entries
   are listed below. Direct mapping: each levy_t5 cluster → one parent
   WMB-34 class.
3. **Extend the SEA-AD crosswalk** `cluster_to_seaad_supertype.csv` to cover
   all 31 levy_t5 clusters (currently 19). The 12 new clusters each get one
   or more SEA-AD supertype assignments with weights (many-to-many supported
   by the existing weighted-mean LFC merge).
4. **Rerun `alz/kinase_attribute.py`** to regenerate `unified_attribution.csv`
   and `unified_attribution_full.csv` keyed on the 31 levy_t5 clusters.
5. **Rerun `alz/attribution_recovery.py`** to regenerate
   `celltype_evidence_table.csv` and `kinase_hypothesis_table.csv` on the
   new spine.
6. **Rebuild the viewer** so the mouse Attribution tab shows 31 levy_t5 rows
   per (kinase, contrast), with the decomp NES/FDR join trivially matching.

### Out-of-scope (separate work)

- Human Attribution sub-tab (Task 2). Will be handled in a separate file
  after Task 1 lands.
- Backfilling levy_t5 specificity into Song's within-cohort outputs — Song
  already keys on cluster_name directly, so this works as an identity join
  for clusters present in Song's snRNA data, and is null for clusters Song
  doesn't observe. That's correct behavior (real missingness, not a join
  bug), so no change needed.

## The 12 new crosswalk entries

These levy_t5 clusters are not present in the current levy19 spine and need
both WMB-class and SEA-AD supertype assignments:

| levy_t5 cluster | proposed WMB class | proposed SEA-AD supertype(s) |
|---|---|---|
| Basal-Ganglia-GABAergic-Neurons | 09 CNU-LGE GABA | (no SEA-AD analog — MTG is cortex; leave empty) |
| Cholinergic-Neurons | 13 CNU-HYa GABA-Glut | (no MTG analog — leave empty) |
| Choroid-Plexus-Epithelial-Cells | 34 Vascular | (no MTG analog — leave empty) |
| Ependymal-cell | 33 Astro-Epen | (no MTG analog — leave empty) |
| Excitatory-neurons-Cajal-Retzius-cells-layer-I-Reelin | 02 NP-CT-L6b Glut | L6b_* supertypes (weighted) |
| GABAergic inhibitory interneurons | 06 CTX-CGE GABA | LAMP5/VIP/SST/PVALB supertypes (broad, equal weight) |
| GABAergic-inhibitory-interneurons-Dlx6os1-Erbb4 | 07 CTX-MGE GABA | SST_* + PVALB_* supertypes |
| GABAergic-inhibitory-interneurons-VIP-positive | 06 CTX-CGE GABA | VIP_* supertypes |
| Glutamatergic-excitatory-neurons-Cortical-layer-2-4-pyramidal-neurons | 01 IT-ET Glut | L2/3 IT_* + L4 IT_* supertypes |
| Inhibitory-Neurons | 06 CTX-CGE GABA | LAMP5/VIP/SST/PVALB supertypes (broad, equal weight) |
| Ptprz1-protoplasmic-astrocytes | 33 Astro-Epen | Astro_* supertypes |
| Vascular-Leptomeningeal-Cells | 34 Vascular | VLMC + Pericyte_* supertypes |

These assignments are biological best-guesses from cluster names against the
Allen WMB-34 taxonomy and SEA-AD MTG supertype catalog. **Need user
confirmation before writing the CSVs** — incorrect mappings here will
silently bias the attribution. The mappings should ideally be sourced from
the Allen Cell Type Mapper or an existing taxonomy crosswalk if available;
otherwise these informed guesses are what I'd write.

Subcortical clusters (Basal-Ganglia, Cholinergic, Choroid-Plexus, Ependymal)
intentionally get empty SEA-AD entries because SEA-AD is MTG-only (cortical);
their `sea_ad_lfc` column will be legitimately null, not a join bug.

## Execution order

1. Confirm the 12 cluster mappings above (or get corrections).
2. Write the extended `cluster_to_wmb_class.csv` (31 rows) + extended
   `cluster_to_seaad_supertype.csv`.
3. Edit `alz/config.py` to repoint the spine file and update comments.
4. Run `python alz/kinase_attribute.py` (regenerates unified attribution
   on the new spine).
5. Run `python alz/attribution_recovery.py --run` (regenerates hypothesis
   tables).
6. Run `python alz/build_unified_viewer.py` (picks up the new attribution
   + the already-swapped levy_t5 decomp + pair-mode incytr).
7. Hard-refresh the viewer; confirm mouse Attribution tab shows 31 rows
   per (kinase, contrast) and decomp NES/FDR populate.

## Rollback

If a step fails partway, revert `config.py`'s `CLUSTER_SPINE_FILE` to the
levy19 path; leave the new crosswalk CSVs in place (additive — extra rows
don't affect the levy19 spine).
