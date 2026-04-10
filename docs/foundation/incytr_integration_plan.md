# Incytr Integration Plan

Integration plan for using the alzheimers multimodal dataset (72-animal bulk phosphoproteomics + 28-animal paired snRNA-seq + cell-type attribution) within the Incytr signaling pathway framework. This document is a design review artifact, not a commitment to build.

## 1. Motivation

The alzheimers pipeline produces kinase activity scores (MEA NES) and cell-type attributions, but stops at "kinase X is active in cell type Y in disease condition Z." It does not answer **what signaling pathways those kinases participate in**, who the upstream sender is, or what downstream targets are affected.

Incytr infers scored, ranked L-R-EM-T (Ligand → Receptor → Effector Molecule → Target) signaling pathways from scRNA-seq + multi-omics data. It can integrate kinase-substrate evidence and phosphoproteomics fold changes into its pathway scoring.

The integration goal is to place the kinase activity findings into a signaling pathway context — identifying putative upstream signals and downstream targets for AD-dysregulated kinases, resolved by cell type.

**This is explicitly a hypothesis-generating step.** The outputs are candidate pathways for experimental validation, not confirmed mechanisms.

## 2. Available Data Inventory

### 2.1 Paired multimodal data (Song et al. cohort)

| Modality | Source | Samples | Resolution |
|----------|--------|---------|------------|
| Bulk phosphoproteomics | TMT IMAC | 72 animals (4 genotype × 3 timepoint × 2 sex) | 16,114 phosphosites |
| Bulk total proteome | TMT | 72 animals (same design) | Protein-level abundance |
| snRNA-seq | 10X Chromium | 28 animals (subset of 72) | 63,695 nuclei, 30,567 genes |

### 2.2 Derived products from the alzheimers pipeline

| Product | File | Description |
|---------|------|-------------|
| Stoichiometry matrix | `stoichiometry_matrix.csv` | log2(phospho) − log2(protein), 16,114 sites × 72 samples |
| Site-level OLS | `site_level_ols.csv` | Per-site LFC and FDR for 9 contrasts (3 disease × 3 timepoint) |
| MEA enrichment | `mea_stoichiometry.csv` | Kinase NES/FDR per contrast, with leading substrates |
| Cell-type attribution | `unified_attribution.csv` | 12,041 kinase × cell-type × contrast triplets with confidence |
| Cell-type evidence | `celltype_evidence_table.csv` | Static localization evidence per kinase × cell-type |
| Pseudobulk expression | `pseudobulk_cpm.csv` | Per-animal, per-subclass CPM (28 animals × 22 subclasses) |
| Within-cohort concordance | `song_concordance.csv` | Per-gene, per-cell-type, per-pathway LFC from snRNA OLS |
| Within-cohort specificity | `song_expression_specificity.csv` | Per-gene, per-subclass expression specificity |

### 2.3 External references

| Reference | Source | Use |
|-----------|--------|-----|
| IncytrDB (mouse) | exFINDER-DB, NicheNet, CellChatDB, NeuronChatDB | L-R-EM-T pathway enumeration |
| Kinase-library | Motif-based kinase-substrate predictions | Kinase-substrate reference (kldata) |
| SEA-AD | Allen Institute MTG snRNA-seq | Cross-species concordance |
| WMB atlas | Allen Whole Mouse Brain 10Xv3 | Expression specificity |

## 3. Three Integration Modes

The integration operates in three modes with increasing novelty. Modes A and B use Incytr as-is. Mode C requires constructing a new pathway database but no code changes.

### Mode A: Intercellular signaling (sender ≠ receiver)

**Question:** Between two cell types in the AD brain, which L-R-EM-T signaling pathways are dysregulated, and do kinase activity findings support them?

**Data flow:**

```
Song snRNA-seq (28 animals, 22 subclasses)
  → Incytr expression input (genes × cells, WT vs AD condition)
  → pathway_inference() with IncytrDB mouse
  → Cal_SigProb() scores pathways by co-expression
  → Integr_multiomics() overlays phospho fold changes (see §4 for how)
  → Integr_kinase_enrichment() with cell-type-filtered MEA results
  → Cal_PDS() → Permutation_test() → Export_results()
```

**Cell-type pairs of interest:** The 22 available subclasses yield 22 × 21 = 462 ordered sender-receiver pairs. Prioritize based on biological relevance to AD:
- Microglia-PVM → L5 IT, L2/3 IT (neuroinflammatory signaling to vulnerable neurons)
- Astrocyte → L5 IT, L5 ET (metabolic support and reactive astrogliosis)
- Oligodendrocyte → L5 IT (myelin-related signaling)
- L5 IT → Microglia-PVM (damage signals, "find-me" and "eat-me" ligands)
- Endothelial → Microglia-PVM (vascular-immune crosstalk)

**Condition design:** Incytr supports exactly 2 conditions. The alzheimers design has 4 genotypes × 3 timepoints. Options:
- WT vs App (amyloid model, single timepoint)
- WT vs Tau (tau model, single timepoint)
- WT vs ApTt (combined model, single timepoint)
- Pool timepoints per genotype for more cells per condition (loses temporal resolution)

### Mode B: Autocrine signaling (sender == receiver)

**Question:** Within a single cell type, which autocrine L-R-EM-T loops are dysregulated in AD?

**Mechanism:** Set `sender = receiver = "Microglia-PVM"` (or any subclass). Incytr's code has no hard block on this — `pathway_inference()` filters genes by sender/receiver pools, which become identical. `Cal_SigProb()` uses the same cell type's expression for all 4 nodes. The pathway represents a cell secreting a ligand that binds its own receptor, triggering an intracellular cascade.

**Biological relevance:** Autocrine signaling is well-documented in AD-relevant cell types:
- Microglial TNF-α/IL-1β autocrine amplification loops
- Astrocyte CXCL10/CXCR3 self-stimulation
- Neuronal BDNF/TrkB autocrine survival signaling
- Oligodendrocyte FGF/FGFR self-renewal

**Data flow:** Identical to Mode A, but sender == receiver. All kinase and phospho evidence is filtered to the single cell type.

### Mode C: Intracellular kinase cascades (custom DB)

**Question:** Within a cell type, which kinase → substrate → downstream target cascades are active in AD, independent of any ligand-receptor initiation?

**Key insight:** Incytr's `pathway_inference()` joins three generic `from → to` tables. It does not enforce that Layer 1 is ligand-receptor — that semantics comes from the database content. A custom intracellular cascade database would work with zero code changes.

**Proposed intracellular DB structure:**

| Layer | Intercellular (IncytrDB) | Intracellular (new) | Source |
|-------|--------------------------|---------------------|--------|
| Layer 1 | Ligand → Receptor | Stimulus → Kinase | SIGNOR, OmniPath, PhosphoSitePlus |
| Layer 2 | Receptor → EM | Kinase → Substrate | Kinase-library (motif predictions) |
| Layer 3 | EM → Target | Substrate → Downstream effector | Reactome, SIGNOR, TF-target DBs |

**Biological semantics of the 4-node chain:**

```
Stimulus (receptor, stress signal, upstream kinase)
  → Kinase (activated by stimulus)
    → Substrate (phosphorylated by kinase)
      → Target (downstream gene regulated by phospho-substrate)
```

**Scoring model — expression enumerates, phospho ranks:**

The expression-based `Cal_SigProb()` computes `Hill(Stimulus × Kinase) × Hill(Kinase × Substrate) × Hill(Substrate × Target)` — co-expression within one cell type is a necessary condition for an intracellular cascade. However, co-expression has a different biological meaning for intracellular cascades than for intercellular signaling:

- **Intercellular (Modes A/B):** Co-expression of ligand and receptor is a proxy for *signaling capacity*. Expression magnitude is informative — higher expression generally means more ligand available to bind.
- **Intracellular (Mode C):** Co-expression of kinase and substrate is closer to a *precondition* than a readout of activity. Most kinase-substrate pairs in the same cell are constitutively co-expressed; the regulatory logic is post-translational (phosphorylation, localization, conformational activation). The Hill-function product provides less discrimination because most plausible cascades will pass the co-expression test.

**Consequence for Mode C results:** The expression layer enumerates biologically plausible cascades (is the kinase expressed in this cell type?), but the phospho overlay via `Cal_PDS()` does the actual biological work of distinguishing active from inactive cascades. Mode C results should be reported with **PDS (including phospho overlay) as the primary ranking**, not the expression-only pathway score.

**Pre-specified empirical validation:** Compute the Spearman correlation between expression-based ranking and phospho-overlay ranking for each mode. If Mode A shows high correlation (phospho confirms transcriptomics) but Mode C shows low correlation (phospho substantially reranks), this empirically validates the argument that Mode C's biological signal lives in the phospho layer. Report this as a methods figure.

**What the phosphoproteomics uniquely contributes to Mode C:**
- `site_level_ols.csv` provides direct evidence for the Kinase → Substrate edge (stoichiometry LFC at the substrate's phosphosite)
- `mea_stoichiometry.csv` provides aggregate evidence for each kinase's activity
- This is no longer a "supplement" to transcriptomic scoring — it is the primary evidence layer for the middle of the cascade

**No code changes required:** The existing `Cal_SigProb()` handles enumeration, and the existing PDS framework (`Cal_PDS()` in `evaluation.R`) handles the phospho overlay. The shift is in how the results are interpreted and reported, not in the scoring code.

**DB construction is a bioinformatics assembly task:**
1. Layer 1 (Stimulus → Kinase): Extract kinase activation relationships from SIGNOR or OmniPath. Format: `from` (upstream regulator gene), `to` (kinase gene), `source`.
2. Layer 2 (Kinase → Substrate): Extract from kinase-library motif predictions (already used in MEA). Format: `from` (kinase gene), `to` (substrate gene), `source`.
3. Layer 3 (Substrate → Target): This is the least well-curated layer. Primary sources: SIGNOR downstream effects, Reactome functional interactions. **Augmentation for coverage:** For substrates that are transcription factors, add TF-target relationships from DoRothEA (confidence A/B only) and TRRUST. For non-TF substrates, add high-confidence experimental edges from STRING (experimental score > 0.7). Report coverage statistics: what fraction of kinases from `mea_stoichiometry.csv` appear in each DB layer.

All three layers are filtered to mouse gene symbols and formatted as 3-element R list matching IncytrDB structure.

## 4. Bridging Bulk Phosphoproteomics to Per-Cell-Type Estimates

Incytr's `Integr_multiomics()` expects per-cell-type phospho fold changes (columns like `Microglia_ps`). The phosphoproteomics is bulk tissue. This section describes how to bridge that gap while carrying forward uncertainty.

### 4.1 The inference layer

The alzheimers pipeline already computes cell-type-level evidence for each kinase:
- **WMB expression specificity**: Is the kinase gene expressed in this cell type?
- **SEA-AD concordance**: Does the kinase's direction match human AD transcriptomic change in this cell type?
- **Song within-cohort concordance**: Does the paired snRNA-seq support this attribution?
- **Combined confidence**: High / Moderate / Low, with evidence basis classification

This inference layer does not measure per-cell-type phosphorylation directly. It estimates which cell types a bulk kinase signal most likely originates from, using convergent evidence from three independent sources. Only WMB specificity overlaps with Incytr's expression input; SEA-AD and Song concordance are independent evidence streams.

### 4.2 Tiered phospho integration

The bulk phospho signal enters Incytr differently depending on attribution confidence. This avoids both the information loss of treating phospho as a binary filter and the false precision of manufacturing synthetic per-cell-type phospho data.

**Three tiers:**

| Tier | Attribution confidence | How phospho enters Incytr | Rationale |
|------|----------------------|---------------------------|-----------|
| **Project** | High | Full bulk stoichiometry LFC assigned to top-ranked cell type (winner-take-all) | Convergent evidence from 3 independent sources supports a definitive cell-type origin |
| **Filter** | Moderate | Binary signal: kinase is flagged as active (MEA FDR < 0.25) but no LFC magnitude is passed | Evidence supports involvement but not strongly enough to justify projecting a specific fold change |
| **Exclude** | Low | No phospho signal enters Incytr | Insufficient evidence; prefer missing data over noisy data |

**Winner-take-all assignment (Project tier):** For high-confidence attributions, the full bulk LFC is assigned to the single top-ranked cell type rather than fractionally apportioned across cell types. This avoids an implicit linearity assumption — that a 60/40 expression-based attribution weight maps to a 60/40 phosphorylation split — for which there is no biological justification. A kinase could be expressed predominantly in microglia but phosphorylate substrates predominantly in astrocytes (via secreted intermediates, scaffolding dependencies, or threshold effects).

Winner-take-all is a Phase 1 simplification. For Phase 4 systematic analysis, revisit this by running the analysis twice — once assigning to the top cell type, once to the second — and reporting which pathways are robust to that choice.

**Implementation:** The tier assignment is derived from `unified_attribution.csv`:
- `combined_confidence == "high"` → Project tier: pass `bulk_stoich_lfc[site]` as `{top_celltype}_ps`
- `combined_confidence == "moderate"` → Filter tier: pass a binary indicator (1/0) to `Integr_kinase_enrichment()` but not to `Integr_multiomics()`
- `combined_confidence == "low"` → Exclude: omit from all phospho inputs

### 4.3 Discordance detection and reporting

When expression-based pathway scores and phospho evidence point in different directions, the discordance is a biological finding (dissociation between transcriptional and post-translational regulation), not noise to be smoothed over by PDS averaging.

**Three discordance scenarios:**

**Scenario A — Expression up, phospho down:** The pathway's transcriptional program is active (high `Cal_SigProb()` score) but the kinase's substrates are dephosphorylated in disease (negative NES). Possible biology: the pathway is being shut down post-translationally while the transcriptional program persists (regulatory lag), or the cell is compensatorily upregulating transcription because post-translational activity has declined.

**Scenario B — Expression low, phospho up:** The pathway components are weakly co-expressed (low `Cal_SigProb()` score) but the kinase shows strong substrate phosphorylation. Possible biology: the pathway operates in a different cell type than expected, or very low transcript levels are sufficient for a functional cascade (common for signaling molecules). **These pathways are filtered out during Incytr's expression-based enumeration** — they never reach PDS scoring. Detecting them requires a separate lookup (see §4.4).

**Scenario C — Direction agreement, magnitude disagreement:** Expression and phospho agree on direction but not magnitude. The PDS weighting determines which dominates, sensitive to weight parameters in ways that are hard to audit.

**Discordance flag definition:** A pathway is flagged as discordant when:
1. Expression-based pathway score is in the top quartile of all enumerated pathways, AND
2. The kinase phospho evidence is in the opposite direction, AND
3. The kinase reaches at least nominal MEA significance (FDR < 0.25) — this prevents flagging pathways where the phospho signal is absent rather than contradictory.

**Three-group reporting:** Results are reported in three groups rather than one combined ranking:

- **Concordant pathways:** Expression and phospho agree. Report PDS as the combined score. Highest-confidence candidates.
- **Discordant (expression up, phospho down):** Report expression score and phospho evidence separately, not combined. Flag as candidate post-translational regulatory events. Narrative: "transcriptionally poised but enzymatically inactive."
- **Discordant (expression low, phospho up):** From Scenario B lookup (§4.4). Lower confidence but potentially most novel — invisible to transcriptomics-only methods. Report with cell-type attribution caveats.

For concordant pathways, PDS combination is appropriate because both evidence streams estimate the same latent variable (pathway activity). For discordant pathways, PDS combination is uninterpretable — the evidence streams are detecting a regulatory dissociation, and any combined score obscures the finding.

### 4.4 Scenario B lookup: expression-dark, phospho-bright pathways

Pathways with low expression scores are filtered out during `Cal_SigProb()` and never reach PDS overlay. To detect Scenario B pathways:

1. Take all kinases with strong MEA evidence (FDR < 0.25) attributed at high confidence to a specific cell type.
2. For each kinase, query IncytrDB (or the Mode C intracellular DB) for pathways containing that kinase as an EM or Target node.
3. Check whether those pathways were absent from Incytr's expression-based output (i.e., filtered by low `Cal_SigProb()`).
4. Report these as Scenario B candidates with the kinase's MEA evidence, the pathway structure from the DB, and the cell-type attribution.

This is a targeted gene-list query against the pathway database, not a full Incytr rerun. It should be run as a post-processing step after each mode's standard pipeline.

### 4.5 Sensitivity analysis

Run the full Incytr pipeline with and without the phospho/kinase layers. This is not an afterthought — it is a core deliverable that enables:

1. **Marginal value assessment**: What fraction of top-20 pathways change rank by >10 positions when phospho is added? This quantifies whether the phosphoproteomics integration matters for each mode.
2. **Discordance detection**: The expression-only run provides the baseline for computing discordance flags (§4.3).
3. **Robustness check**: Pathways significant on transcriptomics alone and reinforced by phospho are more trustworthy than those that depend entirely on attributed phospho signal.

### 4.6 Limitations

- The attribution is correlative, not causal. A kinase attributed to microglia based on expression specificity and transcriptomic concordance may actually be active in a different cell type that happens to co-express the kinase gene.
- Bulk phosphoproteomics cannot distinguish autocrine from paracrine kinase activity. A kinase active in microglia could be phosphorylating substrates in neurons via secreted factors.
- The temporal resolution (2mo, 4mo, 6mo) is coarse. Kinase activity dynamics within a timepoint are not captured.
- Winner-take-all assignment discards information about cases where two cell types have similar attribution scores. High-confidence tier mitigates this (the confidence definition requires a clear winner), but near-ties can occur.

These limitations are inherent to the data design, not to the integration approach. Labeling this step as hypothesis-generating is essential.

## 5. Adapter Requirements

### 5.1 snRNA-seq → Incytr expression input

**Task:** Convert the Song 170_gex_celltypes_00.h5ad (or pseudobulk) into Incytr's expected format.

**Input:** AnnData h5ad with 63,695 nuclei × 30,567 genes, Allen Cell Type Mapper annotations (210 subclass labels mapped to 22 SEA-AD subclasses), experimental metadata (genotype, timepoint, sex, animal_id).

**Output:** Normalized expression matrix (genes × cells) + metadata data frame with columns: cell barcode (rownames), cluster label (one of 22 subclasses), condition label (e.g., "WT" or "App").

**Considerations:**
- Incytr expects 2 conditions. The Song data has 4 genotypes × 3 timepoints. Must subset to a specific comparison per run.
- Males-only filtering should match the primary analysis mode.
- Cell counts per subclass vary widely (Astrocyte: ~90/animal vs Lamp5 Lhx6: ~5/animal). Rare cell types may have insufficient cells for reliable expression estimates.
- Format conversion: h5ad (Python/AnnData) → sparse matrix (R/Matrix). Use `anndata` R package or export to CSV/MTX intermediate.

### 5.2 MEA results → kl_output

**Task:** Reshape `mea_stoichiometry.csv` into Incytr's `kl_output` format, pre-filtered by cell-type attribution.

**Input:** `mea_stoichiometry.csv` (kinase, NES, FDR, leading substrates, contrast) + `unified_attribution.csv` (kinase, cell_type, contrast, combined_confidence).

**Output per cell-type pair:** Data frame with columns:
- `kinase`: kinase gene symbol
- `substrate`: individual substrate gene symbol (parsed from leading substrates)
- `score`: NES value
- `padj`: FDR value

**Pre-filtering logic:**
1. Select contrast matching the Incytr condition comparison (e.g., App_4mo for WT vs App at 4 months)
2. For the target receiver cell type, keep only kinases attributed at moderate+ confidence
3. Parse semicolon-delimited leading substrates into individual rows
4. Map substrate motif identifiers to gene symbols

### 5.3 Kinase-library reference → kldata

**Task:** Extract kinase-substrate motif predictions used by the MEA step into Incytr's `kldata` format.

**Output:** Data frame with columns:
- `gene`: substrate gene symbol
- `site_pos`: phosphosite position (e.g., "S216")
- `motif.geneName`: kinase gene symbol

This is a static reference table, not disease-specific. It populates Incytr's structural kinase evidence (6 SiK cases).

### 5.4 Bulk phospho → tiered per-cell-type phospho input

**Task:** Bridge bulk phosphosite fold changes to cell-type-level Incytr inputs using the tiered integration scheme (§4.2).

**Input:** `site_level_ols.csv` (per-site LFC per contrast) + `unified_attribution.csv` (attribution confidence and top cell type) + `mea_stoichiometry.csv` (kinase NES/FDR).

**Output per cell-type pair:** Two data objects:

1. **Phospho fold changes** (for `Integr_multiomics()`): Data frame with columns `gene_symbol`, `{celltype}_ps`. Only populated for **Project-tier** (high-confidence) kinase-site pairs, using winner-take-all assignment: the full bulk `stoich_lfc` is assigned to the top-ranked cell type, all others receive NA.

2. **Kinase activity flags** (for `Integr_kinase_enrichment()`): For **Filter-tier** (moderate-confidence) kinases, pass the MEA result to `kl_output` (adapter 5.2) but do not pass site-level fold changes to `Integr_multiomics()`.

**Exclusion:** Low-confidence attributions contribute nothing — no phospho fold changes, no kinase activity flags.

**Post-processing:** After Incytr runs, compute discordance flags (§4.3) and Scenario B lookup (§4.4).

### 5.5 Intracellular cascade DB (Mode C only)

**Task:** Assemble a 3-layer intracellular signaling database in IncytrDB format.

**Output:** R list of 3 data frames, each with `from`, `to`, `source` columns:
- Layer 1: Stimulus → Kinase (from SIGNOR/OmniPath)
- Layer 2: Kinase → Substrate (from kinase-library)
- Layer 3: Substrate → Target (from SIGNOR/Reactome)

**Scope:** Mouse gene symbols. Filtered to genes present in the Song snRNA-seq.

## 6. Execution Plan

### Condition selection rationale

The proof of concept uses **WT vs App at 4 months** (males only). This is an early amyloid timepoint — biologically cleaner than late-stage combined pathology (ApTt 6mo), with well-characterized kinase biology (GSK3β, MAPK/ERK in amyloid processing) that enables sanity-checking top-ranked pathways against literature. Although later timepoints may have richer kinase signal, interpretability beats statistical power for a proof of concept.

Pre-specified follow-up: after pipeline validation, run the same analysis on App 6mo and ApTt 6mo to ask whether pathways identified early are amplified or replaced at later disease stages.

### Subclass power tiers for permutation testing

Cell counts per subclass per condition vary widely (Astrocyte ~90 cells/animal vs Lamp5 Lhx6 ~5 cells/animal). Incytr's permutation test (shuffling cell labels) requires sufficient cells for stable null distributions. Subclasses are assigned to power tiers:

| Tier | Cells per condition | Inference mode | Reporting |
|------|-------------------|----------------|-----------|
| **Tier 1** | >100 | Standard cell-level permutation | Primary results with p-values |
| **Tier 2** | 30–100 | Cell-level permutation + stability metric | Report p-value CV across 5 random seeds; flag pathways with CV > 0.5 as unstable |
| **Tier 3** | <30 | No permutation-based inference | Report pathway scores and phospho evidence only, no p-values. Hypothesis-generating only |

**Calibration:** The thresholds (100, 30) should be empirically validated before Phase 1 analysis. Permute labels within WT animals, run Incytr, characterize p-value stability as a function of cell count per subclass. Adjust thresholds based on the observed stability curve. Report calibration results as a methods figure.

### Phase 1: Intercellular (Mode A) — proof of concept

**Goal:** Run Incytr on Song snRNA-seq for one high-priority cell-type pair with full multi-omics integration. Validate the adapter pipeline and quantify the marginal value of phosphoproteomics.

**Condition:** WT vs App, 4mo, males only.

1. Build adapter 5.1 (snRNA → Incytr format) for WT vs App at 4mo
2. Build adapter 5.3 (kinase-library → kldata)
3. Run Incytr standard pipeline: pathway_inference → Cal_SigProb → Pathway_evaluation → Permutation_test
4. **Record expression-only pathway rankings** (baseline for sensitivity analysis and discordance detection)
5. Build adapter 5.2 (MEA → kl_output) for Microglia-PVM as receiver
6. Run Integr_kinase_enrichment() with cell-type-filtered MEA results
7. Build adapter 5.4 (bulk phospho → tiered per-cell-type input) for the same pair
8. Run Integr_multiomics() with tiered phospho input (§4.2), recompute PDS
9. Compute discordance flags (§4.3) and Scenario B lookup (§4.4)
10. **Compute expression-vs-phospho ranking correlation** (Spearman ρ between expression-only rank and PDS rank)
11. Review outputs: do the top-ranked concordant pathways align with known AD neuroinflammatory biology?

**Phase 1 decision gate:** Before proceeding to Phase 2, quantify: *what fraction of the top-20 pathways change rank by more than 10 positions when phospho evidence is added?*

- If substantial reranking occurs → the phospho integration adds value for intercellular signaling. Proceed with phospho integration in all subsequent phases.
- If minimal reranking occurs → the phospho layer is largely redundant for Mode A intercellular signaling. This is a legitimate and publishable finding. Proceed to Phase 2 with the expectation that autocrine mode (same-cell kinase and substrate) may show stronger phospho contribution.

**Deliverables:**
- Ranked pathway table for Microglia-PVM → L5 IT (three-group reporting: concordant, discordant type A, discordant type B)
- Sensitivity comparison (with vs without phospho/kinase layers)
- Expression-vs-phospho ranking correlation (baseline for cross-mode comparison)
- Permutation stability calibration for Microglia-PVM and L5 IT subclasses

### Phase 2: Autocrine (Mode B) — extend to within-cell-type

**Goal:** Run sender == receiver for key AD cell types. Assess whether autocrine mode captures distinct biology and whether phospho signal is stronger when kinase and substrate are in the same cell.

1. Confirm no runtime errors with sender == receiver (expected: works, but verify)
2. Run for Microglia-PVM, Astrocyte, L5 IT, Oligodendrocyte (WT vs App, 4mo)
3. Record expression-only rankings, then run with phospho overlay
4. Compute expression-vs-phospho ranking correlation for each cell type
5. Assess whether autocrine pathways overlap with intercellular findings (same kinases, different pathway context)
6. Compare phospho reranking magnitude in Mode B vs Mode A — is the phospho contribution stronger when kinase and substrate are in the same cell?

**Deliverables:** Autocrine pathway tables (three-group reporting) for 4 cell types, cross-mode ranking correlation comparison.

### Phase 3: Intracellular cascades (Mode C) — new DB construction

**Goal:** Build and validate an intracellular cascade database. This is the most novel mode and the one where phosphoproteomics contributes most directly.

1. Assemble Layer 1 (Stimulus → Kinase) from SIGNOR or OmniPath — assess coverage of AD-relevant kinases
2. Assemble Layer 2 (Kinase → Substrate) from kinase-library — this is well-characterized
3. Assemble Layer 3 (Substrate → Target) — primary: SIGNOR, Reactome. Augment with DoRothEA/TRRUST for TF substrates, STRING experimental edges for non-TF substrates. Report per-layer coverage statistics
4. Filter to mouse gene symbols present in Song snRNA-seq
5. Run Incytr with intracellular DB, sender == receiver, for key cell types
6. **Report with PDS (phospho overlay) as primary ranking**, not expression-only score (see §3 Mode C scoring rationale)
7. Compute expression-vs-phospho ranking correlation — expect low correlation (phospho substantially reranks), validating that Mode C's signal lives in the phospho layer
8. Validate: do the top cascades align with known AD kinase biology (GSK3β, MAPK, CDK5, DYRK1A)?

**Deliverable:** Intracellular cascade DB with coverage report, pathway tables (PDS-ranked, three-group reporting), cross-mode ranking correlation figure (Mode A vs B vs C).

### Phase 4: Systematic analysis

**Goal:** Run all three modes across prioritized cell-type pairs and conditions. Identify convergent hypotheses.

1. Extend condition comparisons: App 6mo, ApTt 4mo, ApTt 6mo (pre-specified progression analysis from proof-of-concept condition)
2. Define the full analysis matrix: cell-type pairs × conditions × modes
3. Implement batch runner
4. **Convergent evidence ranking**: Rank final hypotheses by the number of independent evidence streams supporting them (transcriptomic scoring, bulk phospho, cell-type attribution, cross-mode support) rather than by any single mode's p-value. This is a convergent evidence framework, not a multiple testing correction.
5. Report per-mode results in full (supplementary), but focus the narrative on convergent hits
6. Assemble final hypothesis table integrating intercellular, autocrine, and intracellular findings

## 7. Resolved Design Decisions

These questions were raised in the initial plan and resolved through review:

1. **Condition selection** → WT vs App at 4 months for proof of concept (interpretability over power). App 6mo and ApTt 6mo as pre-specified follow-ups. See §6 rationale.

2. **Phospho attribution method** → Tiered: winner-take-all projection for high-confidence, binary filter for moderate, exclude low. See §4.2.

3. **Multiple testing across modes** → Convergent evidence framework, not cross-mode correction. Rank hypotheses by number of independent evidence streams. Per-mode results in supplementary. See §6 Phase 4.

4. **R/Python boundary** → CSV/MTX intermediates (option a). Each adapter is a standalone Python script that reads pipeline outputs and writes Incytr-formatted files, plus a lightweight R wrapper that reads those files and calls Incytr functions. Intermediates stored in a versioned `integration/` directory with a manifest documenting input files, output files, column names, and data types.

5. **Discordance handling** → Three-group reporting (concordant, discordant type A, discordant type B). Discordant pathways reported with evidence separated, not combined. See §4.3–4.4.

6. **Permutation test power** → Three-tier subclass handling based on cell count per condition. Thresholds empirically calibrated. See §6 subclass power tiers.

## 8. Remaining Open Questions

1. **Mode C database scope**: SIGNOR and OmniPath have variable coverage across kinase families. Should we limit Mode C to kinase families with strong coverage (e.g., MAPK, GSK3, CDK) or attempt genome-wide? The DoRothEA/TRRUST/STRING augmentation for Layer 3 improves coverage, but Layer 1 (Stimulus → Kinase) remains the binding constraint.

2. **Subclass resolution for microglia**: The 22 subclasses lump homeostatic and disease-associated microglia (DAM), which have very different signaling profiles. If Mode B reveals heterogeneous signals in microglia, should we sub-cluster using the snRNA-seq (~90 microglia/animal, sufficient depth for DAM vs homeostatic split)?

3. **Timepoint pooling**: Pooling timepoints per genotype increases cells per condition but assumes signaling architecture is stable across disease progression. For the proof of concept we use a single timepoint (4mo). For Phase 4, should each timepoint be analyzed separately or pooled?

4. **Winner-take-all relaxation**: Phase 1 uses winner-take-all for simplicity. For Phase 4, how should we handle high-confidence attributions where the top two cell types have similar scores? Run twice (top vs second) and report robustness, or develop a principled threshold for when the margin is too narrow?
