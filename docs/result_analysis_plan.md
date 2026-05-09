# Result Analysis Plan

This plan describes how to interpret the current Alzheimer's disease kinase attribution and factorial Incytr results. The goal is to move from generated outputs to biological claims in a disciplined order: first establish what evidence exists, then identify reproducible patterns, then decide which hypotheses are strong enough to highlight.

The analysis is hypothesis-generating. Results should be framed as convergent functional evidence for candidate mechanisms, not as proof of direct signaling, kinase activity in a single cell, or causal disease mechanism.

## 1. A Priori Assumptions From The Experimental Design

### Disease axes

The design separates three disease contexts:

- **App**: amyloid-driven model effect.
- **Tau**: tauopathy-driven model effect.
- **ApTt**: combined APP plus tau model effect.

The primary a priori assumption is that these contexts may produce related but non-identical molecular programs. App and Tau should not be treated as interchangeable AD labels. ApTt should not be assumed to equal App plus Tau; one of the explicit design questions is whether the double model is additive, sub-additive, antagonistic, or super-additive.

### Time

The design includes 2, 4, and 6 month timepoints. The a priori expectation is that disease biology may evolve over time, so temporal patterns are part of the signal rather than nuisance variation. Early signals may indicate initiating or compensatory responses; late signals may indicate progression, remodeling, or accumulated injury.

### Cell type

The analysis assumes that bulk phosphoproteomic kinase activity can be biologically sharpened by cell-type evidence from snRNA-seq and external references, but not perfectly localized. Cell-type attribution should be interpreted as a prioritized cellular context for a kinase or pathway, not direct measurement of phosphorylation in isolated cell types.

### Sender and receiver biology

The Incytr layer assumes that signaling hypotheses can be organized as sender ligand to receiver receptor to receiver-side effector molecule to target. For biological interpretation, the **receiver** usually carries the clearest cellular meaning because receptor, EM, target, pathway differential score, and kinase-substrate wiring are all interpreted inside that receiver context.

### Backbone biology

A pathway backbone is the receiver-side route:

`Receptor -> Effector Molecule -> Target`

Different sender ligands may converge onto the same backbone. More passing backbones for a receiver or contrast suggest broader modeled receiver-side signaling remodeling. Fewer passing backbones suggest a narrower candidate program. Counts alone do not tell us whether the biology is harmful, compensatory, causal, or secondary.

### Null tests

The pathway layer tests many possible backbones, so apparently interesting routes can appear by chance. A backbone is strongest when it passes both null tests in the same contrast:

- **Null 1, enrichment null**: disease-significant kinases are unusually concentrated near this backbone.
- **Null 2, wiring null**: the specific kinase-substrate-cell-type wiring is stronger than expected for that receiver, not just a generic kinase-rich background.

Passing both nulls means the integrated evidence is non-random under these tests. It does not prove physical signaling.

### Directionality

Kinase NES has sign, TPDS has sign, and kinase-pathway concordance has sign, but external kinase support is largely magnitude-based. This is deliberate because inhibitory phosphorylation can create biologically meaningful discordance. Interpretation should therefore separate:

- strength of evidence,
- direction of transcriptomic pathway change,
- direction of kinase enrichment,
- concordant versus discordant kinase-pathway relationships.

## 2. Inventory Of Result Layers

### Quality and preprocessing

Primary files:

- `outputs/reports/data_ingest/data_quality.json`
- `outputs/reports/data_ingest/sample_exclusions.csv`
- `outputs/reports/kinase_attribution/normalization_summary.json`
- `outputs/reports/kinase_attribution/stoichiometry_qc.csv`
- `outputs/reports/kinase_attribution/diagnostic_naive_vs_ols.png`

Interpretation purpose:

Use these results to establish whether downstream signals are interpretable. This is not a place to make biological claims. It should answer whether missingness, plex effects, exclusions, normalization, and OLS contrast construction are acceptable enough to support the later analyses.

### Kinase activity

Primary files:

- `outputs/reports/kinase_attribution/mea_stoichiometry.csv`
- `outputs/reports/kinase_attribution/mechanism_annotation.csv`
- `outputs/reports/attribution_recovery/kinase_activity_matrix.csv`
- `outputs/reports/attribution_recovery/kinase_hypothesis_table.csv`

Interpretation purpose:

Identify kinases with disease-associated phosphosite enrichment, their contrast specificity, temporal behavior, and whether apparent activity is retained after stoichiometry correction. This layer answers: "Which kinases look dysregulated, in which disease contexts, and with what direction?"

Current run anchors:

- 1,196 significant MEA rows are reported in `attribution_summary.json`.
- The kinase hypothesis table contains 12,041 attributed rows.
- High-confidence attribution is a minority of the output, so moderate-confidence findings should be interpreted cautiously unless supported by downstream convergence.

### Cell-type attribution

Primary files:

- `outputs/reports/kinase_attribution/unified_attribution.csv`
- `outputs/reports/kinase_attribution/unified_attribution_full.csv`
- `outputs/reports/attribution_recovery/celltype_evidence_table.csv`
- `outputs/reports/snrna_integration/song_concordance.csv`
- `outputs/reports/wmb_expression/wmb_kinase_expression.csv`

Interpretation purpose:

Assign candidate cell-type contexts to kinase activity by combining within-cohort snRNA-seq concordance, SEA-AD concordance, and whole mouse brain expression specificity. This layer answers: "Where is a kinase most plausibly acting?"

Interpretive standard:

Give more weight to kinases whose cell-type attribution is supported by multiple evidence streams and whose attributed cell type also appears in pathway/backbone results. Treat single-source or weak evidence as directional context rather than a finding.

### Factorial Incytr pathway scores

Primary files:

- `alz/integration/intermediates/factorial/all_pairs/recv_*.parquet`
- `alz/integration/intermediates/factorial/all_pairs/aggregation/hub_matrix_by_contrast.csv`
- `alz/integration/intermediates/factorial/all_pairs/aggregation/contrast_comparison.csv`
- `alz/integration/intermediates/factorial/all_pairs/aggregation/target_convergence_by_contrast.csv`
- `alz/integration/intermediates/factorial/all_pairs/aggregation/temporal_dynamics.csv`

Interpretation purpose:

Identify sender-receiver pathway changes across nine genotype-by-timepoint contrasts. This layer answers: "Which cell-cell signaling routes are changing, when, and in which receiver contexts?"

Current run anchors:

- ApTt_2mo has the largest number of significant pathway calls in the examination summary.
- Top mean absolute TPDS sender-receiver pairs are heavily Chandelier receiver oriented.
- Broad pathway counts should be interpreted with care because the model is animal-level and the snRNA-seq subset is small.

### Kinase-supported backbone aggregation

Primary files:

- `alz/integration/intermediates/factorial/all_pairs/aggregation/backbone_recurrence_by_contrast.csv`
- `alz/integration/intermediates/factorial/all_pairs/aggregation/backbone_permutation_pvalues_by_contrast.csv`
- `alz/integration/intermediates/factorial/all_pairs/aggregation/kinase_backbone_edges.parquet`
- `outputs/reports/kinase_backbone_edges_sig.parquet`

Interpretation purpose:

Prioritize receiver-side routes where transcriptomic pathway change and kinase evidence converge. This layer answers: "Which receptor-EM-target routes have non-random kinase support in a receiver cell type?"

Interpretive standard:

The strongest backbones should show:

- meaningful TPDS magnitude,
- one or more contrasts passing both nulls,
- repeated sender convergence or receiver recurrence,
- interpretable kinase edges,
- consistency with the kinase attribution layer.

### Additivity, temporal dynamics, centrality, and concordance

Primary files:

- `alz/integration/intermediates/factorial/all_pairs/aggregation/examination/additivity_summary.csv`
- `alz/integration/intermediates/factorial/all_pairs/aggregation/examination/additivity_by_pair_timepoint.csv`
- `alz/integration/intermediates/factorial/all_pairs/aggregation/examination/trajectory_classification.csv`
- `alz/integration/intermediates/factorial/all_pairs/aggregation/examination/celltype_centrality.csv`
- `alz/integration/intermediates/factorial/all_pairs/aggregation/examination/kinase_validation_summary.txt`
- `alz/integration/intermediates/factorial/all_pairs/aggregation/examination/figure_overview.png`

Interpretation purpose:

Convert high-dimensional pathway output into result themes:

- Does ApTt behave like App plus Tau?
- Which effects are early, late, monotonic, or peaked?
- Which cell types act as signaling senders or receivers?
- Do kinase and TPDS signs agree or disagree?

Current run anchors:

- Additivity summary shows substantial sub-additive and antagonistic calls, especially at 4mo and 6mo.
- Mean kinase-pathway coverage is 74.6%.
- Mean kinase-pathway concordance is 53.3%, close to chance overall but higher in selected contrasts such as Tau_2mo and ApTt_2mo.

### Unified viewer

Primary files:

- `outputs/reports/unified_viewer/index.html`
- `outputs/reports/unified_viewer.payload.json`
- `outputs/reports/unified_viewer/pipeline_overview.html`

Interpretation purpose:

Use the viewer for interactive triage and evidence auditing, not as the only analysis record. The viewer should help move from high-level patterns to specific kinases, receiver cell types, backbones, contrasts, and supporting edges.

## 3. Recommended Interpretation Order

### Step 1: Establish analysis validity

Questions:

- Which samples were excluded and why?
- Are missingness and plex effects acceptable after normalization?
- Does stoichiometry correction materially change kinase conclusions?
- Do OLS contrasts differ from naive group means in ways that justify the factorial model?

Output:

A short QC paragraph stating which caveats must travel with downstream interpretation.

### Step 2: Define the kinase-level result set

Questions:

- Which kinases are significant across the most contrasts?
- Which kinases have the largest absolute NES?
- Which kinases are App-specific, Tau-specific, ApTt-specific, shared, progressive, peaked, or mixed?
- Which kinase findings are abundance-driven, activity-driven, or retained in both raw and stoichiometry-corrected phospho?

Output:

A ranked kinase result table grouped by disease axis and trajectory, with a separate flag for mechanism annotation.

### Step 3: Attach cell-type context to kinases

Questions:

- For top kinases, what are the top attributed cell types?
- Are the attributions high-confidence or supported by multiple evidence streams?
- Do attributed cell types align with known AD-relevant biology?
- Are there cell types with broad kinase attribution but weak pathway support, or vice versa?

Output:

A kinase-by-cell-type interpretation matrix that distinguishes strong convergent assignments from weak candidates.

### Step 4: Read receiver-level pathway structure

Questions:

- Which receiver cell types show broad passing-backbone signal across contrasts?
- Which contrasts show widespread receiver remodeling?
- Are non-neuronal, vascular, glial, or neuronal receiver compartments dominant?
- Are results driven by a few broad receivers or many narrow receiver-specific events?

Output:

A receiver-first summary of major signaling-remodeling compartments.

### Step 5: Identify high-confidence backbones

Questions:

- Which receptor-EM-target backbones pass both nulls?
- In how many contrasts do they pass both nulls?
- Are they recurrent across many senders?
- Are they associated with meaningful TPDS magnitude?
- Which kinases and substrates support them?

Output:

A prioritized backbone table with columns for receiver, contrast count, passing contrasts, TPDS direction, kinase support, and biological interpretation.

### Step 6: Interpret sender-receiver structure

Questions:

- Are signals receiver-dominant, sender-dominant, or pair-specific?
- Are apparent hubs supported by repeated backbones or by many unrelated one-off routes?
- Do sender identities suggest plausible extracellular sources for receiver-side remodeling?

Output:

A sender-receiver network summary that separates broad cell-type hubs from specific candidate communication routes.

### Step 7: Interpret temporal patterns

Questions:

- Which backbones or receivers are early, late-onset, monotonic, or peaked?
- Do App, Tau, and ApTt show similar or divergent timing?
- Are early signals dominated by amyloid-related contrasts and late signals by tau-related contrasts, as expected, or does the data contradict that?

Output:

A temporal result narrative organized by disease axis and receiver cell type.

### Step 8: Interpret ApTt additivity

Questions:

- Does ApTt approximate App + Tau, or does it deviate?
- Are deviations mostly sub-additive, antagonistic, or super-additive?
- Which receivers and backbones drive the deviations?
- Are deviations stronger at specific timepoints?

Output:

An interaction-focused result section describing where combined APP plus tau biology departs from the additive expectation.

### Step 9: Separate concordant and discordant kinase-pathway evidence

Questions:

- Where does kinase NES direction agree with TPDS direction?
- Where is evidence discordant, suggesting possible inhibitory phosphorylation or compensatory response?
- Are top-ranked backbones enriched for concordant, discordant, or mixed evidence?

Output:

A kinase-pathway concordance summary that avoids treating all magnitude-based support as the same biological direction.

### Step 10: Synthesize candidate mechanisms

Questions:

- Which kinase-celltype-backbone-contrast combinations are supported across multiple layers?
- Which are strong but narrow context-specific hypotheses?
- Which are broad but mechanistically vague and require more caution?
- Which results are surprising relative to the a priori design expectations?

Output:

A final shortlist of candidate mechanisms, each stated in this format:

`In [contrast/timepoint], [kinase or kinase family] is inferred to be dysregulated and attributed to [cell type], converging with [receiver] [receptor-EM-target] backbone evidence, suggesting [plain-language biological hypothesis].`

## 4. Evidence Strength Tiers For Biological Claims

### Tier 1: Strong integrated hypothesis

Use when a claim has:

- significant kinase enrichment,
- credible cell-type attribution,
- receiver/pathway evidence in the same biological context,
- backbone passing both nulls,
- interpretable TPDS direction,
- temporal or contrast recurrence,
- no major QC or sensitivity red flags.

### Tier 2: Focused candidate

Use when a claim has strong evidence in one layer and partial support in another, or a single sharp contrast-specific result that passes both nulls but lacks recurrence.

### Tier 3: Exploratory lead

Use when a result is visually prominent or statistically suggestive but lacks cell-type confidence, dual-null support, recurrence, or directional clarity.

### Do not promote as a main result

Avoid main-text claims based only on:

- raw pathway abundance without null support,
- backbone counts without effect direction,
- kinase NES without cell-type attribution,
- cell-type attribution without kinase significance,
- ApTt effects without acknowledging limited snRNA-seq animal counts,
- magnitude-only kinase support without concordance stratification.

## 5. Expected Biological Readouts To Test

The result analysis should explicitly test these a priori expectations:

- App effects may be stronger in earlier or amyloid-relevant contexts.
- Tau effects may be stronger in later or tau-relevant contexts.
- ApTt may depart from the additive sum of App and Tau.
- Glial, vascular, and border-associated receiver compartments may carry broad remodeling, but neuronal subclass-specific signals may identify more selective routes.
- Receiver-side convergence is more interpretable than isolated sender-specific hits.
- Kinase activity and RNA pathway direction may not always agree, especially for inhibitory phosphorylation or compensatory responses.
- Findings that recur across contrast, receiver, sender, and kinase evidence layers deserve higher priority than findings that are extreme in only one metric.

## 6. Deliverables From The Result Analysis

The next analysis pass should produce:

- A one-page result map listing the major result layers and their top-level conclusions.
- A ranked kinase result table with trajectory and cell-type context.
- A receiver-first pathway/backbone table.
- A focused additivity table for ApTt deviations.
- A temporal dynamics summary by genotype and receiver.
- A concordance-stratified kinase-pathway table.
- A shortlist of Tier 1 and Tier 2 candidate mechanisms.
- A caveats paragraph tied directly to sample size, shared-animal dependence, attribution uncertainty, and null-test interpretation.

## 7. Positive-Control And Outside-Reference Validation Plan

The result interpretation should include a dedicated validation pass against public AD references. The purpose is not to prove that every novel result is true. The purpose is to test whether the pipeline recovers known AD-relevant biology more often than expected by chance, and to separate internally strong but novel hypotheses from results that also agree with external disease evidence.

### Validation principles

Each reference dataset should be assigned an independence label before testing:

- **Independent**: not used anywhere in this pipeline and not directly derived from the same animals or the same reference tables.
- **Semi-independent**: related to a source already used in the pipeline, but testing a different endpoint, region, modality, or summary.
- **Internal consistency**: useful for checking coherence, but not independent validation.

External agreement should be interpreted as one evidence layer, not as a replacement for the pipeline's own statistical controls. A result can be biologically meaningful without being in a known AD reference set, but main claims are stronger when internal evidence and outside evidence agree.

### Candidate reference datasets

#### AMP-AD / ROSMAP

Expected role:

- Broad human AD anchor for transcriptomic, proteomic, clinical, and pathology-associated signatures.
- Useful for testing whether top genes, kinase substrates, pathway targets, and cell-type themes align with human AD severity, cognition, Braak/tau, amyloid, and diagnosis.

Likely validation targets:

- Kinase genes and kinase substrates.
- Backbone targets.
- Receiver-cell pathway target sets.
- Disease-axis themes such as synaptic, glial, myelin, vascular, mitochondrial, and inflammatory modules.

Independence label:

- Usually **independent** if using ROSMAP-derived signatures not already used by the pipeline.
- **Semi-independent** if a harmonized AMP-AD resource overlaps with any external evidence already used for attribution.

Acquisition notes:

- Access may require AD Knowledge Portal / Synapse credentials and data-use acknowledgement.
- Raw assays may be large; first pass should prefer precomputed differential expression, differential protein, module, or pathology-association tables when available.

#### SEA-AD

Expected role:

- Cell-type and disease-progression anchor for human AD single-nucleus, multiome, and spatial data.
- Useful for testing whether receiver-cell and target-gene interpretations are plausible in human AD cell states.

Likely validation targets:

- Receiver cell-type enrichment.
- Backbone target disease trajectories.
- Cell-type-specific expression or differential expression of receptor, EM, and target genes.
- Human AD progression alignment by pathology severity.

Independence label:

- **Semi-independent** if using SEA-AD because SEA-AD already contributes to current attribution logic.
- Can be treated as closer to **independent** only for endpoints not used by the pipeline, such as spatial localization or held-out region/cell-state summaries.

Acquisition notes:

- Prefer processed cell-type differential or trajectory summaries rather than raw count matrices for the first validation pass.
- Harmonize SEA-AD taxonomy to the viewer's 22 receiver subclasses or to a coarser shared class level.

#### Human AD phosphoproteomics

Expected role:

- Most direct outside reference for kinase and phosphorylation biology.
- Useful for checking whether inferred kinase activities and substrate-level findings agree with human AD phosphoproteomic changes.

Likely validation targets:

- Kinase families.
- Kinase genes.
- Substrate genes and phosphosites.
- Pathway EM/target phosphosite support.

Candidate sources:

- Human cortex TMT proteome + IMAC phosphoproteome across control, asymptomatic AD, and symptomatic AD.
- ProteomeXchange PXD020087, "Quantitative phosphoproteomics uncovers dysregulated kinase networks in Alzheimer's disease."

Independence label:

- **Independent**, unless a given kinase prior or substrate catalog is shared with the analysis in a way that creates circularity.

Acquisition notes:

- Phosphosite coordinate harmonization will be the hardest part.
- First pass can validate at gene or kinase-family level before attempting exact site matching.
- If exact sites are tested, record protein accession mapping, residue numbering system, isoform assumptions, and species mapping decisions.

#### NeuroPro or other literature-derived AD proteome compendia

Expected role:

- Robust literature-derived protein-level positive-control set.
- Useful for testing whether pathway targets and kinase substrates overlap reproducible human AD protein changes.

Likely validation targets:

- Backbone target genes.
- Receptor/EM/target genes.
- Kinase substrates.
- Receiver-specific pathway gene sets.

Independence label:

- **Independent**, but not mechanistically specific. It validates broad AD protein relevance, not kinase wiring.

Acquisition notes:

- Prefer curated protein lists with direction, disease stage, and brain-region annotations when available.
- Separate early-stage, advanced-stage, plaque-associated, tangle-associated, and CAA-associated sets if the source provides them.

### Dataset acquisition and cleanup workflow

#### Step 1: Build a reference registry

Create a machine-readable registry, for example:

`data/external_references/reference_registry.yaml`

Recommended fields:

- `reference_id`
- `source_name`
- `source_url`
- `accession`
- `download_date`
- `license_or_data_use`
- `requires_credentials`
- `raw_path`
- `processed_path`
- `organism`
- `tissue_or_region`
- `modality`
- `disease_labels`
- `cell_type_labels`
- `pipeline_overlap`
- `independence_label`
- `intended_validation_tests`
- `citation`

This registry should be treated as part of the analysis record. Any result that uses an outside reference should point back to the registry entry.

#### Step 2: Acquire references reproducibly

Create one acquisition script per source family rather than one monolithic downloader:

- `alz/validation/acquire_amp_ad.py`
- `alz/validation/acquire_sea_ad.py`
- `alz/validation/acquire_phosphoproteomics.py`
- `alz/validation/acquire_neuropro.py`

The scripts should:

- download or locate source files,
- verify checksums when possible,
- write a small manifest of acquired files,
- avoid committing restricted or very large raw data,
- fail clearly when credentials are missing,
- never silently overwrite manually curated files.

For restricted datasets, the script can support a "local file registration" mode where the user downloads data manually and the pipeline records file paths, checksums, and metadata.

#### Step 3: Normalize identifiers

Create shared identifier maps:

- mouse gene symbol to human ortholog,
- Ensembl ID to gene symbol,
- UniProt accession to gene symbol,
- phosphosite protein/residue to gene-level fallback,
- SEA-AD cell taxonomy to viewer receiver taxonomy,
- broad class labels such as neuron, interneuron, astrocyte, microglia, oligodendrocyte, OPC, endothelial, VLMC.

Recommended outputs:

- `outputs/validation/id_maps/gene_ortholog_map.csv`
- `outputs/validation/id_maps/protein_gene_map.csv`
- `outputs/validation/id_maps/phosphosite_map.csv`
- `outputs/validation/id_maps/celltype_crosswalk.csv`

The first validation pass should support gene-level tests even if phosphosite-level harmonization is incomplete.

#### Step 4: Convert each reference into standard validation sets

Each cleaned reference should produce a small set of standardized tables:

- `reference_gene_sets.csv`: named AD gene/protein sets.
- `reference_ranked_genes.csv`: ranked disease-associated genes or proteins.
- `reference_celltype_scores.csv`: cell-type disease scores where available.
- `reference_phosphosite_scores.csv`: phosphosite-level disease scores where available.
- `reference_metadata.json`: provenance, filtering, and caveats.

Standard columns for gene sets:

- `reference_id`
- `set_id`
- `set_label`
- `gene_symbol`
- `direction`
- `evidence_type`
- `disease_stage`
- `brain_region`
- `cell_type`
- `score`
- `q_value`

Standard columns for ranked genes:

- `reference_id`
- `gene_symbol`
- `rank_metric`
- `rank_value`
- `direction`
- `disease_stage`
- `brain_region`
- `cell_type`

#### Step 5: Create pipeline result sets to validate

Export standardized result sets from the current analysis:

- top kinases by absolute NES and FDR,
- high-confidence kinase-cell-type attributions,
- top backbone targets,
- passing-both-null backbones,
- receiver-specific target sets,
- sender-receiver hub genes,
- additivity-deviation backbones,
- temporally classified backbone sets.

Recommended output:

`outputs/validation/pipeline_result_sets/`

Each exported set should record the exact filter used, such as top-N, FDR threshold, passing-both-null requirement, receiver filter, or contrast filter.

### Statistical tests

#### Overlap tests

Use Fisher exact or hypergeometric tests for binary overlap:

- top kinases versus known AD kinase sets,
- backbone targets versus AD protein sets,
- receiver-specific targets versus cell-type AD gene sets.

Report:

- overlap count,
- expected overlap,
- odds ratio or enrichment ratio,
- p-value,
- FDR across all tested set pairs.

#### Rank enrichment tests

Use GSEA-style or Mann-Whitney rank tests for full ranked lists:

- genes ranked by pathway/backbone support,
- kinases ranked by absolute NES or attribution-supported NES,
- targets ranked by passing contrast count or recurrence.

Report:

- normalized enrichment score or rank-biserial effect size,
- p-value,
- FDR,
- leading-edge genes where applicable.

#### Matched permutation tests

Use matched permutations when degree, expression, or pathway membership could inflate overlap:

- Match genes by expression detectability.
- Match substrates by kinase-degree or substrate promiscuity.
- Match backbones by receiver cell type and number of kinase edges.
- Match pathways by number of genes and number of tested contrasts.

This should be the preferred test for main claims because it better respects the structure of the generated result space.

#### Classification-style recovery

If a reference provides binary positive labels, compute:

- AUROC,
- average precision,
- precision at K,
- recall at K.

This is useful for asking whether pipeline scores recover known AD genes or kinases better than random ranking.

### Interpretation standards for validation results

Validation results should be reported as evidence-strength modifiers:

- **External support present**: internal Tier 1 or Tier 2 result also enriched or ranked highly in independent AD reference.
- **Known biology recovered**: positive-control gene sets are enriched above chance, supporting pipeline calibration.
- **Novel but internally strong**: result has strong pipeline evidence but no external positive-control match.
- **Externally plausible but internally weak**: known AD gene or pathway appears but lacks internal null, attribution, or direction support.
- **Potential circularity**: agreement uses a semi-independent or internal-consistency reference and should not be presented as external validation.

### Implementation milestones

1. Create `data/external_references/` and the reference registry.
2. Decide which references are first-pass targets based on access burden and independence.
3. Implement acquisition or local-registration scripts.
4. Build identifier maps and cell-type crosswalks.
5. Convert each reference to standardized validation tables.
6. Export standardized pipeline result sets.
7. Implement overlap, rank enrichment, matched permutation, and recovery metrics.
8. Generate `outputs/validation/validation_summary.csv`.
9. Add a validation report figure/table to the viewer or methods.
10. Use validation results to annotate final candidate mechanisms as externally supported, novel, or weakly supported.

