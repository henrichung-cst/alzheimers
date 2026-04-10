# Integrating Phosphoproteomics Kinase Findings with Cell Signaling Pathway Inference

A methodology overview for review. For implementation details, see [incytr_integration_plan.md](./incytr_integration_plan.md).

## 1. Background

### 1.1 The Alzheimer's kinase attribution pipeline

We have a 72-animal mouse cohort (Song et al.) with a factorial design: 4 genotypes (WT, App, Tau, App×Tau) × 3 timepoints (2, 4, 6 months) × 2 sexes. Two modalities are available:

- **Bulk phosphoproteomics** (TMT, all 72 animals): measures ~16,000 phosphorylation sites across the whole-tissue proteome. We correct for parent protein abundance using stoichiometry (log2 phospho − log2 protein), removing confounding from protein expression changes and isolating phosphorylation activity.
- **Paired snRNA-seq** (28 of the 72 animals): ~64,000 nuclei classified into 22 brain cell types (e.g., Microglia, Astrocyte, L5 IT neurons, Oligodendrocytes) using Allen Brain Cell Atlas taxonomy.

The current pipeline works as follows:

1. **Kinase activity inference**: Stoichiometry-corrected phosphosite fold changes are ranked per disease contrast (e.g., App vs WT at 4 months). GSEA-based Motif Enrichment Analysis (MEA) tests whether each kinase's known substrate set is enriched among the most changed phosphosites, producing a Normalized Enrichment Score (NES) and FDR per kinase per contrast. A positive NES means the kinase's substrates are more phosphorylated in disease; negative means less.

2. **Cell-type attribution**: The phosphoproteomics is bulk tissue — it cannot directly tell us which cell type a kinase signal comes from. We resolve this by combining three independent lines of evidence:
   - **Expression specificity** (Allen Whole Mouse Brain atlas): Is the kinase gene preferentially expressed in a specific cell type?
   - **Human AD concordance** (SEA-AD, Allen Institute): Does the kinase gene's transcriptomic change in human AD point in the same direction as the mouse kinase activity, in a given cell type?
   - **Within-cohort concordance** (paired snRNA-seq from the same 28 animals): Does the kinase gene's expression change in the paired single-cell data agree with the bulk kinase activity signal, per cell type?

   Each kinase × cell-type × contrast triplet receives a combined confidence score (high, moderate, low) reflecting the convergence of these three evidence sources.

**What the pipeline produces:** "Kinase X shows altered activity in disease condition Y, most likely originating from cell type Z." It stops there — it does not identify what signaling pathways those kinases participate in, what upstream signals triggered them, or what downstream targets they affect.

### 1.2 Incytr: cell signaling pathway inference

Incytr (He et al., 2025) is an R framework that infers and scores cell-to-cell signaling pathways from single-cell transcriptomics and optional multi-omics data. Its core model is a four-node signaling chain:

```
Ligand (sender cell) → Receptor (receiver cell) → Effector Molecule → Target
```

A curated database (IncytrDB, compiled from exFINDER-DB, NicheNet, CellChatDB, and NeuronChatDB) enumerates known biological relationships for each link: which ligands bind which receptors, which receptors activate which effectors, and which effectors regulate which targets.

**Scoring model**: For each candidate pathway, Incytr asks whether all four genes are expressed in the appropriate cell types and conditions. Expression products at each link (Ligand × Receptor, Receptor × Effector, Effector × Target) are passed through Hill functions — sigmoid curves that produce high scores when both genes in a pair are well-expressed and low scores when either is absent. The pathway's signaling probability is the product of these three Hill scores, yielding a value between 0 and 1.

**Multi-omics integration**: Incytr can overlay proteomics and phosphoproteomics fold changes onto pathway nodes. If a pathway's receptor shows increased phosphorylation in disease, this boosts the pathway's score. Each omics layer contributes a weighted term to a combined Pathway Discovery Score (PDS).

**Kinase evidence**: Incytr can incorporate kinase-substrate relationships in two ways:
- **Structural**: A reference database of known kinase-substrate pairs is checked against pathway nodes — does a kinase connect the receptor to the effector, or the effector to the target?
- **Activity-based**: External kinase enrichment results (e.g., from our MEA pipeline) can be integrated as evidence that a kinase is active, weighted by whether the kinase is expressed in the receiver cell type (the Exclusiveness Index, or EI).

**Statistical testing**: Pathway significance is assessed by permutation — cell labels are shuffled and signaling probabilities recomputed to generate an empirical null distribution.

**Output**: A ranked table of candidate signaling pathways with scores, multi-omics evidence, kinase support, and empirical p-values.

### 1.3 Why integrate them

The two tools answer complementary questions:

| | Alzheimers pipeline | Incytr |
|---|---|---|
| **Primary evidence** | Phosphoproteomics (protein activity) | Transcriptomics (gene expression) |
| **Biological question** | Which kinases are dysregulated? | Which signaling pathways are active? |
| **Cell-type resolution** | Inferred from convergent evidence | Direct from single-cell data |
| **Pathway context** | None — stops at individual kinases | Full L-R-EM-T chains |
| **Temporal/disease design** | 9 contrasts (3 disease × 3 timepoint) | 2 conditions per run |

The integration places kinase activity findings into a signaling pathway context: not just "GSK3β is active in microglia in the App model at 4 months" but "GSK3β connects receptor X to target Y in microglia, within a pathway initiated by ligand Z from astrocytes."

**This is explicitly hypothesis-generating.** The outputs are candidate pathways for experimental validation, not confirmed mechanisms.

## 2. Three Integration Modes

We propose three complementary modes that address different biological questions, ordered by increasing novelty.

### Mode A: Intercellular signaling

**Question**: Between two cell types in the AD brain, which signaling pathways are dysregulated, and does the kinase activity evidence support them?

**Approach**: Run Incytr in its standard configuration using the paired snRNA-seq (28 animals, 22 cell types) as the expression input, with IncytrDB providing the pathway database. For a given cell-type pair (e.g., Microglia → L5 IT neurons), Incytr enumerates all candidate L-R-EM-T pathways, scores them by co-expression, and tests significance by permutation.

The kinase activity findings from the phosphoproteomics pipeline enter as additional evidence: for kinases attributed to the receiver cell type at moderate or high confidence, their MEA enrichment scores and substrate relationships are passed to Incytr's kinase integration layer. This boosts pathways where a dysregulated kinase connects pathway nodes, and is gated by whether that kinase is actually expressed in the receiver cell type.

**Prioritized cell-type pairs** (based on AD biology):
- Microglia → excitatory neurons (neuroinflammatory signaling to vulnerable populations)
- Astrocyte → excitatory neurons (metabolic support, reactive astrogliosis)
- Neurons → Microglia (damage-associated signals, complement, "find-me/eat-me" cues)
- Endothelial → Microglia (vascular-immune interface)

**Condition design**: Incytr compares exactly two conditions. Our factorial design (4 genotypes × 3 timepoints) requires choosing a specific comparison per run — e.g., WT vs App at 4 months. This can be run for each disease model and timepoint of interest, mirroring the contrast structure of the kinase pipeline.

### Mode B: Autocrine signaling

**Question**: Within a single cell type, which self-stimulatory signaling loops are dysregulated in AD?

**Approach**: Set the sender and receiver to the same cell type (e.g., Microglia → Microglia). Incytr's code does not prohibit this. The pathway inference finds L-R-EM-T chains where all four genes are expressed in the same cell type, representing autocrine signaling — a cell secreting a ligand that binds its own receptor, triggering an intracellular cascade.

**Biological motivation**: Autocrine signaling is well-documented in AD-relevant cell types:
- Microglial TNF-α/IL-1β amplification loops that sustain neuroinflammation
- Astrocyte CXCL10/CXCR3 self-stimulation in reactive astrogliosis
- Neuronal BDNF/TrkB autocrine survival signaling
- Oligodendrocyte FGF/FGFR self-renewal circuits

In this mode, the phosphoproteomics evidence is particularly well-matched: the kinase activity and its substrates are in the same cell, so the cell-type attribution is less ambiguous.

### Mode C: Intracellular kinase cascades

**Question**: Within a cell type, which kinase → substrate → downstream target cascades are active in AD, independent of any secreted ligand?

**Motivation**: Modes A and B both require a ligand-receptor initiation step because IncytrDB encodes extracellular signaling relationships. But many disease-relevant kinase cascades are triggered by intracellular events — oxidative stress activating stress kinases, DNA damage activating checkpoint kinases, or misfolded protein accumulation activating inflammatory kinases. These cascades have no secreted ligand.

**Key insight**: Incytr's computational machinery is generic. It joins three "from → to" relationship tables and scores chains by co-expression. The biological semantics (ligand-receptor, receptor-effector, effector-target) come entirely from the database content, not the code. A custom intracellular cascade database would work with no software changes.

**Proposed database structure**:

| Link | Standard (intercellular) | Proposed (intracellular) |
|------|--------------------------|--------------------------|
| Layer 1 | Ligand → Receptor | Stimulus → Kinase |
| Layer 2 | Receptor → Effector | Kinase → Substrate |
| Layer 3 | Effector → Target | Substrate → Downstream target |

The four-node chain becomes: **Stimulus → Kinase → Substrate → Target**, all within one cell type.

**How the scoring model works differently here**: The Hill function scoring still asks "are all four genes co-expressed in this cell type?" — a necessary condition for any intracellular cascade. However, co-expression has a different biological meaning for intracellular cascades than for intercellular signaling. For ligand-receptor pairs, co-expression magnitude is a proxy for signaling capacity (more ligand = more binding). For kinase-substrate pairs, co-expression is closer to a precondition than a readout of activity — most kinase-substrate pairs in the same cell are constitutively co-expressed, and the regulatory logic is post-translational (phosphorylation, localization, conformational activation).

**Consequence**: The expression layer enumerates biologically plausible cascades (is the kinase expressed in this cell type?), but it provides less discrimination than for Modes A/B because most plausible cascades will pass the co-expression test. The phosphoproteomics overlay does the actual biological work of distinguishing active from inactive cascades. Mode C results are therefore reported with the phospho-overlay score as the primary ranking, not the expression-only pathway score. A pre-specified comparison — the correlation between expression-based and phospho-overlay rankings — is computed across all three modes. If this correlation is high in Mode A (phospho confirms transcriptomics) but low in Mode C (phospho substantially reranks), it empirically validates the argument that Mode C's biological signal lives in the phospho layer.

**Database sources**: Layer 1 (what activates kinases) from signaling databases such as SIGNOR or OmniPath. Layer 2 (kinase-substrate relationships) from the same kinase-library motif predictions already used in our MEA analysis. Layer 3 (downstream effects of phosphorylated substrates) from Reactome or SIGNOR, augmented by TF-target databases (DoRothEA, TRRUST) for substrates that are transcription factors and high-confidence protein interaction data (STRING) for non-TF substrates. Layer 3 is the least well-curated, and coverage statistics (what fraction of our MEA-significant kinases appear in each layer) will be reported as a quality metric. Assembling this database is a bioinformatics curation task, not an algorithmic one.

**This is the most novel aspect of the proposed integration**, and the one where reviewer input on biological plausibility would be most valuable.

## 3. Bridging Bulk Phosphoproteomics to Cell-Type Resolution

The central methodological challenge is that Incytr expects per-cell-type data, while the phosphoproteomics is bulk tissue. This section describes how we propose to bridge that gap.

### 3.1 The problem

A phosphosite fold change measured in bulk tissue is a weighted average across all cell types in the sample. If GSK3β substrates show increased phosphorylation in App mice, that signal could originate from neurons (where GSK3β phosphorylates tau), microglia (where GSK3β regulates inflammatory signaling), or both. The bulk measurement cannot distinguish these.

### 3.2 The inference layer

Our existing pipeline addresses this through convergent evidence from three independent sources:

1. **Expression specificity** (Allen Whole Mouse Brain atlas): If GSK3β is 3× more expressed in neurons than in other cell types, a bulk GSK3β signal is more likely to reflect neuronal activity.
2. **Human disease concordance** (SEA-AD): If GSK3β is transcriptomically upregulated specifically in human AD neurons (not microglia), this supports a neuronal origin for the mouse phosphoproteomic signal.
3. **Within-cohort concordance** (paired snRNA-seq, same animals): If GSK3β expression increases specifically in neurons in the paired single-cell data from the same animals, this provides same-species, same-cohort support.

Each attribution receives a confidence level based on how many evidence sources agree:
- **High confidence**: Within-cohort concordance + strong expression specificity + concordant effect direction
- **Moderate confidence**: Fewer sources or weaker agreement
- **Low confidence**: Single weak source

A key point for assessing circularity: only the expression specificity source overlaps with what Incytr uses for its transcriptomic scoring. The SEA-AD concordance is cross-species human data, and the within-cohort concordance is a separate statistical model on paired snRNA-seq. These are genuinely independent evidence streams.

### 3.3 Tiered phospho integration

Rather than projecting the bulk phospho signal uniformly to all cell types, the phosphoproteomics enters Incytr differently depending on how confidently it can be attributed to a specific cell type. This avoids both the information loss of treating phospho purely as a binary filter and the false precision of manufacturing synthetic per-cell-type phospho data from bulk measurements.

**Three tiers:**

- **High-confidence attributions (project):** The full bulk fold change is assigned to the single top-ranked cell type. This "winner-take-all" assignment avoids an implicit assumption that has no biological justification — that a kinase expressed 60% in microglia and 40% in astrocytes produces a 60/40 phosphorylation split. A kinase could be expressed predominantly in one cell type but phosphorylate substrates predominantly in another (via secreted intermediates, scaffolding dependencies, or threshold effects). Winner-take-all says "this signal most likely comes from microglia," which is a defensible statement when three independent evidence sources converge.

- **Moderate-confidence attributions (filter):** The kinase is flagged as active in the cell type (based on its MEA enrichment score), but no specific fold-change magnitude is passed. This binary signal can boost pathways containing that kinase but cannot drive pathway rankings by itself.

- **Low-confidence attributions (exclude):** No phospho signal enters the analysis. We prefer missing data over noisy data.

This is a proof-of-concept design. For the later systematic analysis, the winner-take-all simplification is revisited by running the analysis twice — once assigning to the top cell type, once to the second — to identify which pathway findings are robust to cell-type assignment.

### 3.4 Handling discordance between expression and phospho evidence

When Incytr's transcriptomic scoring and the phosphoproteomics evidence point in different directions for the same pathway, the discordance is a biological finding — a dissociation between transcriptional and post-translational regulation — not noise to be smoothed over.

We identify three discordance scenarios:

**Scenario A — Transcriptionally active, enzymatically inactive.** The pathway's genes are well co-expressed (Incytr gives it a high transcriptomic score), but the kinase's substrates are *less* phosphorylated in disease. Possible biology: the pathway is being shut down post-translationally while the transcriptional program persists (regulatory lag), or the cell is compensatorily upregulating gene expression because the post-translational activity has declined.

**Scenario B — Transcriptionally dark, enzymatically active.** The pathway components are weakly co-expressed (Incytr's expression-based scoring filters them out), but the kinase shows strong substrate phosphorylation in disease. Possible biology: very low transcript levels are sufficient for a functional kinase cascade (common for signaling molecules), or the kinase is actually operating in a different cell type. These pathways are invisible to transcriptomics-only methods and are detected through a separate database lookup rather than through Incytr's standard pipeline.

**Scenario C — Same direction, different magnitude.** The two evidence streams agree directionally but differ in magnitude. This is handled naturally by the combined scoring.

**Reporting:** Results are presented in three groups rather than one combined ranking:

- **Concordant pathways** (expression and phospho agree): Combined score reported. Highest-confidence candidates.
- **Discordant type A** (expression up, phospho down): Expression and phospho evidence reported separately, not combined. Flagged as candidate post-translational regulatory events.
- **Discordant type B** (expression low, phospho up): From the Scenario B database lookup. Lower confidence but potentially the most novel — these are the pathways that phosphoproteomics uniquely reveals.

The discordance flag requires the phospho evidence to reach nominal statistical significance (FDR < 0.25) to prevent flagging pathways where the phospho signal is simply absent rather than contradictory.

### 3.5 Sensitivity analysis

Every analysis is run with and without the phospho/kinase layer. This is a core deliverable, not an afterthought. It enables:

1. **Marginal value assessment**: What fraction of top-ranked pathways change rank substantially when phospho is added? This quantifies whether the phosphoproteomics integration matters for each mode.
2. **Discordance baseline**: The expression-only run provides the reference for identifying discordant pathways.
3. **Robustness check**: Pathways significant on transcriptomics alone and reinforced by phospho are more trustworthy than those that depend entirely on attributed phospho signal.

A pre-specified figure — expression-based rank versus phospho-overlay rank, faceted by mode — summarizes when and where the phosphoproteomics adds information that transcriptomics alone misses.

### 3.6 Limitations

These are inherent to the data design and should be stated in any downstream analysis:

- **Attribution is correlative, not causal.** A kinase attributed to microglia based on expression specificity and transcriptomic concordance could actually be active in a co-expressing cell type.
- **Bulk phosphoproteomics cannot distinguish autocrine from paracrine kinase activity.** A kinase active in microglia could be phosphorylating substrates in neurons via secreted intermediates.
- **Winner-take-all discards near-tie information.** When the top two cell types have similar attribution scores, assigning the full signal to the top-ranked type introduces false certainty. The high-confidence tier mitigates this (convergent evidence is required), but near-ties can occur.
- **Temporal resolution is coarse.** The three timepoints (2, 4, 6 months) cannot capture rapid kinase dynamics.
- **Cross-species inference.** The human SEA-AD concordance evidence assumes directional conservation between human AD and the 5xFAD mouse model. This is well-supported for certain pathways (MAPK, GSK3) and cell types (microglia, astrocytes) but is an open question for others.

## 4. Experimental Design

### 4.1 Condition selection

The proof of concept uses **WT vs App at 4 months** (males only). This is an early amyloid timepoint — biologically cleaner than late-stage combined pathology (App×Tau at 6 months), with well-characterized kinase biology (GSK3β and MAPK/ERK in amyloid processing) that enables sanity-checking top-ranked pathways against the existing literature. Although later timepoints may have richer kinase signal (more kinases reaching statistical significance), interpretability is more important than statistical power for a proof of concept.

After pipeline validation, pre-specified follow-ups extend to App at 6 months and App×Tau at 6 months to ask whether pathways identified early are amplified or replaced at later disease stages.

### 4.2 Phased approach with decision gates

We propose four phases. Each phase has explicit deliverables and a quantitative decision gate before proceeding.

**Phase 1 — Proof of concept (Mode A, intercellular)**: Run intercellular signaling for Microglia → L5 IT neurons, WT vs App at 4 months. Run with and without the phospho/kinase layers. Assess biological plausibility of top-ranked pathways.

*Decision gate*: Quantify the marginal value of the phosphoproteomics — what fraction of the top-20 pathways change rank by more than 10 positions when phospho evidence is added? If minimal reranking occurs, the phospho layer is largely redundant for intercellular signaling. This is a legitimate and informative finding, not a failure. Proceed to Phase 2 with the expectation that autocrine mode (where kinase and substrate are in the same cell) may show stronger phospho contribution.

*Key deliverable*: The expression-versus-phospho ranking correlation (Spearman), computed in Phase 1, becomes the baseline for comparison across all subsequent modes.

**Phase 2 — Autocrine (Mode B)**: Run sender == receiver for Microglia, Astrocyte, L5 IT, and Oligodendrocyte. Compare expression-versus-phospho ranking correlations to Mode A. Assess whether the phospho contribution is stronger when kinase and substrate are in the same cell.

**Phase 3 — Intracellular cascades (Mode C)**: Assemble the intracellular database, validate coverage for AD-relevant kinase families (GSK3, MAPK, CDK5, DYRK1A), run for key cell types. Report with the phospho-overlay score as the primary ranking (see §2 Mode C rationale). The expression-versus-phospho ranking correlation is expected to be low (phospho substantially reranks), empirically validating the argument that Mode C's biological signal lives in the phospho layer.

**Phase 4 — Systematic analysis**: Run all three modes across prioritized cell-type pairs and disease conditions. Rank final hypotheses using a convergent evidence framework (see §4.4).

### 4.3 Primary analysis choices

- **Sex**: Males only (33 animals after outlier exclusion), consistent with the primary kinase analysis that avoids hormonal confounds on phosphoproteomics.
- **Disease comparisons**: WT vs App for the proof of concept. WT vs Tau and WT vs App×Tau deferred to Phase 4.
- **Timepoints**: Single timepoint (4 months) for the proof of concept. Per-timepoint analysis for Phase 4, with the pre-specified progression comparison (4mo → 6mo).

### 4.4 Handling the multiple hypothesis problem

Running three modes across multiple cell-type pairs and conditions generates a large hypothesis space. We do not apply formal multiple testing correction across modes, because the modes test different biological questions (intercellular, autocrine, and intracellular signaling) and correcting across them is statistically incoherent.

Instead, final hypotheses are ranked by a **convergent evidence framework**: the number of independent evidence streams supporting each pathway finding. A pathway is highest-confidence when it is supported by:
- Transcriptomic co-expression scoring (Incytr base)
- Bulk phospho evidence for its kinase/substrate nodes
- Cell-type attribution evidence placing those kinases in the relevant cell type
- Support across multiple modes (e.g., a kinase cascade identified in both autocrine and intracellular analysis)

Per-mode results are reported in full as supplementary material. The main narrative focuses on convergent hits.

### 4.5 Statistical power for rare cell types

Cell counts per cell type per condition vary widely in the snRNA-seq (Astrocyte: ~90 cells/animal vs Lamp5 Lhx6: ~5 cells/animal). Incytr's permutation test (shuffling cell labels to generate null distributions) requires sufficient cells for stable results. We assign cell types to three power tiers:

- **Tier 1** (>100 cells per condition): Standard permutation testing. Report with p-values.
- **Tier 2** (30–100 cells per condition): Permutation testing with a stability check — rerun with 5 different random seeds and report the variability in p-values. Flag pathways where p-values are unstable across seeds.
- **Tier 3** (<30 cells per condition): No permutation-based inference. Report pathway scores and phospho evidence only, without p-values. These results are hypothesis-generating only.

The cell-count thresholds are empirically calibrated before the main analysis by permuting labels within control animals and characterizing p-value stability as a function of cell count.

## 5. Resolved Design Decisions

The following questions were raised during the initial design and resolved through review. The rationale for each is documented in the relevant section above.

1. **Attribution confidence threshold** → Tiered: high-confidence attributions get the full projected fold change (winner-take-all to top cell type), moderate get a binary active/inactive flag, low are excluded. See §3.3.

2. **Multiple hypothesis burden** → Convergent evidence framework, not cross-mode correction. See §4.4.

3. **Permutation test power** → Three-tier subclass handling based on cell count. See §4.5.

4. **Condition selection** → WT vs App at 4 months for proof of concept (interpretability over power). See §4.1.

5. **Discordance handling** → When expression and phospho evidence conflict, results are reported in three groups (concordant, discordant type A, discordant type B) rather than combined into a single score. See §3.4.

6. **R/Python boundary** → CSV file intermediates. Python scripts export alzheimers pipeline outputs in Incytr-compatible formats; R wrappers read those files and call Incytr functions.

## 6. Remaining Open Questions

1. **Mode C biological plausibility**: Is repurposing Incytr's four-node scoring model for intracellular cascades methodologically sound? We have addressed the co-expression concern by using expression for cascade enumeration and the phospho overlay for ranking (see §2 Mode C). The remaining question is whether the Hill function product — even as a permissive filter — introduces systematic biases (e.g., favoring cascades involving highly-expressed housekeeping kinases over lowly-expressed but disease-relevant ones).

2. **Intracellular database coverage**: The quality of Mode C depends on the completeness of upstream kinase activation relationships (Layer 1) and downstream substrate effects (Layer 3). Layer 3 is augmented with TF-target databases and protein interaction data, but Layer 1 (Stimulus → Kinase) remains the binding constraint. Should we restrict Mode C to kinase families with strong Layer 1 coverage (e.g., MAPK, GSK3, CDK), or attempt genome-wide and accept gaps?

3. **Condition pooling**: Pooling timepoints per genotype increases statistical power but assumes the signaling architecture is stable across disease progression. For the proof of concept we use a single timepoint (4 months). For Phase 4, should each timepoint be analyzed separately or pooled?

4. **Microglial heterogeneity**: The 22 cell types lump homeostatic and disease-associated microglia (DAM), which have very different signaling profiles. If the autocrine analysis (Mode B) reveals heterogeneous microglial signals, should we sub-cluster the microglia and rerun?
