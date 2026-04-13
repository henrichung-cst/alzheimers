# Incytr Integration: Connecting Bulk Kinase Activity to Cell-Cell Signaling Pathways

## Background

We have two independent lines of evidence about kinase signaling in an Alzheimer's disease mouse model (App knock-in, 4-month males):

1. **Bulk tissue phosphoproteomics** — From 72-animal TMT proteomics, we compute stoichiometry-corrected phosphorylation (log2 phospho minus log2 parent protein abundance), then run motif enrichment analysis (MEA, GSEA-based) to identify kinases whose substrate phosphorylation patterns change in disease. This produces a normalized enrichment score (NES) and FDR per kinase per contrast. We then attribute each kinase to cell types using three evidence sources: SEA-AD human AD transcriptomic concordance, Whole Mouse Brain (WMB) expression specificity, and within-cohort paired snRNA-seq concordance.

2. **Single-cell transcriptomics** — From paired snRNA-seq of 28 animals in the same cohort (63K nuclei, Allen Cell Type Mapper annotations), we have cell-type-resolved gene expression. Incytr uses this to infer intercellular signaling pathways between specific cell-type pairs.

The goal is to integrate these two evidence types: use Incytr's cell-type-resolved expression to identify active signaling pathways between Microglia-PVM (sender) and L5 IT cortical neurons (receiver), then use our bulk kinase activity evidence to assess which pathways have additional phosphoproteomic support.

## Phase 1 scope

- **Sender:** Microglia-PVM (brain-resident immune cells)
- **Receiver:** L5 IT (layer 5 intratelencephalic-projecting excitatory neurons)
- **Contrast:** WT vs App knock-in, 4 months, males only
- **Cell counts:** 185 Microglia-PVM nuclei (113 WT, 72 App), 37 L5 IT nuclei (16 WT, 21 App)

## How Incytr models signaling pathways

Incytr represents each intercellular signaling pathway as a four-gene chain:

```
Ligand (sender) --> Receptor (receiver) --> EM (receiver) --> Target (receiver)
```

- **Ligand**: a secreted or membrane-bound molecule produced by the sender cell type
- **Receptor**: the receptor on the receiver cell surface that binds the ligand
- **EM** (Effector Molecule): an intracellular signaling mediator in the receiver
- **Target**: the downstream gene whose expression or activity is ultimately affected

Which four-gene combinations are valid is determined by a curated reference database (IncytrDB) of known molecular interactions. For a given sender-receiver pair, Incytr enumerates all valid chains, then scores each one.

### Pathway scoring

The primary score is **TPDS** (Transcriptomic Pathway Differential Score), which measures how much the expression of all four genes changes between conditions (WT vs App). Specifically, TPDS is derived from a signaling probability model:

```
SigProb = Hill(L x R) x Hill(R x EM) x Hill(EM x T)
```

where each Hill function captures the joint expression of adjacent nodes, and expression values are per-cell-type averages from the snRNA-seq. TPDS is then a logistic transformation of SigProb to the range [-1, 1]. A pathway with TPDS near +1 has strongly upregulated co-expression of all nodes in disease; near -1 means strongly downregulated.

Incytr can also incorporate additional omic layers (proteomics, phosphoserine, phosphotyrosine, acetylation, ubiquitination, methylation) and kinase activity evidence. Each layer contributes an additive term to the final score (PDS = Pathway Differential Score):

```
PDS = TPDS + (PTM layer contributions) + (kinase activity contributions)
```

### Gene filtering and the combinatorial constraint

Before pathway enumeration, genes must pass an expression detection threshold: a gene must be detected (non-zero UMI count) in at least N% of cells in the relevant cell type. This is necessary because snRNA-seq is sparse --- most genes in most cells read as zero. The threshold filters out genes whose expression estimates are unreliable.

The threshold also controls computational feasibility. Pathway count scales combinatorially with gene count (roughly |Ligand| x |Receptor| x |EM| x |Target|), so:

| Detection threshold | Sender genes | Receiver genes | Estimated pathways |
|---|---|---|---|
| 5% | 4,064 | 10,027 | ~9.5 million |
| 10% | 2,357 | 7,464 | ~44,000 |
| 20% | 1,127 | 4,994 | ~20,000 |
| 50% | 210 | 2,114 | 3,544 (actual) |

The current implementation uses 50% to fit within available memory (~30 GB). Note that the receiver gene list supplies three of the four node positions (Receptor, EM, and Target), which is why receiver gene count has a disproportionate effect on pathway count.

## What happened in Phase 1

### Phospho fold changes enter the model but are small

The stoichiometry-corrected phosphorylation data enters Incytr's scoring through the phosphoserine (PhPDS_ps) channel. For each pathway, Incytr computes the phospho fold change at each of the four node positions, transforms them through a logistic function, and averages them. This produces a per-pathway phospho score in the range [-0.21, +0.26].

However, TPDS (the expression score) ranges from [-1.0, +1.0] with a standard deviation of 0.45, while PhPDS_ps has a standard deviation of 0.035 --- roughly 13 times smaller. At the median pathway, the phospho contribution is ~2.4% of the total PDS. The phospho data is real and correctly propagated, but it's a minor perturbation on an expression-dominated score.

### Kinase activity evidence is completely disconnected

This is the central problem. Incytr's kinase activity channel works by checking whether any of the four pathway node genes are kinases with significant enrichment evidence. If a pathway contains `Kinase X` as its EM or Target node, and our MEA results show Kinase X has NES = 2.3 at FDR = 0.08, then Incytr incorporates that activity score into the pathway's PDS.

But for a kinase gene to be a pathway node, it must first pass the expression detection threshold. At 50%, only 14 kinase genes survive in the sender and 99 in the receiver. We have 114 kinases with significant MEA results attributed to Microglia-PVM or L5 IT at moderate-or-higher confidence. Most of these kinases are not detected in >= 50% of cells, so they are excluded from the gene pool, never become pathway nodes, and their activity evidence has nowhere to attach.

The result: **activity scores are zero for all 3,544 pathways.** Only 13 pathways have any kinase structural scores at all (all sharing the Sirpa-Ptprd ligand-receptor pair). The kinase activity channel is architecturally intact but effectively severed by the expression filter.

### The integration barely changes rankings

Spearman rho between expression-only and full-integration pathway rankings: **0.9984**. The phospho and kinase evidence combined shift the median pathway score by 2.4%. The integration is operationally working but not meaningfully contributing.

## The fundamental mismatch

The two evidence types operate at different resolutions:

| Property | Bulk phosphoproteomics | snRNA-seq / Incytr |
|---|---|---|
| **Resolution** | Whole tissue (72 animals) | Single-cell (63K nuclei) |
| **Measurement** | Protein phosphorylation state | mRNA transcript counts |
| **Cell-type assignment** | Indirect (WMB specificity, SEA-AD concordance, within-cohort snRNA-seq correlation) | Direct (each nucleus is classified) |
| **What it captures** | Post-translational kinase activity | Transcriptomic co-expression |
| **Kinase information** | Which kinases are active, how strongly, with what substrates | Whether a kinase gene's mRNA is detected in a cell type |

Incytr expects cell-type-resolved molecular measurements at each pathway node. Our kinase evidence is tissue-level activity with probabilistic cell-type attribution. Forcing kinase genes into pathway nodes by lowering the expression threshold would misrepresent indirect evidence as direct measurement.

At the same time, the cell-type attribution is not arbitrary. It integrates three independent lines of evidence: (1) cross-species concordance with human AD single-nucleus RNA-seq from SEA-AD, (2) expression specificity from the Whole Mouse Brain atlas, and (3) within-cohort concordance from paired snRNA-seq in the same animals. The attribution represents a reasonable assignment given available data, but it is not a direct observation of kinase activity within a specific cell type.

## Implemented approach: substrate-based external reranking

Rather than injecting kinase evidence into Incytr's internal scoring (which requires kinases to be pathway nodes), we use kinase evidence externally to rerank Incytr's expression-based pathway results.

### Core idea

For each pathway, ask: are any of the pathway's node genes (particularly EM and Target) known substrates of kinases that are active in disease?

The kinase-substrate relationships come from kinase-library motif predictions (kldata). If a pathway's Target gene is a known substrate of Kinase X, and our MEA shows Kinase X is significantly activated in the App contrast, then this pathway has convergent support --- expression says the signaling chain is active, and phosphoproteomics says a kinase that phosphorylates the endpoint is also active.

This approach has several properties:

1. **It does not require the kinase to be a pathway node.** The kinase acts on the substrate (which is a pathway node), but the kinase itself does not need to be in the four-gene chain. This resolves the expression threshold bottleneck.

2. **It does not require the kinase to be expressed in the receiver cell type.** A kinase can be catalytically active at low transcript levels, or its activity may be regulated post-translationally. The phosphoproteomic evidence captures the functional state of the kinase regardless of its mRNA abundance.

3. **It keeps the two evidence types separate.** Incytr's expression-based ranking is computed purely from snRNA-seq, which it is designed to analyze. The kinase evidence is applied as a separate layer on top, rather than being mixed into the expression model.

### Dual-channel architecture

Incytr's internal kinase channel is preserved for the ~99 kinases that pass the expression threshold and earn their way into pathway nodes (23 pathways currently have kinase node genes). The external substrate layer covers the remaining majority that cannot become nodes. The two channels are complementary:

- **Internal (Incytr native):** "Is this kinase node transcriptionally active and enzymatically enriched?" --- requires kinase gene to be a pathway node, scored by SiK/activity within Incytr.
- **External (substrate-based):** "Are downstream proteins in this pathway being phosphorylated by disease-activated kinases, regardless of whether those kinases are nodes?" --- connects through kinase-substrate relationships, applied as postprocessing.

**Deduplication rule:** When computing the external substrate score for a given pathway, kinases that are already pathway nodes (EM or Target) for that pathway are excluded. Their contribution is already captured by Incytr's internal channel. This prevents double-counting while preserving both evidence streams.

### Kinase-imputed pathway expansion

The expression detection threshold (50%) excludes genes whose mRNA is sparse in snRNA-seq. But some of these genes have direct protein-level evidence of activity: they appear in kldata as substrates of kinases with significant MEA enrichment (FDR < 0.25). The phosphoproteomic data is observing post-translational modification of these proteins regardless of mRNA detection rate.

Kinase-imputed expansion adds these genes to the receiver gene list for pathway inference, alongside the expression-threshold genes. This allows Incytr to discover pathways where some nodes are supported by kinase-substrate evidence rather than expression evidence alone.

**How it works:**

1. `export_kinase_imputed_genes.py` identifies all genes in the expression matrix that are known substrates (via kldata) of at least one kinase with MEA FDR < 0.25 for the current contrast.
2. `run_incytr.R` reads this list and unions it with the expression-threshold receiver genes. The sender gene list stays at 50% (unchanged).
3. After `pathway_inference()`, every pathway is labeled:
   - **expression-confirmed**: all 4 nodes (Ligand, Receptor, EM, Target) pass the 50% expression threshold in their respective cell type
   - **kinase-imputed**: one or more receiver-side nodes were admitted via kinase-substrate evidence
4. The `imputed_nodes` column records which specific positions were imputed (e.g., "Target", "EM;Target").

**Properties of kinase-imputed genes:**

The 2,537 additional receiver genes are not phantom genes --- they have real mRNA signal (median 16.2% detection rate in L5 IT), just below the 50% threshold. Kinase-substrate evidence provides an independent reason to include them: the protein is being phosphorylated by a disease-activated kinase, so it is present and post-translationally active regardless of its mRNA abundance.

**Pathway expansion and SigProb filtering:**

| Stage | Expression-confirmed | Kinase-imputed | Total |
|---|---|---|---|
| After pathway_inference | 3,544 | 10,284 | 13,828 |
| After SigProb cutoff (0.01) | 3,544 | 2,217 | 5,761 |

The SigProb cutoff naturally prunes kinase-imputed pathways where the imputed node has very low expression --- the Hill function products that drive SigProb are small when any node is weakly expressed. The 2,217 kinase-imputed pathways that survive have at least moderate signaling probability, typically because 3 of 4 nodes are strongly expressed and only the imputed node is sparse.

**Imputed node positions:**

| Imputed nodes | Count | Fraction |
|---|---|---|
| Target only | 1,808 | 81.5% |
| EM only | 320 | 14.4% |
| EM + Target | 89 | 4.0% |
| Receptor | 0 | 0% |

Targets dominate because Target is the most downstream position and draws from the largest gene pool. No Receptors or Ligands are imputed (receptors require expression in the receiver to form L1 edges, and the sender threshold is unchanged).

**Interpretation:** Kinase-imputed pathways represent a lower evidence tier than expression-confirmed pathways. The expression-confirmed label means the entire signaling chain has direct transcriptomic support. The kinase-imputed label means the chain is plausible based on expression at most nodes plus protein-level kinase activity evidence at the imputed node(s). Both tiers flow through the same scoring pipeline (TPDS, PDS, kinase support reranking), and downstream analysis can filter or stratify by the `pathway_evidence` column.

### Coverage

Of the 5,761 pathways after SigProb filtering:

- **4,548 pathways (78.9%)** have at least one EM or Target gene that is a known substrate of an attributed kinase with significant MEA evidence
- **1,213 pathways (21.1%)** have no kinase-substrate connection and are unaffected by reranking
- 134 kinases are both significant (FDR < 0.25) and attributed to sender or receiver

By evidence tier:

| Tier | Total | With kinase support | Coverage |
|---|---|---|---|
| Expression-confirmed | 3,544 | 2,405 | 67.9% |
| Kinase-imputed | 2,217 | 2,143 | 96.7% |

The near-complete kinase support coverage of kinase-imputed pathways is expected: these nodes were admitted specifically because they are substrates of significant kinases.

## Open questions for discussion

### 1. Substrate promiscuity and phosphorylation hubs

Some proteins are substrates of many kinases. In our data, the Target gene Srrm2 is a known substrate of 80 of our 114 attributed kinases. Map1b, Cep170, and Map2 are substrates of 50--75 kinases. These are heavily phosphorylated scaffold and structural proteins.

A naive kinase support score would assign very high scores to pathways ending at these hub substrates, not because those pathways are specifically disease-relevant, but because the Target happens to be phosphorylated by nearly everything.

**Question:** How should we handle substrate promiscuity? Options include:
- Inverse-frequency weighting (a kinase-substrate connection is worth more if the substrate has few kinases)
- Restricting to kinases with strong directional concordance rather than counting all connections
- Focusing on the identity of the connected kinases rather than the count

### 2. Double-counting phosphoproteomic evidence

The phospho fold change data (PhPDS_ps) already enters Incytr's internal scoring through the phosphoserine channel. The proposed kinase support score also derives from phosphoproteomics (via MEA on the same stoichiometry data). These are not identical measurements --- one is a per-site fold change, the other is a set-level enrichment statistic --- but they originate from the same underlying data.

**Question:** Is using the same phosphoproteomic dataset in two different ways (per-site fold changes inside Incytr, kinase enrichment scores outside) a form of double-counting? Or are these sufficiently different transformations of the data that they provide complementary information?

### 3. Causal interpretation

A pathway `Sirpa --> Ptprd --> Xrn2 --> Srrm2` models a specific signaling chain: Sirpa (ligand from microglia) binds Ptprd (receptor on L5 IT neurons), which activates Xrn2 (intracellular mediator), which affects Srrm2 (target). If we find that kinases phosphorylating Srrm2 are active in disease, that tells us Srrm2's phosphorylation state is altered --- but it does not tell us whether the phosphorylation occurred through the Sirpa-Ptprd-Xrn2 chain or through an entirely independent mechanism.

**Question:** How should we interpret a pathway that has both expression support (the four-gene chain is co-regulated) and kinase substrate support (the endpoint is phosphorylated by active kinases)? Is convergent evidence at the Target gene meaningful even if we cannot establish that the phosphorylation flows through the modeled chain? Are there biological scenarios where the signaling chain and the kinase activity would be expected to converge on the same target independently?

### 4. Directionality and inhibitory signaling

If a kinase has positive NES (activated in App vs WT) and the pathway's expression score is also positive (upregulated), this is concordant. But what if the kinase is activated and the pathway is downregulated? This could mean:

- The kinase phosphorylation is inhibitory (phosphorylation of the Target suppresses its function or expression)
- The kinase is activated as a compensatory response to the pathway being downregulated
- The two signals are genuinely unrelated

**Question:** Should we treat directional concordance as a requirement for boosting a pathway, or should directionally discordant kinase-pathway pairs be flagged as a distinct category of interest (e.g., potential inhibitory regulation)?

### 5. Attribution confidence and the cell-type question

Our kinase-to-cell-type attribution uses three evidence sources but remains indirect. For this Phase 1 test case:

- 71 kinases are attributed to Microglia-PVM at moderate+ confidence
- 97 kinases are attributed to L5 IT at moderate+ confidence
- Many kinases are attributed to both

A kinase attributed to L5 IT (the receiver) with high confidence has more direct relevance to receiver-side pathway nodes than a kinase attributed to Microglia-PVM (the sender). But a kinase active in the sender could still be biologically relevant if it affects ligand production or secretion.

**Question:** How should attribution confidence and cell-type identity modulate the kinase support score? Should we weight differently depending on whether the kinase is attributed to the sender vs receiver? Should we require a minimum attribution confidence, or let the graded confidence propagate through as a continuous weight?

### 6. Null expectation and statistical significance

If we randomly shuffled kinase-substrate assignments, some pathways would still receive kinase support by chance, simply because ~56% of Target genes are substrates of at least one attributed kinase. We need a null model to determine whether the observed kinase support for any given pathway exceeds what would be expected by chance.

**Question:** What is the appropriate null model? Permuting kinase-substrate assignments preserves pathway structure but breaks biological relationships. Permuting kinase significance (shuffling NES/FDR across kinases) preserves substrate relationships but breaks disease associations. The choice of null affects what we can claim about the results.

## Consultation: molecular biology review (April 2026)

The open questions above were reviewed by a molecular biologist with experience in biostatistics and cellular pathway analysis. Their assessment and recommendations follow.

### Overall assessment

The substrate-based external reranking was endorsed as the right approach. The reviewer confirmed that the Phase 1 failure mode is expected biologically: kinases are often expressed at low mRNA levels and regulated post-translationally, so the 50% snRNA-seq detection threshold filters out most functionally relevant kinases. The zero activity scores across all 3,544 pathways reflect an architectural mismatch, not a data quality issue.

The orthogonality between expression and kinase signals (rho = 0.179) was noted as genuinely promising --- it indicates real complementary information rather than redundancy.

### Recommendations per open question

**1. Substrate promiscuity** --- This was identified as the most dangerous statistical pitfall. The recommendation is a layered approach:

- Apply inverse-frequency weighting: weight each kinase-substrate edge by `1/log(N)` where N is the number of kinases known to phosphorylate that substrate. This is analogous to inverse document frequency (IDF) in text mining --- a connection to a promiscuous substrate is worth less than a connection to a specific one.
- Additionally require a minimum NES magnitude and FDR stringency for each contributing kinase. A substrate hit by 80 kinases of which 70 are barely significant is not the same as a substrate hit by 3 kinases with NES > 2.5 and FDR < 0.05.
- As a diagnostic: compute the score with and without inverse-frequency weighting and check whether the same hub substrates dominate regardless. If so, consider capping the maximum contributing kinases per substrate, or switching from a "sum of kinases" to a "best kinase" formulation.

**2. Double-counting** --- Judged manageable. The per-site fold change (PhPDS_ps) and the set-level enrichment statistic (MEA NES) are sufficiently different transformations of the same underlying phosphoproteomic data. However:

- In any manuscript, frame these as different *views* of the same dataset, not independent evidence.
- Report the correlation between PhPDS_ps and the kinase support score across pathways. If low (expected, given rho = 0.179 between TPDS and kinase connectivity), this empirically supports complementarity. If high, it indicates redundancy and PhPDS_ps should be dropped from Incytr's internal scoring when the external kinase layer is applied.

**3. Causal interpretation** --- The reviewer urged the most caution here. Convergent evidence at the Target gene does not establish that phosphorylation flows through the modeled signaling chain. However, convergence *is* biologically meaningful in a weaker but still valuable sense:

- A target gene that is both transcriptionally co-regulated with an active signaling chain *and* phosphorylated by disease-activated kinases is more likely to be functionally relevant to disease than a target supported by expression alone.
- The appropriate framing is **convergent functional evidence**, not **mechanistic pathway validation**. Manuscript language should make this distinction explicit.
- There are biological scenarios where independent convergence is expected and interesting. For example, in neuroinflammatory signaling, microglial ligands can activate receptor tyrosine kinases on neurons that feed into the same downstream targets (e.g., MAP kinases) that are also activated by cell-autonomous stress responses. This kind of multi-modal regulatory convergence on the same target may be more biologically informative than a single linear chain.

**4. Directionality** --- The reviewer strongly advised against treating concordance as a hard requirement.

- Discordant kinase-pathway pairs (kinase activated, pathway downregulated) are some of the most biologically interesting findings in phosphoproteomics. They are classic signatures of inhibitory phosphorylation or compensatory feedback loops. Filtering them out risks discarding exactly the biology that makes the integration worthwhile.
- Recommendation: compute the kinase support score as an **unsigned quantity** (magnitude of convergence, not direction). Annotate each pathway with a concordance flag: concordant, discordant, or mixed (when multiple kinases point in different directions). Present concordant and discordant pathways as separate categories in results, rather than baking a directional assumption into the score.

**5. Attribution confidence** --- Weight differently by cell-type identity, using continuous (not thresholded) confidence scores:

- For EM and Target nodes (receiver-side), a kinase attributed to L5 IT (the receiver) is making a direct claim: this kinase is active in the cell type where the substrate resides. Full weight.
- A kinase attributed only to Microglia-PVM (the sender) is making an indirect claim: the kinase is active in the sender, and we hypothesize it affects a substrate in the receiver. This requires an additional biological leap (e.g., that the kinase is secreted or operates extracellularly), which is uncommon. Apply a substantial discount (suggested 0.2--0.3x).
- Exception: for the Ligand node (sender-side), sender attribution is directly relevant.
- Let the graded attribution confidence propagate as a continuous weight rather than imposing a hard threshold.

**6. Null model** --- Recommended a dual permutation approach, both of which are necessary:

- **Primary null: permute kinase significance labels.** Shuffle NES/FDR across kinases while preserving the kinase-substrate graph. This tests: "given that these substrates are in the pathway, is the observed kinase activity evidence stronger than expected by chance?" This is the more conservative and defensible null.
- **Secondary null: permute kinase-substrate assignments** (preserving the degree distribution). This tests: "given the overall density of kinase-substrate relationships, is this pathway's connectivity to active kinases unusual?" This specifically addresses the promiscuity concern from Question 1.
- If a pathway is significant under both nulls, the evidence is robust. Significant only under the first null means the result is driven by which kinases are active (good). Significant only under the second means it is driven by unusual substrate connectivity (suspicious --- likely a hub artifact).
- Minimum 10,000 permutations. Compute empirical p-values with Benjamini-Hochberg correction across pathways. Report both null models and their concordance.

### Additional concerns raised by reviewer

**L5 IT cell count.** With only 37 nuclei (16 WT, 21 App), per-cell-type expression estimates are unreliable. Dropout noise at n = 16 is severe --- many genes will appear differentially expressed due to sampling variance alone. The reviewer recommended a bootstrap sensitivity analysis: resample L5 IT nuclei and recompute TPDS rankings across 500 iterations. If the top pathways are unstable across bootstraps, the expression-based ranking is unreliable and the kinase evidence becomes even more important as an orthogonal stabilizer.

**Detection threshold sensitivity.** The 50% threshold excludes the majority of the transcriptome. It would be valuable to report what happens at 20% detection (~20,000 estimated pathways) even if a more efficient enumeration strategy is needed. If the same pathways rank at the top under both thresholds, confidence is strengthened. If rankings are threshold-dependent, the results are fragile.

**Kinase-substrate prediction confidence.** The kldata substrate predictions are motif-based (from kinase-library) and have non-trivial false positive rates. If the kinase-substrate mapping includes a confidence score, it should be incorporated as an additional weight in the kinase support score, or at minimum used in a sensitivity analysis restricted to high-confidence substrate assignments.

## Implementation

Based on the consultation, the substrate-based external reranking was implemented as `adapters/compute_kinase_support.py` (Adapter 5.5). All configurable parameters are centralized in `config_integration.py`.

### Kinase support score per pathway

For each pathway P with nodes (Ligand, Receptor, EM, Target):

1. **Identify connected kinases.** For each EM and Target gene, look up kinases that phosphorylate it (from kldata via `build_substrate_kinase_map()`). Retain only kinases with MEA FDR < 0.25 for the App_4mo contrast. **Exclude kinases that are already pathway nodes** (deduplication --- Incytr's internal channel already scores them).

2. **Weight each kinase-substrate edge.** For kinase K connected to substrate gene G:

   ```
   edge_weight = |NES_K| x IDF_G x attribution_weight_K

   where:
     IDF_G              = 1 / log(N), N = #significant attributed kinases targeting G
                          (returns 1.0 when N <= 1)
     attribution_weight = max over relevant cell types of:
                          combined_score x cell_type_relevance
     cell_type_relevance:
       = 1.0   if K attributed to receiver (L5 IT)
       = 0.25  if K attributed only to sender (Microglia-PVM)
   ```

3. **Aggregate per pathway.** Take the **median** of edge weights = `kinase_support_score` (unsigned --- magnitude only, per reviewer recommendation). Median aggregation is robust to both hub-substrate inflation (many weak edges from promiscuous substrates like Srrm2 with 80+ kinases) and single-outlier dominance (one strong edge in a low-degree pathway). The sum is retained as `kinase_support_score_sum` for reference. Each pathway is annotated with:
   - Concordance flag: concordant / discordant / mixed / none (based on mean sign of NES vs sign of TPDS)
   - Number of distinct contributing kinases
   - Number of node kinases excluded (deduplication count)
   - Identity of top contributing kinases (semicolon-delimited)

   The choice of median over sum is empirically motivated. Under sum aggregation, the kinase support score correlates strongly with pathway degree (Spearman rho = 0.68 with n_kinases), meaning hub substrates dominate rankings. Under median aggregation, this correlation inverts (rho = -0.47) --- pathways with many kinases actually have *lower* per-edge quality, confirming that most kinases in high-degree pathways are uninformative noise. The bottom half of kinases per pathway contributes only ~10% of total edge weight across all degree buckets.

4. **Produce adjusted rankings.** Additive combination across a sweep of mixing weights:

   ```
   adjusted_score = TPDS + lambda x kinase_support_score
   ```

   for lambda in {0.1, 0.25, 0.5, 1.0, 2.0}. Each lambda produces a separate ranking.

#### Naming bridge

Three naming conventions are bridged by `common.load_mouse_gene_to_kinase_mapping()`:

| Context | Example | Convention |
|---|---|---|
| MEA / unified_attribution `kinase` column | MNK1, GSK3A | Kinase abbreviation |
| unified_attribution `gene_symbol` column | MKNK1, GSK3A | Human gene symbol (not always == abbreviation) |
| kldata `motif.geneName` / Incytr pathway nodes | Mknk1, Gsk3a | Mouse gene symbol |

### Statistical validation via dual permutation

The permutation tests are designed specifically for the median-aggregated score. Because median is not a linear operation, we cannot use a sparse matrix-vector shortcut --- instead, per-pathway edge structures are stored and medians recomputed per permutation. Pathways are grouped by degree for vectorized batch processing (10,000 iterations in ~2 minutes).

The null models draw from the **full MEA kinase universe** (311 kinases tested for App_4mo), not just the 134 significant+attributed kinases used in the observed score. This is critical: the original null design (shuffling NES among significant kinases) had low dynamic range because all significant kinases have similar |NES| by definition (range 1.09--2.5). The redesigned nulls compare observed scores against a background that includes non-significant kinases (|NES| range 0.8--1.15), providing the contrast needed to detect enrichment.

**Null 1 (enrichment null):** For each pathway with N edges, sample N kinases from the full 311-kinase MEA universe. Sampled kinases use their actual |NES| but a uniform attribution weight (median of observed weights). Tests: **"Does this pathway's median kinase evidence reflect enrichment for disease-significant, cell-type-attributed kinases, or could random kinases produce the same score?"**

**Null 2 (wiring null):** Reassign each pathway's edges to random kinases from the full MEA universe, keeping IDF coefficients from the original edges but sampling attribution weights from the observed distribution. Tests: **"Does the specific kinase-substrate wiring matter, or would random connections give a similar median?"**

Empirical p-values with Benjamini-Hochberg correction across all 5,761 pathways.

#### Permutation results (10,000 iterations)

Permutation results below were computed on the pre-imputation pathway set (3,544 pathways). They should be re-run on the expanded set (5,761 pathways) for updated significance calls, but the framework and interpretation remain the same.

- **Null 1 (enrichment):** 1,225 / 2,405 pathways significant at FDR < 0.25 (51%). Evenly distributed across degree buckets (34--56%), confirming median aggregation is not biased by hub size. These pathways are enriched for kinases that are specifically disease-activated and cell-type-attributed.

- **Null 2 (wiring):** 7 pathways significant at FDR < 0.25. This is the stringent test --- most pathways fail because their substrates connect to a broad pool of kinases, so random re-wiring produces similar medians. The 7 that pass have specific kinase-substrate connections that cannot be replicated by chance.

- **Both nulls:** 7 pathways pass both tests (all Null 2 significant pathways also pass Null 1). These form the highest-confidence set: Eea1, Tsc22d2, Rapgef5, Senp7, Satb2, Wdfy3 pathway targets with 4--21 contributing kinases. Five are concordant (kinase direction agrees with expression), two are discordant (potential inhibitory phosphorylation or compensatory feedback).

The practical framework for downstream use:
- **Null 1 FDR** gates which pathways receive a kinase boost
- **Null 2 FDR** identifies a high-confidence subset where specific kinase biology drives the signal
- `n_distinct_kinases` provides a third confidence dimension (convergent evidence from multiple independent kinases)

Gated behind `--permutations` flag (or `RUN_PERMUTATIONS=1` in `run_phase1.sh`).

### Sensitivity analyses

Computed alongside the main scoring:

1. **PhPDS_ps redundancy:** Spearman correlation between Incytr's internal PhPDS_ps and the external kinase support score. Result: **rho = -0.185** (p = 2.8e-36), confirming the two signals are not redundant. The negative correlation suggests that pathways with high internal phospho scores tend to have lower external kinase support, and vice versa --- the two channels capture different aspects of the phosphoproteomic landscape.

2. **IDF sensitivity:** Top-20 overlap between scores computed with and without IDF weighting. Result: **52.4% overlap** --- IDF changes about half of the top pathways, confirming that hub substrate discounting has material impact.

3. **Lambda sensitivity:** Kendall tau-b between rankings at adjacent lambda values. Results show smooth degradation from tau = 0.97 (lambda 0.1 vs 0.25) to tau = 0.88 (lambda 1.0 vs 2.0), indicating rankings stabilize at higher lambda values as kinase evidence dominates.

4. **Rank divergence from TPDS:** Kendall tau between TPDS-only ranking and adjusted ranking at each lambda:

   | Lambda | Tau vs TPDS |
   |--------|-------------|
   | 0.1 | 0.97 |
   | 0.25 | 0.93 |
   | 0.5 | 0.89 |
   | 1.0 | 0.81 |
   | 2.0 | 0.70 |

   Note: the median-aggregated score produces more conservative rank shifts than the sum-based score. This is expected: median values are smaller in magnitude than sums, so the lambda multiplier needs to be larger to achieve the same degree of reranking. The lambda sweep should be recalibrated for median aggregation if a specific target rank divergence is desired.

Additional sensitivity analyses requiring R/Incytr are implemented in `wrappers/bootstrap_sensitivity.R` (gated behind `RUN_BOOTSTRAP=1`):

5. **L5 IT bootstrap:** Resample L5 IT nuclei with replacement (500 iterations), rerun Incytr TPDS computation with fixed pathway structure, report rank stability (cv_rank, frac_in_top20/50).
6. **Detection threshold:** Run Incytr at 20% detection threshold; compare top-50 overlap with 50% results. Wrapped in tryCatch for OOM (20% produces ~20K pathways).

## Pipeline structure

The full pipeline is orchestrated by `run_phase1.sh`:

```
1. Python adapters (alzheimers env)
   export_expression.py          — snRNA-seq to sparse MTX + metadata
   export_kldata.py              — kinase-substrate reference from kinase-library
   export_kl_output.py           — MEA results as kinase-substrate pairs
   export_phospho.py             — attribution-weighted phospho per cell type
   export_kinase_imputed_genes.py — kinase-substrate-supported receiver genes

2. R wrappers (incytr env)
   run_incytr.R             — pathway inference (with kinase-imputed expansion),
                               expression + phospho + kinase scoring, pathway labeling
   postprocess.R            — sensitivity analysis, discordance detection, redundancy check

3. Kinase support scoring (alzheimers env)
   compute_kinase_support.py            — substrate-based reranking (always runs, ~30s)
   compute_kinase_support.py --permutations  — dual null models (optional, ~10-30 min)

4. Bootstrap sensitivity (incytr env, optional)
   bootstrap_sensitivity.R  — L5 IT bootstrap + threshold sensitivity
```

Steps 1--2 and the base scoring in step 3 always run. Kinase-imputed expansion runs by default (gated behind `ENABLE_KINASE_IMPUTATION=1`; set to `0` to disable). Permutation tests and bootstrap sensitivity are gated behind `RUN_PERMUTATIONS=1` and `RUN_BOOTSTRAP=1` environment variables.

### Key outputs

| File | Description |
|---|---|
| `intermediates/results_expronly.csv` | Expression-only pathway rankings (TPDS baseline) |
| `intermediates/results_full.csv` | Full Incytr integration (TPDS + phospho + kinase) |
| `intermediates/kinase_imputed_genes.csv` | Receiver genes admitted via kinase-substrate evidence |
| `intermediates/kinase_imputation_summary.csv` | Gene/pathway counts by evidence tier |
| `intermediates/kinase_support_scores.csv` | Per-pathway substrate-based kinase support scores |
| `intermediates/adjusted_rankings.csv` | Combined rankings at each lambda value |
| `intermediates/reranking_summary.json` | Summary statistics and sensitivity results |
| `intermediates/permutation_pvalues.csv` | Empirical p-values under dual null models (optional) |
| `intermediates/bootstrap_stability.csv` | Rank stability across L5 IT bootstrap (optional) |
| `intermediates/ranking_correlation.json` | Spearman rho between expression-only and full rankings |

Both `results_expronly.csv` and `results_full.csv` include `pathway_evidence` (expression-confirmed or kinase-imputed) and `imputed_nodes` (which positions were imputed) columns for downstream stratification.

## Summary

| What we want | Why it's hard | Resolution |
|---|---|---|
| Kinase activity evidence to influence pathway rankings | Incytr's internal kinase channel requires kinase genes to be pathway nodes, which requires them to pass the expression threshold, which most fail | Dual-channel architecture: internal channel for node-kinases, external substrate-based reranking for the rest, with deduplication at the boundary |
| An integration that respects both data types | Bulk kinase activity is tissue-level; Incytr expects cell-type-resolved data | Keep evidence types separate: expression inside Incytr, kinase evidence as external reranking layer weighted by cell-type attribution confidence |
| Pathway discovery beyond expression limits | The 50% detection threshold excludes genes with real kinase-substrate evidence; lowering the threshold indiscriminately creates millions of pathways | Kinase-imputed expansion: add genes with protein-level kinase-substrate evidence to the receiver gene list, label pathways by evidence tier, let SigProb naturally filter weak candidates |
| Statistical rigor for the combined ranking | Substrate promiscuity can inflate scores; two evidence layers share an upstream dataset | Median aggregation (hub-robust), IDF weighting, dual null model (enrichment null + wiring null against full MEA universe), redundancy check against PhPDS_ps (rho = -0.246, confirming complementarity) |
| Honest interpretation | Convergent evidence at a target gene does not prove mechanistic pathway flow | Frame as convergent functional evidence, not mechanistic validation; present concordant and discordant pathways as separate categories (unsigned score + concordance flag); label kinase-imputed pathways as lower evidence tier |

The core constraint: cell-type attributions of kinase activity are not precise, but they represent a reasonable assignment given available data (paired within-cohort snRNA-seq, cross-species SEA-AD concordance, WMB expression specificity). The integration is indirect but meaningful --- kinase evidence informs pathway interpretation through substrate relationships, weighted by attribution confidence and cell-type relevance, without being misrepresented as cell-type-resolved measurement.
