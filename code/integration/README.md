# Incytr Integration: Connecting Bulk Kinase Activity to Cell-Cell Signaling Pathways

## Background

We have two independent lines of evidence about kinase signaling in an Alzheimer's disease mouse model (App knock-in, 4-month males):

1. **Bulk tissue phosphoproteomics** — From 72-animal TMT proteomics, we compute stoichiometry-corrected phosphorylation (log2 phospho minus log2 parent protein abundance), then run motif enrichment analysis (MEA, GSEA-based) to identify kinases whose substrate phosphorylation patterns change in disease. This produces a normalized enrichment score (NES) and FDR per kinase per contrast. We then attribute each kinase to cell types using three evidence sources: SEA-AD human AD transcriptomic concordance, Whole Mouse Brain (WMB) expression specificity, and within-cohort paired snRNA-seq concordance.

2. **Single-cell transcriptomics** — From paired snRNA-seq of 28 animals in the same cohort (63K nuclei, Allen Cell Type Mapper annotations), we have cell-type-resolved gene expression. Incytr uses this to infer intercellular signaling pathways between specific cell-type pairs.

The goal is to integrate these two evidence types: use Incytr's cell-type-resolved expression to identify active signaling pathways between cell-type pairs, then use our bulk kinase activity evidence to assess which pathways have additional phosphoproteomic support.

## Scope

- **Cell types:** 22 subclasses (all SEA-AD-mapped subclasses with snRNA-seq coverage)
- **Pairs:** 462 sender-receiver combinations (22 x 21, excluding self-pairs)
- **Contrast:** WT vs App knock-in, 4 months, males only
- **Phase 1 reference pair:** Microglia-PVM (sender, 185 nuclei) -> L5 IT (receiver, 37 nuclei)

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
| 10% | 2,357 | 7,464 | ~26,400 (actual, after SigProb) |
| 20% | 1,127 | 4,994 | ~20,000 |
| 50% | 210 | 2,114 | 3,544 (actual) |

The current implementation uses **10%** with DuckDB-based enumeration and in-query SigProb filtering. At 10%, the raw pathway count is much larger than at 50%, but the SigProb cutoff (0.01) prunes pathways where any node has negligible expression, producing a similar final count. The 10% threshold admits more biologically relevant genes while DuckDB handles the combinatorial explosion efficiently (see "DuckDB pathway enumeration" below).

Note that the receiver gene list supplies three of the four node positions (Receptor, EM, and Target), which is why receiver gene count has a disproportionate effect on pathway count.

## What happened in Phase 1 (50% threshold, single pair)

The initial Phase 1 run used a 50% expression detection threshold and the single Microglia-PVM -> L5 IT pair. The key findings that motivated the current architecture:

### Kinase activity evidence was disconnected at 50% threshold

At 50% detection, only 14 kinase genes survived in the sender and 99 in the receiver. We have 114 kinases with significant MEA results attributed to Microglia-PVM or L5 IT at moderate-or-higher confidence. Most kinases were excluded from the gene pool, never became pathway nodes, and their activity evidence had nowhere to attach. Activity scores were zero for all 3,544 pathways.

### The integration barely changed rankings

Spearman rho between expression-only and full-integration pathway rankings: **0.9984**. The phospho and kinase evidence combined shifted the median pathway score by 2.4%. This motivated (1) lowering the threshold to 10%, (2) kinase-imputed pathway expansion, and (3) DuckDB enumeration to handle the resulting combinatorial increase.

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
2. The R pipeline reads this list and unions it with the expression-threshold receiver genes. The sender gene list stays at 10% (unchanged).
3. After enumeration, every pathway is labeled:
   - **expression-confirmed**: all 4 nodes (Ligand, Receptor, EM, Target) pass the 10% expression threshold in their respective cell type
   - **kinase-imputed**: one or more receiver-side nodes were admitted via kinase-substrate evidence
4. The `imputed_nodes` column records which specific positions were imputed (e.g., "Target", "EM;Target").

**Properties of kinase-imputed genes:**

The 986 additional receiver genes (at 10% threshold) are not phantom genes --- they have real mRNA signal, just below the detection threshold. Kinase-substrate evidence provides an independent reason to include them: the protein is being phosphorylated by a disease-activated kinase, so it is present and post-translationally active regardless of its mRNA abundance.

**Kinase-imputed pathway prevalence across all pairs:**

With DuckDB SigProb pre-filtering at 10% threshold, kinase-imputed pathways are rare: 0.33% of all pathways across 462 pairs. Most pairs have a small number (median 42 per pair), concentrated in specific receiver types (Microglia-PVM, VLMC, Endothelial). The SigProb cutoff naturally prunes kinase-imputed pathways where the imputed node has very low expression.

**Interpretation:** Kinase-imputed pathways represent a lower evidence tier than expression-confirmed pathways. The expression-confirmed label means the entire signaling chain has direct transcriptomic support. The kinase-imputed label means the chain is plausible based on expression at most nodes plus protein-level kinase activity evidence at the imputed node(s). Both tiers flow through the same scoring pipeline (TPDS, PDS, kinase integration), and downstream analysis can filter or stratify by the `pathway_evidence` column.

### Coverage (reference pair: Microglia-PVM -> L5 IT)

Of the 26,399 pathways after SigProb filtering (10% threshold):

- **21,199 pathways (80.3%)** have kinase evidence sources
- **5,200 pathways (19.7%)** have no kinase-substrate connection
- 114 kinases are attributed at moderate+ confidence to any cell type (all-pairs mode)

By evidence tier:

| Tier | Total | With kinase evidence | Coverage |
|---|---|---|---|
| Expression-confirmed | 26,392 | 21,192 | 80.3% |
| Kinase-imputed | 7 | 7 | 100% |

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

## DuckDB pathway enumeration

Incytr's built-in `pathway_inference()` uses data.table cartesian joins, which OOM at large gene counts. DuckDB-based enumeration (`wrappers/duckdb_enumeration.R`) replaces this with R-side edge pre-pruning + DuckDB 3-way join + inline SigProb filtering.

### How it works

1. **Edge pre-pruning**: For each edge layer (L1: Ligand-Receptor, L2: Receptor-EM, L3: EM-Target), compute per-condition Hill function values from the weighted-quantile expression. Discard edges where `max(Hill_WT, Hill_App) < cutoff_SigProb` (0.01). This reduces millions of edges to ~20K (L3), ~500 (L2), ~200 (L1).

2. **DuckDB 3-way join**: Register pruned edge tables in DuckDB. A single SQL query joins L1-L2-L3 with a WHERE clause requiring `SigProb_WT >= 0.01 OR SigProb_App >= 0.01`. DuckDB handles the join combinatorics in-process with disk spilling.

3. **EM promiscuity weighting**: Each pathway's SigProb includes an EM degree weight `1/log2(1+degree)` via LEFT JOIN, penalizing promiscuous effector molecules.

### Performance

For the reference pair (Microglia-PVM -> L5 IT at 10% threshold):
- Raw edge counts: L1=330, L2=685, L3=20,464
- After pre-pruning: L1=30, L2=202, L3=12,144
- DuckDB query: 26,399 pathways in ~2 seconds

### All-pairs receiver-centric pipeline

The all-pairs pipeline (`wrappers/run_incytr_all_pairs.R`) uses a two-phase architecture that exploits the receiver-centric structure of Incytr pathways (3 of 4 nodes — Receptor, EM, Target — are receiver-determined):

- **Phase A+B (DuckDB enumeration)**: Outer loop over 22 receivers prunes L2/L3 edges once per receiver, then inner loop over 21 senders prunes L1 edges and runs the DuckDB 3-way join. This produces a unified `all_pathways_df` data.table with `sender` as a column.

- **Phase C (vectorized scoring)**: All senders for a given receiver are scored in a single vectorized pass (`wrappers/receiver_scoring.R`). Receiver-side computations (expression lookups, fold changes, phospho normalization, SiK/EI, kinase activity) are computed once and shared across all 21 senders. Only sender-side computations (Ligand expression, Ligand fold change, per-sender SigProb threshold) vary. This eliminates 440 redundant receiver-side computations and avoids creating 462 Incytr S4 objects.

Output is 22 receiver-indexed Parquet files (`recv_{receiver}.parquet`) with `sender` as a column, replacing the previous 462 per-pair CSV structure.

### All-pairs results (462 pairs)

| Metric | Value |
|---|---|
| Pairs processed | 462 |
| Total pathways (pre-SigProb) | 42,207,765 |
| Pathways per pair (median) | 93,167 |
| Pathways per pair (range) | 3,682 -- 252,699 |
| Pairs with 0 pathways | 0 |
| Total runtime | 318 minutes |

#### Edge weight distributions

| Layer | Mean edges/pair | Median weight | Max weight |
|---|---|---|---|
| L1 (Ligand->Receptor) | 192 | 138 pathways | 10,198 |
| L2 (Receptor->EM) | 486 | 30 | 12,168 |
| L3 (EM->Target) | 19,106 | 3.4 | 73 |

L3 generates the combinatorial explosion (19K edges x 3-5 pathways each). L1 edges are fewer but heavier (hub ligand-receptor pairs carry thousands of pathways).

#### SigProb and score distributions

Most pathways have low SigProb (80% below 0.05, only 0.5% above 0.5). The SigProb cutoff at 0.01 is permissive --- essentially passing everything with non-zero expression at all four nodes.

PDS scores are bimodal with peaks at -0.75 and +0.75. 71% of pathways have |PDS| > 0.5 (strong directionality).

### kinase_boost column

Each receiver Parquet file includes a `kinase_boost` column: `PDS - TPDS`. This directly measures how much kinase and phospho evidence changed the pathway score relative to expression alone. Across all pairs:

- 31% of pathways: kinase_boost = 0 (no kinase effect)
- 42% have |kinase_boost| > 0.01 (moderate shift)
- 1.6% have |kinase_boost| > 0.1 (strong shift)
- Top boosted pathways route through kinase-node EMs (e.g., Mapk8), with boosts up to +0.27

### Checkpoint/restart and operational features

- **Checkpoint**: Receivers with existing `recv_{receiver}.parquet` are skipped. Set `FORCE_RERUN=1` to override.
- **Memory guard**: R memory is checked after each receiver. Aborts cleanly if > `MEMORY_LIMIT_GB` (default 10).
- **Pair filter**: `PAIR_FILTER="Microglia-PVM:L5 IT"` for single-pair testing; `PAIR_FILTER="*:L5 IT"` for single-receiver.
- **Shell runner**: `run_all_pairs.sh` runs Python adapters, R pipeline under `systemd-run --user --scope -p MemoryMax=12G`, kinase support scoring, and cross-pair aggregation. Use `--skip-adapters` on checkpoint-resume.

### Output structure

```
intermediates/all_pairs/
  recv_{receiver}.parquet          (22 files, sender as column, all scores included)
  {sender}__{receiver}/            (462 subdirectories)
    kinase_support_scores.csv      substrate-based kinase support scores (per pathway)
    adjusted_rankings.csv          lambda-sweep adjusted rankings
    reranking_summary.json         per-pair scoring statistics
  pair_summary.csv                 462-row summary (n_pathways, timing, status)
  kinase_support_summary.csv       462-row kinase support summary
  aggregation/
    backbone_recurrence.csv        R-EM-T triples shared across senders
    hub_matrix.csv                 22x22 sender x receiver signaling summary (long format)
    hub_matrix_wide.csv            22x22 pivoted (mean_abs_pds)
    target_convergence.csv         genes targeted by multiple senders/routes
    aggregation_metadata.json      params, thresholds, row counts, timestamp
```

The Parquet files are the sole data format for pathway results. Each contains all pathways for one receiver with `sender` as a column, enabling predicate pushdown for efficient per-pair queries. File-level metadata includes receiver name, pipeline version, and timestamp. The kinase support scoring adapter (`compute_kinase_support_all_pairs.py`) reads Parquet files with predicate pushdown. The cross-pair aggregation (`aggregate_cross_pair.py`) reads all 22 Parquet files via DuckDB glob into a single view for SQL-based analysis.

## Substrate-based external reranking

Based on the consultation, the substrate-based external reranking was implemented as `adapters/compute_kinase_support.py` (single-pair) and `adapters/compute_kinase_support_all_pairs.py` (all-pairs orchestrator). All configurable parameters are centralized in `config_integration.py`.

### Kinase support score per pathway

For each pathway P with nodes (Ligand, Receptor, EM, Target):

1. **Identify connected kinases.** For each EM and Target gene, look up kinases that phosphorylate it (from kldata via `build_substrate_kinase_map()`). Retain only kinases with MEA FDR < 0.25 for the App_4mo contrast. **Exclude kinases that are already pathway nodes** (deduplication --- Incytr's internal channel already scores them).

2. **Weight each kinase-substrate edge.** For kinase K connected to substrate gene G:

   ```
   edge_weight = |NES_K| x IDF_G x attribution_weight_K

   where:
     IDF_G              = 1 / log(N), N = #significant kinases targeting G
                          (pair-independent: counts all MEA-significant kinases,
                           not just those attributed to the pair's cell types;
                           returns 1.0 when N <= 1)
     attribution_weight = max over relevant cell types of:
                          combined_score x cell_type_relevance
     cell_type_relevance:
       = 1.0   if K attributed to receiver
       = 0.25  if K attributed only to sender (SENDER_ATTRIBUTION_DISCOUNT)
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

Permutation results below were computed on the original Phase 1 pathway set (3,544 pathways at 50% threshold). They should be re-run on the current expanded set (26,399 pathways at 10% threshold for the reference pair, or across all 462 pairs) for updated significance calls. The framework and interpretation remain the same.

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

### Single-pair pipeline (`run_phase1.sh`)

```
1. Python adapters (alzheimers env)
   export_expression.py          — snRNA-seq to sparse MTX + metadata
   export_kldata.py              — kinase-substrate reference from kinase-library
   export_kl_output.py           — MEA results as kinase-substrate pairs
   export_phospho.py             — attribution-weighted phospho per cell type
   export_kinase_imputed_genes.py — kinase-substrate-supported receiver genes

2. R wrappers (incytr env)
   run_incytr.R             — DuckDB enumeration, expression + phospho + kinase scoring
   postprocess.R            — sensitivity analysis, discordance detection, redundancy check

3. Kinase support scoring (alzheimers env)
   compute_kinase_support.py            — substrate-based reranking (always runs, ~30s)
   compute_kinase_support.py --permutations  — dual null models (optional, ~10-30 min)

4. Bootstrap sensitivity (incytr env, optional)
   bootstrap_sensitivity.R  — L5 IT bootstrap + threshold sensitivity
```

### All-pairs pipeline (`run_all_pairs.sh`)

```
1. Python adapters (alzheimers env) — same as single-pair, but:
   export_kl_output.py --all-pairs  — include kinases attributed to ANY cell type

2. R all-pairs orchestrator (incytr env)
   run_incytr_all_pairs.R   — Phase A+B: DuckDB backbone enumeration per receiver
                               Phase C: vectorized scoring via receiver_scoring.R
                               Output: 22 receiver Parquet files
   Runs under: systemd-run --user --scope -p MemoryMax=12G

3. Kinase support scoring (alzheimers env)
   compute_kinase_support_all_pairs.py — substrate-based reranking across all 462 pairs (~8 min)
                                          Parquet input with predicate pushdown,
                                          precomputed edge table + pair-independent IDF,
                                          per-pair attribution weights, checkpoint/restart

4. Cross-pair aggregation (alzheimers env)
   aggregate_cross_pair.py  — DuckDB aggregation over 22 receiver Parquet files
                               3a: backbone recurrence (R-EM-T triples across senders)
                               3b: 22x22 cell-type hub matrix
                               3c: target gene convergence
```

Use `--skip-adapters` to skip Python adapters on checkpoint-resume. Kinase-imputed expansion runs by default (gated behind `ENABLE_KINASE_IMPUTATION=1`; set to `0` to disable). Permutation tests and bootstrap sensitivity are gated behind `RUN_PERMUTATIONS=1` and `RUN_BOOTSTRAP=1` environment variables. The kinase support step forwards `PAIR_FILTER` and `FORCE_RERUN` env vars from the shell runner.

### Key outputs

**Single-pair** (`intermediates/`):

| File | Description |
|---|---|
| `results_expronly.csv` | Expression-only pathway rankings (TPDS baseline) |
| `results_full.csv` | Full Incytr integration (TPDS + phospho + kinase + kinase_boost) |
| `kinase_imputed_genes.csv` | Receiver genes admitted via kinase-substrate evidence |
| `kinase_support_scores.csv` | Per-pathway substrate-based kinase support scores |
| `edge_list_l1.csv`, `l2`, `l3` | Per-layer edge lists with pathway counts |

**All-pairs** (`intermediates/all_pairs/`):

| File | Description |
|---|---|
| `recv_{receiver}.parquet` | Receiver-indexed Parquet (sender as column, all scores + kinase_boost) |
| `{sender}__{receiver}/kinase_support_scores.csv` | Per-pathway substrate-based kinase support scores |
| `{sender}__{receiver}/adjusted_rankings.csv` | Lambda-sweep adjusted rankings (TPDS + λ × score) |
| `{sender}__{receiver}/reranking_summary.json` | Per-pair scoring statistics and timing |
| `pair_summary.csv` | 462-row summary: sender, receiver, n_pre, n_post, time_sec, status |
| `kinase_support_summary.csv` | 462-row kinase support summary (n_pathways, n_nonzero, median_score) |
| `aggregation/backbone_recurrence.csv` | R-EM-T triples shared across senders, with direction consistency |
| `aggregation/hub_matrix.csv` | 22×22 sender × receiver signaling summary (long format) |
| `aggregation/hub_matrix_wide.csv` | 22×22 pivoted hub matrix (mean |PDS|) |
| `aggregation/target_convergence.csv` | Per-receiver target genes with sender/route convergence |
| `aggregation/aggregation_metadata.json` | Aggregation parameters, row counts, timestamp |

All results include `pathway_evidence` (expression-confirmed or kinase-imputed), `imputed_nodes` (which positions were imputed), and `kinase_boost` (PDS - TPDS) columns for downstream stratification.

## Summary

| What we want | Why it's hard | Resolution |
|---|---|---|
| Kinase activity evidence to influence pathway rankings | Incytr's internal kinase channel requires kinase genes to be pathway nodes, which requires them to pass the expression threshold, which most fail | Dual-channel architecture: internal channel for node-kinases, external substrate-based reranking for the rest, with deduplication at the boundary |
| An integration that respects both data types | Bulk kinase activity is tissue-level; Incytr expects cell-type-resolved data | Keep evidence types separate: expression inside Incytr, kinase evidence as external reranking layer weighted by cell-type attribution confidence |
| Pathway discovery beyond expression limits | High detection thresholds exclude genes with real kinase-substrate evidence; lowering the threshold indiscriminately creates millions of pathways | DuckDB enumeration with in-query SigProb filtering handles the combinatorial explosion at 10% threshold; kinase-imputed expansion adds genes with protein-level evidence; SigProb naturally filters weak candidates |
| Scale to all cell-type pairs | Single-pair processing is ~3 min but doesn't share work across pairs; 462 independent runs would take ~23 hours with redundant computation | Two-phase receiver-centric architecture: DuckDB enumeration shares L2/L3 pruning per receiver, vectorized scoring eliminates 440 redundant receiver-side computations and 462 S4 object copies. Output as 22 receiver-indexed Parquet files with predicate pushdown. Cross-pair aggregation via DuckDB over Parquet (backbone recurrence, hub matrix, target convergence). Checkpoint/restart per receiver, memory guard under systemd-run 12GB cap |
| Statistical rigor for the combined ranking | Substrate promiscuity can inflate scores; two evidence layers share an upstream dataset | Median aggregation (hub-robust), pair-independent IDF weighting (computed once, reused across pairs), dual null model (enrichment null + wiring null against full MEA universe), redundancy check against PhPDS_ps (rho = -0.246, confirming complementarity) |
| Transparent kinase contribution | PDS integrates multiple evidence types opaquely | `kinase_boost` column (PDS - TPDS) directly measures the kinase/phospho contribution per pathway |
| Honest interpretation | Convergent evidence at a target gene does not prove mechanistic pathway flow | Frame as convergent functional evidence, not mechanistic validation; present concordant and discordant pathways as separate categories (unsigned score + concordance flag); label kinase-imputed pathways as lower evidence tier |

The core constraint: cell-type attributions of kinase activity are not precise, but they represent a reasonable assignment given available data (paired within-cohort snRNA-seq, cross-species SEA-AD concordance, WMB expression specificity). The integration is indirect but meaningful --- kinase evidence informs pathway interpretation through substrate relationships, weighted by attribution confidence and cell-type relevance, without being misrepresented as cell-type-resolved measurement.

## Next steps

### 1. All-pairs substrate-based reranking --- DONE

The external substrate-based kinase support scoring has been extended to all 462 sender-receiver pairs via `compute_kinase_support_all_pairs.py`. This is integrated into `run_all_pairs.sh` as a third stage after the R pipeline.

**Key optimizations over the single-pair implementation:**

- **Pair-independent IDF:** The IDF term (`1/log(N_sig)` where N_sig = number of significant kinases targeting each substrate) is computed once from the full MEA universe and reused across all pairs. Substrate promiscuity is an intrinsic property, not a pair-dependent one --- the old pair-dependent IDF conflated promiscuity with cell-type relevance (already handled by the multiplicative `attribution_weight` term).

- **Precomputed edge table:** Pair-independent substrate→kinase edge arrays (|NES|×IDF products, NES signs, kinase identities) are built once from kldata + MEA results. Per-pair processing applies attribution weights via `apply_pair_weights()` without recomputing the substrate-kinase graph.

- **Fast/slow path split:** 95%+ of pathways have no kinase-gene nodes, so the deduplication check (excluding kinases already scored by Incytr's internal channel) is skipped entirely for the fast path. Only pathways with kinase-gene nodes (EM or Target) take the slow path.

- **Python-native median:** `sorted()` + index for 6--10 element arrays is 47x faster than `np.median` due to numpy's per-call type-checking and nan-handling overhead on small arrays.

- **Profiled performance:** Reference pair (26K pathways): 0.15s scoring, 0.5s total. Estimated full run: ~8 minutes for all 462 pairs (vs ~135 min with the naive iterrows approach).

**Per-pair outputs** (written to each `{sender}__{receiver}/` directory):
- `kinase_support_scores.csv` --- per-pathway substrate-based kinase support scores
- `adjusted_rankings.csv` --- lambda-sweep adjusted rankings (TPDS + λ × kinase_support)
- `reranking_summary.json` --- per-pair statistics (n_pathways, n_nonzero, median_score, timing)

**Cross-pair output:** `kinase_support_summary.csv` (462-row summary, one row per pair).

**CLI:** `python compute_kinase_support_all_pairs.py [--profile-pair DIRNAME] [--force] [--no-sensitivity] [--pair-filter PATTERN]`

Checkpoint/restart is supported: pairs with existing `kinase_support_scores.csv` are skipped unless `--force` is passed.

### 2. Cross-pair aggregation and analysis --- DONE

Implemented as `aggregate_cross_pair.py`, which reads all 22 `recv_*.parquet` files via DuckDB glob into a single view and produces three aggregation outputs:

- **3a. Backbone recurrence** (`backbone_recurrence.csv`): Groups pathways by receiver × Receptor × EM × Target triple and counts how many senders contribute each backbone. Includes `n_senders_significant` (senders where |PDS| exceeds threshold), direction consistency (fraction of senders agreeing with the group mean PDS sign), and mean kinase_boost. Pathways consistently altered across many senders are stronger findings than single-pair results.

- **3b. Cell-type hub matrix** (`hub_matrix.csv`, `hub_matrix_wide.csv`): 22×22 sender × receiver signaling summary. Per pair: pathway count, number/fraction significant, mean |PDS|, mean PDS (signed), mean kinase_boost, and up/down-regulated counts. The wide-format pivot provides a heatmap-ready matrix of mean |PDS| across all cell-type pairs.

- **3c. Target gene convergence** (`target_convergence.csv`): Per-receiver target genes ranked by how many senders and distinct signaling routes (Receptor::EM combinations) converge on each target. Includes per-sender mean PDS aggregated to gene level, identifying genes at the convergence point of multiple cell-type signaling routes.

**CLI:** `python aggregate_cross_pair.py [--pds-threshold 0.1] [--receiver NAME] [--force]`

Configurable via `config_integration.py`: `PDS_SIGNIFICANCE_THRESHOLD` (default 0.1) and `AGGREGATION_DIR`. Integrated into `run_all_pairs.sh` as stage 4.

### 3. Updated permutation tests

The permutation results in this document are from Phase 1 (3,544 pathways, 50% threshold, single pair). They need re-running:

- The current reference pair has 26,399 pathways at 10% threshold. Degree distributions, IDF values, and null distributions will differ. The 51% enrichment-null significance rate from Phase 1 may not hold.

- At all-pairs scale (462 pairs), running independent permutations per pair creates a massive multiple testing burden. An alternative is to run permutations on the aggregated cross-pair results (e.g., recurrence-weighted scores), which is a different statistical question but potentially more defensible.

The framework is implemented and ready; it needs re-execution on the current data.

### 4. Nested experimental design: beyond a single contrast

The all-pairs pipeline currently runs on a single contrast: WT vs App knock-in, 4-month males. This is one cell of a factorial design that spans 3 genotypes (WT, App, Tau, plus the App×Tau interaction), 3 timepoints (2mo, 4mo, 6mo), and 2 sexes (with males as primary). The bulk phosphoproteomics pipeline produces 9 time-resolved contrasts from the full design.

Running Incytr on a single contrast means:

- **42 million pathways describe one comparison.** The 462 sender-receiver pairs × ~93K pathways/pair represent a single snapshot (App_4mo). Extending to all 9 contrasts would produce ~380 million pathway evaluations, each requiring snRNA-seq expression estimates stratified by condition --- which may not be available for all contrast arms given the snRNA-seq sample composition (28 animals, not all genotype×timepoint cells equally populated).

- **The contrasts are not independent.** The 9 contrasts share animals (the WT baseline is the same across App, Tau, and interaction contrasts at each timepoint), share the same snRNA-seq expression data (Incytr's TPDS is computed from the same nuclei), and share the same kinase-substrate reference. Treating each contrast's Incytr results as independent observations would overstate the evidence.

- **Temporal and genotype structure is informative.** A pathway that is active in App_4mo but not App_2mo or App_6mo tells a different biological story than one that is consistently active across all timepoints. Similarly, pathways active in both App and Tau contrasts vs. those specific to one genotype have different implications for disease mechanism. The current single-contrast approach cannot capture these dynamics.

Addressing this requires thinking about how to either (a) run Incytr across multiple contrasts and handle the non-independence in downstream analysis, or (b) define a summary statistic across contrasts that the Incytr integration can target. This is an architectural question that interacts with snRNA-seq sample availability and the factorial OLS structure in the bulk pipeline.

## Known limitations and future robustness checks

The pipeline is explicitly hypothesis-generating. Cell-type attribution of kinase activity is correlational (SEA-AD concordance, WMB specificity, within-cohort snRNA-seq concordance), and we accept this. The following concerns go beyond attribution and affect the biological inference at other stages. They are ranked by a combination of impact on conclusions and practical addressability.

### Tier 1: Significant and addressable

**SigProb cutoff is nearly vacuous.** The SigProb threshold of 0.01 passes essentially everything with non-zero expression at all four nodes --- 80% of pathways have SigProb below 0.05. This means ~93K pathways per pair are carried forward with minimal biological filtering, and every downstream score (TPDS, kinase support, permutation tests) operates on a set that is mostly combinatorial noise. The permutation tests are also computed against this inflated background, diluting statistical power. *Action:* Sweep SigProb cutoffs (0.01, 0.05, 0.10, 0.20) and report how the significant pathway set and top-ranked results change. Consider adding a secondary filter (minimum |TPDS| magnitude) to reduce the pathway set before kinase support scoring.

**Small per-condition cell counts in many receiver types.** The 37 L5 IT nuclei (16 WT, 21 App) problem generalizes --- many receiver subclasses have thin per-condition representation, and TPDS inherits that noise at every pathway node. Dropout noise at small n produces spurious differential expression that propagates into pathway rankings. *Action:* Run the bootstrap sensitivity analysis (already implemented in `bootstrap_sensitivity.R`) at scale across receiver types. Flag or exclude receiver types below a minimum per-condition nuclei threshold rather than treating all 22 subclasses as equal-confidence results. Report rank stability (cv_rank, frac_in_top20) per receiver type.

**Shared animals between the two evidence streams.** The 28 snRNA-seq animals are a subset of the 72 TMT proteomics animals. Song concordance scores (used for kinase attribution weights) are derived from the same biological samples generating the TPDS rankings being reranked. This is not full circularity --- the measurements are different modalities (mRNA vs phosphoprotein) --- but it means the two streams are not statistically independent. *Action:* Report results with and without Song concordance contributing to attribution weights (SEA-AD + WMB only). If kinase support scores are materially unchanged, the dependence is immaterial. If they change, the circularity should be disclosed and the Song-excluded results treated as the conservative estimate.

### Tier 2: Significant but inherent to the approach

**Unsigned kinase support score obscures mechanistic interpretation.** The score is unsigned by design (per molecular biology reviewer recommendation, to avoid discarding inhibitory phosphorylation). But this means the primary ranking conflates opposite biology: a pathway boosted by concordant kinases (activated kinase, upregulated pathway) and one boosted by discordant kinases (activated kinase, downregulated pathway) receive the same score magnitude. The concordance flag exists as metadata but is not reflected in the ranking. *Mitigation:* Downstream analyses that use ranked pathways (top-N lists, cross-pair aggregation, cell-type hub summaries) must condition on the concordance flag. Any summary that does not stratify by concordance direction mixes biologically opposite signals.

**IncytrDB constrains discoverable biology.** Pathway enumeration is limited to four-gene chains present in IncytrDB's curated interaction catalog. Non-canonical interactions, recently discovered pathways, and chains with different architecture (longer, branching, or shorter than four genes) are invisible. This is the same constraint as using any curated gene set database (GO, KEGG, Reactome) for enrichment --- the pipeline prioritizes known templates, it does not discover novel signaling architectures. *Mitigation:* State explicitly in any manuscript that the integration identifies convergent evidence within known signaling frameworks, not novel pathway discovery.

### Tier 3: Real but acceptable on current merits

**Motif-based substrate predictions (kldata).** The external reranking channel rests on kinase-library motif predictions, which have non-trivial false positive rates. However, kldata is the best available kinase-substrate resource at this scale, and the pipeline includes structural mitigations (IDF weighting, median aggregation, dual permutation nulls) that limit the influence of individual false edges. *Optional future check:* If kldata provides a per-prediction confidence score, run a sensitivity analysis restricted to high-confidence substrate assignments and report whether top pathways are stable.

**Sender attribution discount (0.25x) is a point estimate.** The discount for sender-attributed kinases is based on reviewer judgment (0.2--0.3x range), not derived from data. The biological logic is sound (intracellular kinases in the sender rarely phosphorylate receiver-side substrates), and most kinase evidence connects through receiver-attributed kinases anyway, limiting the parameter's influence. *Optional future check:* Sweep sender discount (0.1--0.5x) and report which pathways are sensitive to this parameter.

**Median aggregation for low-degree pathways.** For pathways with 1--3 contributing kinases, the median has no robustness advantage over other aggregation methods and is dominated by a single edge. However, these are also the pathways where any aggregation method carries minimal information --- a single contributing kinase is a single contributing kinase regardless of how it is summarized. The practical impact is limited to interpretation: low-degree pathways with one strong kinase edge are boosted proportionally, which is reasonable for hypothesis generation.
