# Receiver-Centric Pathway Enumeration: Refactor Plan

Refactor the all-pairs Incytr pipeline from pair-centric (462 independent loops) to receiver-centric (22 backbone enumerations with sender as a dimension). This eliminates redundant computation, produces a unified output structure, and makes cross-pair aggregation a natural byproduct rather than a separate post-hoc step.

## Motivation

### The structural asymmetry

An Incytr pathway is a 4-gene chain: **L → R → EM → T**

- **L** (Ligand): expressed by the **sender** cell type
- **R, EM, T** (Receptor, Effector Molecule, Target): all expressed by the **receiver** cell type

Three of four nodes are receiver-determined. The **backbone** (R→EM→T) is entirely a property of the receiver --- it does not change when the sender changes. Only the first link (which ligands connect to which receptors via L1 edges) depends on the sender.

### Measured redundancy

For Chandelier as receiver (3.7M total pathways across 21 senders):

| Metric | Value |
|--------|-------|
| Unique R-EM-T backbones | 471,682 |
| Total L-R-EM-T pathways | 3,712,905 |
| Backbone reuse factor | 7.9x |
| Backbones in 10+ senders | 183,551 (39%) |
| Backbones in all 21 senders | 3,301 (0.7%) |
| Backbones in 1 sender only | 129,833 (28%) |

The current architecture runs the DuckDB L2×L3 join **21 times per receiver** (once per sender), even though L2 and L3 are receiver-only tables that do not change between senders. DuckDB has no persistent intermediate results between queries --- each sender query re-traverses the identical backbone join.

### SigProb decomposes cleanly

The pathway significance check factorizes:

```
SigProb = Hill(L×R) × Hill(R×EM) × Hill(EM×T×em_weight)
          ╰──────╯   ╰────────────────────────────────╯
          sender-dep       receiver-only component
```

Since Hill ∈ [0, 1], if the receiver component `Hill_L2 × Hill_L3 < cutoff`, no sender can rescue the backbone. Pre-filtering backbones by receiver-only SigProb is **lossless** --- zero pathways are missed versus the current approach.

## Architecture overview

### Current: pair-centric (462 loops)

```
for receiver in 22 cell types:
    prune L2/L3 for this receiver
    register L2, L3, receiver_expr in DuckDB
    
    for sender in 21 other cell types:          # 462 iterations
        prune L1 for this sender
        register L1, sender_expr in DuckDB
        RUN 4-WAY JOIN (L1×L2×L3×expr)          # re-traverses L2×L3 backbone
        create Incytr object
        Expr_bygroup → Cal_SigProb → Cal_scFC → Pathway_evaluation → Cal_PDS
        write results_full.csv to pair directory
```

Output: 462 directories, each with independent results_full.csv.

### Proposed: receiver-centric (22 + 22 queries)

```
for receiver in 22 cell types:
    PHASE A: backbone enumeration (1 query)
        L2×L3 join → all R-EM-T triples with receiver SigProb components
    
    PHASE B: all-sender ligand attachment (1 query)
        register ALL sender expressions as long-format table
        join sender_ligands × backbones ON Receptor
        apply full SigProb filter
        → unified pathway table with sender as a column
    
    PHASE C: vectorized scoring
        pre-compute receiver-side expression/FC (shared across senders)
        per-sender: attach sender-side expression/FC
        vectorized SigProb, scFC, pathway evaluation, PDS
    
    write receiver_{name}.parquet (all senders in one table)
```

Output: 22 receiver-indexed files. Cross-pair aggregation = GROUP BY.

## Implementation phases

### Phase 1: Backbone enumeration + sender attachment (immediate)

**Goal:** Replace the per-sender DuckDB query with backbone-first enumeration. Keep per-pair downstream scoring (Incytr functions) unchanged.

**Changes to `run_incytr_all_pairs.R`:**

1. **New function: `enumerate_backbones()`**
   - Input: pruned L2, L3, receiver expression, em_degree
   - SQL: 3-way join (L2→L3→receiver_expr) with receiver-side SigProb pre-computation
   - Output: data.table of (Receptor, EM, Target, h_recv_c1, h_recv_c2, h_l3_c1, h_l3_c2)
   - Filter: `h_recv_c1 * h_l3_c1 >= cutoff OR h_recv_c2 * h_l3_c2 >= cutoff`

2. **New function: `attach_all_senders()`**
   - Input: backbone table, all sender expressions, L1 edges
   - Build a long-format sender expression table (gene, cell_type, c1, c2)
   - SQL: join L1 × backbones × sender_expr with full SigProb filter
   - Output: data.table of (sender, Ligand, Receptor, EM, Target) --- all pathways for all senders of this receiver in one table

3. **Modified main loop:**
   - Outer loop still iterates over 22 receivers
   - Call `enumerate_backbones()` once per receiver
   - Call `attach_all_senders()` once per receiver
   - Inner loop iterates over senders, but only to filter the pre-enumerated table and run downstream scoring (no DuckDB query per sender)

**What stays the same:**
- Per-pair Incytr object creation
- `Expr_bygroup`, `Cal_SigProb`, `Cal_scFC`, `Pathway_evaluation`, `Cal_PDS`
- Per-pair output files (462 directories preserved for compatibility)
- Checkpoint/restart logic

**Expected gains:**
- DuckDB queries: 462 → 44 (22 backbone + 22 attachment)
- L2×L3 join: 462 → 22 executions
- Total enumeration time: estimated 40-60% reduction (backbone join is the expensive part; downstream scoring is unchanged)

**Verification:** Per-pair pathway counts must match the current pipeline exactly. Run both approaches on a subset of receivers and diff the results.

### Phase 2: Vectorized scoring (dedicated sprint)

**Goal:** Replace per-pair Incytr S4 object creation + downstream scoring with vectorized operations over the unified receiver table.

**Key insight:** For a given receiver, the receiver-side computation in every downstream function is identical across senders. Only the sender-side (Ligand expression, Ligand fold-change) varies.

#### 2a. Replace `Expr_bygroup` with pre-computed lookup

Current: `Expr_bygroup()` calls `compute_group_expr()` which computes weighted-quantile expression for all pathway genes in the sender and receiver cell types.

Refactor: The all-pairs pipeline already pre-computes `wq_expr[[ct]][[cond]]` for all 22 cell types (Section 3 of `run_incytr_all_pairs.R`). `Expr_bygroup` re-derives what we already have. Replace it with direct lookup into the pre-computed vectors.

For the unified receiver table:
- Receiver expression for R, EM, T: same for all senders → compute once, broadcast
- Sender expression for L: varies per sender → lookup from `wq_expr[[sender]][[cond]]`

Output: per-pathway per-condition expression for all 4 nodes, as columns in the unified table.

#### 2b. Replace `Cal_SigProb` with vectorized Hill computation

Current: `Cal_SigProb()` calls `compute_sigprob()` which computes Hill products per pathway.

Refactor: Already have receiver-side Hill components from backbone enumeration. Compute sender-side Hill component per (sender, Ligand, Receptor) triple. Multiply:

```r
SigProb_c1 = Hill_L1_c1 * h_recv_c1 * h_l3_c1
SigProb_c2 = Hill_L1_c2 * h_recv_c2 * h_l3_c2
```

This is a vectorized multiply over the full receiver table --- no per-pair loop needed.

Also pre-compute `em_target_weight` once per receiver (it depends only on em_degree and edge_source_count, which are receiver-only).

#### 2c. Replace `Cal_scFC` with vectorized fold-change

Current: `Cal_scFC()` computes log2FC per gene for sender and receiver cell types.

Refactor: Pre-compute per-gene log2FC for all 22 cell types:

```r
scFC <- list()
for (ct in cell_types) {
  scFC[[ct]] <- Cal_foldchange(data.frame(
    gene_symbol = all_genes,
    condition1 = wq_expr[[ct]][["WT"]],
    condition2 = wq_expr[[ct]][["App"]]
  ))
}
```

Then for each pathway row, look up:
- `Ligand_sclog2FC = scFC[[sender]][Ligand]`
- `Receptor_sclog2FC = scFC[[receiver]][Receptor]` (same for all senders)
- `EM_sclog2FC = scFC[[receiver]][EM]` (same for all senders)
- `Target_sclog2FC = scFC[[receiver]][Target]` (same for all senders)

Receiver-side FCs computed once per backbone, broadcast. Sender-side FC is per (sender, Ligand).

#### 2d. Replace `Pathway_evaluation` with vectorized scoring

Current: `Pathway_evaluation()` combines SigProb, scFC, and optional multi-omics into a TPDS score via weighted logistic combination.

Refactor: The scoring formula is:
```
TPDS = weighted_logistic(SigProb_log2FC, sc_up, sc_down, ps_up, ps_down, ...)
```

Where `sc_up/sc_down` count concordant/discordant fold-change directions and `ps_up/ps_down` count phospho concordances. All inputs are already available as columns --- the logistic is an elementwise transformation.

The score weights (`score.weight`) are constant across pairs. This entire function becomes a vectorized column operation.

#### 2e. Replace `Integr_multiomics` with column operations

Current: `Integr_multiomics()` adds phospho fold-change columns per pathway node.

Refactor: Phospho data (`ps_condition1/2.csv`) has columns per cell type. For each pathway:
- `Ligand_ps = ps[[sender_col]][Ligand]` (varies by sender)
- `Receptor_ps = ps[[receiver_col]][Receptor]` (constant per receiver)
- `EM_ps = ps[[receiver_col]][EM]` (constant per receiver)
- `Target_ps = ps[[receiver_col]][Target]` (constant per receiver)

Same broadcast pattern: receiver-side phospho computed once, sender-side per (sender, Ligand).

#### 2f. Replace `Integr_kinasedata` + `Integr_kinase_enrichment` with batched operations

Current: Per-pair, filter kldata to pathway genes, compute SiK scores and EI.

Refactor: The kldata filtering depends on which genes appear as R/EM/T in pathways. Since all backbones for a receiver share the same R/EM/T gene pool, filter kldata once per receiver. The SiK (structural kinase) scores and EI (Exclusiveness Index) are per-edge properties that depend on which kinases connect which pathway nodes --- these can be pre-computed per (EM, Target) and (Receptor, EM) pair and looked up.

Activity kinase evidence (`kl_output`) similarly depends on receiver-side genes + kinase NES values that are pair-independent. Pre-compute per receiver, broadcast.

#### 2g. Replace `Cal_PDS` with vectorized formula

Current: `Cal_PDS()` combines TPDS + kinase evidence (KPDS, AKPDS) into final PDS.

Refactor: The PDS formula is:
```
PDS = TPDS + KPDS.weight * KPDS + AKPDS.weight * AKPDS
```
(simplified --- actual formula involves gating by EI). All components are columns. Vectorized multiply + add.

#### 2h. Output structure

Replace 462 per-pair CSVs with 22 receiver-indexed Parquet files:

```
all_pairs/
  recv_Astrocyte.parquet         (459K rows, sender as column)
  recv_Chandelier.parquet        (3.7M rows)
  ...
  recv_VLMC.parquet              (250K rows)
  cross_pair_summary.parquet     (22×21 = 462 row pair summary)
```

Each Parquet file contains ALL columns from the current `results_full.csv` plus:
- `sender` column (cell type name)
- `backbone_id` (hash of R-EM-T triple, for efficient grouping)
- `n_senders` (how many senders share this backbone --- pre-computed)

Retain backward-compatible CSV export as an option (`--export-csv` flag to split into 462 directories).

### Phase 3: Cross-pair aggregation (follows naturally from Phase 2)

With receiver-centric Parquet tables, the three aggregation angles become DuckDB queries:

#### 3a. Pathway recurrence

```sql
SELECT Receptor, EM, Target,
       COUNT(DISTINCT sender) AS n_senders,
       AVG(PDS) AS mean_pds,
       GROUP_CONCAT(DISTINCT sender) AS sender_list
FROM recv_{receiver}
WHERE PDS > threshold
GROUP BY Receptor, EM, Target
ORDER BY n_senders DESC, mean_pds DESC
```

Backbones appearing in 15+ senders with consistent PDS direction are robust findings. Backbones in 1-2 senders are pair-specific.

#### 3b. Cell-type hub analysis

```sql
-- Sender hub: which senders drive the most signaling to this receiver?
SELECT sender,
       COUNT(*) AS n_pathways,
       AVG(ABS(PDS)) AS mean_abs_pds,
       AVG(kinase_boost) AS mean_kinase_boost
FROM recv_{receiver}
GROUP BY sender
ORDER BY mean_abs_pds DESC
```

Across all receivers, this produces the 22×22 signaling matrix (one pivot).

#### 3c. Gene-level convergence

```sql
-- Which target genes are endpoints of disease-altered pathways from many senders?
SELECT Target,
       COUNT(DISTINCT sender) AS n_senders,
       COUNT(DISTINCT Receptor || '-' || EM) AS n_routes,
       AVG(PDS) AS mean_pds
FROM recv_{receiver}
WHERE ABS(PDS) > threshold
GROUP BY Target
ORDER BY n_senders DESC
```

### Phase 4: Multi-contrast extension (future, depends on snRNA-seq availability)

With the receiver-centric structure, extending to multiple contrasts means:
- Each receiver Parquet file gains a `contrast` dimension
- The backbone table is contrast-independent (it depends on expression, not disease status)
- Only the scoring columns (scFC, phospho, PDS) vary per contrast
- Cross-contrast pathway dynamics become a GROUP BY (backbone, contrast) operation

This is architecturally compatible with the receiver-centric design but depends on snRNA-seq sample availability per contrast arm (see README section 4).

## File changes summary

### Phase 1 (immediate)

| File | Change |
|------|--------|
| `wrappers/run_incytr_all_pairs.R` | Add `enumerate_backbones()`, `attach_all_senders()`; replace per-sender DuckDB query with filter from pre-enumerated table |
| `wrappers/duckdb_enumeration.R` | Add backbone-only SQL builder (optional, can inline in all_pairs) |

### Phase 2 (sprint)

| File | Change |
|------|--------|
| `wrappers/run_incytr_all_pairs.R` | Major rewrite: replace per-pair Incytr object loop with vectorized receiver-batch scoring |
| `wrappers/receiver_scoring.R` | New: vectorized implementations of Expr_bygroup, Cal_SigProb, Cal_scFC, Pathway_evaluation, Cal_PDS |
| `wrappers/postprocess.R` | Adapt for receiver-centric Parquet input |
| `adapters/compute_kinase_support_all_pairs.py` | Read 22 Parquet files instead of 462 CSVs; score with sender column |
| `run_all_pairs.sh` | Update pipeline stages |

### Phase 3 (follows Phase 2)

| File | Change |
|------|--------|
| `wrappers/cross_pair_aggregation.R` (new) | Pathway recurrence, hub analysis, gene convergence |
| `adapters/aggregate_results.py` (new, optional) | Python-side aggregation if preferred over R/DuckDB |

## Verification strategy

1. **Phase 1 correctness:** Run backbone enumeration on 3 diverse receivers (VLMC=small, Astrocyte=medium, Chandelier=large). For each, verify that the set of (sender, Ligand, Receptor, EM, Target) tuples exactly matches the current per-pair enumeration. Zero diff tolerance.

2. **Phase 2 correctness:** For each scoring column (SigProb, sclog2FC, TPDS, PDS, kinase_boost), verify that vectorized scores match per-pair Incytr scores to within floating-point tolerance (|delta| < 1e-10). Run on full 462 pairs.

3. **Performance:** Measure wall-clock time for enumeration (Phase 1) and full pipeline (Phase 2) versus current. Target: 3-5x overall speedup.

## Risks

- **Incytr S4 internals:** If future Incytr package updates change scoring logic, the vectorized replacements must be updated in sync. Mitigate by pinning Incytr version and documenting which functions are replaced.
- **Memory:** A receiver's unified table can be large (4.2M rows for Lamp5 Lhx6). At ~60 columns × 8 bytes, that's ~2 GB per receiver. Parquet compression and DuckDB's out-of-core processing handle this, but R data.frame may not. Use data.table or DuckDB-native processing.
- **Parquet dependency:** Adds `arrow` R package dependency. Already available in the incytr environment.
