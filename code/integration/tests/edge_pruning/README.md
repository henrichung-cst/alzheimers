# Pathway Enumeration Optimization

## Goal

Remove the 50% expression detection threshold from `run_incytr.R` by making pathway enumeration efficient enough to handle all genes. Currently, lowering the threshold causes a combinatorial explosion in `pathway_inference()` that exceeds available memory (~30 GB).

## Background

Incytr enumerates 4-gene signaling chains (Ligand->Receptor->EM->Target) by joining three database layers:

| Layer | Edges   | Description            |
|-------|---------|------------------------|
| L1    | 5,635   | Ligand -> Receptor     |
| L2    | 2.8M    | Receptor -> EM         |
| L3    | 3.7M    | EM -> Target           |

(After filtering to genes in the expression matrix.)

`pathway_inference()` filters each layer to edges where both genes pass a detection threshold, then joins via cartesian products on shared genes (Receptor joins L1-L2, EM joins L2-L3). Without a threshold, the cartesian joins on multi-million-edge tables exhaust memory.

## Approach: edge-level SigProb pre-pruning

SigProb is a product of three pairwise Hill function terms, one per edge:

```
SigProb = hill(L * R) * hill(R * EM) * hill(EM * T)
```

Since Hill values are in [0, 1], if any edge-level Hill value falls below `cutoff_SigProb`, the full pathway product is guaranteed below cutoff. We pre-compute Hill values per edge using per-condition group mean expression and remove dead edges before the cartesian join.

## Results on real data (Microglia-PVM -> L5 IT)

### Edge reduction

| Threshold | L1 edges | L1 pruned | L2 edges  | L2 pruned | L3 edges  | L3 pruned |
|-----------|----------|-----------|-----------|-----------|-----------|-----------|
| 50%       | 3        | 0%        | 65,972    | 0%        | 68,310    | 0%        |
| 20%       | 81       | 0%        | 397,436   | 8%        | 429,153   | 7%        |
| 10%       | 187      | 3%        | 801,960   | 35%       | 887,713   | 34%       |
| 5%        | 333      | 20%       | 1,220,350 | 53%       | 1,378,036 | 53%       |
| 0%        | 5,635    | 93%       | 2,795,348 | 79%       | 3,721,401 | 82%       |

### Pathway counts

| Threshold | Sender genes | Receiver genes | Pathways | Time   | Memory  |
|-----------|-------------|----------------|----------|--------|---------|
| 50%       | 210         | 2,114          | 3,544    | 1.0s   | ~8 MB   |
| 20%       | 1,127       | 4,994          | 164,880  | 3.8s   | ~0 MB*  |
| 10%       | 2,357       | 7,464          | 169,462  | 9.9s   | ~198 MB |
| 5%        | 4,064       | 10,027         | 169,462  | 14.3s  | ~217 MB |
| 0%        | 30,567      | 30,567         | 169,462  | 17.0s  | ~279 MB |

*Memory delta can be noisy due to GC timing.

### Key finding

**The pathway count converges at ~169,462.** From 10% threshold through 0% (all genes), edge pruning produces the same pathway set. This is the complete set of biologically plausible 4-gene chains for this sender-receiver pair given a SigProb cutoff of 0.01 --- all pathways with sufficient co-expression at adjacent nodes to transmit a signal.

The 50% threshold discards 165,918 pathways (97.9%) that have legitimate SigProb. The "9.5M pathways at 5%" estimate was the unpruned enumeration count; with pruning, only 169K survive.

### Correctness

At 50% threshold, the pruned pipeline produces identical surviving pathway sets and SigProb values compared to the standard pipeline (max numerical difference: 0.00e+00). The pruning is conservative: an edge is kept if its Hill value exceeds the cutoff in either condition.

## Files

- `README.md` --- this file
- `profile_enumeration.R` --- baseline measurements at different thresholds (standard pipeline)
- `test_edge_pruning.R` --- edge pruning prototype on real data
