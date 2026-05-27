# Primary-method divergence: derived inputs vs frozen oracle (2026-05-24)

Goal context: reproduce sce4 (full + Top300) **starting from raw omics** via the
active levy_t5 forward-projection decomposition. A divergence between our
**derived** pair-mode inputs and the **frozen** sce4 oracle inputs signals the
primary method is wrong. This doc reproduces the divergence, names the
mechanisms, and poses the one decision that gates the fix.

Benchmark cell: gene `Acvr1`, cluster `Cholinergic-Neurons`, contrast
`ma_2mo AppP vs WTyp` (the Microglia→Cholinergic benchmark pair routes ~83% of
paths through Acvr1).

## Ground truth (three sources, field-name-matched columns)

| Source | AppP | WTyp | ratio | log2FC |
|--------|------|------|-------|--------|
| sce4 published (per CLAUDE.md) | — | — | ~4× | 1.997 |
| frozen file `v2_46clusters/incytr input/pr_yuyu_deconvoluted.csv` | 50.74 | 28.22 | 1.80× | 0.847 |
| our derived (current parquet) | 1654.4 | 1479.2 | 1.12× | 0.163 |

**Cascade of signal loss: sce4 (4×) → frozen file (1.80×) → derived (1.12×).**
The frozen file already loses signal vs sce4 (CLAUDE.md records this as "genuine
pr file divergence" for Acvr1/Grm7 — |pr_log2FC|<1 in the frozen file vs sce4's
1.997/1.877). Our derived loses more.

## Exact decomposition of our 1.12×

`P_c = bulk_protein[gene,animal] × f_percell[gene,animal,cluster]` (inner join).

- bulk `total_proteome_normalized.csv` Acvr1: AppP(animal 9)=36.40, WTyp(animal 15)=36.17 → **bulk ratio 1.006** (flat)
- f_percell Acvr1/Cholinergic: AppP=45.45, WTyp=40.90 → **share ratio 1.11**
- product: 1.006 × 1.11 = **1.12×** ✓ (reproduces parquet 1654.4 / 1479.2)

So the "up in AppP" signal is gone in **both** factors: our normalized bulk is
flat, and our share barely moves.

## Mechanisms (named, reproduced)

1. **No zero-imputation in `snrna_proportions.py`.** `f_percell = (expr_c/Σexpr)×(N_total/N_c)`;
   any cluster with `expr_c = 0` gets `f_percell = 0` (hard zero), and genes with
   `Σexpr = 0` are dropped entirely. The oracle (`protein-ms-by-cell-type.py:80-87,191-196`)
   imputes **every** zero scRNA value to `min/10000` before computing shares, so
   no cluster share is ever zero. This is why our parquet has hard zeros (Acvr1
   = 0 for 4mo/6mo male groups) where the oracle never does.
2. **Different bulk source / normalization (the ~30× scale + ratio shift).** Ours
   uses `total_proteome_normalized.csv` (Acvr1 ≈ 36, ratio 1.006). Oracle uses
   `pr_median.csv` (per-group median of raw MS intensities; Acvr1 ≈ 1.1 implied,
   ratio ≈ 1.6 implied). A uniform per-gene scale cancels in Incytr's
   ratio/prG/sclog2FC logic — but normalization that is **not** a per-gene
   constant (quantile/median normalization) shifts the AppP/WTyp ratio, which
   does not cancel. Our flat 1.006 bulk ratio is the larger half of the lost
   signal.
3. **Aggregation ordering (minor for this benchmark).** Oracle medians the MS
   bulk first, then deconvolves once per group. Ours deconvolves per-animal then
   medians in `export_decomposition_for_pair.py::_pivot`. For ma_2mo there is
   only **one** male animal per (age,geno) group, so ordering has no effect on
   the benchmark; it matters only where a group has >1 animal.

## Two non-defects ruled out

- The pre-compaction "derived Acvr1 = 0/0" was a **stale export**: the CSV
  (07:52) predates the current parquet (07:59). Re-running the export against the
  current parquet gives 1654.4 / 1479.2, not 0/0. The "0/0" was not a live bug.
- The "median across 3 males" concern: there is exactly **one** male per
  (age,geno) group in the parquet (12 male animals total), so the export median
  is a no-op for males. Not the zero-source.

## The decision that gates the fix

Perfect input parity with the frozen file is **impossible for Acvr1**, because
the frozen file itself diverges from sce4 (1.80× vs 4×). So "make derived ==
frozen" tops out at 1.80×, and "make derived == sce4" is unreachable from the
frozen oracle. The real fork:

- **(A) Re-point the active decomposition at the oracle's inputs + imputation.**
  Switch decomposition bulk from `total_proteome_normalized.csv` to the raw
  per-group-median MS intensities the oracle used, and add the `min/10000`
  zero-imputation floor into `snrna_proportions.py`. Closes mechanisms 1+2;
  targets frozen's 1.80× (not sce4's 4×). This reshapes the "Active" levy_t5
  method to mimic the oracle (anti-shim: it *replaces*, no flag).
- **(B) Keep levy_t5 as-is at the input level; judge reconstruction only at the
  Incytr-output level.** Accept that absolute inputs differ; run pair-mode on
  derived inputs and ask whether the Top300 PDS ranking still reproduces sce4
  (rank-based, scale-invariant). If the ranking holds despite input scale
  differences, the divergence is cosmetic for the goal.
- **(C) Add only the zero-imputation floor** (mechanism 1), leave bulk as-is, and
  re-measure. Smallest change; tests whether the hard zeros (not the bulk) are
  the dominant defect for off-benchmark genes.

Prior work (tasks A6–A8) already implemented and probed zero-imputation variants
against the oracle; the SOLVED recipe (A31) is defined against **frozen** inputs.
That history should inform which fork is worth re-running.
