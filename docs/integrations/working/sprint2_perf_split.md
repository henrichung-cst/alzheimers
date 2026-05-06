# Sprint 2 D1 working note — `c1b6fb9` split

Working scratchpad for Sprint 2 D1. May be deleted after Sprint 2 sign-off.

`c1b6fb9` "Optimize performance, fix correctness issues, improve code organization"
bundles three categories. Per-fix verdict below; the bundled commit is split into
two ledger rows (`INC-37` perf, `INC-37.b` correctness). No piece is re-routed to
Sprint 3/4: every correctness fix is numerically neutral (or fixes a path that was
unreachable in practice).

## Performance (INC-37, bucket B)

| Hunk | File | Description | Numerical impact |
|---|---|---|---|
| pathway dedup | `R/analysis.R` ~64-72 | `apply(.,1,duplicated)` → vectorized boolean OR over six gene-pair comparisons | Identical pathway set; bitwise-equivalent (any-row-pair-equal ⇔ duplicated row). |
| permutation densify | `R/analysis.R` ~463-487 | Pre-extract `as.matrix(object@data[geneuse, cells])` outside `run_one_permutation`; only label vector changes per permutation | Identical SigProb permutation distribution; same RNG draws, same data. |
| `Cal_EI` row-stats | `R/kinases.R` ~23-58 | Hoist `apply(mat,1,max/min/second)` out of the per-`i` loop | Identical EI values (max/min/second-highest are loop-invariant). |
| `rbind` → `do.call` | `R/evaluation.R` ~189-192 | Replace iterative `rbind` with `do.call(rbind, list)` | Identical resulting data.frame. |
| permutation p-value | `R/analysis.R` ~526 | `m <- t(sapply(obs, rep, nboot)) - perm; mean(m <= 0)` → `mean(perm >= obs)` | Algebraically identical: `obs - perm <= 0` ⇔ `perm >= obs`. |

## Correctness (INC-37.b, bucket A — fix-forward / wash)

| Hunk | File | Description | Native bug? | Verdict |
|---|---|---|---|---|
| null-gene fallback | `R/analysis.R` ~21-29 (`pathway_inference`) | When only one of `gene.use_Sender`/`gene.use_Receiver` is null, native sets the unrelated `gene_use` to all genes but leaves the actual filter args `gene.use_Sender`/`gene.use_Receiver` NULL — downstream `%in% NULL` yields empty SigPath. Fix sets each null arg to all genes. | **Yes** (present in `93b9881:R/analysis.R:382-387`). | Fix-forward, signed off. Numerical effect only on inputs that hit the buggy branch — our wrapper always passes both args, so the live two-condition golden is unaffected (consistent with Sprint 0 SigProb bitwise match). Flag for upstream PR. |
| `setDT` side effect | `R/math.R` ~5 | `setDT(df)` mutates the caller's `df` in place by reference. Switched to `as.data.table(df)` (copy). | **Yes** (native uses `setDT` since `93b9881` introduction of math.R helpers). | Fix-forward, signed off. No numerical change to outputs (the function returns the same value); only caller's `df` no longer secretly becomes a data.table. Flag for upstream PR. |
| `\|` → `\|\|` (7 sites) | `R/Incytr_class.R`, `R/analysis.R` (4×), `R/utils.R` | Defensive lazy-evaluation in `is.null(x) \| foo(x)` patterns. | Pre-existing pattern in native. | Wash (no behavior change for vector-length-1 args; eliminates `\|` warning when foo errors on NULL). No numerical change. |

## Organization (INC-37, absorbed)

- `R/utils.R` `object_update`: deduplicate three identical `pr_FC`/`ps_FC`/`py_FC` blocks via `slot()` loop. Bytes-identical output.
- `R/evaluation.R`: `print()` → `message()` (3 sites). Suppressible via `suppressMessages()`; no algorithmic change.

## Bottom-line

Every numerical claim of `c1b6fb9` ("end-to-end reference identical, all 123 tests pass, 13/13 snapshots match") is reproduced by Sprint 0's golden comparison: `sigprob`, `sc_FC`, `p_value`, `pr_FC`, `ps_FC`, `py_FC` all bitwise-match native `93b9881`. The PDS / SiK_score drifts attributed to other commits (INC-25, INC-28) are not touched by `c1b6fb9`.

`INC-37` and `INC-37.b` both clear Sprint-2 sign-off without re-routing.
