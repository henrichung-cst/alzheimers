# MEA Executor Performance Pass - 2026-06-18

## Scope

This pass benchmarked and profiled the projected T-cell state MEA executor
added during cross-cohort MEA standardization. The benchmark focuses on code
owned by this repo: input loading, state matrix construction, contrast assembly,
output stacking/writing, aggregate writing, and mechanism attribution.

Full permutation-level profiling was not possible in the active shell because
`kinase_library` and `py-spy` were not importable. The added benchmark harness
therefore defaults to a deterministic no-op MEA caller and can be re-run with
`--caller real` inside the project environment.

## Reproducible Command

```bash
python -m alz.core.benchmark_mea_executor --donor both --track both --iterations 5
python -m alz.core.benchmark_mea_executor --donor donor1 --track st --iterations 1 --profile
```

Use the real enrichment caller when dependencies are available:

```bash
python -m alz.core.benchmark_mea_executor --donor donor1 --track st --iterations 3 --caller real --profile
```

## Finding

The pre-change no-op benchmark showed that donor1/ST projected-state executor
overhead was dominated by repeated per-state matrix preparation. The same
donor/track bundle has 62,807 phosphosite rows, 8,125 protein rows, 13 states,
and 48 state-day columns. The original loop rebuilt parsed state columns,
numeric projected values, protein-by-gene tables, and projected-gene protein
reindexing once per state.

## Change

`alz/cohorts/tcells/state_mea.py` now has a reusable
`ProjectedStateMatrixCache`. `run_projected_state_mea()` builds the cache lazily
for a donor/track only when at least one state actually runs, then reuses it for
each state. This preserves the public `build_state_matrices()` API and does not
change MEA caller inputs, contrast math, output schemas, or canonical analysis
paths.

## Measured Executor-Only Result

No-op caller median timings in this shell:

| case | before | after | interpretation |
| --- | ---: | ---: | --- |
| donor1/ST | 0.7603s | 0.5579s | ~27% faster executor overhead |
| donor1/pY | 0.1857s | 0.1604s | ~14% faster executor overhead |
| donor2/pY | 0.0565s | 0.0584s | effectively unchanged; all states skip |

The post-change profile for donor1/ST shows CSV loading as the largest remaining
fixed executor cost, followed by cached matrix builds. The real end-to-end gain
will be smaller if `kinase_library` permutation runtime dominates wall time,
but the executor path is materially cheaper for added MEA tests and repeated
scratch runs.

## Caveats

- This is not a full `kinase_library` benchmark because the current shell lacks
  the dependency.
- The optimization is deliberately limited to deterministic data preparation.
- Further significant gains likely require profiling `kinase_library` itself or
  reducing repeated real MEA invocations, which should only be attempted with
  strict output fingerprint/parity checks.
