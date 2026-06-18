# MEA Executor Performance Pass - 2026-06-18

## Scope

This pass benchmarked and profiled the projected T-cell state MEA executor
added during cross-cohort MEA standardization. The benchmark focuses on code
owned by this repo: input loading, state matrix construction, contrast assembly,
output stacking/writing, aggregate writing, and mechanism attribution.

Full permutation-level profiling was not possible in the system Python shell
because `kinase_library` and `py-spy` were not importable there. The benchmark
harness therefore defaults to a deterministic no-op MEA caller, and can be run
with `--caller real` inside the project environment.

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

- The first optimization was limited to deterministic data preparation.
- Further significant gains after this pass likely require reducing the number
  of real GSEA prerank invocations, changing permutation strategy, or modifying
  `kinase_library`/`gseapy` internals. Those should only be attempted with
  strict output fingerprint/parity checks.

## Real `kinase_library` Follow-Up

The repo's `.pixi/envs/default/bin/python` environment was available even
though the `pixi` launcher itself was not on `PATH`. That environment contains
`kinase_library==1.7.0` and the compiled `py-spy` executable.

Single-state real profiling command:

```bash
.pixi/envs/default/bin/python -m alz.core.benchmark_mea_executor \
  --donor donor1 --track st --state CD8Naive \
  --iterations 1 --caller real --profile --profile-limit 40
```

Profile result for one `donor1/ST/CD8Naive` run:

| frame | cumulative time |
| --- | ---: |
| `state_mea.run_projected_state_mea` | 98.2s |
| `alz.bulk_mea.enrich._run_mea` | 96.8s |
| `kinase_library.enrichment.mea.RankedPhosData.mea` | 91.6s |
| `gseapy.gse.prerank_rs` | 82.0s |
| `PhosphoProteomics.percentile` | 7.7s |

Interpretation: the remaining bottleneck is the real GSEA prerank/permutation
work inside `gseapy`, followed by repeated kinase-library percentile scoring.
Repo-owned wrapper and output work is small by comparison.

## Optimized Shared MEA Executor

Two shared changes were added after the real profile:

- `alz.shared.config.MEA_THREADS`, defaulting to `8` and overridable with
  `ALZ_MEA_THREADS`, is passed through to `kinase_library`/`gseapy.prerank`.
- `alz.bulk_mea.enrich._run_mea` now keeps an internal LRU percentile cache
  keyed by the exact motif universe, kinase type, and KL method. It reuses
  `kinase_library` percentiles when raw/stoich, state, or contrast runs share
  the same motif universe. The cache is bounded by
  `ALZ_MEA_PERCENTILE_CACHE_MAX` and can be disabled with
  `ALZ_MEA_PERCENTILE_CACHE_MAX=0`.

Controlled parity check:

```bash
ALZ_MEA_THREADS=4 ALZ_MEA_PERCENTILE_CACHE_MAX=0 \
  .pixi/envs/default/bin/python -m alz.cohorts.tcells.state_mea \
  --donor donor1 --track st --state CD8Naive --runner-scratch-dir old

.pixi/envs/default/bin/python -m alz.cohorts.tcells.state_mea \
  --donor donor1 --track st --state CD8Naive --runner-scratch-dir new
```

All generated CSV outputs and `projected_state_mea_manifest.json` were exact
byte matches between old settings and optimized settings.

Measured runtime for that parity case:

| settings | runtime |
| --- | ---: |
| old: 4 threads, percentile cache disabled | 116.05s |
| optimized: 8 threads, percentile cache enabled | 40.10s |

This supports continuing with the upstream `kinase_library` package for now.
A local fork/vendor copy may still be useful later, but the first high-value
improvements were achievable at the repo integration layer while preserving
outputs exactly.

## Fork/Batched-GSEA Probe

After the shared optimization, a direct probe tested the main fork idea:
use `gseapy`'s DataFrame/2D prerank backend (`prerank2d_rs`) to evaluate
multiple contrasts against one kinase-substrate set in a single call. This is
the kind of API a local `kinase_library` fork could expose.

Important implementation details:

- Raw motif strings cannot be passed directly to 2D `gseapy`; `kinase_library`
  first canonicalizes motifs into substrate strings where lowercase central
  residue encodes the phosphosite.
- `gseapy` has an uppercase-gene heuristic that is unsafe for motif strings if
  raw motifs are used, because case is biologically meaningful.
- Using the processed `kinase_library` substrate strings makes the 2D call
  technically runnable.

Probe case:

| case | current per-contrast path | 2D prerank path |
| --- | ---: | ---: |
| donor1/ST/CD4CTLeomes stoich, 3 contrasts | 66.19s | 26.56s |

The speed signal is real, but this path is **not output compatible**:

| check | result |
| --- | --- |
| row shape | matched, 933 rows each |
| `(contrast, kinase)` keys | matched |
| ES max absolute difference | 0.4387 |
| NES max absolute difference | 2.1106 |
| p-value max absolute difference | 1.0 |
| FDR max absolute difference | 0.9138 |
| `Subs fraction` | differed |
| `Leading substrates` | differed |

Conclusion: do not replace canonical MEA with `gseapy`'s 2D prerank path. A
fork may still be worthwhile, but it would need a lower-level implementation
that preserves the exact single-contrast `prerank_rs` semantics while reducing
setup overhead or scheduling multiple independent single-contrast calls more
efficiently. The simple batched backend is a useful performance lead, not a
valid drop-in replacement.
