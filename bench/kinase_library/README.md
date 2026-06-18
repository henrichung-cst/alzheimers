# Kinase-Library MEA Profiling Bench

This bench area is for testing local `kinase-library` source changes without
changing production imports. Production analysis still imports the installed
package from the active Python environment.

## Bootstrap Local Source

Copy the installed package into an ignored local source tree:

```bash
.pixi/envs/default/bin/python bench/kinase_library/bootstrap_local_source.py
```

This writes:

- `bench/kinase_library/local_src/kinase_library/`
- `bench/kinase_library/local_src/SOURCE_METADATA.json`

Both are intentionally ignored by git.

## Public vs Local Parity

Before editing the local source, compare it against the installed public package:

```bash
.pixi/envs/default/bin/python bench/kinase_library/compare_public_local.py \
  --donor donor1 --track st --state CD8Naive
```

The command runs projected-state MEA twice in isolated subprocesses:

- `source=public`: imports the installed `kinase-library` package.
- `source=local`: prepends `bench/kinase_library/local_src` to `PYTHONPATH`.

It fails nonzero if generated CSV/JSON outputs are not byte-identical.

Use `--workload tcells-timecourse` to run the non-projected T-cell timecourse
MEA path through the shared runner into scratch outputs:

```bash
.pixi/envs/default/bin/python bench/kinase_library/compare_public_local.py \
  --workload tcells-timecourse --donor donor1 --track py
```

## Profiling

Profile the local source only after the parity check passes:

```bash
.pixi/envs/default/bin/python bench/kinase_library/compare_public_local.py \
  --donor donor1 --track st --state CD8Naive \
  --profile-source local --profile-limit 60
```

Run artifacts are written under `bench/kinase_library/runs/`, which is ignored.

## Scheduler Benchmarks

Use `benchmark_scheduler.py` to test exact-backend scheduling with explicit
thread control. The command runs each case in its own Python subprocess and sets
`ALZ_MEA_THREADS` per process.

Example sequential run:

```bash
.pixi/envs/default/bin/python bench/kinase_library/benchmark_scheduler.py \
  --mode sequential --source local --threads-per-process 16 \
  --case workload=projected-state,donor=donor1,track=st,state=CD8Naive \
  --case workload=projected-state,donor=donor1,track=py,state=CD8Naive
```

Example parallel run on a 16-CPU host:

```bash
.pixi/envs/default/bin/python bench/kinase_library/benchmark_scheduler.py \
  --mode parallel --source local --max-workers 2 --threads-per-process 8 \
  --case workload=projected-state,donor=donor1,track=st,state=CD8Naive \
  --case workload=projected-state,donor=donor1,track=py,state=CD8Naive
```

The invariant to preserve is simple: `max-workers * threads-per-process`
should stay near the machine CPU count for CPU-bound S/T MEA workloads.

## Initial Baseline

On 2026-06-18, the unmodified local copy of `kinase-library==1.7.0` matched the
installed package exactly for:

```bash
.pixi/envs/default/bin/python bench/kinase_library/compare_public_local.py \
  --donor donor1 --track st --state CD8Naive
```

The comparison checked 14 generated CSV/JSON files and found no hash
mismatches. Timings for that run were:

| source | elapsed |
| --- | ---: |
| public installed package | 40.65s |
| local source copy | 38.09s |

The first local-source `cProfile` pass used:

```bash
.pixi/envs/default/bin/python bench/kinase_library/run_source_benchmark.py \
  --source local --donor donor1 --track st --state CD8Naive \
  --profile --profile-limit 60 \
  --out-dir bench/kinase_library/runs/local_profile_cd8naive_st
```

Top cumulative frames:

| frame | cumulative time |
| --- | ---: |
| `alz.cohorts.tcells.state_mea.run_projected_state_mea` | 43.94s |
| `alz.bulk_mea.enrich._run_mea` | 42.15s |
| `kinase_library.enrichment.mea.RankedPhosData.mea` | 34.75s |
| `gseapy.prerank` / `gsea.Prerank.run` | 23.45s |
| compiled `gseapy.gse.prerank_rs` | 21.98s |
| `kinase_library.objects.phosphoproteomics.PhosphoProteomics.percentile` | 10.85s |

Practical optimization targets, in order:

1. Preserve single-contrast `prerank_rs` semantics while reducing scheduling
   overhead or parallelizing independent calls more explicitly.
2. Reduce repeated percentile work beyond the repo-level cache, especially
   inside `PhosphoProteomics.percentile`.
3. Avoid re-building identical kinase substrate sets when KL values and
   thresholds are unchanged.

## First Local-Source Optimization Probe

The first behavior-preserving local-source probe is saved as:

```text
bench/kinase_library/patches/0001-cache-default-phosprot-vector-substrate-sets.patch
```

It makes two changes inside the ignored local source copy:

- `PhosphoProteomics.percentile()` reuses `_global_vars.all_scored_phosprot`
  when the default phosphoproteome path is requested, instead of constructing
  another `ScoredPhosphoProteome`.
- `create_kin_sub_sets()` converts the threshold mask to a NumPy boolean array
  once, then slices the substrate index by column, instead of repeatedly slicing
  a dataframe per kinase.

Parity checks against the installed public package passed exactly for 14
generated CSV/JSON files:

| local-source change | public elapsed | local elapsed | parity |
| --- | ---: | ---: | --- |
| global default phosphoproteome reuse | 36.94s | 34.46s | exact |
| plus vectorized substrate-set builder | 38.58s | 35.08s | exact |
| plus vectorized builder, tyrosine `CD8Naive` | 1.68s | 0.92s | exact |
| non-projected T-cell timecourse pY | 5.82s | 3.02s | exact |

The combined local-source profile used:

```bash
.pixi/envs/default/bin/python bench/kinase_library/run_source_benchmark.py \
  --source local --donor donor1 --track st --state CD8Naive \
  --profile --profile-limit 60 \
  --out-dir bench/kinase_library/runs/local_profile_global_phosprot_vector_sets_cd8naive_st
```

Top cumulative frames after both local-source changes:

| frame | cumulative time |
| --- | ---: |
| `alz.cohorts.tcells.state_mea.run_projected_state_mea` | 37.04s |
| `alz.bulk_mea.enrich._run_mea` | 35.58s |
| `kinase_library.enrichment.mea.RankedPhosData.mea` | 29.30s |
| `gseapy.prerank` / `gsea.Prerank.run` | 22.98s |
| compiled `gseapy.gse.prerank_rs` | 21.46s |
| `kinase_library.objects.phosphoproteomics.PhosphoProteomics.percentile` | 6.18s |

This is promising but not yet a production dependency change. The next
promotion decision is whether to maintain a patched local package or upstream
the small changes to `kinase-library`.

## GSEA Backend Investigation

The installed environment uses `gseapy==1.1.13`. Its `prerank_rs` backend is a
compiled Rust extension (`gseapy/gse.cpython-311-x86_64-linux-gnu.so`), not
Python. The matching source distribution shows:

- `prerank_rs()` is implemented in Rust and uses Rayon for parallelism.
- `RAYON_NUM_THREADS` is set from the `threads` argument.
- The prerank implementation builds gene permutations once, then for each gene
  set materializes permutation-specific tag vectors before calculating
  enrichment scores.

Thread scaling on the optimized local `kinase-library` source, projected-state
`donor1/ST/CD8Naive` workload:

| `ALZ_MEA_THREADS` | elapsed |
| ---: | ---: |
| 1 | 323.79s |
| 2 | 182.53s |
| 4 | 120.80s |
| 8 | 41.26s |
| 16 | 36.46s |

The host reports 16 CPUs, so 16 threads is a reasonable single-process ceiling
for this machine. The improvement from 8 to 16 is smaller than the improvement
from 4 to 8, so parallelizing multiple MEA units and increasing per-unit
threads at the same time risks oversubscription.

Potential backend directions:

1. **Keep exact `gseapy.prerank_rs`, improve scheduling.** Run independent MEA
   units in parallel only when `ALZ_MEA_THREADS` is reduced per process so total
   Rayon threads stay near physical CPU count.
2. **Patch the Rust backend.** Preserve `prerank_rs` semantics but avoid
   materializing full tag vectors for every permutation/gene-set pair. Compute
   ES directly from permutation indices and hit positions, or reuse compact
   boolean/bitset representations.
3. **Evaluate newer GSEApy.** GSEApy `v1.2.1` release notes mention speed
   improvements and an fgsea multilevel algorithm in the Rust backend, but
   `kinase-library==1.7.0` pins `gseapy~=1.1.8`, so testing this requires a
   controlled dependency override and full parity checks.
4. **Evaluate approximate/alternative engines.** `blitzgsea` and R `fgsea` are
   promising fast preranked GSEA implementations, but they change p-value
   estimation semantics. They should be treated as exploratory or sensitivity
   engines unless they pass strict row/value parity checks.

A C++ rewrite is possible but not the first choice: the current bottleneck is
already native Rust. The more direct path is either a Rust patch to GSEApy or a
careful GSEApy version experiment.

### GSEApy 1.2.1 Temporary Override Probe

GSEApy 1.2.1 was installed into `/tmp/alz_gseapy_1_2_1` with `pip --target`
and tested via `PYTHONPATH`, without modifying pixi.

Projected-state `donor1/pY/CD8Naive`:

| backend | elapsed | parity |
| --- | ---: | --- |
| installed GSEApy 1.1.13 | 0.83s | baseline |
| temporary GSEApy 1.2.1 | 0.81s | exact hash match |

Projected-state `donor1/ST/CD8Naive` with `ALZ_MEA_THREADS=16`:

| backend | elapsed | parity |
| --- | ---: | --- |
| installed GSEApy 1.1.13 | 31.33s | baseline |
| temporary GSEApy 1.2.1 | 29.79s | not byte-identical |

For the ST probe, `ES`, `NES`, nominal `p-value`, `Subs fraction`, and
`Leading substrates` matched exactly across 311 kinases. The changed values
were FDR only:

- 161/311 kinase rows had different FDR values.
- Maximum absolute FDR difference was `0.0009904730882456293`.
- The mechanism call category counts were unchanged in this one probe.

Conclusion: GSEApy 1.2.1 is promising but cannot be adopted as a silent
drop-in upgrade. It changes canonical FDR outputs for this workload, likely due
to changed zero-FDR handling or FDR calculation details in the newer backend.
Any upgrade should be treated as an explicit analysis-output migration, not a
pure performance patch.

## Rules for Optimization Attempts

- Keep local-source edits inside `bench/kinase_library/local_src` until parity
  and performance evidence justify a production change.
- Treat public/local byte parity as the first gate for an unmodified source copy.
- For any optimization, compare against the public package before adopting it.
- Do not use `gseapy`'s 2D prerank backend as a drop-in replacement unless a
  future investigation explains and eliminates the known semantic differences.
