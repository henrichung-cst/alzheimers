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

## Profiling

Profile the local source only after the parity check passes:

```bash
.pixi/envs/default/bin/python bench/kinase_library/compare_public_local.py \
  --donor donor1 --track st --state CD8Naive \
  --profile-source local --profile-limit 60
```

Run artifacts are written under `bench/kinase_library/runs/`, which is ignored.

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

## Rules for Optimization Attempts

- Keep local-source edits inside `bench/kinase_library/local_src` until parity
  and performance evidence justify a production change.
- Treat public/local byte parity as the first gate for an unmodified source copy.
- For any optimization, compare against the public package before adopting it.
- Do not use `gseapy`'s 2D prerank backend as a drop-in replacement unless a
  future investigation explains and eliminates the known semantic differences.
