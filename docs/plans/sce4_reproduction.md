# sce4 reproduction plan

This is the active control document for the sce4 pair-mode reproduction. We are taking over the
analysis under a trust-but-verify model: every claim must be tied to a local artifact, a command, and
an acceptance criterion. Prior exploratory hypotheses are not treated as settled unless they survive
the checks below.

## Primary Conclusion as of 2026-06-01

The current blocker is score parity, not broad pathway enumeration. After removing the three model
transgene genes (`App`, `Psen1`, `Mapt`) as an explicit sensitivity analysis, the best source-`ps`
diagnostic shares 3,466,037 / 3,466,192 sce4 non-transgene gated rows (99.996%), but still misses 155
rows, adds 162 rows, and shares only 399,209 / 431,458 sce4 Top300 rows (92.53%). The remaining
Top300 and threshold-edge differences are driven by `PDS` drift.

The `PDS` drift is not a binary nuisance; it is large enough to affect rank and gate membership.
Across source-`ps` non-transgene shared rows, `abs(delta PDS)` exceeds 0.05 in 346,766 rows. The
largest weighted contributors are phospho subscores: `PhPDS_ps` is the most frequent dominant
contributor, while `PhPDS_py` has the largest tail drift. `PPDS` is secondary. `TPDS` is negligible
for non-transgene rows after the q0 path-level fix.

This means the reproduction should now be framed as: we have nearly recovered the non-transgene
candidate/gated universe, but our score reconstruction is not exact enough to reproduce sce4's
threshold and Top300 behavior.

## Operating Decision and Downstream Policy

Exact sce4 parity is not achieved. Working forward, the defensible claim is bounded: with frozen
sce4 effective gene-use and the source-`ps` diagnostic, removing `App`, `Psen1`, and `Mapt` paths
recovers 3,466,037 / 3,466,192 non-transgene gated rows (99.996%) and 399,209 / 431,458 Top300 rows
(92.53%). This is good enough to proceed operationally, but it is not an exact reproduction and must
not be described as one.

All future production Incytr runs in this repo must use the same input and filter contract unless a
run is explicitly labeled forensic/diagnostic:

- Use `data/derived/incytr_inputs/` as the canonical matched AD input bundle. Do not mix
  transcriptomics, markers, protein, phospho, or kinase files across provenance folders for a
  production claim.
- Preserve the sce4-style score floor for production AD and downstream Incytr outputs:
  `(SigProb_condition1 > 0.1 OR SigProb_condition2 > 0.1) AND abs(PDS) >= 0.2`, uncapped.
- Do not apply the per sender/receiver Top300 up/down cap to canonical Song/AD or T-cell viewer
  outputs. Top300 remains available only as an explicit sce4 table-compatibility diagnostic because
  it is rank-sensitive to the unresolved `PDS` drift.
- Do not add a p-value, FDR, or q-value gate to sce4-style AD outputs. The frozen sce4 reference did
  not use that arm.
- Keep the AD transgene exclusion (`App`, `Psen1`, `Mapt` in Ligand/Receptor/EM/Target) as an
  explicit sensitivity flag, not as a hidden default. The interpretation is experimental-design
  specific: remove paths directly touching the model transgenes when the analysis goal is downstream
  pathway behavior rather than direct transgene effects.
- Any use of `data/incytr_frozen/v2_46clusters/incytr input/`,
  `data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/`,
  `data/derived/incytr_inputs_source_ps_diag/`, or `data/derived/_sce4_input_scratch/` must be
  labeled diagnostic and must write to diagnostic output directories. These folders remain useful for
  forensic tests, but they are not the canonical production base.
- Before relying on a run, execute the input-provenance audit:

```bash
pixi run python alz/incytr_pair/audit_incytr_input_provenance.py
```

The policy consequence is direct: downstream Incytr applications should inherit the canonical input
root, sce4-style floor gate, uncapped row universe, no-pvalue gate, and explicit
transgene-sensitivity flag rather than silently copying whichever forensic command happened to
improve a residual.

Operational artifact status after applying this policy on 2026-06-01:

- Song/AD central wide outputs: 9 contrasts, 4,480,480 floor-gated uncapped rows.
- Song/AD unified-viewer Incytr shards: 653 sender/receiver shards, 4,480,480 rows.
- Song/AD unified-viewer decomp OLS shards: 389 kinase shards, 23,362,857 substrate-site OLS rows,
  generated from `outputs/reports/decomposition/levy_t5/per_animal/site_level_ols.parquet`.
- T-cell donor1 wide outputs: 3 contrasts, 737,081 floor-gated uncapped rows.
- T-cell donor2 wide outputs: 4 contrasts, 1,832,503 floor-gated uncapped rows.
- T-cell viewer Incytr shards: donor1 165 shards / 737,081 rows; donor2 100 shards /
  1,832,503 rows.

## 1. Objective

Reproduce sce4 AD pair-mode results for all nine contrasts:

- `ma_2mo_AppP` vs `ma_2mo_WTyp`
- `ma_2mo_ApTt` vs `ma_2mo_WTyp`
- `ma_2mo_Ttau` vs `ma_2mo_WTyp`
- `ma_4mo_AppP` vs `ma_4mo_WTyp`
- `ma_4mo_ApTt` vs `ma_4mo_WTyp`
- `ma_4mo_Ttau` vs `ma_4mo_WTyp`
- `ma_6mo_AppP` vs `ma_6mo_WTyp`
- `ma_6mo_ApTt` vs `ma_6mo_WTyp`
- `ma_6mo_Ttau` vs `ma_6mo_WTyp`

The reference is sce4's `10302025` DEG/PRG output under:

```text
data/incytr_frozen/outputs/Analysis_new cluster labels_cutoff_0.1/
```

The primary reproducibility target is the gated pre-cap pathway tuple set:

```text
(SigProb_condition1 > 0.1 OR SigProb_condition2 > 0.1) AND abs(PDS) >= 0.2
```

Top300/capped table agreement is reported separately because it is PDS-ranked and currently sensitive
to phospho and transgene score residuals.

## 2. Canonical Inputs

For the independent rerun, use one matched input bundle. Do not mix omics files from different
provenance directories.

Canonical bundle:

```text
data/derived/incytr_inputs/incytr_obj.rds
data/derived/incytr_inputs/allmarkers.csv
data/derived/incytr_inputs/pr_yuyu_deconvoluted.csv
data/derived/incytr_inputs/ps_yuyu_deconvoluted.csv
data/derived/incytr_inputs/py_yuyu_deconvoluted.csv
data/derived/incytr_inputs/kldata.csv
```

Frozen/source deconvolution files exist under:

```text
data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/
```

Those files are diagnostic only unless we also identify the matching transcriptomics object, markers,
and kinase inputs from the same provenance. A previous source-omics-only diagnostic run was ambiguous:
its one-contrast parquet was byte-identical to the derived-input parquet, but a later controlled
one-pair rerun with `INPUTS_DIR_OVERRIDE` explicitly pointed at the source-omics scratch bundle showed
that source and derived outputs do differ on phospho columns as expected. Do not use the old
source-omics parquet as evidence that the source bundle cannot improve anything.

Phospho file identity check:

| channel | file | md5 |
|---|---|---|
| `ps` | `data/derived/incytr_inputs/ps_yuyu_deconvoluted.csv` | `d20fb75065f3a8cd6a4b9cc0909b19e0` |
| `ps` | `data/incytr_frozen/v2_46clusters/incytr input/ps_yuyu_deconvoluted.csv` | `3445726dd5be9e7b175bffa26d0c39e9` |
| `ps` | `data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/ps_yuyu_deconvoluted.csv` | `3445726dd5be9e7b175bffa26d0c39e9` |
| `ps` | `data/derived/incytr_inputs_source_ps_diag/ps_yuyu_deconvoluted.csv` | `3445726dd5be9e7b175bffa26d0c39e9` |
| `py` | `data/derived/incytr_inputs/py_yuyu_deconvoluted.csv` | `da6c85906491f5a900d6eb7273458032` |
| `py` | `data/incytr_frozen/v2_46clusters/incytr input/py_yuyu_deconvoluted.csv` | `6522f82ab643336b25da418e835f3155` |
| `py` | `data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/py_yuyu_deconvoluted.csv` | `ca17469ba3fde425d542b7e81f3683a5` |
| `py` | `data/derived/incytr_inputs_source_ps_diag/py_yuyu_deconvoluted.csv` | `da6c85906491f5a900d6eb7273458032` |

Conclusion from the hash check: the three phospho sources are not all the same. V2 and Drive-adjacent
source `ps` are byte-identical, and the source-`ps` diagnostic intentionally uses that file. The
canonical derived `ps` differs. For `py`, the canonical derived, v2, and Drive-adjacent source files
all differ from each other; the source-`ps` diagnostic did not swap `py` and still uses canonical
derived `py`.

Therefore the observed phospho score drift does not, by itself, prove an `../incytr` engine bug. The
remaining possibilities are: historical input provenance, preprocessing/normalization/collapse
behavior, driver-side handling of phospho inputs, or package scoring behavior. The evidence does prove
that our current score reconstruction is off relative to sce4 frozen outputs.

Follow-up engine trace on the 155 source-`ps` non-transgene missing rows shows that the driver/package
path from current inputs to current output is internally consistent: recomputing package-default
phospho handling from `data/derived/incytr_inputs_source_ps_diag/` reproduces our `PhPDS_ps` and
`PhPDS_py` to floating-point precision. The mismatch is between that current package-default
reconstruction and sce4's frozen scores. Therefore the leading issue is not an accidental mismatch
between our audit math and the current engine output; it is historical score reconstruction:
historical input/preprocessing details or historical package behavior differ from the current
reconstruction.

Verification command:

```bash
md5sum data/derived/incytr_inputs/ps_yuyu_deconvoluted.csv \
  data/derived/incytr_inputs/py_yuyu_deconvoluted.csv \
  'data/incytr_frozen/v2_46clusters/incytr input/ps_yuyu_deconvoluted.csv' \
  'data/incytr_frozen/v2_46clusters/incytr input/py_yuyu_deconvoluted.csv' \
  data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/ps_yuyu_deconvoluted.csv \
  data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/py_yuyu_deconvoluted.csv \
  data/derived/incytr_inputs_source_ps_diag/ps_yuyu_deconvoluted.csv \
  data/derived/incytr_inputs_source_ps_diag/py_yuyu_deconvoluted.csv
```

## 3. Frozen sce4 Gene-Use

The current AD reproduction uses sce4's effective per-pair gene-use extracted from sce4's own pre-cap
Pairwise RDS files:

```text
data/incytr_frozen/sce4_geneuse/*.csv
```

Extractor:

```text
alz/incytr_pair/extract_sce4_geneuse.R
```

Status: all nine contrasts have extracted gene-use artifacts. This gives us a controlled way to test
the scorer and gate over sce4's effective candidate set. It does not independently rediscover sce4's
candidate narrowing rule, so we must describe it as "independent scoring over frozen effective
gene-use," not as a fully independent candidate reconstruction.

## 4. Current One-Contrast Status

Latest verified contrast:

```text
ma_2mo_AppP vs ma_2mo_WTyp
```

Output inspected:

```text
outputs/reports/incytr_pair_mode/_sce4_one_contrast_q0/ma_2mo_AppP_ma_2mo_WTyp_incytr_output.parquet
```

Drive-adjacent source-bundle diagnostic output:

```text
outputs/reports/incytr_pair_mode/_sce4_one_contrast_source_bundle/ma_2mo_AppP_ma_2mo_WTyp_incytr_output.parquet
```

Verifier summary after the q0 path-level TPDS fix:

| metric | count |
|---|---:|
| ours gated | 341,758 |
| sce4 gated | 299,857 |
| shared | 298,078 |
| missing | 1,779 |
| missing non-transgene | 6 |
| extra | 43,680 |
| extra non-transgene | 0 |
| Top300 shared | 48,666 / 65,750 = 74.0% |

Verifier summary for the source-bundle diagnostic:

| metric | count |
|---|---:|
| ours gated | 341,758 |
| sce4 gated | 299,857 |
| shared | 298,078 |
| missing | 1,779 |
| missing non-transgene | 6 |
| extra | 43,680 |
| extra non-transgene | 0 |
| Top300 shared | 48,687 / 65,750 = 74.0% |

Interpretation:

- Non-transgene gated tuple reproduction is nearly exact: 235,239 reproduced of 235,245 reference
  non-transgene rows, with 6 non-transgene missing and 0 non-transgene extras.
- The 43,680 overproduced gated rows are all transgene-associated under the definition below.
- Top300 is not reproduced exactly. Current capped overlap is 74.0% for the tested contrast.
- The Drive-adjacent source-bundle diagnostic does not change gated membership relative to the derived
  q0 run. It improves Top300 by 21 rows and modestly improves some `PhPDS_ps`/PDS residual tails, but
  it is not a reproduction win.

## 4b. Current Full Nine-Contrast Status

Controlled full rerun completed on `2026-05-31` with frozen sce4 per-pair gene-use and the canonical
derived input bundle.

Launcher:

```text
alz/incytr_pair/run_sce4_full_unfiltered.sh
```

Run artifacts:

```text
outputs/reports/incytr_pair_mode/_sce4_full_q0/
outputs/reports/incytr_pair_mode/sce4_full_unfiltered_run.log
outputs/reports/incytr_pair_mode/sce4_full_verify_full_q0.csv
```

The run produced all nine unfiltered parquets. The full verifier failed for every contrast:

| contrast | ours gated | sce4 gated | shared | missing | missing non-transgene | extra | extra non-transgene | Top300 shared |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ma_2mo_AppP` | 341,758 | 299,857 | 298,078 | 1,779 | 6 | 43,680 | 0 | 48,666 / 65,750 |
| `ma_2mo_ApTt` | 1,078,354 | 1,021,830 | 992,063 | 29,767 | 37 | 86,291 | 31 | 104,287 / 154,518 |
| `ma_2mo_Ttau` | 518,419 | 518,289 | 518,240 | 49 | 16 | 179 | 12 | not reported |
| `ma_4mo_AppP` | 677,817 | 636,639 | 627,588 | 9,051 | 18 | 50,229 | 23 | 49,976 / 65,692 |
| `ma_4mo_ApTt` | 922,365 | 841,759 | 822,430 | 19,329 | 9 | 99,935 | 29 | 65,482 / 92,146 |
| `ma_4mo_Ttau` | 385,824 | 385,775 | 385,450 | 325 | 298 | 374 | 206 | 45,915 / 65,638 |
| `ma_6mo_AppP` | 165,233 | 161,332 | 151,094 | 10,238 | 10 | 14,139 | 8 | 37,379 / 48,830 |
| `ma_6mo_ApTt` | 191,581 | 181,581 | 177,525 | 4,056 | 15 | 14,056 | 16 | 39,191 / 51,444 |
| `ma_6mo_Ttau` | 199,129 | 199,109 | 199,037 | 72 | 14 | 92 | 3 | 27,842 / 34,919 |

Totals across the nine contrasts:

| metric | count |
|---|---:|
| missing | 74,666 |
| missing non-transgene | 423 |
| extra | 308,975 |
| extra non-transgene | 328 |

Interpretation:

- The full current-q0 run is complete and reproducible from local artifacts.
- Exact sce4 parity is not achieved.
- `ma_2mo_AppP` remains the cleanest non-transgene case: 6 non-transgene misses and 0 non-transgene
  extras.
- The other contrasts introduce non-transgene extras as well as misses, so the one-contrast
  "nearly exact non-transgene tuple reproduction" claim does not generalize to all nine contrasts.
- Ttau contrasts have much smaller total set differences than AppP/ApTt contrasts, but `ma_4mo_Ttau`
  has a comparatively large non-transgene mismatch class.

## 4c. Current Source-PS Diagnostic Status

A full nine-contrast diagnostic rerun completed on `2026-06-01` using canonical derived
transcriptomics, markers, protein, `py`, and kinase inputs, but replacing canonical `ps` with the
v2/source `ps_yuyu_deconvoluted.csv`.

This is a deliberately mixed input bundle and is not a valid final reproduction claim under the
matched-bundle rule. It tests whether the phospho-serine provenance result from `ma_4mo_Ttau`
generalizes across all nine contrasts.

Run artifacts:

```text
data/derived/incytr_inputs_source_ps_diag/
outputs/reports/incytr_pair_mode/_sce4_full_source_ps_diag/
outputs/reports/incytr_pair_mode/sce4_full_source_ps_diag_run.log
outputs/reports/incytr_pair_mode/sce4_full_verify_full_source_ps_diag.csv
outputs/reports/incytr_pair_mode/forensics_source_ps_full_diag/sce4_shared_score_residual_summary.csv
```

Verification against frozen sce4 `10302025` Pairwise outputs:

| contrast | ours gated | sce4 gated | shared | missing | missing non-transgene | extra | extra non-transgene | Top300 shared |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ma_2mo_AppP` | 341,758 | 299,857 | 298,078 | 1,779 | 6 | 43,680 | 0 | 48,684 / 65,750 |
| `ma_2mo_ApTt` | 1,078,353 | 1,021,830 | 992,062 | 29,768 | 37 | 86,291 | 31 | 104,382 / 154,518 |
| `ma_2mo_Ttau` | 518,419 | 518,289 | 518,240 | 49 | 16 | 179 | 12 | not reported |
| `ma_4mo_AppP` | 677,817 | 636,639 | 627,588 | 9,051 | 17 | 50,229 | 23 | 50,101 / 65,692 |
| `ma_4mo_ApTt` | 922,359 | 841,759 | 822,424 | 19,335 | 8 | 99,935 | 29 | 66,770 / 92,146 |
| `ma_4mo_Ttau` | 385,918 | 385,775 | 385,710 | 65 | 32 | 208 | 40 | 49,064 / 65,638 |
| `ma_6mo_AppP` | 165,233 | 161,332 | 151,094 | 10,238 | 10 | 14,139 | 8 | 37,378 / 48,830 |
| `ma_6mo_ApTt` | 191,583 | 181,581 | 177,527 | 4,054 | 15 | 14,056 | 16 | 39,194 / 51,444 |
| `ma_6mo_Ttau` | 199,129 | 199,109 | 199,037 | 72 | 14 | 92 | 3 | 27,845 / 34,919 |

Comparison to canonical q0:

- The full source-`ps` diagnostic does not achieve exact parity; all nine contrasts still fail.
- The diagnostic materially improves `ma_4mo_Ttau`: missing non-transgene rows drop from 298 to 32,
  and extra non-transgene rows drop from 206 to 40.
- The other two Ttau contrasts were already close and remain close: `ma_2mo_Ttau` is unchanged in
  non-transgene mismatch counts, while `ma_6mo_Ttau` remains at 14 missing and 3 extra non-transgene
  rows.
- AppP/ApTt contrasts still have large transgene-dominated extra sets and score residuals. The
  source-`ps` swap improves some Top300 overlaps but does not fix the broad AppP/ApTt parity gap.
- The rerun completed without OOM using `NPAIR_WORKERS=1`, `NPERM_WORKERS=1`, `CHUNK_PARALLEL=1`, and
  `N_CHUNK_MULT=48`. Driver high-water RSS stayed around 5.2 GB.

## 5. Transgene Definition and Current Counts

For this analysis, a transgene-associated pathway is any row where any of the four path nodes is one
of:

```text
App, Psen1, Mapt
```

For the tested `ma_2mo_AppP` contrast, the 43,680 extra rows break down as:

| transgene set in path | extra rows |
|---|---:|
| App only | 42,852 |
| Psen1 only | 777 |
| App + Psen1 | 51 |
| Mapt | 0 |

By position:

| position | gene | extra rows |
|---|---|---:|
| Ligand | Psen1 | 450 |
| Receptor | App | 41,761 |
| EM | App | 851 |
| EM | Psen1 | 218 |
| Target | App | 291 |
| Target | Psen1 | 160 |

The extra rows include 2,922 non-transgene partner genes, but every extra row contains `App` and/or
`Psen1`. Therefore the overproduction is not ordinary-gene overproduction in this tested contrast.

Important nuance: sce4 does contain transgene rows. The mismatch is not that sce4 excludes transgenes
globally. The current hypothesis is that frozen per-pair gene-use plus current scoring still permits
additional App/Psen1-associated paths to cross the gate because transgene values differ from sce4's
effective values.

Operational sensitivity rule added on `2026-06-01`:

- `alz/incytr_pair/filter_significant_paths.py --exclude-transgenes` now removes any path containing
  `App`, `Psen1`, or `Mapt` in `Ligand`, `Receptor`, `EM`, or `Target` before the per-pair Top300
  cap.
- This is not the default sce4 reproduction rule. It is an explicit AD-model sensitivity analysis
  under the interpretation that the authors may have wanted pathway effects not directly driven by
  the model transgenes.
- The paired audit script is `alz/incytr_pair/audit_transgene_excluded_reproduction.R`, which applies
  the same transgene removal to both our reruns and frozen sce4 before comparing gated and Top300
  membership.

## 6. Phospho Residuals

The phospho residuals are significant for ranking and can move threshold-edge rows across
`abs(PDS) >= 0.2`.

Across the 298,078 shared gated rows in `ma_2mo_AppP`:

| component | median abs delta | p95 abs delta | p99 abs delta | max abs delta | rows > 0.05 |
|---|---:|---:|---:|---:|---:|
| `PhPDS_ps` | 0.0018619 | 0.0970834 | 0.2491548 | 0.6988417 | 31,167 |
| `PhPDS_py` | 0 | 0.25 | 0.2787629 | 0.7493184 | 31,326 |
| `PDS` | 0.0032646 | 0.1262656 | 1.187335 | 2.149155 | 47,927 |

The six non-transgene misses are all Astrocytes-receiver EGFR/FGFR3/MLC1 paths. They are present in
our raw output but fall below the `abs(PDS) >= 0.2` gate because our `PhPDS_ps` and `PhPDS_py` values
are lower than sce4's for those paths.

The `Pathway_evaluation` formula itself is not the first suspect. Sce4's frozen `PhPDS_ps` and
`PhPDS_py` values are exactly reproduced from sce4's frozen per-node phospho `aFC` columns by:

```text
rowMeans(logi(Ligand/Receptor/EM/Target phospho aFC, k = 2), NA -> 0)
```

The current evidence points upstream: the phospho `aFC` values produced from our available matched
`ps/py` inputs differ from sce4's frozen per-node `ps/py_aFC` values. This may be data provenance or
preprocessing provenance. It is not yet proven to be an `../incytr` package bug.

Additional forensic check:

```text
alz/incytr_pair/forensic_sce4_afc.R
outputs/reports/incytr_pair_mode/forensics/
```

For `ma_2mo_AppP`, recomputed per-node `aFC` values from both the derived input bundle and the
Drive-adjacent source-omics bundle were compared against sce4 frozen per-node values. On the six
non-transgene missing rows, the source bundle does not resolve the EGFR/MCL1 phospho gap:

- Astrocytes `Egfr` `ps_aFC`: sce4 `0.691719`, derived `0.377014`, source `0.377016`.
- Astrocytes `Mlc1` `py_aFC`: sce4 `0.282934`, derived `0.230854`, source `0.230886`.

The source bundle can materially change some other phospho values, so a full source-bundle rerun
was run as a separate diagnostic path. It did not change gated membership.

Shared-row component residuals for source-bundle vs derived q0:

| run | component | median abs delta | p95 abs delta | p99 abs delta | max abs delta | rows > 0.05 |
|---|---|---:|---:|---:|---:|---:|
| derived | `PhPDS_ps` | 0.0018619 | 0.0970834 | 0.2491548 | 0.6988417 | 31,167 |
| source bundle | `PhPDS_ps` | 0.0018620 | 0.0961183 | 0.2154972 | 0.3872742 | 29,311 |
| derived | `PDS` | 0.0032646 | 0.1262656 | 1.1873345 | 2.1491551 | 47,927 |
| source bundle | `PDS` | 0.0031665 | 0.1253081 | 1.1871965 | 2.1491126 | 46,437 |

The source-bundle run used Drive `renamed_sobj.rds` and Drive `pr/ps/py`, but still used the current
Song kinase fallback because no matching source-side `kldata.csv` was identified. The 46-cluster
Drive object expanded the pair loop to 2,116 sender/receiver combinations, with 410 non-empty shards.

## 7. Trust-But-Verify Plan

### Step A: Freeze the Input Contract - Complete

Use only `data/derived/incytr_inputs/` for the independent rerun. Record file hashes before the full
run. This was done before the full q0 rerun.

Recorded hashes:

| file | md5 |
|---|---|
| `data/derived/incytr_inputs/incytr_obj.rds` | `718e6fcf19b48411c1cb717c815f67ca` |
| `data/derived/incytr_inputs/allmarkers.csv` | `21ac99fe892ce4b9f8f201acda03c640` |
| `data/derived/incytr_inputs/pr_yuyu_deconvoluted.csv` | `1441b27e30916149501668b431fb8c07` |
| `data/derived/incytr_inputs/ps_yuyu_deconvoluted.csv` | `d20fb75065f3a8cd6a4b9cc0909b19e0` |
| `data/derived/incytr_inputs/py_yuyu_deconvoluted.csv` | `da6c85906491f5a900d6eb7273458032` |
| `data/derived/incytr_inputs/kldata.csv` | `dbd4f1f25f6d77d2d4571af91cf8933f` |

Acceptance:

- `pr`, `ps`, `py`, transcriptomics object, markers, and kinase table all come from the same derived
  bundle.
- Any diagnostic run using alternative omics must be clearly labeled and must not overwrite the
  baseline.

### Step B: Verify Frozen Gene-Use Coverage - Complete

Confirm that `data/incytr_frozen/sce4_geneuse/` has one CSV for each of the nine contrasts and that
every pair maps cleanly to the 31-cluster spine.

Acceptance:

- 9/9 contrast gene-use files present.
- No unmapped cluster labels.
- No duplicate/colliding normalized cluster labels.

### Step C: Run Full Independent Scoring - Complete

Ran all nine contrasts with:

```text
SCE4_GENEUSE_DIR=data/incytr_frozen/sce4_geneuse
OUTPUT_DIR_OVERRIDE=outputs/reports/incytr_pair_mode/_sce4_full_q0
FULL_NBOOT=0 / NBOOT=0
NPERM_WORKERS=1
NPAIR_WORKERS=1
CHUNK_PARALLEL=1
N_CHUNK_MULT=48
```

Actual memory wrapper:

```bash
systemd-run --user --scope --slice=alz-incytr.slice \
  -p MemoryMax=24G -p MemorySwapMax=0 \
  --unit=alz-sce4-full-unfiltered \
  --description="sce4 full unfiltered reproduction" \
  bash alz/incytr_pair/run_sce4_full_unfiltered.sh
```

Acceptance:

- All nine unfiltered parquet outputs were produced in `_sce4_full_q0/`.
- The run completed inside the 24 GB cap; observed driver HWM stayed far below the cap in the live
  logs.
- The run did not use `run_pair_mode.sh` because that wrapper filters `wide/` in place after
  verification; the controlled launcher preserves pre-cap evidence for `verify_sce4_full.R`.

### Step D: Verify Gated Tuple Identity by Class - Complete

Ran:

```bash
/home/hchung/.pixi/bin/pixi run Rscript alz/incytr_pair/verify_sce4_full.R \
  --wide-dir outputs/reports/incytr_pair_mode/_sce4_full_q0 \
  --report-csv outputs/reports/incytr_pair_mode/sce4_full_verify_full_q0.csv
```

Acceptance tiers:

- Green: 0 non-transgene extras and 0 non-transgene missing for every contrast.
- Amber: non-transgene differences exist only at threshold edges and are explained by component-level
  residuals.
- Red: non-transgene extras indicate candidate/enumeration drift and must stop the full reproduction
  claim.

Transgene-associated differences are reported separately and must be quantified by gene and node
position.

Result: Red for the full reproduction claim. All nine contrasts failed exact tuple parity. Several
contrasts have non-transgene extras, so the next step is row-level classification rather than another
full rerun.

### Step E: Classify Non-Transgene Mismatches - Complete First Pass

For every contrast, emit row-level tables for non-transgene missing and extra tuples. Each table must
include sender, receiver, ligand, receptor, EM, target, sce4 and rerun PDS components where available,
and a reason code.

Reason-code targets:

- `threshold_residual`: tuple exists in both raw outputs but crosses `abs(PDS) >= 0.2` in only one.
- `phospho_afc_residual`: threshold movement is driven by `PhPDS_ps` or `PhPDS_py`.
- `tpds_or_sigprob_residual`: threshold movement is driven by transcript/log2FC/SigProb score drift.
- `candidate_or_enumeration`: tuple is not present in our raw output or not present in sce4's frozen
  reference tuple universe before gating.
- `transgene_partner_effect`: tuple is classified non-transgene by path nodes but is affected by a
  sender/receiver or upstream partner pattern linked to transgene-heavy contrasts.

Priority order:

| priority | contrast | why |
|---:|---|---|
| 1 | `ma_4mo_Ttau` | largest non-transgene mismatch class: 298 missing, 206 extra |
| 2 | `ma_2mo_ApTt` | large full-set mismatch and non-transgene extras/misses |
| 3 | `ma_4mo_ApTt` | large full-set mismatch with non-transgene extras |
| 4 | `ma_4mo_AppP` | AppP contrast with non-transgene extras, unlike `ma_2mo_AppP` |
| 5 | remaining contrasts | confirm whether the same mechanisms explain smaller mismatch classes |

Acceptance:

- Every non-transgene missing/extra row is classified.
- Classification separates true candidate/enumeration drift from score-threshold drift.
- If non-transgene extras are raw candidate/enumeration drift, stop claiming near-independent
  reproduction beyond the one clean contrast until that rule is identified.

Full audit artifact:

```bash
/home/hchung/.pixi/bin/pixi run Rscript alz/incytr_pair/audit_sce4_mismatches.R \
  --contrast <contrast>
```

Outputs:

```text
outputs/reports/incytr_pair_mode/forensics/ma_4mo_Ttau_missing_nontransgene_audit.csv
outputs/reports/incytr_pair_mode/forensics/ma_4mo_Ttau_extra_nontransgene_audit.csv
outputs/reports/incytr_pair_mode/forensics/ma_4mo_Ttau_mismatch_audit_summary.csv
outputs/reports/incytr_pair_mode/forensics/sce4_all_contrasts_mismatch_audit_summary.csv
```

Result across all nine contrasts:

| contrast | missing NT | missing present in raw | phospho threshold | non-phospho PDS threshold | extra NT | extra absent from sce4 Pairwise | extra path seen elsewhere |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ma_2mo_ApTt` | 37 | 37 | 36 | 1 | 31 | 31 | 27 |
| `ma_2mo_AppP` | 6 | 6 | 6 | 0 | 0 | 0 | 0 |
| `ma_2mo_Ttau` | 16 | 16 | 13 | 3 | 12 | 12 | 4 |
| `ma_4mo_ApTt` | 9 | 9 | 9 | 0 | 29 | 29 | 12 |
| `ma_4mo_AppP` | 18 | 18 | 12 | 6 | 23 | 23 | 11 |
| `ma_4mo_Ttau` | 298 | 298 | 298 | 0 | 206 | 206 | 129 |
| `ma_6mo_ApTt` | 15 | 15 | 15 | 0 | 16 | 16 | 9 |
| `ma_6mo_AppP` | 10 | 10 | 9 | 1 | 8 | 8 | 2 |
| `ma_6mo_Ttau` | 14 | 14 | 13 | 1 | 3 | 3 | 3 |
| **Total** | **423** | **423** | **411** | **12** | **328** | **328** | **197** |

All 423 missing non-transgene rows are present in our raw parquet and none fail the rerun SigProb
gate. They are PDS threshold misses: 411 are dominated by `PhPDS_ps`/`PhPDS_py` residuals, and 12 are
dominated by non-phospho PDS component residuals. For `ma_4mo_Ttau`, the largest class, the missing
row `abs(delta_PDS)` distribution has median `0.1103892`, p95 `0.1434598`, and max `0.1554634`; its
largest component residual is `PhPDS_ps` for 270 rows and `PhPDS_py` for 28 rows.

All 328 extra non-transgene rows are present in our gated output but absent from sce4's frozen
Pairwise reference set. With the currently available reference artifact, record this as
reference-set/candidate drift, not as a proven score-threshold residual. For 197 of those extra rows,
the same ligand/receptor/EM/target path is seen elsewhere in sce4's Pairwise reference, so at least
part of the extra class may be sender/receiver-specific candidate narrowing rather than wholly novel
path enumeration. If we identify a larger sce4 pre-gate universe, rerun the audit against that universe
to distinguish "absent from sce4 entirely" from "present in sce4 but below gate."

### Step F: Quantify Score Residuals

For every contrast, compute component residual distributions on shared rows:

- `TPDS`
- `PPDS`
- `PhPDS_ps`
- `PhPDS_py`
- kinase score columns
- `multimodel_score`
- `PDS`

Acceptance:

- Transcript and protein residuals should remain small except documented App/Psen1/Mapt cases.
- Phospho residuals must be summarized as effect sizes and threshold impacts, not hand-waved.
- Any residual that changes non-transgene gated membership must be listed row-by-row.

### Step G: Investigate Phospho Provenance

Investigate in this order:

1. Recompute ps/py fold changes from the canonical matched bundle and compare per-gene/per-cluster
   values to sce4 frozen `*_ps_aFC` and `*_py_aFC`.
2. Test whether differences are caused by normalization, duplicate-site collapse, gene-vs-site
   aggregation, zero handling, or `q`/correction settings.
3. Only after data/preprocessing explanations are exhausted, test whether `../incytr` has diverged
   from the sce4-era `Integr_multiomics` behavior.

Acceptance:

- We can classify the phospho residual as data provenance, preprocessing provenance, or package
  implementation drift.
- We do not change the canonical input bundle to chase parity unless we have a complete matched
  replacement bundle.

### Step H: Report Top300 Separately

Compute Top300/capped overlap for every contrast, but do not use it as the sole reproduction gate
until phospho and transgene residuals are explained.

Acceptance:

- Top300 overlap, missing, and extra rows reported per contrast.
- Rows entering or leaving Top300 because of phospho/transgene PDS changes are separated from true
  tuple-set differences.

### Step G2: PDS Provenance First Pass - Complete

Script:

```bash
/home/hchung/.pixi/bin/pixi run Rscript alz/incytr_pair/audit_pds_provenance.R
/home/hchung/.pixi/bin/pixi run Rscript alz/incytr_pair/audit_pds_provenance.R \
  --source-dir "data/incytr_frozen/v2_46clusters/incytr input"
```

Outputs:

```text
outputs/reports/incytr_pair_mode/forensics/sce4_pds_provenance_summary.csv
outputs/reports/incytr_pair_mode/forensics/sce4_pds_provenance_node_detail.csv
outputs/reports/incytr_pair_mode/forensics/sce4_pds_provenance_summary_v2_46input.csv
outputs/reports/incytr_pair_mode/forensics/sce4_pds_provenance_node_detail_v2_46input.csv
outputs/reports/incytr_pair_mode/forensics/sce4_pds_provenance_gate_projection.csv
```

The audit compared sce4 frozen node-level `ps_aFC`/`py_aFC` values on the missing non-transgene rows
against recomputed candidates varying:

- input bundle: canonical derived vs source/v2;
- duplicate phosphosite handling: first row, mean raw, median raw, mean site-aFC, max-abs site-aFC;
- limma normalization on/off;
- aFC shrinkage `q = 0.75` vs `q = 0`;
- correction `0.001` vs `0.0001`.

Current driver behavior is already gene-level mean collapse before limma normalization. This is not an
unfixed package first-duplicate bug in the reproduction path; `slice_omics()` mean-collapses duplicate
gene rows before handing `ps_1`, `ps_2`, `py_1`, and `py_2` to `Incytr::Cal_pairwise_grid`.

Best missing-row node-level matches:

| suffix | best local source | duplicate mode | normalize | q | median abs delta | p95 abs delta | max abs delta |
|---|---|---|---|---:|---:|---:|---:|
| `ps` | v2/source phospho | mean raw | TRUE | 0.75 | 0.0212275 | 0.2894061 | 0.7218666 |
| `ps` | canonical derived | mean raw | TRUE | 0.75 | 0.0586667 | 1.2720492 | 2.0335021 |
| `py` | canonical derived | mean raw | TRUE | 0.75 | 0.0586326 | 0.4266547 | 0.7371584 |
| `py` | v2/source phospho | mean raw | TRUE | 0.75 | 0.0586331 | 0.4266604 | 0.7371637 |

Interpretation:

- The sce4 missing-row `ps` residual is primarily input provenance: source/v2 `ps_yuyu_deconvoluted.csv`
  is materially closer to frozen sce4 node `ps_aFC` than canonical derived `ps_yuyu_deconvoluted.csv`.
- `py` does not materially distinguish canonical derived from v2/source for this missing-row subset.
- `q = 0.75` with limma normalization remains the best-supported scorer setting; `q = 0` improves some
  threshold projections but worsens node-level residual tails and is not a defensible formula patch.
- Gate projection over the 423 missing rows is diagnostic only, not a verifier replacement, but it
  confirms the same direction: source/v2 mean-raw `q = 0.75` projects 324/423 rows back over the
  `abs(PDS) >= 0.2` gate, while canonical derived mean-raw `q = 0.75` projects only 56/423. Source/v2
  `q = 0` projects 357/423 but has worse node-level residual tails.
- A source-side complete bundle exists locally at `data/incytr_frozen/v2_46clusters/incytr input/`.
  It includes `incytr_obj.rds`, `allmarkers.csv`, `pr/ps/py`, and a `kldata.csv` symlink. However, it is
  a 46-cluster source bundle and is not interchangeable with the canonical 31-cluster derived bundle
  without a separate full rerun/verification claim.
- Do not silently mix canonical transcriptomics/proteomics with source phospho to chase PDS parity.
  If we choose the v2/source bundle, rerun and verify it as a complete bundle.

### Step G3: Complete v2/source Bundle One-Contrast Test - Complete

Rationale:

The source/v2 phospho file gave the best node-level `ps` match in Step G2, but the project constraint is
not to mix omics files from different locations. The lowest-cost valid test was therefore to run one
contrast with the complete adjacent v2/source bundle:

```text
data/incytr_frozen/v2_46clusters/incytr input/
```

This bundle contains matched `incytr_obj.rds`, `allmarkers.csv`, `HEG_df.csv`, `input_gene_list.csv`,
`pr_yuyu_deconvoluted.csv`, `ps_yuyu_deconvoluted.csv`, `py_yuyu_deconvoluted.csv`, and `kldata.csv`.
The v2 object has 46 clusters, while the canonical derived object has 31 clusters. The run still used
sce4 frozen per-pair gene-use, so only sce4-relevant pair shards were emitted.

Command:

```bash
systemd-run --user --scope --slice=alz-incytr.slice \
  -p MemoryMax=24G -p MemorySwapMax=0 \
  --unit=alz-sce4-v2-4mo-ttau \
  --description="sce4 v2 46 bundle ma_4mo_Ttau" \
  env INPUTS_DIR_OVERRIDE="data/incytr_frozen/v2_46clusters/incytr input" \
      OUTPUT_DIR_OVERRIDE="outputs/reports/incytr_pair_mode/_sce4_one_contrast_v2_46bundle" \
      SCE4_GENEUSE_DIR="data/incytr_frozen/sce4_geneuse" \
      NBOOT=0 NPAIR_WORKERS=1 NPERM_WORKERS=1 CHUNK_PARALLEL=1 N_CHUNK_MULT=48 \
      /home/hchung/.pixi/bin/pixi run Rscript alz/incytr_pair/incytr_commandline.R \
      ma_4mo_Ttau ma_4mo_WTyp
```

Outputs and supporting residual aggregate:

```text
outputs/reports/incytr_pair_mode/_sce4_one_contrast_v2_46bundle/ma_4mo_Ttau_ma_4mo_WTyp_incytr_output.parquet
outputs/reports/incytr_pair_mode/sce4_full_verify_ma_4mo_Ttau_v2_46bundle.csv
outputs/reports/incytr_pair_mode/forensics_v2_46bundle/
```

Run behavior:

- Completed under the 24 GB cgroup cap.
- High-water RSS reported by the driver was about 8.4 GB.
- Raw output rows were `510,631`, the same raw count as the canonical derived `ma_4mo_Ttau` rerun.
- The 46-cluster object produced 2,116 possible sender/receiver pairs, but only 260 non-empty shards
  because frozen sce4 gene-use exists only for sce4-relevant pairs.

Verification against frozen sce4 `ma_4mo_Ttau`:

| input bundle | ours gated | sce4 gated | shared | missing | missing non-transgene | extra | extra non-transgene | Top300 shared/total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical derived | 385,824 | 385,775 | 385,450 | 325 | 298 | 374 | 206 | 45,915/65,638 |
| complete v2/source | 385,889 | 385,775 | 385,442 | 333 | 300 | 447 | 279 | 46,136/65,638 |

The complete v2/source bundle is therefore not a tuple-parity improvement for this contrast. It
slightly improves Top300 overlap, but it worsens exact gated tuple parity.

Mismatch audit for complete v2/source `ma_4mo_Ttau`:

| measure | count |
|---|---:|
| missing non-transgene | 300 |
| missing present in our raw output | 300 |
| missing absent from our raw output | 0 |
| missing below SigProb gate | 0 |
| missing dominated by phospho PDS residual | 20 |
| missing dominated by non-phospho PDS residual | 280 |
| extra non-transgene | 279 |
| extra absent from frozen sce4 Pairwise reference | 279 |
| extra path seen elsewhere in frozen sce4 Pairwise reference | 194 |

Interpretation:

- The v2/source bundle validates the phospho provenance diagnosis: the canonical `ma_4mo_Ttau`
  missing non-transgene class was 298/298 phospho-dominated, while the complete v2/source run leaves
  only 20 phospho-dominated missing non-transgene rows.
- The same run exposes a larger non-phospho PDS problem, mostly protein/`PPDS`: 280 missing
  non-transgene rows are now dominated by non-phospho PDS residuals.
- The v2/source bundle also increases non-transgene extras from 206 to 279.
- Do not run the full nine-contrast v2/source bundle yet as the next parity attempt. It is useful as a
  provenance diagnostic, but not currently a better reproduction claim than the canonical derived
  bundle.

### Step G4: PPDS/Protein Provenance Audit - Complete

Script:

```bash
/home/hchung/.pixi/bin/pixi run Rscript alz/incytr_pair/audit_ppds_provenance.R \
  --missing-glob outputs/reports/incytr_pair_mode/forensics/ma_4mo_Ttau_missing_nontransgene_audit.csv \
  --out-dir outputs/reports/incytr_pair_mode/forensics

/home/hchung/.pixi/bin/pixi run Rscript alz/incytr_pair/audit_ppds_provenance.R \
  --missing-glob outputs/reports/incytr_pair_mode/forensics_v2_46bundle/ma_4mo_Ttau_missing_nontransgene_audit.csv \
  --out-dir outputs/reports/incytr_pair_mode/forensics_v2_46bundle
```

Outputs:

```text
outputs/reports/incytr_pair_mode/forensics/ma_4mo_Ttau_ppds_provenance_node_summary.csv
outputs/reports/incytr_pair_mode/forensics/ma_4mo_Ttau_ppds_provenance_row_summary.csv
outputs/reports/incytr_pair_mode/forensics_v2_46bundle/ma_4mo_Ttau_ppds_provenance_node_summary.csv
outputs/reports/incytr_pair_mode/forensics_v2_46bundle/ma_4mo_Ttau_ppds_provenance_row_summary.csv
```

The audit compared frozen sce4 per-node `*_pr_aFC` and row-level `PPDS` against protein fold changes
recomputed from both complete input bundles. It varied duplicate handling, floor-to-1 protein handling,
limma normalization, `q`, and correction. The driver's actual protein preprocessing is
`mean_raw + floor_lt1 + limma normalize + q=0.75 + correction=0.001`.

Best row-level `PPDS` agreement by input:

| missing-row set | input tested | best preprocessing | median abs delta vs sce4 `PPDS` | p95 abs delta | max abs delta | rows > 0.05 | rows > 0.10 |
|---|---|---|---:|---:|---:|---:|---:|
| canonical derived missing rows | canonical derived | mean raw, floor <1, limma, q=0.75, corr=0.001 | 0.0008083 | 0.0061927 | 0.0132304 | 0 | 0 |
| canonical derived missing rows | v2/source | median raw, floor <1, limma, q=0.75, corr=0.001 | 0.1885854 | 0.2210320 | 0.2515480 | 230 | 222 |
| v2/source missing rows | canonical derived | mean raw, floor <1, limma, q=0.75, corr=0.001 | 0.0008392 | 0.0062228 | 0.0245363 | 0 | 0 |
| v2/source missing rows | v2/source | mean raw, floor <1, limma, q=0.75, corr=0.001 | 0.1883693 | 0.2387156 | 0.2683968 | 267 | 264 |

Best node-level `pr_aFC` agreement by input:

| missing-row set | input tested | median node abs delta | p95 node abs delta | max node abs delta | nodes > 0.05 | nodes > 0.10 |
|---|---|---:|---:|---:|---:|---:|
| canonical derived missing rows | canonical derived | 0.0038208 | 0.0258433 | 0.1244881 | 17 | 1 |
| canonical derived missing rows | v2/source | 1.0609302 | 1.3289917 | 2.1068417 | 798 | 793 |
| v2/source missing rows | canonical derived | 0.0040711 | 0.0306502 | 0.3676944 | 15 | 6 |
| v2/source missing rows | v2/source | 1.0609302 | 1.3747982 | 2.7149538 | 826 | 824 |

Interpretation:

- The PPDS residual in the complete v2/source run is a protein input provenance problem, not an
  `../incytr` scorer problem. The v2/source protein input exactly reproduces the v2 rerun's `PPDS`
  values, but those values are far from sce4 frozen `PPDS`.
- Sce4 frozen `PPDS` matches the canonical derived protein input closely for both the canonical
  missing-row set and the v2 missing-row set.
- This explains why the complete v2/source bundle fixed most phospho misses but worsened exact tuple
  parity: v2/source `ps` is closer to sce4 phospho, while canonical derived `pr` is closer to sce4
  protein.
- A mixed bundle of v2/source phospho plus canonical derived protein would be a useful diagnostic but
  should not become a reproduction claim under the current "single matched data bundle" constraint.

### Step G5: Source-PS Mixed Diagnostic Rerun - Complete

Rationale:

Step G2 showed that v2/source `ps_yuyu_deconvoluted.csv` is closer to sce4 frozen `ps_aFC` than the
canonical derived `ps` file, while Step G4 showed that canonical derived protein is closer to sce4
`PPDS` than the complete v2/source protein file. The cleanest diagnostic was therefore a deliberately
mixed scratch bundle: canonical transcriptomics, markers, protein, `py`, and kinase, but v2/source
`ps`.

Scratch input bundle:

```text
data/derived/incytr_inputs_source_ps_diag/
```

Symlink targets:

| file | source |
|---|---|
| `incytr_obj.rds` | canonical derived |
| `allmarkers.csv` | canonical derived |
| `pr_yuyu_deconvoluted.csv` | canonical derived |
| `py_yuyu_deconvoluted.csv` | canonical derived |
| `kldata.csv` | canonical derived/Song symlink |
| `ps_yuyu_deconvoluted.csv` | `data/incytr_frozen/v2_46clusters/incytr input/ps_yuyu_deconvoluted.csv` |

Command:

```bash
env INPUTS_DIR_OVERRIDE="data/derived/incytr_inputs_source_ps_diag" \
    OUTPUT_DIR_OVERRIDE="outputs/reports/incytr_pair_mode/_sce4_one_contrast_source_ps_diag" \
    SCE4_GENEUSE_DIR="data/incytr_frozen/sce4_geneuse" \
    NBOOT=0 NPAIR_WORKERS=1 NPERM_WORKERS=1 CHUNK_PARALLEL=1 N_CHUNK_MULT=48 \
    /home/hchung/.pixi/bin/pixi run Rscript alz/incytr_pair/incytr_commandline.R \
    ma_4mo_Ttau ma_4mo_WTyp
```

Output:

```text
outputs/reports/incytr_pair_mode/_sce4_one_contrast_source_ps_diag/ma_4mo_Ttau_ma_4mo_WTyp_incytr_output.parquet
outputs/reports/incytr_pair_mode/sce4_full_verify_ma_4mo_Ttau_source_ps_diag.csv
outputs/reports/incytr_pair_mode/forensics_source_ps_diag/
```

Run behavior:

- Completed with 260 non-empty shards.
- Raw output rows were `510,631`, same as the canonical and complete v2/source `ma_4mo_Ttau` reruns.
- High-water RSS reported by the driver was about 5.1 GB.

Verification against frozen sce4 `ma_4mo_Ttau`:

| input bundle | ours gated | sce4 gated | shared | missing | missing non-transgene | extra | extra non-transgene | Top300 shared/total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical derived | 385,824 | 385,775 | 385,450 | 325 | 298 | 374 | 206 | 45,915/65,638 |
| complete v2/source | 385,889 | 385,775 | 385,442 | 333 | 300 | 447 | 279 | 46,136/65,638 |
| canonical + source `ps` | 385,918 | 385,775 | 385,710 | 65 | 32 | 208 | 40 | 49,064/65,638 |

Mismatch audit for canonical + source `ps`:

| measure | count |
|---|---:|
| missing non-transgene | 32 |
| missing present in our raw output | 32 |
| missing absent from our raw output | 0 |
| missing below SigProb gate | 0 |
| missing dominated by phospho PDS residual | 31 |
| missing dominated by non-phospho PDS residual | 1 |
| extra non-transgene | 40 |
| extra absent from frozen sce4 Pairwise reference | 40 |
| extra path seen elsewhere in frozen sce4 Pairwise reference | 30 |

Interpretation:

- The source-`ps` swap is the best `ma_4mo_Ttau` tuple-parity diagnostic so far: non-transgene misses
  drop from 298 to 32, and non-transgene extras drop from 206 to 40.
- This strongly supports phospho `ps` input provenance as the dominant cause of the canonical
  `ma_4mo_Ttau` threshold residuals.
- The remaining 32 non-transgene misses are still mostly phospho-threshold residuals, so source `ps`
  is closer but not exact.
- This is not a valid final reproduction bundle under the current matched-bundle rule because it mixes
  canonical transcript/protein/py with v2/source `ps`. It is a diagnostic showing which input channel
  controls most of the parity gap.

### Step G6: Full Source-PS Mixed Diagnostic Rerun - Complete

Rationale:

The one-contrast source-`ps` diagnostic in Step G5 was strong enough to justify a full nine-contrast
diagnostic rerun. The purpose was to test whether v2/source `ps` fixes phospho-serine-driven threshold
residuals globally, while keeping canonical derived protein because Step G4 showed canonical protein
best matches sce4 `PPDS`.

Command:

```bash
env INPUTS_DIR_OVERRIDE="data/derived/incytr_inputs_source_ps_diag" \
    OUTPUT_DIR_OVERRIDE="outputs/reports/incytr_pair_mode/_sce4_full_source_ps_diag" \
    LOG_OVERRIDE="outputs/reports/incytr_pair_mode/sce4_full_source_ps_diag_run.log" \
    SCE4_GENEUSE_DIR="data/incytr_frozen/sce4_geneuse" \
    FULL_NBOOT=0 NPAIR_WORKERS=1 NPERM_WORKERS=1 CHUNK_PARALLEL=1 N_CHUNK_MULT=48 \
    bash alz/incytr_pair/run_sce4_full_unfiltered.sh
```

Verifier:

```bash
/home/hchung/.pixi/bin/pixi run Rscript alz/incytr_pair/verify_sce4_full.R \
  --wide-dir outputs/reports/incytr_pair_mode/_sce4_full_source_ps_diag \
  --report-csv outputs/reports/incytr_pair_mode/sce4_full_verify_full_source_ps_diag.csv
```

Shared-score residual audit:

```bash
/home/hchung/.pixi/bin/pixi run Rscript alz/incytr_pair/audit_shared_score_residuals.R \
  --wide-dir outputs/reports/incytr_pair_mode/_sce4_full_source_ps_diag \
  --out-csv outputs/reports/incytr_pair_mode/forensics_source_ps_full_diag/sce4_shared_score_residual_summary.csv
```

All nine parquets were produced. Raw row counts:

| contrast | raw rows |
|---|---:|
| `ma_2mo_AppP` | 464,266 |
| `ma_2mo_ApTt` | 1,576,688 |
| `ma_2mo_Ttau` | 774,407 |
| `ma_4mo_AppP` | 1,198,989 |
| `ma_4mo_ApTt` | 1,385,679 |
| `ma_4mo_Ttau` | 510,631 |
| `ma_6mo_AppP` | 357,680 |
| `ma_6mo_ApTt` | 366,756 |
| `ma_6mo_Ttau` | 334,043 |

The full source-`ps` diagnostic improves phospho-serine residuals but does not eliminate score
residuals:

| component | canonical rows > 1e-4 | source-`ps` rows > 1e-4 | delta | canonical max abs | source-`ps` max abs |
|---|---:|---:|---:|---:|---:|
| `PDS` | 3,865,356 | 3,856,531 | -8,825 | 2.2669839 | 2.1825392 |
| `multimodel_score` | 3,863,546 | 3,854,692 | -8,854 | 2.2669839 | 2.1491236 |
| `PPDS` | 3,465,832 | 3,466,083 | +251 | 0.2509957 | 0.2509957 |
| `PhPDS_ps` | 3,204,520 | 3,112,426 | -92,094 | 0.7100294 | 0.5321491 |
| `PhPDS_py` | 1,485,001 | 1,485,080 | +79 | 0.7493184 | 0.7493184 |
| `TPDS` | 112,874 | 112,861 | -13 | 1.9999919 | 1.9999919 |

Ttau-specific residual impact:

| contrast | component | canonical rows > 1e-4 | source-`ps` rows > 1e-4 | delta | canonical p95 abs | source-`ps` p95 abs |
|---|---|---:|---:|---:|---:|---:|
| `ma_2mo_Ttau` | `PhPDS_ps` | 443,386 | 443,285 | -101 | 0.0619212 | 0.0519856 |
| `ma_4mo_Ttau` | `PhPDS_ps` | 304,092 | 253,977 | -50,115 | 0.1930650 | 0.0891182 |
| `ma_6mo_Ttau` | `PhPDS_ps` | 176,532 | 176,398 | -134 | 0.1141669 | 0.1134168 |

Interpretation:

- The source-`ps` file is a real phospho-serine provenance improvement, especially for
  `ma_4mo_Ttau`.
- It is not an exact historical phospho reconstruction. `PhPDS_ps` still differs by more than `1e-4`
  on 3,112,426 shared rows, and `PhPDS_py` is essentially unchanged because this diagnostic did not
  swap `py`.
- Overall `PDS` residuals remain large because AppP/ApTt score differences, `PPDS`, `PhPDS_py`, and
  threshold-edge transgene rows are still unresolved.
- This diagnostic confirms that a single available local matched bundle cannot currently reproduce
  sce4 exactly: canonical derived protein is closest to sce4 protein, while v2/source `ps` is closer
  to sce4 phospho-serine for the largest Ttau residual class.

### Step I: Non-Transgene Extra-Row Universe Audit - Complete

Script:

```bash
/home/hchung/.pixi/bin/pixi run Rscript alz/incytr_pair/audit_extra_universe.R
```

Latest rerun after extracting the complete local Drive-download Allpathway set used:

```bash
Rscript alz/incytr_pair/audit_extra_universe.R
```

Outputs:

```text
outputs/reports/incytr_pair_mode/forensics/sce4_extra_universe_detail.csv
outputs/reports/incytr_pair_mode/forensics/sce4_extra_universe_summary.csv
```

This audit tested the canonical full nine-contrast extra non-transgene rows against available local
sce4 universes:

- frozen `Pairwise_pathway_table_10302025.rds`;
- frozen `Top300_table_10302025.csv`;
- frozen `Allpathway_table_10302025.csv`.

Important artifact finding:

- Live Google Drive search through the connector did not expose the sce4 folder/files: no shared
  drives were visible and searches for `sce4`, `DEG_PRG`, `Incytr`, `Allpathway`, `.csv`, `.zip`,
  `Analysis_new`, and `10302025` returned no relevant artifacts.
- The local Drive-download zips under `data/incytr_frozen/outputs/` did contain the missing
  contrast-level `Allpathway_table_10302025.csv` files. We extracted the missing Allpathway CSVs from:
  `Analysis_new cluster labels_cutoff_0.1-20260519T145546Z-3-001.zip`,
  `Analysis_new cluster labels_cutoff_0.1-20260519T145546Z-3-002.zip`, and
  `Analysis_new cluster labels_cutoff_0.1-20260519T145546Z-3-003.zip`.
- After extraction, all 9/9 contrast-level `sce4_DEG_PRG_Allpathway_table_10302025.csv` files are
  present under `data/incytr_frozen/outputs/Analysis_new cluster labels_cutoff_0.1/`.
- The `10302025` Allpathway files are not wider pre-gate universes for this audit. The full rerun
  against all nine Allpathway files still finds 0/328 extra non-transgene rows in `Allpathway`.
- The separate score-bearing candidate
  `data/incytr_frozen/outputs/sce4_DEG_PRG_Allpathway_table_09062025.csv` has 1,849,877 data rows and
  includes `TPDS`, `PPDS`, `PhPDS_ps`, `PhPDS_py`, `multimodel_score`, and `PDS`, but it is not the
  missing pre-gate universe for the 328 extra non-transgene rows. Key matching found only 1/328 rows
  in that file, and the matched row's scores differ materially from the rerun (`PDS` delta `0.3108218`,
  `PhPDS_ps` delta `0.0650284`, `PhPDS_py` delta `0`).

Extra non-transgene universe results:

| contrast | extra NT | in Pairwise RDS | in Top300 | in Allpathway 10302025 | all nodes role-present in same pair | path seen elsewhere in Pairwise |
|---|---:|---:|---:|---:|---:|---:|
| `ma_2mo_ApTt` | 31 | 0 | 0 | 0 | 31 | 27 |
| `ma_2mo_AppP` | 0 | 0 | 0 | n/a | 0 | 0 |
| `ma_2mo_Ttau` | 12 | 0 | 0 | 0 | 11 | 4 |
| `ma_4mo_ApTt` | 29 | 0 | 0 | 0 | 29 | 12 |
| `ma_4mo_AppP` | 23 | 0 | 0 | 0 | 23 | 11 |
| `ma_4mo_Ttau` | 206 | 0 | 0 | 0 | 206 | 129 |
| `ma_6mo_ApTt` | 16 | 0 | 0 | 0 | 16 | 9 |
| `ma_6mo_AppP` | 8 | 0 | 0 | 0 | 8 | 2 |
| `ma_6mo_Ttau` | 3 | 0 | 0 | 0 | 3 | 3 |
| **Total** | **328** | **0** | **0** | **0** | **327** | **197** |

Interpretation:

- The extra rows are not explained by an obvious role-specific gene-use bug. In 327/328 extra
  non-transgene rows, all four nodes are present in sce4's same-pair role-specific observed sets
  (`Ligand` as ligand, `Receptor` as receptor, `EM` as EM, and `Target` as target).
- The one role-set exception is a `ma_2mo_Ttau` target-role miss. This is negligible relative to the
  328-row extra class.
- The extra rows are also not present in Top300 or in the complete extracted `10302025` Allpathway
  set.
- Their scores are threshold-edge: across all 328 rows, our `abs(PDS)` ranges from `0.2001827` to
  `0.3325088`, median `0.2229492`; 254/328 are `<= 0.25` and 308/328 are `<= 0.30`.
- With the currently available artifacts, the most defensible interpretation is: these are local
  threshold-edge gated rows that sce4 did not emit in its `10302025` reference outputs. We still
  cannot prove their exact sce4 pre-gate scores because neither the complete `10302025` Allpathway set
  nor the `09062025` score-bearing candidate provides the missing broader pre-gate universe.

### Step J: Transgene-Excluded Sensitivity Rule - Complete

Script:

```bash
/home/hchung/.pixi/bin/pixi run Rscript alz/incytr_pair/audit_transgene_excluded_reproduction.R
```

Output:

```text
outputs/reports/incytr_pair_mode/forensics/transgene_removed_top300_summary.csv
outputs/reports/incytr_pair_mode/forensics_source_ps_full_diag/sce4_all_contrasts_mismatch_audit_summary.csv
```

This audit removes paths containing `App`, `Psen1`, or `Mapt` from both the rerun and frozen sce4
references before gated-set comparison and before the per-pair Top300 reranking. This tests the
biological/design interpretation separately from the literal sce4 table reproduction.

Aggregate results after transgene removal:

| run | gated ours | gated sce4 | gated shared | gated missing | gated extra | Top300 ours | Top300 sce4 | Top300 shared | Top300 missing | Top300 extra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | 3,466,097 | 3,466,192 | 3,465,769 | 423 | 328 | 431,431 | 431,458 | 395,430 | 36,028 | 36,001 |
| source-`ps` diagnostic | 3,466,199 | 3,466,192 | 3,466,037 | 155 | 162 | 431,434 | 431,458 | 399,209 | 32,249 | 32,225 |

Source-`ps` per-contrast results after transgene removal:

| contrast | gated ours | gated sce4 | gated shared | gated missing | gated extra | Top300 shared/sce4 |
|---|---:|---:|---:|---:|---:|---:|
| `ma_2mo_AppP` | 235,239 | 235,245 | 235,239 | 6 | 0 | 39,129 / 42,528 |
| `ma_2mo_ApTt` | 721,120 | 721,126 | 721,089 | 37 | 31 | 81,500 / 88,134 |
| `ma_2mo_Ttau` | 518,134 | 518,138 | 518,122 | 16 | 12 | 48,060 / 52,687 |
| `ma_4mo_AppP` | 489,526 | 489,520 | 489,503 | 17 | 23 | 41,669 / 45,675 |
| `ma_4mo_ApTt` | 680,903 | 680,882 | 680,874 | 8 | 29 | 57,870 / 61,649 |
| `ma_4mo_Ttau` | 385,521 | 385,513 | 385,481 | 32 | 40 | 49,014 / 53,605 |
| `ma_6mo_AppP` | 87,653 | 87,655 | 87,645 | 10 | 8 | 25,251 / 26,025 |
| `ma_6mo_ApTt` | 149,195 | 149,194 | 149,179 | 15 | 16 | 28,994 / 31,294 |
| `ma_6mo_Ttau` | 198,908 | 198,919 | 198,905 | 14 | 3 | 27,722 / 29,861 |

Remaining source-`ps` residual classification after transgene removal:

| class | count |
|---|---:|
| missing non-transgene rows | 155 |
| missing present in our raw output | 155 |
| missing raw-absent | 0 |
| missing below SigProb gate | 0 |
| missing dominated by phospho PDS residual | 142 |
| missing dominated by non-phospho PDS residual | 13 |
| extra non-transgene rows | 162 |
| extra absent from frozen sce4 Pairwise reference | 162 |
| extra path seen elsewhere in frozen sce4 Pairwise reference | 98 |

Interpretation:

- Removing transgene paths collapses the literal gated-universe problem from hundreds of thousands of
  transgene-associated discrepancies to 155 missing and 162 extra rows in the best source-`ps`
  diagnostic.
- The remaining missing rows are not a gene-list absence: every one is present in our raw rerun and
  none fails the SigProb floor. They are PDS threshold-edge rows, mostly phospho-driven.
- The remaining extra rows are not explained by the available sce4 `10302025` Allpathway files; they
  are absent from the frozen Pairwise reference for the same contrast. Most paths are observed
  elsewhere in sce4, so this still looks like threshold-edge scoring/history rather than impossible
  pathway enumeration.
- Top300 parity remains materially worse than gated-universe parity because Top300 is rank-sensitive.
  Even after removing transgenes, source-`ps` shares 399,209 / 431,458 sce4 Top300 rows (92.53%).
  The remaining Top300 residual is therefore a score/ranking issue, not a broad candidate-universe
  issue.

### Step K: PDS Score-Influence Audit - Complete

Script:

```bash
/home/hchung/.pixi/bin/pixi run Rscript alz/incytr_pair/audit_pds_score_influence.R
```

Output:

```text
outputs/reports/incytr_pair_mode/forensics/sce4_pds_score_influence_summary.csv
```

This audit decomposes `delta_PDS = ours_PDS - sce4_PDS` on shared gated rows into weighted
contributors matching the current frozen-geneuse formula:

```text
TPDS
0.5 * PPDS
0.5 * PhPDS_ps
0.5 * PhPDS_py
signed SiK adjustment = (PDS - multimodel_score)
other multimodel residual
```

Source-`ps` diagnostic, dominant weighted contributor on shared rows:

| scope | shared rows | rows with abs(delta PDS) > 0.05 | dominant `PhPDS_ps` | dominant `PhPDS_py` | dominant `PPDS` | dominant `TPDS` | dominant SiK |
|---|---:|---:|---:|---:|---:|---:|---:|
| non-transgene | 3,466,037 | 346,766 | 1,764,072 | 868,261 | 780,271 | 466 | 52,967 |
| transgene | 705,723 | 122,285 | 264,317 | 102,679 | 272,668 | 62,159 | 3,900 |

Source-`ps` non-transgene drift magnitude by weighted contributor:

| weighted contributor | dominant rows | worst median abs drift | worst p95 abs drift | worst p99 abs drift | max abs drift |
|---|---:|---:|---:|---:|---:|
| `0.5 * PhPDS_py` | 868,261 | 0.0001540 | 0.1250000 | 0.2471049 | 0.3746592 |
| `0.5 * PhPDS_ps` | 1,764,072 | 0.0018417 | 0.0711975 | 0.1250000 | 0.2660745 |
| signed SiK adjustment | 52,967 | 0 | 0 | 0.0825000 | 0.2173180 |
| `0.5 * PPDS` | 780,271 | 0.0008160 | 0.0069520 | 0.0109810 | 0.1254979 |
| `TPDS` | 466 | 0 | 0 | 0 | 0.0975122 |

Canonical comparison for context:

| scope | shared rows | rows with abs(delta PDS) > 0.05 | dominant `PhPDS_ps` | dominant `PhPDS_py` | dominant `PPDS` | dominant `TPDS` | dominant SiK |
|---|---:|---:|---:|---:|---:|---:|---:|
| non-transgene | 3,465,769 | 410,987 | 1,839,185 | 848,025 | 725,930 | 385 | 52,244 |
| transgene | 705,736 | 148,874 | 293,237 | 100,065 | 249,552 | 59,132 | 3,750 |

Interpretation:

- `PDS` differences are mostly inherited from `multimodel_score` differences, not from a separate
  downstream `PDS` formula discrepancy.
- In the non-transgene shared universe, the largest weighted contributor is usually phospho,
  especially `PhPDS_ps`; source-`ps` reduces but does not eliminate this class.
- `PhPDS_py` remains substantial because the source-`ps` diagnostic intentionally swapped only `ps`,
  not `py`.
- `PPDS` is the next meaningful contributor. This is consistent with the earlier protein provenance
  audit: canonical protein is closer to sce4 than the complete v2/source protein, but small protein
  residuals remain widespread.
- `TPDS` is negligible for non-transgene shared rows after the q0 path-level fix, but it is still a
  visible contributor in transgene rows. That is consistent with transgene effects being a separate
  score/provenance problem, not ordinary pathway biology.
- For the 155 source-`ps` non-transgene missing rows after transgene removal, the row-level mismatch
  audit agrees with this decomposition: 142 are phospho-threshold residuals and 13 are non-phospho
  PDS residuals. By largest raw component, 72 are `PhPDS_py`, 70 are `PhPDS_ps`, 12 are `PPDS`, and 1
  is `multimodel_score`.
- The source-`ps` diagnostic improves but does not solve score parity: canonical non-transgene rows
  with `abs(delta PDS) > 0.05` total 410,987, while source-`ps` reduces that to 346,766.

### Step L: Phospho Engine Trace - Complete

Script:

```bash
/home/hchung/.pixi/bin/pixi run Rscript alz/incytr_pair/audit_phospho_engine_trace.R
```

Outputs:

```text
outputs/reports/incytr_pair_mode/forensics_source_ps_full_diag/phospho_engine_trace/phospho_engine_trace_detail.csv
outputs/reports/incytr_pair_mode/forensics_source_ps_full_diag/phospho_engine_trace/phospho_engine_trace_summary.csv
```

This trace recomputes source-`ps` diagnostic phospho scores from the actual current input bundle
`data/derived/incytr_inputs_source_ps_diag/` using the current package-default handling:

```text
duplicate gene rows: mean raw collapse
normalization: limma::normalizeBetweenArrays on condition1/condition2
fold-change correction: 0.001
q: 0.75
PhPDS: rowMeans(logi(node aFC, k = 2), NA -> 0)
```

Result on the 155 source-`ps` non-transgene missing rows:

| metric | rows | median abs candidate-vs-ours | p95 abs candidate-vs-ours | max abs candidate-vs-ours | median abs candidate-vs-sce4 | p95 abs candidate-vs-sce4 | max abs candidate-vs-sce4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PhPDS_ps` | 155 | 1.25e-16 | 4.44e-16 | 5.55e-16 | 0.0123351 | 0.2089998 | 0.2450114 |
| `PhPDS_py` | 155 | 2.08e-17 | 3.89e-16 | 5.27e-16 | 0.0121950 | 0.2494172 | 0.2695209 |

Interpretation:

- The current driver/package path from the available input bundle to our output is internally
  consistent for phospho scores.
- The residual is not caused by the audit recomputation using the wrong current formula.
- The residual is between current package-default phospho handling and sce4's frozen score values.
  That points to historical input/preprocessing/provenance or historical package behavior, not a
  simple current-driver arithmetic error.
- The preprocessing-grid audit in
  `outputs/reports/incytr_pair_mode/forensics_source_ps_full_diag/phospho_input_handling/` found that
  the best tested node-level match is still the current package-default style (`mean_raw`,
  `normalize=TRUE`, `q=0.75`, `correction=0.001`), but even that remains materially off from sce4:
  best source `ps` node median abs delta `0.0319426`, p95 `0.3305288`; best derived `py` node median
  abs delta `0.0580835`, p95 `0.4562811`.

## 8. Commands and Artifacts

Key scripts:

```text
alz/incytr_pair/incytr_commandline.R
alz/incytr_pair/run_pair_mode.sh
alz/incytr_pair/run_sce4_full_unfiltered.sh
alz/incytr_pair/extract_sce4_geneuse.R
alz/incytr_pair/verify_sce4_full.R
alz/incytr_pair/audit_sce4_mismatches.R
alz/incytr_pair/audit_pds_provenance.R
alz/incytr_pair/audit_ppds_provenance.R
alz/incytr_pair/audit_extra_universe.R
alz/incytr_pair/audit_transgene_excluded_reproduction.R
alz/incytr_pair/audit_pds_score_influence.R
alz/incytr_pair/audit_phospho_engine_trace.R
alz/incytr_pair/filter_significant_paths.py
```

Current diagnostic outputs:

```text
outputs/reports/incytr_pair_mode/_sce4_one_contrast_q0/
outputs/reports/incytr_pair_mode/sce4_full_verify_one_contrast_q0.csv
outputs/reports/incytr_pair_mode/_sce4_one_contrast_source_omics_q0/
outputs/reports/incytr_pair_mode/sce4_full_verify_one_contrast_source_q0.csv
outputs/reports/incytr_pair_mode/_sce4_full_q0/
outputs/reports/incytr_pair_mode/sce4_full_verify_full_q0.csv
outputs/reports/incytr_pair_mode/_sce4_one_contrast_v2_46bundle/
outputs/reports/incytr_pair_mode/sce4_full_verify_ma_4mo_Ttau_v2_46bundle.csv
outputs/reports/incytr_pair_mode/_sce4_one_contrast_source_ps_diag/
outputs/reports/incytr_pair_mode/sce4_full_verify_ma_4mo_Ttau_source_ps_diag.csv
outputs/reports/incytr_pair_mode/_sce4_full_source_ps_diag/
outputs/reports/incytr_pair_mode/sce4_full_source_ps_diag_run.log
outputs/reports/incytr_pair_mode/sce4_full_verify_full_source_ps_diag.csv
outputs/reports/incytr_pair_mode/forensics/ma_4mo_Ttau_*_audit.csv
outputs/reports/incytr_pair_mode/forensics_v2_46bundle/
outputs/reports/incytr_pair_mode/forensics_source_ps_diag/
outputs/reports/incytr_pair_mode/forensics_source_ps_full_diag/
outputs/reports/incytr_pair_mode/forensics_source_ps_full_diag/sce4_all_contrasts_mismatch_audit_summary.csv
outputs/reports/incytr_pair_mode/forensics/sce4_pds_provenance_*.csv
outputs/reports/incytr_pair_mode/forensics/ma_4mo_Ttau_ppds_provenance_*.csv
outputs/reports/incytr_pair_mode/forensics_v2_46bundle/ma_4mo_Ttau_ppds_provenance_*.csv
outputs/reports/incytr_pair_mode/forensics/sce4_extra_universe_*.csv
outputs/reports/incytr_pair_mode/forensics/sce4_extra_in_09062025_allpathway_matches.csv
outputs/reports/incytr_pair_mode/forensics/transgene_removed_top300_summary.csv
outputs/reports/incytr_pair_mode/forensics/sce4_pds_score_influence_summary.csv
outputs/reports/incytr_pair_mode/forensics_source_ps_full_diag/phospho_input_handling/
outputs/reports/incytr_pair_mode/forensics_source_ps_full_diag/phospho_engine_trace/
```

Known important implementation note:

- `incytr_commandline.R` currently recalibrates path-level `aFC`, `TPDS`, `multimodel_score`, and
  `PDS` in frozen sce4 gene-use mode so path-level SigProb `aFC` follows sce4's observed q0 behavior.
  This improved the tested contrast from 33 non-transgene missing rows to 6.

## 9. Current Reproduction Claim

As of this document:

We can nearly reproduce the non-transgene gated tuple set for one tested contrast using independent
scoring over sce4 frozen per-pair gene-use and the matched derived input bundle.

We cannot claim full exact sce4 reproduction because:

- the tested contrast still has 43,680 extra gated transgene-associated rows;
- the tested contrast still has 6 missing non-transgene gated rows caused by phospho/PDS residuals;
- Top300/capped membership is only 74.0% overlapped for the tested contrast;
- the completed full nine-contrast rerun has 423 missing non-transgene rows and 328 extra
  non-transgene rows across all contrasts;
- several contrasts have non-transgene extras, so this is no longer only a transgene/phospho-edge
  problem outside the original one-contrast baseline.
- the row-level audit across all nine contrasts shows two mechanisms: all 423 missing non-transgene
  rows are PDS threshold residuals already present in our raw output, while all 328 extra
  non-transgene rows are absent from the frozen sce4 Pairwise reference set.
- the complete v2/source bundle one-contrast test fixed most of the phospho-specific missing-row
  mechanism for `ma_4mo_Ttau`, but worsened exact tuple parity and shifted the missing-row mechanism
  to non-phospho PDS/`PPDS` residuals.
- the PPDS provenance audit shows the v2/source `PPDS` residual is protein input provenance: sce4
  frozen `PPDS` matches canonical derived protein closely, while v2/source protein reproduces the v2
  rerun but diverges from sce4 by about 0.19 median absolute `PPDS`.
- first-pass PDS provenance points to phospho input provenance, especially `ps`, not an obvious
  `../incytr` scorer formula bug. The source/v2 phospho file is closer to sce4 than canonical derived
  phospho, but using it requires a complete-bundle rerun rather than omics mixing.
- a targeted mixed diagnostic using canonical derived inputs plus v2/source `ps` sharply improves
  `ma_4mo_Ttau` tuple parity: missing non-transgene rows drop from 298 to 32, extra non-transgene rows
  drop from 206 to 40, and Top300 overlap improves from 45,915 to 49,064. This strongly implicates
  `ps` provenance, but the run is diagnostic only because it mixes input provenance.
- the extra-row universe audit did not find a wider local sce4 pre-gate universe. After extracting the
  complete local Drive-download `10302025` Allpathway set, the 328 extra non-transgene rows are still
  absent from Pairwise, Top300, and all 9 contrast-level Allpathway artifacts. Nearly all have
  same-pair role-present nodes and all are near the `abs(PDS) >= 0.2` threshold. The separate
  `09062025` score-bearing Allpathway candidate matches only 1/328 extra rows and does not provide
  exact pre-gate scores for this mismatch class.
- the opt-in transgene-excluded sensitivity rule substantially improves the biological-design
  comparison: with source-`ps`, non-transgene gated membership is 3,466,037 / 3,466,192 shared
  (99.996%), leaving 155 missing and 162 extra gated rows. It does not solve Top300 ranking parity:
  source-`ps` shares 399,209 / 431,458 transgene-excluded sce4 Top300 rows (92.53%).
- score-influence decomposition shows the remaining `PDS` residual is mainly subscore provenance:
  in source-`ps` non-transgene shared rows, the dominant weighted contributor is `PhPDS_ps` for
  1,764,072 rows, `PhPDS_py` for 868,261 rows, and `PPDS` for 780,271 rows; `TPDS` dominates only
  466 non-transgene rows after the q0 fix.
- phospho engine trace shows current input handling exactly reproduces our current `PhPDS_ps` and
  `PhPDS_py` outputs on the residual rows, but not sce4. This narrows the phospho problem to
  historical score reconstruction rather than an inconsistency between the audit and the current
  driver/package output.

The next milestone is to decide whether exact parity now requires either a true sce4 pre-gate artifact
or the exact historical sce4 scoring environment. With current local artifacts, the defensible
reproduction claim remains bounded by threshold-edge PDS residuals and rank-sensitive Top300 movement
rather than by an identifiable candidate-universe file mismatch.
