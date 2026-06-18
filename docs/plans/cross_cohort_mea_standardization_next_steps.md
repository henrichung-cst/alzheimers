# Cross-Cohort MEA Standardization Next Steps

Date: 2026-06-18
Status: implementation plan for small-agent execution

## Purpose

Standardize comparable kinase MEA analyses across structurally different
cohorts, building on the completed cohort abstraction refactor. The goal is not
to force identical biology onto every cohort. The goal is to apply the same
analysis conventions wherever the data support them:

- paired stoichiometry-vs-raw mechanism attribution for applicable bulk MEA
  cohorts;
- projected per-cell-state MEA for the T-cell cohort, using the same
  interpretation convention already used for projected cell-type MEA in other
  cohorts.

## Operating Rules For Agents

Each packet should be executable by a small agent with a narrow file scope. Do
not combine unrelated packets unless the user explicitly asks for a larger pass.

Default policy:

- Do not change `_run_mea` in `alz/bulk_mea/enrich.py`.
- Do not change MEA thresholds or permutation settings.
- Do not export synthetic numeric mechanism scores.
- Do not regenerate canonical outputs during helper/test packets.
- Use scratch outputs or `--dry-run` style checks until a packet explicitly
  updates a producer.
- Keep cohort-specific loading and contrast design in `alz/cohorts/...`.
- Put reusable classification/output logic in `alz/core/...`.
- If a cohort lacks paired raw/stoich evidence, emit `not_evaluable` or record a
  skip; do not infer a mechanism call.
- Preserve old canonical outputs unless the packet explicitly defines a new
  additive output file.

Every packet should finish with:

- `git status --short`
- `python -m py_compile` on changed Python files
- focused unit or smoke checks listed in the packet
- a short note listing files changed, commands run, and known skips

## Key Caveats

"All cohorts" means all applicable paired raw-vs-stoich bulk MEA cohorts:

- Song bulk
- Mukesh per-donor
- T-cell bulk time-course
- 5xFAD bulk

Do not automatically extend mechanism attribution to Song decomposition MEA or
5xFAD cell-type MEA unless a paired raw/stoich projected input is explicitly
defined.

T-cell per-state MEA should be labeled as projected state MEA, not direct
cell-state phosphoproteomics. This is the same convention already used for
Song/5xFAD projected cell-type MEA, so it is not a blocker.

T-cell donor2 is likely limited:

- donor2 has no IMAC/ST track;
- donor2 pY may lack usable flanking motifs;
- donor2 should be attempted only if the module can record structured skips
  rather than failing.

The first T-cell projected-state MEA target should be:

```text
donor = donor1
track = st
kind = stoich + raw if both can be built
states = ProjecTILs functional.cluster labels
contrast = later day vs baseline day
```

## Deliverable Summary

Primary new code targets:

```text
alz/core/mechanism_attribution.py
alz/cohorts/tcells/state_mea.py
```

Likely modified producers:

```text
alz/bulk_mea/mechanism.py
alz/cohorts/mukesh/mea.py
alz/cohorts/tcells/mea.py
alz/cohorts/fivexfad/ingest.py
alz/core/validate_cohort.py
pixi.toml
```

Likely new tests:

```text
tests/test_mechanism_attribution.py
tests/test_tcells_state_mea.py
```

Likely new docs:

```text
docs/foundation/mechanism_attribution_contract.md
docs/foundation/projected_state_mea_contract.md
```

## Phase 0: Contracts And Inventory

### Packet 0A: Mechanism Attribution Contract

File scope:

```text
docs/foundation/mechanism_attribution_contract.md
```

Instructions:

1. Define mechanism attribution as a categorical comparison between paired
   stoich MEA and raw-phospho MEA evidence.
2. State that no numeric mechanism score is exported.
3. Define required evidence columns.
4. Define allowed categorical calls.
5. Define cohort-specific context keys.
6. Define skip/not-evaluable behavior.

Suggested required columns:

```text
cohort
track
kinase
stoich_NES
stoich_FDR
raw_NES
raw_FDR
stoich_significant
raw_significant
sign_relation
mechanism_call
skip_reason
```

Allowed extra context columns:

```text
contrast
donor
timepoint
tissue
state
kind
```

Categorical calls:

```text
both
activity_driven
abundance_driven
discordant
not_significant
not_evaluable
```

Suggested call definitions:

- `both`: stoich and raw are significant with compatible NES signs.
- `activity_driven`: stoich is significant and raw is not.
- `abundance_driven`: raw is significant and stoich is not.
- `discordant`: stoich and raw are significant with opposite NES signs.
- `not_significant`: neither row is significant.
- `not_evaluable`: one or both paired rows are absent or malformed.

Defer/stop conditions:

- If a proposed call would require a new numeric score, do not add it.
- If a cohort-specific context cannot be represented without changing existing
  MEA output schemas, record the issue and defer wiring for that cohort.

Verification:

```bash
rg -n "mechanism_score|score" docs/foundation/mechanism_attribution_contract.md
```

Any hit must be explanatory only, not a proposed exported field.

### Packet 0B: Projected State MEA Contract

File scope:

```text
docs/foundation/projected_state_mea_contract.md
```

Instructions:

1. Define projected state MEA as MEA on T-cell state-projected substrate values.
2. State clearly that this is not direct cell-state phosphoproteomics.
3. Reuse the same interpretation convention as Song and 5xFAD projected
   cell-type MEA.
4. Define eligible inputs and skip reasons.
5. Define minimum manifest fields.
6. Define first implementation target: donor1 ST.

Required manifest fields:

```text
donor
state
track
kind
baseline_day
days_available
days_run
n_cells_by_day
n_sites
n_motif_sites
input_files
skip_reason
```

Defer/stop conditions:

- If the T-cell deconvolution inputs are missing, do not regenerate them in this
  packet. Record the prerequisite command.
- If donor2 pY has no motif support, mark donor2 pY as not evaluable in the
  contract.

Verification:

```bash
rg -n "direct cell-state phosphoproteomics|projected" docs/foundation/projected_state_mea_contract.md
```

The contract must include both the limitation and the projected interpretation.

### Packet 0C: Input Inventory Script Or Manual Audit

File scope:

```text
docs/audits/cross_cohort_mea_standardization_input_inventory.md
```

Instructions:

Create a short audit table listing whether required inputs exist locally for:

- Song bulk stoich/raw MEA
- Mukesh stoich/raw MEA
- T-cell bulk stoich/raw MEA
- 5xFAD bulk stoich/raw MEA
- T-cell projected state inputs

For each input family, record:

```text
cohort
path pattern
present locally: yes/no
expected producer
safe to run now: yes/no
notes
```

Use shell checks only. Do not regenerate data.

Suggested commands:

```bash
ls outputs/reports/kinase_attribution/*mea* 2>/dev/null
ls outputs/reports/kinase_attribution_human/perdonor/*mea* 2>/dev/null
ls outputs/reports/kinase_attribution_tcells/donor*/mea/*mea* 2>/dev/null
ls outputs/reports/kinase_attribution_5xfad/*mea* 2>/dev/null
ls data/derived/tcells_incytr_inputs/donor1/*deconvoluted.csv 2>/dev/null
```

Defer/stop conditions:

- If large files are missing, do not run full pipelines automatically.
- If an expected path pattern differs from the plan, update the plan or audit
  before implementing code against the wrong path.

Verification:

The audit file exists and contains no generated analysis outputs.

## Phase 1: Shared Mechanism Attribution Helper

### Packet 1A: Pure Classification Helper

File scope:

```text
alz/core/mechanism_attribution.py
tests/test_mechanism_attribution.py
.gitignore  # only if tests/ is not already tracked
```

Instructions:

Implement a pure helper that takes two MEA long DataFrames and returns a
classified DataFrame. Do not read or write files in the core function.

Suggested public API:

```python
def classify_mechanisms(
    stoich_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    context_cols: list[str],
    fdr_thresh: float = config.MEA_FDR_THRESH,
) -> pd.DataFrame:
    ...
```

Expected input columns:

```text
kinase
NES
FDR
```

plus every `context_cols` field.

Implementation notes:

1. Select only required columns from each input.
2. Rename evidence columns:
   - `NES` -> `stoich_NES` / `raw_NES`
   - `FDR` -> `stoich_FDR` / `raw_FDR`
3. Outer-join on `context_cols + ["kinase"]`.
4. Mark rows missing one side as `not_evaluable`.
5. Compute boolean significance flags using `fdr_thresh`.
6. Compute `sign_relation` as:
   - `same`
   - `opposite`
   - `stoich_only`
   - `raw_only`
   - `none`
   - `not_evaluable`
7. Assign `mechanism_call`.
8. Return deterministic sorted rows by context columns and kinase.

Do not add any numeric score besides raw evidence columns and boolean flags.

Suggested tests:

- both significant, same sign -> `both`
- stoich only -> `activity_driven`
- raw only -> `abundance_driven`
- both significant, opposite sign -> `discordant`
- neither significant -> `not_significant`
- missing raw row -> `not_evaluable`
- missing stoich row -> `not_evaluable`
- custom context columns are preserved

Defer/stop conditions:

- If existing MEA tables use inconsistent column names beyond `kinase`, `NES`,
  and `FDR`, do not add cohort-specific hacks to the helper. Add small
  cohort-specific adapter functions later.
- If tests require large real files, stop and replace them with in-memory
  fixtures.

Verification:

```bash
python tests/test_mechanism_attribution.py
python -m py_compile alz/core/mechanism_attribution.py tests/test_mechanism_attribution.py
```

### Packet 1B: File-Level Wrapper Functions And CLI

File scope:

```text
alz/core/mechanism_attribution.py
tests/test_mechanism_attribution.py
```

Instructions:

Add a thin file-level wrapper around the pure helper.

Suggested API:

```python
def classify_mechanism_files(
    stoich_path: str | Path,
    raw_path: str | Path,
    out_path: str | Path,
    *,
    context_cols: list[str],
    cohort: str | None = None,
    extra_constant_cols: dict[str, str] | None = None,
) -> pd.DataFrame:
    ...
```

Suggested CLI:

```bash
python -m alz.core.mechanism_attribution \
  --stoich <CSV> \
  --raw <CSV> \
  --out <CSV> \
  --context contrast --context track \
  --cohort song
```

Implementation notes:

1. Read CSV inputs.
2. Apply `extra_constant_cols` before classification if needed.
3. Add `cohort` column to output if provided.
4. Write output CSV only in the wrapper, not the pure helper.
5. Make `--help` work without importing large cohort modules.

Defer/stop conditions:

- If argparse complexity grows beyond a thin wrapper, keep CLI minimal and
  leave producer integration to cohort modules.

Verification:

```bash
python -m alz.core.mechanism_attribution --help
python tests/test_mechanism_attribution.py
python -m py_compile alz/core/mechanism_attribution.py
```

## Phase 2: Mechanism Attribution Wiring For Bulk Cohorts

### Packet 2A: Refactor Song Mechanism To Shared Helper

File scope:

```text
alz/bulk_mea/mechanism.py
alz/core/mechanism_attribution.py  # only if a small adapter is needed
tests/test_mechanism_attribution.py
```

Instructions:

Refactor Song's existing raw-vs-stoich classification to call
`classify_mechanisms`. Preserve the existing outputs unless the user approves a
schema change.

Current expected Song files:

```text
outputs/reports/kinase_attribution/mea_stoichiometry{,_pY}.csv
outputs/reports/kinase_attribution/mea_raw_phospho{,_pY}.csv
outputs/reports/kinase_attribution/mechanism_annotation.csv
```

Suggested approach:

1. Keep the raw MEA generation logic in `mechanism.py`.
2. Replace local `_classify_mechanisms` logic with shared helper call.
3. If legacy output columns differ, write legacy columns from shared output by a
   small conversion function.
4. Optionally add standardized output:

```text
outputs/reports/kinase_attribution/mechanism_attribution.csv
```

Do not remove `mechanism_annotation.csv` in this packet.

Defer/stop conditions:

- If preserving `mechanism_annotation.csv` conflicts with the standardized
  schema, keep the legacy file and add the standardized file separately.
- If current Song raw MEA outputs are missing, do not run full Song analysis
  unless explicitly requested. Use small fixture tests and document that runtime
  verification is pending.

Verification:

```bash
python -m py_compile alz/bulk_mea/mechanism.py alz/core/mechanism_attribution.py
python alz/bulk_mea/mechanism.py --help 2>/dev/null || true
```

If local data are present and the user has approved producer runs:

```bash
python alz/bulk_mea/mechanism.py
```

### Packet 2B: Mukesh Mechanism Output

File scope:

```text
alz/cohorts/mukesh/mea.py
alz/core/mechanism_attribution.py  # import only, avoid logic edits
alz/cohorts/mukesh/README.md
```

Instructions:

Add mechanism attribution output for paired Mukesh per-donor MEA files.

Inputs per track:

```text
mea_perdonor{suffix}.csv
mea_perdonor_raw{suffix}.csv
```

where `suffix` is `""` for ST and `"_pY"` for pY.

Outputs:

```text
mechanism_attribution{suffix}.csv
```

Implementation suggestion:

1. Add a helper:

```python
def _write_mechanism_attribution(track: str, out_dir: str = PERDONOR_DIR) -> None:
    ...
```

2. Use context columns:

```text
donor
```

3. Add constant columns:

```text
cohort = mukesh
track = st or py
```

4. Call after both stoich and raw `_run_track_kind` calls complete for a track.
5. If one file is absent, write no output and print a clear skip; do not fail
   the whole MEA command.

Defer/stop conditions:

- If Mukesh raw MEA is absent by design for a track, emit a skip message and
  avoid creating a misleading partial file.
- If adding this to the production MEA command risks modifying existing
  canonical behavior, add an opt-in flag first:

```text
--mechanism-attribution
```

Verification:

```bash
python -m alz.cohorts.mukesh.mea --help
python -m py_compile alz/cohorts/mukesh/mea.py
```

If local outputs exist, run a read-only file wrapper into a scratch path before
production wiring.

### Packet 2C: T-cell Bulk Mechanism Output

File scope:

```text
alz/cohorts/tcells/mea.py
alz/cohorts/tcells/README.md
```

Instructions:

Add mechanism attribution output for paired T-cell time-course MEA files.

Inputs per donor/track:

```text
mea_timecourse{suffix}.csv
mea_timecourse_raw{suffix}.csv
```

Outputs:

```text
mechanism_attribution{suffix}.csv
```

Implementation suggestion:

1. Add helper:

```python
def _write_mechanism_attribution(donor: str, track: str, out_dir: str) -> None:
    ...
```

2. Use context columns:

```text
timepoint
```

3. Add constant columns:

```text
cohort = tcells
donor = donor1/donor2
track = st/py
```

4. Call after each donor's track/kind loop completes, or after both kinds for a
   specific track complete.
5. Preserve the existing `mea_manifest.json` structure. If useful, add a new
   manifest key such as `mechanism_attribution` but do not change existing keys.

Defer/stop conditions:

- If donor2 has no MEA long tables, write no mechanism file and keep manifest
  skips clear.
- If pY motif limitations produce empty MEA, classify as not evaluable only if
  a paired row exists; otherwise skip file creation and record in manifest.

Verification:

```bash
python -m alz.cohorts.tcells.mea --help
python -m py_compile alz/cohorts/tcells/mea.py
```

### Packet 2D: 5xFAD Bulk Mechanism Output

File scope:

```text
alz/cohorts/fivexfad/ingest.py
alz/cohorts/fivexfad/README.md
```

Instructions:

Add mechanism attribution output for paired 5xFAD bulk MEA files.

Inputs per tissue/track prefix:

```text
{prefix}_mea_stoichiometry.csv
{prefix}_mea_raw_phospho.csv
```

Output:

```text
{prefix}_mechanism_attribution.csv
```

Implementation suggestion:

1. Add helper:

```python
def _write_mechanism_attribution(prefix: str, tissue: str, track: str) -> None:
    ...
```

2. Use context columns:

```text
contrast
```

3. Add constant columns:

```text
cohort = fivexfad
tissue = cortex/hippocampus
track = st/py
```

4. Call in `run_mea()` after `fit_track()` writes paired MEA results.
5. Do not change `fit_track()` return shape unless needed. Prefer a post-write
   helper that reads the two DataFrames already returned in `results`.

Defer/stop conditions:

- If one of the two MEA DataFrames is empty, write a mechanism file only if the
  helper can represent missing paired evidence honestly. Otherwise skip and
  print a reason.

Verification:

```bash
python -m alz.cohorts.fivexfad.ingest --help
python -m py_compile alz/cohorts/fivexfad/ingest.py
```

### Packet 2E: Pixi Tasks For Mechanism Attribution

File scope:

```text
pixi.toml
README.md  # minimal command documentation only
```

Instructions:

Add tasks only after at least one cohort integration exists and is verified.
Prefer cohort-specific tasks rather than one large "all mechanism" task at
first.

Suggested tasks:

```toml
human-mechanism = "python -m alz.cohorts.mukesh.mea --mechanism-attribution"
tcells-mechanism = "python -m alz.cohorts.tcells.mea --mechanism-attribution"
5xfad-mechanism = "python -m alz.cohorts.fivexfad.ingest --mechanism-attribution"
```

If mechanism files are written automatically during the existing MEA commands,
do not add duplicate tasks unless the user wants explicit rerun commands.

Defer/stop conditions:

- If producers auto-write mechanism outputs, skip this packet and document that
  the existing MEA tasks cover mechanism attribution.

Verification:

```bash
rg -n "mechanism" pixi.toml README.md
```

## Phase 3: T-cell Projected State MEA

### Packet 3A: State MEA Input Loader And Matrix Builder

File scope:

```text
alz/cohorts/tcells/state_mea.py
tests/test_tcells_state_mea.py
```

Instructions:

Create the new module but do not call `_run_mea` yet. Implement only input
parsing and matrix construction from small fixtures.

Inputs to support:

```text
data/derived/tcells_incytr_inputs/<donor>/pr_deconvoluted.csv
data/derived/tcells_incytr_inputs/<donor>/ps_deconvoluted.csv
data/derived/tcells_incytr_inputs/<donor>/py_deconvoluted.csv
data/derived/tcells_incytr_inputs/<donor>/scrna/cell_counts.csv
```

Column pattern from deconvolution:

```text
d{day}_{state}
```

Implementation suggestions:

1. Add parser:

```python
def parse_state_day_columns(columns: Iterable[str]) -> list[StateDayColumn]:
    ...
```

2. Add loader:

```python
def load_projected_inputs(donor: str, track: str, root: Path | None = None) -> ProjectedInputs:
    ...
```

3. Add matrix builder:

```python
def build_state_matrices(inputs: ProjectedInputs, state: str, track: str) -> dict[str, pd.DataFrame]:
    ...
```

4. For raw matrix, use projected phospho values.
5. For stoich matrix, compute:

```text
log2(projected phospho) - log2(projected protein[gene])
```

6. Handle non-positive projected values as NaN before log2.
7. Preserve metadata columns:

```text
site_id
gene_symbol
motif
```

Defer/stop conditions:

- If projected protein cannot be matched by `gene_symbol`, do not invent a
  fallback. Emit skip/not-evaluable for stoich and allow raw-only matrix
  construction.
- If state/day columns do not match `d{day}_{state}`, stop and update the parser
  based on actual files before proceeding.
- If tests require real large files, replace with tiny fixture DataFrames.

Verification:

```bash
python tests/test_tcells_state_mea.py
python -m py_compile alz/cohorts/tcells/state_mea.py tests/test_tcells_state_mea.py
python -m alz.cohorts.tcells.state_mea --help
```

### Packet 3B: State/Day QC And Manifest Skeleton

File scope:

```text
alz/cohorts/tcells/state_mea.py
tests/test_tcells_state_mea.py
```

Instructions:

Add QC functions and a manifest writer before MEA execution.

Suggested functions:

```python
def summarize_state_qc(inputs: ProjectedInputs) -> pd.DataFrame:
    ...

def should_run_state(
    *,
    n_cells_by_day: dict[int, int],
    n_motif_sites: int,
    baseline_day: int,
    days_available: list[int],
) -> tuple[bool, str | None]:
    ...
```

Use implementation-internal gates for mechanics only. Do not export them as
analysis scores. If thresholds are needed, name them as minimum requirements,
for example `min_motif_sites`, and record them in manifest as criteria, not
scores.

Suggested initial gates:

- baseline day exists;
- at least one later day exists;
- motif-bearing sites are nonzero;
- state has cells at baseline and target day.

Do not choose a high minimum cell threshold without reviewing current
`cell_counts.csv`. Start with "present and nonzero" unless a documented
threshold already exists.

Defer/stop conditions:

- If choosing a cell-count threshold would materially affect biological
  inclusion, stop and ask for a threshold decision.
- If motif availability is zero for all states, stop before implementing MEA.

Verification:

```bash
python tests/test_tcells_state_mea.py
python -m alz.cohorts.tcells.state_mea --donor donor1 --track st --dry-run
```

`--dry-run` should write or print QC only, not run MEA.

### Packet 3C: Projected State MEA Execution For Donor1 ST

File scope:

```text
alz/cohorts/tcells/state_mea.py
alz/core/mea_runner.py  # import only; avoid edits unless necessary
tests/test_tcells_state_mea.py
```

Instructions:

Wire donor1 ST projected state matrices into the shared MEA runner or directly
into `_run_mea` through the same runner contract used by other cohorts.

Preferred approach:

- Use `MeaRunner` with a small adapter in `state_mea.py`.
- Treat each `(donor, state, track, kind)` as a `MeaUnit`.
- Use `long_table_stem="mea_projected_state"`.
- Store `state` and `donor` in `unit.meta`.

Contrast construction:

1. Identify baseline day using existing T-cell baseline convention.
2. For each later day, compute:

```text
lfc = value_at_day - value_at_baseline
```

3. Use contrast names compatible with T-cell time-course naming where possible.

Outputs for first implementation:

```text
outputs/reports/kinase_attribution_tcells/donor1/state_mea/
  mea_projected_state.csv
  mea_projected_state_raw.csv
  mea_global_shift_projected_state.csv
  winsorized_sites_projected_state.csv
  mea_substrate_sets_projected_state.csv
  projected_state_mea_manifest.json
```

If the shared runner's fixed filenames are awkward, write to a scratch directory
first and only promote final filenames after review.

Defer/stop conditions:

- If `MeaRunner` cannot represent state-level output cleanly without changing
  its public contract, do not edit the runner first. Implement a local wrapper
  around `_run_mea` and record the runner limitation.
- If output filename choices conflict with existing validators/viewer paths,
  write under a scratch directory and stop for review.
- If donor1 ST inputs are absent locally, complete code and fixture tests but
  mark runtime verification pending.

Verification:

```bash
python -m alz.cohorts.tcells.state_mea --donor donor1 --track st --dry-run
python -m alz.cohorts.tcells.state_mea --donor donor1 --track st --runner-scratch-dir outputs/reports/refactor_audit/tcells_state_mea_donor1_st
python -m py_compile alz/cohorts/tcells/state_mea.py
```

### Packet 3D: Extend State MEA To pY And Structured Skips

File scope:

```text
alz/cohorts/tcells/state_mea.py
tests/test_tcells_state_mea.py
```

Instructions:

Extend CLI choices to:

```text
--donor donor1|donor2|both
--track st|py|both
```

Attempt pY only through the same QC gate. Donor2 should not fail the run if pY
has no usable motifs; it should produce structured skips.

Manifest must include skip records for:

- missing projected phospho file;
- missing projected protein file;
- missing baseline day;
- no later day;
- no motif-bearing sites;
- no cell count for state/day;
- empty MEA result.

Defer/stop conditions:

- If pY motif parsing differs from existing T-cell pY matrix motifs, stop and
  reuse the existing T-cell motif handling rather than writing a new parser.
- If donor2 creates only skips, that is acceptable. Do not force outputs.

Verification:

```bash
python -m alz.cohorts.tcells.state_mea --donor both --track both --dry-run
python tests/test_tcells_state_mea.py
```

### Packet 3E: Aggregates For State MEA

File scope:

```text
alz/cohorts/tcells/state_mea.py
alz/core/mea_outputs.py  # import only unless a generic helper is missing
tests/test_tcells_state_mea.py
```

Instructions:

Add wide NES/FDR outputs and recurrence summaries for projected state MEA.

Recommended first aggregate shape:

```text
kinase x (state, timepoint)
```

Because this is new output, prefer explicit long tables over overly wide tables
if a wide shape becomes ambiguous.

Suggested files:

```text
kinase_state_timepoint_nes{,_raw}{,_pY}.csv
kinase_state_timepoint_fdr{,_raw}{,_pY}.csv
recurrence_projected_state{,_raw}{,_pY}.csv
```

Implementation notes:

- Reuse `alz/core/mea_outputs.py` only if it fits naturally.
- If recurrence over both state and timepoint is ambiguous, defer recurrence and
  emit only long MEA plus NES/FDR matrices.
- Do not create a "state score" or rank beyond existing NES/FDR.

Defer/stop conditions:

- If there is no defensible recurrence axis, skip recurrence in this packet and
  document the open design question.

Verification:

Use small fixture MEA output to test aggregate shape before running real data.

## Phase 4: Mechanism Attribution For T-cell Projected State MEA

### Packet 4A: Projected State Mechanism Files

File scope:

```text
alz/cohorts/tcells/state_mea.py
alz/core/mechanism_attribution.py  # import only
tests/test_tcells_state_mea.py
```

Instructions:

After paired projected stoich/raw state MEA outputs exist, apply the shared
mechanism helper.

Input:

```text
mea_projected_state{suffix}.csv
mea_projected_state_raw{suffix}.csv
```

Output:

```text
mechanism_attribution_projected_state{suffix}.csv
```

Context columns:

```text
state
timepoint
```

Constant columns:

```text
cohort = tcells
donor = donor1/donor2
track = st/py
projection = projected_state
```

Defer/stop conditions:

- If only raw or only stoich projected MEA exists, do not force partial
  mechanism calls. Emit `not_evaluable` only where an explicit paired index can
  be constructed.

Verification:

```bash
python tests/test_tcells_state_mea.py
python -m alz.cohorts.tcells.state_mea --donor donor1 --track st --mechanism-attribution --dry-run
```

## Phase 5: Validators

### Packet 5A: Mechanism File Validator

File scope:

```text
alz/core/validate_cohort.py
tests/test_mechanism_attribution.py  # only if validator fixtures are added here
```

Instructions:

Add validation for mechanism attribution files as optional/additive outputs.

Check:

- required columns exist;
- `mechanism_call` values are in allowed vocabulary;
- no forbidden exported score columns exist;
- evidence FDR/NES columns are numeric where present;
- context columns are present for each cohort.

Defer/stop conditions:

- If outputs are not yet generated locally, validator should warn/skip optional
  files rather than fail existing cohort validation.

Verification:

```bash
python -m alz.core.validate_cohort --cohort mukesh
python -m alz.core.validate_cohort --cohort tcells
python -m alz.core.validate_cohort --cohort fivexfad
python -m alz.core.validate_cohort --cohort song
```

### Packet 5B: T-cell State MEA Validator

File scope:

```text
alz/core/validate_cohort.py
```

Instructions:

Add optional validation for:

```text
outputs/reports/kinase_attribution_tcells/<donor>/state_mea/
```

Check:

- manifest JSON is valid if directory exists;
- MEA long tables contain `state`, `timepoint`, `kinase`, `NES`, `FDR`;
- skip records have `reason`;
- audit sidecars exist when MEA outputs exist.

Defer/stop conditions:

- Do not make state MEA required for T-cell validation until the user approves
  promoting it to canonical required output.

Verification:

```bash
python -m alz.core.validate_cohort --cohort tcells
```

## Phase 6: Viewer Integration

Viewer work should happen only after files validate.

### Packet 6A: T-cell Viewer Design Note

File scope:

```text
docs/foundation/viewer_payload_contract.md
docs/foundation/viewer_frontend_contract.md
```

Instructions:

Before coding viewer changes, document where projected state MEA appears:

- optional detail layer;
- bulk MEA remains primary;
- projected state MEA is attribution/supporting evidence;
- direct cell-state phosphoproteomics language is forbidden.

Defer/stop conditions:

- If payload shape is unclear, write the design note and stop before editing
  `alz/build_tcell_viewer.py`.

### Packet 6B: T-cell Viewer Payload

File scope:

```text
alz/build_tcell_viewer.py
alz/tcell_viewer/
alz/viewer_shared/template/js/
```

Instructions:

Add projected state MEA only as optional payload content. If files are absent,
viewer behavior should remain unchanged.

Implementation suggestions:

- Add a compact index first.
- Lazy-load large state MEA rows if needed.
- Surface mechanism attribution categorically.

Defer/stop conditions:

- If payload size increases materially, stop and switch to lazy shards.
- If frontend copy would imply direct cell-state phosphoproteomics, revise copy
  before implementation.

Verification:

```bash
python alz/build_tcell_viewer.py --payload --html --validate
python alz/viewer/verify_template.py
```

### Packet 6C: Unified Viewer Mechanism Labels

File scope:

```text
alz/viewer/cohorts/mukesh.py
alz/viewer/cohorts/fivexfad.py
alz/viewer/cohorts/song.py
alz/viewer/template/js/
alz/viewer_shared/template/js/
```

Instructions:

Expose mechanism attribution as categorical labels and raw evidence columns.
Do not add new numeric scores.

Defer/stop conditions:

- If mechanism files are absent, omit the block and keep viewer output stable.
- If integrating all cohorts at once is too broad, do Mukesh first, then 5xFAD,
  then Song.

Verification:

```bash
python alz/build_unified_viewer.py
python alz/viewer/verify_template.py
```

## Phase 7: End-To-End Checks

Run only after code-level packets pass.

Suggested smoke sequence:

```bash
python -m alz.core.mechanism_attribution --help
python -m alz.cohorts.mukesh.mea --help
python -m alz.cohorts.tcells.mea --help
python -m alz.cohorts.fivexfad.ingest --help
python -m alz.cohorts.tcells.state_mea --help
python alz/viewer/verify_template.py
```

Suggested full or data-dependent checks, only when local inputs are present and
the user approves producer runs:

```bash
python -m alz.cohorts.mukesh.mea --track both
python -m alz.cohorts.tcells.mea --donor both
python -m alz.cohorts.fivexfad.ingest --mea
python -m alz.cohorts.tcells.state_mea --donor donor1 --track st
```

Before accepting outputs:

- compare row counts to expected paired MEA inputs;
- check key uniqueness;
- check allowed categorical calls only;
- check no synthetic scores were exported;
- check old canonical outputs were not changed except for planned additive
  files.

## Recommended Execution Order

1. Packet 0A: mechanism contract.
2. Packet 0B: projected state MEA contract.
3. Packet 0C: input inventory.
4. Packet 1A: pure mechanism helper.
5. Packet 1B: file wrapper and CLI.
6. Packet 2A: Song mechanism refactor.
7. Packet 2B: Mukesh mechanism output.
8. Packet 2C: T-cell bulk mechanism output.
9. Packet 2D: 5xFAD bulk mechanism output.
10. Packet 3A: T-cell state matrix builder.
11. Packet 3B: T-cell state QC and manifest.
12. Packet 3C: donor1 ST projected state MEA.
13. Packet 3D: pY/donor2 structured skips.
14. Packet 3E: projected state aggregates.
15. Packet 4A: projected state mechanism attribution.
16. Packet 5A: mechanism validators.
17. Packet 5B: T-cell state MEA validators.
18. Packet 6A: viewer design note.
19. Packet 6B: T-cell viewer integration.
20. Packet 6C: unified viewer mechanism labels.

If a packet hits a significant barrier, stop at that packet, write a short
decision note under `docs/audits/`, and do not continue into dependent packets.
