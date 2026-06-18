# Mechanism Attribution Contract (Packet 0A)

## Scope

This contract defines Phase 0 Packet 0A outputs: categorical attribution of kinase
mechanism evidence by comparing paired stoichiometry-adjusted and raw-phospho MEA
rows for the same kinase and cohort context.

No numeric mechanism-derived field is exported by this contract. Downstream systems should use
`mechanism_call` as the public mechanism classification.

## Pairing Rule

Each mechanism-attribution row is defined by a pair of evidence rows that match on:

- `cohort`
- `track`
- `kinase`
- any selected context keys (section below)

The pair compares:

- stoich evidence columns (`stoich_*`)
- raw-phospho evidence columns (`raw_*`)

If either side of the pair is missing or malformed, the row is `not_evaluable`.

## Required Columns

All mechanism-attribution tables must contain, at minimum, these columns:

- `cohort` (`str`)
- `track` (`str`)
- `kinase` (`str`)
- `stoich_NES` (`float | null`)
- `stoich_FDR` (`float | null`)
- `raw_NES` (`float | null`)
- `raw_FDR` (`float | null`)
- `stoich_significant` (`bool`) — true when the stoich row exists and is considered significant by the cohort analysis gate
- `raw_significant` (`bool`) — true when the raw row exists and is considered significant by the cohort analysis gate
- `sign_relation` (`str`)
- `mechanism_call` (`str`)
- `skip_reason` (`str | null`)

### `sign_relation`

Allowed values:

- `same` — finite NES values have compatible sign
- `opposite` — finite NES values have opposite sign
- `stoich_only` — only stoich evidence is significant
- `raw_only` — only raw-phospho evidence is significant
- `none` — both evidence rows exist but neither is significant
- `not_evaluable` — context indicates pairing failed before comparison

`not_evaluable` is the only allowed terminal state for malformed pair joins; it is the
expected `sign_relation` when `mechanism_call = not_evaluable`.

## Allowed Categorical Calls

- `both`: stoich and raw are significant and `sign_relation = same`
- `activity_driven`: stoich is significant, raw is not significant, and `sign_relation = stoich_only`
- `abundance_driven`: raw is significant, stoich is not significant, and `sign_relation = raw_only`
- `discordant`: stoich and raw are significant and `sign_relation = opposite`
- `not_significant`: neither stoich nor raw is significant and `sign_relation = none`
- `not_evaluable`: one or both paired rows are absent or malformed

Any implementation should reject invalid combinations (for example, `raw_significant = true`
with `mechanism_call = activity_driven`) as contract violations.

## Cohort-Specific Context Keys

In addition to required columns, rows may include one or more of these context
columns when available:

- `contrast`
- `donor`
- `timepoint`
- `tissue`
- `state`
- `kind`

`kind` is a cohort-level provenance label for the evidence stream (for example,
`bulk`, `deconvolved`, `projected_state`), while `state` is only required for
state-projected workflows.

No other context columns are required by this packet; if a column is not applicable
for a cohort, it may be omitted.

## Skip and Not-Evaluable Rules

Set `mechanism_call = not_evaluable` and keep a human-readable `skip_reason` when:

- stoich row is missing
- raw row is missing
- pairing keys cannot be uniquely resolved (duplicate/multiple matches)
- NES or FDR fields are missing or malformed after parsing
- context prevents a lawful pairing (e.g., unsupported evidence branch)

`skip_reason` should be short and machine-actionable (e.g., `missing_raw_row`,
`missing_stoich_row`, `invalid_numeric_values`, `duplicate_pair_rows`).
