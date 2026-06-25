# G2 — Positive controls (skill)

**Deliverable (per grill):** a **skill**, not a static CSV and not a CI gate. Invoked on demand for whatever cohort is in play; it carries a curated handful of positive-control genes (kinase + non-kinase) with their *externally-expected* cell-type home and disease direction, looks up each one's **actual** values in the existing artifacts, and renders a **non-deterministic** (agent-judged, not threshold-gated) verdict on whether the specificity and enrichment metrics behave as expected. Output is a conversational report — no committed artifact, no pass/fail gate.

This intentionally rejects the three heavier forms that were on the table (static list-only, permanent verification-harness gate, viewer badge). The user's words: *"a local skill file that says pick a handful of kinases and run a non-deterministic signal check."*

## Why a skill, not a harness

The check is a **sanity signal**, not a contract. PHKG1's "expected astrocyte home" and ATP9A's "expected endothelial home" are hypotheses (the TODO writes them with a `?`), so the right output is a judged "agrees / off / borderline / not-built", which a fixed threshold cannot express. A deterministic harness would also have to freeze expected values, which couples the audit to whatever the pipeline currently emits (circular). Keeping it agent-judged keeps the expected values *external* (literature/atlas consensus) and the actual values *from our pipeline*, which is the comparison G2 actually wants.

## The artifact: one file

`.claude/skills/check-controls/SKILL.md` — frontmatter (`name: check-controls`, `description`, `allowed-tools: Read, Bash, Glob`) + numbered procedure. The curated control table lives **inside this file** (single source; "maintain the list" = edit the table). No companion data file — the check is agent-read, not machine-parsed, so a markdown table is sufficient and matches the user's "skill file that says…".

### Curated control table (seed)

Columns: `gene` · `kinase?` · `cohorts` · `expected cell-type home` · `expected AD direction` · `source of expectation` · `notes`.

Seed rows (the named candidates + expectation marked as **hypothesis** where the TODO used `?`):

| gene | kinase? | expected home | expected AD dir | basis |
|---|---|---|---|---|
| PHKG1 | yes | astrocyte *(hypothesis)* | TBD | TODO candidate; in MEA + `unified_attribution` |
| ATP9A | no (flippase) | endothelial / vascular *(hypothesis)* | TBD | TODO candidate; `atp9a_ad_export` precedent |
| APOE | no | astrocyte / microglia | up in AD | established AD biology |

The skill instructs the agent to **append a row** when it confirms a new robust control, and to keep `expected` columns sourced to external knowledge (never to our own pipeline output). Expectations that are still hypotheses stay flagged so the verdict reads "our metric disagrees with a *hypothesis*", not "our metric is wrong".

## Procedure the skill encodes

1. **Resolve cohort.** From the invocation arg, else infer from the working context. Map to a key in `alz/core/cohort_manifest.py` (`song` / `fivexfad` / `mukesh` / `tcells`). Read that cohort's `mea_output_kind` and output roots from the manifest.
2. **Select controls.** Filter the table to rows whose `cohorts` include the target (a control with no meaningful expectation for that cohort is skipped, reported as such). "A handful" — the table is deliberately small; the skill checks all applicable rows.
3. **Per control, branch on `kinase?`:**
   - **Kinase** → cell-type home from the cohort's kinase specificity surface; disease direction from the cohort's **MEA NES** sign.
   - **Non-kinase** → home from the proteome/atlas expression surface; direction from **expression LFC** (no substrate set ⇒ no NES).
4. **Look up actual values** (artifact map below). Show the number, not just the verdict.
5. **Judge** each control on each declared axis: `✓ as-expected` / `⚠ off` / `~ borderline` / `– not built / N/A`. Borderline = right cell-type family but weak tier, or related subtype, or sign present but small. The agent reasons about these; it does not apply a fixed cutoff.
6. **Report** a compact table to the conversation (control × axis × actual × verdict) plus a one-line overall read. **Write nothing to disk.** If a required artifact is missing, say which and which task builds it — do not fail.

## Artifact map (where "actual" comes from)

| surface | kinase cohorts | non-kinase / atlas |
|---|---|---|
| cell-type home (mouse) | `outputs/reports/wmb_expression/wmb_kinase_expression.csv` → top `specificity_score` celltype; or merged `unified_attribution.csv` (`wmb_top_celltype`/`wmb_concentration_tier`, `song_top_celltype`) | `wmb_proteome_expression.csv` |
| cell-type home (song snRNA) | `outputs/reports/snrna_integration/song_expression_specificity.csv` (`top_cluster`, `specificity_score`, `tau`) | same file (all genes) |
| cell-type home (human) | `human_reference_expression/{seaad,hbca}_kinase_specificity.csv` | `data/derived/aggregates/seaad/expression_by_supertype.csv`, `hbca/expression_by_class.csv` *(present on disk)* |
| disease direction | MEA NES: `mea_stoichiometry.csv` (song), `kinase_attribution_5xfad/`, `kinase_attribution_human/perdonor/`, `kinase_attribution_tcells/donor1/mea/` — sign + `FDR<0.25` | expression LFC: `sea_ad_supertype_lfc.csv`, snrna attribution LFC |

Several mouse specificity artifacts (`wmb_expression/`, `snrna_integration/`) are **not currently on disk** (audit finding). The skill falls back to `unified_attribution.csv` (present) and the human atlas aggregates (present), and reports "run `pixi run …`" for anything genuinely absent rather than erroring.

## Open interpretation (resolve at plan approval)

"**Non-deterministic**" — I read it as *agent-judged verdict* (step 5), not *random subset selection*. The table is small enough to check whole, so there is no rotating sample. If you meant "each run spot-checks a varying few", say so and step 2 becomes a sampling step. I built the plan on the judgment reading.

## Out of scope (deliberately not built)

- No committed report / CSV (that is the rejected harness form).
- No `verify_*`/`pixi` gate; the skill never blocks a pipeline run.
- No viewer badge or payload column.
- No T-cell-specific control biology authored now — if invoked for `tcells`, the skill checks whatever table rows apply and notes the gap. (`tcell_marker_validation.py` already validates T-cell *state calls* separately; G2 does not duplicate it.)

## Files touched

- **New:** `.claude/skills/check-controls/SKILL.md` (only file).
- **No** edits to pipeline code, configs, viewers, or `pixi.toml`.

## Verification

Self-test once: invoke for `song` and for `mukesh`, confirm each control resolves to a real artifact value and the verdicts read sensibly (PHKG1 home, APOE direction). Manual, in-session — consistent with a non-deterministic signal check.
