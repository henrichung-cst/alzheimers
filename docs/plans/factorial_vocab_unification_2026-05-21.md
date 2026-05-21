# Factorial Vocabulary Unification — 2026-05-21

## The problem

Two genotype vocabularies are propagating through the codebase:

- **Short form** (`WT`/`APP`/`T22`/`T22/APP`) — comes from raw TMT collaborator labels
- **Long form** (`WTyp`/`AppP`/`Ttau`/`ApTt`) — SAP canonical; used by `config.SAP_FACTORIAL`, snRNA h5ad `mutant` column, contrast names (`App_2mo`, `ApTt_4mo`), viewer outputs, `phospho_group_id`

Eight files carry short-form literals or local conversion dicts. Three files contain a local `GENO_DECODE` map (two going short→long, one going long→short) — the back-and-forth conversion is the smell that gives the duplication away.

Additionally, **`CONTRAST_COEFS`** (the 9 contrast linear-combinations) is duplicated in three files (with one renamed to `CONTRASTS`).

This is **not** a small-diff cleanup — it touches 8+ files. But ignoring it leaves a known-incoherent state in core analysis code, which is exactly what the "research pivots replace, they do not coexist" rule forbids.

## The two real boundaries

Short form enters the codebase from **two distinct upstream sources**, not one:

1. **`sample_mapping.csv:genotype` column** — `data_ingest.py` writes the raw TMT genotype token unchanged into this column.
2. **`animal_id` string** — the verbatim TMT collaborator label (e.g., `"1_C198(L)_M_2mo_WT"`). Embeds short form by historical accident. `animal_id` is a join key with external meaning and **should not be rewritten** — but the embedded token can be parsed out to long form at any consumer.

The cleanup strategy normalizes #1 and provides one shared helper for #2.

## Target end state

- **One** factorial coding: `config.SAP_FACTORIAL` (long-form keys).
- **One** contrast definition: `config.CONTRAST_COEFS` (moved from `kinase_enrich.py`).
- **One** animal_id parser: `config.parse_animal_id(s)` — regex + short→long internally, returns long-form `genotype`.
- `sample_mapping.csv:genotype` column is **long form** on disk.
- Zero local `GENOTYPE_CODING`, `GENO_DECODE`, or short-form literal lists anywhere outside `config.py` and the parser internals.

## Site inventory

Every short-form occurrence, classified by role.

| File | Lines | What it is | Role | Edit |
|------|-------|------------|------|------|
| `alz/config.py` | 531 | `SAP_FACTORIAL` (long-form) | canonical source | keep; add `CONTRAST_COEFS` + `parse_animal_id` |
| `alz/data_ingest.py` | 84 | `GENOTYPE_TO_SAP` short→long map | normalization at write boundary | extend: apply to `genotype` column before write |
| `alz/data_ingest.py` | 134-153 | `_parse_animal_id` regex (short-form group) | source of short form | keep regex (animal_id is raw); but use it to populate long-form `genotype` field via `GENOTYPE_TO_SAP` |
| `alz/kinase_enrich.py` | 44-49 | local `GENOTYPE_CODING` (short-form keys) | duplicate | **delete**; use `config.SAP_FACTORIAL` |
| `alz/kinase_enrich.py` | 52-62 | local `CONTRAST_COEFS` | duplicate (canonical-by-import) | **move to** `config.py`, import here |
| `alz/kinase_normalize.py` | 476 | `for geno in ["WT", "APP", "T22", "T22/APP"]` | short-form literal list | rewrite to long-form (`SAP_FACTORIAL.keys()`) |
| `alz/snrna_integration.py` | 354-364 | local `CONTRAST_COEFS` | pure duplicate | **delete**, import from `config` |
| `alz/decomposition/paths.py` | 71-76 | local `GENOTYPE_CODING` (long-form keys) | duplicate of `SAP_FACTORIAL` | **delete**, import `SAP_FACTORIAL` |
| `alz/decomposition/paths.py` | 78-88 | `CONTRASTS` (= `CONTRAST_COEFS` renamed) | duplicate | **delete**, import `CONTRAST_COEFS` |
| `alz/decomposition/factorial_ols.py` | 33, 78, 100 | `paths.GENOTYPE_CODING`, `paths.CONTRASTS` | consumer | retarget to `config` |
| `alz/decomposition/per_animal_extension.py` | 28-30, 173 | local `GENOTYPE_CODING` (short-form), `paths.CONTRASTS` | duplicate + retarget | **delete** local dict, use `config.SAP_FACTORIAL`; retarget contrasts |
| `alz/supplementary/deconvolution_feasibility.py` | 148-149 | `.isin(["APP", "T22/APP"])`, `.isin(["T22", "T22/APP"])` | short-form literal filter | rewrite to long-form (`["AppP", "ApTt"]`, `["Ttau", "ApTt"]`) |
| `alz/integration/build_omics_trace.py` | 117, 130 | local `GENO_DECODE` short→long + parsing | duplicate parser | **delete**, use `config.parse_animal_id` |
| `alz/incytr/export_decomposition_for_pair.py` | 46-47, 51-55 | local `GENO_DECODE` + `ANIMAL_RE` | duplicate parser | **delete**, use `config.parse_animal_id` |
| `alz/snrna_proportions.py` | 46, 49-61 | local `GENO_DECODE` long→**short** + `_decode_snrna_sample` | converts AWAY from canonical to match short-form sample_mapping | **delete** entire decoder — after sample_mapping is long-form, no conversion needed; snRNA sample IDs are already long form (`ma_2mo_AppP`), can be parsed/joined directly |

## Net change

- **Deleted**: 3 `GENOTYPE_CODING`/equivalent dicts, 3 `CONTRAST_COEFS`/`CONTRASTS` dicts, 3 `GENO_DECODE` dicts, 2 `ANIMAL_RE`/parser-regex duplicates, 1 short-form literal list, 1 long→short decoder function (~40 lines).
- **Added**: 1 `CONTRAST_COEFS` in `config.py` (moved, not new), 1 `parse_animal_id` helper in `config.py` (consolidated from 3 sites), 1 `.map(GENOTYPE_TO_SAP)` line in `data_ingest.py`.
- **Modified**: ~5 consumer sites switch from local literals to imports.

Net deletion across the codebase.

## Execution order (single commit)

1. **`config.py`** — add `parse_animal_id(s) -> dict` (lifted from `data_ingest._parse_animal_id`, normalizes genotype to long form internally). Add `CONTRAST_COEFS` (lifted from `kinase_enrich.py`). Keep `SAP_FACTORIAL` as-is.
2. **`data_ingest.py`** — apply `.map(GENOTYPE_TO_SAP)` to `genotype` column before `to_csv`. Drop the now-redundant `GENOTYPE_TO_SAP[r['genotype']]` lookup inside `phospho_group_id` construction (column is already long form). Keep `_parse_animal_id` as the regex source of truth for the animal_id format; refactor `config.parse_animal_id` to call it (or move the regex to config and have data_ingest import it).
3. **`kinase_enrich.py`** — delete `GENOTYPE_CODING`, delete `CONTRAST_COEFS`. Import `SAP_FACTORIAL`, `CONTRAST_COEFS` from `config`. Replace the `GENOTYPE_CODING[g][f]` lookup with positional indexing into `SAP_FACTORIAL[g]` (which is a 3-tuple).
4. **`kinase_normalize.py`** — change `["WT", "APP", "T22", "T22/APP"]` to `["WTyp", "AppP", "Ttau", "ApTt"]` (or iterate `config.SAP_FACTORIAL`).
5. **`snrna_integration.py`** — delete local `CONTRAST_COEFS`, import from `config`.
6. **`decomposition/paths.py`** — delete `GENOTYPE_CODING` and `CONTRASTS`. Re-export from `config` (or have consumers import directly).
7. **`decomposition/factorial_ols.py`** — retarget imports.
8. **`decomposition/per_animal_extension.py`** — delete local `GENOTYPE_CODING`, retarget.
9. **`supplementary/deconvolution_feasibility.py`** — long-form literals.
10. **`integration/build_omics_trace.py`** — delete `GENO_DECODE`, use `config.parse_animal_id`.
11. **`incytr/export_decomposition_for_pair.py`** — delete `GENO_DECODE` and `ANIMAL_RE`, use `config.parse_animal_id`.
12. **`snrna_proportions.py`** — delete `GENO_DECODE` and `_decode_snrna_sample`; join on long-form `genotype` from `sample_mapping.csv` (now long-form).

## Verification

Before/after byte-equality check:

```bash
# 1. Run pipeline against pre-change state, snapshot key outputs
cp outputs/reports/kinase_attribution/mea_stoichiometry.csv /tmp/mea_before.csv
cp outputs/reports/kinase_attribution/site_level_ols.csv /tmp/site_before.csv
cp outputs/reports/snrna_integration/within_cohort_concordance.csv /tmp/conc_before.csv

# 2. Execute the unification commit

# 3. Re-run and diff
pixi run normalize && pixi run enrich
diff /tmp/mea_before.csv outputs/reports/kinase_attribution/mea_stoichiometry.csv
diff /tmp/site_before.csv outputs/reports/kinase_attribution/site_level_ols.csv

pixi run -- python alz/snrna_integration.py --concordance
diff /tmp/conc_before.csv outputs/reports/snrna_integration/within_cohort_concordance.csv

# 4. Also re-snapshot sample_mapping.csv — the genotype column should change
# from short→long form. This IS the schema change; expected.
```

NES, β, p-values must be byte-identical (or floating-point equal to ≥6 dp). The genotype column in `sample_mapping.csv` changes from short to long form — that's the deliberate schema change.

If anything else differs, stop and find the bug before committing.

## What's NOT in scope

- **Factorial design parametrization** for arbitrary future cohorts (different number of genotypes/timepoints). That's Phase 4 of the master plan. This phase only unifies the Song-cohort vocabulary.
- **Renaming `animal_id` strings** to embed long form. That changes raw-collaborator-derived strings and would require a much wider scan; the `parse_animal_id` helper makes it unnecessary.
- **Kedro pipeline restructuring.** Separate work.

## Risk

The change to `sample_mapping.csv:genotype` is on-disk schema. Any consumer not in the table above that reads that column and expects short form will silently get long form and either crash (KeyError on `GENOTYPE_CODING["WTyp"]`) or filter to zero rows. Mitigation: the grep that produced the inventory used short-form literal strings, which should catch every consumer; verification step (re-run + diff) catches anything missed.

## Success criteria

- One `GENOTYPE_CODING`-equivalent in the codebase (`config.SAP_FACTORIAL`).
- One `CONTRAST_COEFS` (in `config.py`).
- One `animal_id` parser (in `config.py`).
- Zero short-form literals anywhere outside `config.parse_animal_id` internals.
- Pipeline outputs byte-identical (verification step passes).
