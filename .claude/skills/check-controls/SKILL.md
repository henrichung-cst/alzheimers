---
name: check-controls
description: >
  Positive-control sanity check for the kinase-activity pipeline. Given a cohort
  (song / fivexfad / mukesh / tcells), looks up a curated set of control genes in
  the pipeline's existing output artifacts and renders an agent-judged verdict
  (agrees / off / borderline / not-built) on whether specificity and enrichment
  metrics behave as externally expected. Output is conversational — nothing is
  written to disk. Invoke as: /check-controls [cohort]
allowed-tools:
  - Read
  - Bash
  - Glob
---

# Positive-control signal check

## Curated control table

Maintain this table as the single source of control expectations. `expected` columns are sourced to external knowledge, never to our own pipeline output. Hypotheses are flagged — a disagreement against a hypothesis is noted as such, not treated as a pipeline bug.

To add a new robust control: append a row, fill `source of expectation` with a literature citation or atlas reference, and mark the expectation as **hypothesis** if it is not yet consensus.

| gene | kinase? | cohorts | expected cell-type home | expected AD direction | source of expectation | notes |
|---|---|---|---|---|---|---|
| PHKG1 | yes | song, fivexfad, mukesh | astrocyte *(hypothesis)* | TBD | TODO candidate; appears in MEA + unified_attribution | Phosphorylase kinase γ1; astrocyte hypothesis from WMB/Song specificity signal; direction not yet established externally |
| ATP9A | no (flippase) | mukesh | endothelial / vascular *(hypothesis)* | TBD | TODO candidate; atp9a_ad_export precedent | P4-ATPase lipid flippase; endothelial hypothesis from literature (vascular ATP9A expression); direction not established |
| APOE | no | song, fivexfad, mukesh | astrocyte / microglia | up in AD | established AD biology (PMID 2629601, GTEx/Allen atlas) | Canonical AD risk gene; strong astrocyte enrichment in mouse + human scRNA; upregulated in AD across all major study designs |

---

## Procedure

### 1. Resolve cohort

Read the invocation argument. If absent, infer from working context (e.g. a recently opened file path under `outputs/reports/kinase_attribution_human/` → mukesh; `kinase_attribution_5xfad/` → fivexfad; `kinase_attribution_tcells/` → tcells; `kinase_attribution/` with no cohort qualifier → song).

Valid keys: `song` · `fivexfad` · `mukesh` · `tcells`

Confirm the key and note the cohort's MEA output kind:
- song, fivexfad → OLS tables (`mea_stoichiometry.csv` carries NES/FDR per kinase per contrast)
- mukesh, tcells → NES/FDR matrices (`kinase_donor_nes.csv` etc.)

### 2. Select applicable controls

Filter the table above to rows whose `cohorts` field includes the target. Skip a control if the cohort is not listed, and report it as "not applicable for this cohort".

For `tcells`: no control biology is authored in this table yet — report the gap and move on.

### 3. Per control, branch on kinase?

**Kinase path:**

a. **Cell-type home** — check in order of availability:

   - Primary (mouse WMB): `outputs/reports/wmb_expression/wmb_kinase_expression.csv` → column `specificity_score`, top cell-type. **Likely absent** — if missing, report "run `pixi run wmb-export`" and skip to fallback.
   - Fallback (unified attribution): `outputs/reports/kinase_attribution/unified_attribution.csv` → columns `wmb_top_celltype`, `wmb_concentration_tier`, `song_top_celltype`. Filter to `kinase == <GENE>`, read the most-represented `wmb_top_celltype` and `song_top_celltype`.
   - Human specificity: `outputs/reports/kinase_attribution_human/celltype_specificity.csv` → columns `kinase`, `reference`, `celltype`, `specificity_score`, `rank`. Filter to `kinase == <GENE>`, read the rank-1 celltype per reference.
   - snRNA specificity (song): `outputs/reports/snrna_integration/song_expression_specificity.csv` → `top_cluster`, `specificity_score`, `tau`. **Likely absent** — if missing, fall back to `outputs/reports/snrna_integration/song_detection.csv` (`gene_symbol`, `cell_type`, `fraction_cells_expressing`) and take the top-fraction cell type. Note the fallback.

b. **Disease direction** — check MEA NES sign:

   - song: `outputs/reports/kinase_attribution/mea_stoichiometry.csv` → columns `kinase`, `NES`, `FDR`, `contrast`. Filter to `kinase == <GENE>`. Summarise the sign distribution across contrasts (how many positive/negative, which are FDR < 0.25).
   - fivexfad: per-region files under `outputs/reports/kinase_attribution_5xfad/` (e.g. `cortex_st_mea_stoichiometry.csv`). Same columns.
   - mukesh: `outputs/reports/kinase_attribution_human/perdonor/kinase_donor_nes.csv` → wide matrix (kinase × donor). Filter to row `kinase == <GENE>`, read per-donor NES values. Also check `outputs/reports/kinase_attribution_human/perdonor/kinase_donor_fdr.csv` for significance. Summarise sign and recurrence.

**Non-kinase path:**

a. **Cell-type home (mouse snRNA):** `outputs/reports/snrna_integration/song_detection.csv` → `gene_symbol`, `cell_type`, `fraction_cells_expressing`. Mouse gene symbols are lowercase-first (Apoe, Atp9a). Read the top-fraction cell types. For human gene symbols (APOE, ATP9A), try both casings.

b. **Cell-type home (human atlas):** `data/derived/aggregates/seaad/expression_by_supertype.csv` → gene-per-row, supertype-per-column matrix; column `gene`. Read the row for the target gene, identify the highest-expression supertype columns. For vascular/endothelial home, look at `Endo_*` columns. `data/derived/aggregates/hbca/expression_by_class.csv` → same structure, broader cell-class resolution.

c. **Disease direction (expression LFC):** `outputs/reports/kinase_attribution/sea_ad_supertype_lfc.csv` → columns `gene_symbol`, `stratum`, `supertype`, `subclass`, `supertype_lfc`. Note: this file contains only the 384 kinases in the attribution panel. For non-kinase controls (APOE, ATP9A), this file will NOT have the entry — say "LFC not computed for non-kinase genes in current pipeline artifacts; direction sourced from external literature only".

### 4. Show actual values

For each axis, print the raw number(s) — do not summarise to a verdict only. Example output pattern:

```
PHKG1 | song
  home (wmb_top_celltype): '30 Astro-Epen'  [wmb_concentration_tier=5]
  home (song snRNA): Astrocytes (fraction 0.44), fallback via song_detection.csv
  home (human, seaad): <celltype from celltype_specificity.csv>
  NES across contrasts: App_4mo=-1.69 (FDR<0.001), App_6mo=+1.88 (FDR<0.001), [...]
```

If an artifact is missing, write which file and which task produces it, then continue with what is available.

### 5. Judge each axis

Apply one of:
- `✓ as-expected` — actual result agrees with the external expectation
- `⚠ off` — actual result disagrees with a settled (non-hypothesis) expectation
- `~ borderline` — right family but weak tier, related subtype, or sign present but small; or expected direction is TBD and result is weak
- `– not built / N/A` — required artifact absent, or control not applicable to this cohort
- `? vs hypothesis` — expected is flagged *(hypothesis)*; actual agrees or disagrees; note which

This is agent judgment. Do NOT apply a numeric threshold. Reason about cell-type family (astrocyte vs astrocyte subtype is agreement; astrocyte vs microglia is borderline; astrocyte vs neuron is off). For NES sign: consider whether the sign is stable across contrasts and whether FDR < 0.25 in at least some.

### 6. Report to conversation

Render a compact table:

| control | axis | actual | verdict |
|---|---|---|---|
| PHKG1 | cell-type home (mouse) | '30 Astro-Epen' / Astrocytes | ✓ as-expected (? vs hypothesis) |
| PHKG1 | disease direction (song) | mixed: 4/9 neg FDR<0.25, 2/9 pos FDR<0.25 | ~ borderline / TBD |
| APOE | cell-type home (snRNA) | Astrocytes frac=0.61 | ✓ as-expected |
| APOE | disease direction | not computed for non-kinase genes | – not built |

Follow the table with a one-line overall read (e.g. "specificity signal looks credible for song; direction check needs wmb-export and human-specific LFC artifacts").

**Write nothing to disk.** If ALL required artifacts are missing for a given cohort, say which tasks to run first and stop.

---

## Artifact availability summary (as of last audit)

| artifact | path | present? | builds via |
|---|---|---|---|
| unified_attribution (song) | `outputs/reports/kinase_attribution/unified_attribution.csv` | yes | `pixi run attribution` |
| mea_stoichiometry (song) | `outputs/reports/kinase_attribution/mea_stoichiometry.csv` | yes | `pixi run mea` |
| mea_stoichiometry (5xfad) | `outputs/reports/kinase_attribution_5xfad/<region>_<track>_mea_stoichiometry.csv` | yes | `pixi run mea` |
| kinase_donor_nes (mukesh) | `outputs/reports/kinase_attribution_human/perdonor/kinase_donor_nes.csv` | yes | `pixi run attribution` |
| celltype_specificity (human) | `outputs/reports/kinase_attribution_human/celltype_specificity.csv` | yes | `pixi run attribution` |
| song snRNA detection | `outputs/reports/snrna_integration/song_detection.csv` | yes | `pixi run snrna` |
| seaad expression by supertype | `data/derived/aggregates/seaad/expression_by_supertype.csv` | yes | upstream ingest |
| hbca expression by class | `data/derived/aggregates/hbca/expression_by_class.csv` | yes | upstream ingest |
| wmb_kinase_expression | `outputs/reports/wmb_expression/wmb_kinase_expression.csv` | **no** | `pixi run wmb-export` |
| wmb_proteome_expression | `outputs/reports/wmb_expression/wmb_proteome_expression.csv` | **no** | `pixi run wmb-export` |
| song_expression_specificity | `outputs/reports/snrna_integration/song_expression_specificity.csv` | **no** | `pixi run snrna` |
| human_reference_expression | `outputs/reports/kinase_attribution/human_reference_expression/` | **no** | `pixi run attribution` |
| sea_ad_supertype_lfc | `outputs/reports/kinase_attribution/sea_ad_supertype_lfc.csv` | yes (kinases only) | `pixi run attribution` |
