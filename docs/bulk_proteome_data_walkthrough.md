# Bulk Proteome + PTM Number Walkthrough

This document is written for a reader who wants to audit the numerical path,
file by file. The central question is:

> If I point to one number in the raw phospho workbook, where does it go, what
> is done to it, why is that operation done, and where can I verify the result?

The short answer is:

```
raw phosphosite signal
  -> IRS-normalized phosphosite signal
  -> log2 phosphosite signal
  -> subtract log2 parent-protein signal
  -> stoichiometry value
  -> disease/timepoint contrast LFC
  -> kinase enrichment rank
  -> cell-type attribution tables
```

The longer answer is the rest of this file.

The sample and gene matching steps matter because they keep the rows and
columns aligned, but they are not the main focus here. The focus is what
happens to the measured signal values.

---

## 1. The raw numbers

### 1.1 Total proteome workbook

Raw file:

```
data/incytr_collections/song/primary/proteomics/
  song2024_tmttotal_protein_quant_merged_labeled (2).xlsx
```

Shape:

```
6,746 protein rows x 116 columns
```

Each row is a protein group. Each biological sample column contains a TMT
reporter signal-to-noise value, abbreviated here as SN.

Example total-proteome sample column:

```
plex2_130c_sn_mean
```

Read this as:

- `plex2`: TMT plex 2
- `130c`: TMT reporter channel 130C
- `sn_mean`: mean signal-to-noise value after peptide-to-protein aggregation

For one protein and one animal, this number is a relative abundance signal.
It is not an absolute concentration, not copies per cell, and not directly
comparable to samples in other plexes until cross-plex normalization is done.

### 1.2 Phospho sitequant workbook

Raw file:

```
data/incytr_collections/song/primary/proteomics/
  song_IMAC_sitequant_merged_labeled (2).xlsx
```

Shape:

```
about 16,114 phosphosite rows x TMT sample/ref columns
```

Each row is a localized phosphosite or phosphopeptide feature. Each biological
sample column contains the TMT reporter signal for the phosphorylated peptide.

Example phospho sample column:

```
p2_130c_sn_sum
```

This corresponds to the same biological animal as:

```
plex2_130c_sn_mean
```

The phospho workbook and proteome workbook use slightly different column
names, but the plex and channel identify the same TMT channel. The code maps:

```
plex2_130c_sn_mean  ->  p2_130c_sn_sum
```

### 1.3 Sample mapping workbook

Raw file:

```
data/incytr_collections/song/primary/proteomics/
  Sample_list_72mice (1).xlsx
```

Derived file:

```
outputs/reports/data_ingest/sample_mapping.csv
```

This is the column key. It says which animal, genotype, sex, and timepoint
each TMT channel represents.

For the running example in this document:

| Field | Value |
|---|---|
| Proteome column | `plex2_130c_sn_mean` |
| Phospho column | `p2_130c_sn_sum` |
| Plex | 2 |
| Channel | 130C |
| Animal | `31_D224(L)_F_2mo_APP` |
| Mouse ID | `D224` |
| Sex | F |
| Timepoint | 2mo |
| Genotype | APP |

The sample map is not a biological transformation. It does not change the
signal values. It only ensures that a phospho value and proteome value from
the same TMT channel are paired correctly.

---

## 2. Running example: one site, one sample

We will follow one phosphosite in one sample through the pipeline.

Phosphosite:

| Field | Value |
|---|---|
| `site_id` | 2488 |
| `gene_symbol` | `1110004F10Rik` |
| `motif` | `GVKRSASPDDDLG` |

Sample:

| Field | Value |
|---|---|
| Proteome output column | `plex2_130c_sn_mean` |
| Phospho raw column | `p2_130c_sn_sum` |
| Plex | 2 |
| Channel | 130C |

The raw phospho value is found in:

```
song_IMAC_sitequant_merged_labeled (2).xlsx
  row: site_id = 2488
  column: p2_130c_sn_sum
  value: 832.7
```

The raw parent-protein value is found in:

```
song2024_tmttotal_protein_quant_merged_labeled (2).xlsx
  row: Gene Symbol = 1110004F10Rik
  column: plex2_130c_sn_mean
  value: 89.6999
```

At this point these two numbers are still raw TMT signals. They have not been
normalized across plexes, logged, or divided.

---

## 3. Why IRS normalization is the first numerical operation

### 3.1 The problem

The 72 animals are split across six TMT plexes. A raw value in plex 2 and a raw
value in plex 5 do not live on the same scale automatically. Differences can
come from injection amount, ionization, instrument response, or other
plex-specific effects.

Each plex includes a shared reference pool in channel 126. That reference pool
is intended to be the same biological material in every plex.

If the same reference pool is measured differently across plexes, the
difference is treated as a plex measurement effect. IRS normalization uses the
reference pool to remove that effect.

### 3.2 The IRS equation

For one row, one sample, and one plex:

```
+---------------------------------------------------------------+
| global_ref = mean(reference channel across the six plexes)     |
|                                                               |
| normalized_sample = raw_sample x (global_ref / plex_ref)       |
+---------------------------------------------------------------+
```

Where:

- `raw_sample` is the raw SN value for the animal.
- `plex_ref` is the channel 126 reference-pool value in that animal's plex.
- `global_ref` is the mean of the six reference-pool values for that same row.

This is done row by row. For phospho data, each phosphosite gets its own IRS
factor. For proteome data, each protein gets its own IRS factor.

The operation is not a single global scaling factor for all proteins or all
sites. That matters because different peptides/proteins can have different
plex behavior.

---

## 4. IRS normalization for the running phosphosite

Raw file:

```
song_IMAC_sitequant_merged_labeled (2).xlsx
```

Row:

```
site_id = 2488
gene_symbol = 1110004F10Rik
motif = GVKRSASPDDDLG
```

Raw animal value:

```
p2_130c_sn_sum = 832.7
```

Reference-pool values for the same phosphosite:

| Reference column | Value |
|---|---:|
| `p1_126_sn_sum` | 647.34 |
| `p2_126_sn_sum` | 924.23 |
| `p3_126_sn_sum` | 912.44 |
| `p4_126_sn_sum` | 866.39 |
| `p5_126_sn_sum` | 1207.66 |
| `p6_126_sn_sum` | 800.59 |

The global reference for this phosphosite is the mean of those six values:

```
global_ref_phospho
  = (647.34 + 924.23 + 912.44 + 866.39 + 1207.66 + 800.59) / 6
  = 893.1083333333335
```

This animal is in plex 2, so the plex-specific reference is:

```
plex_ref_phospho = p2_126_sn_sum = 924.23
```

The plex 2 scaling factor for this phosphosite is:

```
global_ref_phospho / plex_ref_phospho
  = 893.1083333333335 / 924.23
  = 0.9663269243947215
```

Now apply that factor to the raw animal value:

```
+-------------------------------------------------------------+
| normalized_phospho                                         |
|   = raw_phospho x (global_ref_phospho / plex_ref_phospho)  |
|   = 832.7 x 0.9663269243947215                             |
|   = 804.6604299434846                                      |
+-------------------------------------------------------------+
```

Where to verify this number:

```
outputs/reports/kinase_attribution/raw_phospho_normalized.csv
  row: site_id = 2488
  column: plex2_130c_sn_mean
  value: 804.6604299434846
```

Notice that the output file renames phospho sample columns to match the
proteome naming convention. That is why the normalized phospho value appears
under `plex2_130c_sn_mean`, not `p2_130c_sn_sum`.

The number is still in linear SN-like units. It has only been cross-plex
normalized.

---

## 5. IRS normalization for the matching parent protein

The phosphosite belongs to gene:

```
1110004F10Rik
```

The matching total-proteome row is:

```
song2024_tmttotal_protein_quant_merged_labeled (2).xlsx
  Gene Symbol = 1110004F10Rik
  protein_id = ENSMUSP00000032899.6
```

Raw animal value:

```
plex2_130c_sn_mean = 89.6999
```

Reference-pool values for the same protein:

| Reference column | Value |
|---|---:|
| `plex1_126_sn_mean` | 103.893 |
| `plex2_126_sn_mean` | 96.2143 |
| `plex3_126_sn_mean` | 66.6147 |
| `plex4_126_sn_mean` | 120.99 |
| `plex5_126_sn_mean` | 251.663 |
| `plex6_126_sn_mean` | 109.796 |

The global reference for this protein is:

```
global_ref_protein
  = (103.893 + 96.2143 + 66.6147 + 120.99 + 251.663 + 109.796) / 6
  = 124.86183333333334
```

This animal is again in plex 2:

```
plex_ref_protein = plex2_126_sn_mean = 96.2143
```

The plex 2 scaling factor for this protein is:

```
global_ref_protein / plex_ref_protein
  = 124.86183333333334 / 96.2143
  = 1.2977471470803543
```

Apply that factor to the raw animal value:

```
+-------------------------------------------------------------+
| normalized_protein                                         |
|   = raw_protein x (global_ref_protein / plex_ref_protein)  |
|   = 89.6999 x 1.2977471470803543                           |
|   = 116.40778931839307                                     |
+-------------------------------------------------------------+
```

The normalized proteome matrix is used in memory by
`code/kinase_attribution.py`. It is not written as its own CSV in the current
pipeline. We can still verify it indirectly because it is exactly the
denominator used to compute stoichiometry in the next step.

---

## 6. Why log2 is applied

After IRS normalization, both the phospho value and parent-protein value are
positive linear intensities.

For the running example:

| Quantity | Value |
|---|---:|
| IRS-normalized phosphosite | 804.6604299434846 |
| IRS-normalized parent protein | 116.40778931839307 |

The biological question is not simply "how much phosphopeptide was detected?"
It is closer to:

> How much phosphorylated peptide was detected per unit of parent protein?

In linear form, that is a ratio:

```
phospho / protein
```

The pipeline computes this ratio on the log2 scale:

```
+----------------------------------------------------+
| log2(phospho / protein)                            |
|   = log2(phospho) - log2(protein)                  |
+----------------------------------------------------+
```

There are three reasons for this.

First, log2 turns division into subtraction. This makes the correction
compatible with later linear modeling.

Second, log2 units are fold-change units. A change of +1 means a doubling.
A change of -1 means a halving.

Third, TMT intensity values are usually right-skewed and have larger variance
at larger signal. Log transformation makes the data more symmetric and makes
variance more stable across the signal range.

---

## 7. Stoichiometry for the running site/sample

Take log2 of the IRS-normalized phosphosite value:

```
log2(804.6604299434846) = 9.652236278106182
```

Take log2 of the IRS-normalized parent-protein value:

```
log2(116.40778931839307) = 6.863043787840785
```

Subtract:

```
+----------------------------------------------------+
| stoichiometry                                      |
|   = log2(phospho_IRS) - log2(protein_IRS)          |
|   = 9.652236278106182 - 6.863043787840785          |
|   = 2.789192490265397                              |
+----------------------------------------------------+
```

Where to verify this number:

```
outputs/reports/kinase_attribution/stoichiometry_matrix.csv
  row: site_id = 2488
  column: plex2_130c_sn_mean
  value: 2.789192490265397
```

This number is the per-animal stoichiometry proxy for one phosphosite in one
animal.

It should not be interpreted as an absolute site occupancy percentage. The
phospho and total-proteome experiments have different peptide chemistry,
enrichment, and ionization behavior. The absolute scale is arbitrary.

What is interpretable is comparison of the same site across animals:

```
+-------------------------------------------------------------+
| change in stoichiometry                                     |
|   = change in log2(phospho) - change in log2(parent protein) |
+-------------------------------------------------------------+
```

That quantity asks whether phosphorylation changed more than expected from
parent-protein abundance alone.

---

## 8. What stoichiometry corrects

Suppose a phosphosite doubles in raw phospho signal in APP mice:

```
raw phospho fold change = 2x
log2 phospho fold change = +1
```

If the parent protein also doubles:

```
protein fold change = 2x
log2 protein fold change = +1
```

Then the stoichiometry change is:

```
+-------------------------------+
| +1 - +1 = 0                   |
+-------------------------------+
```

That means the site did not become more phosphorylated per unit of protein.
There was simply more substrate.

If the phosphosite doubles but the parent protein stays constant:

```
+-------------------------------+
| +1 - 0 = +1                   |
+-------------------------------+
```

That means the phosphorylation signal increased relative to substrate
abundance. That is the kind of signal one would want kinase enrichment to see.

This is why MEA is run on stoichiometry contrast values, not directly on raw
phospho abundance.

---

## 9. Files after normalization and stoichiometry

### 9.1 `raw_phospho_normalized.csv`

Path:

```
outputs/reports/kinase_attribution/raw_phospho_normalized.csv
```

What each numerical cell means:

```
IRS-normalized phosphosite intensity, still in linear space
```

For the running example:

```
site_id = 2488
column = plex2_130c_sn_mean
value = 804.6604299434846
```

This file is used later for the "raw phospho" comparison model. Before OLS,
these values are log2-transformed.

### 9.2 `stoichiometry_matrix.csv`

Path:

```
outputs/reports/kinase_attribution/stoichiometry_matrix.csv
```

What each numerical cell means:

```
log2(IRS-normalized phosphosite intensity)
  - log2(IRS-normalized parent-protein intensity)
```

For the running example:

```
site_id = 2488
column = plex2_130c_sn_mean
value = 2.789192490265397
```

Rows whose parent protein was not detected in the total proteome are marked as
unmatched and receive missing stoichiometry values. For example, a site can
have raw phospho values but still have no stoichiometry if its parent protein
is absent from the total-proteome gene list.

### 9.3 `normalization_summary.json`

Path:

```
outputs/reports/kinase_attribution/normalization_summary.json
```

This is the receipt for the normalization run. It records, among other things:

| Quantity | Value |
|---|---|
| Sites total | 16,114 |
| Sites matched to parent proteins | 14,772 |
| Percent matched | 91.7% |
| Valid stoichiometry values | 628,188 |
| Percent valid stoichiometry | 54.1% |
| Normalization method | IRS |

It also records total-proteome plex medians before and after IRS:

| Plex | Before | After |
|---|---:|---:|
| 1 | 70.37 | 71.75 |
| 2 | 74.20 | 71.41 |
| 3 | 64.85 | 75.06 |
| 4 | 65.38 | 71.79 |
| 5 | 72.78 | 72.75 |
| 6 | 73.97 | 72.10 |

These medians are not used to compute IRS. They are a diagnostic summary.
They show that the plex-level distributions became more comparable after the
reference-channel scaling.

The PCA diagnostic also shifts:

| Diagnostic | Before IRS | After IRS |
|---|---:|---:|
| Total-proteome PC1 variance | 21.11% | 17.07% |
| Total-proteome PC2 variance | 15.91% | 12.01% |

This is another diagnostic, not a direct calculation step for downstream
values.

---

## 10. OLS: from per-animal stoichiometry to disease/timepoint LFC

The stoichiometry matrix still has one value per site per animal. The next
goal is to estimate disease effects at each timepoint.

Output file:

```
outputs/reports/kinase_attribution/site_level_ols.csv
```

For each site, the model produces contrast columns such as:

```
stoich_lfc_App_2mo
stoich_lfc_App_4mo
stoich_lfc_App_6mo
stoich_lfc_Tau_2mo
stoich_lfc_Tau_4mo
stoich_lfc_Tau_6mo
stoich_lfc_ApTt_2mo
stoich_lfc_ApTt_4mo
stoich_lfc_ApTt_6mo
```

Each one is a log2 fold-change estimate for stoichiometry, relative to WT at
the same modeled timepoint.

### 10.1 What numbers go into the OLS model

For stoichiometry OLS, the input response values are cells from:

```
stoichiometry_matrix.csv
```

For raw phospho OLS, the input response values are:

```
log2(raw_phospho_normalized.csv values)
```

The current default analysis mode is:

```
ANALYSIS_MODE = males_only
```

For this run, the sample filter was:

```
72 total samples -> 33 samples used
```

The reduction comes from:

- outlier exclusions
- retaining male samples only

Normalization was already done on all 72 samples. This filtering happens only
at the OLS stage.

### 10.2 The design matrix in plain language

For the default males-only run, the OLS model has 10 parameters:

```
const
App
Tau
Int
time_4mo
time_6mo
App_x_time4
App_x_time6
Tau_x_time4
Tau_x_time6
```

The model is not changing the stoichiometry values. It is fitting a surface
through them so that disease effects can be estimated while sharing variance
across the factorial design.

In simplified form:

```
+-------------------------------------------------------------+
| observed stoichiometry                                      |
|   = baseline WT 2mo                                         |
|   + APP effect                                              |
|   + Tau effect                                              |
|   + APP/Tau interaction                                     |
|   + time effects                                            |
|   + disease-by-time adjustments                             |
|   + residual noise                                          |
+-------------------------------------------------------------+
```

### 10.3 Running example OLS coefficients

For site `2488`, the fitted stoichiometry coefficients were:

| Coefficient | Value |
|---|---:|
| `const` | 2.7292428971739917 |
| `App` | -0.01988331404982523 |
| `Tau` | -0.34118259120801053 |
| `Int` | -0.05678985290435811 |
| `time_4mo` | -0.1129008199286347 |
| `time_6mo` | -0.032721840343156644 |
| `App_x_time4` | -0.23202572149279663 |
| `App_x_time6` | 0.19986209464182364 |
| `Tau_x_time4` | 0.2976188283640912 |
| `Tau_x_time6` | 0.16148630457848728 |

These coefficients are intermediate bookkeeping values. They are not written
to `site_level_ols.csv`. They are combined into biologically named contrasts.

For example:

```
+--------------------------------------------+
| App_4mo stoichiometry LFC                  |
|   = App + App_x_time4                      |
|   = -0.01988331404982523                   |
|     + -0.23202572149279663                 |
|   = -0.25190903554262184                   |
+--------------------------------------------+
```

Where to verify the rounded result:

```
outputs/reports/kinase_attribution/site_level_ols.csv
  row: site_id = 2488
  column: stoich_lfc_App_4mo
  value: -0.251909035542622
```

Interpretation:

```
For site 2488, the modeled APP effect at 4 months is a -0.252 log2 change
in phosphorylation per unit parent protein.
```

On a linear fold-change scale:

```
2^(-0.251909) = about 0.84x
```

That is about a 16% lower stoichiometry estimate for this site in the modeled
APP 4mo contrast. In this example the p-value and FDR do not support a
confident site-level effect:

| Column | Value |
|---|---:|
| `n_obs_stoich` | 33 |
| `stoich_lfc_App_4mo` | -0.251909035542622 |
| `stoich_pval_App_4mo` | 0.3230876951141539 |
| `stoich_fdr_App_4mo` | 0.9994066417102875 |
| `raw_lfc_App_4mo` | -0.3276992446032432 |
| `raw_fdr_App_4mo` | 0.9965652561022575 |

### 10.4 How the p-value is attached to the contrast

The model also estimates uncertainty. For the same site and contrast:

| Quantity | Value |
|---|---:|
| Observations | 33 |
| Model parameters | 10 |
| Degrees of freedom | 23 |
| Contrast estimate | -0.25190903554262184 |
| Contrast standard error | 0.24945929304301376 |
| t statistic | -1.0098202094206434 |
| p-value | 0.32308769511415425 |

The simplified equation is:

```
+--------------------------------------+
| t = contrast_LFC / contrast_SE       |
+--------------------------------------+
```

The p-value asks whether that estimate is distinguishable from zero relative
to the residual variability for that site. The FDR then corrects p-values
across all tested sites for the same contrast.

---

## 11. Why OLS instead of simple group means

A skeptical reader might ask why we do not simply average APP 4mo animals,
average WT 4mo animals, and subtract.

That would be easier to inspect, but it would use very small groups and would
not share information across the factorial design. In this dataset, many
cells have about three animals before filtering, and some have fewer after
outlier/sex filtering.

OLS lets the analysis estimate:

- the APP effect
- the Tau effect
- the combined APP/Tau effect
- the 4mo and 6mo time effects
- the disease-by-time deviations

using one coherent model per site.

The contrast still has a direct meaning. For example:

```
App_4mo = App + App_x_time4
```

The model is a way to estimate that contrast with a standard error, not a
different biological endpoint.

---

## 12. From site-level LFCs to MEA kinase enrichment

Input file:

```
outputs/reports/kinase_attribution/site_level_ols.csv
```

Output file:

```
outputs/reports/kinase_attribution/mea_stoichiometry.csv
```

For each contrast, MEA starts with one value per phosphosite:

```
stoich_lfc_<contrast>
```

For example:

```
stoich_lfc_App_4mo
```

This is a ranked list of sites. Sites with positive values increased in
stoichiometry for that contrast. Sites with negative values decreased.

Before kinase enrichment, the pipeline applies two numerical preprocessing
steps to the LFC vector.

### 12.1 Median-centering before MEA

For each contrast:

```
+-------------------------------------------+
| centered_site_LFC = site_LFC - median_LFC |
+-------------------------------------------+
```

The reason is that MEA asks whether a kinase's substrates are specifically
concentrated near the top or bottom of the ranked list. If the entire
phosphoproteome shifts up or down, the global shift can obscure kinase-specific
patterns.

For `App_4mo`, the recorded median shift was:

```
median_shift = -0.039825455760814185
```

So a site with:

```
stoich_lfc_App_4mo = -0.251909035542622
```

would have a centered value:

```
-0.251909035542622 - (-0.039825455760814185)
  = -0.21208357978180782
```

Where to verify the median shift:

```
outputs/reports/kinase_attribution/mea_global_shift.csv
  row: contrast = App_4mo
  column: median_shift
  value: -0.039825455760814185
```

### 12.2 Winsorization before MEA

After median-centering, the pipeline clips the most extreme 1% and 99% tails:

```
+---------------------------------------------------------+
| values below the 1st percentile are set to that bound   |
| values above the 99th percentile are set to that bound  |
+---------------------------------------------------------+
```

This does not remove sites. It prevents a small number of very extreme sites
from dominating the enrichment statistic.

Any clipped sites are logged in:

```
outputs/reports/kinase_attribution/winsorized_sites.csv
```

The MEA-ranked value for each site is therefore:

```
site OLS LFC
  -> minus contrast-wide median
  -> clipped only if it is outside the 1st/99th percentile bounds
```

### 12.3 What `mea_stoichiometry.csv` contains

Each row is a kinase-by-contrast enrichment test.

Important columns:

| Column | Meaning |
|---|---|
| `kinase` | Kinase being tested |
| `contrast` | Disease/timepoint contrast |
| `NES` | Normalized enrichment score |
| `FDR` | Multiple-testing corrected enrichment value |
| `Leading substrates` | Motifs/sites that drove the enrichment |
| `Subs fraction` | Number of matched kinase substrates used |

Numerically, MEA asks:

```
Are predicted substrates of this kinase concentrated near the high end
or low end of the ranked site-LFC list?
```

Positive NES means the kinase's substrates tend to sit toward the positive
end of the contrast ranking. Negative NES means they tend to sit toward the
negative end.

---

## 13. From MEA to cell-type attribution

Input:

```
outputs/reports/kinase_attribution/mea_stoichiometry.csv
```

Output:

```
outputs/reports/kinase_attribution/unified_attribution.csv
```

At this stage, the numerical kinase activity signal is already represented by
MEA columns such as `NES` and `FDR`. The attribution step does not recompute
the phosphosite stoichiometry. It adds cell-type evidence using:

- kinase expression or specificity in reference cell-type data
- cross-species concordance where available
- same-cohort evidence where available

Each row is a kinase, contrast, and cell type combination.

Important numerical columns include:

| Column | Meaning |
|---|---|
| `NES` | kinase enrichment direction/strength from MEA |
| `FDR` | kinase enrichment FDR |
| `wmb_specificity` | kinase expression specificity in WMB reference |
| `sea_ad_lfc` | human SEA-AD transcriptomic effect, when available |
| `song_specificity` / `song_lfc` | same-cohort supporting evidence, when available |
| `combined_score` | weighted summary score |
| `combined_confidence` | high/moderate/low confidence tier |

This stage changes the question from:

```
Which kinases show enriched substrate behavior?
```

to:

```
Which cell types are plausible sources or contexts for those kinase signals?
```

---

## 14. Final hypothesis tables

Final output directory:

```
outputs/reports/attribution_recovery/
```

Important files:

| File | What the numbers summarize |
|---|---|
| `kinase_activity_matrix.csv` | Wide kinase x contrast view of NES/FDR |
| `celltype_evidence_table.csv` | Cell-type evidence after specificity/concordance filtering |
| `kinase_hypothesis_table.csv` | Primary summary table with top cell types and trajectory labels |

These are synthesis tables. They should be traced backward to:

```
kinase_hypothesis_table.csv
  -> unified_attribution.csv
  -> mea_stoichiometry.csv
  -> site_level_ols.csv
  -> stoichiometry_matrix.csv
  -> raw_phospho_normalized.csv
  -> raw phospho and total-proteome Excel workbooks
```

---

## 15. File-by-file audit checklist

### Step 1: Confirm the sample identity

Open:

```
outputs/reports/data_ingest/sample_mapping.csv
```

Check:

```
column_name = plex2_130c_sn_mean
```

For the running example, this should identify:

```
animal_id = 31_D224(L)_F_2mo_APP
mouse_id = D224
sex = F
timepoint = 2mo
genotype = APP
```

This proves which animal the column represents.

### Step 2: Confirm the raw phospho value

Open:

```
song_IMAC_sitequant_merged_labeled (2).xlsx
```

Find:

```
site_id = 2488
column = p2_130c_sn_sum
```

Expected value:

```
832.7
```

Also read the six reference values:

```
p1_126_sn_sum = 647.34
p2_126_sn_sum = 924.23
p3_126_sn_sum = 912.44
p4_126_sn_sum = 866.39
p5_126_sn_sum = 1207.66
p6_126_sn_sum = 800.59
```

Compute:

```
832.7 x mean(reference values) / p2_126_sn_sum
  = 804.6604299434846
```

Verify:

```
raw_phospho_normalized.csv
  site_id = 2488
  plex2_130c_sn_mean = 804.6604299434846
```

### Step 3: Confirm the raw parent-protein value

Open:

```
song2024_tmttotal_protein_quant_merged_labeled (2).xlsx
```

Find:

```
Gene Symbol = 1110004F10Rik
column = plex2_130c_sn_mean
```

Expected value:

```
89.6999
```

Also read the six reference values:

```
plex1_126_sn_mean = 103.893
plex2_126_sn_mean = 96.2143
plex3_126_sn_mean = 66.6147
plex4_126_sn_mean = 120.99
plex5_126_sn_mean = 251.663
plex6_126_sn_mean = 109.796
```

Compute:

```
89.6999 x mean(reference values) / plex2_126_sn_mean
  = 116.40778931839307
```

This normalized protein value is not written to a standalone file, but it is
used immediately in the stoichiometry calculation.

### Step 4: Confirm stoichiometry

Compute:

```
log2(804.6604299434846) - log2(116.40778931839307)
  = 2.789192490265397
```

Verify:

```
stoichiometry_matrix.csv
  site_id = 2488
  plex2_130c_sn_mean = 2.789192490265397
```

### Step 5: Confirm the OLS contrast

Open:

```
site_level_ols.csv
```

Find:

```
site_id = 2488
```

Expected relevant values:

```
n_obs_stoich = 33
stoich_lfc_App_4mo = -0.251909035542622
stoich_pval_App_4mo = 0.3230876951141539
stoich_fdr_App_4mo = 0.9994066417102875
raw_lfc_App_4mo = -0.3276992446032432
raw_fdr_App_4mo = 0.9965652561022575
```

The contrast was computed from the fitted site-level coefficients:

```
App_4mo = App + App_x_time4
        = -0.01988331404982523 + -0.23202572149279663
        = -0.25190903554262184
```

### Step 6: Confirm the MEA ranking transformation

Open:

```
mea_global_shift.csv
```

Find:

```
contrast = App_4mo
median_shift = -0.039825455760814185
```

For the example site:

```
centered App_4mo value
  = -0.251909035542622 - (-0.039825455760814185)
  = -0.21208357978180782
```

Then check `winsorized_sites.csv` if you need to know whether the site was
clipped before MEA ranking.

### Step 7: Trace kinase-level output

Open:

```
mea_stoichiometry.csv
```

Pick a kinase and contrast. The row reports whether that kinase's predicted
substrates are enriched toward the positive or negative end of the ranked
site list.

Then trace the kinase/contrast to:

```
unified_attribution.csv
kinase_hypothesis_table.csv
```

At that point the values are summaries of kinase enrichment and cell-type
support, not direct transformations of a single phosphosite cell.

---

## 16. What each operation does and does not claim

| Operation | What it does | What it does not claim |
|---|---|---|
| IRS normalization | Removes plex-specific scaling using the shared reference pool | Does not make values absolute concentrations |
| log2 transform | Converts ratios to subtractions and stabilizes variance | Does not create biological signal by itself |
| stoichiometry subtraction | Removes parent-protein abundance contribution from phospho signal | Does not measure true absolute occupancy |
| OLS | Estimates disease/timepoint contrasts with uncertainty | Does not prove causality |
| FDR correction | Controls multiple testing across sites or kinases | Does not make a weak effect biologically important |
| MEA | Tests whether kinase substrates are enriched in ranked site changes | Does not directly measure kinase enzymatic activity |
| Attribution | Adds cell-type plausibility evidence | Does not prove the signal originates only from that cell type |

---

## 17. Minimal mental model

If you only remember one chain, remember this:

```
Raw phosphosite SN:
  832.7

IRS-normalized phosphosite:
  832.7 x (893.1083 / 924.23)
  = 804.6604

Raw parent-protein SN:
  89.6999

IRS-normalized parent protein:
  89.6999 x (124.8618 / 96.2143)
  = 116.4078

Stoichiometry:
  log2(804.6604) - log2(116.4078)
  = 2.78919

Site-level disease contrast:
  App_4mo = App coefficient + App-by-4mo coefficient
  = -0.25191

MEA input for that contrast:
  site LFC - contrast median shift
  = -0.25191 - (-0.03983)
  = -0.21208 before any winsorization
```

That is the core numerical path from a raw phosphosite measurement to the
ranked value used for kinase enrichment.
