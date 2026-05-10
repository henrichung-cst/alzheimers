# Cohort-concordance + presence calibration audit

**Source:** `outputs/reports/deconvolution/per_animal/kinase_enrichment_46clusters.csv`

- Bulk-significant rows (FDR<0.25) with finite snRNA LFC: **2,303**

## 1. Cohort sign-concordance (binomial vs 0.5 null)

- Strata (wmb_class × contrast) with ≥5 rows: **43**
- Strata with frac_match > 0.5: **23 / 43**
- Strata passing cohort_fdr < 0.05: **0**
- Strata passing cohort_fdr < 0.10: **0**
- Strata passing cohort_fdr < 0.25: **3**
- Median frac_match across strata: **0.523**

### Per-stratum table (top 20 by frac_match)

| wmb_class | contrast | n_total | n_match | frac_match | cohort_pval | cohort_fdr | cohort_concordant |
|---|---|---|---|---|---|---|---|
| 02 NP-CT-L6b Glut | ApTt_4mo | 5 | 5 | 1.000 | 0.031 | 0.336 | False |
| 11 CNU-HYa GABA | Tau_4mo | 7 | 7 | 1.000 | 0.008 | 0.194 | True |
| 30 Astro-Epen | Tau_2mo | 6 | 6 | 1.000 | 0.016 | 0.224 | True |
| 33 Vascular | App_6mo | 7 | 6 | 0.857 | 0.062 | 0.519 | False |
| 30 Astro-Epen | Tau_4mo | 6 | 5 | 0.833 | 0.109 | 0.519 | False |
| 07 CTX-MGE GABA | App_2mo | 12 | 9 | 0.750 | 0.073 | 0.519 | False |
| 30 Astro-Epen | Tau_6mo | 10 | 7 | 0.700 | 0.172 | 0.672 | False |
| 03 OB-CR Glut | App_4mo | 18 | 12 | 0.667 | 0.119 | 0.519 | False |
| 07 CTX-MGE GABA | App_6mo | 6 | 4 | 0.667 | 0.344 | 0.935 | False |
| 09 CNU-LGE GABA | Tau_4mo | 9 | 6 | 0.667 | 0.254 | 0.910 | False |
| 01 IT-ET Glut | ApTt_2mo | 39 | 24 | 0.615 | 0.100 | 0.519 | False |
| 09 CNU-LGE GABA | ApTt_4mo | 13 | 8 | 0.615 | 0.291 | 0.935 | False |
| 30 Astro-Epen | ApTt_2mo | 5 | 3 | 0.600 | 0.500 | 0.935 | False |
| 11 CNU-HYa GABA | ApTt_2mo | 7 | 4 | 0.571 | 0.500 | 0.935 | False |
| 33 Vascular | App_2mo | 7 | 4 | 0.571 | 0.500 | 0.935 | False |
| 03 OB-CR Glut | ApTt_4mo | 7 | 4 | 0.571 | 0.500 | 0.935 | False |
| 01 IT-ET Glut | ApTt_4mo | 347 | 196 | 0.565 | 0.009 | 0.194 | True |
| 07 CTX-MGE GABA | ApTt_2mo | 22 | 12 | 0.545 | 0.416 | 0.935 | False |
| 30 Astro-Epen | ApTt_4mo | 11 | 6 | 0.545 | 0.500 | 0.935 | False |
| 06 CTX-CGE GABA | ApTt_2mo | 13 | 7 | 0.538 | 0.500 | 0.935 | False |

## 2. Expression presence (log2(CPM+1) in snRNA pseudobulk)

- Sig-rows with a (wmb_class, gene) match in specificity: **2,303**
- Sig-rows missing from specificity (treated as NotExpressed): **0**

### Quantiles of nonzero mean_expression (kinase × wmb_class)

| q | mean_expression |
|---|---|
| 0.05 | 0.950 |
| 0.10 | 1.662 |
| 0.25 | 3.616 |
| 0.50 | 6.073 |
| 0.75 | 7.482 |

### NotExpressed counts at candidate floors

| floor_log2cpm | n_NotExpressed | frac_of_sig |
|---|---|---|
| 0.050 | 0.000 | 0.000 |
| 0.100 | 0.000 | 0.000 |
| 0.250 | 0.000 | 0.000 |
| 0.500 | 39.000 | 0.017 |
| 1.000 | 119.000 | 0.052 |

## Recommended thresholds

- `COHORT_FDR_THRESH = 0.25` — Even FDR<0.10 fails. Use 0.25 as the cohort gate; treat 'Supported' tier with caution.
- `EXPR_PRESENCE_FLOOR = 0.10` (log2(CPM+1); rows below this are NotExpressed).

Review the floor table above; pick the most conservative value that retains a meaningful Supported tier.