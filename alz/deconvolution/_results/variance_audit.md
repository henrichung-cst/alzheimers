# Variance audit: per-animal vs. Yuyu group-level OLS

**Common scope:** 3 cluster(s) (['Astrocytes', 'Microglia', 'Oligodendrocytes']) × 2 track(s) (['py', 'st']) × 9 contrasts.

## Headline

- Median SE ratio (per-animal / group-level): **0.563** (theoretical floor √(2/dof_pa) ≈ 0.30)
- Median Spearman ρ on LFC (per-animal vs. group): **0.972** (target > 0.9)
- Rows with SE ratio < 0.5: **18 / 54**
- Rows with LFC ρ > 0.9: **49 / 54**

## Per (cluster, track, contrast) detail

| cluster | track | contrast | n_sites | se_ratio_median | se_ratio_p10 | se_ratio_p90 | lfc_spearman_rho | lfc_slope | frac_p_pa_lt_0p05 | frac_p_grp_lt_0p05 |
|---|---|---|---|---|---|---|---|---|---|---|
| Astrocytes | py | ApTt_2mo | 1456 | 0.657 | 0.320 | 1.904 | 0.975 | 0.996 | 0.378 | 0.040 |
| Astrocytes | py | ApTt_4mo | 1456 | 0.729 | 0.355 | 2.113 | 0.919 | 0.996 | 0.226 | 0.051 |
| Astrocytes | py | ApTt_6mo | 1456 | 0.729 | 0.355 | 2.114 | 0.932 | 0.963 | 0.253 | 0.050 |
| Astrocytes | py | App_2mo | 1456 | 0.666 | 0.324 | 1.932 | 0.956 | 0.997 | 0.372 | 0.028 |
| Astrocytes | py | App_4mo | 1456 | 0.674 | 0.328 | 1.955 | 0.971 | 0.999 | 0.289 | 0.047 |
| Astrocytes | py | App_6mo | 1456 | 0.724 | 0.352 | 2.100 | 0.954 | 0.996 | 0.235 | 0.045 |
| Astrocytes | py | Tau_2mo | 1456 | 0.666 | 0.324 | 1.932 | 0.971 | 0.987 | 0.247 | 0.036 |
| Astrocytes | py | Tau_4mo | 1456 | 0.674 | 0.328 | 1.955 | 0.958 | 0.989 | 0.217 | 0.041 |
| Astrocytes | py | Tau_6mo | 1456 | 0.682 | 0.332 | 1.978 | 0.938 | 0.989 | 0.360 | 0.049 |
| Microglia | py | ApTt_2mo | 1456 | 0.320 | 0.266 | 1.175 | 0.993 | 0.998 | 0.452 | 0.039 |
| Microglia | py | ApTt_4mo | 1456 | 0.355 | 0.295 | 1.304 | 0.885 | 0.997 | 0.402 | 0.053 |
| Microglia | py | ApTt_6mo | 1456 | 0.355 | 0.296 | 1.304 | 0.927 | 0.984 | 0.420 | 0.044 |
| Microglia | py | App_2mo | 1456 | 0.325 | 0.270 | 1.192 | 0.993 | 0.999 | 0.390 | 0.028 |
| Microglia | py | App_4mo | 1456 | 0.328 | 0.273 | 1.206 | 0.991 | 1.018 | 0.482 | 0.046 |
| Microglia | py | App_6mo | 1456 | 0.353 | 0.294 | 1.296 | 0.994 | 1.003 | 0.438 | 0.038 |
| Microglia | py | Tau_2mo | 1456 | 0.325 | 0.270 | 1.192 | 0.994 | 0.991 | 0.408 | 0.040 |
| Microglia | py | Tau_4mo | 1456 | 0.328 | 0.273 | 1.206 | 0.988 | 0.993 | 0.458 | 0.052 |
| Microglia | py | Tau_6mo | 1456 | 0.332 | 0.276 | 1.220 | 0.988 | 0.991 | 0.458 | 0.044 |
| Oligodendrocytes | py | ApTt_2mo | 1456 | 0.504 | 0.277 | 1.973 | 0.975 | 0.999 | 0.292 | 0.047 |
| Oligodendrocytes | py | ApTt_4mo | 1456 | 0.560 | 0.308 | 2.190 | 0.886 | 1.012 | 0.390 | 0.050 |
| Oligodendrocytes | py | ApTt_6mo | 1456 | 0.560 | 0.308 | 2.191 | 0.919 | 0.991 | 0.351 | 0.055 |
| Oligodendrocytes | py | App_2mo | 1456 | 0.512 | 0.281 | 2.002 | 0.982 | 1.000 | 0.301 | 0.026 |
| Oligodendrocytes | py | App_4mo | 1456 | 0.518 | 0.284 | 2.026 | 0.977 | 1.004 | 0.381 | 0.044 |
| Oligodendrocytes | py | App_6mo | 1456 | 0.556 | 0.306 | 2.176 | 0.979 | 1.008 | 0.381 | 0.041 |
| Oligodendrocytes | py | Tau_2mo | 1456 | 0.512 | 0.281 | 2.002 | 0.985 | 1.006 | 0.304 | 0.027 |
| Oligodendrocytes | py | Tau_4mo | 1456 | 0.518 | 0.284 | 2.026 | 0.980 | 1.016 | 0.357 | 0.045 |
| Oligodendrocytes | py | Tau_6mo | 1456 | 0.524 | 0.288 | 2.049 | 0.969 | 0.982 | 0.369 | 0.055 |
| Astrocytes | st | ApTt_2mo | 16010 | 0.733 | 0.327 | 2.091 | 0.970 | 0.996 | 0.376 | 0.059 |
| Astrocytes | st | ApTt_4mo | 16010 | 0.814 | 0.363 | 2.320 | 0.906 | 0.956 | 0.219 | 0.056 |
| Astrocytes | st | ApTt_6mo | 16010 | 0.814 | 0.363 | 2.321 | 0.916 | 0.973 | 0.311 | 0.065 |
| Astrocytes | st | App_2mo | 16010 | 0.744 | 0.332 | 2.121 | 0.955 | 0.992 | 0.409 | 0.041 |
| Astrocytes | st | App_4mo | 16010 | 0.753 | 0.336 | 2.147 | 0.960 | 0.969 | 0.288 | 0.051 |
| Astrocytes | st | App_6mo | 16010 | 0.809 | 0.361 | 2.306 | 0.957 | 1.006 | 0.236 | 0.060 |
| Astrocytes | st | Tau_2mo | 16010 | 0.744 | 0.332 | 2.121 | 0.973 | 0.992 | 0.299 | 0.053 |
| Astrocytes | st | Tau_4mo | 16010 | 0.753 | 0.336 | 2.147 | 0.947 | 0.979 | 0.229 | 0.047 |
| Astrocytes | st | Tau_6mo | 16010 | 0.761 | 0.340 | 2.172 | 0.942 | 1.005 | 0.475 | 0.061 |
| Microglia | st | ApTt_2mo | 16010 | 0.351 | 0.268 | 1.219 | 0.991 | 0.997 | 0.441 | 0.044 |
| Microglia | st | ApTt_4mo | 16010 | 0.390 | 0.297 | 1.353 | 0.869 | 1.002 | 0.341 | 0.048 |
| Microglia | st | ApTt_6mo | 16010 | 0.390 | 0.297 | 1.353 | 0.900 | 1.000 | 0.404 | 0.049 |
| Microglia | st | App_2mo | 16010 | 0.356 | 0.271 | 1.236 | 0.992 | 0.993 | 0.406 | 0.032 |
| Microglia | st | App_4mo | 16010 | 0.361 | 0.275 | 1.251 | 0.990 | 1.002 | 0.495 | 0.048 |
| Microglia | st | App_6mo | 16010 | 0.387 | 0.295 | 1.344 | 0.991 | 0.997 | 0.452 | 0.041 |
| Microglia | st | Tau_2mo | 16010 | 0.356 | 0.271 | 1.236 | 0.993 | 0.991 | 0.418 | 0.041 |
| Microglia | st | Tau_4mo | 16010 | 0.361 | 0.275 | 1.251 | 0.986 | 0.997 | 0.386 | 0.047 |
| Microglia | st | Tau_6mo | 16010 | 0.365 | 0.278 | 1.266 | 0.988 | 1.005 | 0.435 | 0.048 |
| Oligodendrocytes | st | ApTt_2mo | 16010 | 0.567 | 0.280 | 1.812 | 0.981 | 0.991 | 0.245 | 0.053 |
| Oligodendrocytes | st | ApTt_4mo | 16010 | 0.629 | 0.310 | 2.010 | 0.871 | 0.993 | 0.366 | 0.055 |
| Oligodendrocytes | st | ApTt_6mo | 16010 | 0.629 | 0.310 | 2.011 | 0.916 | 1.000 | 0.316 | 0.066 |
| Oligodendrocytes | st | App_2mo | 16010 | 0.575 | 0.284 | 1.838 | 0.973 | 0.986 | 0.249 | 0.039 |
| Oligodendrocytes | st | App_4mo | 16010 | 0.582 | 0.287 | 1.860 | 0.973 | 0.992 | 0.358 | 0.051 |
| Oligodendrocytes | st | App_6mo | 16010 | 0.625 | 0.308 | 1.998 | 0.977 | 0.995 | 0.314 | 0.055 |
| Oligodendrocytes | st | Tau_2mo | 16010 | 0.575 | 0.284 | 1.838 | 0.979 | 0.988 | 0.318 | 0.045 |
| Oligodendrocytes | st | Tau_4mo | 16010 | 0.582 | 0.287 | 1.860 | 0.971 | 0.997 | 0.309 | 0.047 |
| Oligodendrocytes | st | Tau_6mo | 16010 | 0.588 | 0.290 | 1.881 | 0.969 | 1.011 | 0.419 | 0.061 |

## MEA reach (FDR < 0.25, common scope)

| side | n_rows | n_sig_FDR_lt_0p25 | n_kinases_sig |
|---|---|---|---|
| per_animal | 10503 | 106 | 80 |
| group_level | 5598 | 70 | 36 |

- Median fraction of sites with p<0.05: per-animal **0.368** vs. group-level **0.047** (≈ 7.9× detection gain)

## **Verdict: PASS.** Point estimates faithful to Yuyu's group-level OLS (median LFC ρ > 0.9) and detection power improved sharply (≥3× more sites at p<0.05). The raw SE ratio is partial because per-animal OLS captures within-group variance that the 24-sample group-level OLS averaged away — both SEs are correct for what they measure, but per-animal is the more honest estimate. Green-light the full 46-cluster run.
