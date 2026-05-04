"""Cohort-level snRNA cross-check.

Two evidence helpers attached to the Stage 4 MEA table:

1. ``compute_cohort_concordance``: per (wmb_class, contrast), test whether
   the sign-match rate between bulk NES and snRNA kinase-gene LFC exceeds
   chance via a one-sided binomial. The per-row snRNA FDR is saturated
   (n=15 male animals); aggregate sign concordance is what survives.
2. ``expression_presence``: per row, look up the kinase gene's
   ``mean_expression`` (log2(CPM+1)) in the matching WMB class. Rows below
   ``EXPR_PRESENCE_FLOOR`` (or missing from the specificity table) return
   False so a downstream reader can drop them regardless of bulk FDR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests

from deconvolution import paths


def compute_cohort_concordance(annotated_df: pd.DataFrame,
                               min_rows: int = 5) -> pd.DataFrame:
    """One row per (wmb_class, contrast) stratum.

    Restricts to bulk-significant rows (FDR < paths.DECON_FDR_THRESH) with
    finite snRNA LFC; counts sign-matches; tests P(X >= n_match | n, p=0.5)
    via the survival function of the binomial. BH across strata.
    """
    sub = annotated_df[
        (annotated_df["FDR"] < paths.DECON_FDR_THRESH)
        & annotated_df["kinase_gene_LFC_snRNA"].notna()
        & annotated_df["NES"].notna()
    ].copy()
    sub["match"] = np.sign(sub["NES"]) == np.sign(sub["kinase_gene_LFC_snRNA"])

    rows = []
    for (cl, contrast), grp in sub.groupby(["wmb_class", "contrast"]):
        n = len(grp)
        if n < min_rows:
            continue
        m = int(grp["match"].sum())
        # P(X >= m | n, p=0.5) = sf(m-1, n, 0.5)
        pval = float(binom.sf(m - 1, n, 0.5))
        rows.append({
            "wmb_class": cl, "contrast": contrast,
            "n_total": n, "n_match": m,
            "frac_match": m / n,
            "cohort_pval": pval,
        })
    cohort = pd.DataFrame(rows)
    if not len(cohort):
        cohort["cohort_fdr"] = np.array([], dtype=float)
        cohort["cohort_concordant"] = np.array([], dtype=bool)
        return cohort
    _, fdr, _, _ = multipletests(cohort["cohort_pval"].values, method="fdr_bh")
    cohort["cohort_fdr"] = fdr
    cohort["cohort_concordant"] = (
        (cohort["cohort_fdr"] < paths.COHORT_FDR_THRESH)
        & (cohort["frac_match"] > 0.5)
    )
    return cohort


def expression_presence(annotated_df: pd.DataFrame,
                        specificity_df: pd.DataFrame) -> pd.Series:
    """Boolean Series, indexed like annotated_df. True ↔ kinase gene is
    detectably expressed in the row's WMB class.

    Specificity rows below EXPR_PRESENCE_FLOOR or missing from the lookup
    return False. Missing wmb_class or gene_symbol on the MEA side also
    returns False (cannot corroborate presence → fail safe).
    """
    spec = specificity_df.set_index(["cell_type", "gene_symbol"])
    expr = spec["mean_expression"]
    keys = list(zip(annotated_df["wmb_class"].astype(object),
                    annotated_df["gene_symbol"].astype(object)))
    looked = expr.reindex(keys)
    looked.index = annotated_df.index
    return looked.fillna(0.0) >= paths.EXPR_PRESENCE_FLOOR
