#!/usr/bin/env python3
"""Standard cell-type attribution metric — one definition, every cohort.

Implements the metric agreed in ``docs/plans/attribution/standard_attribution_metric.md``.
It answers "how specific is this kinase to a cell type?" with one denominator:
all cell types/states in the cohort. Detection is reported as an independent
evidence column, but it does not change the denominator for concentration,
effective number, top cell type, or concentration tier. The load-bearing rule is:

1. **One denominator** — every specificity quantity is computed over all labels
   with finite expression in the cohort. Detection never filters the share basis.
2. **Linear weights** — concentration / breadth are computed on linear per-cell
   expression (de-logged from the stored log mean), not on log means. Log means
   compress high values and make broad kinases look concentrated.

Two questions, built from one foundation:

* **Q1 — specific to *this* cell type?** ``detected`` (✓/✗) is paired with
  ``concentration`` = the cell type's share of the kinase's total linear
  expression across ALL cell types. ``concentration_of_total`` is retained as an
  alias for downstream compatibility; it has the same denominator and value.
  The ``concentration_tier`` pill (≥2×/5×/10×) is the fold of that all-label
  share over the even ``1/N_total`` share, so a 10× pill is the same bar for
  every kinase.
* **Q2 — specific overall?** ``effective_n`` = ``1 / Σ concentration²`` — the
  magnitude-aware "effective number of cell types" (≈1 specific, ≈n broad). A
  count, not a fold, so no tier bins apply to it.

``compute`` returns the metric at the native cell-type resolution and, when a
coarse grouping is supplied, a second time over the coarse lineages — the data
legitimately gives different answers at different resolutions (e.g. a pan-T
kinase is effectively ~1 lineage but ~14 T-states), and the reader should see
both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Detection gate: fraction of cells with a non-zero count. Normalization-free,
# so it is the one cross-cohort presence bar (the scale-dependent mean>1 co-gate
# is deliberately NOT used here).
DETECTION_FRAC_MIN = 0.10

# Fold-over-even-share bins, applied to ``concentration_of_total`` (the cell
# type's share of the kinase's total linear expression) over a 1/N_total
# baseline — an even split across all cell types, so the fold is comparable
# across kinases.
TIER_MULTIPLES = (10, 5, 2, 1)


def delog2(mean_log2) -> float:
    """Linear per-cell expression from a stored mean of log2(x+1). Clipped ≥0."""
    v = float(mean_log2)
    if not np.isfinite(v):
        return 0.0
    return max(2.0 ** v - 1.0, 0.0)


def concentration_tier(share_of_total, n_total: int) -> int:
    """First TIER_MULTIPLE t where share_of_total >= t/n_total, else 0.

    Baseline is ``1/n_total`` — an even split across ALL cell types in the
    cohort — so a 10× pill means the same bar (≥10× the uniform share) for every
    kinase, independent of how many cell types it is detected in.
    """
    if n_total < 1 or share_of_total is None or not np.isfinite(share_of_total):
        return 0
    uniform = 1.0 / n_total
    for t in TIER_MULTIPLES:
        if share_of_total >= t * uniform:
            return t
    return 0


def _breadth(labels: list[str], weights: np.ndarray, detected: np.ndarray) -> dict:
    """All-label share basis + effective number of cell types.

    weights: linear per-cell expression per label (same order as labels).
    detected: boolean mask per label. Used only for the reported n_detected field;
    it never filters the specificity denominator.

    Returns ``concentration`` and ``concentration_of_total`` as the same per-label
    share: TOTAL linear expression over ALL labels. The duplicated output keeps
    old consumers working while enforcing one denominator.
    """
    n = len(labels)
    conc = np.full(n, np.nan)
    tier = np.zeros(n, dtype=int)
    w_all = np.clip(weights, 0.0, None)
    tot_all = float(w_all.sum())
    share_total = (w_all / tot_all) if tot_all > 0 else np.zeros(n)
    n_det = int(np.sum(detected))
    summary = {"n_detected": n_det, "effective_n": np.nan,
               "top_label": "", "top_concentration": np.nan}
    if n == 0 or tot_all <= 0:
        return {"concentration": conc, "concentration_of_total": share_total,
                "concentration_tier": tier, **summary}
    conc = share_total.copy()
    # Tier from all-label share over the even 1/n_total share.
    s = share_total
    uniform = 1.0 / n
    for t in TIER_MULTIPLES:
        tier[(tier == 0) & np.isfinite(s) & (s >= t * uniform)] = t
    summary["effective_n"] = float(1.0 / np.sum(share_total * share_total))
    top_k = int(np.argmax(share_total))
    summary["top_label"] = labels[top_k]
    summary["top_concentration"] = float(share_total[top_k])
    return {"concentration": conc, "concentration_of_total": share_total,
            "concentration_tier": tier, **summary}


def compute(df: pd.DataFrame, *, gene_col: str, label_col: str,
            mean_log2_col: str, frac_col: str, ncells_col: str,
            group_col: str | None = None,
            detection_frac_min: float = DETECTION_FRAC_MIN
            ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Compute the standard attribution metric.

    Input: one row per (gene, cell-type label) with a mean log2(x+1) expression,
    a fraction-cells-expressing, and an n_cells.

    Returns (per_label, per_group, per_gene):
      per_label  — input rows + linear_expression, detected, concentration,
                   concentration_of_total, concentration_tier (native resolution).
      per_group  — one row per (gene, coarse group): group_fraction,
                   group_detected, concentration_coarse, concentration_of_total_coarse,
                   concentration_tier_coarse (None when group_col is None).
      per_gene   — one row per gene: native + coarse {n_detected, effective_n,
                   top_celltype/top_group, top_concentration}.
    """
    per_label = df.copy()
    # Vectorized delog2: linear per-cell expression from mean log2(x+1), clipped ≥0.
    _ml = per_label[mean_log2_col].to_numpy(dtype=float)
    per_label["linear_expression"] = np.where(
        np.isfinite(_ml), np.maximum(np.exp2(_ml) - 1.0, 0.0), 0.0)
    per_label["detected"] = per_label[frac_col].astype(float) >= detection_frac_min
    per_label["concentration"] = np.nan
    per_label["concentration_of_total"] = np.nan
    per_label["concentration_tier"] = 0

    gene_rows: list[dict] = []
    group_rows: list[dict] = []

    for gene, g in per_label.groupby(gene_col, sort=False):
        labels = g[label_col].astype(str).tolist()
        w = g["linear_expression"].to_numpy(dtype=float)
        det = g["detected"].to_numpy(dtype=bool)
        b = _breadth(labels, w, det)
        per_label.loc[g.index, "concentration"] = b["concentration"]
        per_label.loc[g.index, "concentration_of_total"] = b["concentration_of_total"]
        per_label.loc[g.index, "concentration_tier"] = b["concentration_tier"]
        rec = {gene_col: gene,
               "n_detected_native": b["n_detected"],
               "effective_n_native": b["effective_n"],
               "top_celltype_native": b["top_label"],
               "top_concentration_native": b["top_concentration"]}

        if group_col is not None:
            # Cell-weighted aggregation of native labels to coarse groups.
            gg = g.copy()
            gg["_wf"] = gg[frac_col].astype(float) * gg[ncells_col].astype(float)
            gg["_wl"] = gg["linear_expression"] * gg[ncells_col].astype(float)
            agg = gg.groupby(group_col, sort=False).agg(
                n=(ncells_col, "sum"), wf=("_wf", "sum"), wl=("_wl", "sum"))
            agg["group_fraction"] = agg["wf"] / agg["n"].clip(lower=1)
            agg["group_linear"] = agg["wl"] / agg["n"].clip(lower=1)
            grp_labels = agg.index.astype(str).tolist()
            grp_w = agg["group_linear"].to_numpy(dtype=float)
            grp_det = (agg["group_fraction"].to_numpy(dtype=float)
                       >= detection_frac_min)
            gb = _breadth(grp_labels, grp_w, grp_det)
            for k, glab in enumerate(grp_labels):
                group_rows.append({
                    gene_col: gene, group_col: glab,
                    "group_fraction": round(float(agg["group_fraction"].iloc[k]), 6),
                    "group_detected": bool(grp_det[k]),
                    "concentration_coarse": (None if not np.isfinite(gb["concentration"][k])
                                             else round(float(gb["concentration"][k]), 6)),
                    "concentration_of_total_coarse": round(float(gb["concentration_of_total"][k]), 6),
                    "concentration_tier_coarse": int(gb["concentration_tier"][k]),
                })
            rec.update({
                "n_detected_coarse": gb["n_detected"],
                "effective_n_coarse": gb["effective_n"],
                "top_group_coarse": gb["top_label"],
                "top_concentration_coarse": gb["top_concentration"],
            })
        gene_rows.append(rec)

    per_label["concentration"] = per_label["concentration"].round(6)
    per_label["concentration_of_total"] = per_label["concentration_of_total"].round(6)
    per_gene = pd.DataFrame(gene_rows)
    per_group = pd.DataFrame(group_rows) if group_col is not None else None
    return per_label, per_group, per_gene
