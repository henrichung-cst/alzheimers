#!/usr/bin/env python3
"""Standard cell-type attribution metric — one definition, every cohort.

Implements the metric agreed in ``docs/plans/standard_attribution_metric.md``.
It answers "how specific is this kinase to a cell type?" *without* the
share-is-not-presence failure that the per-cohort ``specificity_score`` shares
suffer (a near-zero kinase scores a high share wherever competition is lowest;
inversely predictive of truth). The fix is two-fold and load-bearing:

1. **Detection gate** — a cell type counts only when the kinase is genuinely
   present there (``fraction_cells_expressing >= DETECTION_FRAC_MIN``). This is
   count-based, so it needs no normalization and means the same thing in every
   pipeline. Noise cell types are dropped *before* any concentration math, which
   is what removes the phantom specificity.
2. **Linear weights** — concentration / breadth are computed on linear per-cell
   expression (de-logged from the stored log mean), not on log means. Log means
   compress high values and make broad kinases look concentrated.

Two questions, built from one foundation:

* **Q1 — specific to *this* cell type?** ``detected`` (✓/✗) paired with
  ``concentration`` = the cell type's share of expression **among detected cell
  types only**, binned to the familiar ``concentration_tier`` (≥2×/5×/10× over
  ``1/n_detected`` — an even split among the cell types it is actually in, *not*
  ``1/N_total`` padded with empty cell types). Meaningful only when
  ``n_detected >= 2``; with one detected cell type, ``effective_n = 1`` carries
  the specificity and the tier is trivially 1×.
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

# Familiar fold-over-even-share bins, applied to ``concentration`` (the one
# surviving share, now gated + linear) over a 1/n_detected baseline.
TIER_MULTIPLES = (10, 5, 2, 1)


def delog2(mean_log2) -> float:
    """Linear per-cell expression from a stored mean of log2(x+1). Clipped ≥0."""
    v = float(mean_log2)
    if not np.isfinite(v):
        return 0.0
    return max(2.0 ** v - 1.0, 0.0)


def concentration_tier(conc, n_detected: int) -> int:
    """First TIER_MULTIPLE t where conc >= t/n_detected, else 0 (1× = even share)."""
    if n_detected < 1 or conc is None or not np.isfinite(conc):
        return 0
    uniform = 1.0 / n_detected
    for t in TIER_MULTIPLES:
        if conc >= t * uniform:
            return t
    return 0


def _breadth(labels: list[str], weights: np.ndarray, detected: np.ndarray) -> dict:
    """Concentration over detected labels + effective number of cell types.

    weights: linear per-cell expression per label (same order as labels).
    detected: boolean mask per label.
    Returns per-label concentration/tier arrays + scalar summary.
    """
    n = len(labels)
    conc = np.full(n, np.nan)
    tier = np.zeros(n, dtype=int)
    det_idx = np.where(detected)[0]
    n_det = int(len(det_idx))
    summary = {"n_detected": n_det, "effective_n": np.nan,
               "top_label": "", "top_concentration": np.nan}
    if n_det == 0:
        return {"concentration": conc, "concentration_tier": tier, **summary}
    w = np.clip(weights[det_idx], 0.0, None)
    total = float(w.sum())
    if total <= 0:
        # Detected by prevalence but zero linear weight (all log-means ~0):
        # fall back to an even split so the kinase is not silently dropped.
        c = np.full(n_det, 1.0 / n_det)
    else:
        c = w / total
    conc[det_idx] = c
    for k, idx in enumerate(det_idx):
        tier[idx] = concentration_tier(float(c[k]), n_det)
    summary["effective_n"] = float(1.0 / np.sum(c * c))
    top_k = int(np.argmax(c))
    summary["top_label"] = labels[det_idx[top_k]]
    summary["top_concentration"] = float(c[top_k])
    return {"concentration": conc, "concentration_tier": tier, **summary}


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
                   concentration_tier (native resolution).
      per_group  — one row per (gene, coarse group): group_fraction,
                   group_detected, concentration_coarse, concentration_tier_coarse
                   (None when group_col is None).
      per_gene   — one row per gene: native + coarse {n_detected, effective_n,
                   top_celltype/top_group, top_concentration}.
    """
    per_label = df.copy()
    per_label["linear_expression"] = per_label[mean_log2_col].map(delog2)
    per_label["detected"] = per_label[frac_col].astype(float) >= detection_frac_min
    per_label["concentration"] = np.nan
    per_label["concentration_tier"] = 0

    gene_rows: list[dict] = []
    group_rows: list[dict] = []

    for gene, g in per_label.groupby(gene_col, sort=False):
        labels = g[label_col].astype(str).tolist()
        w = g["linear_expression"].to_numpy(dtype=float)
        det = g["detected"].to_numpy(dtype=bool)
        b = _breadth(labels, w, det)
        per_label.loc[g.index, "concentration"] = b["concentration"]
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
    per_gene = pd.DataFrame(gene_rows)
    per_group = pd.DataFrame(group_rows) if group_col is not None else None
    return per_label, per_group, per_gene
