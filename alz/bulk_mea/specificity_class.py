"""Recalculate the attribution confidence pill as within-cohort cell-type exclusivity.

The confidence pill keeps its familiar vocabulary
(``none`` / ``low`` / ``moderate`` / ``high`` / ``very_high``) but its meaning is
recalculated: the top tiers now mean "highly and *exclusively* expressed in one
cell type, with the reference data agreeing." This replaces the prior
disease-*direction* concordance tier, which is preserved as a secondary signal
(``direction_tier`` / ``direction_basis``) for the tooltip.

Resolution is the **curated specificity unit** (``config.load_specificity_unit_map``),
not blindly the 31 native Song clusters nor a flat coarse rollup. The native
clusters over-split some cell types (excitatory neurons → 6 pyramidal subtypes),
which dilutes a pan-class kinase and makes it look non-specific; those WMB classes
are collapsed into a parent. WMB classes whose clusters are genuinely distinct cell
types (vascular = endothelial + pericyte + …) stay split — the native cluster is
the correct unit there. The collapse is never silent: the viewer shows a collapsed
unit as an expandable parent over its child clusters.

Design (see ``docs/plans/cross_reference_exclusivity_regrouping.md``):

* **Within-cohort Song is the primary** and sets the tier. Fold the all-label
  ``song_concentration`` onto the specificity units and measure exclusivity by the
  effective number of units ``eff = 1 / Σ unit_share²``. The dominant unit is the
  one with the largest share; its top child cluster is the home cell type.
* **Reference data is corroborative, never required and never a veto.** The atlases
  have no cluster vocabulary, so they vote at the WMB-class level. The dominant
  unit's home cluster is projected to its WMB class and compared against WMB's true
  top class (``wmb_top_celltype``) and the human references' strongest location.

Tiers (Song effective # units ``eff``; ``corroborated`` = WMB or human agrees on
the WMB class of the dominant unit's home cluster):

    none       no measurable Song expression distribution
    low        expressed but broadly distributed (eff > BROAD_EFF_MAX)
    moderate   Song-exclusive (eff <= BROAD_EFF_MAX), no reference corroboration
    high       Song-exclusive + corroborated   — OR — Song-very-exclusive uncorroborated
    very_high  Song very-exclusive (eff <= EXCLUSIVE_EFF_MAX) + corroborated
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alz.shared import config
from alz.bulk_mea.confidence import HUMAN_STRONG_LOG2_SPECIFICITY as HUMAN_STRONG
from alz.bulk_mea.exclusivity_tier import (
    EXCLUSIVE_EFF_MAX,
    BROAD_EFF_MAX,
    exclusivity_tier,
)

_UNIT_MAP = config.load_specificity_unit_map()


def unit_concentration_shares(cell_types, concentrations) -> pd.Series:
    """Collapse per-cluster concentration shares onto curated specificity units.

    This is the shared **Song convention** for cell-type exclusivity: native
    clusters are mapped to their specificity unit via
    ``config.load_specificity_unit_map()`` (unmapped clusters fall back to their
    own name — no collapse), and the all-label ``concentration`` shares are
    summed within each unit. Returns the per-unit summed shares (units with
    share > 0 only). Used by both Song and 5xFAD so the confidence-pill ``eff`` is
    computed over the *same* unit vocabulary in every cohort.
    """
    s = pd.DataFrame({
        "_unit": [str(c) for c in cell_types],
        "_conc": pd.to_numeric(list(concentrations), errors="coerce"),
    })
    s["_unit"] = s["_unit"].map(lambda c: _UNIT_MAP.get(c, {}).get("unit", c))
    s["_conc"] = s["_conc"].fillna(0.0)
    shares = s.groupby("_unit")["_conc"].sum()
    return shares[shares > 0]


def unit_effective_n(shares: pd.Series) -> float:
    """Effective number of curated specificity units: ``eff = 1 / Σ unit_share²``.

    Input is the per-unit shares from :func:`unit_concentration_shares`. This is
    the number the confidence pill consumes — NOT the native per-cluster
    ``effective_n`` (the "subtype spread"), which over-counts split cell types.
    """
    if shares is None or not len(shares):
        return float("nan")
    ss = float(np.sum(shares.values ** 2))
    return (1.0 / ss) if ss > 0 else float("nan")


def _first_str(series: pd.Series) -> str:
    for v in series:
        if isinstance(v, str) and v:
            return v
    return ""


def _recalc_kinase(g: pd.DataFrame) -> dict:
    """Recalculate the confidence tier for one kinase from its per-cell-type rows.

    Expects columns: cell_type (Song Levy-T5 cluster), wmb_class,
    song_concentration, wmb_top_celltype, human_location_score.
    """
    sub = g.copy()
    sub["_unit"] = sub["cell_type"].astype(str).map(
        lambda c: _UNIT_MAP.get(c, {}).get("unit", c))
    sub["_conc"] = pd.to_numeric(sub["song_concentration"], errors="coerce").fillna(0.0)

    # --- Within-cohort Song (primary): exclusivity over specificity units ---
    shares = unit_concentration_shares(sub["cell_type"], sub["_conc"])
    if not len(shares):
        return _none_result()
    dom_unit = str(shares.idxmax())
    song_eff = unit_effective_n(shares)

    # Home cluster = the dominant unit's top child by all-label concentration.
    dom_rows = sub[sub["_unit"] == dom_unit]
    home_cluster = str(dom_rows.loc[dom_rows["_conc"].idxmax(), "cell_type"])
    info = _UNIT_MAP.get(home_cluster, {})
    unit_label = info.get("label", dom_unit)
    collapsed = bool(info.get("collapsed", False))
    dom_wmb_class = info.get("wmb_class", "") or _first_str(dom_rows["wmb_class"])

    # --- WMB corroboration (mouse atlas, at its class level) ---
    wmb_top = _first_str(g["wmb_top_celltype"])
    wmb_agree = bool(dom_wmb_class) and dom_wmb_class == wmb_top

    # --- Human corroboration (SEA-AD/HBCA, only when location is strong) ---
    hscore = pd.to_numeric(g["human_location_score"], errors="coerce")
    human_agree = False
    if hscore.notna().any() and dom_wmb_class:
        j = hscore.idxmax()
        if float(hscore.loc[j]) >= HUMAN_STRONG:
            human_agree = str(g.loc[j, "wmb_class"]) == dom_wmb_class
    corroborated = wmb_agree or human_agree

    # --- Tier: shared exclusivity helper (single source of truth) ---
    tier, _ = exclusivity_tier(True, song_eff, corroborated)

    return {
        "confidence_tier": tier,
        "confidence_basis": _basis(tier, unit_label, song_eff, wmb_agree, human_agree),
        "specificity_unit": dom_unit,
        "specificity_unit_label": unit_label,
        "specificity_celltype": home_cluster,
        "specificity_collapsed": collapsed,
        # The unit-level eff that SET the tier (1 / Σ unit_share²), distinct from
        # the native-cluster song_effective_n. This is the number the pill uses.
        "song_unit_effective_n": float(song_eff) if np.isfinite(song_eff) else np.nan,
    }


def _none_result() -> dict:
    return {
        "confidence_tier": "none",
        "confidence_basis": _basis("none", "", np.nan, False, False),
        "specificity_unit": "",
        "specificity_unit_label": "",
        "specificity_celltype": "",
        "specificity_collapsed": False,
        "song_unit_effective_n": np.nan,
    }


def _basis(tier, unit_label, eff, wmb_agree, human_agree) -> str:
    if tier == "none":
        return "no measurable within-cohort Song expression distribution"
    if tier == "low":
        return "broadly expressed across cell types (not cell-type-specific)"
    eff_s = f"{eff:.1f}" if np.isfinite(eff) else "?"
    corr = [r for r, ok in (("WMB", wmb_agree), ("human", human_agree)) if ok]
    corr_s = ("corroborated by " + "+".join(corr)) if corr else "within-cohort only"
    return f"Song exclusive to {unit_label} (eff {eff_s} cell types); {corr_s}"


def assign_specificity_class(unified: pd.DataFrame) -> pd.DataFrame:
    """Recalculate ``confidence_tier``/``confidence_basis`` as cell-type exclusivity.

    Snapshots the prior direction-concordance tier into ``direction_tier`` /
    ``direction_basis`` (secondary signal), then overwrites the headline pill.
    """
    out = unified.copy()

    # Preserve the prior (disease-direction) tier as a secondary signal.
    out["direction_tier"] = out.get("confidence_tier", "none")
    out["direction_basis"] = out.get("confidence_basis", "")

    need = ["cell_type", "wmb_class", "song_detected", "song_concentration",
            "wmb_top_celltype", "human_location_score"]
    for c in need:
        if c not in out.columns:
            out[c] = "" if c in ("wmb_class", "wmb_top_celltype") else np.nan

    # Expression signals are contrast-invariant: collapse to one row per
    # (kinase, cell_type) before recalculating, then broadcast back per kinase.
    ct = out.drop_duplicates(["kinase", "cell_type"])[["kinase"] + need]
    recs = [{"kinase": k, **_recalc_kinase(g)}
            for k, g in ct.groupby("kinase", sort=False)]
    per_kinase = pd.DataFrame(recs)

    out = out.drop(columns=["confidence_tier", "confidence_basis"], errors="ignore")
    out = out.merge(per_kinase, on="kinase", how="left")
    out["confidence_tier"] = out["confidence_tier"].fillna("none")
    out["confidence_basis"] = out["confidence_basis"].fillna("")
    for c in ("specificity_unit", "specificity_unit_label", "specificity_celltype"):
        out[c] = out[c].fillna("")
    out["specificity_collapsed"] = out["specificity_collapsed"].fillna(False).astype(bool)
    return out
