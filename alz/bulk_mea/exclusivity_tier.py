"""Shared cell-type exclusivity confidence tier.

Single source of truth for the confidence pill used across all three cohorts
(Song, 5xFAD, T-cell). Each cohort supplies its own within-cohort
``effective_n`` and a boolean ``corroborated`` flag from its own reference;
the tier formula is identical for all.

Tiers (constants match the Song calibration in ``specificity_class.py``):

    none       no measurable within-cohort expression distribution
    low        broadly expressed (eff > BROAD_EFF_MAX)
    moderate   exclusive (eff ≤ BROAD_EFF_MAX), no reference corroboration
    high       (eff ≤ BROAD_EFF_MAX, corroborated)
               OR (eff ≤ EXCLUSIVE_EFF_MAX, uncorroborated)
    very_high  eff ≤ EXCLUSIVE_EFF_MAX, corroborated

Direction concordance is a SEPARATE info-only axis — it never gates the tier.
"""

from __future__ import annotations

import math

EXCLUSIVE_EFF_MAX = 1.5   # inverse-Simpson ≈ 1 → essentially one cell type
BROAD_EFF_MAX = 3.0       # above this → broadly expressed


def exclusivity_tier(
    detected: bool,
    eff: float | None,
    corroborated: bool,
) -> tuple[str, str]:
    """Return (confidence_tier, confidence_basis) for one (kinase, group) entry.

    Args:
        detected:      True if the kinase has a measurable within-cohort
                       expression distribution. Detection fractions are reported
                       separately and do not define the specificity denominator.
        eff:           Within-cohort effective number of cell types
                       (inverse-Simpson breadth, 1/Σ concentration²).
                       None / NaN → treated as broadly expressed (low).
        corroborated:  True if an independent reference agrees the kinase
                       lives in the same home cell class.  The reference is
                       cohort-specific (Song: WMB+human; 5xFAD: WMB+SEA-AD;
                       T-cell: NSCLC lineage).  Absence from the probe panel
                       counts as uncorroborated (not as disconfirming).

    Returns:
        (tier, basis) where tier ∈ {none, low, moderate, high, very_high}.
    """
    if not detected:
        return "none", "no measurable within-cohort expression distribution"

    eff_finite = eff is not None and math.isfinite(float(eff))
    eff_val = float(eff) if eff_finite else float("inf")

    if not eff_finite or eff_val > BROAD_EFF_MAX:
        return "low", _basis_low(eff_val if eff_finite else None)

    if eff_val <= EXCLUSIVE_EFF_MAX:
        if corroborated:
            tier = "very_high"
        else:
            tier = "high"
    else:
        # ENRICHED range: EXCLUSIVE_EFF_MAX < eff ≤ BROAD_EFF_MAX
        if corroborated:
            tier = "high"
        else:
            tier = "moderate"

    return tier, _basis(tier, eff_val, corroborated)


def _basis_low(eff: float | None) -> str:
    if eff is None:
        return "effective_n is undefined; treated as broadly expressed"
    return f"broadly expressed across cell types (eff {eff:.1f} > {BROAD_EFF_MAX})"


def _basis(tier: str, eff: float, corroborated: bool) -> str:
    eff_s = f"{eff:.2f}"
    corr_s = "corroborated by reference" if corroborated else "reference absent or not corroborating"
    return f"eff {eff_s} cell types ({tier}); {corr_s}"
