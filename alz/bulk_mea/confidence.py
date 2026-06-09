"""Canonical cell-type attribution confidence model.

Song within-cohort evidence is the primary high-confidence signal. Mouse WMB,
human SEA-AD/HBCA location specificity, and decomposition are cross-checks
that can support moderate calls or promote a Song-high row to very_high.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from alz.shared import config


CONFIDENCE_RANK = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "very_high": 4,
}

CONFIDENCE_COLUMNS = [
    "confidence_tier",
    "confidence_basis",
    "song_direction_support",
    "song_location_tier",
    "wmb_crosscheck_tier",
    "human_location_tier",
    "decomp_agrees_bulk",
]

HUMAN_STRONG_LOG2_SPECIFICITY = 1.0
DECOMP_FDR_AGREEMENT = 0.25


def _tier_from_share(value: float, uniform: float) -> str:
    if not np.isfinite(value) or value <= 0:
        return "none"
    if value >= 2.0 * uniform:
        return "high"
    if value >= uniform:
        return "above_uniform"
    return "below_uniform"


def _human_location_tier(value: float) -> str:
    if not np.isfinite(value) or value <= 0:
        return "none"
    if value >= HUMAN_STRONG_LOG2_SPECIFICITY:
        return "strong"
    return "positive"


def load_decomposition_crosscheck() -> pd.DataFrame | None:
    """Load per-cluster decomposition rows needed for confidence promotion."""
    base_dir = Path(config.REPO_ROOT) / "outputs" / "reports" / "decomposition" / "levy_t5"
    paths = [
        (base_dir / "mea_per_cluster.parquet", "st"),
        (base_dir / "mea_per_cluster_pY.parquet", "py"),
    ]
    frames = []
    for path, track in paths:
        if not path.exists():
            continue
        cols = ["kinase", "cluster", "contrast", "NES", "FDR"]
        try:
            df = pd.read_parquet(path, columns=cols + ["track"])
            df = df[df["track"].astype(str).str.lower() == track].drop(columns=["track"])
        except Exception:
            df = pd.read_parquet(path, columns=cols)
        frames.append(df.rename(columns={
            "cluster": "cell_type",
            "NES": "decomp_nes",
            "FDR": "decomp_fdr",
        }))
    if not frames:
        return None
    decomp = pd.concat(frames, ignore_index=True, sort=False)
    return (decomp.sort_values("decomp_fdr", ascending=True)
            .drop_duplicates(["kinase", "contrast", "cell_type"], keep="first"))


def prepare_human_location_specificity() -> pd.DataFrame | None:
    """Roll SEA-AD MTG and HBCA specificity matrices onto Levy-T5 clusters."""
    try:
        from alz.cross_reference.human_celltype_attribution import (
            _filter_hbca_cortex_hpc,
            _rollup_matrix_to_levy_t5,
        )
    except Exception as exc:
        print(f"  Human specificity helpers unavailable: {exc}")
        return None

    parts: list[pd.DataFrame] = []
    for ref, path in [
        ("seaad", config.SEAAD_KINASE_SPECIFICITY_FILE),
        ("hbca", config.HBCA_KINASE_SPECIFICITY_FILE),
    ]:
        if not os.path.exists(path):
            continue
        spec = pd.read_csv(path, index_col=0)
        if ref == "hbca":
            spec = _filter_hbca_cortex_hpc(spec)
        rolled, _ = _rollup_matrix_to_levy_t5(spec, "seaad_mtg" if ref == "seaad" else "allen_hbca")
        long = (rolled.reset_index(names="_gene_upper")
                .melt(id_vars="_gene_upper",
                      var_name="cell_type",
                      value_name=f"{ref}_location_score"))
        long["_gene_upper"] = long["_gene_upper"].astype(str).str.upper()
        parts.append(long)

    if not parts:
        return None

    human = parts[0]
    for part in parts[1:]:
        human = human.merge(part, on=["_gene_upper", "cell_type"], how="outer")
    score_cols = [c for c in ("seaad_location_score", "hbca_location_score")
                  if c in human.columns]
    human["human_location_score"] = human[score_cols].max(axis=1, skipna=True)
    print(f"  Human location specificity: {len(human)} (gene, cell_type) pairs loaded")
    return human


def assign_confidence(unified: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical confidence fields and component labels."""
    out = unified.copy()
    idx = out.index

    for col, default in [
        ("song_lfc", np.nan),
        ("song_specificity", np.nan),
        ("wmb_specificity", np.nan),
        ("sea_ad_lfc", np.nan),
        ("human_location_score", np.nan),
        ("decomp_nes", np.nan),
        ("decomp_fdr", np.nan),
        ("_effective_concordance", 0.0),
        ("concordance_source", "none"),
    ]:
        if col not in out.columns:
            out[col] = default

    eligible = out["mea_significant"].astype(bool) & (out["_effective_concordance"].fillna(0.0) > 0)
    song_lfc = pd.to_numeric(out["song_lfc"], errors="coerce")
    song_spec = pd.to_numeric(out["song_specificity"], errors="coerce")
    wmb_spec = pd.to_numeric(out["wmb_specificity"], errors="coerce")
    human_score = pd.to_numeric(out["human_location_score"], errors="coerce")
    sea_lfc = pd.to_numeric(out["sea_ad_lfc"], errors="coerce")
    bulk_nes = pd.to_numeric(out["NES"], errors="coerce")
    decomp_nes = pd.to_numeric(out["decomp_nes"], errors="coerce")
    decomp_fdr = pd.to_numeric(out["decomp_fdr"], errors="coerce")

    song_contributed = out["concordance_source"].isin(("song", "both"))
    seaad_contributed = out["concordance_source"].isin(("sea_ad", "both"))
    song_direction_support = song_contributed & (song_lfc.abs() > config.SONG_LFC_MIN)
    song_location_high = song_spec >= (2.0 / config.N_CELL_TYPES)
    wmb_crosscheck = wmb_spec >= config.wmb_specificity_uniform()
    strong_human_location = human_score >= HUMAN_STRONG_LOG2_SPECIFICITY
    decomp_agrees_bulk = (
        (decomp_fdr < DECOMP_FDR_AGREEMENT)
        & np.isfinite(decomp_nes)
        & np.isfinite(bulk_nes)
        & (decomp_nes != 0)
        & (bulk_nes != 0)
        & ((decomp_nes > 0) == (bulk_nes > 0))
    )

    high = eligible & song_contributed & song_direction_support & song_location_high
    very_high = high & decomp_agrees_bulk
    moderate_song = eligible & song_contributed & ~high
    moderate_seaad = eligible & ~song_contributed & seaad_contributed & (
        wmb_crosscheck | strong_human_location
    )
    low = eligible & ~(very_high | high | moderate_song | moderate_seaad)

    tier = pd.Series("none", index=idx, dtype=object)
    tier.loc[low] = "low"
    tier.loc[moderate_song | moderate_seaad] = "moderate"
    tier.loc[high] = "high"
    tier.loc[very_high] = "very_high"
    out["confidence_tier"] = tier

    basis = pd.Series("none", index=idx, dtype=object)
    basis.loc[low] = "low_concordance"
    basis.loc[moderate_seaad & strong_human_location] = "seaad_human_moderate"
    basis.loc[moderate_seaad & ~strong_human_location & wmb_crosscheck] = "seaad_wmb_moderate"
    basis.loc[moderate_song] = "song_moderate"
    basis.loc[high] = "song_high"
    basis.loc[very_high] = "song_high_decomp"
    out["confidence_basis"] = basis

    out["song_direction_support"] = song_direction_support.astype(bool)
    out["song_location_tier"] = [
        _tier_from_share(float(v), 1.0 / config.N_CELL_TYPES) for v in song_spec.fillna(np.nan)
    ]
    out["wmb_crosscheck_tier"] = [
        _tier_from_share(float(v), config.wmb_specificity_uniform()) for v in wmb_spec.fillna(np.nan)
    ]
    out["human_location_tier"] = [
        _human_location_tier(float(v)) for v in human_score.fillna(np.nan)
    ]
    out["decomp_agrees_bulk"] = decomp_agrees_bulk.astype(bool)

    return out
