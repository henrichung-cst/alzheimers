"""Nodes for the attribution-recovery pipeline.

Thin wrappers around helpers in `alz.attribution_recovery`. The
combiner mirrors the legacy `_load_mea_stoichiometry` (concat of
per-track MEA stoichiometry tables, with `residue_type` / `track`
columns backfilled when missing).
"""

from __future__ import annotations

import pandas as pd

from alz import config
from alz.attribution_recovery import step_attribution_recovery


def combine_mea_stoichiometry(st_mea_stoichiometry: pd.DataFrame,
                              py_mea_stoichiometry: pd.DataFrame
                              ) -> pd.DataFrame:
    """Concatenate per-track MEA stoichiometry tables for downstream recovery.

    Mirrors `attribution_recovery._load_mea_stoichiometry`: backfills
    `residue_type` and `track` columns from the per-track config when
    they're missing (legacy compat for files predating the pY refactor).
    """
    inputs = {"st": st_mea_stoichiometry, "py": py_mea_stoichiometry}
    frames = []
    for track_name, df in inputs.items():
        if df is None or df.empty:
            continue
        track_cfg = config.PHOSPHO_TRACKS[track_name]
        df_out = df.copy()
        if "residue_type" not in df_out.columns:
            df_out["residue_type"] = track_cfg["residue"]
        if "track" not in df_out.columns:
            df_out["track"] = track_cfg["name"]
        frames.append(df_out)
    if not frames:
        raise FileNotFoundError(
            "No MEA stoichiometry tracks available; run enrich first.")
    return pd.concat(frames, ignore_index=True, sort=False)


def compute_recovery_tables(mea_stoichiometry_combined: pd.DataFrame,
                            unified_attribution_full: pd.DataFrame):
    """Build the three hypothesis tables (kinase activity, celltype evidence, hypothesis)."""
    return step_attribution_recovery(mea_stoichiometry_combined,
                                     unified_attribution_full)
