from __future__ import annotations

import pandas as pd

from alz import config
from alz.attribution_recovery import step_attribution_recovery


def combine_mea_stoichiometry(st_mea_stoichiometry: pd.DataFrame,
                              py_mea_stoichiometry: pd.DataFrame
                              ) -> pd.DataFrame:
    frame_by_track = {"st": st_mea_stoichiometry, "py": py_mea_stoichiometry}
    frames = []
    for track_name, track_cfg in config.PHOSPHO_TRACKS.items():
        df = frame_by_track[track_name]
        if df is None or df.empty:
            continue
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


def compute_recovery_tables(
    mea_stoichiometry_combined: pd.DataFrame,
    unified_attribution_full: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return step_attribution_recovery(mea_stoichiometry_combined,
                                     unified_attribution_full)
