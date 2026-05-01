"""Stage 6: roll up the 46-cluster table to 34 WMB classes.

For each (kinase × WMB-class × contrast × track), keep the strongest-
evidence row from the 46-cluster table (highest |NES| at deconv
FDR < 0.25; ties broken by lowest FDR), with the originating cluster
annotated. ``Unclassified`` clusters are excluded from the rolled-up
view (per the plan).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from deconvolution import paths


def aggregate(primary: pd.DataFrame) -> pd.DataFrame:
    df = primary.copy()
    df = df[df["wmb_class"].notna() & (df["wmb_class"] != "Unclassified")]
    if df.empty:
        return df

    # Restrict to deconvolution-significant rows for the aggregation.
    sig = df[df["FDR"] < paths.DECON_FDR_THRESH].copy()
    if sig.empty:
        return sig

    sig["abs_NES"] = sig["NES"].abs()
    sig = sig.sort_values(
        ["kinase", "wmb_class", "contrast", "track", "abs_NES", "FDR"],
        ascending=[True, True, True, True, False, True],
    )
    rolled = sig.drop_duplicates(
        subset=["kinase", "wmb_class", "contrast", "track"], keep="first"
    ).copy()
    rolled = rolled.rename(columns={"cluster": "cluster_of_origin"})
    rolled = rolled.drop(columns=["abs_NES"])
    return rolled.reset_index(drop=True)
