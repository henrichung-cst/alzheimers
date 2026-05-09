"""Nodes for the attribute pipeline.

Thin wrappers over helpers in `alz.kinase_attribute` so the legacy CLI shim
and the Kedro pipeline run the same code paths. Optional inputs (Song
specificity/concordance, WMB expression) are loaded inside nodes via path
parameters because Kedro's catalog cannot natively express "may not exist"
files; SEA-AD h5ads are loaded inside the node because anndata is not a
built-in Kedro dataset.
"""

from __future__ import annotations

import os

import pandas as pd

from alz import config
from alz.kinase_attribute import (
    _assemble_unified,
    _combine_mea_tracks,
    _compute_sea_ad_concordance,
    _map_kinases_to_genes,
    _prepare_song_concordance,
    _prepare_song_specificity,
    _prepare_wmb_specificity,
)


def combine_mea_tracks(st_mea: pd.DataFrame, py_mea: pd.DataFrame,
                       kinase_to_gene: pd.DataFrame) -> pd.DataFrame:
    """Concat per-track MEA outputs and inject kinase→gene mapping."""
    print("\n=== Stage 3: Unified Cell-Type Attribution ===\n")
    pairs = []
    for track_name, df in (("st", st_mea), ("py", py_mea)):
        track_cfg = config.PHOSPHO_TRACKS.get(track_name)
        if track_cfg is None or df is None:
            continue
        pairs.append((track_cfg, df))
    sig = _combine_mea_tracks(pairs)
    sig = _map_kinases_to_genes(sig, kinase_to_gene)
    return sig


def sea_ad_concordance(sig: pd.DataFrame,
                       seaad_to_wmb_class: pd.DataFrame,
                       sea_ad_paths: dict):
    """Compute SEA-AD concordance + supertype audit. Returns
    ``(sea_ad_df, sea_ad_supertype_lfc)``."""
    return _compute_sea_ad_concordance(sig, seaad_to_wmb_class, sea_ad_paths)


def assemble_unified(sig: pd.DataFrame,
                     sea_ad_df: pd.DataFrame,
                     wmb_expression_path: str,
                     song_specificity_path: str,
                     song_concordance_path: str):
    """Cross-join sig × WMB classes, layer evidence merges, score, label.

    Returns ``(unified_attributed, unified_full, attribution_summary)``.
    Optional inputs are guarded by path existence."""
    wmb_df = (pd.read_csv(wmb_expression_path)
              if wmb_expression_path and os.path.exists(wmb_expression_path)
              else None)
    if wmb_df is None and wmb_expression_path:
        print(f"  WMB expression file not found at {wmb_expression_path}")
    wmb_top = _prepare_wmb_specificity(wmb_df)

    song_sp_df = (pd.read_csv(song_specificity_path)
                  if song_specificity_path and os.path.exists(song_specificity_path)
                  else None)
    song_spec_top = _prepare_song_specificity(song_sp_df)

    song_cd_df = (pd.read_csv(song_concordance_path)
                  if song_concordance_path and os.path.exists(song_concordance_path)
                  else None)
    song_cd_top, song_key_is_contrast = _prepare_song_concordance(song_cd_df)

    unified, attributed, summary = _assemble_unified(
        sig, sea_ad_df, wmb_top, song_spec_top, song_cd_top, song_key_is_contrast
    )
    print(f"\n  Saved unified_attribution ({len(attributed)} attributed rows)")
    print("  Stage 3 complete.")
    return attributed, unified, summary
