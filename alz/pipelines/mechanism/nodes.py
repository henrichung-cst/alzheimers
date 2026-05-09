from __future__ import annotations

import pandas as pd

from alz.kinase_enrich import _filter_samples, _resolve_track
from alz.kinase_mechanism import (
    _classify_mechanisms,
    _merge_mechanism_into_unified,
    _run_track_raw_mea,
)


def mea_raw_phospho_track(raw_phospho_normalized: pd.DataFrame,
                          sample_mapping: pd.DataFrame,
                          analysis_mode: str,
                          track: str) -> pd.DataFrame:
    track_cfg = _resolve_track(track)
    mapping = _filter_samples(sample_mapping, analysis_mode=analysis_mode)
    mea_raw = _run_track_raw_mea(track_cfg, raw_phospho_normalized, mapping)
    if mea_raw is None:
        return pd.DataFrame()
    return mea_raw


def classify_mechanisms(st_mea_raw: pd.DataFrame, py_mea_raw: pd.DataFrame,
                        st_mea_stoich: pd.DataFrame, py_mea_stoich: pd.DataFrame
                        ) -> pd.DataFrame:
    raw_frames = [df for df in (st_mea_raw, py_mea_raw)
                  if df is not None and not df.empty]
    if not raw_frames:
        raise RuntimeError(
            "No raw-phospho MEA outputs available; cannot classify mechanisms.")
    mea_raw = pd.concat(raw_frames, ignore_index=True)

    stoich_frames = [df for df in (st_mea_stoich, py_mea_stoich)
                     if df is not None and not df.empty]
    if not stoich_frames:
        raise RuntimeError(
            "No stoichiometry MEA outputs available; run enrich first.")
    mea_stoich = pd.concat(stoich_frames, ignore_index=True)

    return _classify_mechanisms(mea_raw, mea_stoich)


def merge_into_unified(unified_attribution: pd.DataFrame,
                       mechanism_annotation: pd.DataFrame) -> pd.DataFrame:
    return _merge_mechanism_into_unified(unified_attribution,
                                         mechanism_annotation)
