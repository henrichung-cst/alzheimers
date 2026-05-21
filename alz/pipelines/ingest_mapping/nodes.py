"""TMT channel-to-animal mapping node.

Phase 2 proof node: smallest real I/O contract on the live arc. Helpers are
imported from alz.data_ingest for now; Phase 4 will refactor data_ingest into
nodes and these helpers move with them.
"""
from __future__ import annotations

import pandas as pd

from alz.data_ingest import (
    GENOTYPE_TO_SAP,
    SEX_TO_SAP,
    _discover_snrna_samples,
    _parse_animal_id,
    _plex_channel_to_colname,
)

EXCLUDE_KEYWORDS = {"Large Pool", "na", "Ref_Pool", "WT_M", "WT_F", "DKO_M", "DKO_F"}

OUTPUT_COLUMNS = [
    "plex", "channel", "column_name", "animal_id", "mouse_id",
    "sex", "timepoint", "genotype", "replicate", "has_snrna_seq",
    "snrna_sample_id", "phospho_group_id",
]


def build_tmt_channel_mapping(
    tmt_layout: pd.DataFrame,
    total_proteome_columns: pd.DataFrame,
    proof_marker: str,
) -> pd.DataFrame:
    """Parse the TMT plex layout, map channels to animals, cross-ref snRNA-seq.

    Output is byte-identical to alz/data_ingest.py:step_sample_mapping.
    """
    print(f"  proof_marker (Kedro param): {proof_marker}")

    layout = tmt_layout.dropna(subset=["Plex"]).copy()
    layout["Plex"] = layout["Plex"].astype(int)

    bio = layout[~layout["Animal #"].isin(EXCLUDE_KEYWORDS)].copy()

    records = []
    for _, row in bio.iterrows():
        plex = int(row["Plex"])
        channel = str(row["Channel"])
        animal_str = str(row["Animal #"])
        parsed = _parse_animal_id(animal_str)
        if parsed is None:
            print(f"  WARNING: could not parse animal ID: '{animal_str}'")
            continue
        records.append({
            "plex": plex,
            "channel": channel,
            "column_name": _plex_channel_to_colname(plex, channel),
            "animal_id": animal_str,
            "mouse_id": parsed["mouse_id"],
            "mouse_id_raw": parsed["mouse_id_raw"],
            "sex": parsed["sex"],
            "timepoint": parsed["timepoint"],
            "genotype": parsed["genotype"],
            "sample_num": parsed["sample_num"],
        })

    df = pd.DataFrame(records)

    tp_cols = set(total_proteome_columns.columns)
    df["column_exists"] = df["column_name"].isin(tp_cols)
    n_missing = (~df["column_exists"]).sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} columns not found in total proteome")

    df = df.sort_values(["sex", "timepoint", "genotype", "sample_num"])
    df["replicate"] = df.groupby(["sex", "timepoint", "genotype"]).cumcount() + 1

    snrna_samples = _discover_snrna_samples()
    df["has_snrna_seq"] = df["mouse_id"].map(lambda mid: mid in snrna_samples)
    df["snrna_sample_id"] = df["mouse_id"].map(lambda mid: snrna_samples.get(mid, ""))

    df["phospho_group_id"] = df.apply(
        lambda r: f"{SEX_TO_SAP[r['sex']]}_{r['timepoint']}_{r['genotype']}",
        axis=1,
    )

    return df[OUTPUT_COLUMNS].reset_index(drop=True)
