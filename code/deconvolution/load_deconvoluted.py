"""Stage 1: load pre-computed Yuyu 46-cluster deconvolution outputs.

Each deconvoluted CSV has metadata columns followed by 24 samples × 46
clusters = 1104 value columns named ``{sample}_{cluster}``. We reshape
to a long, indexed structure: per-site metadata + a (n_sites × 1104)
value matrix with a tidy column index of (sample, cluster).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from deconvolution import paths

PS_META_COLS = [
    "site_id", "protein_id", "gene_symbol", "prot_description",
    "site_position", "motif",
]
PY_META_COLS = [
    "protein_id", "gene_symbol", "prot_description",
    "site_position", "motif", "gene_id",
]


@dataclass
class DeconvoluatedTrack:
    track: str                 # "st" | "py"
    meta: pd.DataFrame         # per-site metadata (n_sites rows)
    values: pd.DataFrame       # n_sites × (n_samples × n_clusters); MultiIndex columns: (sample, cluster)
    samples: list              # ordered, e.g. ["fe_2mo_AppP", ...]
    clusters: list             # ordered cluster names

    def site_id(self) -> pd.Series:
        if "site_id" in self.meta.columns:
            return self.meta["site_id"].astype(str)
        return (
            self.meta["protein_id"].astype(str)
            + "_"
            + self.meta["site_position"].astype(str)
        )


def _split_value_columns(value_cols: list[str]) -> list[tuple[str, str]]:
    """Split ``{sample}_{cluster}`` column names. Sample is the first 3
    underscore-separated tokens (e.g. ``ma_2mo_WTyp``); cluster is the rest."""
    out = []
    for col in value_cols:
        parts = col.split("_", 3)
        if len(parts) < 4:
            raise ValueError(f"Unexpected value column name: {col!r}")
        sample = "_".join(parts[:3])
        cluster = parts[3]
        out.append((sample, cluster))
    return out


def load_track(track: str) -> DeconvoluatedTrack:
    if track == "st":
        path = paths.PS_DECONVOLUTED_FILE
        meta_cols = PS_META_COLS
    elif track == "py":
        path = paths.PY_DECONVOLUTED_FILE
        meta_cols = PY_META_COLS
    else:
        raise ValueError(f"Unknown track: {track!r}")

    df = pd.read_csv(path, low_memory=False)

    # Drop unnamed index relics
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:")]

    # Identify value columns
    value_cols = [c for c in df.columns if c not in meta_cols]
    sample_cluster = _split_value_columns(value_cols)

    meta = df[meta_cols].copy()
    values = df[value_cols].copy()
    values.columns = pd.MultiIndex.from_tuples(sample_cluster, names=["sample", "cluster"])

    samples = sorted({s for s, _ in sample_cluster})
    clusters = sorted({c for _, c in sample_cluster})

    return DeconvoluatedTrack(
        track=track, meta=meta, values=values,
        samples=samples, clusters=clusters,
    )


def load_cluster_sizes() -> pd.DataFrame:
    """Return DataFrame indexed by cluster, columns = sample → cell count."""
    df = pd.read_csv(paths.CLUSTER_SIZE_FILE)
    df = df.rename(columns={df.columns[0]: "cluster"})
    df = df.set_index("cluster")
    return df


def load_cluster_mapping() -> pd.DataFrame:
    return pd.read_csv(paths.CLUSTER_MAPPING_FILE)


def parse_sample_metadata(samples: list[str]) -> pd.DataFrame:
    """Parse `{sex}_{timepoint}_{genotype}` sample names."""
    rows = []
    for s in samples:
        sex, timepoint, genotype = s.split("_")
        rows.append({
            "sample": s, "sex": sex, "timepoint": timepoint,
            "genotype": genotype,
        })
    return pd.DataFrame(rows)


def males_only(samples: list[str]) -> list[str]:
    return [s for s in samples if s.startswith("ma_")]


def safe_log2(x: np.ndarray) -> np.ndarray:
    """log2(x) with non-positive values mapped to NaN."""
    out = np.full_like(x, np.nan, dtype=float)
    mask = np.isfinite(x) & (x > 0)
    out[mask] = np.log2(x[mask])
    return out
