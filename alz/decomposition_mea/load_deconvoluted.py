"""Stage 1: load the CTM-native WMB-class decomposition outputs.

Each `*_wmb_decomposition.csv` file produced by ``build_wmb_decomposition.py``
has metadata columns followed by 24 groups × N WMB classes value columns
named ``{group}_{wmb_class}``. Reshape to a long, indexed structure: per-site
metadata + a (n_sites × n_value_cols) matrix with a tidy column index of
(sample, wmb_class).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alz.decomposition_mea import paths

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
    values: pd.DataFrame       # n_sites × (n_samples × n_classes); MultiIndex columns: (sample, wmb_class)
    samples: list              # ordered, e.g. ["fe_2mo_AppP", ...]
    clusters: list             # ordered WMB-class labels (kept attr name `clusters` for stable callers)

    def site_id(self) -> pd.Series:
        if "site_id" in self.meta.columns:
            return self.meta["site_id"].astype(str)
        return (
            self.meta["protein_id"].astype(str)
            + "_"
            + self.meta["site_position"].astype(str)
        )


def _split_value_columns(value_cols: list[str]) -> list[tuple[str, str]]:
    """Split ``{group}_{wmb_class}`` column names. Group is the first 3
    underscore-separated tokens (e.g. ``ma_2mo_WTyp``); the WMB class is the
    remainder and may contain spaces."""
    out = []
    for col in value_cols:
        parts = col.split("_", 3)
        if len(parts) < 4:
            raise ValueError(f"Unexpected value column name: {col!r}")
        sample = "_".join(parts[:3])
        wmb_class = parts[3]
        out.append((sample, wmb_class))
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
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:")]

    value_cols = [c for c in df.columns if c not in meta_cols]
    sample_class = _split_value_columns(value_cols)

    meta = df[meta_cols].copy()
    values = df[value_cols].copy()
    values.columns = pd.MultiIndex.from_tuples(
        sample_class, names=["sample", "wmb_class"]
    )

    samples = sorted({s for s, _ in sample_class})
    wmb_classes = sorted({c for _, c in sample_class})

    return DeconvoluatedTrack(
        track=track, meta=meta, values=values,
        samples=samples, clusters=wmb_classes,
    )


def load_wmb_class_sizes() -> pd.DataFrame:
    """Return DataFrame indexed by WMB class, columns = group → cell count."""
    df = pd.read_csv(paths.WMB_CLASS_SIZE_FILE)
    df = df.rename(columns={df.columns[0]: "wmb_class"})
    df = df.set_index("wmb_class")
    return df


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
