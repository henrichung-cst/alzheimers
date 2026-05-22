#!/usr/bin/env python3
"""Build per-cluster protein + phospho raw-value shards for the Incytr
Pathways "Evidence" tab.

Substrates (per-animal long-form parquets, canonical home):
  outputs/reports/decomposition/levy_t5/protein_per_cluster.parquet
      schema: (gene_symbol, animal_id, cluster, value, log2_value)
  outputs/reports/decomposition/levy_t5/phospho_per_cluster.parquet   (pS/pT)
      schema: (site_id, gene_symbol, animal_id, cluster, value, log2_value)
  outputs/reports/decomposition/levy_t5/phospho_per_cluster_pY.parquet
      schema: (site_id, gene_symbol, animal_id, cluster, value, log2_value)

Output: per-cluster parquet shards under ``audit_sources/omics_trace/`` with
uniform schema:
  layer        : str, one of {protein, phospho_ps, phospho_py}
  gene_symbol  : str
  site_id      : str or null (null for protein rows; row-index id for phospho)
  motif        : str or null (null for protein rows; 13-mer with central S/T/Y
                 at position 7, sourced from raw_phospho_normalized{,_pY}.csv)
  animal_id    : str  (raw animal identifier, e.g. "37_E50(L)_M_4mo_WT")
  sex          : str  (M or F)
  timepoint    : str  (2mo / 4mo / 6mo)
  genotype     : str  (WTyp / AppP / Ttau / ApTt)
  value        : float64  (forward-projected abundance — raw, not log)
  log2_value   : float64  (log2(value), NaN when value=0; see note below)

Note on log2_value vs LFC:
  ``log2_value`` stored here is log2(value) with no epsilon correction (NaN
  when value == 0). Item 3.4's JS-side LFC computation uses
  ``log2((D + 1e-5) / (W + 1e-5))`` (epsilon = 1e-5 matching
  ``Cal_foldchange``'s driver-passed correction). Do NOT use the stored
  ``log2_value`` directly to compute LFCs — it will disagree by 1e-5 at low
  abundance and produce NaN for zero rows.

Shard structure:
  One shard per cluster present in the incytr_pathways index.  A single shard
  contains rows for all three layers; the ``layer`` column discriminates.
  Shards are gated to pathway clusters exactly (same hard-fail pattern as
  ``build_transcript_trace.py``).

Pathway-cluster coverage hard-fail:
  Every cluster name referenced by ``edge_slices/incytr_pathways/index.json``
  must be present in all three substrate parquets, or the build aborts with a
  clear error. This is non-negotiable per the epic's design.

Animal-level metadata decoding:
  ``animal_id`` encodes sex / timepoint / genotype:
      ``<n>_<lab>_<sex>_<age>_<geno>``  e.g. "37_E50(L)_M_4mo_WT"
  Parsing + short→long genotype normalization is centralized in
  ``config.parse_animal_id``; animals whose ``animal_id`` does not match
  the regex are silently dropped (they are the 72 − 33 = 39 females +
  outliers not used by the Incytr pipeline).

Schema version: 1.  Bump OMICS_TRACE_SCHEMA_VERSION in alz/viewer/paths.py on
any schema change; the viewer rebuild will invalidate existing shards.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # alz/

import config  # noqa: E402

from integration.pair_to_receiver_cache import _sanitize_celltype  # noqa: E402
from viewer.paths import (  # noqa: E402
    EDGE_SLICES_INCYTR_PATHWAYS_DIR,
    OMICS_TRACE_DIR,
    OMICS_TRACE_INDEX,
    OMICS_TRACE_SCHEMA_VERSION,
    TRANSCRIPT_TRACE_SAMPLEKEY,
    UNIFIED_VIEWER_DIR,
)

# ---------------------------------------------------------------------------
# Substrate paths (canonical — per the Phase 2 layout).
# ---------------------------------------------------------------------------
_DEC_DIR = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "decomposition", "levy_t5"
)
PROTEIN_PARQUET = os.path.join(_DEC_DIR, "protein_per_cluster.parquet")
PHOSPHO_PS_PARQUET = os.path.join(_DEC_DIR, "phospho_per_cluster.parquet")
PHOSPHO_PY_PARQUET = os.path.join(_DEC_DIR, "phospho_per_cluster_pY.parquet")

# Site-level motif (13-mer with central S/T/Y at position 7) lives only in the
# upstream IRS-normalized phospho CSV; the per-cluster decomposition drops it.
# We rejoin on site_id at shard build time so the Evidence tab popover can
# label rows by their phosphosite motif instead of opaque integer ids.
_KIN_DIR = os.path.join(config.REPO_ROOT, "outputs", "reports", "kinase_attribution")
PHOSPHO_PS_MOTIF_CSV = os.path.join(_KIN_DIR, "raw_phospho_normalized.csv")
PHOSPHO_PY_MOTIF_CSV = os.path.join(_KIN_DIR, "raw_phospho_normalized_pY.csv")


def _load_site_motif(csv_path: str) -> pd.DataFrame:
    """Return DataFrame with columns (site_id:str, motif:str)."""
    df = pd.read_csv(csv_path, usecols=["site_id", "motif"])
    df["site_id"] = df["site_id"].astype(str)
    df["motif"] = df["motif"].astype(str)
    return df.drop_duplicates(subset=["site_id"])

# ---------------------------------------------------------------------------
# Animal-id decoding — delegates to config.parse_animal_id (long-form genotype).
# ---------------------------------------------------------------------------

def _parse_animal(animal_id: str) -> tuple[str, str, str] | None:
    """Return (sex, timepoint, genotype) for a valid animal_id, else None."""
    parsed = config.parse_animal_id(animal_id)
    if parsed is None:
        return None
    return parsed["sex"], parsed["timepoint"], parsed["genotype"]


def _add_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse animal_id into sex/timepoint/genotype; drop unparseable rows."""
    parsed = df["animal_id"].map(_parse_animal)
    mask = parsed.notna()
    df = df[mask].copy()
    parsed = parsed[mask]
    df["sex"] = [t[0] for t in parsed]
    df["timepoint"] = [t[1] for t in parsed]
    df["genotype"] = [t[2] for t in parsed]
    return df


# ---------------------------------------------------------------------------
# Pathway-cluster discovery (same logic as build_transcript_trace.py).
# ---------------------------------------------------------------------------

def _load_pathway_clusters(index_path: str) -> set[str]:
    """Read incytr_pathways index.json → union of unsanitized sender/receiver."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"incytr_pathways index missing at {index_path}; build the "
            f"pathway shards before the omics-trace step."
        )
    with open(index_path) as f:
        idx = json.load(f)
    present = idx.get("present") or []
    clusters: set[str] = set()
    for pair in present:
        if len(pair) >= 2:
            clusters.add(pair[0])
            clusters.add(pair[1])
    return clusters


# ---------------------------------------------------------------------------
# Substrate loaders.
# ---------------------------------------------------------------------------

def _load_protein(cluster_filter: set[str] | None = None) -> pd.DataFrame:
    """Load protein substrate; add layer='protein', site_id=None, motif=None."""
    print(f"  omics_trace: reading {PROTEIN_PARQUET}", flush=True)
    df = pq.ParquetFile(PROTEIN_PARQUET).read(
        columns=["gene_symbol", "animal_id", "cluster", "value", "log2_value"]
    ).to_pandas()
    if cluster_filter is not None:
        df = df[df["cluster"].isin(cluster_filter)]
    df["layer"] = "protein"
    df["site_id"] = None
    df["motif"] = None
    return df


def _load_phospho(parquet_path: str, motif_csv: str, layer: str,
                  cluster_filter: set[str] | None = None) -> pd.DataFrame:
    """Load phospho substrate and join motif from the upstream IRS-normalized
    CSV. site_id→motif is a 1:1 lookup (~13K rows for pS, ~1.5K for pY)."""
    print(f"  omics_trace: reading {parquet_path}", flush=True)
    df = pq.ParquetFile(parquet_path).read(
        columns=["site_id", "gene_symbol", "animal_id", "cluster", "value",
                 "log2_value"]
    ).to_pandas()
    if cluster_filter is not None:
        df = df[df["cluster"].isin(cluster_filter)]
    df["site_id"] = df["site_id"].astype(str)
    motif = _load_site_motif(motif_csv)
    df = df.merge(motif, on="site_id", how="left")
    n_missing = int(df["motif"].isna().sum())
    if n_missing:
        miss_sites = df.loc[df["motif"].isna(), "site_id"].unique()[:5]
        raise ValueError(
            f"omics_trace: {n_missing} {layer} rows have no motif lookup "
            f"(examples: {list(miss_sites)}). Substrate drift between "
            f"{parquet_path} and {motif_csv}."
        )
    df["layer"] = layer
    return df


def _load_phospho_ps(cluster_filter: set[str] | None = None) -> pd.DataFrame:
    return _load_phospho(PHOSPHO_PS_PARQUET, PHOSPHO_PS_MOTIF_CSV,
                         "phospho_ps", cluster_filter)


def _load_phospho_py(cluster_filter: set[str] | None = None) -> pd.DataFrame:
    return _load_phospho(PHOSPHO_PY_PARQUET, PHOSPHO_PY_MOTIF_CSV,
                         "phospho_py", cluster_filter)


# ---------------------------------------------------------------------------
# Build.
# ---------------------------------------------------------------------------

_SHARD_COLS = [
    "layer", "gene_symbol", "site_id", "motif", "animal_id",
    "sex", "timepoint", "genotype", "value", "log2_value",
]


def build(force: bool = False) -> dict:
    """Build omics_trace shards. Returns the index dict."""
    if not force and os.path.exists(OMICS_TRACE_INDEX):
        with open(OMICS_TRACE_INDEX) as f:
            existing = json.load(f)
        if existing.get("omics_schema_version") == OMICS_TRACE_SCHEMA_VERSION:
            return existing

    if os.path.exists(OMICS_TRACE_DIR):
        shutil.rmtree(OMICS_TRACE_DIR)
    os.makedirs(OMICS_TRACE_DIR, exist_ok=True)

    t0 = time.time()

    # --- Substrate existence checks ---
    for path, label in [
        (PROTEIN_PARQUET, "protein_per_cluster"),
        (PHOSPHO_PS_PARQUET, "phospho_per_cluster (pS/pT)"),
        (PHOSPHO_PY_PARQUET, "phospho_per_cluster_pY"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"omics_trace substrate missing: {path} ({label}). "
                f"Run alz/decomposition_mea/build_celltype_decomposition.py "
                f"before rebuilding the viewer."
            )

    # --- Pathway-cluster discovery ---
    incytr_idx = os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, "index.json")
    pathway_clusters = _load_pathway_clusters(incytr_idx)
    print(f"  omics_trace: {len(pathway_clusters)} pathway clusters", flush=True)

    # --- Load substrates (filtered to pathway clusters for memory) ---
    pr = _load_protein(cluster_filter=pathway_clusters)
    ps = _load_phospho_ps(cluster_filter=pathway_clusters)
    py = _load_phospho_py(cluster_filter=pathway_clusters)

    # --- Coverage check: every substrate must contain every pathway cluster ---
    for df, label in [(pr, "protein"), (ps, "phospho_ps"), (py, "phospho_py")]:
        present_in_layer = set(df["cluster"].unique())
        missing = pathway_clusters - present_in_layer
        if missing:
            raise ValueError(
                f"omics_trace: pathway index references cluster(s) "
                f"{sorted(missing)} absent from {label} substrate. "
                f"Substrate drift — regenerate "
                f"build_celltype_decomposition.py before rebuilding the viewer."
            )

    # --- Add metadata columns (sex / timepoint / genotype from animal_id) ---
    pr = _add_metadata_columns(pr)
    ps = _add_metadata_columns(ps)
    py = _add_metadata_columns(py)

    # --- Concatenate all layers ---
    combined = pd.concat([pr, ps, py], ignore_index=True)
    combined["cluster"] = combined["cluster"].astype(str)

    n_rows = len(combined)
    print(f"  omics_trace: combined {n_rows:,} rows across 3 layers", flush=True)

    # --- Per-cluster shard write ---
    shards_written: dict[str, str] = {}
    for cluster, sub in combined.groupby("cluster", sort=True):
        if cluster not in pathway_clusters:
            continue
        out = sub[_SHARD_COLS].reset_index(drop=True)
        slug = _sanitize_celltype(cluster)
        out_path = os.path.join(OMICS_TRACE_DIR, f"{slug}.parquet")
        out.to_parquet(out_path, index=False, compression="zstd")
        shards_written[cluster] = os.path.relpath(out_path, UNIFIED_VIEWER_DIR)

    # --- Coverage: every pathway cluster must have a shard ---
    missing_shards = pathway_clusters - set(shards_written)
    if missing_shards:
        raise RuntimeError(
            f"omics_trace: {len(missing_shards)} pathway cluster(s) have no "
            f"shard after write: {sorted(missing_shards)}. "
            f"This is a logic error — please report."
        )

    # --- Index ---
    index = {
        "omics_schema_version": OMICS_TRACE_SCHEMA_VERSION,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "label": "Omics raw-value trace (protein + phospho pS/pT + pY)",
        "source_protein": os.path.relpath(PROTEIN_PARQUET, config.REPO_ROOT),
        "source_phospho_ps": os.path.relpath(PHOSPHO_PS_PARQUET, config.REPO_ROOT),
        "source_phospho_py": os.path.relpath(PHOSPHO_PY_PARQUET, config.REPO_ROOT),
        "layers": ["protein", "phospho_ps", "phospho_py"],
        "log2_value_note": (
            "log2_value = log2(value), NaN when value == 0. "
            "For LFC computation use log2((D + 1e-5) / (W + 1e-5)) from raw "
            "value column — not from log2_value — to match Cal_foldchange "
            "(correction=1e-5 passed by incytr_commandline.R)."
        ),
        "aggregation_note": (
            "Phospho site-to-gene aggregation mirrors incytr_commandline.R: "
            "median across males per (site, group), then arithmetic mean "
            "across sites per gene. Rows here are per-(site, animal) — "
            "the JS Evidence tab applies the same two-step aggregation."
        ),
        "sanitize_rule": "replace('/', '-'); replace(' ', '_')",
        "filename_template": "{cluster}.parquet",
        "relative_path": os.path.relpath(OMICS_TRACE_DIR, UNIFIED_VIEWER_DIR),
        "clusters": sorted(shards_written.keys()),
        "shard_files": shards_written,
        "n_shards": len(shards_written),
    }
    with open(OMICS_TRACE_INDEX, "w") as f:
        json.dump(index, f, indent=2)
    print(
        f"  omics_trace: wrote {len(shards_written)} cluster shards in "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if index is current.")
    args = ap.parse_args()
    build(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
