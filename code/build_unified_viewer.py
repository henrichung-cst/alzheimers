#!/usr/bin/env python3
"""Unified viewer builder: single entry point for kinase + pathway views.

Phase 3 adds the HTML shell (template + CSS + JS Store + Overview tab).
Phase 2 artifacts produced:

  - kinase_backbone_edges_sig.parquet  — edges filtered to backbones that
    pass both null permutation tests (significant_both). Sidecar artifact
    (not embedded in HTML; fetched by future tabs only if needed).
  - unified_viewer.payload.json (+ .gz) — columnar JSON payload with
    stable integer IDs for kinases, celltypes, and backbones. Embedded
    inline in the HTML via a <script type="application/json"> tag so the
    viewer is a single-file deliverable usable over file://.

The full 7.14 GB / 2.23B-row edge parquet is streamed via
ParquetFile.iter_batches — it is never materialized in memory.

Usage:
    python code/build_unified_viewer.py              # build + html (default)
    python code/build_unified_viewer.py --summary    # input row counts
    python code/build_unified_viewer.py --sidecar    # sig parquet only
    python code/build_unified_viewer.py --payload    # JSON only (needs sidecar)
    python code/build_unified_viewer.py --build      # sidecar + payload
    python code/build_unified_viewer.py --html       # write HTML (needs payload)
    python code/build_unified_viewer.py --validate   # write report md
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "integration"))

import config  # noqa: E402
import config_integration as icfg  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AGGREGATION_DIR = os.path.join(icfg.FACTORIAL_ALL_PAIRS_DIR, "aggregation")
EDGES_PARQUET = os.path.join(AGGREGATION_DIR, "kinase_backbone_edges.parquet")
EDGE_META_JSON = os.path.join(AGGREGATION_DIR, "edge_index_metadata.json")
BACKBONE_SIG_CSV = os.path.join(AGGREGATION_DIR, "backbone_significant_both_nulls.csv")
BACKBONE_REC_CSV = os.path.join(AGGREGATION_DIR, "backbone_recurrence_by_contrast.csv")

UNIFIED_VIEWER_OUTPUT_DIR = os.path.join(config.REPO_ROOT, "outputs", "reports")
UNIFIED_VIEWER_DIR = os.path.join(UNIFIED_VIEWER_OUTPUT_DIR, "unified_viewer")
PAYLOAD_JSON = os.path.join(UNIFIED_VIEWER_DIR, "unified_viewer.payload.json")
PAYLOAD_JSON_GZ = PAYLOAD_JSON + ".gz"
SIDECAR_PARQUET = os.path.join(UNIFIED_VIEWER_OUTPUT_DIR, "kinase_backbone_edges_sig.parquet")
UNIFIED_VIEWER_HTML = os.path.join(UNIFIED_VIEWER_DIR, "index.html")
PER_KINASE_SUMMARY = os.path.join(UNIFIED_VIEWER_DIR, "edge_summaries",
                                  "per_kinase_summary.parquet")
PER_BACKBONE_SUMMARY = os.path.join(UNIFIED_VIEWER_DIR, "edge_summaries",
                                    "per_backbone_summary.parquet")
EDGE_SLICES_KINASE_DIR = os.path.join(UNIFIED_VIEWER_DIR, "edge_slices", "kinase")
EDGE_SLICES_BACKBONE_DIR = os.path.join(UNIFIED_VIEWER_DIR, "edge_slices", "backbone")
REPORT_MD = os.path.join(config.REPO_ROOT, "pipeline_notes", "phase2_payload_report.md")

SCHEMA_VERSION = 1
TOP_N_KINASES = 5                  # per (backbone, contrast) preview in JSON
EDGE_STREAM_BATCH = 1_000_000      # rows per pyarrow batch; caps RAM

TISSUE_CATEGORIES = {
    "Excitatory": ["L2/3 IT", "L4 IT", "L5 ET", "L5 IT", "L5/6 NP",
                   "L6 CT", "L6 IT", "L6b"],
    "Inhibitory": ["Chandelier", "Lamp5", "Lamp5 Lhx6", "Pvalb", "Sncg",
                   "Sst", "Sst Chodl", "Vip"],
    "Non-neuronal": ["Astrocyte", "Endothelial", "Microglia-PVM", "OPC",
                     "Oligodendrocyte", "VLMC"],
}
RECEIVER_TO_TISSUE = {r: t for t, rs in TISSUE_CATEGORIES.items() for r in rs}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class UnifiedData:
    """All non-edge inputs + a lazy handle to the full edge parquet."""

    # Kinase-side
    kinase_activity: pd.DataFrame
    celltype_evidence: pd.DataFrame
    kinase_hypothesis: pd.DataFrame
    celltype_profiles: pd.DataFrame
    mea_stoichiometry: pd.DataFrame

    # Pathway-side
    backbone_sig: pd.DataFrame
    backbone_recurrence: pd.DataFrame
    unified_attribution: pd.DataFrame

    # Edge parquet — metadata only, streamed at use time
    edges_pf: pq.ParquetFile
    edge_metadata: dict = field(default_factory=dict)

    def summary(self) -> dict:
        md = self.edge_metadata
        return {
            "kinases": len(md.get("kinases", [])),
            "celltypes": len(md.get("celltypes", [])),
            "contrasts": len(md.get("contrasts", [])),
            "backbones": md.get("backbones_n", 0),
            "edges": self.edges_pf.metadata.num_rows,
            "kinase_activity_rows": len(self.kinase_activity),
            "celltype_evidence_rows": len(self.celltype_evidence),
            "kinase_hypothesis_rows": len(self.kinase_hypothesis),
            "celltype_profiles_rows": len(self.celltype_profiles),
            "mea_rows": len(self.mea_stoichiometry),
            "backbone_sig_rows": len(self.backbone_sig),
            "backbone_recurrence_rows": len(self.backbone_recurrence),
            "unified_attribution_rows": len(self.unified_attribution),
        }


def load_all_data() -> UnifiedData:
    """Load every non-edge input; open the edge parquet as a lazy handle."""
    ar_dir = config.ATTRIBUTION_RECOVERY_OUTPUT_DIR
    ka_dir = config.KINASE_ATTRIBUTION_OUTPUT_DIR

    kinase_activity = pd.read_csv(os.path.join(ar_dir, "kinase_activity_matrix.csv"))
    celltype_evidence = pd.read_csv(os.path.join(ar_dir, "celltype_evidence_table.csv"))
    kinase_hypothesis = pd.read_csv(os.path.join(ar_dir, "kinase_hypothesis_table.csv"))
    celltype_profiles = pd.read_csv(os.path.join(ar_dir, "celltype_kinase_profiles.csv"))
    mea = pd.read_csv(os.path.join(ka_dir, "mea_stoichiometry.csv"),
                      usecols=["kinase", "NES", "FDR", "contrast"])

    backbone_sig = pd.read_csv(BACKBONE_SIG_CSV)
    backbone_recurrence = pd.read_csv(
        BACKBONE_REC_CSV,
        usecols=[
            "contrast", "receiver", "Receptor", "EM", "Target",
            "n_senders", "n_senders_significant",
            "mean_tpds", "max_abs_tpds", "sender_list",
        ],
    )
    unified_attribution = pd.read_csv(
        icfg.UNIFIED_ATTRIBUTION_CSV,
        usecols=[
            "kinase", "gene_symbol", "contrast", "cell_type",
            "NES", "FDR", "combined_score", "combined_confidence",
        ],
    )

    edges_pf = pq.ParquetFile(EDGES_PARQUET)
    with open(EDGE_META_JSON) as f:
        edge_metadata = json.load(f)

    return UnifiedData(
        kinase_activity=kinase_activity,
        celltype_evidence=celltype_evidence,
        kinase_hypothesis=kinase_hypothesis,
        celltype_profiles=celltype_profiles,
        mea_stoichiometry=mea,
        backbone_sig=backbone_sig,
        backbone_recurrence=backbone_recurrence,
        unified_attribution=unified_attribution,
        edges_pf=edges_pf,
        edge_metadata=edge_metadata,
    )


# ---------------------------------------------------------------------------
# Vocab helpers
# ---------------------------------------------------------------------------

BACKBONE_VOCAB_CACHE = os.path.join(AGGREGATION_DIR, "backbone_vocab.parquet")


def build_backbone_index(recurrence: pd.DataFrame | None = None) -> pd.DataFrame:
    """Canonical backbone vocabulary matching the edge parquet's IDs.

    The edge parquet at `kinase_backbone_edges.parquet` was built by
    `code/integration/adapters/build_edge_index.py`, whose vocab pass scans
    every pair directory's `kinase_routes.parquet` and takes the backbones
    from `kinase_support_scores.csv` filtered to routes-used paths. This
    produces 832,289 unique (receiver, Receptor, EM, Target) tuples — a
    superset of `backbone_recurrence_by_contrast.csv` (516,583 unique),
    which applies a stricter "recurrence across senders" filter.

    To align IDs, we reconstruct the edge parquet's vocab here. Cached as
    `backbone_vocab.parquet` in the aggregation dir after first build.
    """
    if os.path.exists(BACKBONE_VOCAB_CACHE):
        return pd.read_parquet(BACKBONE_VOCAB_CACHE)

    adapters_dir = os.path.join(HERE, "integration", "adapters")
    if adapters_dir not in sys.path:
        sys.path.insert(0, adapters_dir)
    from build_edge_index import discover_pair_dirs, _collect_vocab  # noqa: E402

    pair_dirs = discover_pair_dirs()
    pairs = [(n, d) for (n, d) in pair_dirs
             if os.path.exists(os.path.join(d, "kinase_routes.parquet"))]
    print(f"  reconstructing backbone vocab from {len(pairs)} pair dirs "
          f"(one-time; ~5 min)...", flush=True)
    _, backbones, _ = _collect_vocab(pairs)
    backbones.to_parquet(BACKBONE_VOCAB_CACHE, index=False)
    print(f"  cached -> {BACKBONE_VOCAB_CACHE}", flush=True)
    return backbones


def compute_sig_sets(data: UnifiedData,
                     bb_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique sig backbone_ids, sorted packed (backbone_id<<8)|contrast_id array).

    The sig-both gate is per (backbone, contrast), not per backbone.
    """
    contrasts = data.edge_metadata["contrasts"]
    contrast_to_id = {c: i for i, c in enumerate(contrasts)}

    sig = data.backbone_sig.merge(
        bb_index, on=["receiver", "Receptor", "EM", "Target"],
        how="inner", validate="many_to_one",
    )
    sig["contrast_id"] = sig["contrast"].map(contrast_to_id).astype("Int16")
    sig = sig.dropna(subset=["contrast_id"])

    bb = sig["backbone_id"].astype(np.int64).to_numpy()
    cn = sig["contrast_id"].astype(np.int64).to_numpy()
    packed_sorted = np.sort((bb << 8) | cn)
    sig_bb_ids = np.sort(np.unique(bb).astype(np.uint32))
    return sig_bb_ids, packed_sorted


# ---------------------------------------------------------------------------
# Step B — stream-filter full edge parquet into sig sidecar
# ---------------------------------------------------------------------------

def write_sig_sidecar(data: UnifiedData,
                      bb_index: pd.DataFrame,
                      progress: bool = True) -> dict:
    """Stream the full edge parquet, keep only rows whose (backbone_id,
    contrast_id) pair is in the sig_pair_set, write to SIDECAR_PARQUET.

    Memory bound: one EDGE_STREAM_BATCH-row batch at a time (~20 MB).
    """
    sig_bb_ids, packed_sorted = compute_sig_sets(data, bb_index)
    if len(packed_sorted) == 0:
        raise RuntimeError("No significant (backbone, contrast) pairs found")

    sig_bb_arr = pa.array(sig_bb_ids, type=pa.uint32())

    os.makedirs(UNIFIED_VIEWER_OUTPUT_DIR, exist_ok=True)
    pf = data.edges_pf
    writer = None
    total_rows = pf.metadata.num_rows
    seen = 0
    kept = 0
    t0 = time.monotonic()

    def _log():
        if progress and seen % (EDGE_STREAM_BATCH * 50) == 0:
            rate = seen / max(time.monotonic() - t0, 1e-6) / 1e6
            print(f"  stream {seen:>14,}/{total_rows:,} "
                  f"({rate:.1f} M/s) kept={kept:,}", flush=True)

    try:
        for batch in pf.iter_batches(batch_size=EDGE_STREAM_BATCH):
            seen += batch.num_rows

            mask1 = pc.is_in(batch["backbone_id"], value_set=sig_bb_arr)
            batch = batch.filter(mask1)
            if batch.num_rows == 0:
                _log()
                continue

            bb_np = batch["backbone_id"].to_numpy().astype(np.int64)
            cn_np = batch["contrast_id"].to_numpy().astype(np.int64)
            packed = (bb_np << 8) | cn_np
            idx = np.clip(np.searchsorted(packed_sorted, packed),
                          0, len(packed_sorted) - 1)
            keep_mask = packed_sorted[idx] == packed
            if not keep_mask.any():
                _log()
                continue
            batch = batch.filter(pa.array(keep_mask))

            if writer is None:
                writer = pq.ParquetWriter(
                    SIDECAR_PARQUET, batch.schema,
                    compression="zstd", compression_level=3,
                )
            writer.write_batch(batch)
            kept += batch.num_rows
            _log()
    finally:
        if writer is not None:
            writer.close()

    dt = time.monotonic() - t0
    print(f"  sidecar: {kept:,} rows / {total_rows:,} scanned "
          f"in {dt:.1f}s -> {SIDECAR_PARQUET}", flush=True)
    return {
        "sidecar_rows": kept,
        "scanned_rows": total_rows,
        "sig_backbones": int(len(sig_bb_ids)),
        "sig_pairs": int(len(packed_sorted)),
        "stream_seconds": round(dt, 1),
    }


# ---------------------------------------------------------------------------
# Step C — JSON payload
# ---------------------------------------------------------------------------

def _sanitize(obj: Any, decimals: int = 4):
    """JSON-safe: NaN/Inf -> None, numpy -> native, floats rounded."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _sanitize(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, decimals) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist(), decimals)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return None
        return round(x, decimals)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NA:
        return None
    return obj


def _build_kinases_slice(data: UnifiedData) -> dict:
    """Columnar kinases table. IDs follow edge_metadata['kinases'] ordering."""
    kinases = data.edge_metadata["kinases"]
    kid = {k: i for i, k in enumerate(kinases)}

    ka = data.kinase_activity.set_index("kinase")
    hyp = data.kinase_hypothesis.set_index("kinase")
    contrasts = data.edge_metadata["contrasts"]

    cols: dict[str, list] = {
        "id": [], "name": [], "gene_symbol": [],
        "trajectory": [], "peak_contrast": [], "peak_NES": [],
        "n_sig_contrasts": [],
        "top_celltype_1": [], "top_celltype_2": [], "top_celltype_3": [],
        "n_celltype_candidates": [], "has_high_conf_attribution": [],
    }
    for c in contrasts:
        cols[f"NES_{c}"] = []
        cols[f"FDR_{c}"] = []

    for k in kinases:
        cols["id"].append(kid[k])
        cols["name"].append(k)
        ka_row = ka.loc[k] if k in ka.index else None
        hyp_row = hyp.loc[k] if k in hyp.index else None

        def _get(r, col, default=None):
            if r is None or col not in r.index:
                return default
            v = r[col]
            return default if pd.isna(v) else v

        cols["gene_symbol"].append(_get(ka_row, "gene_symbol", ""))
        cols["trajectory"].append(_get(ka_row, "trajectory_label", ""))
        cols["peak_contrast"].append(_get(ka_row, "peak_contrast", ""))
        cols["peak_NES"].append(_get(ka_row, "peak_NES"))
        cols["n_sig_contrasts"].append(_get(ka_row, "n_sig_contrasts", 0))
        cols["top_celltype_1"].append(_get(hyp_row, "top_celltype_1", ""))
        cols["top_celltype_2"].append(_get(hyp_row, "top_celltype_2", ""))
        cols["top_celltype_3"].append(_get(hyp_row, "top_celltype_3", ""))
        cols["n_celltype_candidates"].append(_get(hyp_row, "n_celltype_candidates", 0))
        cols["has_high_conf_attribution"].append(
            bool(_get(hyp_row, "has_high_conf_attribution", False))
        )
        for c in contrasts:
            cols[f"NES_{c}"].append(_get(ka_row, f"{c}_NES"))
            cols[f"FDR_{c}"].append(_get(ka_row, f"{c}_FDR"))
    return cols


def _build_celltypes_slice(data: UnifiedData) -> dict:
    celltypes = data.edge_metadata["celltypes"]
    return {
        "id": list(range(len(celltypes))),
        "name": list(celltypes),
        "tissue_category": [RECEIVER_TO_TISSUE.get(c, "Other") for c in celltypes],
    }


def _encode_sender_mask(sender_order: list[str], sender_list: str) -> int:
    idx = {s: i for i, s in enumerate(sender_order)}
    mask = 0
    if not sender_list:
        return mask
    for s in sender_list.split(","):
        s = s.strip()
        if s in idx:
            mask |= (1 << idx[s])
    return mask


def _build_backbones_slice(data: UnifiedData,
                           bb_index: pd.DataFrame) -> tuple[dict, list[str]]:
    """Pivot backbone_recurrence to one row per unique backbone, with
    per-contrast mean_tpds columns and a significant_both mask."""
    contrasts = data.edge_metadata["contrasts"]
    cn_to_id = {c: i for i, c in enumerate(contrasts)}

    # Sender vocabulary (same rule as build_pathway_viewer.py)
    all_senders: set[str] = set()
    for sl in data.backbone_recurrence["sender_list"].dropna():
        for s in str(sl).split(","):
            s = s.strip()
            if s:
                all_senders.add(s)
    sender_order = sorted(all_senders)

    rec = data.backbone_recurrence.copy()
    before = len(rec)
    rec = rec.merge(bb_index, on=["receiver", "Receptor", "EM", "Target"],
                    how="inner", validate="many_to_one")
    dropped = before - len(rec)
    if dropped:
        # Recurrence rows whose backbone has no kinase edges (absent from the
        # Phase 1 edge parquet vocab). We can't surface these in a kinase-aware
        # viewer — no ID to point to — so they're dropped here. Sig backbones
        # must be in the edge vocab (sig is derived from kinase edges), so
        # dropping these does not lose any sig-both data.
        print(f"  dropped {dropped:,}/{before:,} recurrence rows with no "
              f"kinase edges", flush=True)
    rec["backbone_id"] = rec["backbone_id"].astype(np.uint32)

    # Significant-both gate merged in per (contrast, backbone_id)
    sig_bb = data.backbone_sig.merge(
        bb_index, on=["receiver", "Receptor", "EM", "Target"],
        how="inner", validate="many_to_one",
    )
    sig_keys = set(zip(sig_bb["contrast"], sig_bb["backbone_id"].astype(int)))

    # Collapse to one row per backbone, with per-contrast values flattened.
    base = (
        rec[["backbone_id", "receiver", "Receptor", "EM", "Target"]]
        .drop_duplicates("backbone_id")
        .sort_values("backbone_id")
        .reset_index(drop=True)
    )
    celltype_to_id = {c: i for i, c in enumerate(data.edge_metadata["celltypes"])}
    rid = base["receiver"].map(celltype_to_id)
    if rid.isna().any():
        missing = base[rid.isna()]["receiver"].unique().tolist()
        raise RuntimeError(f"Receivers missing from celltype vocab: {missing}")
    base["receiver_id"] = rid.astype(np.uint8)

    cols: dict[str, list] = {
        "id": base["backbone_id"].tolist(),
        "receiver_id": base["receiver_id"].tolist(),
        "Receptor": base["Receptor"].tolist(),
        "EM": base["EM"].tolist(),
        "Target": base["Target"].tolist(),
    }
    first_sender_per_bb = (
        rec.dropna(subset=["sender_list"])
        .drop_duplicates("backbone_id")[["backbone_id", "sender_list",
                                         "n_senders", "n_senders_significant"]]
        .set_index("backbone_id")
    )
    aligned = first_sender_per_bb.reindex(base["backbone_id"])
    sender_lists = aligned["sender_list"].fillna("").astype(str).tolist()
    cols["sender_mask"] = [_encode_sender_mask(sender_order, sl)
                           for sl in sender_lists]
    cols["n_senders"] = aligned["n_senders"].fillna(0).astype(int).tolist()
    cols["n_senders_significant"] = (
        aligned["n_senders_significant"].fillna(0).astype(int).tolist()
    )

    def _flatten_pivot(df: pd.DataFrame, value_col: str) -> None:
        piv = (df.pivot_table(index="backbone_id", columns="contrast",
                              values=value_col, aggfunc="first")
                 .reindex(index=base["backbone_id"], columns=contrasts))
        for c in contrasts:
            cols[f"{value_col}_{c}"] = piv[c].astype(object).where(
                piv[c].notna(), None
            ).tolist()

    _flatten_pivot(rec, "mean_tpds")
    cols["max_abs_tpds"] = base["backbone_id"].map(
        rec.groupby("backbone_id")["max_abs_tpds"].max()
    ).astype(float).tolist()
    _flatten_pivot(sig_bb, "observed_score")

    # significant_both encoded as a 9-bit integer, one bit per contrast.
    sig_by_bb: dict[int, int] = {}
    for c, bid in sig_keys:
        ci = cn_to_id.get(c)
        if ci is None:
            continue
        sig_by_bb[int(bid)] = sig_by_bb.get(int(bid), 0) | (1 << ci)
    cols["significant_both_mask"] = [
        sig_by_bb.get(int(bid), 0) for bid in base["backbone_id"]
    ]

    return cols, sender_order


def _build_overview_slice(data: UnifiedData) -> dict:
    """Pre-aggregate receiver × contrast counts for the Overview tab.

    Keyed by "{contrast}|{receiver}"; empty cells are omitted (JS treats
    missing as zero).
    """
    rec = data.backbone_recurrence
    overview: dict[str, dict] = {}
    for (c, r), g in rec.groupby(["contrast", "receiver"], sort=False):
        tpds = g["mean_tpds"].to_numpy()
        tpds_fin = tpds[np.isfinite(tpds)]
        mean_tpds = float(tpds_fin.mean()) if tpds_fin.size else 0.0
        overview[f"{c}|{r}"] = {
            "n": int(len(g)),
            "n_up": int((tpds > 0).sum()),
            "n_down": int((tpds < 0).sum()),
            "mean_tpds": round(mean_tpds, 4),
        }
    return overview


def _extract_pi0(backbone_sig: pd.DataFrame, contrasts: list[str]) -> dict:
    out = {}
    for c in contrasts:
        sub = backbone_sig[backbone_sig["contrast"] == c]
        if len(sub) > 0:
            out[c] = {
                "pi0_null1": round(float(sub["pi0_null1"].iloc[0]), 4),
                "pi0_null2": round(float(sub["pi0_null2"].iloc[0]), 4),
            }
    return out


def build_payload(data: UnifiedData) -> dict:
    """Assemble the full JSON payload (no edges — that's the sidecar)."""
    from kinase_library.utils._global_vars import family_colors as KL_FAMILY_COLORS
    from kinase_library.modules import data as kl_data

    bb_index = build_backbone_index(data.backbone_recurrence)

    kinases_slice = _build_kinases_slice(data)
    celltypes_slice = _build_celltypes_slice(data)
    backbones_slice, sender_order = _build_backbones_slice(data, bb_index)

    # Kinase family map
    try:
        fam = kl_data.get_kinase_family(data.edge_metadata["kinases"]).to_dict()
    except Exception as e:
        print(f"  (warn) family resolve failed: {e}; using empty map", flush=True)
        fam = {}

    contrasts = data.edge_metadata["contrasts"]

    # Tier-1 edge summary (embedded). Tier-2 slices are loaded lazily by the
    # browser from edge_slices/ — not embedded.
    if not os.path.exists(PER_KINASE_SUMMARY):
        raise SystemExit(
            f"per_kinase_summary missing: {PER_KINASE_SUMMARY}. "
            f"Run: pixi run python code/integration/adapters/build_edge_shards.py"
        )
    pk_summary_tbl = pq.read_table(PER_KINASE_SUMMARY)
    per_kinase_summary = {name: pk_summary_tbl[name].to_pylist()
                          for name in pk_summary_tbl.column_names}

    kinase_index_path = os.path.join(EDGE_SLICES_KINASE_DIR, "index.json")
    backbone_index_path = os.path.join(EDGE_SLICES_BACKBONE_DIR, "index.json")
    with open(kinase_index_path) as f:
        kinase_slice_index = json.load(f)
    with open(backbone_index_path) as f:
        backbone_slice_index = json.load(f)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "contrasts": contrasts,
        "diseaseGroups": list(config.DISEASE_GROUPS),
        "timepoints": list(config.TIMEPOINTS),
        "diseaseColors": dict(config.DISEASE_COLORS),
        "tissueOrder": list(config.TISSUE_ORDER),
        "subclassToTissue": dict(config.SUBCLASS_TO_TISSUE_CATEGORY),
        "tissueCategories": TISSUE_CATEGORIES,
        "receiverToTissue": RECEIVER_TO_TISSUE,
        "senderOrder": sender_order,
        "fdrThreshDefault": config.MEA_FDR_THRESH,
        "specificityHigh": round(config.SPECIFICITY_HIGH, 4),
        "specificityLow": round(config.SPECIFICITY_LOW, 4),
        "seaAdLfcMin": config.SEA_AD_LFC_MIN,
        "familyMap": fam,
        "familyColors": dict(KL_FAMILY_COLORS),
        "pi0": _extract_pi0(data.backbone_sig, contrasts),
    }

    kid = {k: i for i, k in enumerate(data.edge_metadata["kinases"])}
    ev = data.celltype_evidence[
        data.celltype_evidence["kinase"].isin(kid)
    ].copy()
    ev["kinase_id"] = ev["kinase"].map(kid).astype("uint16")
    kinase_celltype_evidence = {
        "kinase_id":  ev["kinase_id"].tolist(),
        "cell_type":  ev["cell_type"].tolist(),
        "wmb_fold":   ev["wmb_fold_over_uniform"].astype(float).round(3).tolist(),
        "sea_ad_lfc": ev["sea_ad_lfc"].astype(float).round(3).tolist(),
        "song_lfc":   ev["song_lfc"].astype(float).round(3).tolist(),
        "wmb_tier":   ev["wmb_tier"].astype(str).tolist(),
    }

    payload = {
        "kinases": kinases_slice,
        "celltypes": celltypes_slice,
        "backbones": backbones_slice,
        "overview": _build_overview_slice(data),
        "per_kinase_summary": per_kinase_summary,
        "kinase_celltype_evidence": kinase_celltype_evidence,
        "edge_slice_ref": {
            "kinase_url": "edge_slices/kinase/",
            "backbone_url": "edge_slices/backbone/",
            "kinase_index": "edge_slices/kinase/index.json",
            "backbone_index": "edge_slices/backbone/index.json",
            "backbone_summary_url": "edge_summaries/per_backbone_summary.parquet",
            "bucket_size": backbone_slice_index["bucket_size"],
            "schema_version": SCHEMA_VERSION,
            "n_kinase_slices": kinase_slice_index["slice_count"],
            "n_backbone_buckets": backbone_slice_index["bucket_count"],
            "present_kinase_ids": kinase_slice_index["present_kinase_ids"],
            "source_sha256": kinase_slice_index.get("source_sha256"),
        },
        "meta": meta,
    }
    return _sanitize(payload)


def write_payload(payload: dict) -> dict:
    os.makedirs(UNIFIED_VIEWER_DIR, exist_ok=True)
    json_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raw = json_str.encode("utf-8")
    with open(PAYLOAD_JSON, "wb") as f:
        f.write(raw)
    gz = gzip.compress(raw, compresslevel=6)
    with open(PAYLOAD_JSON_GZ, "wb") as f:
        f.write(gz)
    return {"raw_bytes": len(raw), "gzip_bytes": len(gz), "json_str": json_str}


# ---------------------------------------------------------------------------
# HTML shell (Phase 3)
# ---------------------------------------------------------------------------

# Raw string + sentinel replacement avoids f-string collisions with CSS/JS braces.
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Unified Kinase + Pathway Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script type="module">
  // hyparquet: ESM-only parquet reader. Attach to window for non-module code.
  import { parquetReadObjects } from "https://cdn.jsdelivr.net/npm/hyparquet@1.8.0/+esm";
  window.hyparquet = { parquetReadObjects };
</script>
<style>
:root {
  --app-red:__APP_COLOR__; --tau-blue:__TAU_COLOR__; --aptt-purple:__APTT_COLOR__;
  --up-red:__APP_COLOR__; --down-blue:__TAU_COLOR__;
  --receptor-color:#1b5e20; --em-color:#e65100; --target-color:#4a148c;
  --bg:#fafafa; --card-bg:#ffffff; --border:#e0e0e0;
  --text:#212121; --text-muted:#757575;
  --near-miss-bg:#fff8e1; --sub-thresh-bg:#f5f5f5;
  --selected-border:#1976d2;
}
* { box-sizing:border-box; }
html, body { margin:0; padding:0; background:var(--bg); color:var(--text);
  font:13px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
header#app-header { background:#455a64; color:#fff; padding:8px 14px;
  display:flex; gap:10px; align-items:center; flex-wrap:wrap;
  border-bottom:1px solid #37474f; }
header#app-header h1 { margin:0 16px 0 0; font-size:15px; font-weight:600;
  letter-spacing:0.2px; }
header#app-header label { display:flex; gap:4px; align-items:center;
  font-size:12px; color:#cfd8dc; }
header#app-header select, header#app-header input[type=number] {
  background:#fff; color:var(--text); border:1px solid #37474f;
  border-radius:3px; padding:2px 5px; font-size:12px; }
header#app-header button#glossary-toggle {
  margin-left:auto; background:#263238; color:#fff; border:1px solid #37474f;
  border-radius:3px; padding:3px 10px; cursor:pointer; font-size:12px; }
header#app-header button#glossary-toggle:hover { background:#37474f; }
nav#tab-bar { background:#37474f; display:flex; gap:0; padding:0 10px;
  border-bottom:1px solid #263238; }
nav#tab-bar button { background:transparent; color:#cfd8dc; border:none;
  border-bottom:3px solid transparent; padding:9px 16px; cursor:pointer;
  font-size:13px; font-weight:500; letter-spacing:0.2px; }
nav#tab-bar button:hover { color:#fff; background:#455a64; }
nav#tab-bar button.active { color:#fff; border-bottom-color:#42a5f5; }
main#app-main { padding:14px; }
.tab-panel { display:none; }
.tab-panel.active { display:block; }
.card { background:var(--card-bg); border:1px solid var(--border);
  border-radius:4px; padding:12px 14px; margin-bottom:12px; }
.card h2 { margin:0 0 6px; font-size:14px; font-weight:600; }
.muted { color:var(--text-muted); font-size:12px; }
#overview-plot { width:100%; height:560px; }
#glossary-panel { position:fixed; top:0; right:0; width:340px; height:100%;
  background:#fff; border-left:1px solid var(--border); padding:16px 18px;
  box-shadow:-4px 0 12px rgba(0,0,0,0.08); transform:translateX(100%);
  transition:transform 0.15s ease-out; z-index:50; overflow-y:auto; }
#glossary-panel.open { transform:translateX(0); }
#glossary-panel h3 { margin-top:0; font-size:14px; }
#glossary-panel dl { margin:0; font-size:12px; }
#glossary-panel dt { font-weight:600; margin-top:8px; color:#37474f; }
#glossary-panel dd { margin:2px 0 0; color:var(--text-muted); }
.tab-stub { color:var(--text-muted); font-style:italic; padding:40px;
  text-align:center; border:1px dashed var(--border); border-radius:4px; }
.explorer-layout { display:grid; grid-template-columns:minmax(0,1.6fr) minmax(320px,1fr);
  gap:12px; align-items:start; }
.detail-card { background:var(--card-bg); border:1px solid var(--border);
  border-radius:4px; padding:12px 14px; position:sticky; top:10px;
  max-height:calc(100vh - 80px); overflow-y:auto; }
.detail-card h3 { margin:0 0 4px; font-size:14px; font-weight:600; }
.detail-card .meta { color:var(--text-muted); font-size:11px; margin-bottom:8px; }
.detail-card h4 { margin:12px 0 4px; font-size:12px; font-weight:600;
  color:#37474f; text-transform:uppercase; letter-spacing:0.3px; }
.ke-toolbar { display:flex; gap:10px; align-items:center; margin-bottom:8px; }
.ke-toolbar input { padding:3px 6px; border:1px solid var(--border);
  border-radius:3px; font-size:12px; width:220px; }
.ke-table-wrap { max-height:70vh; overflow-y:auto; }
.data-table { border-collapse:collapse; font-size:12px; width:100%; }
.data-table th, .data-table td { padding:4px 8px; border-bottom:1px solid var(--border);
  text-align:left; vertical-align:top; white-space:nowrap; }
.data-table thead th { position:sticky; top:0; background:#eceff1; cursor:pointer;
  user-select:none; z-index:1; font-weight:600; }
.data-table thead th:hover { background:#cfd8dc; }
.data-table tbody tr { cursor:pointer; }
.data-table tbody tr:hover { background:#f5f5f5; }
.data-table tbody tr.selected { background:#e3f2fd; box-shadow:inset 3px 0 0 var(--selected-border); }
.data-table tbody tr.sub-thresh { color:var(--text-muted); }
.badge { display:inline-block; padding:1px 6px; border-radius:9px; font-size:10px;
  font-weight:600; letter-spacing:0.2px; }
.badge.hi { background:#c8e6c9; color:#1b5e20; }
.badge.lo { background:#eceff1; color:#546e7a; }
#ke-detail-nes { height:160px; }
#pe-detail-cross { height:180px; }
.pe-chip { display:inline-block; padding:1px 5px; margin:0 2px 2px 0;
  border-radius:3px; font-size:10px; font-weight:600; background:#eceff1; color:#546e7a; }
.pe-chip.on { background:#c8e6c9; color:#1b5e20; }
.detail-chips { display:flex; gap:12px; align-items:center; margin-bottom:8px;
  flex-wrap:wrap; font-size:12px; }
.detail-chips label { display:flex; gap:4px; align-items:center; }
.chip { background:#fff3cd; color:#8a6d3b; border:1px solid #f0ad4e;
  border-radius:3px; padding:2px 8px; font-size:11px; cursor:pointer; }
#graph-container { display:grid; grid-template-columns:1fr 320px; gap:12px;
  height:calc(100vh - 180px); }
#cy { background:#fafafa; border:1px solid var(--border); border-radius:4px;
  min-height:400px; }
#graph-detail { position:relative; top:0; }
.graph-placeholder { display:flex; align-items:center; justify-content:center;
  height:100%; color:var(--text-muted); font-style:italic; text-align:center;
  padding:20px; }
</style>
</head>
<body>
<header id="app-header">
  <h1>Unified Kinase + Pathway Viewer</h1>
  <label>Contrast <select id="f-contrast"></select></label>
  <label>Direction <select id="f-direction">
    <option value="ALL">All</option>
    <option value="up">Up</option>
    <option value="down">Down</option>
  </select></label>
  <label>Receiver <select id="f-receiver"></select></label>
  <label>FDR &lt; <input id="f-fdr" type="number" step="0.05" min="0" max="1"
    value="0.25" style="width:60px;"></label>
  <label>|Score| &gt; <input id="f-score" type="number" step="0.1" min="0"
    value="0" style="width:60px;"></label>
  <button id="glossary-toggle">Glossary</button>
  <button id="f-graph-nodes-clear" class="chip" hidden>Clear graph-node filter</button>
</header>
<nav id="tab-bar">
  <button data-tab="overview" class="active">Overview</button>
  <button data-tab="kinase">Kinase</button>
  <button data-tab="pathway">Pathway</button>
  <button data-tab="graph">Graph</button>
  <button data-tab="senders">Sender&times;Receiver</button>
  <button data-tab="temporal">Temporal</button>
  <button data-tab="additivity">Additivity</button>
</nav>
<main id="app-main">
  <div id="tab-overview" class="tab-panel active">
    <div class="card">
      <h2>Receiver &times; Contrast</h2>
      <div class="muted" id="overview-subtitle">Sig backbone counts
        (significant-both) per receiver cell type across 9 contrasts.
        Click any cell to filter the other tabs to that receiver.</div>
    </div>
    <div class="card">
      <div id="overview-plot"></div>
    </div>
  </div>
  <div id="tab-kinase" class="tab-panel">
    <div class="explorer-layout">
      <div class="card">
        <div class="ke-toolbar">
          <input id="ke-search" placeholder="Search kinase or gene…"/>
          <span class="muted" id="ke-count"></span>
        </div>
        <div class="ke-table-wrap">
          <table class="data-table" id="ke-table">
            <thead><tr>
              <th data-col="name">Kinase</th>
              <th data-col="family">Family</th>
              <th data-col="gene_symbol">Gene</th>
              <th data-col="n_sig">Sig vs WT</th>
              <th data-col="peak_NES">Peak NES</th>
              <th data-col="top_celltype_1">Top cell type</th>
              <th data-col="has_high_conf_attribution">Conf</th>
              <th data-col="n_backbones">#Backbones</th>
            </tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
      <aside class="detail-card" id="ke-detail">
        <div class="muted">Select a kinase to see details.</div>
      </aside>
    </div>
  </div>
  <div id="tab-pathway" class="tab-panel">
    <div class="explorer-layout">
      <div class="card">
        <div class="ke-toolbar">
          <input id="pe-search" placeholder="Search Receptor / EM / Target…"/>
          <label><input id="pe-sig-only" type="checkbox" checked/> Sig-both only</label>
          <span class="muted" id="pe-count"></span>
        </div>
        <div class="ke-table-wrap">
          <table class="data-table" id="pe-table">
            <thead><tr>
              <th data-col="receiver">Receiver</th>
              <th data-col="Receptor">Receptor</th>
              <th data-col="EM">EM</th>
              <th data-col="Target">Target</th>
              <th data-col="tpds">TPDS</th>
              <th data-col="n_sig">Sig</th>
              <th data-col="n_senders">Senders</th>
              <th data-col="max_abs_tpds">Max|TPDS|</th>
            </tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
      <aside class="detail-card" id="pe-detail">
        <div class="muted">Select a backbone to see details.</div>
      </aside>
    </div>
  </div>
  <div id="tab-graph" class="tab-panel">
    <div id="graph-controls" class="detail-chips">
      <label>Layout:
        <select id="graph-layout">
          <option value="concentric">Concentric (R → EM → T)</option>
          <option value="flow">Flow (column-snapped)</option>
          <option value="force">Force-directed</option>
        </select>
      </label>
      <label>Min degree:
        <select id="graph-min-degree">
          <option value="1">1</option>
          <option value="2">2</option>
          <option value="5">5</option>
          <option value="10">10</option>
          <option value="20">20</option>
          <option value="50">50</option>
        </select>
      </label>
      <span id="graph-stats" class="muted"></span>
      <button id="graph-focus-clear" class="chip" hidden>Clear focus</button>
    </div>
    <div id="graph-container">
      <div id="cy"></div>
      <aside class="detail-card" id="graph-detail">
        <div class="muted">Pick a single contrast, then click a node.</div>
      </aside>
    </div>
  </div>
  <div id="tab-senders" class="tab-panel">
    <div class="tab-stub">Sender &times; Receiver — Phase 4.4</div>
  </div>
  <div id="tab-temporal" class="tab-panel">
    <div class="tab-stub">Temporal Dynamics — Phase 4.5</div>
  </div>
  <div id="tab-additivity" class="tab-panel">
    <div class="tab-stub">Additivity — Phase 4.6</div>
  </div>
</main>
<aside id="glossary-panel">
  <h3>Glossary</h3>
  <dl>
    <dt>TPDS</dt><dd>Total Pathway Directional Score per backbone.</dd>
    <dt>Sig-both</dt><dd>Backbone passing both permutation nulls.</dd>
    <dt>NES</dt><dd>Normalized Enrichment Score (MEA on stoichiometry).</dd>
    <dt>Contrast</dt><dd>disease &times; timepoint: App/Tau/ApTt at 2/4/6mo.</dd>
  </dl>
</aside>

<script type="application/json" id="payload-data">__PAYLOAD_SENTINEL__</script>
<script>
"use strict";

// ---------------------------------------------------------------------------
// Payload
// ---------------------------------------------------------------------------
const PAYLOAD = JSON.parse(document.getElementById("payload-data").textContent);
const META = PAYLOAD.meta;
const CONTRASTS = META.contrasts;
const RECEIVERS = PAYLOAD.celltypes.name;
const TISSUE_CAT = PAYLOAD.celltypes.tissue_category;
const DISEASE_COLORS = META.diseaseColors;

// ---------------------------------------------------------------------------
// Store — reducer-style with {selection, filters, view} slices
// ---------------------------------------------------------------------------
const INITIAL_STATE = {
  selection: { kinase:null, backbone:null, celltype:null },
  filters:   { contrast:"ALL", direction:"ALL", receiver:"ALL",
               fdr:0.25, score:0.0, graphNodeIds:null },
  view:      { activeTab:"overview", overviewMode:"count",
               overviewSort:"tissue", glossaryOpen:false,
               graphLayout:"concentric", graphMinDegree:1 },
};

const _clone = (typeof structuredClone === "function")
  ? structuredClone
  : (v) => JSON.parse(JSON.stringify(v));

function reducer(state, action) {
  const s = _clone(state);
  if (action.type === "SET_FILTER") s.filters[action.key] = action.value;
  else if (action.type === "SET_SELECTION") s.selection[action.key] = action.value;
  else if (action.type === "SET_VIEW") s.view[action.key] = action.value;
  else return state;
  return s;
}

const Store = (function(){
  let state = _clone(INITIAL_STATE);
  const subs = [];
  return {
    get state() { return state; },
    subscribe(fn) { subs.push(fn); return () => {
      const i = subs.indexOf(fn); if (i >= 0) subs.splice(i, 1);
    }; },
    dispatch(action) {
      const prev = state;
      const next = reducer(state, action);
      if (next === prev) return;
      state = next;
      for (const fn of subs) fn(next, prev);
    },
  };
})();
window.Store = Store;  // expose for console smoke test

// ---------------------------------------------------------------------------
// Derived-array memoization — keyed on JSON signature of filters slice
// ---------------------------------------------------------------------------
let _filteredCache = { key:null, gnRef:null, indices:null };

function _computeFilteredIndices() {
  const f = Store.state.filters;
  const BB = PAYLOAD.backbones;
  const n = BB.id.length;
  const cIdx = CONTRASTS.indexOf(f.contrast);
  const tpdsCol = cIdx >= 0 ? BB["mean_tpds_" + f.contrast] : null;
  const sigCol = BB.significant_both_mask;
  const rIdx = (f.receiver === "ALL") ? -1 : RECEIVERS.indexOf(f.receiver);
  // graphNodeIds is a transient filter applied after a Pathway Graph node
  // click. Stored as a Set of backbone_id for O(1) membership.
  const gnSet = (f.graphNodeIds && f.graphNodeIds.length)
    ? new Set(f.graphNodeIds) : null;
  const out = [];
  for (let i = 0; i < n; i++) {
    if (rIdx >= 0 && BB.receiver_id[i] !== rIdx) continue;
    if (cIdx >= 0) {
      if (!((sigCol[i] >> cIdx) & 1)) continue;
      const t = tpdsCol[i];
      if (t == null) continue;
      if (f.direction === "up" && !(t > 0)) continue;
      if (f.direction === "down" && !(t < 0)) continue;
      if (f.score > 0 && Math.abs(t) < f.score) continue;
    }
    if (gnSet !== null && !gnSet.has(BB.id[i])) continue;
    out.push(i);
  }
  return out;
}

function getFilteredIndices() {
  const f = Store.state.filters;
  // graphNodeIds array identity changes on each SET_FILTER dispatch (reducer
  // deep-clones state) — use identity, not stringify, to avoid scanning the
  // full array on every read.
  const gnKey = f.graphNodeIds ? ("gn:" + f.graphNodeIds.length) : "gn:null";
  const gnRef = f.graphNodeIds;  // also compare by identity
  const key = f.contrast + "|" + f.direction + "|" + f.receiver + "|"
            + f.fdr + "|" + f.score + "|" + gnKey;
  if (key !== _filteredCache.key || gnRef !== _filteredCache.gnRef) {
    _filteredCache = {
      key, gnRef, indices: _computeFilteredIndices(),
    };
  }
  return _filteredCache.indices;
}
window.getFilteredIndices = getFilteredIndices;

// ---------------------------------------------------------------------------
// SliceCache — lazy loader for per-entity edge parquets (Unit E).
// Kinase slices and backbone-bucket slices are fetched on demand via the
// URLs in PAYLOAD.edge_slice_ref. LRU-capped to avoid unbounded memory.
// Parquet decoding uses hyparquet (CDN-loaded) when available; falls back
// to reporting an error message on the selected entity's side panel.
// ---------------------------------------------------------------------------
const SliceCache = (function(){
  const ESR = PAYLOAD.edge_slice_ref || {};
  const BUCKET_SIZE = ESR.bucket_size || 256;
  const MAX = 16;                          // LRU cap (per side)
  const kCache = new Map();                // kinase_id -> {backbone_id, contrast_id, support_contribution, concordance}
  const bCache = new Map();                // bucket_id -> same shape + kinase_id

  function _lruTouch(cache, key, value){
    if (cache.has(key)) cache.delete(key);
    cache.set(key, value);
    while (cache.size > MAX) cache.delete(cache.keys().next().value);
  }

  async function _fetchParquet(url){
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`fetch ${url} → ${resp.status}`);
    const buf = new Uint8Array(await resp.arrayBuffer());
    if (typeof hyparquet === "undefined") {
      throw new Error("parquet reader not loaded (hyparquet missing)");
    }
    return await hyparquet.parquetReadObjects({ file: buf });
  }

  async function loadKinase(kinase_id){
    if (kCache.has(kinase_id)) {
      const v = kCache.get(kinase_id); _lruTouch(kCache, kinase_id, v); return v;
    }
    const pad = String(kinase_id).padStart(3, "0");
    const url = `${ESR.kinase_url}${pad}.parquet`;
    const rows = await _fetchParquet(url);
    _lruTouch(kCache, kinase_id, rows);
    return rows;
  }

  async function loadBackboneBucket(backbone_id){
    const bkt = Math.floor(backbone_id / BUCKET_SIZE);
    if (bCache.has(bkt)) {
      const v = bCache.get(bkt); _lruTouch(bCache, bkt, v); return v;
    }
    const pad = String(bkt).padStart(3, "0");
    const url = `${ESR.backbone_url}${pad}.parquet`;
    const rows = await _fetchParquet(url);
    _lruTouch(bCache, bkt, rows);
    return rows;
  }

  async function backboneEdges(backbone_id){
    const rows = await loadBackboneBucket(backbone_id);
    return rows.filter(r => r.backbone_id === backbone_id);
  }

  return { loadKinase, loadBackboneBucket, backboneEdges,
           get kinaseCacheSize(){ return kCache.size; },
           get backboneCacheSize(){ return bCache.size; } };
})();
window.SliceCache = SliceCache;

// ---------------------------------------------------------------------------
// Header wiring
// ---------------------------------------------------------------------------
function populateHeader() {
  const fc = document.getElementById("f-contrast");
  fc.innerHTML = ['<option value="ALL">All</option>']
    .concat(CONTRASTS.map(c => `<option value="${c}">${c}</option>`)).join("");
  const fr = document.getElementById("f-receiver");
  fr.innerHTML = ['<option value="ALL">All</option>']
    .concat(RECEIVERS.map(r => `<option value="${r}">${r}</option>`)).join("");
  fc.addEventListener("change", e => Store.dispatch({
    type:"SET_FILTER", key:"contrast", value:e.target.value}));
  fr.addEventListener("change", e => Store.dispatch({
    type:"SET_FILTER", key:"receiver", value:e.target.value}));
  document.getElementById("f-direction").addEventListener("change", e =>
    Store.dispatch({type:"SET_FILTER", key:"direction", value:e.target.value}));
  document.getElementById("f-fdr").addEventListener("change", e =>
    Store.dispatch({type:"SET_FILTER", key:"fdr", value:parseFloat(e.target.value)}));
  document.getElementById("f-score").addEventListener("change", e =>
    Store.dispatch({type:"SET_FILTER", key:"score", value:parseFloat(e.target.value)}));
  document.getElementById("glossary-toggle").addEventListener("click", () =>
    Store.dispatch({type:"SET_VIEW", key:"glossaryOpen",
      value:!Store.state.view.glossaryOpen}));
  const gnClear = document.getElementById("f-graph-nodes-clear");
  if (gnClear) gnClear.addEventListener("click", () =>
    Store.dispatch({type:"SET_FILTER", key:"graphNodeIds", value:null}));
}

function syncHeaderFromStore() {
  const f = Store.state.filters;
  const ids = ["f-contrast","f-direction","f-receiver"];
  const vals = [f.contrast, f.direction, f.receiver];
  for (let i = 0; i < ids.length; i++) {
    const el = document.getElementById(ids[i]);
    if (el && el.value !== String(vals[i])) el.value = vals[i];
  }
  document.getElementById("f-fdr").value = f.fdr;
  document.getElementById("f-score").value = f.score;
  const gnClear = document.getElementById("f-graph-nodes-clear");
  if (gnClear) {
    const on = !!(f.graphNodeIds && f.graphNodeIds.length);
    gnClear.hidden = !on;
    if (on) gnClear.textContent = "Clear graph-node filter ("
      + f.graphNodeIds.length + " backbones)";
  }
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
function wireTabs() {
  document.querySelectorAll("nav#tab-bar button").forEach(btn => {
    btn.addEventListener("click", () => {
      Store.dispatch({type:"SET_VIEW", key:"activeTab", value:btn.dataset.tab});
    });
  });
}

function syncTabsFromStore() {
  const active = Store.state.view.activeTab;
  document.querySelectorAll("nav#tab-bar button").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === active);
  });
  document.querySelectorAll(".tab-panel").forEach(p => {
    p.classList.toggle("active", p.id === "tab-" + active);
  });
}

// ---------------------------------------------------------------------------
// Overview tab — receiver × contrast heatmap
// ---------------------------------------------------------------------------
function receiverOrder() {
  // Sort receivers by tissue_category then alphabetical within category.
  const pairs = RECEIVERS.map((r, i) => [r, TISSUE_CAT[i] || "zzz", i]);
  pairs.sort((a, b) => {
    if (a[1] !== b[1]) return a[1] < b[1] ? -1 : 1;
    return a[0] < b[0] ? -1 : (a[0] > b[0] ? 1 : 0);
  });
  return pairs.map(p => p[0]);
}

function renderOverview() {
  const el = document.getElementById("overview-plot");
  if (!el) return;
  const f = Store.state.filters;
  const mode = Store.state.view.overviewMode;  // 'count' | 'direction'
  const rows = receiverOrder();
  const cols = CONTRASTS;

  // Build z matrix + hover + customdata.
  const z = [], hover = [], cd = [];
  for (const r of rows) {
    const zrow = [], hrow = [], crow = [];
    for (const c of cols) {
      const cell = PAYLOAD.overview[c + "|" + r];
      if (!cell || cell.n === 0) {
        zrow.push(null); hrow.push(`${r} | ${c}<br>(no sig backbones)`);
        crow.push({receiver:r, contrast:c, n:0});
      } else {
        let v;
        if (mode === "direction") v = cell.n_up - cell.n_down;
        else v = Math.log10(1 + cell.n);
        // Direction filter mask
        if (f.direction === "up" && cell.n_up === 0) v = null;
        if (f.direction === "down" && cell.n_down === 0) v = null;
        zrow.push(v);
        hrow.push(
          `${r} | ${c}<br>n=${cell.n} (up=${cell.n_up}, down=${cell.n_down})` +
          `<br>mean TPDS=${cell.mean_tpds}`);
        crow.push({receiver:r, contrast:c, n:cell.n});
      }
    }
    z.push(zrow); hover.push(hrow); cd.push(crow);
  }

  // Contrast filter: dim non-selected columns by blanking cells.
  if (f.contrast !== "ALL") {
    const keep = cols.indexOf(f.contrast);
    for (let i = 0; i < z.length; i++)
      for (let j = 0; j < z[i].length; j++)
        if (j !== keep) z[i][j] = null;
  }

  const colorscale = (mode === "direction")
    ? [[0, DISEASE_COLORS.Tau], [0.5, "#ffffff"], [1, DISEASE_COLORS.App]]
    : "YlOrRd";
  const trace = {
    type:"heatmap", x:cols, y:rows, z, text:hover,
    hovertemplate:"%{text}<extra></extra>", customdata:cd,
    colorscale, showscale:true,
    zmid: (mode === "direction") ? 0 : undefined,
  };
  const layout = {
    margin:{l:130, r:20, t:10, b:90},
    xaxis:{tickangle:-30, automargin:true},
    yaxis:{automargin:true, autorange:"reversed"},
    height:560,
  };
  Plotly.react(el, [trace], layout, {displaylogo:false, responsive:true});

  // Plotly.react preserves the DOM node, so detach prior listeners first.
  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on && el.on("plotly_click", ev => {
    if (!ev.points || !ev.points.length) return;
    const d = ev.points[0].customdata;
    if (!d || d.n === 0) return;
    Store.dispatch({type:"SET_SELECTION", key:"backbone", value:null});
    Store.dispatch({type:"SET_FILTER", key:"receiver", value:d.receiver});
  });
}

// ---------------------------------------------------------------------------
// Kinase Explorer tab
// ---------------------------------------------------------------------------
let keSortCol = "peak_NES";
let keSortAsc = false;
let keSearch = "";
let _keRows = null;
let _keSigFdr = null;
let _kinaseIdxById = null;
let _backboneIdxById = null;
let _evidenceByKinase = null;
let _presentKinaseSet = null;

function _buildKinaseRowModel() {
  const K = PAYLOAD.kinases;
  const PKS = PAYLOAD.per_kinase_summary || {kinase_id:[], n_backbones:[]};
  const famMap = META.familyMap || {};
  const bbByK = new Array(K.id.length).fill(0);
  for (let i = 0; i < PKS.kinase_id.length; i++) {
    bbByK[PKS.kinase_id[i]] += PKS.n_backbones[i];
  }
  const idxById = new Map();
  const out = [];
  for (let i = 0; i < K.id.length; i++) {
    idxById.set(K.id[i], i);
    out.push({
      id: K.id[i],
      name: K.name[i],
      gene_symbol: K.gene_symbol[i] || "",
      family: famMap[K.name[i]] || "",
      trajectory: K.trajectory[i] || "",
      peak_contrast: K.peak_contrast[i] || "",
      peak_NES: K.peak_NES[i],
      top_celltype_1: K.top_celltype_1[i] || "",
      has_high_conf_attribution: !!K.has_high_conf_attribution[i],
      n_backbones: bbByK[i],
      _fdr: CONTRASTS.map(c => K["FDR_" + c][i]),
      _nes: CONTRASTS.map(c => K["NES_" + c][i]),
    });
  }
  _kinaseIdxById = idxById;
  return out;
}

function _ensureBackboneIdx() {
  if (_backboneIdxById !== null) return;
  const BB = PAYLOAD.backbones;
  const m = new Map();
  for (let i = 0; i < BB.id.length; i++) m.set(BB.id[i], i);
  _backboneIdxById = m;
}

function _ensureKinaseIdx() {
  if (_kinaseIdxById !== null) return;
  const K = PAYLOAD.kinases;
  const m = new Map();
  for (let i = 0; i < K.id.length; i++) m.set(K.id[i], i);
  _kinaseIdxById = m;
}

function _ensureKinaseIndexes() {
  if (_keRows === null) _keRows = _buildKinaseRowModel();
  _ensureBackboneIdx();
  if (_evidenceByKinase === null) {
    const EV = PAYLOAD.kinase_celltype_evidence || {kinase_id:[]};
    const m = new Map();
    for (let k = 0; k < EV.kinase_id.length; k++) {
      const kid = EV.kinase_id[k];
      let arr = m.get(kid);
      if (!arr) { arr = []; m.set(kid, arr); }
      arr.push(k);
    }
    _evidenceByKinase = m;
  }
  if (_presentKinaseSet === null) {
    const esr = PAYLOAD.edge_slice_ref || {};
    _presentKinaseSet = new Set(esr.present_kinase_ids || []);
  }
}

function _refreshSigCounts(fdr) {
  if (_keSigFdr === fdr) return;
  for (const r of _keRows) {
    let n = 0;
    for (const v of r._fdr) if (v != null && v < fdr) n++;
    r._sigCount = n;
  }
  _keSigFdr = fdr;
}

function _keCompare(a, b) {
  const col = keSortCol;
  let va, vb;
  if (col === "n_sig") { va = a._sigCount; vb = b._sigCount; }
  else if (col === "peak_NES") {
    va = a.peak_NES == null ? -Infinity : Math.abs(a.peak_NES);
    vb = b.peak_NES == null ? -Infinity : Math.abs(b.peak_NES);
  }
  else { va = a[col]; vb = b[col]; }
  if (va == null && vb == null) return 0;
  if (va == null) return 1;
  if (vb == null) return -1;
  if (typeof va === "string") return keSortAsc
    ? va.localeCompare(vb) : vb.localeCompare(va);
  return keSortAsc ? (va - vb) : (vb - va);
}

function renderKinaseExplorer() {
  const tbody = document.querySelector("#ke-table tbody");
  if (!tbody) return;
  _ensureKinaseIndexes();
  const f = Store.state.filters;
  const fdr = f.fdr;
  const contrast = f.contrast;
  const cIdx = CONTRASTS.indexOf(contrast);
  const selKid = Store.state.selection.kinase;
  const q = keSearch.trim().toLowerCase();

  _refreshSigCounts(fdr);
  const visible = [];
  for (const r of _keRows) {
    if (cIdx >= 0) {
      const fdrC = r._fdr[cIdx];
      if (!(fdrC != null && fdrC < fdr)) continue;
    }
    if (q && !(r.name.toLowerCase().includes(q) ||
               r.gene_symbol.toLowerCase().includes(q))) continue;
    visible.push(r);
  }
  visible.sort(_keCompare);

  document.querySelectorAll("#ke-table thead th").forEach(th => {
    const c = th.dataset.col;
    th.textContent = th.textContent.replace(/[ ▲▼]+$/, "");
    if (c === keSortCol) th.textContent += keSortAsc ? " ▲" : " ▼";
  });

  const parts = [];
  for (const r of visible) {
    const selCls = r.id === selKid ? " selected" : "";
    const subCls = r._sigCount === 0 ? " sub-thresh" : "";
    const conf = r.has_high_conf_attribution
      ? '<span class="badge hi">HIGH</span>'
      : '<span class="badge lo">low</span>';
    const peak = r.peak_NES == null ? "—" : r.peak_NES.toFixed(2);
    parts.push(
      `<tr class="ke-row${selCls}${subCls}" data-kid="${r.id}">` +
      `<td>${r.name}</td>` +
      `<td>${r.family}</td>` +
      `<td>${r.gene_symbol}</td>` +
      `<td>${r._sigCount}</td>` +
      `<td>${peak}</td>` +
      `<td>${r.top_celltype_1 || "—"}</td>` +
      `<td>${conf}</td>` +
      `<td>${r.n_backbones.toLocaleString()}</td>` +
      `</tr>`
    );
  }
  tbody.innerHTML = parts.join("");
  const countEl = document.getElementById("ke-count");
  if (countEl) countEl.textContent = `${visible.length} / ${_keRows.length} kinases`;
}

function _updateRowSelection(tableSel, rowCls, dataAttr, value) {
  const tbody = document.querySelector(`${tableSel} tbody`);
  if (!tbody) return;
  const prev = tbody.querySelector(`tr.${rowCls}.selected`);
  if (prev) prev.classList.remove("selected");
  if (value == null) return;
  const row = tbody.querySelector(`tr.${rowCls}[${dataAttr}="${value}"]`);
  if (row) row.classList.add("selected");
}

function _updateKinaseRowSelection(kid) {
  _updateRowSelection("#ke-table", "ke-row", "data-kid", kid);
}

function _diseaseColorFor(contrast) {
  for (const d of ["App","Tau","ApTt"])
    if (contrast.indexOf(d) === 0) return DISEASE_COLORS[d];
  return "#90a4ae";
}

function renderKinaseDetail(kinase_id) {
  const el = document.getElementById("ke-detail");
  if (!el) return;
  if (kinase_id == null) {
    el.innerHTML = '<div class="muted">Select a kinase to see details.</div>';
    return;
  }
  _ensureKinaseIndexes();
  const K = PAYLOAD.kinases;
  const i = _kinaseIdxById.get(kinase_id);
  if (i == null) {
    el.innerHTML = '<div class="muted">Kinase not found.</div>';
    return;
  }
  const name = K.name[i];
  const family = (META.familyMap || {})[name] || "—";
  const gene = K.gene_symbol[i] || "—";
  const traj = K.trajectory[i] || "—";
  const fdr = Store.state.filters.fdr;

  el.innerHTML =
    `<h3>${name}</h3>` +
    `<div class="meta">Family: ${family} · Gene: ${gene} · Trajectory: ${traj}</div>` +
    `<h4>NES by contrast</h4><div id="ke-detail-nes"></div>` +
    `<h4>Cell-type evidence</h4><div id="ke-detail-evidence"></div>` +
    `<h4>Backbones supported</h4><div id="ke-detail-backbones" class="muted">loading…</div>`;

  const nes = CONTRASTS.map(c => K["NES_" + c][i]);
  const fdrs = CONTRASTS.map(c => K["FDR_" + c][i]);
  const colors = CONTRASTS.map(_diseaseColorFor);
  const outlines = fdrs.map(v => (v != null && v < fdr) ? "#000" : "rgba(0,0,0,0)");
  Plotly.react("ke-detail-nes", [{
    type: "bar", x: CONTRASTS, y: nes,
    marker: { color: colors, line: { color: outlines, width: 1.5 } },
    hovertemplate: "%{x}<br>NES %{y:.2f}<extra></extra>",
  }], {
    margin:{l:40,r:10,t:6,b:60}, height:160,
    yaxis:{zeroline:true, zerolinecolor:"#bbb"},
    xaxis:{tickangle:-35},
  }, {displaylogo:false, responsive:true});

  const EV = PAYLOAD.kinase_celltype_evidence || {kinase_id:[]};
  const evIdx = _evidenceByKinase.get(kinase_id) || [];
  const rows = evIdx.map(k => ({
    cell_type: EV.cell_type[k],
    wmb_fold: EV.wmb_fold[k],
    sea_ad_lfc: EV.sea_ad_lfc[k],
    song_lfc: EV.song_lfc[k],
    wmb_tier: EV.wmb_tier[k],
  }));
  rows.sort((a, b) => {
    const av = a.wmb_fold == null ? -Infinity : a.wmb_fold;
    const bv = b.wmb_fold == null ? -Infinity : b.wmb_fold;
    return bv - av;
  });
  const evEl = document.getElementById("ke-detail-evidence");
  if (rows.length === 0) {
    evEl.innerHTML = '<div class="muted">No evidence rows.</div>';
  } else {
    const evParts = ['<table class="data-table"><thead><tr>',
      '<th>Cell type</th><th>WMB fold</th><th>SEA-AD LFC</th>',
      '<th>Song LFC</th><th>Tier</th></tr></thead><tbody>'];
    for (const r of rows) {
      const tierCls = r.wmb_tier === "high" ? "hi" : "lo";
      const fmt = (v) => v == null ? "—" : Number(v).toFixed(2);
      evParts.push(
        `<tr><td>${r.cell_type}</td>` +
        `<td>${fmt(r.wmb_fold)}</td>` +
        `<td>${fmt(r.sea_ad_lfc)}</td>` +
        `<td>${fmt(r.song_lfc)}</td>` +
        `<td><span class="badge ${tierCls}">${r.wmb_tier}</span></td></tr>`
      );
    }
    evParts.push("</tbody></table>");
    evEl.innerHTML = evParts.join("");
  }

  renderKinaseBackbones(kinase_id);
}

async function renderKinaseBackbones(kinase_id) {
  const container = document.getElementById("ke-detail-backbones");
  if (!container) return;
  _ensureKinaseIndexes();
  if (!_presentKinaseSet.has(kinase_id)) {
    container.innerHTML = '<div class="muted">No significant edges for this kinase.</div>';
    container.classList.remove("muted");
    return;
  }
  let rows;
  try {
    rows = await SliceCache.loadKinase(kinase_id);
  } catch (e) {
    if (Store.state.selection.kinase !== kinase_id) return;
    container.innerHTML = `<div class="muted">Failed to load: ${e.message}</div>`;
    return;
  }
  if (Store.state.selection.kinase !== kinase_id) return;

  const f = Store.state.filters;
  const cIdx = CONTRASTS.indexOf(f.contrast);
  const filtered = (cIdx >= 0)
    ? rows.filter(r => r.contrast_id === cIdx)
    : rows;
  const sorted = filtered.slice().sort(
    (a, b) => Math.abs(b.support_contribution) - Math.abs(a.support_contribution));
  const TOP = 200;
  const shown = sorted.slice(0, TOP);

  const BB = PAYLOAD.backbones;
  const bbIdxById = _backboneIdxById;

  const parts = [
    `<div class="muted">Showing top ${shown.length} of ${filtered.length} edges` +
    (cIdx >= 0 ? ` (contrast ${f.contrast})` : "") + `.</div>`,
    '<table class="data-table"><thead><tr>',
    '<th>Receiver</th><th>Receptor</th><th>EM</th><th>Target</th>',
    '<th>Contrast</th><th>Support</th><th>Conc.</th>',
    '</tr></thead><tbody>',
  ];
  for (const r of shown) {
    const bi = bbIdxById.get(r.backbone_id);
    const rcv = bi != null ? RECEIVERS[BB.receiver_id[bi]] : "?";
    const rcp = bi != null ? BB.Receptor[bi] : "?";
    const em  = bi != null ? BB.EM[bi] : "?";
    const tgt = bi != null ? BB.Target[bi] : "?";
    const contr = CONTRASTS[r.contrast_id] || "?";
    const sup = Number(r.support_contribution).toFixed(3);
    const conc = r.concordance > 0 ? "↑" : (r.concordance < 0 ? "↓" : "—");
    parts.push(
      `<tr><td>${rcv}</td><td>${rcp}</td><td>${em}</td><td>${tgt}</td>` +
      `<td>${contr}</td><td>${sup}</td><td>${conc}</td></tr>`
    );
  }
  parts.push("</tbody></table>");
  container.innerHTML = parts.join("");
}

function wireKinaseTable() {
  const tbl = document.getElementById("ke-table");
  if (!tbl) return;
  tbl.querySelectorAll("thead th").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (keSortCol === col) keSortAsc = !keSortAsc;
      else { keSortCol = col; keSortAsc = false; }
      renderKinaseExplorer();
    });
  });
  tbl.querySelector("tbody").addEventListener("click", ev => {
    const tr = ev.target.closest("tr.ke-row");
    if (!tr) return;
    const kid = parseInt(tr.dataset.kid, 10);
    Store.dispatch({type:"SET_SELECTION", key:"kinase", value: kid});
  });
  const search = document.getElementById("ke-search");
  if (search) search.addEventListener("input", ev => {
    keSearch = ev.target.value;
    renderKinaseExplorer();
  });
}

// ---------------------------------------------------------------------------
// Pathway Explorer tab
// ---------------------------------------------------------------------------
let peSortCol = "tpds";
let peSortAsc = false;
let peSearch = "";
let peSigOnly = true;
let _peRows = null;
let _peSearchTimer = null;

function _popcount(m) {
  m = m - ((m >> 1) & 0x55555555);
  m = (m & 0x33333333) + ((m >> 2) & 0x33333333);
  return (((m + (m >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

function _buildPathwayRowModel() {
  const BB = PAYLOAD.backbones;
  const n = BB.id.length;
  const tpdsCols = CONTRASTS.map(c => BB["mean_tpds_" + c]);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const tpds = new Array(CONTRASTS.length);
    for (let c = 0; c < CONTRASTS.length; c++) tpds[c] = tpdsCols[c][i];
    out[i] = {
      idx: i,
      id: BB.id[i],
      receiver_id: BB.receiver_id[i],
      receiver: RECEIVERS[BB.receiver_id[i]],
      Receptor: BB.Receptor[i] || "",
      EM: BB.EM[i] || "",
      Target: BB.Target[i] || "",
      sender_mask: BB.sender_mask[i],
      n_senders: BB.n_senders[i],
      n_senders_sig: BB.n_senders_significant[i],
      max_abs_tpds: BB.max_abs_tpds[i],
      sig_mask: BB.significant_both_mask[i],
      sig_count: _popcount(BB.significant_both_mask[i]),
      _tpds: tpds,
    };
  }
  return out;
}

function _ensurePathwayIndexes() {
  if (_peRows === null) _peRows = _buildPathwayRowModel();
  _ensureBackboneIdx();
}

function _peCompare(a, b, cIdx) {
  const col = peSortCol;
  let va, vb;
  if (col === "tpds") {
    va = cIdx >= 0 ? a._tpds[cIdx] : a.max_abs_tpds;
    vb = cIdx >= 0 ? b._tpds[cIdx] : b.max_abs_tpds;
    if (va == null) va = -Infinity;
    if (vb == null) vb = -Infinity;
  }
  else if (col === "n_sig") { va = a.sig_count; vb = b.sig_count; }
  else if (col === "receiver") { va = a.receiver; vb = b.receiver; }
  else { va = a[col]; vb = b[col]; }
  if (va == null && vb == null) return 0;
  if (va == null) return 1;
  if (vb == null) return -1;
  if (typeof va === "string") return peSortAsc
    ? va.localeCompare(vb) : vb.localeCompare(va);
  return peSortAsc ? (va - vb) : (vb - va);
}

function renderPathwayExplorer() {
  const tbody = document.querySelector("#pe-table tbody");
  if (!tbody) return;
  _ensurePathwayIndexes();
  const f = Store.state.filters;
  const cIdx = CONTRASTS.indexOf(f.contrast);
  const selBid = Store.state.selection.backbone;
  const q = peSearch.trim().toLowerCase();
  const baseIdx = getFilteredIndices();

  const visible = [];
  for (const i of baseIdx) {
    const r = _peRows[i];
    if (peSigOnly && r.sig_count === 0) continue;
    if (q && !(r.Receptor.toLowerCase().includes(q) ||
               r.EM.toLowerCase().includes(q) ||
               r.Target.toLowerCase().includes(q))) continue;
    visible.push(r);
  }
  visible.sort((a, b) => _peCompare(a, b, cIdx));

  document.querySelectorAll("#pe-table thead th").forEach(th => {
    const c = th.dataset.col;
    th.textContent = th.textContent.replace(/[ ▲▼]+$/, "");
    if (c === peSortCol) th.textContent += peSortAsc ? " ▲" : " ▼";
  });

  const CAP = 2000;
  const shown = visible.slice(0, CAP);
  const parts = [];
  for (const r of shown) {
    const selCls = r.id === selBid ? " selected" : "";
    const t = cIdx >= 0 ? r._tpds[cIdx] : r.max_abs_tpds;
    const tStr = (t == null) ? "—" : t.toFixed(3);
    parts.push(
      `<tr class="pe-row${selCls}" data-bid="${r.id}">` +
      `<td>${r.receiver}</td>` +
      `<td>${r.Receptor}</td>` +
      `<td>${r.EM}</td>` +
      `<td>${r.Target}</td>` +
      `<td>${tStr}</td>` +
      `<td>${r.sig_count}</td>` +
      `<td>${r.n_senders_sig}/${r.n_senders}</td>` +
      `<td>${r.max_abs_tpds == null ? "—" : r.max_abs_tpds.toFixed(3)}</td>` +
      `</tr>`
    );
  }
  tbody.innerHTML = parts.join("");
  const countEl = document.getElementById("pe-count");
  if (countEl) {
    const cap = visible.length > CAP ? ` (first ${CAP} shown)` : "";
    countEl.textContent = `${visible.length.toLocaleString()} / ${_peRows.length.toLocaleString()} backbones${cap}`;
  }
}

function _updatePathwayRowSelection(bid) {
  _updateRowSelection("#pe-table", "pe-row", "data-bid", bid);
}

function renderPathwayDetail(backbone_id) {
  const el = document.getElementById("pe-detail");
  if (!el) return;
  if (backbone_id == null) {
    el.innerHTML = '<div class="muted">Select a backbone to see details.</div>';
    return;
  }
  _ensurePathwayIndexes();
  const BB = PAYLOAD.backbones;
  const i = _backboneIdxById.get(backbone_id);
  if (i == null) {
    el.innerHTML = '<div class="muted">Backbone not found.</div>';
    return;
  }
  const receiver = RECEIVERS[BB.receiver_id[i]];
  const rcp = BB.Receptor[i] || "—";
  const em = BB.EM[i] || "—";
  const tgt = BB.Target[i] || "—";
  const sigMask = BB.significant_both_mask[i];
  const chips = CONTRASTS.map((c, ci) => {
    const on = ((sigMask >> ci) & 1) ? " on" : "";
    return `<span class="pe-chip${on}">${c}</span>`;
  }).join("");
  const nSendSig = BB.n_senders_significant[i];
  const nSend = BB.n_senders[i];

  el.innerHTML =
    `<h3>${rcp} → ${em} → ${tgt}</h3>` +
    `<div class="meta">Receiver: ${receiver} · Senders: ${nSendSig}/${nSend} sig</div>` +
    `<h4>Sig-both by contrast</h4><div>${chips}</div>` +
    `<h4>TPDS across contrasts</h4><div id="pe-detail-cross"></div>` +
    `<h4>Driving kinases</h4><div id="pe-detail-kinases" class="muted">loading…</div>`;

  const tpds = CONTRASTS.map(c => BB["mean_tpds_" + c][i]);
  const barColors = tpds.map(v => {
    if (v == null || v === 0) return "#cfd8dc";
    return v > 0 ? "var(--up-red)" : "var(--down-blue)";
  });
  const outlines = CONTRASTS.map((_, ci) =>
    ((sigMask >> ci) & 1) ? "#000" : "rgba(0,0,0,0)");
  Plotly.react("pe-detail-cross", [{
    type: "bar", x: CONTRASTS, y: tpds.map(v => v == null ? 0 : v),
    marker: { color: barColors, line: { color: outlines, width: 1.5 } },
    hovertemplate: "%{x}<br>TPDS %{y:.3f}<extra></extra>",
  }], {
    margin:{l:40,r:10,t:6,b:60}, height:180,
    yaxis:{zeroline:true, zerolinecolor:"#bbb"},
    xaxis:{tickangle:-35},
  }, {displaylogo:false, responsive:true});

  renderPathwayKinases(backbone_id);
}

async function renderPathwayKinases(backbone_id) {
  const container = document.getElementById("pe-detail-kinases");
  if (!container) return;
  _ensurePathwayIndexes();
  const bi = _backboneIdxById.get(backbone_id);
  if (bi == null) {
    container.innerHTML = '<div class="muted">Backbone not found.</div>';
    return;
  }
  if (PAYLOAD.backbones.significant_both_mask[bi] === 0) {
    container.innerHTML = '<div class="muted">No significant kinase edges.</div>';
    return;
  }
  let rows;
  try {
    rows = await SliceCache.backboneEdges(backbone_id);
  } catch (e) {
    if (Store.state.selection.backbone !== backbone_id) return;
    container.innerHTML = `<div class="muted">Failed to load: ${e.message}</div>`;
    return;
  }
  if (Store.state.selection.backbone !== backbone_id) return;

  const f = Store.state.filters;
  const cIdx = CONTRASTS.indexOf(f.contrast);
  const filtered = (cIdx >= 0)
    ? rows.filter(r => r.contrast_id === cIdx)
    : rows;

  const byK = new Map();
  for (const r of filtered) {
    let g = byK.get(r.kinase_id);
    if (!g) { g = { sum_abs:0, net:0, up:0, down:0, n:0 }; byK.set(r.kinase_id, g); }
    const s = r.support_contribution;
    g.sum_abs += Math.abs(s);
    g.net += s;
    if (r.concordance > 0) g.up++;
    else if (r.concordance < 0) g.down++;
    g.n++;
  }
  const groups = Array.from(byK.entries()).map(([kid, g]) => ({ kid, ...g }));
  groups.sort((a, b) => b.sum_abs - a.sum_abs);

  _ensureKinaseIdx();
  const K = PAYLOAD.kinases;
  const famMap = META.familyMap || {};

  const TOP = 200;
  const shown = groups.slice(0, TOP);
  const header = cIdx >= 0
    ? `Showing ${shown.length} of ${groups.length} kinases (contrast ${f.contrast}).`
    : `Showing ${shown.length} of ${groups.length} kinases (all contrasts).`;
  const parts = [
    `<div class="muted">${header}</div>`,
    '<table class="data-table"><thead><tr>',
    '<th>Kinase</th><th>Family</th><th>Σ|Support|</th>',
    '<th>Net</th><th>Conc.</th><th>#Edges</th>',
    '</tr></thead><tbody>',
  ];
  for (const g of shown) {
    const kIdx = _kinaseIdxById.get(g.kid);
    const name = kIdx != null ? K.name[kIdx] : `kid:${g.kid}`;
    const fam = famMap[name] || "";
    const conc = (g.up > g.down) ? "↑" : (g.down > g.up ? "↓" : "—");
    parts.push(
      `<tr><td>${name}</td><td>${fam}</td>` +
      `<td>${g.sum_abs.toFixed(3)}</td>` +
      `<td>${g.net.toFixed(3)}</td>` +
      `<td>${conc} (${g.up}/${g.down})</td>` +
      `<td>${g.n}</td></tr>`
    );
  }
  parts.push("</tbody></table>");
  container.innerHTML = parts.join("");
}

function wirePathwayTable() {
  const tbl = document.getElementById("pe-table");
  if (!tbl) return;
  tbl.querySelectorAll("thead th").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (peSortCol === col) peSortAsc = !peSortAsc;
      else { peSortCol = col; peSortAsc = false; }
      renderPathwayExplorer();
    });
  });
  tbl.querySelector("tbody").addEventListener("click", ev => {
    const tr = ev.target.closest("tr.pe-row");
    if (!tr) return;
    const bid = parseInt(tr.dataset.bid, 10);
    Store.dispatch({type:"SET_SELECTION", key:"backbone", value: bid});
  });
  const search = document.getElementById("pe-search");
  if (search) search.addEventListener("input", ev => {
    const val = ev.target.value;
    if (_peSearchTimer) clearTimeout(_peSearchTimer);
    _peSearchTimer = setTimeout(() => {
      peSearch = val;
      renderPathwayExplorer();
    }, 250);
  });
  const sig = document.getElementById("pe-sig-only");
  if (sig) sig.addEventListener("change", ev => {
    peSigOnly = ev.target.checked;
    renderPathwayExplorer();
  });
}

// ---------------------------------------------------------------------------
// Pathway Graph (Cytoscape) — aggregates filtered backbones into an
// R → EM → T node DAG where each node is a unique gene across many backbones.
// ---------------------------------------------------------------------------
const GRAPH_MAX_NODES = 600;
const GRAPH_COLORS = { "Receptor":"#43a047", "EM":"#fb8c00", "Target":"#5c6bc0" };

let _cyInstance = null;
let _nodeInfo = null;  // Map<nodeId, {bbs:number[], scoreSum, scoreN, nUp, nDown}>

function _destroyCy() {
  if (_cyInstance) { try { _cyInstance.destroy(); } catch(e) {} _cyInstance = null; }
  _nodeInfo = null;
}

function _graphPlaceholder(msg) {
  const el = document.getElementById("cy");
  if (!el) return;
  el.innerHTML = '<div class="graph-placeholder">' + msg + "</div>";
}

function _buildGraphData(indices, contrast) {
  const BB = PAYLOAD.backbones;
  const scoreCol = BB["observed_score_" + contrast];
  const tpdsCol = BB["mean_tpds_" + contrast];
  const nodeDeg = new Map();
  const nodeType = new Map();
  const nodeInfo = new Map();
  const edgeScores = new Map();
  const edgeTpds = new Map();
  const edgeCounts = new Map();

  for (const i of indices) {
    const bid = BB.id[i];
    const rGene = BB.Receptor[i];
    const emGene = BB.EM[i];
    const tGene = BB.Target[i];
    const rId = "R:" + rGene;
    const eId = "E:" + emGene;
    const tId = "T:" + tGene;
    const score = scoreCol ? scoreCol[i] : null;
    const tpds = tpdsCol ? tpdsCol[i] : null;

    for (const [nid, type] of [[rId, "Receptor"], [eId, "EM"], [tId, "Target"]]) {
      nodeDeg.set(nid, (nodeDeg.get(nid) || 0) + 1);
      if (!nodeType.has(nid)) nodeType.set(nid, type);
      let info = nodeInfo.get(nid);
      if (!info) {
        info = {bbs:[], scoreSum:0, scoreN:0, nUp:0, nDown:0};
        nodeInfo.set(nid, info);
      }
      info.bbs.push(bid);
      if (score != null) { info.scoreSum += score; info.scoreN++; }
      if (tpds != null) { if (tpds > 0) info.nUp++; else if (tpds < 0) info.nDown++; }
    }

    const rek = rId + ">" + eId;
    const etk = eId + ">" + tId;
    const s = (score == null) ? 0 : score;
    const t = (tpds == null) ? 0 : tpds;
    edgeScores.set(rek, Math.max(edgeScores.get(rek) || 0, s));
    edgeScores.set(etk, Math.max(edgeScores.get(etk) || 0, s));
    edgeTpds.set(rek, (edgeTpds.get(rek) || 0) + t);
    edgeTpds.set(etk, (edgeTpds.get(etk) || 0) + t);
    edgeCounts.set(rek, (edgeCounts.get(rek) || 0) + 1);
    edgeCounts.set(etk, (edgeCounts.get(etk) || 0) + 1);
  }

  // Min-degree filter
  const minDeg = Store.state.view.graphMinDegree | 0;
  let keepIds = [...nodeDeg.keys()].filter(id => nodeDeg.get(id) >= minDeg);
  // Node cap (degree-sorted)
  if (keepIds.length > GRAPH_MAX_NODES) {
    keepIds.sort((a,b) => nodeDeg.get(b) - nodeDeg.get(a));
    keepIds = keepIds.slice(0, GRAPH_MAX_NODES);
  }
  const keep = new Set(keepIds);

  // Edges only where both endpoints survive
  const maxDeg = keepIds.reduce((m, id) => Math.max(m, nodeDeg.get(id)), 1);
  const maxScore = [...edgeScores.values()].reduce((m,v) => Math.max(m,v), 0) || 1;

  const nodes = keepIds.map(id => {
    const type = nodeType.get(id);
    const deg = nodeDeg.get(id);
    const sz = 10 + 30 * Math.sqrt(deg / maxDeg);
    const rank = type === "Receptor" ? 0 : type === "EM" ? 1 : 2;
    return { data: {
      id, label: id.slice(2), type, deg, size: sz,
      color: GRAPH_COLORS[type], rank,
    }};
  });

  const edges = [];
  for (const [key, score] of edgeScores.entries()) {
    const [src, tgt] = key.split(">");
    if (!keep.has(src) || !keep.has(tgt)) continue;
    const count = edgeCounts.get(key) || 1;
    const avgTpds = (edgeTpds.get(key) || 0) / count;
    const w = 0.5 + 3 * (score / maxScore);
    const op = 0.2 + 0.6 * (score / maxScore);
    const col = avgTpds > 0 ? "#c62828"
              : avgTpds < 0 ? "#1565c0" : "#999";
    edges.push({ data: {
      id: key, source: src, target: tgt,
      score, width: w, opacity: op, edgeColor: col,
    }});
  }

  const finalInfo = new Map();
  for (const id of keepIds) finalInfo.set(id, nodeInfo.get(id));

  return { nodes, edges, nodeInfo: finalInfo,
           totalNodes: nodeDeg.size, keptNodes: keepIds.length };
}

function _applyFlowSnap(cy) {
  const w = cy.width() || 800;
  const cols = { "Receptor": w * 0.15, "EM": w * 0.50, "Target": w * 0.85 };
  cy.nodes().forEach(n => {
    const xTarget = cols[n.data("type")];
    const xCur = n.position("x");
    n.position("x", xCur * 0.15 + xTarget * 0.85);
  });
}

function _layoutConfig(layoutName, nNodes) {
  if (layoutName === "concentric") {
    return { name:"concentric",
             concentric: node => 3 - (node.data("rank") || 0),
             levelWidth: () => 1,
             minNodeSpacing: 8, animate:false };
  }
  const cose = { name:"cose", animate:false, randomize:true,
                 nodeRepulsion: () => nNodes > 200 ? 80000 : 40000,
                 idealEdgeLength: () => nNodes > 200
                   ? (layoutName === "flow" ? 60 : 50)
                   : (layoutName === "flow" ? 80 : 70),
                 gravity: layoutName === "flow" ? 0.3 : 0.25,
                 nodeOverlap:20 };
  return cose;
}

function _renderNodeDetail(nodeData) {
  const det = document.getElementById("graph-detail");
  if (!det) return;
  const nodeId = nodeData.id;
  const info = (_nodeInfo && _nodeInfo.get(nodeId))
    || {bbs:[], scoreSum:0, scoreN:0, nUp:0, nDown:0};
  const avgScore = info.scoreN ? (info.scoreSum / info.scoreN) : 0;
  det.innerHTML = "<h3>" + nodeData.label
    + ' <span class="meta">(' + nodeData.type + ")</span></h3>"
    + '<div class="meta">Backbones: ' + nodeData.deg
    + " &middot; avg score: " + avgScore.toFixed(3)
    + " &middot; ↑" + info.nUp + " / ↓" + info.nDown + "</div>"
    + '<button id="graph-filter-btn" class="chip" style="margin-top:8px;">'
    + "Filter Pathway Explorer to this node</button>";
  const btn = document.getElementById("graph-filter-btn");
  if (btn) btn.addEventListener("click", () => {
    const uniq = [...new Set(info.bbs)];
    Store.dispatch({type:"SET_FILTER", key:"graphNodeIds", value: uniq});
    Store.dispatch({type:"SET_VIEW", key:"activeTab", value:"pathway"});
  });
}

function renderGraph() {
  const el = document.getElementById("cy");
  if (!el) return;
  const contrast = Store.state.filters.contrast;
  if (contrast === "ALL") {
    _destroyCy();
    _graphPlaceholder("Select a single contrast to view the network graph.");
    const stats = document.getElementById("graph-stats");
    if (stats) stats.textContent = "";
    return;
  }
  el.innerHTML = "";  // clear any placeholder text
  const indices = getFilteredIndices();
  const built = _buildGraphData(indices, contrast);
  _nodeInfo = built.nodeInfo;
  const stats = document.getElementById("graph-stats");
  if (stats) {
    stats.textContent = built.keptNodes + " / " + built.totalNodes
      + " nodes (min-deg " + Store.state.view.graphMinDegree
      + (built.totalNodes > built.keptNodes
         ? ", degree-capped at " + GRAPH_MAX_NODES : "")
      + "), " + built.edges.length + " edges";
  }
  if (!built.nodes.length) {
    _destroyCy();
    _graphPlaceholder("No backbones for the current filters.");
    return;
  }

  _destroyCy();
  const layoutName = Store.state.view.graphLayout || "concentric";
  const nNodes = built.nodes.length;
  const layoutCfg = _layoutConfig(layoutName, nNodes);
  _cyInstance = cytoscape({
    container: el,
    elements: { nodes: built.nodes, edges: built.edges },
    style: [
      { selector:"node", style: {
        label:"data(label)", width:"data(size)", height:"data(size)",
        "background-color":"data(color)", "font-size":8,
        "text-valign":"bottom", "text-margin-y":4,
        "text-outline-color":"#fff", "text-outline-width":1,
        "min-zoomed-font-size":6,
      }},
      { selector:"edge", style: {
        width:"data(width)", "line-color":"data(edgeColor)",
        "target-arrow-color":"data(edgeColor)",
        "target-arrow-shape":"triangle", "curve-style":"bezier",
        opacity:"data(opacity)", "arrow-scale":0.6,
      }},
      { selector:"node.highlighted", style: {
        "border-width":3, "border-color":"#e53935",
        "font-weight":"bold", "font-size":10, "z-index":999,
      }},
      { selector:"node.faded", style: { opacity:0.15 } },
      { selector:"edge.faded", style: { opacity:0.05 } },
      { selector:"node.focus-center", style: {
        "border-width":4, "border-color":"#ff6f00", "border-style":"double",
      }},
    ],
    layout: layoutCfg,
    wheelSensitivity: 0.3,
  });
  if (layoutName === "flow") {
    _cyInstance.one("layoutstop", () => _applyFlowSnap(_cyInstance));
  }

  _cyInstance.on("tap", "node", evt => {
    const n = evt.target;
    _cyInstance.elements().removeClass("highlighted faded focus-center");
    const nbh = n.closedNeighborhood();
    _cyInstance.elements().not(nbh).addClass("faded");
    nbh.nodes().addClass("highlighted");
    n.addClass("focus-center");
    _renderNodeDetail(n.data());
  });
  _cyInstance.on("tap", evt => {
    if (evt.target === _cyInstance) {
      _cyInstance.elements().removeClass("highlighted faded focus-center");
      const det = document.getElementById("graph-detail");
      if (det) det.innerHTML = '<div class="muted">Click a node for details.</div>';
    }
  });
}

function wireGraphControls() {
  const layoutSel = document.getElementById("graph-layout");
  if (layoutSel) {
    layoutSel.value = Store.state.view.graphLayout;
    layoutSel.addEventListener("change", ev => {
      Store.dispatch({type:"SET_VIEW", key:"graphLayout", value: ev.target.value});
    });
  }
  const degSel = document.getElementById("graph-min-degree");
  if (degSel) {
    degSel.value = String(Store.state.view.graphMinDegree);
    degSel.addEventListener("change", ev => {
      Store.dispatch({type:"SET_VIEW", key:"graphMinDegree",
                      value: parseInt(ev.target.value, 10)});
    });
  }
}

// ---------------------------------------------------------------------------
// Glossary
// ---------------------------------------------------------------------------
function syncGlossary() {
  document.getElementById("glossary-panel").classList.toggle(
    "open", Store.state.view.glossaryOpen);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function boot() {
  populateHeader();
  wireTabs();
  wireKinaseTable();
  wirePathwayTable();
  wireGraphControls();
  syncHeaderFromStore();
  syncTabsFromStore();
  syncGlossary();
  renderOverview();

  Store.subscribe((next, prev) => {
    const activeTab = next.view.activeTab;
    if (next.filters !== prev.filters) {
      syncHeaderFromStore();
      if (activeTab === "overview") renderOverview();
      if (activeTab === "kinase") {
        renderKinaseExplorer();
        if (next.selection.kinase != null)
          renderKinaseDetail(next.selection.kinase);
      }
      if (activeTab === "pathway") {
        renderPathwayExplorer();
        if (next.selection.backbone != null)
          renderPathwayDetail(next.selection.backbone);
      }
      if (activeTab === "graph") renderGraph();
    }
    if (next.selection.kinase !== prev.selection.kinase && activeTab === "kinase") {
      _updateKinaseRowSelection(next.selection.kinase);
      renderKinaseDetail(next.selection.kinase);
    }
    if (next.selection.backbone !== prev.selection.backbone && activeTab === "pathway") {
      _updatePathwayRowSelection(next.selection.backbone);
      renderPathwayDetail(next.selection.backbone);
    }
    if (next.view !== prev.view) {
      if (next.view.activeTab !== prev.view.activeTab) {
        syncTabsFromStore();
        if (activeTab === "kinase") {
          renderKinaseExplorer();
          if (next.selection.kinase != null)
            renderKinaseDetail(next.selection.kinase);
        }
        if (activeTab === "pathway") {
          renderPathwayExplorer();
          if (next.selection.backbone != null)
            renderPathwayDetail(next.selection.backbone);
        }
        if (activeTab === "graph") renderGraph();
        if (prev.view.activeTab === "graph" && activeTab !== "graph")
          _destroyCy();
      }
      if (next.view.glossaryOpen !== prev.view.glossaryOpen) syncGlossary();
      if (next.view.overviewMode !== prev.view.overviewMode &&
          activeTab === "overview") renderOverview();
      if ((next.view.graphLayout !== prev.view.graphLayout ||
           next.view.graphMinDegree !== prev.view.graphMinDegree) &&
          activeTab === "graph") renderGraph();
    }
  });
}

if (document.readyState === "loading")
  document.addEventListener("DOMContentLoaded", boot);
else boot();
</script>
</body>
</html>
"""


def write_html(payload: dict, json_str: str | None = None) -> dict:
    """Emit the unified viewer HTML at UNIFIED_VIEWER_DIR/index.html.

    Sibling dirs (edge_slices/, edge_summaries/) are written by
    build_edge_shards.py; this function only writes the HTML.
    """
    os.makedirs(UNIFIED_VIEWER_DIR, exist_ok=True)
    if json_str is None:
        json_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Escape </ so an embedded "</script>" in the JSON can't terminate the tag.
    safe = json_str.replace("</", "<\\/")
    html = HTML_TEMPLATE
    for sentinel, value in (
        ("__APP_COLOR__", config.DISEASE_COLORS["App"]),
        ("__TAU_COLOR__", config.DISEASE_COLORS["Tau"]),
        ("__APTT_COLOR__", config.DISEASE_COLORS["ApTt"]),
        ("__PAYLOAD_SENTINEL__", safe),
    ):
        html = html.replace(sentinel, value)
    raw = html.encode("utf-8")
    with open(UNIFIED_VIEWER_HTML, "wb") as f:
        f.write(raw)
    return {"html_bytes": len(raw), "output": UNIFIED_VIEWER_HTML}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def validate(data: UnifiedData) -> str:
    """Write pipeline_notes/phase2_payload_report.md. Returns the md string."""
    errors: list[str] = []
    warnings: list[str] = []

    md = data.edge_metadata
    n_kinases = len(md["kinases"])
    n_celltypes = len(md["celltypes"])
    n_contrasts = len(md["contrasts"])

    # Payload size
    if not os.path.exists(PAYLOAD_JSON):
        errors.append(f"payload JSON missing: {PAYLOAD_JSON}")
        raw_bytes = gzip_bytes = 0
        payload = None
    else:
        raw_bytes = os.path.getsize(PAYLOAD_JSON)
        gzip_bytes = os.path.getsize(PAYLOAD_JSON_GZ) if os.path.exists(PAYLOAD_JSON_GZ) else 0
        with open(PAYLOAD_JSON) as f:
            payload = json.load(f)

    if raw_bytes >= 100 * 1024 * 1024:
        errors.append(f"payload raw {raw_bytes/1e6:.1f} MB exceeds 100 MB cap")
    if gzip_bytes >= 20 * 1024 * 1024:
        errors.append(f"payload gzip {gzip_bytes/1e6:.1f} MB exceeds 20 MB cap")

    # Edge-summary artifacts (Tier 1, embedded in payload)
    pk_summary_rows = pk_summary_bytes = 0
    pb_summary_rows = pb_summary_bytes = 0
    if not os.path.exists(PER_KINASE_SUMMARY):
        errors.append(f"per_kinase_summary missing: {PER_KINASE_SUMMARY}")
    else:
        pk_summary_rows = pq.ParquetFile(PER_KINASE_SUMMARY).metadata.num_rows
        pk_summary_bytes = os.path.getsize(PER_KINASE_SUMMARY)
    if not os.path.exists(PER_BACKBONE_SUMMARY):
        errors.append(f"per_backbone_summary missing: {PER_BACKBONE_SUMMARY}")
    else:
        pb_summary_rows = pq.ParquetFile(PER_BACKBONE_SUMMARY).metadata.num_rows
        pb_summary_bytes = os.path.getsize(PER_BACKBONE_SUMMARY)

    # Tier-2 slice directories (lazy-loaded; not embedded)
    n_kinase_slices = n_backbone_buckets = 0
    slices_bytes = 0
    if not os.path.isdir(EDGE_SLICES_KINASE_DIR):
        errors.append(f"kinase slice dir missing: {EDGE_SLICES_KINASE_DIR}")
    else:
        n_kinase_slices = sum(1 for f in os.listdir(EDGE_SLICES_KINASE_DIR)
                              if f.endswith(".parquet"))
        slices_bytes += sum(os.path.getsize(os.path.join(EDGE_SLICES_KINASE_DIR, f))
                            for f in os.listdir(EDGE_SLICES_KINASE_DIR)
                            if f.endswith(".parquet"))
    if not os.path.isdir(EDGE_SLICES_BACKBONE_DIR):
        errors.append(f"backbone slice dir missing: {EDGE_SLICES_BACKBONE_DIR}")
    else:
        n_backbone_buckets = sum(1 for f in os.listdir(EDGE_SLICES_BACKBONE_DIR)
                                 if f.endswith(".parquet"))
        slices_bytes += sum(os.path.getsize(os.path.join(EDGE_SLICES_BACKBONE_DIR, f))
                            for f in os.listdir(EDGE_SLICES_BACKBONE_DIR)
                            if f.endswith(".parquet"))

    # Structural
    if payload is not None:
        pk = payload["kinases"]
        pc_ = payload["celltypes"]
        pb = payload["backbones"]

        if len(pk["id"]) != n_kinases:
            errors.append(f"kinases rows {len(pk['id'])} != vocab {n_kinases}")
        if len(pc_["id"]) != n_celltypes:
            errors.append(f"celltypes rows {len(pc_['id'])} != vocab {n_celltypes}")
        # backbones[] only covers recurrence-aware backbones (superset of sig),
        # a strict subset of the edge parquet's 832K vocab. See docstring of
        # build_backbone_index() for why the two sets differ.
        if len(pb["id"]) > md["backbones_n"]:
            errors.append(
                f"backbones rows {len(pb['id'])} > edge vocab {md['backbones_n']}"
            )
        if len(set(pb["id"])) != len(pb["id"]):
            errors.append("duplicate backbone ids in payload")
        bb_index = build_backbone_index(data.backbone_recurrence)
        sig_bb_ids, _ = compute_sig_sets(data, bb_index)
        payload_bb_ids = set(pb["id"])
        missing_sig = [int(b) for b in sig_bb_ids if int(b) not in payload_bb_ids]
        if missing_sig:
            errors.append(
                f"{len(missing_sig)} sig backbone_id(s) absent from payload "
                f"backbones[] (first 3: {missing_sig[:3]})"
            )

        bad = [rid for rid in pb["receiver_id"]
               if rid < 0 or rid >= n_celltypes]
        if bad:
            errors.append(f"{len(bad)} orphan receiver_id(s) in backbones")

        n_bb = len(pb["id"])
        sig_mask_arr = np.asarray(pb["significant_both_mask"], dtype=np.int64)
        for ci, c in enumerate(md["contrasts"]):
            obs_key = f"observed_score_{c}"
            tpds_key = f"mean_tpds_{c}"
            if obs_key not in pb:
                errors.append(f"missing backbones[{obs_key}]")
                continue
            if len(pb[obs_key]) != n_bb:
                errors.append(
                    f"{obs_key} length {len(pb[obs_key])} != id length {n_bb}"
                )
                continue
            obs_notnull = np.array([v is not None for v in pb[obs_key]])
            tpds_notnull = np.array([v is not None for v in pb[tpds_key]])
            bad_rows = int(np.sum(obs_notnull & ~tpds_notnull))
            if bad_rows:
                errors.append(
                    f"{obs_key}: {bad_rows} rows have sig observed_score but "
                    f"no recurrence mean_tpds (sig should imply recurrence)"
                )
            sig_bit = ((sig_mask_arr >> ci) & 1).astype(bool)
            missing_obs = int(np.sum(sig_bit & ~obs_notnull))
            if missing_obs:
                errors.append(
                    f"{obs_key}: {missing_obs} sig-both rows missing "
                    f"observed_score"
                )

        # Tier-1 summary embedded in payload
        pks = payload.get("per_kinase_summary", {})
        if len(pks.get("kinase_id", [])) != pk_summary_rows:
            errors.append(
                f"per_kinase_summary rows in payload "
                f"{len(pks.get('kinase_id', []))} != parquet {pk_summary_rows}"
            )

        # Tier-2 slice reference
        esr = payload.get("edge_slice_ref", {})
        if esr.get("n_kinase_slices") != n_kinase_slices:
            errors.append(
                f"edge_slice_ref.n_kinase_slices={esr.get('n_kinase_slices')} "
                f"but {n_kinase_slices} parquet files on disk"
            )
        if esr.get("n_backbone_buckets") != n_backbone_buckets:
            errors.append(
                f"edge_slice_ref.n_backbone_buckets={esr.get('n_backbone_buckets')} "
                f"but {n_backbone_buckets} parquet files on disk"
            )

        # Every kinase_id referenced in per_kinase_summary must be in kinases[]
        summary_kids = set(pks.get("kinase_id", []))
        payload_kids = set(payload["kinases"]["id"])
        missing = summary_kids - payload_kids
        if missing:
            errors.append(
                f"{len(missing)} per_kinase_summary kinase_id(s) absent from "
                f"kinases[] (first 3: {sorted(missing)[:3]})"
            )

    peak_mb = _peak_rss_mb()

    lines = [
        "# Phase 2 Payload Report",
        "",
        f"_Generated {pd.Timestamp.utcnow().isoformat()}_",
        "",
        "## Sizes",
        "",
        f"- Payload JSON (raw): {raw_bytes/1e6:.2f} MB (cap 100)",
        f"- Payload JSON (gzip): {gzip_bytes/1e6:.2f} MB (cap 20)",
        f"- per_kinase_summary: {pk_summary_bytes/1e3:.1f} KB, {pk_summary_rows:,} rows",
        f"- per_backbone_summary: {pb_summary_bytes/1e3:.1f} KB, {pb_summary_rows:,} rows",
        f"- Edge slice shards total: {slices_bytes/1e6:.1f} MB "
        f"({n_kinase_slices} kinase + {n_backbone_buckets} backbone buckets)",
        "",
        "## Counts",
        "",
        f"- kinases: {n_kinases}",
        f"- celltypes: {n_celltypes}",
        f"- contrasts: {n_contrasts}",
        f"- backbones: {md['backbones_n']:,}",
        f"- full edges (Phase 1): {md['n_edges']:,}",
        "",
        "## Memory",
        "",
        f"- Peak RSS (this process): {peak_mb:.0f} MB",
        "",
        "## Invariants",
        "",
    ]
    if errors:
        lines.append("### FAIL")
        for e in errors:
            lines.append(f"- {e}")
    else:
        lines.append("All structural invariants pass.")
    if warnings:
        lines.append("")
        lines.append("### Warnings")
        for w in warnings:
            lines.append(f"- {w}")
    report = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, "w") as f:
        f.write(report)
    print(report)
    if errors:
        raise SystemExit(f"validation failed: {len(errors)} error(s)")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="Print input counts (Unit 2.1 smoke test)")
    ap.add_argument("--sidecar", action="store_true", help="Stream-filter full edges to _sig sidecar parquet")
    ap.add_argument("--payload", action="store_true", help="Write JSON payload (requires sidecar)")
    ap.add_argument("--build", action="store_true", help="Sidecar then payload")
    ap.add_argument("--html", action="store_true", help="Write unified_viewer.html (requires payload)")
    ap.add_argument("--validate", action="store_true", help="Write Phase 2 validation report")
    args = ap.parse_args(argv)

    if not any([args.summary, args.sidecar, args.payload, args.build,
                args.html, args.validate]):
        args.build = True
        args.html = True

    data = load_all_data()

    if args.summary:
        print(json.dumps(data.summary(), indent=2))

    if args.sidecar:
        bb_index = build_backbone_index(data.backbone_recurrence)
        write_sig_sidecar(data, bb_index)

    payload = None
    json_str = None
    if args.payload or args.build:
        if not os.path.exists(PER_KINASE_SUMMARY):
            raise SystemExit(
                f"edge shards missing; run: "
                f"pixi run python code/integration/adapters/build_edge_shards.py"
            )
        payload = build_payload(data)
        sizes = write_payload(payload)
        json_str = sizes.pop("json_str")
        print(f"  payload raw={sizes['raw_bytes']/1e6:.2f} MB "
              f"gzip={sizes['gzip_bytes']/1e6:.2f} MB")

    if args.html:
        if payload is None:
            if not os.path.exists(PAYLOAD_JSON):
                raise SystemExit(
                    f"payload missing at {PAYLOAD_JSON}; run --payload first"
                )
            with open(PAYLOAD_JSON) as f:
                json_str = f.read()
            payload = json.loads(json_str)
        info = write_html(payload, json_str=json_str)
        print(f"  html {info['html_bytes']/1e6:.2f} MB -> {info['output']}")

    if args.validate:
        validate(data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
