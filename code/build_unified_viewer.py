#!/usr/bin/env python3
"""Unified viewer builder: single entry point for kinase + pathway views.

Phase 2 deliverable. Produces:

  - kinase_backbone_edges_sig.parquet  — edges filtered to backbones that
    pass both null permutation tests (significant_both). Sidecar artifact
    that the HTML viewer will fetch at runtime in later phases.
  - unified_viewer.payload.json (+ .gz) — columnar JSON payload with
    stable integer IDs for kinases, celltypes, and backbones. Does NOT
    embed edges (the sig set is still ~10^7-10^8 rows). Carries an
    `edges_ref` pointer to the sidecar.

The full 7.14 GB / 2.23B-row edge parquet is streamed via
ParquetFile.iter_batches — it is never materialized in memory.

Usage:
    python code/build_unified_viewer.py --summary    # input row counts
    python code/build_unified_viewer.py --sidecar    # sig parquet only
    python code/build_unified_viewer.py --payload    # JSON only (needs sidecar)
    python code/build_unified_viewer.py --build      # sidecar + payload
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
PAYLOAD_JSON = os.path.join(UNIFIED_VIEWER_OUTPUT_DIR, "unified_viewer.payload.json")
PAYLOAD_JSON_GZ = PAYLOAD_JSON + ".gz"
SIDECAR_PARQUET = os.path.join(UNIFIED_VIEWER_OUTPUT_DIR, "kinase_backbone_edges_sig.parquet")
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

    pivot = rec.pivot_table(
        index="backbone_id", columns="contrast",
        values="mean_tpds", aggfunc="first",
    ).reindex(index=base["backbone_id"], columns=contrasts)
    for c in contrasts:
        cols[f"mean_tpds_{c}"] = pivot[c].astype(object).where(
            pivot[c].notna(), None
        ).tolist()
    cols["max_abs_tpds"] = base["backbone_id"].map(
        rec.groupby("backbone_id")["max_abs_tpds"].max()
    ).astype(float).tolist()

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

    sidecar_rows = None
    if os.path.exists(SIDECAR_PARQUET):
        sidecar_rows = pq.ParquetFile(SIDECAR_PARQUET).metadata.num_rows

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

    payload = {
        "kinases": kinases_slice,
        "celltypes": celltypes_slice,
        "backbones": backbones_slice,
        "edges_ref": {
            "path": os.path.relpath(SIDECAR_PARQUET, UNIFIED_VIEWER_OUTPUT_DIR),
            "n_rows": sidecar_rows,
            "schema_version": SCHEMA_VERSION,
        },
        "meta": meta,
    }
    return _sanitize(payload)


def write_payload(payload: dict) -> dict:
    os.makedirs(UNIFIED_VIEWER_OUTPUT_DIR, exist_ok=True)
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    with open(PAYLOAD_JSON, "wb") as f:
        f.write(raw)
    gz = gzip.compress(raw, compresslevel=6)
    with open(PAYLOAD_JSON_GZ, "wb") as f:
        f.write(gz)
    return {"raw_bytes": len(raw), "gzip_bytes": len(gz)}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _peak_rss_mb() -> float:
    # Linux: getrusage.ru_maxrss is in kB
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

    # Sidecar
    if not os.path.exists(SIDECAR_PARQUET):
        errors.append(f"sidecar parquet missing: {SIDECAR_PARQUET}")
        sidecar_rows = sidecar_bytes = 0
    else:
        sidecar_rows = pq.ParquetFile(SIDECAR_PARQUET).metadata.num_rows
        sidecar_bytes = os.path.getsize(SIDECAR_PARQUET)

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

        n_ref = payload["edges_ref"].get("n_rows")
        if n_ref != sidecar_rows:
            errors.append(f"edges_ref.n_rows={n_ref} but sidecar has {sidecar_rows}")

        if sidecar_rows and len(sig_bb_ids):
            # Quick check on a small sample
            sample = pq.ParquetFile(SIDECAR_PARQUET).read_row_group(
                0, columns=["backbone_id"])["backbone_id"].to_numpy()
            if len(sample) and not np.isin(sample[:1000], sig_bb_ids).all():
                warnings.append(
                    "sidecar row-group 0 contains backbone_ids outside sig set"
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
        f"- Sidecar parquet: {sidecar_bytes/1e6:.2f} MB, {sidecar_rows:,} rows",
        "",
        "## Counts",
        "",
        f"- kinases: {n_kinases}",
        f"- celltypes: {n_celltypes}",
        f"- contrasts: {n_contrasts}",
        f"- backbones: {md['backbones_n']:,}",
        f"- full edges (Phase 1): {md['n_edges']:,}",
        f"- sig edges (sidecar): {sidecar_rows:,} "
        f"({100 * sidecar_rows / max(md['n_edges'], 1):.2f}% of full)",
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
    ap.add_argument("--validate", action="store_true", help="Write Phase 2 validation report")
    args = ap.parse_args(argv)

    if not any([args.summary, args.sidecar, args.payload, args.build, args.validate]):
        args.summary = True

    data = load_all_data()

    if args.summary:
        print(json.dumps(data.summary(), indent=2))

    if args.sidecar or args.build:
        bb_index = build_backbone_index(data.backbone_recurrence)
        write_sig_sidecar(data, bb_index)

    if args.payload or args.build:
        if not os.path.exists(SIDECAR_PARQUET):
            raise SystemExit(
                f"sidecar missing at {SIDECAR_PARQUET}; run --sidecar first"
            )
        payload = build_payload(data)
        sizes = write_payload(payload)
        print(f"  payload raw={sizes['raw_bytes']/1e6:.2f} MB "
              f"gzip={sizes['gzip_bytes']/1e6:.2f} MB")

    if args.validate:
        validate(data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
