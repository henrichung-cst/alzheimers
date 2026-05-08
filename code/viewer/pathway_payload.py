"""Pathway-side payload builders for the unified viewer.

Extracted from `code/build_unified_viewer.py` so the pathway tabs can be
restructured (or removed) without touching the kinase code path. All nine
functions here are read by `build_payload` in the main script — every other
caller is internal to this module.

The functions take a `UnifiedData` instance (defined in the main script);
the type is referenced via a string-quoted annotation here to avoid
circular import.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .paths import (
    AGGREGATION_DIR,
    BACKBONE_REC_CSV,
    BACKBONE_VOCAB_CACHE,
    HERE,
)

if TYPE_CHECKING:
    from build_unified_viewer import UnifiedData


# ---------------------------------------------------------------------------
# Backbone vocabulary
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Significance set + sidecar streaming
# ---------------------------------------------------------------------------

def compute_sig_sets(data: "UnifiedData",
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
# Backbones slice (the big one — pivots recurrence to one row per backbone)
# ---------------------------------------------------------------------------

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


def _build_backbones_slice(data: "UnifiedData",
                           bb_index: pd.DataFrame) -> tuple[dict, list[str]]:
    """Pivot backbone_recurrence to one row per unique backbone, with
    per-contrast metrics, pathway provenance, and a significant_both mask."""
    contrasts = data.edge_metadata["contrasts"]
    cn_to_id = {c: i for i, c in enumerate(contrasts)}

    # Sender vocabulary (same rule as the legacy pathway viewer, archived under archive/code/integration/)
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
    _flatten_pivot(rec, "tpds_pvalue")
    _flatten_pivot(rec, "pathway_evidence_backbone")
    _flatten_pivot(rec, "n_expression_confirmed")
    _flatten_pivot(rec, "n_kinase_imputed")
    _flatten_pivot(rec, "imputed_nodes_union")
    cols["max_abs_tpds"] = base["backbone_id"].map(
        rec.groupby("backbone_id")["max_abs_tpds"].max()
    ).astype(float).tolist()
    _flatten_pivot(sig_bb, "observed_score")

    evidence_summary_map: dict[int, str] = {}
    imputed_nodes_summary_map: dict[int, str] = {}
    expr_total_map: dict[int, int] = {}
    kin_imp_total_map: dict[int, int] = {}
    for bid, sub in rec.groupby("backbone_id", sort=False):
        evid = set(sub["pathway_evidence_backbone"].dropna().astype(str))
        evid.discard("")
        if "mixed" in evid or ({"expression-confirmed", "kinase-imputed"} <= evid):
            evidence_summary_map[int(bid)] = "mixed"
        elif "kinase-imputed" in evid:
            evidence_summary_map[int(bid)] = "kinase-imputed"
        else:
            evidence_summary_map[int(bid)] = "expression-confirmed"

        nodes = []
        for raw in sub["imputed_nodes_union"].dropna().astype(str):
            for node in raw.split(";"):
                node = node.strip()
                if node and node not in nodes:
                    nodes.append(node)
        imputed_nodes_summary_map[int(bid)] = ";".join(nodes)
        expr_total_map[int(bid)] = int(pd.to_numeric(
            sub["n_expression_confirmed"], errors="coerce"
        ).fillna(0).sum())
        kin_imp_total_map[int(bid)] = int(pd.to_numeric(
            sub["n_kinase_imputed"], errors="coerce"
        ).fillna(0).sum())
    cols["all_contrasts_pathway_evidence"] = [
        evidence_summary_map.get(int(bid), "expression-confirmed")
        for bid in base["backbone_id"]
    ]
    cols["all_imputed_nodes_union"] = [
        imputed_nodes_summary_map.get(int(bid), "")
        for bid in base["backbone_id"]
    ]
    cols["all_n_expression_confirmed"] = [
        expr_total_map.get(int(bid), 0) for bid in base["backbone_id"]
    ]
    cols["all_n_kinase_imputed"] = [
        kin_imp_total_map.get(int(bid), 0) for bid in base["backbone_id"]
    ]

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

    # TPDS-magnitude significance encoded as three 9-bit masks at 0.01 / 0.05 /
    # 0.10. Lets the viewer toggle a TPDS-pvalue threshold without scanning
    # the full p-value column at filter time. Distinct from significant_both:
    # significant_both gates the chain over-representation (kinase specificity)
    # test; these masks gate the TPDS magnitude (is the chain's flux shift
    # distinguishable from zero) test.
    n_rows = len(base)
    for thresh, name in [(0.01, "tpds_sig_001_mask"),
                          (0.05, "tpds_sig_005_mask"),
                          (0.10, "tpds_sig_010_mask")]:
        masks = [0] * n_rows
        for ci, c in enumerate(contrasts):
            col = cols.get(f"tpds_pvalue_{c}")
            if col is None:
                continue
            bit = 1 << ci
            for i, p in enumerate(col):
                if p is not None and p < thresh:
                    masks[i] |= bit
        cols[name] = masks

    return cols, sender_order


# ---------------------------------------------------------------------------
# Aggregate slices (overview, tpds distribution, sender matrix)
# ---------------------------------------------------------------------------

def _build_overview_slice(data: "UnifiedData") -> dict:
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


def _build_tpds_distribution_slice() -> dict:
    """Per-(receiver, contrast) TPDS magnitude distribution computed from the
    *ungated* recurrence table (every enumerated chain, including non-
    significant ones).

    The backbone payload is hard-gated at load time to chains passing the
    permutation chain test, which means the per-chain `mean_tpds` columns on
    `BB` cannot answer 'how much pathway burden is in this regime' for the
    diffuse phase — late-Tau chains are absent from the payload entirely.
    This summary fills that gap: a small (≤22 × 9 = 198 cells) per-cell
    distribution of |TPDS| computed across all enumerated chains, suitable
    for the temporal magnitude view's mean_tpds and pct_up metrics.

    Schema: {"App_2mo|Astrocyte": {n, mean_abs, median_abs, p75, p95, p99,
    max, n_up, n_down}, ...}. Order is contrast|receiver to match the
    overview slice key convention.
    """
    df = pd.read_csv(
        BACKBONE_REC_CSV,
        usecols=["contrast", "receiver", "mean_tpds"],
    )
    df = df[df["mean_tpds"].notna()]
    abs_tpds = df["mean_tpds"].abs().to_numpy()
    df = df.assign(_abs=abs_tpds)
    out: dict[str, dict] = {}
    for (c, r), g in df.groupby(["contrast", "receiver"], sort=False):
        a = g["_abs"].to_numpy()
        s = g["mean_tpds"].to_numpy()
        if a.size == 0:
            continue
        out[f"{c}|{r}"] = {
            "n": int(a.size),
            "mean_abs": round(float(a.mean()), 6),
            "median_abs": round(float(np.median(a)), 6),
            "p75": round(float(np.quantile(a, 0.75)), 6),
            "p95": round(float(np.quantile(a, 0.95)), 6),
            "p99": round(float(np.quantile(a, 0.99)), 6),
            "max": round(float(a.max()), 6),
            "n_up": int((s > 0).sum()),
            "n_down": int((s < 0).sum()),
        }
    return out


def _build_sender_matrix_slice(data: "UnifiedData",
                               sender_order: list[str]) -> dict:
    """Pre-aggregate (contrast, sender, receiver) cells for the Sender×Receiver
    tab. Keyed "{contrast}|{sender_idx}|{receiver_idx}"; each value is
    {n, n_up, n_down, mean_tpds}. Grid is ≤ 9 × 22 × 22 cells.

    Every backbone recurrence row contributes to one (contrast, receiver) cell
    per sender named in its sender_list (the old pathway viewer's bitmask
    expansion, moved to build time).
    """
    rec = data.backbone_recurrence
    celltypes = data.edge_metadata["celltypes"]
    celltype_to_id = {c: i for i, c in enumerate(celltypes)}
    sender_to_id = {s: i for i, s in enumerate(sender_order)}

    out: dict[str, dict] = {}
    for row in rec.itertuples(index=False):
        c = row.contrast
        r = row.receiver
        rid = celltype_to_id.get(r)
        if rid is None:
            continue
        sl = row.sender_list
        if not isinstance(sl, str) or not sl:
            continue
        tpds = row.mean_tpds
        tpds_ok = isinstance(tpds, (int, float)) and np.isfinite(tpds)
        sign = 0 if not tpds_ok else (1 if tpds > 0 else (-1 if tpds < 0 else 0))
        for s in sl.split(","):
            s = s.strip()
            sid = sender_to_id.get(s)
            if sid is None:
                continue
            key = f"{c}|{sid}|{rid}"
            cell = out.get(key)
            if cell is None:
                cell = {"n": 0, "n_up": 0, "n_down": 0,
                        "_sum": 0.0, "_cnt": 0}
                out[key] = cell
            cell["n"] += 1
            if sign > 0:
                cell["n_up"] += 1
            elif sign < 0:
                cell["n_down"] += 1
            if tpds_ok:
                cell["_sum"] += float(tpds)
                cell["_cnt"] += 1
    for cell in out.values():
        cnt = cell.pop("_cnt")
        s = cell.pop("_sum")
        cell["mean_tpds"] = round(s / cnt, 4) if cnt else 0.0
    return out


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
