#!/usr/bin/env python3
"""NSCLC 10x Expression Reference: cell-type specificity surface for the T-cell cohort.

Analogous to ``wmb_expression.py`` (mouse Song) and ``human_expression.py``
(human Mukesh), this produces per-cell-type kinase expression for the T-cell
cohort from the public 10x "Aggregate of 900k human NSCLC + normal-adjacent
cells" Flex dataset (Cell Ranger multi 7.1.0).

The 10x dataset ships NO cell-type labels — only unsupervised graphclust
clusters (86) + per-cluster diffexp. Cell types are therefore DERIVED by two
complementary annotators (see docs/plans/todo2_tcell_specificity_reference.md):

  --label-clusters : score each of the 86 graphclust clusters against canonical
                     cell-type marker sets (T/NK, B/plasma, Myeloid, Epithelial,
                     Endothelial, Fibroblast, Mast) using the shipped diffexp
                     Mean Counts; assign argmax cell type. This labels the NON-T
                     compartment (which ProjecTILs structurally cannot) and
                     sanity-checks the T calls. Cheap (no matrix load).

  (heavy step)       alz/ingest/nsclc_projectils_map.R projects ALL 897,733
                     cells onto the CD8/CD4 ProjecTILs human refs (14 states).
                     scGate (inside filter.cells=TRUE) is the authoritative
                     T-cell gate — it accepts/rejects every cell, so a true T
                     cell the markers mislabeled is still recovered. Writes
                     projectils_predictions.csv. Run capped (see runner).

  --run            : stream the full 10x CSC matrix (h5py, indptr-bounded cell
                     chunks — NEVER full-load; 1.3 B nnz), assign each cell its
                     final label (ProjecTILs state where scGate gated it as a T
                     cell, else the coarse marker cell type), accumulate per
                     (gene, cell_type). Writes nsclc_kinase_expression.csv.

  --metrics        : recompute the standard attribution metric (detection-gated
                     concentration + enrichment breadth) from the existing
                     expression CSV — no matrix re-stream. Refreshes the metric
                     columns in nsclc_kinase_expression.csv (single canonical
                     output; nsclc_kinase_specificity.csv is no longer written).

  --audit          : cross MEA-predicted kinases (FDR<0.25) against the reference;
                     report panel-covered kinases expressed nowhere. Writes
                     nsclc_kinase_audit.csv.

Usage:
    python alz/reference/nsclc_expression.py --label-clusters
    python alz/reference/nsclc_expression.py --run
    python alz/reference/nsclc_expression.py --metrics
    python alz/reference/nsclc_expression.py --audit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.shared import config
from alz.reference.atlas import get_all_kinase_genes
from alz.cross_reference import specificity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Detection floor — the SINGLE cross-cohort gate (specificity.DETECTION_FRAC_MIN,
# 10%). No per-reference override: a kinase counts as detected in a cell type only
# when ≥10% of its cells express it, identically for the cohort scRNA and the NSCLC
# reference, so "detected ✓" never means two different things across axes.
NSCLC_DETECTION_FRAC_MIN = specificity.DETECTION_FRAC_MIN

CHUNK_CELLS = 5000

# Canonical cell-type markers (HGNC symbols). T and NK share one coarse bucket
# (T_NK) which ProjecTILs later resolves into 14 CD8/CD4 functional states.
CELLTYPE_MARKERS: Dict[str, list] = {
    "T_NK":        ["CD3D", "CD3E", "CD3G", "TRAC", "TRBC1", "TRBC2", "CD2",
                    "IL7R", "CD8A", "CD8B", "CD4", "FOXP3", "NKG7", "GNLY",
                    "KLRD1", "KLRF1", "NCAM1"],
    "B_plasma":    ["CD19", "MS4A1", "CD79A", "CD79B", "MZB1", "IGHG1",
                    "JCHAIN", "IGHM"],
    "Myeloid":     ["LYZ", "CD68", "CD14", "FCGR3A", "C1QA", "C1QB", "ITGAM",
                    "CD163", "S100A8", "S100A9", "CLEC9A", "LILRA4"],
    "Epithelial":  ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "SFTPC",
                    "SCGB1A1", "KRT5"],
    "Endothelial": ["PECAM1", "VWF", "CLDN5", "CDH5", "FLT1"],
    "Fibroblast":  ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRB", "ACTA2"],
    "Mast":        ["TPSAB1", "TPSB2", "CPA3", "MS4A2"],
}

# Margin below which a cluster's argmax cell type is flagged ambiguous.
AMBIGUOUS_MARGIN = 0.30

# Coarse non-T cell types (everything ProjecTILs cannot classify). Any cell_type
# NOT in this set is a ProjecTILs T state (or T_NK_other) and collapses to the
# single "T_NK" group for the specificity-share denominator (guardrail 2).
_COARSE_NON_T = {"B_plasma", "Myeloid", "Epithelial", "Endothelial",
                 "Fibroblast", "Mast", "unlabeled"}


def _spec_group(cell_type: str) -> str:
    return cell_type if cell_type in _COARSE_NON_T else "T_NK"


# ---------------------------------------------------------------------------
# Tarball member extraction (graphclust + diffexp)
# ---------------------------------------------------------------------------


def _ensure_analysis_members() -> None:
    """Extract the graphclust clusters + diffexp CSVs from analysis.tar.gz if
    not already on disk. Only the two needed members are extracted."""
    needed = [config.NSCLC_10X_GRAPHCLUST_FILE, config.NSCLC_10X_DIFFEXP_FILE]
    if all(os.path.exists(p) for p in needed):
        return
    if not os.path.exists(config.NSCLC_10X_ANALYSIS_TGZ):
        raise FileNotFoundError(
            f"{config.NSCLC_10X_ANALYSIS_TGZ} missing — run "
            f"`pixi run nsclc-ingest` first.")
    members = [
        "analysis/clustering/gene_expression_graphclust/clusters.csv",
        "analysis/diffexp/gene_expression_graphclust/differential_expression.csv",
    ]
    print(f"  Extracting {len(members)} analysis members from tarball ...")
    with tarfile.open(config.NSCLC_10X_ANALYSIS_TGZ, "r:gz") as tf:
        for m in members:
            tf.extract(m, path=config.NSCLC_10X_CACHE_DIR)


# ---------------------------------------------------------------------------
# Stage 1 — coarse cluster labeling
# ---------------------------------------------------------------------------


def label_clusters() -> pd.DataFrame:
    """Assign each graphclust cluster a coarse cell type from diffexp markers."""
    _ensure_analysis_members()
    os.makedirs(config.NSCLC_10X_CACHE_DIR, exist_ok=True)

    print("  Reading graphclust diffexp marker table ...")
    de = pd.read_csv(config.NSCLC_10X_DIFFEXP_FILE).rename(
        columns={"Feature Name": "gene"})
    mc_cols = [c for c in de.columns if c.endswith("Mean Counts")]
    clusters = [c.replace(" Mean Counts", "") for c in mc_cols]
    g2i = {g: i for i, g in enumerate(de["gene"].astype(str))}

    # per-gene specificity z-score across clusters (log1p mean counts)
    log_mc = np.log1p(de[mc_cols].to_numpy(dtype=float))
    z = (log_mc - log_mc.mean(axis=1, keepdims=True)) / (
        log_mc.std(axis=1, keepdims=True) + 1e-9)

    def marker_score(markers: list) -> np.ndarray:
        idx = [g2i[g] for g in markers if g in g2i]
        return z[idx, :].mean(axis=0) if idx else np.full(z.shape[1], -np.inf)

    S = pd.DataFrame({ct: marker_score(ms) for ct, ms in CELLTYPE_MARKERS.items()},
                     index=clusters)
    assign = S.idxmax(axis=1)
    top2 = S.apply(lambda r: r.nlargest(2).iloc[-1], axis=1)
    margin = (S.max(axis=1) - top2).round(3)

    gc = pd.read_csv(config.NSCLC_10X_GRAPHCLUST_FILE)
    counts = gc["Cluster"].value_counts()
    counts.index = [f"Cluster {i}" for i in counts.index]

    out = pd.DataFrame({
        "cluster": assign.index,
        "coarse_cell_type": assign.values,
        "margin": margin.reindex(assign.index).values,
        # T/NK marker z-score per cluster — a sanity column. ProjecTILs/scGate
        # (not these markers) gates T cells now, so a non-T cluster with a high
        # t_nk_score is something to cross-check against scGate's calls, not a
        # pre-filter input.
        "t_nk_score": S["T_NK"].reindex(assign.index).round(3).values,
        "n_cells": counts.reindex(assign.index).fillna(0).astype(int).values,
    })
    out["ambiguous"] = out["margin"] < AMBIGUOUS_MARGIN
    out = out.sort_values("n_cells", ascending=False).reset_index(drop=True)
    out.to_csv(config.NSCLC_CLUSTER_LABELS_FILE, index=False)
    print(f"  Wrote {len(out)} cluster labels -> {config.NSCLC_CLUSTER_LABELS_FILE}")

    # Per-barcode coarse marker cell type (cluster -> cell type join). This labels
    # the non-T compartment for the specificity denominator; ProjecTILs/scGate
    # overrides it for cells it gates as T (full-cohort projection).
    cluster_to_celltype = dict(zip(out["cluster"], out["coarse_cell_type"]))
    cells = gc.rename(columns={"Barcode": "barcode", "Cluster": "cluster_num"})
    cells["cluster"] = "Cluster " + cells["cluster_num"].astype(str)
    cells["coarse_cell_type"] = cells["cluster"].map(cluster_to_celltype)
    cells[["barcode", "cluster_num", "coarse_cell_type"]].to_csv(
        config.NSCLC_CELL_LABELS_FILE, index=False)
    print(f"  Wrote {len(cells):,} per-barcode coarse labels "
          f"-> {config.NSCLC_CELL_LABELS_FILE}")

    # Summary
    agg = out.groupby("coarse_cell_type")["n_cells"].agg(["sum", "count"]).sort_values(
        "sum", ascending=False)
    print("\n  Marker cell-type partition (cells / clusters):")
    for ct, row in agg.iterrows():
        print(f"    {ct:<12} {int(row['sum']):>9,}  ({int(row['count'])} clusters)")
    n_amb = int(out["ambiguous"].sum())
    print(f"  Marker T/NK cells: {agg.loc['T_NK','sum']:,} "
          f"(scGate makes the authoritative T call over ALL cells)")
    print(f"  Ambiguous clusters (margin<{AMBIGUOUS_MARGIN}): {n_amb} "
          f"({out.loc[out.ambiguous,'n_cells'].sum():,} cells)")
    return out


# ---------------------------------------------------------------------------
# Final per-cell labels (coarse cell type + ProjecTILs refinement of T/NK)
# ---------------------------------------------------------------------------


def _final_cell_labels() -> pd.Series:
    """barcode -> final cell_type. scGate (inside ProjecTILs, run over ALL cells)
    is the authoritative T gate: a gated cell takes its ProjecTILs functional
    state; every other cell keeps its coarse marker cell type. A marker-T cell
    scGate rejected falls to T_NK_other (T by markers, unplaceable by ProjecTILs)."""
    if not os.path.exists(config.NSCLC_CELL_LABELS_FILE):
        raise FileNotFoundError(
            f"{config.NSCLC_CELL_LABELS_FILE} missing — run --label-clusters first.")
    cells = pd.read_csv(config.NSCLC_CELL_LABELS_FILE)
    label = cells.set_index("barcode")["coarse_cell_type"].copy()
    marker_tnk = label == "T_NK"

    if not os.path.exists(config.NSCLC_PROJECTILS_PREDICTIONS_FILE):
        # No projection yet: marker-T cells have no functional state.
        label[marker_tnk] = "T_NK_other"
        print("  WARNING: no ProjecTILs predictions found — all marker T/NK cells "
              "labeled T_NK_other (run nsclc_projectils_map.R for state resolution).")
        return label

    pred = pd.read_csv(config.NSCLC_PROJECTILS_PREDICTIONS_FILE)
    gated = pred[pred["functional.cluster"].notna()].set_index("barcode")
    proj = gated["functional.cluster"]
    scgate_t = label.index.isin(proj.index)        # cells scGate accepted as T

    # Sanity check: marker-T vs scGate-T agreement (the markers' new role).
    both   = int((marker_tnk.to_numpy() & scgate_t).sum())   # agree: T
    only_m = int((marker_tnk.to_numpy() & ~scgate_t).sum())  # markers say T, scGate no
    only_s = int((~marker_tnk.to_numpy() & scgate_t).sum())  # scGate found T markers missed
    n_marker_t = int(marker_tnk.sum())
    print(f"  [sanity] marker-T vs scGate-T: agree={both:,}  "
          f"marker-only={only_m:,} (-> T_NK_other)  "
          f"scGate-only={only_s:,} (recall recovered by scGate)")
    if n_marker_t:
        print(f"           scGate confirms {100.0*both/n_marker_t:.1f}% of marker T/NK; "
              f"{only_s:,} additional T cells found outside marker-T clusters.")

    # scGate-gated cells (anywhere in the TME) take their ProjecTILs state.
    common = label.index.intersection(proj.index)
    label.loc[common] = proj.loc[common].values
    # Marker-T cells scGate rejected: T by markers, no state -> T_NK_other.
    rej = marker_tnk & ~label.index.isin(proj.index)
    label[rej] = "T_NK_other"
    print(f"  ProjecTILs gated {len(common):,} cells -> {proj.nunique()} states; "
          f"{int((label=='T_NK_other').sum()):,} marker-T cells remain T_NK_other.")
    return label


# ---------------------------------------------------------------------------
# Stage 2 — streaming expression accumulation (direct 10x CSC h5)
# ---------------------------------------------------------------------------


def compute_expression() -> pd.DataFrame:
    """Stream the 10x CSC matrix, accumulate per (kinase gene, cell_type)."""
    import h5py
    from scipy.sparse import csc_matrix

    os.makedirs(config.NSCLC_REFERENCE_OUTPUT_DIR, exist_ok=True)
    labels = _final_cell_labels()

    _, human_kinases = get_all_kinase_genes()

    print("  Opening 10x HDF5 (read-only, streamed) ...")
    f = h5py.File(config.NSCLC_10X_H5_FILE, "r")
    m = f["matrix"]
    n_genes, n_cells = (int(x) for x in m["shape"][:])
    feat_names = np.array([x.decode() for x in m["features"]["name"][:]])
    barcodes = np.array([x.decode() for x in m["barcodes"][:]])

    # kinase genes present in the Flex panel (probe_covered)
    feat_to_row = {g: i for i, g in enumerate(feat_names)}
    kinase_genes = sorted(set(human_kinases) & set(feat_to_row))
    kinase_idx = np.array([feat_to_row[g] for g in kinase_genes])
    print(f"  matrix: {n_genes} genes x {n_cells:,} cells; "
          f"{len(kinase_genes)} kinase genes panel-covered "
          f"(of {len(human_kinases)} kinome)")

    # per-cell final label aligned to matrix barcode order
    cell_label = labels.reindex(barcodes).fillna("unlabeled").to_numpy()
    cell_types = sorted(set(cell_label))
    ct_to_i = {ct: i for i, ct in enumerate(cell_types)}
    cell_ct_idx = np.array([ct_to_i[c] for c in cell_label])
    n_ct = len(cell_types)
    n_k = len(kinase_idx)

    expr_sum = np.zeros((n_ct, n_k), dtype=np.float64)
    nonzero = np.zeros((n_ct, n_k), dtype=np.int64)
    n_cells_ct = np.zeros(n_ct, dtype=np.int64)
    np.add.at(n_cells_ct, cell_ct_idx, 1)

    indptr = m["indptr"]              # len n_cells+1 (CSC: columns = cells)
    data_ds = m["data"]
    idx_ds = m["indices"]
    row_pos = {int(r): k for k, r in enumerate(kinase_idx)}  # gene row -> kinase col
    kinase_row_set = set(int(r) for r in kinase_idx)

    print(f"  Streaming {n_cells:,} cells in chunks of {CHUNK_CELLS} ...")
    for c0 in range(0, n_cells, CHUNK_CELLS):
        c1 = min(c0 + CHUNK_CELLS, n_cells)
        p0, p1 = int(indptr[c0]), int(indptr[c1])
        if p1 == p0:
            continue
        chunk_data = data_ds[p0:p1].astype(np.float64)
        chunk_idx = idx_ds[p0:p1]
        local_indptr = indptr[c0:c1 + 1][:] - p0
        # CSC chunk: genes x (c1-c0). Per-cell CPM normalization (÷ total UMI
        # over ALL genes × 1e6) then log2(CPM+1) — the Allen ABC / WMB-HBCA
        # convention, so mean_log2_expression and the mean>1 binary threshold
        # live on the SAME scale as the sibling references. Raw counts alone
        # (shallow Flex depth) would read systematically low and break
        # cross-reference comparability.
        mat = csc_matrix((chunk_data, chunk_idx, local_indptr),
                         shape=(n_genes, c1 - c0))
        cell_total = np.asarray(mat.sum(axis=0)).ravel()      # per-cell UMI, all genes
        raw = mat[kinase_idx, :].toarray()                    # n_k x chunk raw counts
        cpm = raw / np.maximum(cell_total, 1.0)[None, :] * 1e6
        sub = np.log2(cpm + 1.0)                              # log2(CPM+1)
        chunk_ct = cell_ct_idx[c0:c1]
        # scatter-add per cell type
        for ci in np.unique(chunk_ct):
            cmask = chunk_ct == ci
            block = sub[:, cmask]
            expr_sum[ci] += block.sum(axis=1)
            nonzero[ci] += (block > 0).sum(axis=1)
        if (c0 // CHUNK_CELLS) % 40 == 0:
            print(f"    {c1:,}/{n_cells:,} cells")
    f.close()

    # build output
    rows = []
    mean_all = np.divide(expr_sum, n_cells_ct[:, None],
                         out=np.zeros_like(expr_sum),
                         where=n_cells_ct[:, None] > 0)
    frac_all = np.divide(nonzero, n_cells_ct[:, None],
                         out=np.zeros_like(expr_sum),
                         where=n_cells_ct[:, None] > 0)
    for ci, ct in enumerate(cell_types):
        if n_cells_ct[ci] == 0:
            continue
        for k, gene in enumerate(kinase_genes):
            rows.append({
                "kinase_id": gene,
                "gene_symbol": gene,
                "cell_type": ct,
                "mean_log2_expression": round(float(mean_all[ci, k]), 6),
                "fraction_cells_expressing": round(float(frac_all[ci, k]), 6),
                # Shared 10% detection floor; re-aliased from `detected` in
                # _write_attribution_metrics after specificity.compute().
                "binary_expressed": bool(frac_all[ci, k] >= NSCLC_DETECTION_FRAC_MIN),
                "n_cells": int(n_cells_ct[ci]),
                "probe_covered": True,
            })
    df = pd.DataFrame(rows)

    # spec_group = coarse cell-type grouping. The 14 ProjecTILs T-states (+
    # T_NK_other) collapse to one "T_NK" group; the 6 non-T cell types stay
    # separate. This is the coarse resolution for the standard attribution
    # metric (the data answers "how specific" differently at native vs coarse
    # resolution — both are reported). fraction_cells_expressing / detection
    # stay at native cell-type resolution.
    df["spec_group"] = df["cell_type"].map(_spec_group)
    _write_attribution_metrics(df)

    with open(config.NSCLC_KINASE_EXPRESSION_FILE.replace(".csv", ".scope.json"), "w") as fh:
        json.dump({"dataset": "16plex_900k_32_NSCLC_multiplex",
                   "n_cells": int(n_cells), "n_cell_types": int((n_cells_ct > 0).sum()),
                   "n_kinase_genes_panel_covered": len(kinase_genes),
                   "cell_types": cell_types}, fh, indent=2)

    print("\n  Cells & expressed kinases per cell type:")
    for ci, ct in enumerate(cell_types):
        if n_cells_ct[ci] == 0:
            continue
        ne = int((df[df.cell_type == ct]["binary_expressed"]).sum())
        print(f"    {ct:<14} {int(n_cells_ct[ci]):>9,} cells  "
              f"{ne}/{len(kinase_genes)} kinases expressed")
    return df


# ---------------------------------------------------------------------------
# Standard attribution metric (detection-gated concentration + effective N)
# ---------------------------------------------------------------------------


def _write_attribution_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the standard attribution metric and write the canonical reference
    output. See alz/cross_reference/specificity.py and
    docs/plans/attribution/standard_attribution_metric.md for the definition.

      nsclc_kinase_expression.csv  — per (kinase, cell_type): raw facts (mean,
        fraction, n_cells) + native metrics (detected, linear_expression,
        specificity_score, concentration_tier) + per-kinase breadth columns:
          specificity_count  — N of 7 coarse groups whose cell-weighted
            group_fraction ≥ the shared detection floor (specificity.DETECTION_FRAC_MIN)
          expressing_groups  — comma-separated list of those groups
          top_coarse_group   — dominant coarse group (highest group_fraction)
          nsclc_enrichment_<state>  — per-ProjecTILs-state share of the kinase's
            total T_NK expression (14 states; non-T types are NaN)
          nsclc_enrichment_tier_<state>  — tier binned at multiples of 1/14
    """
    df = df.copy()
    # Re-entrant: keep only the raw per-(kinase, cell_type) facts. Every other
    # column below is recomputed, so a prior pass's derived columns must be dropped
    # or the spec_count/enrichment merges collide on suffixes (specificity_count_x/_y)
    # when --metrics is re-run on an already-metriced CSV.
    _RAW = ["kinase_id", "gene_symbol", "cell_type", "mean_log2_expression",
            "fraction_cells_expressing", "n_cells", "probe_covered"]
    df = df[[c for c in _RAW if c in df.columns]].copy()
    if "spec_group" not in df.columns:
        df["spec_group"] = df["cell_type"].map(_spec_group)

    per_label, per_group, per_gene = specificity.compute(
        df, gene_col="gene_symbol", label_col="cell_type",
        mean_log2_col="mean_log2_expression",
        frac_col="fraction_cells_expressing", ncells_col="n_cells",
        group_col="spec_group")

    # `detected`, `group_detected`, and the per-gene n_detected counts all come
    # from specificity.compute() at the single shared 10% floor — no per-reference
    # re-threshold. The CSV and JS read the `binary_expressed` alias of `detected`.
    per_label["binary_expressed"] = per_label["detected"]

    # Rename concentration → specificity_score (the JS reads this column).
    # The coarse group columns are derived in-memory below; only the native-type
    # rows are written to the canonical CSV.
    per_label = per_label.rename(columns={"concentration": "specificity_score"})

    # --- Per-kinase NSCLC specificity (N-of-7 breadth at the shared 10% floor) ---
    # group_fraction = cell-weighted average fraction_cells_expressing across the
    # native types in each coarse group. Specificity = prevalence count of groups
    # the kinase is detected in, at the same 10% floor as `detected`.
    spec_grp = per_label.copy()
    spec_grp["_wt"] = spec_grp["n_cells"] * spec_grp["fraction_cells_expressing"]
    _grp = spec_grp.groupby(["gene_symbol", "spec_group"]).agg(
        _wt_sum=("_wt", "sum"), _nc=("n_cells", "sum")).reset_index()
    _grp["group_fraction"] = _grp["_wt_sum"] / _grp["_nc"].clip(lower=1)
    _spec_count = (
        _grp[_grp["group_fraction"] >= specificity.DETECTION_FRAC_MIN]
        .groupby("gene_symbol")
        .agg(specificity_count=("spec_group", "count"),
             expressing_groups=("spec_group", lambda s: ",".join(sorted(s))))
        .reset_index())
    per_label = per_label.merge(_spec_count, on="gene_symbol", how="left")
    per_label["specificity_count"] = per_label["specificity_count"].fillna(0).astype(int)
    per_label["expressing_groups"] = per_label["expressing_groups"].fillna("")
    # Dominant coarse group = highest cell-weighted group_fraction (most prevalent),
    # whether or not it clears the floor — so an undetected kinase still reports the
    # cell type it tilts toward. Same prevalence basis as specificity_count.
    _top = (_grp.sort_values("group_fraction", ascending=False)
            .drop_duplicates("gene_symbol")[["gene_symbol", "spec_group"]]
            .rename(columns={"spec_group": "top_coarse_group"}))
    per_label = per_label.merge(_top, on="gene_symbol", how="left")
    per_label["top_coarse_group"] = per_label["top_coarse_group"].fillna("")

    # --- Per-kinase NSCLC enrichment (14-state ProjecTILs share) ---------------
    # Subset T_NK rows → per-state share across 14 ProjecTILs states.
    # Non-T native types get no enrichment columns (NaN).
    t_rows = per_label[per_label["spec_group"] == "T_NK"].copy()
    _gene_total = (t_rows.groupby("gene_symbol")["linear_expression"]
                   .sum().rename("_t_total"))
    t_rows = t_rows.join(_gene_total, on="gene_symbol")
    t_rows["_share"] = np.where(
        t_rows["_t_total"] > 0,
        t_rows["linear_expression"] / t_rows["_t_total"], 0.0)
    n_t_states = int(t_rows["cell_type"].nunique()) or 14
    _uniform = 1.0 / max(n_t_states, 1)
    t_rows["_tier"] = np.floor(t_rows["_share"] / _uniform).clip(upper=n_t_states).astype(int)
    # Pivot: one column per T-state.
    _share_wide = t_rows.pivot_table(
        index="gene_symbol", columns="cell_type",
        values=["_share", "_tier"], aggfunc="first")
    _share_wide.columns = [
        f"nsclc_enrichment_{col}" if lvl == "_share"
        else f"nsclc_enrichment_tier_{col}"
        for lvl, col in _share_wide.columns]
    per_label = per_label.merge(
        _share_wide.reset_index(), on="gene_symbol", how="left")

    expr = per_label[[
        "kinase_id", "gene_symbol", "cell_type", "spec_group",
        "mean_log2_expression", "fraction_cells_expressing", "binary_expressed",
        "n_cells", "probe_covered",
        "linear_expression", "detected", "specificity_score",
        "concentration_of_total", "concentration_tier",
        "specificity_count", "expressing_groups", "top_coarse_group",
        *[c for c in per_label.columns if c.startswith("nsclc_enrichment_")]]]
    expr.to_csv(config.NSCLC_KINASE_EXPRESSION_FILE, index=False)
    print(f"\n  Wrote {len(expr):,} rows -> {config.NSCLC_KINASE_EXPRESSION_FILE}")

    n_det_any = int((per_gene["n_detected_native"] > 0).sum())
    print(f"  Kinases detected in ≥1 cell type: {n_det_any} / {len(per_gene)}; "
          f"median effective # cell types (detected) = "
          f"{per_gene.loc[per_gene['n_detected_native'] > 0, 'effective_n_native'].median():.2f}")
    n_spec = int((_spec_count["specificity_count"] > 0).sum())
    print(f"  Kinases with specificity_count ≥ 1 (group_fraction ≥ "
          f"{specificity.DETECTION_FRAC_MIN:g}): {n_spec} / {len(per_gene)}")
    return expr


def compute_metrics() -> pd.DataFrame:
    """Recompute the standard attribution metric from the existing expression
    CSV — no matrix re-stream. Buildable now from prior --run output."""
    if not os.path.exists(config.NSCLC_KINASE_EXPRESSION_FILE):
        raise FileNotFoundError(
            f"{config.NSCLC_KINASE_EXPRESSION_FILE} missing — run --run first.")
    df = pd.read_csv(config.NSCLC_KINASE_EXPRESSION_FILE)
    needed = {"kinase_id", "gene_symbol", "cell_type", "mean_log2_expression",
              "fraction_cells_expressing", "binary_expressed", "n_cells",
              "probe_covered"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"{config.NSCLC_KINASE_EXPRESSION_FILE} missing columns {missing}")
    return _write_attribution_metrics(df)


# ---------------------------------------------------------------------------
# Stage 3 — MEA audit
# ---------------------------------------------------------------------------


def _mea_predicted_kinases(fdr_max: float = 0.25) -> set:
    """T-cell donor1 MEA kinases significant at FDR<fdr_max in any contrast."""
    mea_dir = os.path.join("outputs", "reports", "kinase_attribution_tcells",
                           "donor1", "mea")
    predicted = set()
    for fname in ("mea_timecourse.csv", "mea_timecourse_pY.csv"):
        path = os.path.join(mea_dir, fname)
        if not os.path.exists(path):
            continue
        t = pd.read_csv(path)
        kcol = next((c for c in ("kinase", "kinase_id", "Kinase") if c in t.columns), None)
        fcol = next((c for c in t.columns if c.lower() in ("fdr", "padj", "q_value", "qvalue")), None)
        if kcol is None:
            continue
        if fcol is not None:
            t = t[t[fcol] < fdr_max]
        predicted |= set(t[kcol].astype(str).str.upper())
    return predicted


def audit() -> pd.DataFrame:
    """Cross MEA-predicted kinases against the NSCLC reference expression."""
    if not os.path.exists(config.NSCLC_KINASE_EXPRESSION_FILE):
        raise FileNotFoundError(
            f"{config.NSCLC_KINASE_EXPRESSION_FILE} missing — run --run first.")
    expr = pd.read_csv(config.NSCLC_KINASE_EXPRESSION_FILE)

    # kinase abbreviation <-> gene symbol mapping
    kmap = pd.read_csv(config.MAPPING_CACHE_FILE)
    abbr_to_gene = dict(zip(kmap["kinase_abbreviation"].str.upper(),
                            kmap["gene_symbol"].str.upper()))

    _, human_kinases = get_all_kinase_genes()
    mea = _mea_predicted_kinases()

    # full kinome universe by gene symbol
    panel_genes = set(expr["gene_symbol"])
    rows = []
    for abbr in sorted(set(abbr_to_gene) | mea):
        gene = abbr_to_gene.get(abbr, abbr)
        sub = expr[expr["gene_symbol"] == gene]
        probe_covered = gene in panel_genes
        bexpr = sub["binary_expressed"].any() if len(sub) else False
        if len(sub):
            mx = sub.loc[sub["fraction_cells_expressing"].idxmax()]
            max_frac = float(mx["fraction_cells_expressing"])
            max_ct = str(mx["cell_type"])
            ct_list = ",".join(sorted(sub.loc[sub["binary_expressed"], "cell_type"]))
        else:
            max_frac, max_ct, ct_list = 0.0, "", ""
        is_mea = abbr in mea
        if not probe_covered:
            flag = "not_in_probe_panel"
        elif bexpr:
            flag = "expressed"
        else:
            flag = "not_expressed_in_panel"
        rows.append({
            "kinase": abbr,
            "gene_symbol": gene,
            "probe_covered": probe_covered,
            "binary_expressed_any": bool(bexpr),
            "expressed_cell_types": ct_list,
            "max_fraction_expressing": round(max_frac, 6),
            "max_fraction_cell_type": max_ct,
            "is_mea_predicted": is_mea,
            "audit_flag": flag,
        })
    adf = pd.DataFrame(rows).sort_values(
        ["is_mea_predicted", "binary_expressed_any", "kinase"],
        ascending=[False, True, True])
    os.makedirs(config.NSCLC_REFERENCE_OUTPUT_DIR, exist_ok=True)
    adf.to_csv(config.NSCLC_KINASE_AUDIT_FILE, index=False)

    mea_df = adf[adf["is_mea_predicted"]]
    panel = mea_df[mea_df["probe_covered"]]
    expressed = panel[panel["binary_expressed_any"]]
    absent = panel[~panel["binary_expressed_any"]]
    print(f"\nNSCLC reference audit: T-cell MEA kinases")
    print(f"  MEA-predicted (FDR<0.25): {len(mea_df)}")
    print(f"  Probe panel covered:      {len(panel)} / {len(mea_df)}")
    print(f"  Expressed in >=1 type:    {len(expressed)} / {len(panel)} (panel-covered)")
    print(f"  NOT expressed (panel-covered): {len(absent)}   <- the finding")
    if len(absent):
        print("    " + ", ".join(absent["kinase"].tolist()))
    print(f"\n  Wrote {len(adf)} rows -> {config.NSCLC_KINASE_AUDIT_FILE}")
    return adf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--label-clusters", action="store_true",
                   help="Coarse marker-based cell-type labeling of graphclust clusters")
    g.add_argument("--run", action="store_true",
                   help="Stream the 10x matrix and compute per-cell-type expression")
    g.add_argument("--metrics", action="store_true",
                   help="Recompute the standard attribution metric from the "
                        "existing expression CSV (no matrix re-stream)")
    g.add_argument("--audit", action="store_true",
                   help="Cross MEA-predicted kinases against the reference")
    args = p.parse_args()
    if args.label_clusters:
        label_clusters()
    elif args.run:
        compute_expression()
    elif args.metrics:
        compute_metrics()
    elif args.audit:
        audit()


if __name__ == "__main__":
    main()
