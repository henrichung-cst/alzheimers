#!/usr/bin/env python3
"""NSCLC 10x Expression Reference: cell-type specificity surface for the T-cell cohort.

Analogous to ``wmb_expression.py`` (mouse Song) and ``human_expression.py``
(human Mukesh), this produces per-cell-type kinase expression for the T-cell
cohort from the public 10x "Aggregate of 900k human NSCLC + normal-adjacent
cells" Flex dataset (Cell Ranger multi 7.1.0).

The 10x dataset ships NO cell-type labels — only unsupervised graphclust
clusters (86) + per-cluster diffexp. Cell types are therefore DERIVED in two
stages (see docs/plans/todo2_tcell_specificity_reference.md, REVISION 2026-06-19):

  --label-clusters : score each of the 86 graphclust clusters against canonical
                     lineage marker sets (T/NK, B/plasma, Myeloid, Epithelial,
                     Endothelial, Fibroblast, Mast) using the shipped diffexp
                     Mean Counts; assign argmax lineage. Writes the per-cluster
                     and per-barcode coarse label tables. Cheap (no matrix load).

  (between the two)  alz/ingest/nsclc_subset_tnk.py exports the T/NK barcodes to
                     a native 10x h5; alz/ingest/nsclc_projectils_map.R projects
                     them onto the CD8/CD4 ProjecTILs human refs (14 states),
                     writing projectils_predictions.csv. Heavy — run capped.

  --run            : stream the full 10x CSC matrix (h5py, indptr-bounded cell
                     chunks — NEVER full-load; 1.3 B nnz), assign each cell its
                     final label (ProjecTILs state for gated T cells, else the
                     coarse lineage), accumulate per (gene, cell_type). Writes
                     nsclc_kinase_expression.csv.

  --audit          : cross MEA-predicted kinases (FDR<0.25) against the reference;
                     report panel-covered kinases expressed nowhere. Writes
                     nsclc_kinase_audit.csv.

Usage:
    python alz/reference/nsclc_expression.py --label-clusters
    python alz/reference/nsclc_expression.py --run
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_CELLS = 5000

# Canonical lineage markers (HGNC symbols). T and NK share one coarse bucket
# (T_NK) which ProjecTILs later resolves into 14 CD8/CD4 functional states.
LINEAGE_MARKERS: Dict[str, list] = {
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

# Margin below which a cluster's argmax lineage is flagged ambiguous.
AMBIGUOUS_MARGIN = 0.30


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
    """Assign each graphclust cluster a coarse lineage from diffexp markers."""
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

    S = pd.DataFrame({lin: marker_score(ms) for lin, ms in LINEAGE_MARKERS.items()},
                     index=clusters)
    assign = S.idxmax(axis=1)
    top2 = S.apply(lambda r: r.nlargest(2).iloc[-1], axis=1)
    margin = (S.max(axis=1) - top2).round(3)

    gc = pd.read_csv(config.NSCLC_10X_GRAPHCLUST_FILE)
    counts = gc["Cluster"].value_counts()
    counts.index = [f"Cluster {i}" for i in counts.index]

    out = pd.DataFrame({
        "cluster": assign.index,
        "lineage": assign.values,
        "margin": margin.reindex(assign.index).values,
        "n_cells": counts.reindex(assign.index).fillna(0).astype(int).values,
    })
    out["ambiguous"] = out["margin"] < AMBIGUOUS_MARGIN
    out = out.sort_values("n_cells", ascending=False).reset_index(drop=True)
    out.to_csv(config.NSCLC_CLUSTER_LABELS_FILE, index=False)
    print(f"  Wrote {len(out)} cluster labels -> {config.NSCLC_CLUSTER_LABELS_FILE}")

    # Per-barcode coarse label (cluster -> lineage join)
    cluster_to_lineage = dict(zip(out["cluster"], out["lineage"]))
    cells = gc.rename(columns={"Barcode": "barcode", "Cluster": "cluster_num"})
    cells["cluster"] = "Cluster " + cells["cluster_num"].astype(str)
    cells["coarse_lineage"] = cells["cluster"].map(cluster_to_lineage)
    cells[["barcode", "cluster_num", "coarse_lineage"]].to_csv(
        config.NSCLC_CELL_LABELS_FILE, index=False)
    print(f"  Wrote {len(cells):,} per-barcode labels -> {config.NSCLC_CELL_LABELS_FILE}")

    # Summary
    agg = out.groupby("lineage")["n_cells"].agg(["sum", "count"]).sort_values(
        "sum", ascending=False)
    print("\n  Lineage partition (cells / clusters):")
    for lin, row in agg.iterrows():
        print(f"    {lin:<12} {int(row['sum']):>9,}  ({int(row['count'])} clusters)")
    n_amb = int(out["ambiguous"].sum())
    print(f"  T/NK cells (-> ProjecTILs): {agg.loc['T_NK','sum']:,}")
    print(f"  Ambiguous clusters (margin<{AMBIGUOUS_MARGIN}): {n_amb} "
          f"({out.loc[out.ambiguous,'n_cells'].sum():,} cells)")
    return out


# ---------------------------------------------------------------------------
# Final per-cell labels (coarse lineage + ProjecTILs refinement of T/NK)
# ---------------------------------------------------------------------------


def _final_cell_labels() -> pd.Series:
    """barcode -> final cell_type. Non-T/NK keep their coarse lineage; T/NK
    cells take their ProjecTILs functional.cluster when gated, else T_NK_other."""
    if not os.path.exists(config.NSCLC_CELL_LABELS_FILE):
        raise FileNotFoundError(
            f"{config.NSCLC_CELL_LABELS_FILE} missing — run --label-clusters first.")
    cells = pd.read_csv(config.NSCLC_CELL_LABELS_FILE)
    label = cells.set_index("barcode")["coarse_lineage"].copy()

    tnk_mask = label == "T_NK"
    label[tnk_mask] = "T_NK_other"

    if os.path.exists(config.NSCLC_PROJECTILS_PREDICTIONS_FILE):
        pred = pd.read_csv(config.NSCLC_PROJECTILS_PREDICTIONS_FILE)
        pred = pred[pred["functional.cluster"].notna()]
        proj = pred.set_index("barcode")["functional.cluster"]
        # only overwrite T/NK barcodes (ProjecTILs only ran on those)
        common = label.index.intersection(proj.index)
        common = common[label.loc[common] == "T_NK_other"]
        label.loc[common] = proj.loc[common].values
        print(f"  ProjecTILs refined {len(common):,} T/NK cells into "
              f"{proj.loc[common].nunique()} states; "
              f"{int((label=='T_NK_other').sum()):,} remain T_NK_other (ungated).")
    else:
        print("  WARNING: no ProjecTILs predictions found — all T/NK cells "
              "labeled T_NK_other (run nsclc_projectils_map.R for state resolution).")
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
        # CSC chunk: genes x (c1-c0); restrict to kinase rows then densify
        mat = csc_matrix((chunk_data, chunk_idx, local_indptr),
                         shape=(n_genes, c1 - c0))
        sub = mat[kinase_idx, :].toarray()          # n_k x chunk
        sub = np.log2(sub + 1.0)                    # mean log2(count+1)
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
                "binary_expressed": bool(mean_all[ci, k] > 1
                                         and frac_all[ci, k] > 0.10),
                "n_cells": int(n_cells_ct[ci]),
                "probe_covered": True,
            })
    df = pd.DataFrame(rows)

    # specificity = share of total mean log2 expr across cell types (per gene)
    df["specificity_score"] = 0.0
    for gene, gdf in df.groupby("gene_symbol"):
        tot = gdf["mean_log2_expression"].sum()
        if tot > 0:
            df.loc[gdf.index, "specificity_score"] = (
                gdf["mean_log2_expression"] / tot).round(6)

    df = df[["kinase_id", "gene_symbol", "cell_type", "mean_log2_expression",
             "fraction_cells_expressing", "binary_expressed",
             "specificity_score", "n_cells", "probe_covered"]]
    df.to_csv(config.NSCLC_KINASE_EXPRESSION_FILE, index=False)
    print(f"\n  Wrote {len(df):,} rows -> {config.NSCLC_KINASE_EXPRESSION_FILE}")

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
                   help="Coarse marker-based lineage labeling of graphclust clusters")
    g.add_argument("--run", action="store_true",
                   help="Stream the 10x matrix and compute per-cell-type expression")
    g.add_argument("--audit", action="store_true",
                   help="Cross MEA-predicted kinases against the reference")
    args = p.parse_args()
    if args.label_clusters:
        label_clusters()
    elif args.run:
        compute_expression()
    elif args.audit:
        audit()


if __name__ == "__main__":
    main()
