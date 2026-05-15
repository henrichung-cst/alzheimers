"""Hypothesis (C) test: scRNA dropout × n=1 at late timepoints.

For high-|TPDS| paths from ApTt_2mo in Astrocytes, check what fraction
of the 4 path genes (Ligand, Receptor, EM, Target) have non-zero
per-animal mean expression within the Astrocytes cluster, for each
animal × timepoint. Hypothesis (C) predicts:

- At 2mo (n=2 for WT/App/Tau): per-condition mean averages over 2
  animals, so even if one animal drops a gene, the other rescues it.
- At 4mo/6mo (n=1 everywhere): dropout in any one of the 4 genes
  collapses the product-form SigProb to zero.

We measure: per-(animal, path) "all 4 genes detected" rate.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.io as sio

REPO_ROOT = "/home/hchung/Projects/work/alzheimers"
INPUT_DIR = f"{REPO_ROOT}/data/incytr_factorial_inputs"
RECV = f"{REPO_ROOT}/outputs/reports/incytr_factorial_5xfad_kldata/receiver_cache/receiver=Astrocytes"
CLUSTER_MAP = f"{REPO_ROOT}/data/incytr/v2_46clusters/barcode_to_cluster.csv"
OUT = f"{REPO_ROOT}/outputs/reports/incytr_factorial_5xfad_kldata/diagnostics"
CLUSTER = "Astrocytes"
TOP_K = 200


def load_paths() -> pd.DataFrame:
    parts = sorted(glob.glob(f"{RECV}/*.parquet"))
    frames = []
    for p in parts:
        d = pq.ParquetFile(p).read(
            columns=["Ligand", "Receptor", "EM", "Target", "contrast", "TPDS",
                     "SigProb_WTyp_2mo", "SigProb_AppP_2mo", "SigProb_Ttau_2mo",
                     "SigProb_ApTt_2mo", "SigProb_WTyp_4mo", "SigProb_AppP_4mo",
                     "SigProb_Ttau_4mo", "SigProb_ApTt_4mo", "SigProb_WTyp_6mo",
                     "SigProb_AppP_6mo", "SigProb_Ttau_6mo", "SigProb_ApTt_6mo"]
        ).to_pandas()
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print("Loading paths …")
    paths = load_paths()
    print(f"  {len(paths):,} path×contrast rows")

    high_aptt2 = paths[(paths["contrast"] == "ApTt_2mo") & (paths["TPDS"].abs() >= 0.5)]
    print(f"  {len(high_aptt2):,} paths with |TPDS|>=0.5 at ApTt_2mo")
    top = high_aptt2.reindex(high_aptt2["TPDS"].abs().sort_values(ascending=False).index).head(TOP_K)
    gene_set = sorted(set(top["Ligand"]) | set(top["Receptor"]) | set(top["EM"]) | set(top["Target"]))
    print(f"  {len(gene_set):,} unique L/R/EM/T genes across top {TOP_K}")

    print("Loading expression matrix + metadata …")
    genes = pd.read_csv(f"{INPUT_DIR}/expression_genes.csv", header=None)[0].tolist()
    if genes[0].lower() in ("gene", "gene_symbol", "0"):
        genes = genes[1:]
    print(f"  genes vocab: {len(genes):,}")
    meta = pd.read_csv(f"{INPUT_DIR}/expression_metadata.csv").reset_index(drop=True)
    bcs = pd.read_csv(f"{INPUT_DIR}/expression_barcodes.csv", header=None)[0].tolist()
    if bcs[0].lower() in ("barcode", "0"):
        bcs = bcs[1:]

    cluster_map = pd.read_csv(CLUSTER_MAP)
    print(f"  barcode_to_cluster cols: {list(cluster_map.columns)}")
    bc2cl = dict(zip(cluster_map["barcode"], cluster_map["cluster_subclass"]))

    if "barcode" in meta.columns:
        meta["cluster"] = meta["barcode"].map(bc2cl)
    else:
        meta["cluster"] = meta.iloc[:, 0].map(bc2cl) if meta.columns[0] == "Unnamed: 0" else None

    if meta["cluster"].isna().all():
        meta["cluster"] = pd.Series(bcs).map(bc2cl).values
    print(f"  cluster coverage: {meta['cluster'].notna().sum():,}/{len(meta):,}")
    print(meta["cluster"].value_counts().head(10).to_string())

    astro_mask = (meta["cluster"] == CLUSTER).to_numpy()
    astro_idx = np.where(astro_mask)[0]
    print(f"  Astrocytes barcodes: {astro_mask.sum():,}")
    if astro_mask.sum() == 0:
        print("WARN: no Astrocytes barcodes found — check cluster naming")
        print("Available clusters:", meta["cluster"].dropna().unique()[:30])
        return

    print("Loading matrix and slicing to Astrocytes barcodes …")
    X = sio.mmread(f"{INPUT_DIR}/expression_matrix.mtx").tocsc()
    print(f"  matrix shape (genes × barcodes): {X.shape}")
    gene2idx = {g: i for i, g in enumerate(genes)}
    gene_idx = [gene2idx[g] for g in gene_set if g in gene2idx]
    print(f"  {len(gene_idx):,}/{len(gene_set):,} genes present in vocab")

    X_sub = X[gene_idx, :][:, astro_idx].toarray()
    astro_meta = meta.iloc[astro_idx].reset_index(drop=True)
    print(f"  Astrocytes submatrix: {X_sub.shape}")

    rows = []
    for animal, sub in astro_meta.groupby("animal_id"):
        bc_pos = sub.index.to_numpy()
        per_gene_nonzero_frac = (X_sub[:, bc_pos] > 0).mean(axis=1)
        per_gene_mean = X_sub[:, bc_pos].mean(axis=1)
        rows.append(pd.DataFrame({
            "animal_id": animal,
            "genotype": sub["genotype"].iloc[0],
            "timepoint": sub["timepoint"].iloc[0],
            "n_barcodes": len(bc_pos),
            "gene": [genes[i] for i in gene_idx],
            "nonzero_frac": per_gene_nonzero_frac,
            "mean_expr": per_gene_mean,
            "detected": per_gene_mean > 0,
        }))
    long = pd.concat(rows, ignore_index=True)

    per_animal = long.groupby(["timepoint", "genotype", "animal_id"]).agg(
        n_genes=("gene", "size"),
        n_detected=("detected", "sum"),
        n_barcodes=("n_barcodes", "first"),
    ).reset_index()
    per_animal["detected_frac"] = per_animal["n_detected"] / per_animal["n_genes"]
    print("\n=== Per-animal gene detection (top-200 path genes) in Astrocytes ===")
    print(per_animal.to_string(index=False))

    detect_map = long.set_index(["animal_id", "gene"])["detected"].to_dict()
    path_rows = []
    for _, p in top.iterrows():
        genes4 = [p["Ligand"], p["Receptor"], p["EM"], p["Target"]]
        for animal in per_animal["animal_id"].unique():
            dets = [int(detect_map.get((animal, g), 0)) for g in genes4]
            path_rows.append({
                "animal_id": animal,
                "Ligand": genes4[0], "Receptor": genes4[1],
                "EM": genes4[2], "Target": genes4[3],
                "n_detected_of_4": sum(dets),
            })
    pdf = pd.DataFrame(path_rows)
    pdf = pdf.merge(per_animal[["animal_id", "genotype", "timepoint"]], on="animal_id")

    print("\n=== Per (timepoint, genotype): %% of top-200 paths with all 4 genes detected ===")
    out = pdf.groupby(["timepoint", "genotype", "animal_id"]).agg(
        n_paths=("n_detected_of_4", "size"),
        frac_all4=("n_detected_of_4", lambda s: float((s == 4).mean())),
        frac_ge3=("n_detected_of_4", lambda s: float((s >= 3).mean())),
        frac_ge2=("n_detected_of_4", lambda s: float((s >= 2).mean())),
        mean_detected=("n_detected_of_4", "mean"),
    ).reset_index()
    print(out.to_string(index=False))

    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/dropout_coverage_per_animal.csv", index=False)
    per_animal.to_csv(f"{OUT}/dropout_coverage_gene_detection.csv", index=False)
    print(f"\nWrote {OUT}/dropout_coverage_per_animal.csv")


if __name__ == "__main__":
    main()
