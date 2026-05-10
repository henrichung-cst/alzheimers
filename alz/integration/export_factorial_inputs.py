"""Export Song snRNA-seq h5ad to the on-disk contract consumed by load.R.

Produces five files in OUT_DIR:
  expression_matrix.mtx     genes x cells, sparse
  expression_genes.csv      gene symbol per row
  expression_barcodes.csv   barcode per cell
  expression_metadata.csv   per-cell labels, animal_id, genotype, timepoint
  animal_metadata.csv       per-animal design matrix matching kinase_enrich.py OLS
"""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "alz"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as main_config  # noqa: E402
import config_integration as icfg  # noqa: E402


def build_design_row(genotype: str, timepoint: str) -> dict:
    geno = icfg.MUTANT_TO_DESIGN[genotype]
    time = icfg.TIMEPOINT_TO_DESIGN[timepoint]
    row = {"const": 1}
    row.update(geno)
    row.update(time)
    row["App_x_time4"] = geno["App"] * time["time_4mo"]
    row["App_x_time6"] = geno["App"] * time["time_6mo"]
    row["Tau_x_time4"] = geno["Tau"] * time["time_4mo"]
    row["Tau_x_time6"] = geno["Tau"] * time["time_6mo"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=icfg.FACTORIAL_INPUT_DIR)
    parser.add_argument("--h5ad", default=icfg.H5AD_PATH)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.h5ad} ...")
    adata = ad.read_h5ad(args.h5ad)
    print(f"  Full dataset: {adata.n_obs} cells x {adata.n_vars} genes")

    mask = (
        (adata.obs["sex"] == icfg.FACTORIAL_SEX)
        & (adata.obs["mutant"].isin(icfg.FACTORIAL_GENOTYPES))
        & (adata.obs["age"].isin(icfg.FACTORIAL_TIMEPOINTS))
    )
    adata = adata[mask].copy()
    print(f"  After sex/genotype/timepoint filter: {adata.n_obs} cells")

    if "subclass_prob" in adata.obs.columns:
        prob_mask = adata.obs["subclass_prob"] >= main_config.SONG_MIN_SUBCLASS_PROB
        adata = adata[prob_mask].copy()
        print(
            f"  After subclass_prob >= {main_config.SONG_MIN_SUBCLASS_PROB}: "
            f"{adata.n_obs} cells"
        )

    adata.obs["sea_ad_subclass"] = adata.obs["subclass_name"].map(
        main_config.SONG_SUBCLASS_MAP
    )
    unmapped = adata.obs["sea_ad_subclass"].isna().sum()
    if unmapped > 0:
        print(f"  Dropping {unmapped} cells with unmapped subclass names")
        adata = adata[adata.obs["sea_ad_subclass"].notna()].copy()
    print(f"  After subclass mapping: {adata.n_obs} cells")

    meta = pd.DataFrame(index=adata.obs.index)
    meta["labels"] = adata.obs["sea_ad_subclass"].values
    meta["animal_id"] = adata.obs["sample"].values
    meta["genotype"] = adata.obs["mutant"].values
    meta["timepoint"] = adata.obs["age"].values

    animals = meta[["animal_id", "genotype", "timepoint"]].drop_duplicates()
    print(f"\n  Animals: {len(animals)}")
    for geno in icfg.FACTORIAL_GENOTYPES:
        for tp in icfg.FACTORIAL_TIMEPOINTS:
            n_cells = ((meta["genotype"] == geno) & (meta["timepoint"] == tp)).sum()
            n_animals = animals[
                (animals["genotype"] == geno) & (animals["timepoint"] == tp)
            ].shape[0]
            print(f"    {geno:>4s} x {tp}: {n_cells:>5d} cells, {n_animals} animals")

    animal_meta_rows = []
    for _, row in animals.iterrows():
        design_row = build_design_row(row["genotype"], row["timepoint"])
        design_row["animal_id"] = row["animal_id"]
        design_row["genotype"] = row["genotype"]
        design_row["timepoint"] = row["timepoint"]
        animal_meta_rows.append(design_row)

    animal_meta = pd.DataFrame(animal_meta_rows)
    id_cols = ["animal_id", "genotype", "timepoint"]
    animal_meta = animal_meta[id_cols + icfg.DESIGN_COLUMNS]
    animal_meta = animal_meta.sort_values("animal_id").reset_index(drop=True)

    rank = np.linalg.matrix_rank(animal_meta[icfg.DESIGN_COLUMNS].values.astype(float))
    print(
        f"\n  Design matrix: {len(animal_meta)} animals x "
        f"{len(icfg.DESIGN_COLUMNS)} parameters, rank {rank}"
    )
    if rank < len(icfg.DESIGN_COLUMNS):
        raise SystemExit("Design matrix is rank-deficient; refusing to write outputs.")

    X = adata.X
    X_t = X.T.tocsc() if scipy.sparse.issparse(X) else scipy.sparse.csc_matrix(X.T)

    paths = {
        "expression_matrix.mtx": (scipy.io.mmwrite, X_t),
        "expression_genes.csv": (
            lambda p, df: df.to_csv(p, index=False),
            pd.DataFrame({"gene": adata.var_names}),
        ),
        "expression_barcodes.csv": (
            lambda p, df: df.to_csv(p, index=False),
            pd.DataFrame({"barcode": adata.obs_names}),
        ),
        "expression_metadata.csv": (
            lambda p, df: df.to_csv(p, index=True),
            meta,
        ),
        "animal_metadata.csv": (
            lambda p, df: df.to_csv(p, index=False),
            animal_meta,
        ),
    }

    for name, (writer, payload) in paths.items():
        path = os.path.join(args.out_dir, name)
        writer(path, payload)
        print(f"  Wrote {path}")


if __name__ == "__main__":
    main()
