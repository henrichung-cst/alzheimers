"""Export Song snRNA-seq h5ad for factorial Incytr analysis.

Reads 170_gex_celltypes_00.h5ad, filters to males across all genotypes and
timepoints, maps subclass names to SEA-AD subclasses, and exports a sparse
expression matrix, cell metadata (with animal_id), and an animal-level design
matrix for the R factorial pipeline.

Design matrix matches kinase_attribution.py OLS:
  const, App, Tau, Int, time_4mo, time_6mo,
  App_x_time4, App_x_time6, Tau_x_time4, Tau_x_time6
"""

import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse

# common.py sets up sys.path — must be imported before config
from common import ensure_intermediates_dir  # noqa: E402
from config import SONG_SUBCLASS_MAP, SONG_MIN_SUBCLASS_PROB  # noqa: E402
import config_integration as icfg  # noqa: E402


def build_design_row(genotype: str, timepoint: str) -> dict:
    """Build a single row of the design matrix from genotype + timepoint."""
    geno = icfg.MUTANT_TO_DESIGN[genotype]
    time = icfg.TIMEPOINT_TO_DESIGN[timepoint]
    row = {"const": 1}
    row.update(geno)
    row.update(time)
    # Interaction terms
    row["App_x_time4"] = geno["App"] * time["time_4mo"]
    row["App_x_time6"] = geno["App"] * time["time_6mo"]
    row["Tau_x_time4"] = geno["Tau"] * time["time_4mo"]
    row["Tau_x_time6"] = geno["Tau"] * time["time_6mo"]
    return row


def main():
    ensure_intermediates_dir()
    out = os.path.join(icfg.FACTORIAL_DIR)
    os.makedirs(out, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load h5ad
    # ------------------------------------------------------------------
    print(f"Loading {icfg.H5AD_PATH} ...")
    adata = ad.read_h5ad(icfg.H5AD_PATH)
    print(f"  Full dataset: {adata.n_obs} cells x {adata.n_vars} genes")

    # ------------------------------------------------------------------
    # 2. Filter to males, all genotypes, all timepoints
    # ------------------------------------------------------------------
    mask = (
        (adata.obs["sex"] == icfg.FACTORIAL_SEX)
        & (adata.obs["mutant"].isin(icfg.FACTORIAL_GENOTYPES))
        & (adata.obs["age"].isin(icfg.FACTORIAL_TIMEPOINTS))
    )
    adata = adata[mask].copy()
    print(f"  After sex filter (males only): {adata.n_obs} cells")

    # ------------------------------------------------------------------
    # 3. Filter by subclass probability
    # ------------------------------------------------------------------
    if "subclass_prob" in adata.obs.columns:
        prob_mask = adata.obs["subclass_prob"] >= SONG_MIN_SUBCLASS_PROB
        adata = adata[prob_mask].copy()
        print(f"  After subclass_prob >= {SONG_MIN_SUBCLASS_PROB}: {adata.n_obs} cells")

    # ------------------------------------------------------------------
    # 4. Map Allen subclass names to SEA-AD subclasses
    # ------------------------------------------------------------------
    adata.obs["sea_ad_subclass"] = adata.obs["subclass_name"].map(SONG_SUBCLASS_MAP)
    unmapped = adata.obs["sea_ad_subclass"].isna().sum()
    if unmapped > 0:
        print(f"  Dropping {unmapped} cells with unmapped subclass names")
        adata = adata[adata.obs["sea_ad_subclass"].notna()].copy()
    print(f"  After subclass mapping: {adata.n_obs} cells")

    # ------------------------------------------------------------------
    # 5. Build cell-level metadata for Incytr
    # ------------------------------------------------------------------
    meta = pd.DataFrame(index=adata.obs.index)
    meta["labels"] = adata.obs["sea_ad_subclass"].values
    meta["animal_id"] = adata.obs["sample"].values
    meta["genotype"] = adata.obs["mutant"].values
    meta["timepoint"] = adata.obs["age"].values

    # Summary
    animals = meta[["animal_id", "genotype", "timepoint"]].drop_duplicates()
    print(f"\n  Animals: {len(animals)}")
    print("  Cells per genotype x timepoint:")
    for geno in icfg.FACTORIAL_GENOTYPES:
        for tp in icfg.FACTORIAL_TIMEPOINTS:
            n = ((meta["genotype"] == geno) & (meta["timepoint"] == tp)).sum()
            n_animals = animals[
                (animals["genotype"] == geno) & (animals["timepoint"] == tp)
            ].shape[0]
            print(f"    {geno:>4s} x {tp}: {n:>5d} cells, {n_animals} animals")

    print("\n  Cells per subclass:")
    for ct in sorted(meta["labels"].unique()):
        n = (meta["labels"] == ct).sum()
        n_a = meta.loc[meta["labels"] == ct, "animal_id"].nunique()
        print(f"    {ct:<20s}  {n:>5d} cells  ({n_a} animals)")

    # ------------------------------------------------------------------
    # 6. Build animal-level design matrix
    # ------------------------------------------------------------------
    animal_meta_rows = []
    for _, row in animals.iterrows():
        design_row = build_design_row(row["genotype"], row["timepoint"])
        design_row["animal_id"] = row["animal_id"]
        design_row["genotype"] = row["genotype"]
        design_row["timepoint"] = row["timepoint"]
        animal_meta_rows.append(design_row)

    animal_meta = pd.DataFrame(animal_meta_rows)
    # Reorder: animal_id, genotype, timepoint, then design columns
    id_cols = ["animal_id", "genotype", "timepoint"]
    animal_meta = animal_meta[id_cols + icfg.DESIGN_COLUMNS]
    animal_meta = animal_meta.sort_values("animal_id").reset_index(drop=True)

    print(f"\n  Design matrix: {len(animal_meta)} animals x "
          f"{len(icfg.DESIGN_COLUMNS)} parameters")
    rank = np.linalg.matrix_rank(animal_meta[icfg.DESIGN_COLUMNS].values.astype(float))
    print(f"  Design matrix rank: {rank} (need {len(icfg.DESIGN_COLUMNS)})")
    if rank < len(icfg.DESIGN_COLUMNS):
        print("  WARNING: Design matrix is rank-deficient!")

    # ------------------------------------------------------------------
    # 7. Export sparse matrix (genes x cells, as R expects)
    # ------------------------------------------------------------------
    X = adata.X
    if scipy.sparse.issparse(X):
        X_t = X.T.tocsc()
    else:
        X_t = scipy.sparse.csc_matrix(X.T)

    mtx_path = os.path.join(out, "expression_matrix.mtx")
    scipy.io.mmwrite(mtx_path, X_t)
    print(f"\n  Wrote {mtx_path} ({X_t.shape[0]} genes x {X_t.shape[1]} cells)")

    # Gene names
    genes_path = os.path.join(out, "expression_genes.csv")
    pd.DataFrame({"gene": adata.var_names}).to_csv(genes_path, index=False)
    print(f"  Wrote {genes_path}")

    # Cell barcodes
    barcodes_path = os.path.join(out, "expression_barcodes.csv")
    pd.DataFrame({"barcode": adata.obs_names}).to_csv(barcodes_path, index=False)
    print(f"  Wrote {barcodes_path}")

    # Cell metadata
    meta_path = os.path.join(out, "expression_metadata.csv")
    meta.to_csv(meta_path, index=True)
    print(f"  Wrote {meta_path}")

    # Animal design matrix
    animal_meta_path = os.path.join(out, "animal_metadata.csv")
    animal_meta.to_csv(animal_meta_path, index=False)
    print(f"  Wrote {animal_meta_path}")

    print("\nexport_expression_factorial complete.")


if __name__ == "__main__":
    main()
