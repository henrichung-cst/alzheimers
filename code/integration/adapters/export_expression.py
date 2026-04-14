"""Adapter 5.1: Export Song snRNA-seq h5ad to Incytr-compatible MTX + CSV.

Reads the 170_gex_celltypes_00.h5ad, filters to the Phase 1 comparison
(males, 4mo, WT vs App), maps subclass names to SEA-AD subclasses, and
exports a sparse expression matrix plus metadata for R consumption.
"""

import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse

from config import SONG_SUBCLASS_MAP, SONG_MIN_SUBCLASS_PROB  # noqa: E402
from common import ensure_intermediates_dir  # noqa: E402
import config_integration as icfg  # noqa: E402


def main():
    ensure_intermediates_dir()
    out = icfg.INTERMEDIATES_DIR

    # ------------------------------------------------------------------
    # 1. Load h5ad
    # ------------------------------------------------------------------
    print(f"Loading {icfg.H5AD_PATH} ...")
    adata = ad.read_h5ad(icfg.H5AD_PATH)
    print(f"  Full dataset: {adata.n_obs} cells x {adata.n_vars} genes")

    # ------------------------------------------------------------------
    # 2. Filter to Phase 1 comparison
    # ------------------------------------------------------------------
    mask = (
        (adata.obs["sex"] == icfg.SEX_FILTER)
        & (adata.obs["age"] == icfg.TIMEPOINT)
        & (adata.obs["mutant"].isin([icfg.CONDITION_WT, icfg.CONDITION_DISEASE]))
    )
    adata = adata[mask].copy()
    print(f"  After sex/timepoint/genotype filter: {adata.n_obs} cells")

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
    # 5. Build metadata for Incytr
    # ------------------------------------------------------------------
    meta = pd.DataFrame(index=adata.obs.index)
    meta["labels"] = adata.obs["sea_ad_subclass"].values
    meta["condition"] = adata.obs["mutant"].map(icfg.MUTANT_TO_CONDITION).values

    # Summary
    print("\n  Cells per condition:")
    for cond in icfg.INCYTR_CONDITIONS:
        n = (meta["condition"] == cond).sum()
        print(f"    {cond}: {n}")
    print("\n  Cells per subclass (sender/receiver):")
    for ct in [icfg.SENDER, icfg.RECEIVER]:
        for cond in icfg.INCYTR_CONDITIONS:
            n = ((meta["labels"] == ct) & (meta["condition"] == cond)).sum()
            print(f"    {ct} / {cond}: {n}")

    # ------------------------------------------------------------------
    # 6. Export sparse matrix (genes x cells, as R expects)
    # ------------------------------------------------------------------
    # adata.X is cells x genes; Incytr wants genes x cells
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

    # Metadata
    meta_path = os.path.join(out, "expression_metadata.csv")
    meta.to_csv(meta_path, index=True)
    print(f"  Wrote {meta_path}")

    print("\nAdapter 5.1 complete.")


if __name__ == "__main__":
    main()
