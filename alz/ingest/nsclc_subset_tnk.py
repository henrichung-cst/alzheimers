#!/usr/bin/env python3
"""Export the T/NK compartment of the 10x NSCLC matrix to a native 10x HDF5.

The full matrix is 18,082 genes x 897,733 cells (1.3 B nnz). We select only
the ~182 K barcodes whose graphclust cluster was coarse-labeled ``T_NK``
(by nsclc_expression.py --label-clusters) and write them to a Cell Ranger
v3-format HDF5 that ``Read10X_h5`` reads directly, so ProjecTILs can project
just the T/NK cells (alz/ingest/nsclc_projectils_map.R).

Memory-safe: streams the source CSC matrix in cell-column chunks, appending to
resizable output datasets — never materializes the full matrix. Peak RAM is
one chunk's nnz, not the whole subset.

Usage:  python alz/ingest/nsclc_subset_tnk.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from alz.shared import config

CHUNK_CELLS = 20000


def main() -> None:
    import h5py
    import pandas as pd

    if not os.path.exists(config.NSCLC_CELL_LABELS_FILE):
        raise FileNotFoundError(
            f"{config.NSCLC_CELL_LABELS_FILE} missing — run "
            f"`pixi run nsclc-label` first.")

    cells = pd.read_csv(config.NSCLC_CELL_LABELS_FILE)
    # ProjecTILs candidates = T/NK compartment + leak-risk clusters (guardrail 1;
    # scGate inside filter.cells arbitrates, so over-inclusion is safe).
    tnk = set(cells.loc[cells["projectils_candidate"], "barcode"])
    print(f"  ProjecTILs-candidate barcodes to export: {len(tnk):,}")

    src = h5py.File(config.NSCLC_10X_H5_FILE, "r")
    m = src["matrix"]
    n_genes, n_cells = (int(x) for x in m["shape"][:])
    barcodes = np.array([x.decode() for x in m["barcodes"][:]])
    keep_mask = np.array([b in tnk for b in barcodes])
    keep_cols = np.flatnonzero(keep_mask)
    print(f"  source {n_genes} x {n_cells:,}; keeping {keep_cols.size:,} columns")

    indptr = m["indptr"][:]            # int (n_cells+1); small enough to hold
    data_ds = m["data"]
    idx_ds = m["indices"]

    out_path = config.NSCLC_10X_TNK_SUBSET_H5
    tmp_path = f"{out_path}.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with h5py.File(tmp_path, "w") as out:
        om = out.create_group("matrix")
        d_out = om.create_dataset("data", shape=(0,), maxshape=(None,),
                                  dtype="int32", chunks=(1 << 20,))
        i_out = om.create_dataset("indices", shape=(0,), maxshape=(None,),
                                  dtype="int64", chunks=(1 << 20,))
        new_indptr = [0]
        written = 0

        for c0 in range(0, n_cells, CHUNK_CELLS):
            c1 = min(c0 + CHUNK_CELLS, n_cells)
            cols = keep_cols[(keep_cols >= c0) & (keep_cols < c1)]
            if cols.size == 0:
                continue
            p0, p1 = int(indptr[c0]), int(indptr[c1])
            chunk_data = data_ds[p0:p1]
            chunk_idx = idx_ds[p0:p1]
            # gather all kept columns' slices, then bulk-write once per chunk
            d_parts, i_parts = [], []
            for col in cols:
                a = int(indptr[col]) - p0
                b = int(indptr[col + 1]) - p0
                if b > a:
                    d_parts.append(chunk_data[a:b])
                    i_parts.append(chunk_idx[a:b])
                written += (b - a)
                new_indptr.append(written)
            if d_parts:
                cat_d = np.concatenate(d_parts)
                cat_i = np.concatenate(i_parts)
                start = d_out.shape[0]
                d_out.resize(start + cat_d.size, axis=0)
                i_out.resize(start + cat_i.size, axis=0)
                d_out[start:] = cat_d
                i_out[start:] = cat_i
            print(f"    {c1:,}/{n_cells:,} cells scanned; {written:,} nnz written")

        om.create_dataset("indptr", data=np.asarray(new_indptr, dtype="int64"))
        om.create_dataset("shape", data=np.array([n_genes, keep_cols.size],
                                                 dtype="int32"))
        om.create_dataset("barcodes", data=m["barcodes"][:][keep_mask])
        # features: copy datasets only (target_sets is a group; genes unchanged)
        fg = om.create_group("features")
        for k in m["features"].keys():
            obj = m["features"][k]
            if isinstance(obj, h5py.Dataset):
                fg.create_dataset(k, data=obj[:])
    src.close()
    os.replace(tmp_path, out_path)
    print(f"  wrote {keep_cols.size:,} cells x {n_genes} genes, "
          f"{written:,} nnz -> {out_path}")


if __name__ == "__main__":
    main()
