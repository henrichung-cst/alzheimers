"""Hand-compute per-animal SigProb for D095 (6mo WT) and a few top-TPDS-at-ApTt_2mo
paths in Astrocytes. Compare against parquet's SigProb_WTyp_6mo (which is 0).

If hand-computed SigProb is non-zero but parquet is exactly 0, something filters
or zeros the per-animal SigProb at 4mo/6mo before the per-condition mean.
"""
from __future__ import annotations
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.io as sio

ROOT = "/home/hchung/Projects/work/alzheimers"
IN = f"{ROOT}/data/incytr_factorial_inputs"
RECV = f"{ROOT}/outputs/reports/incytr_factorial_5xfad_kldata/receiver_cache/receiver=Astrocytes"
K, N = 0.5, 2
KN = K ** N


def hill(x):
    x2 = x ** N
    return x2 / (x2 + KN)


def per_animal_cluster_mean(X, meta, animal, cluster, gene_idx):
    mask = (meta["animal_id"].values == animal) & (meta["cluster"].values == cluster)
    if mask.sum() == 0:
        return np.zeros(len(gene_idx))
    Xs = X[gene_idx, :][:, np.where(mask)[0]]
    return np.asarray(Xs.mean(axis=1)).ravel()


def main():
    bcs = pd.read_csv(f"{IN}/expression_barcodes.csv", header=None)[0].tolist()
    if bcs[0].lower() in ("barcode", "0", "x"):
        bcs = bcs[1:]
    genes = pd.read_csv(f"{IN}/expression_genes.csv", header=None)[0].tolist()
    if genes[0].lower() in ("gene", "0"):
        genes = genes[1:]
    gene2i = {g: i for i, g in enumerate(genes)}
    meta = pd.read_csv(f"{IN}/expression_metadata.csv").reset_index(drop=True)
    cm = pd.read_csv(f"{ROOT}/data/incytr/v2_46clusters/barcode_to_cluster.csv")
    bc2cl = dict(zip(cm["barcode"], cm["cluster_subclass"]))
    meta["cluster"] = pd.Series(bcs).map(bc2cl).values

    print("Loading matrix...")
    X = sio.mmread(f"{IN}/expression_matrix.mtx").tocsc()

    parts = sorted(glob.glob(f"{RECV}/*.parquet"))
    frames = []
    for p in parts:
        d = pq.ParquetFile(p).read(
            columns=["Ligand", "Receptor", "EM", "Target", "Sender.group",
                     "Receiver.group", "contrast", "TPDS",
                     "SigProb_WTyp_2mo", "SigProb_WTyp_6mo",
                     "SigProb_ApTt_2mo", "SigProb_ApTt_6mo"]
        ).to_pandas()
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    top = df[(df["contrast"] == "ApTt_2mo") & (df["TPDS"].abs() >= 0.5)].copy()
    top = top.reindex(top["TPDS"].abs().sort_values(ascending=False).index).head(10)
    print(f"\n{len(top)} top |TPDS|>=0.5 paths at ApTt_2mo")

    rows = []
    for _, p in top.iterrows():
        sender = p["Sender.group"]
        L, R, EM, TG = p["Ligand"], p["Receptor"], p["EM"], p["Target"]
        if not all(g in gene2i for g in [L, R, EM, TG]):
            continue
        # For each path component
        gi = [gene2i[L], gene2i[R], gene2i[EM], gene2i[TG]]
        for animal, label in [
            ("D095_ma_6mo_WTyp", "WTyp_6mo"),
            ("D092_ma_6mo_ApTt", "ApTt_6mo"),
            ("C201_ma_2mo_ApTt", "ApTt_2mo"),
            ("E137_ma_4mo_ApTt", "ApTt_4mo"),
        ]:
            send_expr = per_animal_cluster_mean(X, meta, animal, sender, gi)
            recv_expr = per_animal_cluster_mean(X, meta, animal, "Astrocytes", gi)
            L_v, R_v, EM_v, T_v = send_expr[0], recv_expr[1], recv_expr[2], recv_expr[3]
            sp = hill(L_v * R_v) * hill(R_v * EM_v) * hill(EM_v * T_v)
            parquet_col = f"SigProb_{label}"
            parquet_val = p[parquet_col] if parquet_col in p else None
            rows.append({
                "Path": f"{L}>{R}>{EM}>{TG}",
                "Sender": sender, "Animal": animal, "Label": label,
                "L": round(L_v, 4), "R": round(R_v, 4),
                "EM": round(EM_v, 4), "T": round(T_v, 4),
                "L*R": round(L_v * R_v, 4),
                "hand_sigprob": sp,
                "parquet_sigprob": parquet_val,
            })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
