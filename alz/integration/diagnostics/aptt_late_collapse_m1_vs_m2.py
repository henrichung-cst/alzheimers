"""ApTt-late TPDS collapse — M1 vs M2 saturated-model diagnostic.

For each Incytr pathway in the Astrocytes receiver, compare:
- M1 TPDS: current value from the 10-column factorial OLS (carried in the
  `TPDS` column).
- M2 TPDS: condition-level proxy for a saturated 12-column model
  (`+ Int_x_time4 + Int_x_time6`). Computed as
  `logi(log(SigProb_alt + eps) - log(SigProb_ref + eps))` using the
  group-mean SigProb columns.

If M2 estimates for ApTt_4mo / ApTt_6mo are materially larger than M1,
the collapse is partly a design-misspecification artifact (constant `Int`
extrapolated to 4mo/6mo). If M2 lands where M1 did, the late-ApTt
transcriptome is genuinely WT-like and the collapse reflects the data.

Caveat: M2 proxy uses log(group-mean) instead of mean(log-per-animal).
These differ when per-animal SigProb is dispersed within a cell; with
n=1 ApTt animal at 4mo/6mo there is no within-cell variance to lose, so
the proxy is exact for those cells. For multi-animal cells (WT, App, Tau)
it is a Jensen approximation.
"""

from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RECEIVER_CACHE = os.path.join(
    REPO_ROOT,
    "outputs/reports/incytr_factorial_5xfad_kldata/receiver_cache",
)
OUT_DIR = os.path.join(REPO_ROOT, "outputs/reports/incytr_factorial_5xfad_kldata/diagnostics")
EPS = 1e-10
K_LOGI = 2.0 / math.log(2.0)

CONTRAST_CELLS = {
    "App_2mo":  ("WTyp_2mo", "AppP_2mo"),
    "App_4mo":  ("WTyp_4mo", "AppP_4mo"),
    "App_6mo":  ("WTyp_6mo", "AppP_6mo"),
    "Tau_2mo":  ("WTyp_2mo", "Ttau_2mo"),
    "Tau_4mo":  ("WTyp_4mo", "Ttau_4mo"),
    "Tau_6mo":  ("WTyp_6mo", "Ttau_6mo"),
    "ApTt_2mo": ("WTyp_2mo", "ApTt_2mo"),
    "ApTt_4mo": ("WTyp_4mo", "ApTt_4mo"),
    "ApTt_6mo": ("WTyp_6mo", "ApTt_6mo"),
}


def logi(x: np.ndarray, k: float = K_LOGI) -> np.ndarray:
    return 2.0 / (1.0 + np.exp(-k * x)) - 1.0


def load_receiver(receiver: str) -> pd.DataFrame:
    parts = sorted(glob.glob(os.path.join(RECEIVER_CACHE, f"receiver={receiver}", "*.parquet")))
    if not parts:
        raise FileNotFoundError(f"no parquet for receiver={receiver}")
    frames = []
    for p in parts:
        d = pq.ParquetFile(p).read().to_pandas()
        if "receiver" in d.columns:
            d = d.drop(columns=["receiver"])
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def compute_m2(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for cname, (ref, alt) in CONTRAST_CELLS.items():
        sub = df[df["contrast"] == cname]
        if sub.empty:
            continue
        ref_col = f"SigProb_{ref}"
        alt_col = f"SigProb_{alt}"
        if ref_col not in sub.columns or alt_col not in sub.columns:
            continue
        est_m2 = np.log(sub[alt_col].to_numpy() + EPS) - np.log(sub[ref_col].to_numpy() + EPS)
        tpds_m2 = logi(est_m2)
        out_rows.append(
            pd.DataFrame({
                "contrast": cname,
                "ID_1": sub["ID_1"].to_numpy(),
                "ID_2": sub["ID_2"].to_numpy(),
                "tpds_m1": sub["TPDS"].to_numpy(),
                "tpds_m2": tpds_m2,
                "sigprob_ref": sub[ref_col].to_numpy(),
                "sigprob_alt": sub[alt_col].to_numpy(),
            })
        )
    return pd.concat(out_rows, ignore_index=True)


def summarize(joined: pd.DataFrame) -> pd.DataFrame:
    g = joined.groupby("contrast", observed=True)
    summary = g.agg(
        n=("tpds_m1", "size"),
        mean_abs_tpds_m1=("tpds_m1", lambda s: float(np.mean(np.abs(s)))),
        mean_abs_tpds_m2=("tpds_m2", lambda s: float(np.mean(np.abs(s)))),
        n_high_m1=("tpds_m1", lambda s: int((np.abs(s) >= 0.5).sum())),
        n_high_m2=("tpds_m2", lambda s: int((np.abs(s) >= 0.5).sum())),
    ).reset_index()
    summary["delta_mean_abs"] = summary["mean_abs_tpds_m2"] - summary["mean_abs_tpds_m1"]
    summary["delta_n_high"] = summary["n_high_m2"] - summary["n_high_m1"]
    return summary


def main() -> None:
    receivers = sys.argv[1:] or ["Astrocytes"]
    os.makedirs(OUT_DIR, exist_ok=True)
    all_summary = []
    for rec in receivers:
        df = load_receiver(rec)
        joined = compute_m2(df)
        joined.to_parquet(os.path.join(OUT_DIR, f"m1_vs_m2_{rec}.parquet"), index=False)
        s = summarize(joined)
        s.insert(0, "receiver", rec)
        all_summary.append(s)
        print(f"\n=== {rec} (n_paths={len(df):,}) ===")
        print(s.to_string(index=False))
    pd.concat(all_summary, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "m1_vs_m2_summary.csv"), index=False
    )
    print(f"\nWrote: {OUT_DIR}/m1_vs_m2_summary.csv")


if __name__ == "__main__":
    main()
