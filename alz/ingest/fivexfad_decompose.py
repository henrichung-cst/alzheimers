"""5xFAD cohort — per-(cell_type, condition) protein deconvolution for Incytr.

Cohort analog of `alz/ingest/tcells_decompose.py` (which is itself the analog of
the AD `export_decomposition_for_pair.py`). Applies the provenance deconvolution:

    P_c = (N_total / N_c) × bulk × (specific_c / Σ_cell_types specific)

per (cell_type, condition), with min/10000 zero-imputation on the share. The
scRNA share and size factors come from `alz/ingest/fivexfad_scrna_extract.R`
(`aggexp_data.csv` + `cell_counts.csv`); the linear per-group bulk comes from
`alz/cohorts/fivexfad/ingest.py --export-bulk` (`{pr,ps,py,ack,kgg}_bulk_linear.csv`).
Both are pre-computed per tissue; this module never touches the raw RDS.

Channels (pr/ps/py always present both tissues; ack/kgg 5xFAD-only):
    pr (gene-keyed)   — total proteome  → pr_deconvoluted.csv
    ps (site-keyed)   — IMAC/ST         → ps_deconvoluted.csv
    py (site-keyed)   — pY              → py_deconvoluted.csv
    ack (site-keyed)  — acetylation     → ack_deconvoluted.csv  (if bulk present)
    kgg (site-keyed)  — ubiquitination  → kgg_deconvoluted.csv  (if bulk present)

Self-gating: if a bulk_linear.csv is absent for ack or kgg (e.g. hippocampus AcK
Mo6-12 report missing 3mo), the channel is skipped gracefully with a printed note.

Output wide schema: row keys + columns `<condition>_<cell_type>` (condition =
`<geno>_<age>`, e.g. "TG_3mo"; cell_type = a 31-spine label). The Incytr driver's
`slice_omics` strips the anchored `^<condition>_` prefix, so condition may carry
the internal `_` and cell_type may carry hyphens/spaces (identical to the AD
`ma_<age>_<geno>_<cluster>` scheme).

Mass identity (sanity gate): Σ_c [P_c × (N_c / N_total)] ≈ bulk per (gene/site,
condition); holds when share sums to 1 over the cell types present. Max relative
error is reported.

Usage:  pixi run python alz/ingest/fivexfad_decompose.py        (both tissues)
        python alz/ingest/fivexfad_decompose.py --tissue cortex
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.cohorts.fivexfad.ingest import INCYTR_INPUT_DIR, TISSUES

_SITE_KEYS = ["site_id", "gene_symbol", "motif"]
KEY_COLS = {
    "pr":  ["gene_symbol"],
    "ps":  _SITE_KEYS,
    "py":  _SITE_KEYS,
    "ack": _SITE_KEYS,
    "kgg": _SITE_KEYS,
}
BULK_FILE = {
    "pr":  "pr_bulk_linear.csv",
    "ps":  "ps_bulk_linear.csv",
    "py":  "py_bulk_linear.csv",
    "ack": "ack_bulk_linear.csv",
    "kgg": "kgg_bulk_linear.csv",
}
OUT_FILE = {
    "pr":  "pr_deconvoluted.csv",
    "ps":  "ps_deconvoluted.csv",
    "py":  "py_deconvoluted.csv",
    "ack": "ack_deconvoluted.csv",
    "kgg": "kgg_deconvoluted.csv",
}


def _load_aggexp(tissue: str) -> tuple[pd.DataFrame, list[tuple[str, str]], float]:
    """Load aggexp (columns `<cell_type>__<condition>`). Floor = (nonzero min)/10000."""
    path = os.path.join(INCYTR_INPUT_DIR, tissue, "scrna", "aggexp_data.csv")
    raw = pd.read_csv(path).set_index("gene")
    parsed: list[tuple[str, str]] = []
    for c in raw.columns:
        cell_type, sep, condition = c.partition("__")
        if not sep:
            raise ValueError(f"unparsed aggexp column {c!r} — expected <cell_type>__<condition>")
        parsed.append((cell_type, condition))
    nz = raw.values[raw.values > 0]
    floor = float(nz.min()) / 10000.0
    return raw, parsed, floor


def _shares_by_condition(
    agg: pd.DataFrame, parsed: list[tuple[str, str]], floor: float
) -> dict[str, pd.DataFrame]:
    """Per condition: gene × cell_type share = specific_c / Σ_cell_types specific
    (zeros imputed to ``floor`` first). Share within a condition sums to 1 over
    the cell types present."""
    by_cond: dict[str, list[tuple[str, str]]] = {}
    for col, (cl, cond) in zip(agg.columns, parsed):
        by_cond.setdefault(cond, []).append((cl, col))
    out: dict[str, pd.DataFrame] = {}
    for cond, items in by_cond.items():
        clusters = [cl for cl, _ in items]
        sub = agg[[col for _, col in items]].copy()
        sub.columns = clusters
        arr = sub.values.astype("float64")
        arr[arr == 0.0] = floor
        denom = arr.sum(axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        out[cond] = pd.DataFrame(arr / denom, index=sub.index, columns=clusters)
    return out


def _load_counts(tissue: str
                 ) -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    """cell_counts.csv → (per (cell_type, condition), per condition total)."""
    path = os.path.join(INCYTR_INPUT_DIR, tissue, "scrna", "cell_counts.csv")
    cc = pd.read_csv(path)
    cc["cell_type"] = cc["cell_type"].astype(str)
    cc["condition"] = cc["condition"].astype(str)
    per = {(r.cell_type, r.condition): int(r.n_cells) for r in cc.itertuples()}
    totals = cc.groupby("condition")["n_cells"].sum().astype(int).to_dict()
    return per, totals


def _deconvolve(
    bulk: pd.DataFrame,
    gene_col: str,
    key_cols: list[str],
    shares: dict[str, pd.DataFrame],
    n_per: dict[tuple[str, str], int],
    n_total: dict[str, int],
) -> tuple[pd.DataFrame, dict]:
    """P_c = (N_total/N_c) × bulk × share per shared condition × cell_type.

    Bulk value columns ARE the conditions. Sites whose gene_symbol is not in the
    share matrix are emitted with NaN (honest — no transcript support)."""
    bulk_conditions = [c for c in bulk.columns if c not in key_cols]
    shared = sorted(set(bulk_conditions) & set(shares))
    skipped = sorted(set(bulk_conditions) - set(shared))
    out = bulk[key_cols].copy()
    mass: dict = {"per_condition": {}}
    gidx = pd.Index(bulk[gene_col].astype(str))
    for cond in shared:
        share_aligned = shares[cond].reindex(gidx)
        bvals = bulk[cond].values.astype("float64")
        clusters = list(share_aligned.columns)
        for cl in clusters:
            nc = n_per.get((cl, cond), 0)
            if nc <= 0:
                continue
            sf = n_total[cond] / nc
            out[f"{cond}_{cl}"] = sf * bvals * share_aligned[cl].values
        has_share = ~share_aligned.iloc[:, 0].isna().values
        sums = np.zeros_like(bvals)
        for cl in clusters:
            nc = n_per.get((cl, cond), 0)
            if nc <= 0:
                continue
            sums = sums + out[f"{cond}_{cl}"].values * (nc / n_total[cond])
        ratio = np.where(has_share & (bvals != 0), sums / bvals, np.nan)
        mass["per_condition"][cond] = {
            "max_abs_rel_err": float(np.nanmax(np.abs(ratio - 1.0))),
            "median_ratio": float(np.nanmedian(ratio)),
            "n_genes_compared": int(np.sum(~np.isnan(ratio))),
            "n_rows_no_scrna_support": int(np.sum(~has_share)),
        }
    mass["shared_conditions"] = shared
    mass["bulk_conditions_skipped_no_scrna"] = skipped
    return out, mass


def _decompose_tissue(tissue: str) -> dict:
    indir = os.path.join(INCYTR_INPUT_DIR, tissue)
    print(f"\n=== decompose {tissue} ===")
    agg, parsed, floor = _load_aggexp(tissue)
    shares = _shares_by_condition(agg, parsed, floor)
    n_per, n_total = _load_counts(tissue)
    cell_types = sorted({cl for cl, _ in parsed})
    print(f"  spine: {len(cell_types)} cell types")
    print(f"  aggexp: {len(agg)} genes, {len(parsed)} (cell_type,condition) groups; "
          f"share floor = {floor:.4g}")
    print(f"  conditions: {sorted(shares)}; cells/condition: "
          f"{ {c: n_total[c] for c in sorted(n_total)} }")

    summary: dict = {"tissue": tissue, "share_floor": floor,
                     "conditions": sorted(shares),
                     "cells_per_condition": {c: int(n_total[c]) for c in sorted(n_total)},
                     "cell_types": cell_types, "channels": {}}

    def _run(channel: str) -> None:
        bulk_path = os.path.join(indir, BULK_FILE[channel])
        if not os.path.exists(bulk_path):
            print(f"  {channel}: bulk missing → skip ({bulk_path})")
            return
        bulk = pd.read_csv(bulk_path)
        out, mass = _deconvolve(bulk, "gene_symbol", KEY_COLS[channel],
                                shares, n_per, n_total)
        out.to_csv(os.path.join(indir, OUT_FILE[channel]), index=False)
        value_cols = [c for c in out.columns if c not in KEY_COLS[channel]]
        max_err = max((v["max_abs_rel_err"] for v in mass["per_condition"].values()),
                      default=float("nan"))
        print(f"  {channel}: {len(out)} rows × {len(value_cols)} value cols "
              f"({len(mass['shared_conditions'])} conditions × cell types); "
              f"mass-identity max |rel err| = {max_err:.3g}")
        summary["channels"][channel] = {
            "rows": int(len(out)), "value_cols": len(value_cols),
            "shared_conditions": mass["shared_conditions"],
            "bulk_conditions_skipped_no_scrna": mass["bulk_conditions_skipped_no_scrna"],
            "mass_identity": mass["per_condition"],
        }

    for channel in ("pr", "ps", "py", "ack", "kgg"):
        _run(channel)

    summary["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(os.path.join(indir, "decompose_manifest.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tissue", choices=[*TISSUES, "all"], default="all")
    args = p.parse_args(argv)
    tissues = list(TISSUES) if args.tissue == "all" else [args.tissue]
    for t in tissues:
        _decompose_tissue(t)
    print("\n[5xfad] decompose done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
