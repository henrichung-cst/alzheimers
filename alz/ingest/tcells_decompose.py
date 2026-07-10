"""T-cell exhaustion cohort — per-(cluster, day) protein deconvolution.

Time-course analog of `alz/incytr_pair/export_decomposition_for_pair.py`. Applies
the provenance deconvolution formula:

    P_c = (N_total / N_c) × bulk × (specific_c / Σ_clusters specific)

per (cluster, day), with min/10000 zero-imputation on the share. The scRNA share
and size factors come from the memory-safe extraction
(`alz/ingest/tcells_scrna_extract.R`); the linear per-day bulk comes from
`tcells.py --export-bulk`. Both are pre-computed; this module never touches the
raw `.rds`.

Per-donor channels (donor1 has IMAC, donor2 does not):
    pr (gene-keyed)        — Total proteome → pr_deconvoluted.csv
    py (site-keyed)        — pY            → py_deconvoluted.csv
    ps (site-keyed)        — IMAC          → ps_deconvoluted.csv  (donor1 only)

Output wide schema: row keys + columns `d{day}_c{cluster}` for the days present
in BOTH the bulk and the scRNA share (the contrast set is built from this
intersection downstream). Numeric cluster IDs are used here; rename to T-cell
subset labels after annotation (the spine `cluster` half must stay underscore-free
so the Incytr driver's `<condition>_<cluster>` split works).

Mass identity (sanity gate, not a closed path):
    Σ_c [P_c × (N_c / N_total)] ≈ bulk   per (gene/site, day)
holds by construction when share sums to 1 over clusters — which it does because
aggexp and cell_counts only carry (cluster, day) groups that have cells. Max
relative error is reported.

Usage:  pixi run tcells-decompose   (both donors)
        python alz/ingest/tcells_decompose.py --donor donor1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.cohorts.tcells.ingest import INCYTR_INPUT_DIR, _donor_prefix

_AGG_COL_RE = re.compile(r"^(?P<st>[A-Za-z0-9]+)__d(?P<d>\d+)$")  # state-keyed
_BULK_COL_RE = re.compile(r"^D\d+_d(?P<d>\d+)$")


def _load_aggexp(donor: str
                 ) -> tuple[pd.DataFrame, list[tuple[str, int]], float]:
    """Load aggexp (state-keyed). Columns are already `<state>__d<day>` — the
    extract step keys on the sanitized native-cluster state, so no
    cluster→label collapse is needed here. Floor = (nonzero min) / 10000."""
    path = os.path.join(INCYTR_INPUT_DIR, donor, "scrna", "aggexp_data.csv")
    raw = pd.read_csv(path).set_index("gene")
    parsed: list[tuple[str, int]] = []
    for c in raw.columns:
        m = _AGG_COL_RE.match(c)
        if not m:
            raise ValueError(f"unparsed aggexp column {c!r} — expected <state>__d<day>")
        parsed.append((m["st"], int(m["d"])))
    nz = raw.values[raw.values > 0]
    floor = float(nz.min()) / 10000.0
    return raw, parsed, floor


def _shares_by_day(
    agg: pd.DataFrame, parsed: list[tuple[str, int]], floor: float
) -> dict[int, pd.DataFrame]:
    """Per day d: gene × cluster share matrix = specific_c / Σ_clusters specific
    (zeros imputed to ``floor`` first). Cells where a (cluster, day) is absent
    from aggexp simply don't appear; share within day sums to 1 across present
    clusters."""
    by_day: dict[int, list[tuple[str, str]]] = {}
    for col, (cl, d) in zip(agg.columns, parsed):
        by_day.setdefault(d, []).append((cl, col))
    out: dict[int, pd.DataFrame] = {}
    for d, items in by_day.items():
        clusters = [cl for cl, _ in items]
        sub = agg[[col for _, col in items]].copy()
        sub.columns = clusters
        arr = sub.values.astype("float64")
        arr[arr == 0.0] = floor
        denom = arr.sum(axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0  # defensive; with floor impute denom>0 always
        share = pd.DataFrame(arr / denom, index=sub.index, columns=clusters)
        out[d] = share
    return out


def _load_counts(donor: str
                 ) -> tuple[dict[tuple[str, int], int], dict[int, int]]:
    """Load state-keyed cell counts. cell_counts.csv has columns (state, day,
    n_cells) — already filtered upstream to non-contaminant evidence labels.
    N_total reflects only retained T cells, keeping mass identity
    Σ_s P_s × N_s/N_total = bulk exact on the annotated subset."""
    path = os.path.join(INCYTR_INPUT_DIR, donor, "scrna", "cell_counts.csv")
    cc = pd.read_csv(path)
    cc["state"] = cc["state"].astype(str)
    cc["day"] = cc["day"].astype(int)
    per_lab = (cc.groupby(["state", "day"])["n_cells"].sum().astype(int).to_dict())
    per: dict[tuple[str, int], int] = {(st, d): n for (st, d), n in per_lab.items()}
    totals = cc.groupby("day")["n_cells"].sum().astype(int).to_dict()
    return per, totals


def _bulk_day_cols(bulk: pd.DataFrame) -> dict[int, str]:
    out: dict[int, str] = {}
    for c in bulk.columns:
        m = _BULK_COL_RE.match(c)
        if m:
            out[int(m["d"])] = c
    return out


def _deconvolve(
    bulk: pd.DataFrame,
    gene_col: str,
    key_cols: list[str],
    shares: dict[int, pd.DataFrame],
    n_per: dict[tuple[str, int], int],
    n_total: dict[int, int],
) -> tuple[pd.DataFrame, dict]:
    """Apply P_c = (N_total/N_c) × bulk × share per shared day × cluster.

    Vectorized per (day, cluster): one column-wise multiply against the share
    column, joined on gene_symbol. Sites whose gene_symbol is not in the share
    matrix are emitted with NaN (honest — the scRNA has no transcript support).
    """
    bulk_days = _bulk_day_cols(bulk)
    shared_days = sorted(set(bulk_days) & set(shares))
    skipped_days = sorted(set(bulk_days) - set(shared_days))
    out = bulk[key_cols].copy()
    mass = {"per_day": {}}
    for d in shared_days:
        share_d = shares[d]
        bcol = bulk_days[d]
        bvals = bulk[bcol].values.astype("float64")
        # gene index into share (NaN for unmatched)
        gidx = pd.Index(bulk[gene_col].astype(str))
        share_aligned = share_d.reindex(gidx)
        clusters_d = list(share_aligned.columns)
        for cl in clusters_d:
            nc = n_per.get((cl, d), 0)
            if nc <= 0:
                continue
            sf = n_total[d] / nc
            col = f"d{d}_{cl}"   # cl is now a lineage label, underscore-free
            out[col] = sf * bvals * share_aligned[cl].values
        # mass identity check: Σ_c P_c × N_c/N_total per gene at this day.
        # Restrict to rows with scRNA share (unmatched genes get honest NaN in
        # P_c by design — excluding them keeps the check on the rows we actually
        # deconvolved).
        has_share = ~share_aligned.iloc[:, 0].isna().values
        sums = np.zeros_like(bvals)
        for cl in clusters_d:
            nc = n_per.get((cl, d), 0)
            if nc <= 0:
                continue
            sums = sums + out[f"d{d}_{cl}"].values * (nc / n_total[d])
        ratio = np.where(has_share & (bvals != 0), sums / bvals, np.nan)
        mass["per_day"][d] = {
            "max_abs_rel_err": float(np.nanmax(np.abs(ratio - 1.0))),
            "median_ratio": float(np.nanmedian(ratio)),
            "n_genes_compared": int(np.sum(~np.isnan(ratio))),
            "n_rows_no_scrna_support": int(np.sum(~has_share)),
        }
    mass["shared_days"] = shared_days
    mass["bulk_days_skipped_no_scrna"] = skipped_days
    return out, mass


def _decompose_donor(donor: str) -> dict:
    indir = os.path.join(INCYTR_INPUT_DIR, donor)
    outdir = indir
    print(f"\n=== decompose {donor} ===")
    agg, parsed, floor = _load_aggexp(donor)
    shares = _shares_by_day(agg, parsed, floor)
    n_per, n_total = _load_counts(donor)
    states = sorted({st for st, _ in parsed})
    print(f"  spine: {len(states)} states → {states}")
    print(f"  aggexp: {len(agg)} genes, {len(parsed)} (state,day) groups; "
          f"share floor = {floor:.4g}")
    print(f"  scrna days: {sorted(shares)}; cells/day (annotated): "
          f"{ {d: n_total[d] for d in sorted(n_total)} }")

    summary: dict = {"donor": donor, "share_floor": floor,
                     "scrna_days": sorted(shares),
                     "cells_per_day_annotated": {int(d): int(n_total[d])
                                                 for d in sorted(n_total)},
                     "states": states,
                     "channels": {}}

    def _run(channel: str, bulk_path: str, gene_col: str, key_cols: list[str],
             out_name: str) -> None:
        if not os.path.exists(bulk_path):
            print(f"  {channel}: bulk missing → skip ({bulk_path})")
            return
        bulk = pd.read_csv(bulk_path)
        out, mass = _deconvolve(bulk, gene_col, key_cols, shares, n_per, n_total)
        out_path = os.path.join(outdir, out_name)
        out.to_csv(out_path, index=False)
        value_cols = [c for c in out.columns if c not in key_cols]
        max_err = max((v["max_abs_rel_err"] for v in mass["per_day"].values()),
                      default=float("nan"))
        print(f"  {channel}: {len(out)} rows × {len(value_cols)} value cols "
              f"({len(mass['shared_days'])} shared days × clusters); "
              f"mass-identity max |rel err| = {max_err:.3g}")
        summary["channels"][channel] = {
            "rows": int(len(out)),
            "value_cols": len(value_cols),
            "shared_days": mass["shared_days"],
            "bulk_days_skipped_no_scrna": mass["bulk_days_skipped_no_scrna"],
            "mass_identity": mass["per_day"],
        }

    _run("pr", os.path.join(indir, "pr_bulk_linear.csv"),
         "gene_symbol", ["gene_symbol"], "pr_deconvoluted.csv")
    _run("py", os.path.join(indir, "py_bulk_linear.csv"),
         "gene_symbol", ["site_id", "gene_symbol", "motif"], "py_deconvoluted.csv")
    if donor == "donor1":
        _run("ps", os.path.join(indir, "ps_bulk_linear.csv"),
             "gene_symbol", ["site_id", "gene_symbol", "motif"], "ps_deconvoluted.csv")

    summary["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(os.path.join(outdir, "decompose_manifest.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--donor", choices=["donor1", "donor2", "both"], default="both")
    args = p.parse_args(argv)
    donors = ["donor1", "donor2"] if args.donor == "both" else [args.donor]
    for d in donors:
        _decompose_donor(d)
    print("\n[tcells] decompose done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
