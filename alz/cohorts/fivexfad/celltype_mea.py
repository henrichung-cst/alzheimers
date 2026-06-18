#!/usr/bin/env python3
"""5xFAD per-cell-type phosphosite decomposition MEA.

This is a 5xFAD-native analogue of the Song/Mukesh decomposition cross-check.
It uses matched 5xFAD snRNA `new_clusters` expression to reweight 5xFAD raw
phosphosite signal per cell type, fits age-specific TG-vs-WT site effects, and
runs the shared kinase-library MEA routine on those per-cell-type site ranks.
"""

from __future__ import annotations

import json
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.bulk_mea import enrich as kinase_enrich  # noqa: E402
from alz.cohorts.fivexfad import ingest as fivexfad  # noqa: E402
from alz.shared import config  # noqa: E402

KINASE_DIR = Path(config.REPO_ROOT) / "outputs" / "reports" / "kinase_attribution_5xfad"
OUT_DIR = KINASE_DIR / "celltype_mea"
PSEUDOBULK_PATH = OUT_DIR / "fivexfad_snrna_pseudobulk_linear.csv.gz"
GENE_MAP_PATH = OUT_DIR / "fivexfad_snrna_gene_map.csv"
COUNTS_PATH = OUT_DIR / "fivexfad_snrna_pseudobulk_counts.csv"

TRACKS = {
    "st": {"kl_track": "st", "assay": "IMAC", "residue_type": "ST"},
    "py": {"kl_track": "py", "assay": "pY", "residue_type": "Y"},
}
TISSUES = ("cortex", "hippocampus")
AGES = (3, 6, 9, 12)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    return kinase_enrich._bh_fdr(pvals)


def _contrast_coefs() -> dict[str, dict[str, float]]:
    return fivexfad._contrast_coefs()


def _build_design_matrix(meta: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    rows = meta.drop_duplicates("sample_id").set_index("sample_id").loc[sample_cols].reset_index()
    x = pd.DataFrame(index=range(len(rows)))
    x["const"] = 1.0
    x["age_6mo"] = (rows["age_months"] == 6).astype(float)
    x["age_9mo"] = (rows["age_months"] == 9).astype(float)
    x["age_12mo"] = (rows["age_months"] == 12).astype(float)
    x["TG"] = (rows["genotype"] == "TG").astype(float)
    x["TG_x_age6"] = x["TG"] * x["age_6mo"]
    x["TG_x_age9"] = x["TG"] * x["age_9mo"]
    x["TG_x_age12"] = x["TG"] * x["age_12mo"]
    return x


def _build_contrast_vec(param_names: list[str], coef_map: dict[str, float]) -> np.ndarray | None:
    c = np.zeros(len(param_names), dtype=float)
    for name, weight in coef_map.items():
        if name not in param_names:
            return None
        c[param_names.index(name)] = float(weight)
    return c


def _is_estimable(c: np.ndarray, x: np.ndarray, tol: float = 1e-8) -> bool:
    projected = c @ np.linalg.pinv(x) @ x
    return bool(np.allclose(projected, c, atol=tol, rtol=0))


def _run_ols_pinv(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_sites, n_samples = y.shape
    n_params = x.shape[1]
    betas = np.full((n_sites, n_params), np.nan)
    n_obs = np.zeros(n_sites, dtype=int)
    xtxinv = np.full((n_sites, n_params, n_params), np.nan)

    complete = np.all(np.isfinite(y), axis=1)
    if complete.any():
        pinv = np.linalg.pinv(x.T @ x)
        betas[complete] = (pinv @ x.T @ y[complete].T).T
        n_obs[complete] = n_samples
        xtxinv[complete] = pinv

    for i in np.where(~complete)[0]:
        valid = np.isfinite(y[i])
        n_valid = int(valid.sum())
        if n_valid < n_params + 2:
            continue
        xi = x[valid]
        yi = y[i, valid]
        pinv_i = np.linalg.pinv(xi.T @ xi)
        betas[i] = pinv_i @ xi.T @ yi
        n_obs[i] = n_valid
        xtxinv[i] = pinv_i
    return betas, n_obs, xtxinv


def _contrast_stats(
    y: np.ndarray,
    betas: np.ndarray,
    xtxinv: np.ndarray,
    n_obs: np.ndarray,
    c_vec: np.ndarray,
    x_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_params = x_np.shape[1]
    lfc = betas @ c_vec
    residuals = y - (x_np @ betas.T).T
    dof = n_obs - n_params
    sigma2 = np.full(y.shape[0], np.nan)
    ok = dof > 0
    sigma2[ok] = np.nansum(residuals[ok] ** 2, axis=1) / dof[ok]
    var_c = np.einsum("p,ipq,q->i", c_vec, xtxinv, c_vec)
    se = np.sqrt(np.where(np.isfinite(var_c * sigma2) & (var_c * sigma2 >= 0), var_c * sigma2, np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = lfc / se
    pval = 2 * sp_stats.t.sf(np.abs(t_stat), df=np.maximum(dof, 1))
    return lfc, se, t_stat, pval, _bh_fdr(pval)


def _contrast_group_counts(y: np.ndarray, meta: pd.DataFrame, sample_cols: list[str], age: int) -> tuple[np.ndarray, np.ndarray]:
    rows = meta.drop_duplicates("sample_id").set_index("sample_id").loc[sample_cols].reset_index()
    wt_idx = np.where((rows["age_months"].values == age) & (rows["genotype"].values == "WT"))[0]
    tg_idx = np.where((rows["age_months"].values == age) & (rows["genotype"].values == "TG"))[0]
    wt = np.isfinite(y[:, wt_idx]).sum(axis=1) if len(wt_idx) else np.zeros(y.shape[0], dtype=int)
    tg = np.isfinite(y[:, tg_idx]).sum(axis=1) if len(tg_idx) else np.zeros(y.shape[0], dtype=int)
    return wt.astype(int), tg.astype(int)


def _age_from_contrast(contrast: str) -> int:
    m = re.search(r"_(3|6|9|12)mo$", contrast)
    if not m:
        raise ValueError(f"Could not parse age from {contrast!r}")
    return int(m.group(1))


def _load_pseudobulk() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not PSEUDOBULK_PATH.exists():
        raise FileNotFoundError(
            f"{PSEUDOBULK_PATH} missing. Run "
            "`Rscript alz/ingest/build_5xfad_snrna_decomposition_pseudobulk.R` first."
        )
    pb = pd.read_csv(PSEUDOBULK_PATH)
    gene_map = pd.read_csv(GENE_MAP_PATH)
    counts = pd.read_csv(COUNTS_PATH)
    for df in (pb, counts):
        if "cell_type" in df.columns:
            named = ~df["cell_type"].astype(str).str.match(r"^cluster-\d+$", na=False)
            df.drop(df.index[~named], inplace=True)
    pb["age_months"] = pd.to_numeric(pb["age_months"], errors="coerce").astype("Int64")
    counts["age_months"] = pd.to_numeric(counts["age_months"], errors="coerce").astype("Int64")
    return pb, gene_map, counts


def _weights_for_tissue(pb: pd.DataFrame, gene_map: pd.DataFrame, tissue: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    sub = pb[pb["tissue"] == tissue].copy()
    meta_cols = ["tissue", "age_months", "genotype", "sample_id", "cell_type", "n_cells"]
    gene_cols = [c for c in sub.columns if c not in meta_cols]
    symbol_for_matched = dict(zip(gene_map["matched_gene"].astype(str), gene_map["gene_symbol"].astype(str)))
    keep_gene_cols = [g for g in gene_cols if g in symbol_for_matched]
    expr = sub[keep_gene_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    expr.columns = [symbol_for_matched[g] for g in keep_gene_cols]
    meta = sub[meta_cols].copy()
    meta["n_cells"] = pd.to_numeric(meta["n_cells"], errors="coerce").fillna(0).astype(float)

    weights_by_cell: dict[str, pd.DataFrame] = {}
    for cell_type, idx in meta.groupby("cell_type", sort=False).groups.items():
        rows = list(idx)
        cell_meta = meta.loc[rows].reset_index(drop=True)
        cell_expr = expr.loc[rows].reset_index(drop=True)
        out = pd.DataFrame(index=cell_meta["sample_id"].astype(str), columns=cell_expr.columns, dtype=float)
        for sample, sample_idx in meta.groupby("sample_id", sort=False).groups.items():
            sample_rows = list(sample_idx)
            sample_expr = expr.loc[sample_rows]
            denom = sample_expr.sum(axis=0)
            total_cells = float(meta.loc[sample_rows, "n_cells"].sum())
            this = meta.index[meta["sample_id"].astype(str).eq(str(sample)) & meta["cell_type"].astype(str).eq(str(cell_type))]
            if len(this) == 0:
                continue
            n_cells = float(meta.loc[this[0], "n_cells"])
            if n_cells <= 0 or total_cells <= 0:
                continue
            share = cell_expr.loc[cell_meta["sample_id"].astype(str).eq(str(sample))].iloc[0] / denom.replace(0, np.nan)
            out.loc[str(sample)] = share * (total_cells / n_cells)
        weights_by_cell[str(cell_type)] = out
    return weights_by_cell, meta


def _fit_one_celltype(
    tissue: str,
    track: str,
    cell_type: str,
    raw: pd.DataFrame,
    weights: pd.DataFrame,
    meta: pd.DataFrame,
    mea_caller: Callable | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    meta_cols = ["site_id", "gene_symbol", "motif", "site_position", "residue_type"]
    sample_cols = [c for c in raw.columns if c not in meta_cols and c in set(weights.index)]
    sample_cols = [c for c in sample_cols if c in set(meta["sample_id"].astype(str))]
    if len(sample_cols) < 8:
        audit = {"status": "skipped", "reason": f"only {len(sample_cols)} matched samples"}
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), audit

    site_genes = raw["gene_symbol"].astype(str)
    w = weights.reindex(index=sample_cols, columns=site_genes.unique())
    log_w_by_gene = np.log2(w.replace([np.inf, -np.inf], np.nan).where(w > 0))
    raw_vals = raw[sample_cols].to_numpy(dtype=float)
    log_w = np.vstack([
        log_w_by_gene.loc[sample_cols, gene].to_numpy(dtype=float)
        if gene in log_w_by_gene.columns else np.full(len(sample_cols), np.nan)
        for gene in site_genes
    ])
    y = raw_vals + log_w

    x = _build_design_matrix(meta, sample_cols)
    x_np = x.values
    param_names = list(x.columns)
    rank = int(np.linalg.matrix_rank(x_np))
    rank_deficient = rank < x_np.shape[1]
    contrast_vectors: dict[str, np.ndarray] = {}
    unestimable: list[str] = []
    for contrast, coefs in _contrast_coefs().items():
        c_vec = _build_contrast_vec(param_names, coefs)
        if c_vec is None or not _is_estimable(c_vec, x_np):
            unestimable.append(contrast)
        else:
            contrast_vectors[contrast] = c_vec
    if not contrast_vectors:
        audit = {
            "status": "skipped",
            "reason": f"no estimable contrasts (rank {rank} < {x_np.shape[1]})",
            "design_rank": rank,
            "design_n_params": int(x_np.shape[1]),
            "unestimable_contrasts": unestimable,
        }
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), audit

    if rank_deficient:
        betas, n_obs, xtxinv = _run_ols_pinv(y, x_np)
    else:
        betas, _, n_obs, xtxinv = kinase_enrich._run_ols_all_sites(y, x_np)

    site_rows = []
    mea_input = {}
    for contrast in _contrast_coefs():
        age = _age_from_contrast(contrast)
        if contrast not in contrast_vectors:
            lfc = np.full(y.shape[0], np.nan)
            se = np.full(y.shape[0], np.nan)
            t_stat = np.full(y.shape[0], np.nan)
            pval = np.full(y.shape[0], np.nan)
            fdr = np.full(y.shape[0], np.nan)
        else:
            lfc, se, t_stat, pval, fdr = _contrast_stats(y, betas, xtxinv, n_obs, contrast_vectors[contrast], x_np)
        wt_n, tg_n = _contrast_group_counts(y, meta, sample_cols, age)
        mea_input[contrast] = {"lfc": lfc}
        site_rows.append(pd.DataFrame({
            "tissue": tissue,
            "track": track,
            "cell_type": cell_type,
            "contrast": contrast,
            "site_id": raw["site_id"].astype(str).values,
            "gene_symbol": raw["gene_symbol"].astype(str).values,
            "motif": raw["motif"].astype(str).values,
            "lfc": lfc,
            "se": se,
            "t": t_stat,
            "pval": pval,
            "fdr": fdr,
            "n_obs": n_obs,
            "n_wt": wt_n,
            "n_tg": tg_n,
        }))
    site_ols = pd.concat(site_rows, ignore_index=True)

    _call_mea = mea_caller if mea_caller is not None else kinase_enrich._run_mea
    mea_df, shift_df, wins_df, subs_df = _call_mea(
        motif_series=raw["motif"],
        results_by_contrast=mea_input,
        lfc_key="lfc",
        site_ids=raw["site_id"].astype(str).values,
        gene_symbols=raw["gene_symbol"].astype(str).values,
        track=TRACKS[track]["kl_track"],
    )
    for df in (mea_df, shift_df, wins_df, subs_df):
        if isinstance(df, pd.DataFrame) and not df.empty:
            df["cell_type"] = cell_type
            df["track"] = track
            df["tissue"] = tissue
    audit = {
        "status": "ok",
        "n_samples_kept": len(sample_cols),
        "n_sites": int(y.shape[0]),
        "n_sites_fit": int(np.isfinite(betas[:, 0]).sum()),
        "design_rank": rank,
        "design_n_params": int(x_np.shape[1]),
        "rank_deficient": rank_deficient,
        "unestimable_contrasts": unestimable,
        "n_estimable_contrasts": len(contrast_vectors),
    }
    return mea_df, site_ols, shift_df, wins_df, subs_df, audit


def run(
    tissue_filter: set[str] | None = None,
    track_filter: set[str] | None = None,
    celltype_filter: set[str] | None = None,
    max_celltypes: int | None = None,
    out_dir: Path = OUT_DIR,
    mea_caller: Callable | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pb, gene_map, counts = _load_pseudobulk()
    all_mea: list[pd.DataFrame] = []
    all_site: list[pd.DataFrame] = []
    all_shift: list[pd.DataFrame] = []
    all_wins: list[pd.DataFrame] = []
    all_subs: list[pd.DataFrame] = []
    audit: dict[str, dict] = {}

    for tissue in TISSUES:
        if tissue_filter and tissue not in tissue_filter:
            continue
        weights_by_cell, meta = _weights_for_tissue(pb, gene_map, tissue)
        for track in TRACKS:
            if track_filter and track not in track_filter:
                continue
            raw_path = KINASE_DIR / f"{tissue}_{track}_raw_phospho_normalized.csv"
            if not raw_path.exists():
                continue
            raw = pd.read_csv(raw_path)
            raw = raw[raw["gene_symbol"].astype(str).isin(set(gene_map["gene_symbol"].astype(str)))].copy()
            raw["site_id"] = raw["site_id"].astype(str)
            print(f"[5xfad-celltype-mea] {tissue}/{track}: {len(raw):,} sites, {len(weights_by_cell)} cell types", flush=True)
            selected = list(weights_by_cell.items())
            if celltype_filter:
                selected = [(ct, w) for ct, w in selected if ct in celltype_filter]
            if max_celltypes is not None:
                selected = selected[:max_celltypes]
            for cell_type, weights in selected:
                print(f"  cell_type={cell_type}", flush=True)
                mea, site, shift, wins, subs, cell_audit = _fit_one_celltype(
                    tissue, track, cell_type, raw, weights, meta, mea_caller=mea_caller
                )
                audit[f"{tissue}|{track}|{cell_type}"] = cell_audit
                if not mea.empty:
                    all_mea.append(mea)
                if not site.empty:
                    all_site.append(site)
                if not shift.empty:
                    all_shift.append(shift)
                if not wins.empty:
                    all_wins.append(wins)
                if not subs.empty:
                    all_subs.append(subs)

    outputs = {
        "fivexfad_celltype_mea.parquet": pd.concat(all_mea, ignore_index=True) if all_mea else pd.DataFrame(),
        "fivexfad_celltype_site_level_ols.parquet": pd.concat(all_site, ignore_index=True) if all_site else pd.DataFrame(),
    }
    for name, df in outputs.items():
        if not df.empty:
            df.to_parquet(out_dir / name, index=False)
            print(f"[5xfad-celltype-mea] wrote {out_dir / name} rows={len(df):,}", flush=True)
    csv_outputs = {
        "fivexfad_celltype_mea_global_shift.csv": pd.concat(all_shift, ignore_index=True) if all_shift else pd.DataFrame(),
        "fivexfad_celltype_winsorized_sites.csv": pd.concat(all_wins, ignore_index=True) if all_wins else pd.DataFrame(),
        "fivexfad_celltype_substrate_sets.csv": pd.concat(all_subs, ignore_index=True) if all_subs else pd.DataFrame(),
        "fivexfad_celltype_counts.csv": counts,
    }
    for name, df in csv_outputs.items():
        if not df.empty:
            df.to_csv(out_dir / name, index=False)
            print(f"[5xfad-celltype-mea] wrote {out_dir / name} rows={len(df):,}", flush=True)
    audit_payload = config.provenance_stamp(
        cohort="5xFAD",
        analysis="celltype_decomposition_mea",
        n_groups=len(audit),
        n_ok=sum(1 for v in audit.values() if v.get("status") == "ok"),
        n_skipped=sum(1 for v in audit.values() if v.get("status") == "skipped"),
        per_group=audit,
    )
    (out_dir / "fivexfad_celltype_mea_audit.json").write_text(
        json.dumps(audit_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[5xfad-celltype-mea] wrote {out_dir / 'fivexfad_celltype_mea_audit.json'}", flush=True)


def run_mea_via_runner(
    scratch_dir: str | Path,
    tissue_filter: set[str] | None = None,
    track_filter: set[str] | None = None,
    celltype_filter: set[str] | None = None,
    max_celltypes: int | None = None,
) -> None:
    """Run per-cell-type MEA through the Phase-3 shared runner, writing all output to scratch_dir.

    Opt-in entry point.  Does NOT overwrite canonical outputs under OUT_DIR.
    Invoke via:
        pixi run python alz/cohorts/fivexfad/celltype_mea.py --runner-scratch-dir <DIR>
    or via the adapter directly:
        from alz.core.fivexfad_celltype_mea_adapter import run_via_runner
        run_via_runner(scratch_dir, ...)
    """
    from alz.core.fivexfad_celltype_mea_adapter import run_via_runner
    run_via_runner(
        scratch_dir=scratch_dir,
        tissue_filter=tissue_filter,
        track_filter=track_filter,
        celltype_filter=celltype_filter,
        max_celltypes=max_celltypes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tissue", action="append", choices=TISSUES)
    parser.add_argument("--track", action="append", choices=sorted(TRACKS))
    parser.add_argument("--cell-type", action="append", dest="cell_type")
    parser.add_argument("--max-celltypes", type=int)
    parser.add_argument("--runner-scratch-dir", metavar="DIR",
                        help="Run per-cell-type MEA through the Phase-3 shared runner; "
                             "writes to DIR (never to canonical OUT_DIR)")
    args = parser.parse_args()
    if args.runner_scratch_dir:
        run_mea_via_runner(
            scratch_dir=args.runner_scratch_dir,
            tissue_filter=set(args.tissue) if args.tissue else None,
            track_filter=set(args.track) if args.track else None,
            celltype_filter=set(args.cell_type) if args.cell_type else None,
            max_celltypes=args.max_celltypes,
        )
    else:
        run(
            tissue_filter=set(args.tissue) if args.tissue else None,
            track_filter=set(args.track) if args.track else None,
            celltype_filter=set(args.cell_type) if args.cell_type else None,
            max_celltypes=args.max_celltypes,
        )
