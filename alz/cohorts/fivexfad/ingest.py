#!/usr/bin/env python3
"""5xFAD mouse proteomics ingest and kinase enrichment.

This module reshapes the local 5xFAD Spectronaut reports into the same broad
artifact family as the Song and Mukesh kinase workflows, while keeping the
5xFAD cohort as supporting evidence. Cortex and hippocampus are modeled
independently; viewer integration combines them as a tissue filter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.bulk_mea import enrich as kinase_enrich
from alz.shared import config

DATASET_DIR = Path(config.REPO_ROOT) / "data" / "datasets" / "5xFAD"
PRIMARY_DIR = DATASET_DIR / "primary"
OUTPUT_DIR = Path(config.REPO_ROOT) / "outputs" / "reports" / "kinase_attribution_5xfad"
# Per-tissue Incytr pair-mode input root (mirrors data/derived/tcells_incytr_inputs).
INCYTR_INPUT_DIR = Path(config.REPO_ROOT) / "data" / "derived" / "5xfad_incytr_inputs"

ASSAY_LABELS = {
    "total": "total",
    "imac": "IMAC",
    "py": "pY",
    "kgg": "KGG",
    "ack": "AcK",
}
KINASE_TRACKS = {
    "st": {"assay": "imac", "residue_type": "ST", "kl_track": "st", "label": "IMAC/ST"},
    "py": {"assay": "py", "residue_type": "Y", "kl_track": "py", "label": "pY"},
}
TISSUES = ("cortex", "hippocampus")
AGES = (3, 6, 9, 12)

# Delivered sample-list DOCX files are present in the Lucie proteomics report
# bundle. These calls are centralized, exposed in sample_manifest.csv, and
# stamped with an explicit provenance string.
GENOTYPE_BY_AGE_SAMPLE: dict[int, dict[str, str]] = {
    3: {
        "1": "WT", "2": "WT", "4": "WT",
        "3": "TG", "9": "TG", "10": "TG", "11": "TG",
    },
    6: {
        "11": "WT", "12": "WT", "13": "WT", "14": "WT", "15": "WT",
        "16": "TG", "17": "TG", "18": "TG", "19": "TG",
    },
    9: {
        "10": "WT", "11": "WT",
        "12": "TG", "13": "TG", "14": "TG",
    },
    12: {
        "4": "WT", "7": "WT", "8": "WT", "10": "WT", "12": "WT",
        "3": "TG", "5": "TG", "6": "TG", "9": "TG", "11": "TG",
    },
}
GENOTYPE_PROVENANCE = (
    "delivered_lucie_proteomics_docx_sample_lists"
)

SENSITIVITY_RAW_RUNS = {"101525_Hippo_IMAC_3mon_10.raw"}

@dataclass(frozen=True)
class ReportSpec:
    tissue: str
    assay: str
    path: Path
    source_priority: str


REPORT_SPECS = [
    ReportSpec(
        "cortex", "total",
        PRIMARY_DIR / "cortex" / "proteomics" / "total"
        / "26Mar2026_MaleCtx5xFAD_Total_ProteinSiteReport.tsv",
        "delivered",
    ),
    ReportSpec(
        "cortex", "imac",
        PRIMARY_DIR / "cortex" / "proteomics" / "imac"
        / "20260527_104420_102325_LD_5xfad_IMAC_cortex_PTMSiteReport.tsv",
        "delivered",
    ),
    ReportSpec(
        "cortex", "py",
        PRIMARY_DIR / "cortex" / "proteomics" / "py"
        / "26Mar2026_MaleCtx5xFAD_pY_PTMSiteReport.tsv",
        "delivered",
    ),
    ReportSpec(
        "cortex", "kgg",
        PRIMARY_DIR / "cortex" / "proteomics" / "kgg"
        / "20260501_102617_260203_LD_cortex_KGG_Report.tsv",
        "available_not_kinase_mea_v1",
    ),
    ReportSpec(
        "cortex", "ack",
        PRIMARY_DIR / "cortex" / "proteomics" / "ack"
        / "31Mar2026_MaleCtx5xFAD_AcK_PTMSiteReport.tsv",
        "available_not_kinase_mea_v1",
    ),
    ReportSpec(
        "hippocampus", "total",
        PRIMARY_DIR / "hippocampus" / "proteomics" / "total"
        / "20260501_094814_5xFAD_hippocampus_3-6-9-12mo_totalproteome_Report.tsv",
        "delivered",
    ),
    ReportSpec(
        "hippocampus", "imac",
        PRIMARY_DIR / "hippocampus" / "proteomics" / "imac"
        / "23Mar2026_MaleHippo5XFAD_IMAC_EnsemblDB_PTMSiteReport.tsv",
        "delivered",
    ),
    ReportSpec(
        "hippocampus", "py",
        PRIMARY_DIR / "ensembl_corrections" / "hippocampus" / "py"
        / "23Mar2026_MaleHippo5XFAD_pY_CorrectEnsembleDB__PTMSiteReport.tsv",
        "corrected_ensembl_preferred",
    ),
    ReportSpec(
        "hippocampus", "kgg",
        PRIMARY_DIR / "ensembl_corrections" / "hippocampus" / "kgg"
        / "23Mar2026_MaleHippo5xFAD_KGG_CorrectEnsembleDB__PTMSiteReport.tsv",
        "available_not_kinase_mea_v1",
    ),
    ReportSpec(
        "hippocampus", "ack",
        PRIMARY_DIR / "hippocampus" / "proteomics" / "ack"
        / "20260501_092018_011926_Lucie_Hippocampus_male_Mo6-12_5xFAD_Report.tsv",
        "available_not_kinase_mea_v1",
    ),
]


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_header(path: Path) -> list[str]:
    return pd.read_csv(path, sep="\t", nrows=0).columns.tolist()


def _extract_raw_run(column: str, metric: str) -> str | None:
    if not column.endswith(metric):
        return None
    m = re.search(r"\]\s+(.+?\.raw)\.", column)
    return m.group(1) if m else None


def _quantity_columns(columns: Iterable[str], assay: str) -> dict[str, str]:
    metric = ".PG.Quantity" if assay == "total" else ".PTM.Quantity"
    out: dict[str, str] = {}
    for col in columns:
        raw_run = _extract_raw_run(str(col), metric)
        if raw_run:
            out[raw_run] = str(col)
    return out


def _parse_age_sample(raw_run: str) -> tuple[int | None, str | None, bool, str | None]:
    raw_no_ext = re.sub(r"\.raw$", "", raw_run, flags=re.IGNORECASE)
    age_match = re.search(
        r"(?:^|_)(?:M)?(?P<age>3|6|9|12)(?:mon|month|mth)?(?:_|$)",
        raw_no_ext,
        flags=re.IGNORECASE,
    )
    age = int(age_match.group("age")) if age_match else None
    if re.search(r"pool", raw_no_ext, flags=re.IGNORECASE):
        return age, None, True, None

    sample_match = re.search(
        r"(?:^|_)(?:M)?(?:3|6|9|12)(?:mon|month|mth)?_(?P<sample>\d+a?)(?:_|$)",
        raw_no_ext,
        flags=re.IGNORECASE,
    )
    if sample_match is None:
        sample_match = re.search(r"_(?P<sample>\d+a?)$", raw_no_ext, flags=re.IGNORECASE)
    if not age_match or not sample_match:
        return None, None, False, "unparsed_run_name"
    sample = sample_match.group("sample")
    return age, sample, False, None


def _duplicate_group(tissue: str, assay: str, age: int | None, sample: str | None) -> str:
    if tissue == "hippocampus" and assay == "imac" and age == 3 and sample:
        return re.sub(r"a$", "", sample)
    return sample or ""


def _genotype_for(age: int | None, sample: str | None) -> tuple[str | None, str]:
    if age is None or sample is None:
        return None, "not_applicable"
    base = re.sub(r"a$", "", sample)
    genotype = GENOTYPE_BY_AGE_SAMPLE.get(age, {}).get(base)
    if genotype is None:
        return None, "unassigned"
    return genotype, GENOTYPE_PROVENANCE


def _biological_sample_id(
    tissue: str,
    age: int | None,
    sample: str | None,
    genotype: str | None,
) -> str:
    if age is None or sample is None or genotype is None:
        return ""
    base = re.sub(r"a$", "", sample)
    return f"{tissue}_{age}mo_{genotype}_{base}"


def parse_raw_run(tissue: str, assay: str, raw_run: str) -> dict:
    age, sample, is_pool, parse_note = _parse_age_sample(raw_run)
    duplicate_group = _duplicate_group(tissue, assay, age, sample)
    genotype, genotype_source = _genotype_for(age, duplicate_group or sample)
    sensitivity = raw_run in SENSITIVITY_RAW_RUNS
    action = "exclude_pool" if is_pool else "primary"
    if not is_pool and (parse_note or genotype is None):
        action = "metadata_unassigned"
    return {
        "tissue": tissue,
        "assay": assay,
        "raw_run": raw_run,
        "age_months": age,
        "age": f"{age}mo" if age else "",
        "genotype": genotype or "",
        "biological_sample_id": _biological_sample_id(tissue, age, duplicate_group or sample, genotype),
        "pool": bool(is_pool),
        "duplicate_group": duplicate_group,
        "analysis_action": action,
        "sensitivity_flag": bool(sensitivity),
        "genotype_source": genotype_source,
        "parse_note": parse_note or "",
    }


def discover_reports() -> pd.DataFrame:
    records: list[dict] = []
    for spec in REPORT_SPECS:
        if not spec.path.exists():
            records.append({
                "tissue": spec.tissue,
                "assay": spec.assay,
                "source_path": str(spec.path.relative_to(config.REPO_ROOT)),
                "source_priority": spec.source_priority,
                "exists": False,
                "raw_run_count": 0,
                "analysis_scope": "kinase_mea_v1" if spec.assay in {"imac", "py"} else "provenance_only",
            })
            continue
        qcols = _quantity_columns(_read_header(spec.path), spec.assay)
        records.append({
            "tissue": spec.tissue,
            "assay": spec.assay,
            "source_path": str(spec.path.relative_to(config.REPO_ROOT)),
            "source_priority": spec.source_priority,
            "exists": True,
            "raw_run_count": len(qcols),
            "analysis_scope": "kinase_mea_v1" if spec.assay in {"imac", "py"} else "provenance_only",
        })
    return pd.DataFrame(records)


def build_sample_manifest() -> pd.DataFrame:
    records: list[dict] = []
    for spec in REPORT_SPECS:
        if not spec.path.exists():
            continue
        qcols = _quantity_columns(_read_header(spec.path), spec.assay)
        for raw_run in qcols:
            rec = parse_raw_run(spec.tissue, spec.assay, raw_run)
            rec["source_path"] = str(spec.path.relative_to(config.REPO_ROOT))
            rec["source_priority"] = spec.source_priority
            rec["analysis_scope"] = (
                "kinase_mea_v1" if spec.assay in {"imac", "py"} else "provenance_only"
            )
            records.append(rec)
    cols = [
        "tissue", "assay", "raw_run", "age_months", "age", "genotype",
        "biological_sample_id", "pool", "duplicate_group", "analysis_action",
        "sensitivity_flag", "genotype_source", "parse_note", "analysis_scope",
        "source_priority", "source_path",
    ]
    return pd.DataFrame(records, columns=cols)


def _extract_gene(row: pd.Series) -> str:
    for col in ("PG.Genes", "Gene Symbol", "gene_symbol"):
        val = row.get(col)
        if pd.notna(val):
            text = str(val).strip()
            if text and text.lower() not in {"nan", "pep"}:
                return re.split(r"[;,]", text)[0].strip()
    for col in ("PG.ProteinDescriptions", "PG.ProteinNames"):
        val = row.get(col)
        if pd.isna(val):
            continue
        m = re.search(r"gene_symbol:([A-Za-z0-9_.-]+)", str(val))
        if m:
            return m.group(1)
    return ""


def _site_id(row: pd.Series) -> str:
    collapse_key = row.get("PTM.CollapseKey")
    if pd.notna(collapse_key) and str(collapse_key).strip():
        return str(collapse_key).strip()
    protein = str(row.get("PTM.ProteinId", row.get("PG.ProteinAccessions", ""))).strip()
    aa = str(row.get("PTM.SiteAA", "")).strip()
    loc = str(row.get("PTM.SiteLocation", "")).strip()
    return f"{protein}_{aa}{loc}"


def _motif(row: pd.Series) -> str:
    flank = row.get("PTM.FlankingRegion")
    if pd.isna(flank):
        return ""
    motif = str(flank).strip().replace(".", "X")
    if not motif:
        return ""
    idx = len(motif) // 2
    chars = list(motif)
    if 0 <= idx < len(chars):
        chars[idx] = chars[idx].lower()
    return "".join(chars)


def _make_unique(values: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for val in values:
        key = str(val)
        n = seen.get(key, 0) + 1
        seen[key] = n
        out.append(key if n == 1 else f"{key}__dup{n}")
    return out


def _median_center_log2(mat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = mat.median(axis=0, skipna=True)
    global_med = float(np.nanmedian(med.values)) if len(med) else float("nan")
    norm = mat.subtract(med, axis=1).add(global_med)
    summary = pd.DataFrame({
        "sample": med.index.astype(str),
        "sample_median_log2_quantity": med.values,
        "global_median_log2_quantity": global_med,
        "median_shift_log2_quantity": med.values - global_med,
    })
    return norm, summary


def _read_log2_quantity_matrix(spec: ReportSpec, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(spec.path, sep="\t")
    qcols = _quantity_columns(df.columns, spec.assay)
    rows = manifest[
        (manifest["tissue"] == spec.tissue)
        & (manifest["assay"] == spec.assay)
        & (manifest["analysis_action"] == "primary")
    ].copy()
    rows = rows[rows["raw_run"].isin(qcols)]
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw_cols = rows["raw_run"].tolist()
    quant = df[[qcols[r] for r in raw_cols]].apply(pd.to_numeric, errors="coerce")
    quant.columns = raw_cols
    quant = quant.mask(quant <= 0)
    with np.errstate(divide="ignore"):
        log2q = np.log2(quant)
    log2q = pd.DataFrame(log2q, columns=raw_cols, index=df.index)
    norm, norm_summary = _median_center_log2(log2q)

    sample_ids = dict(zip(rows["raw_run"], rows["biological_sample_id"]))
    norm = norm.rename(columns=sample_ids)
    # Technical duplicates are deliberately averaged after log2 transform.
    norm = norm.T.groupby(level=0, sort=False).mean().T
    return df, norm


def _load_total_by_gene(tissue: str, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = next(s for s in REPORT_SPECS if s.tissue == tissue and s.assay == "total")
    src, mat = _read_log2_quantity_matrix(spec, manifest)
    if src.empty or mat.empty:
        return pd.DataFrame(), pd.DataFrame()
    genes = src.apply(_extract_gene, axis=1)
    keep = genes.astype(bool)
    total = mat.loc[keep].copy()
    total.insert(0, "gene_symbol", genes.loc[keep].values)
    total = total.groupby("gene_symbol", sort=False).median(numeric_only=True).reset_index()
    norm_summary_path = OUTPUT_DIR / f"{tissue}_total_normalization_summary.csv"
    return total, pd.DataFrame({"normalization_summary_path": [str(norm_summary_path)]})


def _track_prefix(tissue: str, track: str) -> str:
    return f"{tissue}_{track}"


def _track_spec(tissue: str, track: str) -> ReportSpec:
    assay = KINASE_TRACKS[track]["assay"]
    return next(s for s in REPORT_SPECS if s.tissue == tissue and s.assay == assay)


def build_track_matrices(tissue: str, track: str, manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    spec = _track_spec(tissue, track)
    src, raw_log2 = _read_log2_quantity_matrix(spec, manifest)
    if src.empty or raw_log2.empty:
        raise ValueError(f"No primary sample quantities for {tissue}/{track}")

    residue = KINASE_TRACKS[track]["residue_type"]
    site_aa = src["PTM.SiteAA"].fillna("").astype(str).str.upper()
    keep = site_aa.isin(list(residue))
    src = src.loc[keep].copy().reset_index(drop=True)
    raw_log2 = raw_log2.loc[keep.values].reset_index(drop=True)

    site_ids = _make_unique(src.apply(_site_id, axis=1))
    meta = pd.DataFrame({
        "site_id": site_ids,
        "gene_symbol": src.apply(_extract_gene, axis=1),
        "motif": src.apply(_motif, axis=1),
        "site_position": src.get("PTM.SiteLocation", pd.Series([""] * len(src))).astype(str),
        "residue_type": src.get("PTM.SiteAA", pd.Series([""] * len(src))).astype(str),
        "matched_protein": False,
    })
    raw_out = pd.concat([meta.drop(columns=["matched_protein"]), raw_log2], axis=1)

    total, _ = _load_total_by_gene(tissue, manifest)
    if total.empty:
        matched = pd.DataFrame(np.nan, index=raw_log2.index, columns=raw_log2.columns)
    else:
        total_idx = total.set_index("gene_symbol")
        matched = pd.DataFrame(index=raw_log2.index, columns=raw_log2.columns, dtype=float)
        for col in raw_log2.columns:
            if col not in total_idx.columns:
                matched[col] = np.nan
                continue
            matched[col] = meta["gene_symbol"].map(total_idx[col])
    meta["matched_protein"] = matched.notna().any(axis=1)
    stoich_vals = raw_log2 - matched
    stoich = pd.concat([meta, stoich_vals], axis=1)
    matched_total = pd.concat([
        meta[["site_id", "gene_symbol", "matched_protein"]],
        matched,
    ], axis=1)
    return {
        "raw_phospho_normalized": raw_out,
        "matched_total_protein": matched_total,
        "stoichiometry_matrix": stoich,
    }


def _build_design_matrix(mapping: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    meta = mapping.drop_duplicates("biological_sample_id").set_index("biological_sample_id")
    meta = meta.loc[sample_cols].reset_index()
    x = pd.DataFrame(index=range(len(meta)))
    x["const"] = 1.0
    x["age_6mo"] = (meta["age_months"] == 6).astype(float)
    x["age_9mo"] = (meta["age_months"] == 9).astype(float)
    x["age_12mo"] = (meta["age_months"] == 12).astype(float)
    x["TG"] = (meta["genotype"] == "TG").astype(float)
    x["TG_x_age6"] = x["TG"] * x["age_6mo"]
    x["TG_x_age9"] = x["TG"] * x["age_9mo"]
    x["TG_x_age12"] = x["TG"] * x["age_12mo"]
    return x


def _contrast_coefs() -> dict[str, dict[str, float]]:
    return {
        "TG_vs_WT_3mo": {"TG": 1.0},
        "TG_vs_WT_6mo": {"TG": 1.0, "TG_x_age6": 1.0},
        "TG_vs_WT_9mo": {"TG": 1.0, "TG_x_age9": 1.0},
        "TG_vs_WT_12mo": {"TG": 1.0, "TG_x_age12": 1.0},
    }


def _contrast_qc(tissue: str, track: str, mapping: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    meta = mapping.drop_duplicates("biological_sample_id").set_index("biological_sample_id")
    meta = meta.loc[sample_cols].reset_index()
    rows = []
    for age in AGES:
        sub = meta[meta["age_months"] == age]
        n_wt = int((sub["genotype"] == "WT").sum())
        n_tg = int((sub["genotype"] == "TG").sum())
        rows.append({
            "tissue": tissue,
            "track": track,
            "contrast": f"TG_vs_WT_{age}mo",
            "age_months": age,
            "n_wt": n_wt,
            "n_tg": n_tg,
            "contrast_status": "primary" if n_wt > 0 and n_tg > 0 else "missing_group",
        })
    return pd.DataFrame(rows)


def _age_from_contrast(contrast_name: str) -> int:
    m = re.search(r"_(3|6|9|12)mo$", contrast_name)
    if not m:
        raise ValueError(f"Could not parse contrast age from {contrast_name!r}")
    return int(m.group(1))


def _contrast_group_counts(
    y: np.ndarray,
    mapping: pd.DataFrame,
    sample_cols: list[str],
    age: int,
) -> tuple[np.ndarray, np.ndarray]:
    meta = mapping.drop_duplicates("biological_sample_id").set_index("biological_sample_id")
    meta = meta.loc[sample_cols].reset_index()
    wt_idx = np.where((meta["age_months"].values == age) & (meta["genotype"].values == "WT"))[0]
    tg_idx = np.where((meta["age_months"].values == age) & (meta["genotype"].values == "TG"))[0]
    wt_counts = np.isfinite(y[:, wt_idx]).sum(axis=1) if len(wt_idx) else np.zeros(y.shape[0], dtype=int)
    tg_counts = np.isfinite(y[:, tg_idx]).sum(axis=1) if len(tg_idx) else np.zeros(y.shape[0], dtype=int)
    return wt_counts.astype(int), tg_counts.astype(int)


def fit_track(
    tissue: str,
    track: str,
    manifest: pd.DataFrame,
    mea_caller: Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fit OLS contrasts and run MEA for one (tissue, track) combination.

    Parameters
    ----------
    tissue : str
        One of TISSUES (``"cortex"`` or ``"hippocampus"``).
    track : str
        One of KINASE_TRACKS (``"st"`` or ``"py"``).
    manifest : pd.DataFrame
        Loaded sample_manifest.csv.
    mea_caller : Callable | None
        Injectable MEA function.  Signature must match ``kinase_enrich._run_mea``
        exactly (positional motif_series, results_by_contrast, lfc_key; keyword
        site_ids, gene_symbols, track) and return the same 4-tuple
        ``(mea_df, shift_df, wins_df, substrate_df)``.

        ``None`` (default) → calls ``kinase_enrich._run_mea`` directly; this is
        the canonical path used by ``run_mea()`` and produces byte-identical
        output to historical runs.

        Provide a wrapper around ``MeaRunner._call_mea_unit`` when routing
        through the Phase-3 runner (e.g. from ``run_via_runner`` in the adapter)
        to record SkipRecords without duplicating OLS/concat logic here.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``site_level_ols``, ``mea_stoichiometry``, ``mea_raw_phospho``,
        ``mea_global_shift``, ``winsorized_sites``, ``mea_substrate_sets``,
        ``contrast_qc``.  The caller is responsible for writing these to disk.
    """
    _call_mea = mea_caller if mea_caller is not None else kinase_enrich._run_mea

    prefix = _track_prefix(tissue, track)
    stoich = pd.read_csv(OUTPUT_DIR / f"{prefix}_stoichiometry_matrix.csv")
    raw = pd.read_csv(OUTPUT_DIR / f"{prefix}_raw_phospho_normalized.csv")
    sample_cols = [
        c for c in stoich.columns
        if c not in {"site_id", "gene_symbol", "motif", "site_position", "residue_type", "matched_protein"}
    ]
    mapping = manifest[
        (manifest["tissue"] == tissue)
        & (manifest["assay"] == KINASE_TRACKS[track]["assay"])
        & (manifest["analysis_action"] == "primary")
    ].copy()
    mapping = mapping[mapping["biological_sample_id"].isin(sample_cols)]

    x = _build_design_matrix(mapping, sample_cols)
    x_np = x.values
    param_names = list(x.columns)
    contrast_coefs = _contrast_coefs()

    y_stoich = stoich[sample_cols].to_numpy(dtype=float)
    y_raw = raw[sample_cols].to_numpy(dtype=float)
    betas_s, _, nobs_s, xtxinv_s = kinase_enrich._run_ols_all_sites(y_stoich, x_np)
    betas_r, _, nobs_r, xtxinv_r = kinase_enrich._run_ols_all_sites(y_raw, x_np)

    results_by_contrast: dict[str, dict[str, np.ndarray]] = {}
    site_results = pd.DataFrame({
        "site_id": stoich["site_id"].values,
        "gene_symbol": stoich["gene_symbol"].values,
        "matched_protein": stoich["matched_protein"].values,
        "n_obs_stoich": nobs_s,
        "n_obs_raw": nobs_r,
    })
    n_params = len(param_names)
    for contrast_name, coefs in contrast_coefs.items():
        age = _age_from_contrast(contrast_name)
        c_vec = np.zeros(n_params)
        for param, weight in coefs.items():
            c_vec[param_names.index(param)] = weight
        lfc_s, p_s, fdr_s = kinase_enrich._contrast_stats(
            y_stoich, betas_s, xtxinv_s, nobs_s, c_vec, x_np, n_params
        )
        lfc_r, p_r, fdr_r = kinase_enrich._contrast_stats(
            y_raw, betas_r, xtxinv_r, nobs_r, c_vec, x_np, n_params
        )
        wt_s, tg_s = _contrast_group_counts(y_stoich, mapping, sample_cols, age)
        wt_r, tg_r = _contrast_group_counts(y_raw, mapping, sample_cols, age)
        results_by_contrast[contrast_name] = {
            "stoich_lfc": lfc_s,
            "stoich_pval": p_s,
            "stoich_fdr": fdr_s,
            "raw_lfc": lfc_r,
            "raw_pval": p_r,
            "raw_fdr": fdr_r,
        }
        site_results[f"stoich_lfc_{contrast_name}"] = lfc_s
        site_results[f"stoich_pval_{contrast_name}"] = p_s
        site_results[f"stoich_fdr_{contrast_name}"] = fdr_s
        site_results[f"stoich_n_wt_{contrast_name}"] = wt_s
        site_results[f"stoich_n_tg_{contrast_name}"] = tg_s
        site_results[f"raw_lfc_{contrast_name}"] = lfc_r
        site_results[f"raw_pval_{contrast_name}"] = p_r
        site_results[f"raw_fdr_{contrast_name}"] = fdr_r
        site_results[f"raw_n_wt_{contrast_name}"] = wt_r
        site_results[f"raw_n_tg_{contrast_name}"] = tg_r

    kl_track = KINASE_TRACKS[track]["kl_track"]
    mea_stoich, shift_stoich, wins_stoich, subs_stoich = _call_mea(
        stoich["motif"],
        results_by_contrast,
        "stoich_lfc",
        site_ids=stoich["site_id"].values,
        gene_symbols=stoich["gene_symbol"].values,
        track=kl_track,
    )
    mea_raw, shift_raw, wins_raw, subs_raw = _call_mea(
        stoich["motif"],
        results_by_contrast,
        "raw_lfc",
        site_ids=stoich["site_id"].values,
        gene_symbols=stoich["gene_symbol"].values,
        track=kl_track,
    )
    for df, analysis_track in [
        (mea_stoich, "stoichiometry"),
        (shift_stoich, "stoichiometry"),
        (wins_stoich, "stoichiometry"),
        (subs_stoich, "stoichiometry"),
        (mea_raw, "raw_phospho"),
        (shift_raw, "raw_phospho"),
        (wins_raw, "raw_phospho"),
        (subs_raw, "raw_phospho"),
    ]:
        if df is not None and not df.empty:
            df.insert(0, "analysis_track", analysis_track)
            df.insert(0, "tissue", tissue)

    return {
        "site_level_ols": site_results,
        "mea_stoichiometry": mea_stoich,
        "mea_raw_phospho": mea_raw,
        "mea_global_shift": pd.concat([shift_stoich, shift_raw], ignore_index=True),
        "winsorized_sites": pd.concat([wins_stoich, wins_raw], ignore_index=True),
        "mea_substrate_sets": pd.concat([subs_stoich, subs_raw], ignore_index=True),
        "contrast_qc": _contrast_qc(tissue, track, mapping, sample_cols),
    }


def _sample_group_map(manifest: pd.DataFrame, tissue: str) -> dict[str, str]:
    """biological_sample_id -> `<geno>_<age>mo` for primary, individual,
    genotyped samples of this tissue. Pools (analysis_action=exclude_pool) and
    ungenotyped rows are dropped — the group-level bulk uses every available
    individual sample per `<geno>_<age>`, independent of the per-animal snRNA
    join (matches the AD pr_median convention)."""
    rows = manifest[
        (manifest["tissue"] == tissue)
        & (manifest["analysis_action"] == "primary")
        & (manifest["genotype"].isin(["WT", "TG"]))
    ].drop_duplicates("biological_sample_id")
    return {
        str(r["biological_sample_id"]): f"{r['genotype']}_{int(r['age_months'])}mo"
        for _, r in rows.iterrows()
        if str(r["biological_sample_id"])
    }


def _linear_group_bulk(
    norm_log2: pd.DataFrame, key_cols: list[str], group_map: dict[str, str]
) -> pd.DataFrame:
    """Collapse a global-median-anchored log2 per-sample matrix to a LINEAR
    per-group bulk for the multiplicative `P_c = (N_total/N_c)×bulk×share`
    deconvolution.

    `_median_center_log2` already loading-equalizes each sample and re-anchors to
    the global median (so the scale stays in MS-intensity magnitude — the
    `pmax(pr,1)` Incytr floor would otherwise clobber a median-0 scale). Here we
    just exponentiate to linear and average the member samples within each
    `<geno>_<age>` group. Undetected (row, group) cells stay NaN (honestly
    missing). Columns are the `<geno>_<age>mo` group labels."""
    sample_cols = [c for c in norm_log2.columns if c in group_map]
    lin = np.power(2.0, norm_log2[sample_cols].apply(pd.to_numeric, errors="coerce"))
    out = norm_log2[key_cols].copy()
    by_group: dict[str, list[str]] = {}
    for c in sample_cols:
        by_group.setdefault(group_map[c], []).append(c)
    for group in sorted(by_group):
        out[group] = lin[by_group[group]].mean(axis=1, skipna=True)
    return out


def run_export_bulk() -> None:
    """Write per-tissue LINEAR per-group bulk for the Incytr pair-mode
    deconvolution: pr (total proteome, gene-keyed) + ps (IMAC/ST) + py (pY),
    both site-keyed. Consumed by alz/ingest/fivexfad_decompose.py."""
    _ensure_output_dir()
    manifest_path = OUTPUT_DIR / "sample_manifest.csv"
    manifest = (
        pd.read_csv(manifest_path) if manifest_path.exists() else build_sample_manifest()
    )
    site_keys = ["site_id", "gene_symbol", "motif"]
    for tissue in TISSUES:
        outdir = INCYTR_INPUT_DIR / tissue
        outdir.mkdir(parents=True, exist_ok=True)
        group_map = _sample_group_map(manifest, tissue)

        total, _ = _load_total_by_gene(tissue, manifest)
        pr_bulk = _linear_group_bulk(total, ["gene_symbol"], group_map)
        pr_bulk.to_csv(outdir / "pr_bulk_linear.csv", index=False)

        track_out = {"st": "ps_bulk_linear.csv", "py": "py_bulk_linear.csv"}
        for track, out_name in track_out.items():
            raw = build_track_matrices(tissue, track, manifest)["raw_phospho_normalized"]
            bulk = _linear_group_bulk(raw, site_keys, group_map)
            bulk.to_csv(outdir / out_name, index=False)
        groups = sorted(set(group_map.values()))
        print(f"[5xfad-export-bulk] {tissue}: pr={len(pr_bulk)} genes, "
              f"{len(groups)} groups {groups} -> {outdir}")


def run_ingest() -> None:
    _ensure_output_dir()
    manifest = build_sample_manifest()
    manifest.to_csv(OUTPUT_DIR / "sample_manifest.csv", index=False)
    discover_reports().to_csv(OUTPUT_DIR / "dataset_index.csv", index=False)
    for tissue in TISSUES:
        total, _ = _load_total_by_gene(tissue, manifest)
        total.to_csv(OUTPUT_DIR / f"{tissue}_total_proteome_normalized.csv", index=False)
        for track in KINASE_TRACKS:
            matrices = build_track_matrices(tissue, track, manifest)
            prefix = _track_prefix(tissue, track)
            for name, df in matrices.items():
                df.to_csv(OUTPUT_DIR / f"{prefix}_{name}.csv", index=False)
    summary = config.provenance_stamp(
        cohort="5xFAD",
        output_dir=str(OUTPUT_DIR.relative_to(config.REPO_ROOT)),
        genotype_provenance=GENOTYPE_PROVENANCE,
    )
    (OUTPUT_DIR / "ingest_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def run_mea() -> None:
    _ensure_output_dir()
    manifest_path = OUTPUT_DIR / "sample_manifest.csv"
    if not manifest_path.exists():
        run_ingest()
    manifest = pd.read_csv(manifest_path)
    for tissue in TISSUES:
        for track in KINASE_TRACKS:
            prefix = _track_prefix(tissue, track)
            required = OUTPUT_DIR / f"{prefix}_stoichiometry_matrix.csv"
            if not required.exists():
                matrices = build_track_matrices(tissue, track, manifest)
                for name, df in matrices.items():
                    df.to_csv(OUTPUT_DIR / f"{prefix}_{name}.csv", index=False)
            results = fit_track(tissue, track, manifest)
            for name, df in results.items():
                df.to_csv(OUTPUT_DIR / f"{prefix}_{name}.csv", index=False)


def run_mea_via_runner(scratch_dir: str) -> None:
    """Run bulk MEA through the Phase-3 shared runner to scratch_dir.

    Opt-in entry point.  Does NOT overwrite canonical outputs under OUTPUT_DIR.
    Invoke via:
        pixi run python alz/cohorts/fivexfad/ingest.py --runner-scratch-dir <DIR>
    or via the adapter directly:
        from alz.core.fivexfad_bulk_mea_adapter import run_via_runner
        run_via_runner(scratch_dir, manifest)
    """
    from alz.core.fivexfad_bulk_mea_adapter import run_via_runner
    _ensure_output_dir()
    manifest_path = OUTPUT_DIR / "sample_manifest.csv"
    if not manifest_path.exists():
        run_ingest()
    manifest = pd.read_csv(manifest_path)
    run_via_runner(scratch_dir=scratch_dir, manifest=manifest)


def print_summary() -> None:
    manifest = build_sample_manifest()
    index = discover_reports()
    print("\n5xFAD report index")
    print(index.to_string(index=False))
    print("\nPrimary sample counts by tissue/assay/age/genotype")
    primary = manifest[manifest["analysis_action"] == "primary"]
    if primary.empty:
        print("(none)")
    else:
        counts = (
            primary.drop_duplicates(["tissue", "assay", "biological_sample_id"])
            .groupby(["tissue", "assay", "age", "genotype"])
            .size()
            .rename("n_biological_samples")
            .reset_index()
        )
        print(counts.to_string(index=False))
    pools = manifest[manifest["pool"]]
    print(f"\nPool runs excluded from primary contrasts: {len(pools)}")
    if not pools.empty:
        print(pools[["tissue", "assay", "raw_run", "analysis_action"]].to_string(index=False))
    dups = manifest[
        (manifest["analysis_action"] == "primary")
        & (manifest.duplicated(["tissue", "assay", "biological_sample_id"], keep=False))
    ]
    print(f"\nTechnical duplicate raw runs collapsed after log2 transform: {len(dups)}")
    if not dups.empty:
        print(dups[["tissue", "assay", "raw_run", "biological_sample_id"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true", help="Print discovery and manifest summary")
    parser.add_argument("--ingest", action="store_true", help="Write manifest and normalized matrices")
    parser.add_argument("--mea", action="store_true", help="Run age-aware OLS and MEA")
    parser.add_argument("--run", action="store_true", help="Run ingest and MEA")
    parser.add_argument("--export-bulk", action="store_true",
                        help="Write per-tissue linear per-group bulk for Incytr deconvolution")
    parser.add_argument("--runner-scratch-dir", metavar="DIR",
                        help="Run bulk MEA through the Phase-3 shared runner; "
                             "writes to DIR (never to canonical OUTPUT_DIR)")
    args = parser.parse_args()
    if not any([args.summary, args.ingest, args.mea, args.run, args.export_bulk,
                args.runner_scratch_dir]):
        args.summary = True
    if args.summary:
        print_summary()
    if args.ingest or args.run:
        run_ingest()
    if args.mea or args.run:
        run_mea()
    if args.export_bulk:
        run_export_bulk()
    if args.runner_scratch_dir:
        run_mea_via_runner(args.runner_scratch_dir)


if __name__ == "__main__":
    main()
