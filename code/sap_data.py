"""
SAP data preparation: ingestion, pooling, RNA preprocessing, and diagnostics.

Phase 0: Loads the four input matrices (A_obs, X^base, bulk phospho, aggexp),
applies 5+1 cell-type pooling, MoR normalization, feature filtering, and
runs §6.0 pre-fit diagnostics.

Phase 1 (--phase1): Resolves aggexp sample mapping, preprocesses pseudobulk
RNA (§2.2), computes condition-variation scores (§2.3), and constructs
RNA-derived auxiliary covariates for the Hurdle-Tweedie model (§3.2.1).

Usage:
    python code/sap_data.py              # full data readiness report (Phase 0)
    python code/sap_data.py --check-only # diagnostics only (skip heavy loads)
    python code/sap_data.py --phase1     # Phase 0 + RNA preprocessing
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SAPData:
    """Container for all SAP model inputs after pooling and alignment."""

    a_obs: pd.DataFrame          # (24 samples × 6 cell types) composition fractions
    x_base: pd.DataFrame         # (J sites × 6 cell types) DESP baseline intensities
    bulk_phospho: pd.DataFrame   # (J sites × 24 samples) MoR-normalized bulk intensities
    sample_meta: pd.DataFrame    # (24 samples) with gender, timepoint, condition columns
    site_meta: pd.DataFrame      # (J sites) with site_id, protein_id, gene_symbol, etc.

    # Set after feature filtering
    n_sites_raw: int = 0
    n_sites_filtered: int = 0

    # Phase 1: RNA preprocessing outputs (set by load_all with include_rna=True)
    sample_id_map: Optional[Dict[int, str]] = None  # aggexp num → canonical sample ID
    gkp: Optional[pd.DataFrame] = None              # centered/scaled KP expression (§2.2)
    cvs: Optional[pd.DataFrame] = None              # condition-variation scores (§2.3)
    r_tensor: Optional[np.ndarray] = None            # (6, 24, n_sites) RNA covariates (§3.2.1)
    kinase_genes: Optional[List[str]] = None
    phosphatase_genes: Optional[List[str]] = None

    # Phase 2: intensity stratification (set by load_all with include_rna=True)
    intensity_strata: Optional[np.ndarray] = None    # (n_sites,) stratum index 0..Q-1


@dataclass
class DiagnosticResult:
    """Result of a single pre-fit diagnostic check."""

    name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""


@dataclass
class DiagnosticReport:
    """Collection of all §6.0 pre-fit diagnostics."""

    results: List[DiagnosticResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        lines = ["SAP §6.0 Pre-Fit Diagnostics", "=" * 40]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.name}: {r.value:.4g} (threshold: {r.threshold:.4g})")
            if r.detail:
                lines.append(f"         {r.detail}")
        lines.append("-" * 40)
        gate = "ALL PASSED" if self.all_passed else "BLOCKED — model cannot proceed"
        lines.append(f"  Gate: {gate}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sample metadata parsing
# ---------------------------------------------------------------------------

def parse_sample_id(sample_id: str) -> Dict[str, str]:
    """Parse a sample ID like 'fe_2mo_AppP' into components."""
    parts = sample_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected sample_id format: {sample_id!r}")
    return {"gender": parts[0], "timepoint": parts[1], "condition": parts[2]}


def build_sample_meta(sample_ids: List[str]) -> pd.DataFrame:
    """Build sample metadata DataFrame from A_obs row index."""
    records = [parse_sample_id(s) for s in sample_ids]
    return pd.DataFrame(records, index=sample_ids)


# ---------------------------------------------------------------------------
# A_obs loading and pooling (SAP §2.1)
# ---------------------------------------------------------------------------

def load_a_obs() -> pd.DataFrame:
    """Load and pool A_obs fractions from 10 cell types to 6 resolved types."""
    df = pd.read_csv(config.A_OBS_FILE, sep="\t", index_col="sample_id")

    # Validate expected columns
    missing = set(config.AOBS_POOL_MAP.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"A_obs missing expected cell types: {missing}")

    # Pool: sum fractions within each resolved type
    pooled = pd.DataFrame(0.0, index=df.index, columns=config.SAP_CELLTYPES)
    for orig_ct, resolved_ct in config.AOBS_POOL_MAP.items():
        pooled[resolved_ct] += df[orig_ct]

    # Sanity: rows should still sum to ~1.0
    row_sums = pooled.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        bad = row_sums[~np.isclose(row_sums, 1.0, atol=1e-4)]
        raise ValueError(f"Pooled A_obs rows don't sum to 1: {bad.to_dict()}")

    return pooled


# ---------------------------------------------------------------------------
# DESP baseline loading and pooling
# ---------------------------------------------------------------------------

def _parse_desp_column(col: str) -> Optional[Tuple[str, str, str, str]]:
    """Parse DESP column like 'ma_2mo_WTyp_Astrocytes' → (gender, tp, cond, celltype)."""
    # Match pattern: gender_timepoint_condition_celltype
    # Cell type may contain spaces so we split on first 3 underscores
    m = re.match(r'^(ma|fe)_(2mo|4mo|6mo)_(WTyp|Ttau|AppP|ApTt)_(.+)$', col)
    if m:
        return m.group(1), m.group(2), m.group(3), m.group(4)
    return None


def load_x_base() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load DESP baselines and pool to 6 cell types.

    The DESP file has columns like 'ma_2mo_WTyp_Glut' — one column per
    (sample × cell type). The global baseline X^base is the mean intensity
    across all 24 samples for each (site, cell type) pair.

    Returns:
        x_base: DataFrame (J sites × 6 cell types), mean intensities.
        site_meta: DataFrame (J sites) with site_id, protein_id, gene_symbol, etc.
    """
    df = pd.read_csv(config.DESP_BASELINE_FILE, low_memory=False)

    # Identify metadata vs sample columns
    meta_cols = ["Unnamed: 0", "site_id", "protein_id", "gene_symbol",
                 "prot_description", "site_position", "motif"]
    existing_meta = [c for c in meta_cols if c in df.columns]
    sample_cols = [c for c in df.columns if c not in existing_meta]

    site_meta = df[existing_meta].copy()
    if "Unnamed: 0" in site_meta.columns:
        site_meta = site_meta.drop(columns=["Unnamed: 0"])
    site_meta = site_meta.reset_index(drop=True)

    # Parse each sample column to extract cell type, then average across samples
    ct_data: Dict[str, List[str]] = {ct: [] for ct in config.SAP_CELLTYPES}
    for col in sample_cols:
        parsed = _parse_desp_column(col)
        if parsed is None:
            continue
        _, _, _, desp_ct = parsed
        resolved = config.DESP_POOL_MAP.get(desp_ct)
        if resolved is None:
            raise ValueError(f"DESP cell type {desp_ct!r} not in DESP_POOL_MAP")
        ct_data[resolved].append(col)

    # For each resolved cell type, compute mean across all samples that map to it.
    # For "Other" pool, we sum the DESP baselines (since the bulk signal from
    # multiple minor types combines additively in the mixing model).
    x_base = pd.DataFrame(index=range(len(df)))
    for ct in config.SAP_CELLTYPES:
        cols = ct_data[ct]
        if not cols:
            x_base[ct] = 0.0
        elif ct == "Other":
            # Sum across pooled types (each already averaged over samples),
            # but first average within each DESP type, then sum across types.
            by_desp_type: Dict[str, List[str]] = {}
            for col in cols:
                parsed = _parse_desp_column(col)
                desp_ct = parsed[3]
                by_desp_type.setdefault(desp_ct, []).append(col)
            x_base[ct] = sum(
                df[type_cols].mean(axis=1) for type_cols in by_desp_type.values()
            )
        else:
            x_base[ct] = df[cols].mean(axis=1)

    return x_base, site_meta


# ---------------------------------------------------------------------------
# Bulk phospho loading with MoR normalization (SAP §2.4)
# ---------------------------------------------------------------------------

def _parse_bulk_column(col: str) -> Optional[Tuple[str, str, str]]:
    """Parse bulk column like 'M_2mo_WT' → (gender, timepoint, condition).

    Bulk uses different condition suffixes: WT, T22, APP, T22/APP.
    """
    m = re.match(r'^([MF])_(2mo|4mo|6mo)_(.+)$', col)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def _bulk_condition_to_canonical(bulk_cond: str) -> str:
    """Map bulk condition suffix to canonical SAP condition name."""
    reverse_map = {v: k for k, v in config.BULK_CONDITION_MAP.items()}
    reverse_map["WT"] = "WTyp"
    return reverse_map.get(bulk_cond, bulk_cond)


def _bulk_gender_to_canonical(bulk_gender: str) -> str:
    """Map bulk gender code to canonical SAP gender."""
    return {"M": "ma", "F": "fe"}[bulk_gender]


def load_bulk_phospho() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load bulk phospho matrix and apply Median-of-Ratios normalization.

    Returns:
        intensities: DataFrame (J sites × 24 samples), MoR-normalized.
                     Columns renamed to canonical sample IDs matching A_obs.
        site_meta: DataFrame (J sites) with site_id, protein_id, etc.
        sample_order: List of canonical sample IDs in column order.
    """
    df = pd.read_csv(config.BULK_PHOSPHO_FILE, low_memory=False)

    # Separate metadata from intensity columns
    meta_cols = ["", "site_id", "protein_id", "gene_symbol",
                 "prot_description", "site_position", "motif"]
    existing_meta = [c for c in meta_cols if c in df.columns]
    sample_cols = [c for c in df.columns if _parse_bulk_column(c) is not None]

    site_meta = df[["site_id", "protein_id", "gene_symbol",
                     "prot_description", "site_position", "motif"]].copy()
    site_meta = site_meta.reset_index(drop=True)

    intensities = df[sample_cols].copy().reset_index(drop=True)

    # Rename columns to canonical sample IDs (e.g. M_2mo_WT → ma_2mo_WTyp)
    col_rename = {}
    for col in sample_cols:
        parsed = _parse_bulk_column(col)
        gender = _bulk_gender_to_canonical(parsed[0])
        tp = parsed[1]
        cond = _bulk_condition_to_canonical(parsed[2])
        col_rename[col] = f"{gender}_{tp}_{cond}"
    intensities = intensities.rename(columns=col_rename)
    sample_order = list(intensities.columns)

    # MoR normalization (SAP §2.4)
    intensities = _mor_normalize(intensities)

    return intensities, site_meta, sample_order


def _mor_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Median-of-Ratios normalization (Anders & Huber, 2010).

    For each sample, compute the ratio of each site's intensity to its
    geometric mean across samples; the normalization factor is the median
    of these ratios. Sites with zero geometric mean are excluded from
    factor computation.
    """
    # Replace zeros with NaN for geometric mean computation
    log_vals = np.log(df.replace(0, np.nan))
    geo_mean = np.exp(log_vals.mean(axis=1))

    # Exclude sites with undefined geometric mean
    valid = geo_mean > 0
    ratios = df.loc[valid].div(geo_mean[valid], axis=0)
    size_factors = ratios.median(axis=0)

    # Guard against zero size factors
    size_factors = size_factors.replace(0, 1.0)

    return df.div(size_factors, axis=1)


# ---------------------------------------------------------------------------
# Feature filtering (SAP §2.5)
# ---------------------------------------------------------------------------

def filter_sites(bulk: pd.DataFrame) -> pd.Series:
    """Return boolean mask of sites detected in >= MIN_SAMPLE_DETECTION samples.

    A site is "detected" in a sample if the value is finite and > 0.
    NaN values count as not detected.
    """
    detected = bulk.notna() & (bulk > 0)
    counts = detected.sum(axis=1)
    return counts >= config.MIN_SAMPLE_DETECTION


# ---------------------------------------------------------------------------
# Aggexp loading and pooling (SAP §2.2 — partial, structure only)
# ---------------------------------------------------------------------------

def load_aggexp_pooled() -> pd.DataFrame:
    """Load aggexp.csv, pool fine-grained clusters into 6 resolved types.

    Returns a MultiIndex DataFrame with index=(cell_type, sample_num) and
    ~30K gene columns. The sample number suffix (0-23) corresponds to the
    24 experimental groups, but the mapping to specific sample IDs requires
    the original Seurat object metadata. For Phase 0 we validate structure
    and pooling coverage only.
    """
    df = pd.read_csv(config.AGGEXP_FILE, index_col=0)

    # Parse row names: "Astrocytes14" → ("Astrocytes", 14)
    # Greedy prefix matching: try the full name first, then strip trailing digits.
    # This handles names like "Foxp2-Excitatory-Neurons-layers-6-and-2-3" where
    # the trailing "3" is part of the cluster name, not a sample number.
    row_ct = []
    row_num = []
    unmapped_prefixes = set()
    for row_name in df.index:
        # Strategy: try longest prefix match against known keys first
        resolved = None
        sample_num = 0
        for known_prefix in sorted(config.AGGEXP_POOL_MAP.keys(), key=len, reverse=True):
            if row_name == known_prefix:
                resolved = config.AGGEXP_POOL_MAP[known_prefix]
                sample_num = 0
                break
            if row_name.startswith(known_prefix):
                suffix = row_name[len(known_prefix):]
                if suffix.isdigit():
                    resolved = config.AGGEXP_POOL_MAP[known_prefix]
                    sample_num = int(suffix)
                    break

        if resolved is None:
            # Fallback: strip trailing digits
            m = re.match(r'^(.+?)(\d+)$', row_name)
            if m:
                prefix, sample_num = m.group(1), int(m.group(2))
            else:
                prefix, sample_num = row_name, 0
            unmapped_prefixes.add(prefix)
            resolved = "Other"

        row_ct.append(resolved)
        row_num.append(sample_num)

    if unmapped_prefixes:
        print(f"  Warning: unmapped aggexp prefixes (assigned to Other): {unmapped_prefixes}")

    df["_resolved_ct"] = row_ct
    df["_sample_num"] = row_num

    # Sum expression within each (resolved_cell_type, sample_num) group.
    # This implements the pooling described in SAP §2.1 for the pseudobulk.
    pooled = df.groupby(["_resolved_ct", "_sample_num"]).sum(numeric_only=True)
    pooled.index.names = ["cell_type", "sample_num"]

    return pooled


# ---------------------------------------------------------------------------
# Phase 1: Aggexp sample mapping
# ---------------------------------------------------------------------------

def build_aggexp_sample_map(
    aggexp_pooled: pd.DataFrame,
) -> Dict[int, str]:
    """Map aggexp sample numbers (0-23) to canonical sample IDs.

    Uses Hungarian algorithm matching of multi-cell-type library-size
    fingerprints against cell-count profiles from yuyu_clustersize.csv.
    Validated against transgene (hsAPP, hsMAPT) and sex-gene (Xist,
    Ddx3y/Uty/Kdm5d) expression patterns.
    """
    from scipy.optimize import linear_sum_assignment

    # Library-size matrix from aggexp: (24 samples × 5 cell types)
    ct_list = config.SAP_ESTIMATED_CELLTYPES
    lib_matrix = np.zeros((24, len(ct_list)))
    for j, ct in enumerate(ct_list):
        for i in range(24):
            lib_matrix[i, j] = pd.to_numeric(
                aggexp_pooled.loc[(ct, i)], errors="coerce"
            ).sum()

    # Cell-count matrix from clustersize: (24 samples × 5 cell types)
    cs = pd.read_csv(config.CLUSTERSIZE_FILE, index_col=0)
    sample_ids = list(cs.columns)
    cs_matrix = np.zeros((24, len(ct_list)))
    for j, ct in enumerate(ct_list):
        clusters = [c for c in config.CLUSTERSIZE_POOL_MAP[ct] if c in cs.index]
        cs_matrix[:, j] = cs.loc[clusters].sum().values

    # Standardize and match via Hungarian algorithm
    lib_std = (lib_matrix - lib_matrix.mean(axis=0)) / (lib_matrix.std(axis=0) + 1e-10)
    cs_std = (cs_matrix - cs_matrix.mean(axis=0)) / (cs_matrix.std(axis=0) + 1e-10)
    cost = np.zeros((24, 24))
    for i in range(24):
        for j in range(24):
            cost[i, j] = np.sum((lib_std[i] - cs_std[j]) ** 2)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {int(row_ind[k]): sample_ids[col_ind[k]] for k in range(24)}

    # Validate against transgene and sex-gene expression
    _validate_sample_map(aggexp_pooled, mapping)

    return mapping


def _validate_sample_map(
    aggexp_pooled: pd.DataFrame,
    mapping: Dict[int, str],
) -> None:
    """Validate sample mapping using transgene and sex-linked gene expression."""
    ct_list = config.SAP_ESTIMATED_CELLTYPES
    records = []
    for num in range(24):
        app = sum(aggexp_pooled.loc[(ct, num), "hsAPP"] for ct in ct_list)
        mapt = sum(aggexp_pooled.loc[(ct, num), "hsMAPT"] for ct in ct_list)
        xist = sum(aggexp_pooled.loc[(ct, num), "Xist"] for ct in ct_list)
        male = sum(
            aggexp_pooled.loc[(ct, num), g]
            for ct in ct_list
            for g in ["Ddx3y", "Uty", "Kdm5d"]
        )
        records.append({"num": num, "hsAPP": app, "hsMAPT": mapt,
                        "xist": xist, "male": male})

    df = pd.DataFrame(records)
    # Rank-based condition inference (top 12 by each transgene)
    app_top12 = set(df.nlargest(12, "hsAPP")["num"])
    mapt_top12 = set(df.nlargest(12, "hsMAPT")["num"])

    mismatches = []
    for num in range(24):
        sid = mapping[num]
        parts = sid.split("_")
        exp_gender, exp_cond = parts[0], parts[2]

        inf_gender = "fe" if df.loc[num, "xist"] > df.loc[num, "male"] else "ma"
        has_app = num in app_top12
        has_mapt = num in mapt_top12
        if has_app and has_mapt:
            inf_cond = "ApTt"
        elif has_app:
            inf_cond = "AppP"
        elif has_mapt:
            inf_cond = "Ttau"
        else:
            inf_cond = "WTyp"

        if inf_gender != exp_gender or inf_cond != exp_cond:
            mismatches.append(
                f"  sample {num} → {sid}: expected {exp_gender}/{exp_cond}, "
                f"inferred {inf_gender}/{inf_cond}"
            )

    if mismatches:
        raise ValueError(
            f"Aggexp sample mapping failed validation ({len(mismatches)}/24 mismatches):\n"
            + "\n".join(mismatches)
        )


# ---------------------------------------------------------------------------
# Phase 1: RNA preprocessing (SAP §2.2)
# ---------------------------------------------------------------------------

def _remap_aggexp_samples(
    aggexp_pooled: pd.DataFrame,
    sample_map: Dict[int, str],
) -> pd.DataFrame:
    """Replace numeric sample indices with canonical sample IDs."""
    new_index = []
    for ct, num in aggexp_pooled.index:
        sid = sample_map.get(num)
        if sid is not None:
            new_index.append((ct, sid))
        else:
            new_index.append((ct, f"unmapped_{num}"))
    aggexp_pooled = aggexp_pooled.copy()
    aggexp_pooled.index = pd.MultiIndex.from_tuples(new_index, names=["cell_type", "sample_id"])
    return aggexp_pooled


def _cpm_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """CPM normalization: divide each row by its total and multiply by 1e6."""
    row_sums = df.sum(axis=1)
    row_sums = row_sums.replace(0, 1.0)
    return df.div(row_sums, axis=0) * 1e6


def _filter_genes(df: pd.DataFrame, min_samples: int = None) -> pd.DataFrame:
    """Remove genes detected in fewer than min_samples of 24 samples.

    A gene is considered detected if nonzero in any cell type for that sample.
    """
    if min_samples is None:
        min_samples = config.MIN_GENE_DETECTION
    sample_ids = df.index.get_level_values("sample_id").unique()
    # For each gene, count how many unique samples have nonzero expression
    # (across any cell type)
    gene_detected = pd.DataFrame(0, index=sample_ids, columns=df.columns)
    for sid in sample_ids:
        sid_rows = df.xs(sid, level="sample_id")
        gene_detected.loc[sid] = (sid_rows > 0).any(axis=0).astype(int)
    detection_counts = gene_detected.sum(axis=0)
    keep_genes = detection_counts[detection_counts >= min_samples].index
    return df[keep_genes]


def _get_kinase_genes(aggexp_columns: pd.Index) -> List[str]:
    """Extract kinase gene symbols present in aggexp from kldata.csv."""
    kldata = pd.read_csv(config.KLDATA_FILE, usecols=["GENE_NAME"], low_memory=False)
    # kldata uses human symbols (uppercase); convert to mouse title-case
    human_genes = kldata["GENE_NAME"].dropna().unique()
    mouse_genes = set()
    for g in human_genes:
        mouse = g[0].upper() + g[1:].lower() if len(g) > 1 else g.upper()
        mouse_genes.add(mouse)
    # Also load the existing kinase-to-gene mapping cache
    if os.path.exists(config.MAPPING_CACHE_FILE):
        cache = pd.read_csv(config.MAPPING_CACHE_FILE)
        if "gene_symbol" in cache.columns:
            for g in cache["gene_symbol"].dropna():
                mouse_genes.add(g)
    return sorted(set(aggexp_columns) & mouse_genes)


def _get_phosphatase_genes(aggexp_columns: pd.Index) -> List[str]:
    """Build phosphatase gene list from config prefixes and extras."""
    genes = set()
    for prefix in config.PHOSPHATASE_GENE_PREFIXES:
        for col in aggexp_columns:
            if col.startswith(prefix):
                genes.add(col)
    for g in config.PHOSPHATASE_GENES_EXTRA:
        if g in aggexp_columns:
            genes.add(g)
    return sorted(genes)


def _center_scale_per_celltype(df: pd.DataFrame) -> pd.DataFrame:
    """Within each cell type, z-score each gene across 24 samples."""
    result_parts = []
    for ct in df.index.get_level_values("cell_type").unique():
        ct_data = df.loc[ct].copy()
        means = ct_data.mean(axis=0)
        stds = ct_data.std(axis=0)
        stds = stds.replace(0, 1.0)  # zero-variance genes → 0 after centering
        ct_data = (ct_data - means) / stds
        ct_data.index = pd.MultiIndex.from_tuples(
            [(ct, sid) for sid in ct_data.index], names=["cell_type", "sample_id"]
        )
        result_parts.append(ct_data)
    return pd.concat(result_parts)


def preprocess_aggexp(
    aggexp_pooled: pd.DataFrame,
    sample_map: Dict[int, str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Full §2.2 pseudobulk preprocessing pipeline.

    Returns:
        gkp: Centered/scaled kinase-phosphatase expression matrix.
             MultiIndex (cell_type, sample_id), ~600-700 gene columns.
        kinase_genes: List of kinase genes in gkp.
        phosphatase_genes: List of phosphatase genes in gkp.
    """
    print("  Remapping sample indices...")
    df = _remap_aggexp_samples(aggexp_pooled, sample_map)
    # Keep only the 5 estimated cell types (drop "Other")
    est_types = config.SAP_ESTIMATED_CELLTYPES
    df = df.loc[df.index.get_level_values("cell_type").isin(est_types)]

    print("  CPM normalization...")
    df = _cpm_normalize(df)

    print("  Gene filtering (≥{} samples)...".format(config.MIN_GENE_DETECTION))
    n_before = df.shape[1]
    df = _filter_genes(df)
    print(f"    {n_before} → {df.shape[1]} genes")

    print("  Kinase/phosphatase subsetting...")
    kinase_genes = _get_kinase_genes(df.columns)
    phosphatase_genes = _get_phosphatase_genes(df.columns)
    kp_genes = sorted(set(kinase_genes) | set(phosphatase_genes))
    df = df[kp_genes]
    print(f"    {len(kinase_genes)} kinases + {len(phosphatase_genes)} phosphatases "
          f"= {len(kp_genes)} KP genes")

    print("  Per-cell-type centering and scaling...")
    gkp = _center_scale_per_celltype(df)

    return gkp, kinase_genes, phosphatase_genes


# ---------------------------------------------------------------------------
# Phase 1: Condition-Variation Score (SAP §2.3)
# ---------------------------------------------------------------------------

def compute_cvs(
    gkp: pd.DataFrame,
    sample_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Compute condition-variation scores (CVS) per cell type × condition.

    CVS_{k,c} = mean over KP genes of |mean(G̃_{k,c,g}) - mean(G̃_{k,WTyp,g})|

    Returns: DataFrame with index=estimated cell types, columns=non-WTyp conditions.
    """
    ct_list = config.SAP_ESTIMATED_CELLTYPES
    conditions = [c for c in config.SAP_CONDITIONS if c != "WTyp"]

    # Build sample-to-condition lookup
    cond_samples: Dict[str, List[str]] = {}
    for sid, row in sample_meta.iterrows():
        cond_samples.setdefault(row["condition"], []).append(sid)

    cvs_data = {}
    for ct in ct_list:
        ct_data = gkp.loc[ct]  # (24 samples × genes)
        wtyp_mean = ct_data.loc[cond_samples["WTyp"]].mean(axis=0)
        for cond in conditions:
            cond_mean = ct_data.loc[cond_samples[cond]].mean(axis=0)
            cvs_data[(ct, cond)] = float((cond_mean - wtyp_mean).abs().mean())

    result = pd.DataFrame(
        {cond: [cvs_data[(ct, cond)] for ct in ct_list] for cond in conditions},
        index=ct_list,
    )
    return result


# ---------------------------------------------------------------------------
# Phase 1: RNA-derived auxiliary covariates (SAP §3.2.1)
# ---------------------------------------------------------------------------

def build_kinase_substrate_map(
    site_meta: pd.DataFrame,
    kinase_genes_in_gkp: List[str],
) -> Dict[int, List[str]]:
    """Map each phosphosite to its upstream kinase gene(s) via kldata.csv.

    Joins site_meta (gene_symbol, site_position) with kldata (gene_symbol,
    site_position → GENE_NAME). Returns dict: site_index → list of kinase
    gene symbols present in the G^KP matrix.
    """
    kldata = pd.read_csv(
        config.KLDATA_FILE,
        usecols=["gene_symbol", "site_position", "GENE_NAME"],
        low_memory=False,
    )
    # kldata gene_symbol is the substrate; GENE_NAME is the kinase
    # Convert kinase names to mouse case
    kldata["kinase_mouse"] = kldata["GENE_NAME"].apply(
        lambda g: g[0].upper() + g[1:].lower() if isinstance(g, str) and len(g) > 1
        else (g.upper() if isinstance(g, str) else "")
    )

    # Build lookup: (substrate_gene, site_pos) → set of kinase genes
    kp_set = set(kinase_genes_in_gkp)
    substrate_to_kinases: Dict[Tuple[str, str], set] = {}
    for _, row in kldata.iterrows():
        key = (row["gene_symbol"], str(row["site_position"]))
        kinase = row["kinase_mouse"]
        if kinase in kp_set:
            substrate_to_kinases.setdefault(key, set()).add(kinase)

    # Match against site_meta
    result: Dict[int, List[str]] = {}
    matched = 0
    for idx in range(len(site_meta)):
        gene = site_meta.iloc[idx].get("gene_symbol", "")
        pos = str(site_meta.iloc[idx].get("site_position", ""))
        kinases = substrate_to_kinases.get((gene, pos), set())
        if kinases:
            result[idx] = sorted(kinases)
            matched += 1

    print(f"  Kinase-substrate mapping: {matched}/{len(site_meta)} sites "
          f"({100 * matched / len(site_meta):.1f}%) have upstream kinases in G^KP")
    return result


def compute_r_tensor(
    gkp: pd.DataFrame,
    sample_meta: pd.DataFrame,
    kinase_map: Dict[int, List[str]],
    phosphatase_genes: List[str],
    n_sites: int,
) -> np.ndarray:
    """Construct the RNA-derived covariate tensor r_{k,i,j} (SAP §3.2.1).

    r_{k,i,j} = mean(G̃^KP for upstream kinases of site j)
                - mean(G̃^KP for phosphatases)  [global, not site-specific]

    Shape: (n_celltypes, n_samples, n_sites) = (6, 24, n_sites).
    The 6th cell type ("Other") gets zeros since it receives no condition effects.

    For sites with no known upstream kinase, r_{k,i,j} = 0.
    """
    ct_list = config.SAP_CELLTYPES  # all 6 including Other
    est_types = config.SAP_ESTIMATED_CELLTYPES
    sample_ids = list(sample_meta.index)

    r = np.zeros((len(ct_list), len(sample_ids), n_sites))

    # Precompute global phosphatase mean per (cell_type, sample)
    phos_available = [g for g in phosphatase_genes if g in gkp.columns]
    phos_means = {}  # (ct_idx, sample_idx) → mean phosphatase expression
    for ki, ct in enumerate(est_types):
        for si, sid in enumerate(sample_ids):
            if phos_available:
                phos_means[(ki, si)] = float(gkp.loc[(ct, sid), phos_available].mean())
            else:
                phos_means[(ki, si)] = 0.0

    # Compute r for each site
    for j in range(n_sites):
        kinases = kinase_map.get(j)
        if not kinases:
            continue  # r stays 0 for unmapped sites
        kin_available = [g for g in kinases if g in gkp.columns]
        if not kin_available:
            continue
        for ki, ct in enumerate(est_types):
            for si, sid in enumerate(sample_ids):
                kin_mean = float(gkp.loc[(ct, sid), kin_available].mean())
                r[ki, si, j] = kin_mean - phos_means[(ki, si)]

    return r


# ---------------------------------------------------------------------------
# Pre-fit diagnostics (SAP §6.0)
# ---------------------------------------------------------------------------

def check_composition_rank(a_obs: pd.DataFrame) -> DiagnosticResult:
    """SVD of the 24×6 pooled composition matrix. Rank >= 5, min SV > 0.01."""
    U, sv, Vt = np.linalg.svd(a_obs.values, full_matrices=False)
    rank = int(np.sum(sv > 1e-10))
    min_sv = float(sv[-1]) if len(sv) > 0 else 0.0

    passed = rank >= config.COMPOSITION_MIN_RANK and min_sv > config.COMPOSITION_MIN_SV
    return DiagnosticResult(
        name="Composition rank",
        passed=passed,
        value=rank,
        threshold=config.COMPOSITION_MIN_RANK,
        detail=f"rank={rank}, singular values=[{', '.join(f'{s:.4f}' for s in sv)}]",
    )


def check_composition_min_sv(a_obs: pd.DataFrame) -> DiagnosticResult:
    """Smallest singular value of composition matrix must exceed threshold."""
    _, sv, _ = np.linalg.svd(a_obs.values, full_matrices=False)
    min_sv = float(sv[-1])
    return DiagnosticResult(
        name="Composition min singular value",
        passed=min_sv > config.COMPOSITION_MIN_SV,
        value=min_sv,
        threshold=config.COMPOSITION_MIN_SV,
        detail=f"smallest SV={min_sv:.6f}",
    )


def build_effective_design_matrix(
    a_obs: pd.DataFrame,
    sample_meta: pd.DataFrame,
) -> np.ndarray:
    """Construct the effective design matrix D for condition number check.

    The design matrix includes:
    - A_obs columns for the 5 estimated cell types × 3 factorial indicators
      = 15 columns (the penalized condition parameters)
    - 1 gender indicator column
    - 2 timepoint indicator columns
    = 18 columns total (the fixed design; RNA covariates excluded here).
    """
    cols = []

    # Build indicator vectors from sample metadata
    conditions = sample_meta["condition"].values
    app_ind = np.array([config.SAP_FACTORIAL[c][0] for c in conditions], dtype=float)
    tau_ind = np.array([config.SAP_FACTORIAL[c][1] for c in conditions], dtype=float)
    int_ind = np.array([config.SAP_FACTORIAL[c][2] for c in conditions], dtype=float)

    # Cell-type × factorial Hadamard products (15 columns)
    for ct in config.SAP_ESTIMATED_CELLTYPES:
        a_k = a_obs[ct].values
        cols.append(a_k * app_ind)
        cols.append(a_k * tau_ind)
        cols.append(a_k * int_ind)

    # Global gender indicator (1 column: female=1)
    female_ind = (sample_meta["gender"] == "fe").astype(float).values
    cols.append(female_ind)

    # Global timepoint indicators (2 columns: 4mo, 6mo)
    cols.append((sample_meta["timepoint"] == "4mo").astype(float).values)
    cols.append((sample_meta["timepoint"] == "6mo").astype(float).values)

    D = np.column_stack(cols)
    return D


def _condition_number(D: np.ndarray) -> float:
    """Condition number κ(D̃) = σ_max/σ_min of column-standardized D."""
    norms = np.linalg.norm(D, axis=0)
    norms[norms == 0] = 1.0
    D_std = D / norms
    sv = np.linalg.svd(D_std, compute_uv=False)
    sv = sv[sv > 1e-15]
    if len(sv) == 0:
        return np.inf
    return float(sv[0] / sv[-1])


def check_design_conditioning(
    a_obs: pd.DataFrame,
    sample_meta: pd.DataFrame,
) -> DiagnosticResult:
    """Condition number κ(D̃) of column-standardized design matrix < threshold."""
    D = build_effective_design_matrix(a_obs, sample_meta)
    kappa = _condition_number(D)

    n_zero_cols = int(np.sum(np.linalg.norm(D, axis=0) == 0))
    detail = f"kappa={kappa:.1f}, matrix shape={D.shape}"
    if n_zero_cols > 0:
        detail += f", {n_zero_cols} zero-norm columns"

    return DiagnosticResult(
        name="Design matrix conditioning",
        passed=kappa < config.CONDITION_NUMBER_MAX,
        value=kappa,
        threshold=config.CONDITION_NUMBER_MAX,
        detail=detail,
    )


def check_effective_dof(
    a_obs: pd.DataFrame,
    sample_meta: pd.DataFrame,
) -> DiagnosticResult:
    """Effective DOF audit: tr(H) of hat matrix at initial values < threshold.

    Under the global-covariate model with full Group Lasso activation (worst case),
    the effective DOF equals the number of non-zero-variance design columns.
    This is an upper bound; sparsity reduces it in practice.
    """
    D = build_effective_design_matrix(a_obs, sample_meta)

    # Remove zero columns (e.g. interaction for WTyp-only samples)
    col_norms = np.linalg.norm(D, axis=0)
    D_active = D[:, col_norms > 1e-10]

    # Hat matrix trace = rank of D_active (at initial values, no penalization)
    rank = np.linalg.matrix_rank(D_active)
    n_samples = D.shape[0]
    residual_dof = n_samples - rank

    passed = rank <= config.MAX_EFFECTIVE_DOF and residual_dof >= config.MIN_RESIDUAL_DOF
    return DiagnosticResult(
        name="Effective DOF (unpenalized upper bound)",
        passed=passed,
        value=rank,
        threshold=config.MAX_EFFECTIVE_DOF,
        detail=f"rank={rank}, residual DOF={residual_dof} (need >= {config.MIN_RESIDUAL_DOF})",
    )


def build_design_matrix_no_interaction(
    a_obs: pd.DataFrame,
    sample_meta: pd.DataFrame,
) -> np.ndarray:
    """Design matrix without interaction terms (simplification cascade step 1)."""
    cols = []
    conditions = sample_meta["condition"].values
    app_ind = np.array([config.SAP_FACTORIAL[c][0] for c in conditions], dtype=float)
    tau_ind = np.array([config.SAP_FACTORIAL[c][1] for c in conditions], dtype=float)

    # Cell-type × main effects only (10 columns instead of 15)
    for ct in config.SAP_ESTIMATED_CELLTYPES:
        a_k = a_obs[ct].values
        cols.append(a_k * app_ind)
        cols.append(a_k * tau_ind)

    # Global covariates (3 columns)
    cols.append((sample_meta["gender"] == "fe").astype(float).values)
    cols.append((sample_meta["timepoint"] == "4mo").astype(float).values)
    cols.append((sample_meta["timepoint"] == "6mo").astype(float).values)

    return np.column_stack(cols)


def check_design_conditioning_no_interaction(
    a_obs: pd.DataFrame,
    sample_meta: pd.DataFrame,
) -> DiagnosticResult:
    """Conditioning check after dropping interaction terms (cascade step 1)."""
    D = build_design_matrix_no_interaction(a_obs, sample_meta)
    kappa = _condition_number(D)

    return DiagnosticResult(
        name="Design conditioning (no interaction, cascade step 1)",
        passed=kappa < config.CONDITION_NUMBER_MAX,
        value=kappa,
        threshold=config.CONDITION_NUMBER_MAX,
        detail=f"kappa={kappa:.1f}, matrix shape={D.shape}",
    )


def check_rna_covariate_vif(
    r_tensor: np.ndarray,
    a_obs: pd.DataFrame,
    sample_meta: pd.DataFrame,
) -> DiagnosticResult:
    """VIF of r_{k,i,j} against factorial design columns (§6.0).

    For each site j, the RNA covariate vector (stacked across 5 estimated
    cell types × 24 samples = 120 observations) is regressed against the
    factorial design columns (A_{i,k} * indicator). VIF = 1/(1 - R²).
    Reports fraction of sites with VIF < threshold.
    """
    # Build factorial design: (24 × 15) — same as condition columns in D
    conditions = sample_meta["condition"].values
    app_ind = np.array([config.SAP_FACTORIAL[c][0] for c in conditions], dtype=float)
    tau_ind = np.array([config.SAP_FACTORIAL[c][1] for c in conditions], dtype=float)
    int_ind = np.array([config.SAP_FACTORIAL[c][2] for c in conditions], dtype=float)

    # For VIF, we check collinearity of r with the design per cell type.
    # Stack: for each site, r is (5 cell types × 24 samples) = 120 values.
    # Design X is (5 × 24) × 15 (cell-type-specific factorial indicators).
    n_est = len(config.SAP_ESTIMATED_CELLTYPES)
    n_samples = a_obs.shape[0]

    # Build stacked design matrix (n_est * n_samples, 15)
    X_blocks = []
    for ki, ct in enumerate(config.SAP_ESTIMATED_CELLTYPES):
        a_k = a_obs[ct].values  # (24,)
        X_blocks.append(np.column_stack([a_k * app_ind, a_k * tau_ind, a_k * int_ind]))
    X_design = np.vstack(X_blocks)  # (120, 3) per cell type → but we need all 15

    # Actually: the full design has 15 columns (5 cell types × 3 indicators).
    # For each cell type k, columns 3k:3k+3 are nonzero only for rows from ct k.
    X_full = np.zeros((n_est * n_samples, n_est * 3))
    for ki, ct in enumerate(config.SAP_ESTIMATED_CELLTYPES):
        a_k = a_obs[ct].values
        row_start = ki * n_samples
        row_end = (ki + 1) * n_samples
        X_full[row_start:row_end, ki * 3] = a_k * app_ind
        X_full[row_start:row_end, ki * 3 + 1] = a_k * tau_ind
        X_full[row_start:row_end, ki * 3 + 2] = a_k * int_ind

    n_sites = r_tensor.shape[2]
    vifs = np.zeros(n_sites)

    # Precompute (X^T X)^{-1} X^T for projection
    # Use pseudoinverse for stability
    XtX = X_full.T @ X_full
    try:
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(XtX.shape[0]))
        hat_matrix_factor = XtX_inv @ X_full.T  # (15, 120)
    except np.linalg.LinAlgError:
        # Singular — VIF undefined, report as failure
        return DiagnosticResult(
            name="RNA covariate VIF",
            passed=False,
            value=0.0,
            threshold=config.VIF_PASS_FRACTION,
            detail="Design matrix singular; cannot compute VIF",
        )

    for j in range(n_sites):
        # Stack r across 5 estimated cell types (exclude "Other" = index 5)
        r_vec = r_tensor[:n_est, :, j].ravel()  # (120,)
        if np.all(r_vec == 0):
            vifs[j] = 1.0  # no collinearity for zero covariate
            continue
        # R² of r regressed on X
        r_hat = X_full @ (hat_matrix_factor @ r_vec)
        ss_res = np.sum((r_vec - r_hat) ** 2)
        ss_tot = np.sum((r_vec - r_vec.mean()) ** 2)
        if ss_tot < 1e-15:
            vifs[j] = 1.0
            continue
        r_squared = 1.0 - ss_res / ss_tot
        r_squared = min(r_squared, 1.0 - 1e-10)  # cap to avoid inf
        vifs[j] = 1.0 / (1.0 - r_squared)

    frac_below = float(np.mean(vifs < config.VIF_THRESHOLD))
    median_vif = float(np.median(vifs))
    max_vif = float(np.max(vifs))
    passed = frac_below >= config.VIF_PASS_FRACTION

    return DiagnosticResult(
        name="RNA covariate VIF",
        passed=passed,
        value=frac_below,
        threshold=config.VIF_PASS_FRACTION,
        detail=(f"{frac_below * 100:.1f}% of sites have VIF < {config.VIF_THRESHOLD} "
                f"(median={median_vif:.2f}, max={max_vif:.1f})"),
    )


def compute_intensity_strata(x_base: pd.DataFrame) -> np.ndarray:
    """Partition sites into Q intensity strata by mean baseline intensity (§4.2).

    Returns array of stratum indices (0 to Q-1), one per site.
    """
    mean_intensity = x_base.mean(axis=1).values
    n_strata = config.N_INTENSITY_STRATA
    # Use quantile-based binning
    quantiles = np.quantile(mean_intensity, np.linspace(0, 1, n_strata + 1))
    # digitize: bin edges, right=False gives left-closed intervals
    strata = np.digitize(mean_intensity, quantiles[1:-1])  # 0 to Q-1
    return strata


def run_diagnostics(
    a_obs: pd.DataFrame,
    sample_meta: pd.DataFrame,
    r_tensor: Optional[np.ndarray] = None,
) -> DiagnosticReport:
    """Run all §6.0 pre-fit diagnostics."""
    report = DiagnosticReport()
    report.results.append(check_composition_rank(a_obs))
    report.results.append(check_composition_min_sv(a_obs))
    report.results.append(check_design_conditioning(a_obs, sample_meta))
    report.results.append(check_effective_dof(a_obs, sample_meta))

    # If full design fails conditioning, test simplification cascade
    if not report.results[2].passed:
        report.results.append(
            check_design_conditioning_no_interaction(a_obs, sample_meta))

    # VIF diagnostic (only when r_tensor is available, i.e. Phase 1+)
    if r_tensor is not None:
        report.results.append(
            check_rna_covariate_vif(r_tensor, a_obs, sample_meta))

    return report


# ---------------------------------------------------------------------------
# Full data loading pipeline
# ---------------------------------------------------------------------------

def load_all(include_rna: bool = False) -> Tuple[SAPData, DiagnosticReport]:
    """Load all inputs, pool, filter, and run diagnostics.

    Args:
        include_rna: If True, also run Phase 1 RNA preprocessing (§2.2-§3.2.1).

    Returns the validated SAPData container and a DiagnosticReport.
    """
    print("Loading A_obs fractions...")
    a_obs = load_a_obs()
    sample_meta = build_sample_meta(list(a_obs.index))
    print(f"  {a_obs.shape[0]} samples × {a_obs.shape[1]} cell types")

    print("Loading DESP baselines (X^base)...")
    x_base, site_meta_desp = load_x_base()
    print(f"  {x_base.shape[0]} sites × {x_base.shape[1]} cell types")

    print("Loading bulk phospho (with MoR normalization)...")
    bulk, site_meta_bulk, sample_order = load_bulk_phospho()
    print(f"  {bulk.shape[0]} sites × {bulk.shape[1]} samples")

    # Align bulk samples to A_obs sample order
    aobs_samples = set(a_obs.index)
    bulk_samples = set(bulk.columns)
    common = aobs_samples & bulk_samples
    if len(common) < 24:
        missing_from_bulk = aobs_samples - bulk_samples
        missing_from_aobs = bulk_samples - aobs_samples
        print(f"  Warning: only {len(common)} samples overlap")
        if missing_from_bulk:
            print(f"    Missing from bulk: {missing_from_bulk}")
        if missing_from_aobs:
            print(f"    Missing from A_obs: {missing_from_aobs}")
    bulk = bulk[list(a_obs.index)]  # reorder to match A_obs

    # Align DESP and bulk by site_id
    desp_sites = set(site_meta_desp["site_id"])
    bulk_sites = set(site_meta_bulk["site_id"])
    common_sites = sorted(desp_sites & bulk_sites)
    print(f"  DESP sites: {len(desp_sites)}, bulk sites: {len(bulk_sites)}, "
          f"common: {len(common_sites)}")

    # Index both frames by site_id for alignment
    site_meta_desp = site_meta_desp.set_index("site_id")
    x_base = x_base.set_index(site_meta_desp.index)
    site_meta_bulk = site_meta_bulk.set_index("site_id")
    bulk = bulk.set_index(site_meta_bulk.index)

    # Restrict to common sites
    x_base = x_base.loc[common_sites]
    bulk = bulk.loc[common_sites]
    site_meta_aligned = site_meta_bulk.loc[common_sites]

    # Feature filtering (SAP §2.5)
    n_raw = len(bulk)
    keep_mask = (bulk > 0).sum(axis=1) >= config.MIN_SAMPLE_DETECTION
    n_filtered = int(keep_mask.sum())
    print(f"Feature filtering: {n_raw} → {n_filtered} sites "
          f"(dropped {n_raw - n_filtered} detected in < {config.MIN_SAMPLE_DETECTION} samples)")

    bulk_filtered = bulk.loc[keep_mask].reset_index(drop=True)
    x_base_filtered = x_base.loc[keep_mask].reset_index(drop=True)
    site_meta_filtered = site_meta_aligned.loc[keep_mask].reset_index()

    data = SAPData(
        a_obs=a_obs,
        x_base=x_base_filtered,
        bulk_phospho=bulk_filtered,
        sample_meta=sample_meta,
        site_meta=site_meta_filtered,
        n_sites_raw=n_raw,
        n_sites_filtered=n_filtered,
    )

    if include_rna:
        print("\n--- Phase 1: RNA Preprocessing ---")
        print("Loading and pooling aggexp...")
        aggexp = load_aggexp_pooled()

        print("Resolving aggexp sample mapping...")
        sample_map = build_aggexp_sample_map(aggexp)
        data.sample_id_map = sample_map

        print("Preprocessing pseudobulk RNA (§2.2)...")
        gkp, kinase_genes, phosphatase_genes = preprocess_aggexp(aggexp, sample_map)
        data.gkp = gkp
        data.kinase_genes = kinase_genes
        data.phosphatase_genes = phosphatase_genes

        print("Computing condition-variation scores (§2.3)...")
        data.cvs = compute_cvs(gkp, sample_meta)

        print("Building kinase-substrate map and r_tensor (§3.2.1)...")
        kinase_map = build_kinase_substrate_map(site_meta_filtered, kinase_genes)
        data.r_tensor = compute_r_tensor(
            gkp, sample_meta, kinase_map, phosphatase_genes, n_filtered,
        )

        print("Computing intensity strata (§4.2)...")
        data.intensity_strata = compute_intensity_strata(x_base_filtered)

    # Run diagnostics (after r_tensor is available so VIF is included)
    print("\nRunning pre-fit diagnostics...")
    report = run_diagnostics(a_obs, sample_meta, r_tensor=data.r_tensor)

    return data, report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def print_data_summary(data: SAPData) -> None:
    """Print a concise summary of loaded data."""
    print("\n" + "=" * 50)
    print("SAP Data Readiness Summary")
    print("=" * 50)

    print(f"\nSamples:        {data.a_obs.shape[0]}")
    print(f"Cell types:     {data.a_obs.shape[1]} ({', '.join(config.SAP_CELLTYPES)})")
    print(f"Sites (raw):    {data.n_sites_raw}")
    print(f"Sites (filtered): {data.n_sites_filtered}")

    print("\nA_obs composition (mean fractions):")
    means = data.a_obs.mean()
    for ct in config.SAP_CELLTYPES:
        print(f"  {ct:25s} {means[ct]:.3f}")

    print(f"\nX^base range: [{data.x_base.values.min():.2f}, {data.x_base.values.max():.2f}]")
    print(f"X^base zeros: {(data.x_base.values == 0).sum()} / {data.x_base.values.size} "
          f"({100 * (data.x_base.values == 0).sum() / data.x_base.values.size:.1f}%)")

    bulk_vals = data.bulk_phospho.values
    bulk_nans = int(np.isnan(bulk_vals).sum())
    print(f"\nBulk phospho range: [{np.nanmin(bulk_vals):.2f}, {np.nanmax(bulk_vals):.2f}]")
    print(f"Bulk phospho NaNs: {bulk_nans} / {bulk_vals.size} "
          f"({100 * bulk_nans / bulk_vals.size:.1f}%)")

    print(f"\nFactorial design: {len(config.SAP_CONDITIONS)} conditions × "
          f"{len(config.SAP_TIMEPOINTS)} timepoints × {len(config.SAP_GENDERS)} genders "
          f"= {len(config.SAP_CONDITIONS) * len(config.SAP_TIMEPOINTS) * len(config.SAP_GENDERS)} groups")


def print_rna_summary(data: SAPData) -> None:
    """Print Phase 1 RNA preprocessing summary."""
    if data.gkp is None:
        return

    print("\n" + "=" * 50)
    print("Phase 1: RNA Preprocessing Summary")
    print("=" * 50)

    print(f"\nG^KP matrix: {data.gkp.shape}")
    ct_count = len(data.gkp.index.get_level_values("cell_type").unique())
    sample_count = len(data.gkp.index.get_level_values("sample_id").unique())
    print(f"  {ct_count} cell types × {sample_count} samples × {data.gkp.shape[1]} KP genes")
    print(f"  Kinase genes: {len(data.kinase_genes)}")
    print(f"  Phosphatase genes: {len(data.phosphatase_genes)}")

    # Verify centering/scaling
    for ct in config.SAP_ESTIMATED_CELLTYPES:
        ct_data = data.gkp.loc[ct]
        print(f"  {ct[:20]:20s}  mean={ct_data.values.mean():.4f}  std={ct_data.values.std():.4f}")

    print(f"\nCVS (condition-variation scores):")
    print(data.cvs.to_string(float_format="{:.4f}".format))

    if data.r_tensor is not None:
        nonzero = np.count_nonzero(data.r_tensor)
        total = data.r_tensor.size
        print(f"\nr_tensor shape: {data.r_tensor.shape}")
        print(f"  Nonzero entries: {nonzero}/{total} ({100 * nonzero / total:.1f}%)")
        print(f"  Range: [{data.r_tensor.min():.4f}, {data.r_tensor.max():.4f}]")

    if data.intensity_strata is not None:
        print(f"\nIntensity strata (§4.2): {config.N_INTENSITY_STRATA} quartile-based bins")
        for q in range(config.N_INTENSITY_STRATA):
            n_q = int(np.sum(data.intensity_strata == q))
            print(f"  Stratum {q}: {n_q} sites")


def main() -> None:
    parser = argparse.ArgumentParser(description="SAP data preparation and diagnostics")
    parser.add_argument("--check-only", action="store_true",
                        help="Run composition diagnostics only (skip heavy loads)")
    parser.add_argument("--phase1", action="store_true",
                        help="Include Phase 1 RNA preprocessing (§2.2-§3.2.1)")
    args = parser.parse_args()

    if args.check_only:
        print("Loading A_obs for diagnostics...")
        a_obs = load_a_obs()
        sample_meta = build_sample_meta(list(a_obs.index))
        report = run_diagnostics(a_obs, sample_meta)
        print("\n" + report.summary())
        sys.exit(0 if report.all_passed else 1)

    data, report = load_all(include_rna=args.phase1)
    print_data_summary(data)
    print("\n" + report.summary())

    if args.phase1:
        print_rna_summary(data)
    else:
        # Phase 0 only: validate aggexp pooling coverage
        print("\nValidating aggexp pooling coverage...")
        aggexp = load_aggexp_pooled()
        cts_in_aggexp = sorted(aggexp.index.get_level_values("cell_type").unique())
        print(f"  Resolved cell types in aggexp: {cts_in_aggexp}")
        samples_per_ct = aggexp.groupby("cell_type").size()
        print(f"  Samples per cell type:\n{samples_per_ct.to_string()}")
        print(f"  Genes: {aggexp.shape[1]}")

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
