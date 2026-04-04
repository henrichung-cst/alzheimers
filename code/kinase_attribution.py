#!/usr/bin/env python3
"""Kinase attribution: stoichiometry-corrected MEA enrichment and unified cell-type attribution.

Stage 1: Cross-plex normalization + stoichiometry computation (N=72)
Stage 2: OLS site-level models + MEA (GSEA-based) kinase enrichment on stoichiometry
Stage 3: Unified cell-type attribution combining SEA-AD concordance + WMB expression

Optional: --mechanism-annotation for supplementary abundance/activity classification.

Inputs:
  data/incytr_collections/song/primary/proteomics/
    song2024_tmttotal_protein_quant_merged_labeled (2).xlsx
    song_IMAC_sitequant_merged_labeled (2).xlsx
  outputs/reports/data_ingest/sample_mapping.csv
  outputs/reports/wmb_expression/wmb_kinase_expression.csv
  data/external/sea_ad/effect_sizes.h5ad
  code/config.py

Outputs (all under outputs/reports/kinase_attribution/):
  stoichiometry_matrix.csv, raw_phospho_normalized.csv
  mea_stoichiometry.csv, site_level_ols.csv
  unified_attribution.csv, attribution_summary.json
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.KINASE_ATTRIBUTION_OUTPUT_DIR
DATA_INGEST_DIR = config.DATA_INGEST_OUTPUT_DIR

TOTAL_PROTEOME_FILE = os.path.join(
    config.SONG_PRIMARY_PROTEOMICS_DIR,
    "song2024_tmttotal_protein_quant_merged_labeled (2).xlsx",
)
IMAC_SITEQUANT_FILE = os.path.join(
    config.SONG_PRIMARY_PROTEOMICS_DIR,
    "song_IMAC_sitequant_merged_labeled (2).xlsx",
)

REF_CHANNEL = "126"  # Ref_Pool channel in each plex

# Factorial genotype coding for OLS
GENOTYPE_CODING = {
    "WT":      {"App": 0, "Tau": 0, "Int": 0},
    "APP":     {"App": 1, "Tau": 0, "Int": 0},
    "T22":     {"App": 0, "Tau": 1, "Int": 0},
    "T22/APP": {"App": 1, "Tau": 1, "Int": 1},
}

# Contrast definitions: how to derive effective LFC from OLS coefficients
# App: APP vs WT main effect
# Tau: T22 vs WT main effect
# ApTt: T22/APP vs WT (App + Tau + Int)
CONTRAST_COEFS = {
    "App":  {"App": 1, "Tau": 0, "Int": 0},
    "Tau":  {"App": 0, "Tau": 1, "Int": 0},
    "ApTt": {"App": 1, "Tau": 1, "Int": 1},
}

# Genes for stoichiometry QC spot-checks
QC_GENES = ["Mapt", "Gsk3b", "Akt1", "Mapk1", "Camk2a"]

WMB_EXPRESSION_FILE = config.WMB_EXPRESSION_FILE

from atlas_reference import SUBCLASS_TO_5PLUS1 as _SUBCLASS_TO_5PLUS1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "pca_plots"), exist_ok=True)


def load_sample_mapping():
    """Load sample mapping from data ingestion stage."""
    path = os.path.join(DATA_INGEST_DIR, "sample_mapping.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sample mapping not found at {path}. Run data_ingest.py --mapping first."
        )
    return pd.read_csv(path)


def _proteome_ref_col(plex):
    return f"plex{plex}_{REF_CHANNEL}_sn_mean"


def _phospho_ref_col(plex):
    return f"p{plex}_{REF_CHANNEL}_sn_sum"


def _proteome_to_phospho_col(col):
    """Convert plex1_128n_sn_mean -> p1_128n_sn_sum."""
    parts = col.split("_", 1)  # ['plex1', '128n_sn_mean']
    plex_num = parts[0].replace("plex", "")
    rest = parts[1].rsplit("_sn_mean", 1)[0]
    return f"p{plex_num}_{rest}_sn_sum"


def _irs_normalize(quant_df, ref_cols, sample_to_plex):
    """Internal Reference Scaling normalization.

    Parameters
    ----------
    quant_df : DataFrame
        Proteins (rows) x samples+refs (columns).
    ref_cols : dict[int, str]
        Plex number -> reference column name.
    sample_to_plex : dict[str, int]
        Sample column name -> plex number.

    Returns
    -------
    DataFrame with IRS-normalized values for sample columns.
    """
    # Global mean reference per protein (across plexes)
    ref_mat = pd.DataFrame(
        {p: quant_df[col] for p, col in ref_cols.items() if col in quant_df.columns}
    )
    # Use nanmean so proteins missing from one plex ref still get normalized
    global_ref = ref_mat.mean(axis=1, skipna=True)

    normalized = quant_df.copy()
    for col, plex in sample_to_plex.items():
        ref_col = ref_cols[plex]
        if ref_col not in quant_df.columns:
            continue
        ref_vals = quant_df[ref_col].values
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = quant_df[col].values / ref_vals
            normalized[col] = ratio * global_ref.values
    return normalized


def _median_center_normalize(quant_df, sample_to_plex):
    """Fallback: per-plex median centering."""
    global_median = np.nanmedian(quant_df.values)
    normalized = quant_df.copy()
    for plex in set(sample_to_plex.values()):
        plex_cols = [c for c, p in sample_to_plex.items() if p == plex]
        plex_vals = quant_df[plex_cols].values
        plex_med = np.nanmedian(plex_vals)
        if plex_med > 0:
            normalized[plex_cols] = plex_vals * (global_median / plex_med)
    return normalized


def _run_pca_and_plot(quant_df, mapping, title_prefix, out_prefix):
    """PCA on log2-transformed data, 4 factor-colored plots."""
    mat = quant_df.T.copy()  # samples x proteins
    mat = mat.replace(0, np.nan)
    with np.errstate(divide="ignore"):
        mat = np.log2(mat)
    # Drop proteins with any NaN (for clean PCA)
    mat = mat.dropna(axis=1)
    if mat.shape[1] < 10:
        print(f"  WARNING: only {mat.shape[1]} complete proteins for PCA")
        return None

    pca = PCA(n_components=min(10, mat.shape[0], mat.shape[1]))
    coords = pca.fit_transform(mat.values)
    var_exp = pca.explained_variance_ratio_ * 100

    # Build sample metadata aligned to quant columns
    col_order = quant_df.columns.tolist()
    meta = mapping.set_index("column_name").loc[col_order].reset_index()

    for factor, col_name in [("plex", "plex"), ("genotype", "genotype"),
                              ("sex", "sex"), ("timepoint", "timepoint")]:
        fig, ax = plt.subplots(figsize=(8, 6))
        groups = meta[col_name].unique()
        cmap = plt.cm.get_cmap("tab10", len(groups))
        for i, g in enumerate(sorted(groups)):
            mask = meta[col_name] == g
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)],
                       label=str(g), s=40, alpha=0.7, edgecolors="k", linewidths=0.3)
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
        ax.set_title(f"{title_prefix} — colored by {factor}")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "pca_plots",
                                 f"{out_prefix}_by_{factor}.png"), dpi=150)
        plt.close(fig)

    return {"pc1_var": round(var_exp[0], 2), "pc2_var": round(var_exp[1], 2),
            "n_proteins": mat.shape[1]}


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction, NaN-safe."""
    valid = ~np.isnan(pvals)
    result = np.full_like(pvals, np.nan)
    if valid.sum() == 0:
        return result
    _, adj, _, _ = multipletests(pvals[valid], method="fdr_bh")
    result[valid] = adj
    return result


# ===========================================================================
# Stage 1: Cross-plex normalization + stoichiometry
# ===========================================================================

def step_normalize():
    """Stage 1: IRS normalization and stoichiometry computation."""
    _ensure_output_dir()
    print("\n=== Stage 1: Cross-Plex Normalization + Stoichiometry ===\n")

    # --- 1.0 Load sample mapping ---
    mapping = load_sample_mapping()
    sample_to_plex = dict(zip(mapping["column_name"], mapping["plex"]))
    bio_cols = mapping["column_name"].tolist()
    print(f"  {len(bio_cols)} biological samples across "
          f"{mapping['plex'].nunique()} plexes")

    # --- 1.1 Load and IRS-normalize total proteome ---
    print("\n--- 1.1 Total proteome normalization ---")
    tp = pd.read_excel(TOTAL_PROTEOME_FILE, header=1)
    tp_gene = tp["Gene Symbol"].copy()

    # Build ref column dict and check availability
    ref_cols_tp = {}
    for plex in sorted(mapping["plex"].unique()):
        rc = _proteome_ref_col(plex)
        if rc in tp.columns:
            ref_cols_tp[plex] = rc
        else:
            print(f"  WARNING: reference column {rc} not found in total proteome")

    # Extract all needed columns (bio + refs)
    all_tp_cols = bio_cols + list(ref_cols_tp.values())
    tp_quant_raw = tp[[c for c in all_tp_cols if c in tp.columns]].copy()
    tp_quant_raw = tp_quant_raw.apply(pd.to_numeric, errors="coerce")

    # PCA before normalization
    print("  PCA before normalization...")
    tp_bio_raw = tp_quant_raw[bio_cols]
    pca_before = _run_pca_and_plot(tp_bio_raw, mapping,
                                    "Total Proteome (raw)", "tp_raw")

    # IRS normalize
    if len(ref_cols_tp) >= 4:
        print(f"  Applying IRS normalization using {len(ref_cols_tp)} "
              f"reference channels...")
        tp_quant_norm = _irs_normalize(tp_quant_raw, ref_cols_tp, sample_to_plex)
        norm_method = "IRS"
    else:
        print("  Fewer than 4 reference channels found, falling back to "
              "median centering...")
        tp_quant_norm = _median_center_normalize(tp_quant_raw, sample_to_plex)
        norm_method = "median_centering"

    tp_norm = tp_quant_norm[bio_cols]

    # PCA after normalization
    print("  PCA after normalization...")
    pca_after = _run_pca_and_plot(tp_norm, mapping,
                                   f"Total Proteome ({norm_method})", "tp_norm")

    # Plex median comparison
    plex_medians_before = {}
    plex_medians_after = {}
    for plex in sorted(mapping["plex"].unique()):
        plex_cols = mapping.loc[mapping["plex"] == plex, "column_name"].tolist()
        plex_medians_before[str(plex)] = round(
            float(np.nanmedian(tp_bio_raw[plex_cols].values)), 2)
        plex_medians_after[str(plex)] = round(
            float(np.nanmedian(tp_norm[plex_cols].values)), 2)

    print(f"  Plex medians before: {plex_medians_before}")
    print(f"  Plex medians after:  {plex_medians_after}")

    # --- 1.2 Load and IRS-normalize phospho sitequant ---
    print("\n--- 1.2 Phospho sitequant normalization ---")
    sq = pd.read_excel(IMAC_SITEQUANT_FILE, header=1)
    print(f"  Loaded {len(sq)} phosphosites")

    # Map bio columns from proteome to phospho naming
    phospho_bio_cols = [_proteome_to_phospho_col(c) for c in bio_cols]
    missing_pcols = [c for c in phospho_bio_cols if c not in sq.columns]
    if missing_pcols:
        print(f"  WARNING: {len(missing_pcols)} phospho columns not found: "
              f"{missing_pcols[:3]}")
    phospho_bio_cols = [c for c in phospho_bio_cols if c in sq.columns]

    # Build phospho ref columns
    ref_cols_ph = {}
    for plex in sorted(mapping["plex"].unique()):
        rc = _phospho_ref_col(plex)
        if rc in sq.columns:
            ref_cols_ph[plex] = rc

    # Build phospho sample-to-plex map
    phospho_s2p = {}
    for tp_col, plex in sample_to_plex.items():
        ph_col = _proteome_to_phospho_col(tp_col)
        if ph_col in sq.columns:
            phospho_s2p[ph_col] = plex

    all_ph_cols = phospho_bio_cols + [c for c in ref_cols_ph.values()
                                       if c in sq.columns]
    sq_quant_raw = sq[[c for c in all_ph_cols if c in sq.columns]].copy()
    sq_quant_raw = sq_quant_raw.apply(pd.to_numeric, errors="coerce")

    if len(ref_cols_ph) >= 4:
        print(f"  Applying IRS normalization using {len(ref_cols_ph)} "
              f"reference channels...")
        sq_quant_norm = _irs_normalize(sq_quant_raw, ref_cols_ph, phospho_s2p)
    else:
        print("  Falling back to median centering...")
        sq_quant_norm = _median_center_normalize(sq_quant_raw, phospho_s2p)

    sq_norm = sq_quant_norm[phospho_bio_cols]

    # --- 1.3 Compute stoichiometry ---
    print("\n--- 1.3 Computing stoichiometry ---")

    # Match phosphosites to total proteome proteins by gene symbol
    sq_genes = sq["gene_symbol"].fillna("").astype(str).str.upper()
    tp_genes = tp_gene.fillna("").astype(str).str.upper()

    # Build protein-level intensity lookup: gene_symbol_upper -> row indices in tp
    gene_to_tp_idx = {}
    for idx, g in enumerate(tp_genes):
        if g and g != "0":
            gene_to_tp_idx.setdefault(g, []).append(idx)

    # For each phosphosite, find the matching protein row (use first match)
    n_sites = len(sq)
    n_matched = 0
    stoich_data = {}  # col -> array of stoich values

    # Build column-name mapping: phospho col -> proteome col
    ph_to_tp_col = {}
    for tp_col in bio_cols:
        ph_col = _proteome_to_phospho_col(tp_col)
        if ph_col in phospho_bio_cols:
            ph_to_tp_col[ph_col] = tp_col

    # Pre-extract normalized matrices as numpy for speed
    tp_norm_vals = tp_norm.values  # (n_proteins, 72)
    sq_norm_vals = sq_norm.values  # (n_sites, len(phospho_bio_cols))

    # Map phospho column index to proteome column index
    ph_col_to_tp_col_idx = {}
    for j, ph_col in enumerate(phospho_bio_cols):
        tp_col = ph_to_tp_col.get(ph_col)
        if tp_col and tp_col in bio_cols:
            ph_col_to_tp_col_idx[j] = bio_cols.index(tp_col)

    # Compute stoichiometry matrix
    stoich_matrix = np.full((n_sites, len(bio_cols)), np.nan)
    site_matched = np.zeros(n_sites, dtype=bool)
    site_protein_gene = [""] * n_sites

    for i in range(n_sites):
        gene_upper = sq_genes.iloc[i]
        if gene_upper not in gene_to_tp_idx:
            continue
        tp_row = gene_to_tp_idx[gene_upper][0]  # first matching protein
        site_matched[i] = True
        site_protein_gene[i] = gene_upper
        n_matched += 1

        for ph_j, tp_j in ph_col_to_tp_col_idx.items():
            ph_val = sq_norm_vals[i, ph_j]
            tp_val = tp_norm_vals[tp_row, tp_j]
            if ph_val > 0 and tp_val > 0 and np.isfinite(ph_val) and np.isfinite(tp_val):
                stoich_matrix[i, tp_j] = np.log2(ph_val) - np.log2(tp_val)

    pct_matched = n_matched / n_sites * 100
    n_total_values = stoich_matrix.size
    n_valid = np.sum(np.isfinite(stoich_matrix))
    n_valid_matched = np.sum(np.isfinite(stoich_matrix[site_matched]))
    print(f"  {n_matched}/{n_sites} sites matched to proteins ({pct_matched:.1f}%)")
    print(f"  Stoichiometry matrix: {n_valid}/{n_total_values} valid values "
          f"({n_valid/n_total_values*100:.1f}%)")
    if n_matched > 0:
        print(f"  Among matched sites: {n_valid_matched}/{n_matched*len(bio_cols)} "
              f"valid ({n_valid_matched/(n_matched*len(bio_cols))*100:.1f}%)")

    # Build output DataFrame
    stoich_df = pd.DataFrame(stoich_matrix, columns=bio_cols)
    stoich_df.insert(0, "site_id", sq["site_id"].values)
    stoich_df.insert(1, "gene_symbol", sq["gene_symbol"].values)
    stoich_df.insert(2, "motif", sq["motif"].values)
    stoich_df.insert(3, "matched_protein", site_matched)

    # Also save raw (normalized) phospho intensities for Stage 2 comparison
    raw_phospho_df = pd.DataFrame(sq_norm_vals, columns=phospho_bio_cols)
    # Rename phospho columns to match proteome column names for consistency
    rename_map = {ph: ph_to_tp_col[ph] for ph in phospho_bio_cols
                  if ph in ph_to_tp_col}
    raw_phospho_df = raw_phospho_df.rename(columns=rename_map)
    raw_phospho_df.insert(0, "site_id", sq["site_id"].values)
    raw_phospho_df.insert(1, "gene_symbol", sq["gene_symbol"].values)
    raw_phospho_df.insert(2, "motif", sq["motif"].values)

    # --- 1.4 Quality check ---
    print("\n--- 1.4 Stoichiometry QC spot-checks ---")
    qc_rows = []
    for gene in QC_GENES:
        gene_upper = gene.upper()
        mask = sq_genes == gene_upper
        n_sites_gene = mask.sum()
        if n_sites_gene == 0:
            print(f"  {gene}: not found in phospho data")
            qc_rows.append({"gene": gene, "n_sites": 0, "found": False})
            continue
        # Use first matching site for spot-check
        site_idx = np.where(mask)[0][0]
        site_id = sq["site_id"].iloc[site_idx]
        site_pos = sq.get("site_position", sq.get("site_pos", pd.Series()))
        pos_str = site_pos.iloc[site_idx] if len(site_pos) > site_idx else "?"

        # Compute per-genotype means for raw phospho and stoichiometry
        for geno in ["WT", "APP", "T22", "T22/APP"]:
            geno_mask = mapping["genotype"] == geno
            geno_cols = mapping.loc[geno_mask, "column_name"].tolist()
            geno_idx = [bio_cols.index(c) for c in geno_cols if c in bio_cols]
            raw_vals = sq_norm_vals[site_idx, [
                phospho_bio_cols.index(_proteome_to_phospho_col(bio_cols[j]))
                for j in geno_idx
                if _proteome_to_phospho_col(bio_cols[j]) in phospho_bio_cols
            ]]
            stoich_vals = stoich_matrix[site_idx, geno_idx]
            qc_rows.append({
                "gene": gene, "site_id": int(site_id),
                "site_position": str(pos_str),
                "genotype": geno,
                "raw_phospho_mean": float(np.nanmean(raw_vals)) if len(raw_vals) else np.nan,
                "stoichiometry_mean": float(np.nanmean(stoich_vals)),
                "n_valid_stoich": int(np.sum(np.isfinite(stoich_vals))),
                "n_sites_for_gene": int(n_sites_gene),
            })
        print(f"  {gene} ({pos_str}): {n_sites_gene} sites, "
              f"matched={site_matched[site_idx]}")

    qc_df = pd.DataFrame(qc_rows)

    # --- 1.5 Save outputs ---
    print("\n--- 1.5 Saving outputs ---")
    stoich_path = os.path.join(OUTPUT_DIR, "stoichiometry_matrix.csv")
    stoich_df.to_csv(stoich_path, index=False)
    print(f"  Saved {stoich_path} ({stoich_df.shape})")

    raw_path = os.path.join(OUTPUT_DIR, "raw_phospho_normalized.csv")
    raw_phospho_df.to_csv(raw_path, index=False)
    print(f"  Saved {raw_path}")

    qc_path = os.path.join(OUTPUT_DIR, "stoichiometry_qc.csv")
    qc_df.to_csv(qc_path, index=False)
    print(f"  Saved {qc_path}")

    norm_summary = {
        "normalization_method": norm_method,
        "n_sites_total": int(n_sites),
        "n_sites_matched": int(n_matched),
        "pct_matched": round(pct_matched, 1),
        "n_valid_stoich_values": int(n_valid),
        "pct_valid_stoich": round(n_valid / n_total_values * 100, 1),
        "plex_medians_before": plex_medians_before,
        "plex_medians_after": plex_medians_after,
        "pca_before": pca_before,
        "pca_after": pca_after,
    }
    norm_path = os.path.join(OUTPUT_DIR, "normalization_summary.json")
    with open(norm_path, "w") as f:
        json.dump(norm_summary, f, indent=2)
    print(f"  Saved {norm_path}")

    print("\n  Stage 1 complete.")
    return stoich_df, raw_phospho_df


# ===========================================================================
# Stage 2: OLS site-level models + kinase enrichment
# ===========================================================================

def _build_design_matrix(mapping, bio_cols):
    """Build the factorial OLS design matrix (72 x 7).

    Columns: const, App, Tau, Int, female, time_4mo, time_6mo
    Row order matches bio_cols.
    """
    # Align mapping to bio_cols order
    meta = mapping.set_index("column_name").loc[bio_cols].reset_index()

    X = pd.DataFrame(index=range(len(bio_cols)))
    X["const"] = 1.0
    for factor, val in [("App", None), ("Tau", None), ("Int", None)]:
        X[factor] = meta["genotype"].map(
            lambda g, f=factor: GENOTYPE_CODING[g][f]).astype(float)
    X["female"] = (meta["sex"] == "F").astype(float)
    X["time_4mo"] = (meta["timepoint"] == "4mo").astype(float)
    X["time_6mo"] = (meta["timepoint"] == "6mo").astype(float)

    return X


def _run_ols_all_sites(Y, X):
    """Vectorized OLS for all sites.

    Parameters
    ----------
    Y : ndarray (n_sites, n_samples)
        Response matrix (may contain NaN).
    X : ndarray (n_samples, n_params)
        Design matrix.

    Returns
    -------
    betas : ndarray (n_sites, n_params)
    pvals : ndarray (n_sites, n_params)
    n_obs : ndarray (n_sites,)
    """
    n_sites, n_samples = Y.shape
    n_params = X.shape[1]
    betas = np.full((n_sites, n_params), np.nan)
    pvals = np.full((n_sites, n_params), np.nan)
    n_obs = np.zeros(n_sites, dtype=int)

    # Separate sites into complete-data (fast path) and partial-data
    complete_mask = np.all(np.isfinite(Y), axis=1)
    n_complete = complete_mask.sum()

    # Fast path: vectorized OLS for complete-data sites
    if n_complete > 0:
        Y_c = Y[complete_mask]
        XtX_inv = np.linalg.inv(X.T @ X)
        B_c = (XtX_inv @ X.T @ Y_c.T).T  # (n_complete, n_params)
        residuals = Y_c - (X @ B_c.T).T
        dof = n_samples - n_params
        sigma2 = np.sum(residuals ** 2, axis=1) / dof  # (n_complete,)
        # Standard errors: sqrt(diag(XtX_inv) * sigma2)
        cov_diag = np.diag(XtX_inv)  # (n_params,)
        se = np.sqrt(np.outer(sigma2, cov_diag))  # (n_complete, n_params)
        t_stats = B_c / se
        p_c = 2 * sp_stats.t.sf(np.abs(t_stats), df=dof)

        betas[complete_mask] = B_c
        pvals[complete_mask] = p_c
        n_obs[complete_mask] = n_samples

    # Slow path: per-site OLS for sites with missing data
    partial_idx = np.where(~complete_mask)[0]
    for i in partial_idx:
        valid = np.isfinite(Y[i])
        n_valid = valid.sum()
        if n_valid < n_params + 2:  # need at least params + 2 for meaningful fit
            continue
        Xi = X[valid]
        yi = Y[i, valid]
        try:
            XtX_inv_i = np.linalg.inv(Xi.T @ Xi)
        except np.linalg.LinAlgError:
            continue
        bi = XtX_inv_i @ Xi.T @ yi
        resid = yi - Xi @ bi
        dof = n_valid - n_params
        s2 = np.sum(resid ** 2) / dof
        se_i = np.sqrt(np.diag(XtX_inv_i) * s2)
        t_i = bi / se_i
        p_i = 2 * sp_stats.t.sf(np.abs(t_i), df=dof)
        betas[i] = bi
        pvals[i] = p_i
        n_obs[i] = n_valid

    print(f"  OLS: {n_complete} complete-data sites (fast), "
          f"{len(partial_idx)} partial-data sites")
    print(f"  Sites with valid fits: {np.sum(np.isfinite(betas[:, 0]))}")

    return betas, pvals, n_obs


def _run_mea(motif_series, results_by_contrast, lfc_key):
    """Run MEA (GSEA-based) kinase enrichment across contrasts."""
    from kinase_library import RankedPhosData

    enrichment_results = {}
    for contrast_name, res in results_by_contrast.items():
        enrich_df = pd.DataFrame({
            "motif": motif_series.values,
            "log2_fold_change": res[lfc_key],
        })
        enrich_df = enrich_df.dropna(subset=["log2_fold_change"])
        enrich_df = enrich_df[enrich_df["motif"].notna() &
                              (enrich_df["motif"] != "")]
        if len(enrich_df) < 100:
            print(f"  WARNING: only {len(enrich_df)} sites for "
                  f"{contrast_name}, skipping")
            continue
        rpd = RankedPhosData(
            dp_data=enrich_df,
            rank_col="log2_fold_change",
            seq_col="motif",
        )
        result = rpd.mea(
            kin_type="ser_thr",
            kl_method=config.KL_METHOD,
            kl_thresh=config.KL_THRESH,
            permutation_num=config.MEA_PERMUTATION_NUM,
            seed=config.MEA_SEED,
        )
        er = result.enrichment_results.copy()
        er.index.name = "kinase"
        er = er.reset_index()
        er["contrast"] = contrast_name
        enrichment_results[contrast_name] = er
        n_sig = (er["FDR"] < config.MEA_FDR_THRESH).sum()
        print(f"  {contrast_name}: {len(er)} kinases tested, "
              f"{n_sig} significant (FDR<{config.MEA_FDR_THRESH})")
    if enrichment_results:
        return pd.concat(enrichment_results.values(), ignore_index=True)
    return pd.DataFrame()


def step_enrich():
    """Stage 2: OLS models + kinase enrichment + classification."""
    _ensure_output_dir()
    print("\n=== Stage 2: OLS Models + Kinase Enrichment ===\n")

    # Load data
    mapping = load_sample_mapping()
    bio_cols = mapping["column_name"].tolist()

    stoich_path = os.path.join(OUTPUT_DIR, "stoichiometry_matrix.csv")
    raw_path = os.path.join(OUTPUT_DIR, "raw_phospho_normalized.csv")
    if not os.path.exists(stoich_path):
        raise FileNotFoundError(f"{stoich_path} not found. Run --normalize first.")
    stoich_df = pd.read_csv(stoich_path)
    raw_df = pd.read_csv(raw_path)

    # Build design matrix
    X = _build_design_matrix(mapping, bio_cols)
    X_np = X.values
    print(f"  Design matrix: {X_np.shape} (samples x params)")
    print(f"  Columns: {list(X.columns)}")

    # --- 2.1 OLS on stoichiometry ---
    print("\n--- 2.1 OLS on stoichiometry ---")
    Y_stoich = stoich_df[bio_cols].values
    betas_s, pvals_s, nobs_s = _run_ols_all_sites(Y_stoich, X_np)

    # --- 2.2 OLS on raw phospho (log2) ---
    print("\n--- 2.2 OLS on raw phospho (log2-transformed) ---")
    raw_vals = raw_df[bio_cols].values.copy()
    raw_vals[raw_vals <= 0] = np.nan
    with np.errstate(divide="ignore"):
        Y_raw = np.log2(raw_vals)
    betas_r, pvals_r, nobs_r = _run_ols_all_sites(Y_raw, X_np)

    # BH FDR per contrast for each model
    param_names = list(X.columns)  # const, App, Tau, Int, female, time_4mo, time_6mo

    # Compute effective LFC and p-values for each contrast
    print("\n--- 2.3 Computing contrast LFCs and running kinase enrichment ---")
    results_by_contrast = {}
    for contrast_name, coefs in CONTRAST_COEFS.items():
        # Effective beta = weighted sum of App, Tau, Int coefficients
        idx_app = param_names.index("App")
        idx_tau = param_names.index("Tau")
        idx_int = param_names.index("Int")

        # Stoichiometry
        lfc_s = (coefs["App"] * betas_s[:, idx_app] +
                 coefs["Tau"] * betas_s[:, idx_tau] +
                 coefs["Int"] * betas_s[:, idx_int])

        # For p-value of a linear combination, compute Var(c'beta) = c' (X'X)^{-1} c * sigma2
        c_vec = np.zeros(len(param_names))
        c_vec[idx_app] = coefs["App"]
        c_vec[idx_tau] = coefs["Tau"]
        c_vec[idx_int] = coefs["Int"]

        # Use complete-data XtX_inv for the fast path
        complete = np.all(np.isfinite(Y_stoich), axis=1)
        XtX_inv = np.linalg.inv(X_np.T @ X_np)
        var_c = c_vec @ XtX_inv @ c_vec  # scalar

        # sigma2 per site
        residuals_s = Y_stoich - (X_np @ betas_s.T).T
        dof_s = nobs_s - len(param_names)
        dof_s[dof_s <= 0] = 1  # avoid division by zero
        sigma2_s = np.nansum(residuals_s ** 2, axis=1) / dof_s
        se_contrast_s = np.sqrt(var_c * sigma2_s)
        t_contrast_s = lfc_s / se_contrast_s
        p_contrast_s = 2 * sp_stats.t.sf(np.abs(t_contrast_s),
                                           df=dof_s)
        fdr_s = _bh_fdr(p_contrast_s)

        # Raw phospho
        lfc_r = (coefs["App"] * betas_r[:, idx_app] +
                 coefs["Tau"] * betas_r[:, idx_tau] +
                 coefs["Int"] * betas_r[:, idx_int])
        residuals_r = Y_raw - (X_np @ betas_r.T).T
        dof_r = nobs_r - len(param_names)
        dof_r[dof_r <= 0] = 1
        sigma2_r = np.nansum(residuals_r ** 2, axis=1) / dof_r
        se_contrast_r = np.sqrt(var_c * sigma2_r)
        t_contrast_r = lfc_r / se_contrast_r
        p_contrast_r = 2 * sp_stats.t.sf(np.abs(t_contrast_r), df=dof_r)
        fdr_r = _bh_fdr(p_contrast_r)

        results_by_contrast[contrast_name] = {
            "stoich_lfc": lfc_s, "stoich_pval": p_contrast_s,
            "stoich_fdr": fdr_s,
            "raw_lfc": lfc_r, "raw_pval": p_contrast_r, "raw_fdr": fdr_r,
        }

        n_sig_s = np.sum(fdr_s < 0.05)
        n_sig_r = np.sum(fdr_r < 0.05)
        print(f"  {contrast_name}: stoich {n_sig_s} sig sites (FDR<0.05), "
              f"raw {n_sig_r} sig sites")

    print("\n--- 2.4 Running MEA kinase enrichment (stoichiometry) ---")
    mea_stoich = _run_mea(
        stoich_df["motif"], results_by_contrast, "stoich_lfc")
    mea_path = os.path.join(OUTPUT_DIR, "mea_stoichiometry.csv")
    mea_stoich.to_csv(mea_path, index=False)
    n_sig_total = (mea_stoich["FDR"] < config.MEA_FDR_THRESH).sum() if len(mea_stoich) > 0 else 0
    print(f"\n  Saved {mea_path} ({len(mea_stoich)} rows, {n_sig_total} significant)")

    # Also save the per-site OLS results for reference
    site_results = pd.DataFrame({
        "site_id": stoich_df["site_id"].values,
        "gene_symbol": stoich_df["gene_symbol"].values,
        "matched_protein": stoich_df["matched_protein"].values,
        "n_obs_stoich": nobs_s,
    })
    for cn in CONTRAST_COEFS:
        res = results_by_contrast[cn]
        site_results[f"stoich_lfc_{cn}"] = res["stoich_lfc"]
        site_results[f"stoich_pval_{cn}"] = res["stoich_pval"]
        site_results[f"stoich_fdr_{cn}"] = res["stoich_fdr"]
        site_results[f"raw_lfc_{cn}"] = res["raw_lfc"]
        site_results[f"raw_fdr_{cn}"] = res["raw_fdr"]
    site_path = os.path.join(OUTPUT_DIR, "site_level_ols.csv")
    site_results.to_csv(site_path, index=False)
    print(f"  Saved {site_path} ({len(site_results)} rows)")

    print("\n  Stage 2 complete.")


# ===========================================================================
# Stage 3: Unified cell-type attribution (SEA-AD concordance + WMB expression)
# ===========================================================================

def _assign_confidence(concordance_score, wmb_specificity, sea_ad_lfc):
    """Assign attribution confidence from combined evidence."""
    if concordance_score <= 0:
        return "none"
    if wmb_specificity >= config.SPECIFICITY_HIGH and abs(sea_ad_lfc) > config.SEA_AD_LFC_MIN:
        return "high"
    if wmb_specificity >= config.SPECIFICITY_LOW or abs(sea_ad_lfc) > config.SEA_AD_LFC_MIN:
        return "moderate"
    return "low"


def step_attribute():
    """Stage 3: Unified cell-type attribution (SEA-AD concordance + WMB expression)."""
    _ensure_output_dir()
    print("\n=== Stage 3: Unified Cell-Type Attribution ===\n")
    # 3a. Load MEA results and filter significant kinases
    mea_path = os.path.join(OUTPUT_DIR, "mea_stoichiometry.csv")
    if not os.path.exists(mea_path):
        raise FileNotFoundError(f"{mea_path} not found. Run --enrich first.")
    mea = pd.read_csv(mea_path)
    sig = mea[mea["FDR"] < config.MEA_FDR_THRESH].copy()
    print(f"  MEA results: {len(mea)} total, {len(sig)} significant "
          f"(FDR<{config.MEA_FDR_THRESH})")
    if len(sig) == 0:
        print("  No significant kinases found. Stage 3 complete.")
        return

    # 3b. Map kinases to genes
    k2g = pd.read_csv(config.MAPPING_CACHE_FILE)
    kinase_to_gene = dict(zip(k2g["kinase_abbreviation"], k2g["gene_symbol"]))
    sig["gene_symbol"] = sig["kinase"].map(
        lambda k: kinase_to_gene.get(k, k))

    # 3c. SEA-AD concordance (per kinase × cell type)
    sea_ad_rows = []
    try:
        import anndata as ad
        sea_ad_path = os.path.join(config.SEA_AD_DIR, "effect_sizes.h5ad")
        if not os.path.exists(sea_ad_path):
            raise FileNotFoundError(sea_ad_path)
        print("  Loading SEA-AD effect sizes...")
        adata = ad.read_h5ad(sea_ad_path)
        sea_ad_genes_upper = {g.upper(): g for g in adata.obs_names}
        supertypes = list(adata.var_names)

        # Build supertype → subclass mapping from SEA-AD metadata
        st_to_subclass = dict(zip(adata.var_names, adata.var["Subclass"]))

        # Pre-build gene index for O(1) lookups
        gene_to_idx = {g: i for i, g in enumerate(adata.obs_names)}

        for _, row in sig.iterrows():
            kinase = row["kinase"]
            contrast = row["contrast"]
            nes = row["NES"]
            fdr = row["FDR"]
            gene = row["gene_symbol"]
            gene_upper = gene.upper() if isinstance(gene, str) else ""

            if gene_upper not in sea_ad_genes_upper:
                continue

            sea_ad_gene = sea_ad_genes_upper[gene_upper]
            gene_idx = gene_to_idx[sea_ad_gene]

            effects = adata.X[gene_idx, :]
            if hasattr(effects, "toarray"):
                effects = effects.toarray().flatten()
            else:
                effects = np.asarray(effects).flatten()

            # Aggregate to subclass level (24 subclasses)
            sc_effects = {}
            sc_counts = {}
            for i, st in enumerate(supertypes):
                subclass = st_to_subclass[st]
                val = effects[i]
                if not np.isfinite(val):
                    continue
                sc_effects.setdefault(subclass, []).append(val)
                sc_counts[subclass] = sc_counts.get(subclass, 0) + 1

            for subclass, vals in sc_effects.items():
                median_lfc = float(np.median(vals))
                concordance = np.sign(nes) * median_lfc
                parent_ct = _SUBCLASS_TO_5PLUS1.get(subclass, "Other")
                sea_ad_rows.append({
                    "kinase": kinase,
                    "gene_symbol": gene,
                    "contrast": contrast,
                    "NES": nes,
                    "FDR": fdr,
                    "cell_type": subclass,
                    "cell_type_class": parent_ct,
                    "sea_ad_lfc": median_lfc,
                    "sea_ad_n_supertypes": sc_counts[subclass],
                    "concordance_score": concordance,
                })

        print(f"  SEA-AD concordance: {len(sea_ad_rows)} "
              f"(kinase, contrast, cell_type) rows")

    except (ImportError, FileNotFoundError) as e:
        print(f"  SEA-AD not available ({e}), skipping concordance")

    sea_ad_df = pd.DataFrame(sea_ad_rows)

    # 3d. WMB expression specificity (per kinase × subclass)
    wmb_spec = {}
    if os.path.exists(WMB_EXPRESSION_FILE):
        wmb = pd.read_csv(WMB_EXPRESSION_FILE)
        wmb_grouped = wmb.groupby(
            [wmb["gene_symbol"].str.upper(), "cell_type"]
        )["specificity_score"].max()
        wmb_spec = wmb_grouped.to_dict()
        print(f"  WMB specificity: {len(wmb_spec)} (gene, cell_type) pairs loaded")
    else:
        print(f"  WMB expression file not found at {WMB_EXPRESSION_FILE}")

    # 3e. Combine into unified attribution table
    if len(sea_ad_df) == 0:
        print("  No SEA-AD data available — building WMB-only attributions")
        attribution_rows = []
        for _, row in sig.iterrows():
            gene_upper = row["gene_symbol"].upper() if isinstance(
                row["gene_symbol"], str) else ""
            for subclass, parent_ct in _SUBCLASS_TO_5PLUS1.items():
                spec = wmb_spec.get((gene_upper, subclass), 0.0)
                attribution_rows.append({
                    "kinase": row["kinase"],
                    "gene_symbol": row["gene_symbol"],
                    "contrast": row["contrast"],
                    "NES": row["NES"],
                    "FDR": row["FDR"],
                    "cell_type": subclass,
                    "cell_type_class": parent_ct,
                    "sea_ad_lfc": np.nan,
                    "sea_ad_n_supertypes": 0,
                    "wmb_specificity": spec,
                    "concordance_score": 0.0,
                    "combined_score": 0.0,
                    "combined_confidence": "none",
                })
        unified = pd.DataFrame(attribution_rows)
    else:
        unified = sea_ad_df.copy()
        genes_upper = unified["gene_symbol"].fillna("").str.upper()
        keys = list(zip(genes_upper, unified["cell_type"]))
        unified["wmb_specificity"] = [wmb_spec.get(k, 0.0) for k in keys]
        unified["combined_score"] = (
            unified["concordance_score"] * (0.5 + unified["wmb_specificity"]))
        unified["combined_confidence"] = unified.apply(
            lambda r: _assign_confidence(
                r["concordance_score"], r["wmb_specificity"],
                r["sea_ad_lfc"]),
            axis=1)

    # Filter to attributed rows (confidence != none)
    attributed = unified[unified["combined_confidence"] != "none"].copy()
    attributed = attributed.sort_values("combined_score", ascending=False)

    out_path = os.path.join(OUTPUT_DIR, "unified_attribution.csv")
    attributed.to_csv(out_path, index=False)
    print(f"\n  Saved {out_path} ({len(attributed)} attributed rows)")

    # Also save the full table for reference
    full_path = os.path.join(OUTPUT_DIR, "unified_attribution_full.csv")
    unified.to_csv(full_path, index=False)

    # Summary
    if len(attributed) > 0:
        print(f"\n  Attribution summary:")
        n_kinase_contrast = attributed.groupby(["kinase", "contrast"]).ngroups
        n_unique_kinases = attributed["kinase"].nunique()
        print(f"    {n_kinase_contrast} kinase-contrast pairs attributed "
              f"({n_unique_kinases} unique kinases)")
        print(f"\n  By confidence:")
        for conf, cnt in attributed[
                "combined_confidence"].value_counts().items():
            print(f"    {conf}: {cnt}")
        print(f"\n  By cell type (subclass):")
        for ct, cnt in attributed[
                "cell_type"].value_counts().items():
            print(f"    {ct}: {cnt}")
        print(f"\n  By cell type class (5+1):")
        for ct, cnt in attributed[
                "cell_type_class"].value_counts().items():
            print(f"    {ct}: {cnt}")

    # Save summary JSON
    summary = {
        "n_mea_significant": int(len(sig)),
        "n_total_rows": int(len(unified)),
        "n_attributed": int(len(attributed)),
        "by_confidence": (attributed["combined_confidence"].value_counts()
                          .to_dict() if len(attributed) > 0 else {}),
        "by_cell_type": (attributed["cell_type"].value_counts()
                         .to_dict() if len(attributed) > 0 else {}),
        "by_cell_type_class": (attributed["cell_type_class"].value_counts()
                               .to_dict() if len(attributed) > 0 else {}),
        "by_contrast": (attributed["contrast"].value_counts()
                        .to_dict() if len(attributed) > 0 else {}),
    }
    summary_path = os.path.join(OUTPUT_DIR, "attribution_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  Stage 3 complete.")


# ===========================================================================
# Optional: Mechanism annotation (raw phospho MEA + classification)
# ===========================================================================

def step_mechanism_annotation():
    """Optional: Run raw phospho MEA and classify abundance/activity/both."""
    _ensure_output_dir()
    print("\n=== Mechanism Annotation (supplementary) ===\n")

    # Load data
    mapping = load_sample_mapping()
    bio_cols = mapping["column_name"].tolist()
    raw_path = os.path.join(OUTPUT_DIR, "raw_phospho_normalized.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"{raw_path} not found. Run --normalize first.")
    raw_df = pd.read_csv(raw_path)

    # Build design matrix and run OLS on raw phospho
    X = _build_design_matrix(mapping, bio_cols)
    X_np = X.values
    param_names = list(X.columns)
    raw_vals = raw_df[bio_cols].values.copy()
    raw_vals[raw_vals <= 0] = np.nan
    with np.errstate(divide="ignore"):
        Y_raw = np.log2(raw_vals)
    betas_r, pvals_r, nobs_r = _run_ols_all_sites(Y_raw, X_np)

    # Compute contrast LFCs for raw phospho
    results_by_contrast = {}
    for contrast_name, coefs in CONTRAST_COEFS.items():
        idx_app = param_names.index("App")
        idx_tau = param_names.index("Tau")
        idx_int = param_names.index("Int")
        lfc_r = (coefs["App"] * betas_r[:, idx_app] +
                 coefs["Tau"] * betas_r[:, idx_tau] +
                 coefs["Int"] * betas_r[:, idx_int])
        results_by_contrast[contrast_name] = {"raw_lfc": lfc_r}

    # Run MEA on raw phospho
    print("  Running MEA on raw phospho...")
    mea_raw = _run_mea(raw_df["motif"], results_by_contrast, "raw_lfc")
    mea_raw_path = os.path.join(OUTPUT_DIR, "mea_raw_phospho.csv")
    mea_raw.to_csv(mea_raw_path, index=False)
    print(f"  Saved {mea_raw_path} ({len(mea_raw)} rows)")

    # Load stoichiometry MEA
    mea_stoich_path = os.path.join(OUTPUT_DIR, "mea_stoichiometry.csv")
    if not os.path.exists(mea_stoich_path):
        raise FileNotFoundError(f"{mea_stoich_path} not found. Run --enrich first.")
    mea_stoich = pd.read_csv(mea_stoich_path)

    # Classify mechanism
    annotation_rows = []
    for contrast_name in CONTRAST_COEFS:
        stoich_c = mea_stoich[mea_stoich["contrast"] == contrast_name]
        raw_c = mea_raw[mea_raw["contrast"] == contrast_name]
        stoich_sig = set(stoich_c[stoich_c["FDR"] < config.MEA_FDR_THRESH]["kinase"])
        raw_sig = set(raw_c[raw_c["FDR"] < config.MEA_FDR_THRESH]["kinase"])

        all_kinases = stoich_sig | raw_sig
        for kinase in all_kinases:
            in_stoich = kinase in stoich_sig
            in_raw = kinase in raw_sig
            if in_stoich and in_raw:
                mechanism = "both"
            elif in_stoich:
                mechanism = "activity_driven"
            elif in_raw:
                mechanism = "abundance_driven"
            else:
                mechanism = "non_significant"

            s_fdr = stoich_c[stoich_c["kinase"] == kinase]["FDR"].values
            r_fdr = raw_c[raw_c["kinase"] == kinase]["FDR"].values
            annotation_rows.append({
                "kinase": kinase,
                "contrast": contrast_name,
                "stoich_FDR": float(s_fdr[0]) if len(s_fdr) > 0 else np.nan,
                "raw_FDR": float(r_fdr[0]) if len(r_fdr) > 0 else np.nan,
                "mechanism": mechanism,
            })

    annotation_df = pd.DataFrame(annotation_rows)
    ann_path = os.path.join(OUTPUT_DIR, "mechanism_annotation.csv")
    annotation_df.to_csv(ann_path, index=False)
    print(f"\n  Saved {ann_path} ({len(annotation_df)} rows)")

    if len(annotation_df) > 0:
        print("\n  Mechanism counts:")
        for mech, cnt in annotation_df["mechanism"].value_counts().items():
            print(f"    {mech}: {cnt}")

    # Merge into unified_attribution.csv if it exists
    unified_path = os.path.join(OUTPUT_DIR, "unified_attribution.csv")
    if os.path.exists(unified_path):
        unified = pd.read_csv(unified_path)
        mech_map = {}
        for _, row in annotation_df.iterrows():
            mech_map[(row["kinase"], row["contrast"])] = row["mechanism"]
        unified["mechanism_annotation"] = unified.apply(
            lambda r: mech_map.get((r["kinase"], r["contrast"]), ""), axis=1)
        unified.to_csv(unified_path, index=False)
        print(f"  Merged mechanism annotations into {unified_path}")

    print("\n  Mechanism annotation complete.")


# ===========================================================================
# Summary
# ===========================================================================

def print_summary():
    """Print cached results summary."""
    print("\n" + "=" * 72)
    print("Kinase Attribution Pipeline — Summary")
    print("=" * 72)

    # Stage 1: Normalization
    norm_path = os.path.join(OUTPUT_DIR, "normalization_summary.json")
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            ns = json.load(f)
        print(f"\n--- Stage 1: Normalization ---")
        print(f"  Method: {ns.get('normalization_method', '?')}")
        print(f"  Sites matched to proteins: {ns.get('n_sites_matched', '?')}"
              f"/{ns.get('n_sites_total', '?')} "
              f"({ns.get('pct_matched', '?')}%)")
        print(f"  Valid stoichiometry values: {ns.get('pct_valid_stoich', '?')}%")
        if ns.get("pca_after"):
            pa = ns["pca_after"]
            print(f"  PCA (after norm): PC1={pa.get('pc1_var', '?')}%, "
                  f"PC2={pa.get('pc2_var', '?')}% "
                  f"({pa.get('n_proteins', '?')} proteins)")
    else:
        print("\n--- Stage 1: Not yet run ---")

    # Stage 2: MEA Enrichment
    mea_path = os.path.join(OUTPUT_DIR, "mea_stoichiometry.csv")
    if os.path.exists(mea_path):
        mea = pd.read_csv(mea_path)
        print(f"\n--- Stage 2: OLS + MEA Kinase Enrichment ---")
        print(f"  MEA stoichiometry entries: {len(mea)}")
        n_sig = (mea["FDR"] < config.MEA_FDR_THRESH).sum()
        print(f"  Significant (FDR<{config.MEA_FDR_THRESH}): {n_sig}")
        if "contrast" in mea.columns:
            for cn in mea["contrast"].unique():
                sub = mea[mea["contrast"] == cn]
                n_s = (sub["FDR"] < config.MEA_FDR_THRESH).sum()
                print(f"    {cn}: {n_s} significant")
    else:
        print("\n--- Stage 2: Not yet run ---")

    # Stage 3: Unified Attribution
    attr_path = os.path.join(OUTPUT_DIR, "unified_attribution.csv")
    summ_path = os.path.join(OUTPUT_DIR, "attribution_summary.json")
    print(f"\n--- Stage 3: Unified Attribution ---")
    if os.path.exists(summ_path):
        with open(summ_path) as f:
            summ = json.load(f)
        print(f"  MEA significant kinases: {summ.get('n_mea_significant', '?')}")
        print(f"  Attributed rows: {summ.get('n_attributed', '?')}")
        by_conf = summ.get("by_confidence", {})
        if by_conf:
            print(f"  By confidence:")
            for conf, cnt in by_conf.items():
                print(f"    {conf}: {cnt}")
        by_ct = summ.get("by_cell_type", {})
        if by_ct:
            print(f"  By cell type (subclass):")
            for ct, cnt in sorted(by_ct.items(), key=lambda x: -x[1]):
                print(f"    {ct}: {cnt}")
        by_ctc = summ.get("by_cell_type_class", {})
        if by_ctc:
            print(f"  By cell type class (5+1):")
            for ct, cnt in sorted(by_ctc.items(), key=lambda x: -x[1]):
                print(f"    {ct}: {cnt}")
    elif os.path.exists(attr_path):
        attr = pd.read_csv(attr_path)
        print(f"  Attributed rows: {len(attr)}")
        if "combined_confidence" in attr.columns:
            for conf, cnt in attr["combined_confidence"].value_counts().items():
                print(f"    {conf}: {cnt}")
    else:
        print("  Not yet run")

    # Optional: Mechanism annotation
    mech_path = os.path.join(OUTPUT_DIR, "mechanism_annotation.csv")
    if os.path.exists(mech_path):
        mech = pd.read_csv(mech_path)
        print(f"\n--- Mechanism Annotation (supplementary) ---")
        print(f"  Total entries: {len(mech)}")
        for m, c in mech["mechanism"].value_counts().items():
            print(f"    {m}: {c}")

    print()


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kinase attribution: Stoichiometry-corrected MEA enrichment "
                    "and unified cell-type attribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--normalize", action="store_true",
                       help="Stage 1: Cross-plex normalization + stoichiometry")
    group.add_argument("--enrich", action="store_true",
                       help="Stage 2: OLS + MEA kinase enrichment on stoichiometry")
    group.add_argument("--attribute", action="store_true",
                       help="Stage 3: Unified cell-type attribution (SEA-AD + WMB)")
    group.add_argument("--mechanism-annotation", action="store_true",
                       help="Optional: raw phospho MEA + mechanism classification")
    group.add_argument("--run", action="store_true",
                       help="Run stages 1-3 in order")
    group.add_argument("--summary", action="store_true",
                       help="Print cached results summary")

    args = parser.parse_args()

    if args.normalize or args.run:
        step_normalize()
    if args.enrich or args.run:
        step_enrich()
    if args.attribute or args.run:
        step_attribute()
    if args.mechanism_annotation:
        step_mechanism_annotation()
    if args.summary:
        print_summary()

    if args.run:
        print_summary()


if __name__ == "__main__":
    main()
