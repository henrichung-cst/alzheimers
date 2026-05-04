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
# Legacy alias: existing call sites and external scripts may still reference
# IMAC_SITEQUANT_FILE. Track-aware code paths read config.PHOSPHO_TRACKS instead.
IMAC_SITEQUANT_FILE = config.SONG_IMAC_SITEQUANT_FILE

REF_CHANNEL = "126"  # Ref_Pool channel in each plex

# Factorial genotype coding for OLS
GENOTYPE_CODING = {
    "WT":      {"App": 0, "Tau": 0, "Int": 0},
    "APP":     {"App": 1, "Tau": 0, "Int": 0},
    "T22":     {"App": 0, "Tau": 1, "Int": 0},
    "T22/APP": {"App": 1, "Tau": 1, "Int": 1},
}

# Contrast definitions: how to derive effective LFC from OLS coefficients
# Time-resolved: disease effect at each timepoint (2mo is reference)
# App_Xmo:  APP vs WT main effect at timepoint X
# Tau_Xmo:  T22 vs WT main effect at timepoint X
# ApTt_Xmo: T22/APP vs WT (full combined effect) at timepoint X
# Note: Int (synergy) is assumed constant across timepoints.
CONTRAST_COEFS = {
    "App_2mo":  {"App": 1},
    "App_4mo":  {"App": 1, "App_x_time4": 1},
    "App_6mo":  {"App": 1, "App_x_time6": 1},
    "Tau_2mo":  {"Tau": 1},
    "Tau_4mo":  {"Tau": 1, "Tau_x_time4": 1},
    "Tau_6mo":  {"Tau": 1, "Tau_x_time6": 1},
    "ApTt_2mo": {"App": 1, "Tau": 1, "Int": 1},
    "ApTt_4mo": {"App": 1, "Tau": 1, "Int": 1, "App_x_time4": 1, "Tau_x_time4": 1},
    "ApTt_6mo": {"App": 1, "Tau": 1, "Int": 1, "App_x_time6": 1, "Tau_x_time6": 1},
}

# Genes for stoichiometry QC spot-checks
QC_GENES = ["Mapt", "Gsk3b", "Akt1", "Mapk1", "Camk2a"]

WMB_EXPRESSION_FILE = config.WMB_EXPRESSION_FILE



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


def load_sample_exclusions():
    """Load excluded mouse IDs from outlier analysis."""
    path = os.path.join(DATA_INGEST_DIR, "sample_exclusions.csv")
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df.loc[df["excluded"], "mouse_id"])


def _filter_samples(mapping):
    """Apply outlier exclusion + sex filter based on config.ANALYSIS_MODE.

    Returns filtered mapping DataFrame.
    """
    excluded = load_sample_exclusions()
    n0 = len(mapping)
    # Step 1: outlier exclusion
    filt = mapping[~mapping["mouse_id"].isin(excluded)].copy()
    n_excl = n0 - len(filt)
    # Step 2: sex filter
    if config.ANALYSIS_MODE == "males_only":
        filt = filt[filt["sex"] == "M"].copy()
    n_final = len(filt)
    print(f"  Sample filter ({config.ANALYSIS_MODE}): {n0} -> {n_final} "
          f"({n_excl} outliers excluded, {n0 - n_excl - n_final} sex-filtered)")
    return filt


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


def _resolve_track(track):
    """Look up a phospho-track config by name; return the dict from config."""
    if isinstance(track, dict):
        return track
    if track not in config.PHOSPHO_TRACKS:
        raise ValueError(
            f"Unknown phospho track {track!r}; "
            f"valid: {list(config.PHOSPHO_TRACKS)}"
        )
    return config.PHOSPHO_TRACKS[track]


def _load_phospho_track(track_cfg):
    """Load a phospho-track xlsx and normalize it to the canonical IMAC schema.

    Canonical schema (matches the legacy IMAC sitequant file):
      - site_id (synthesized for tracks where the source file lacks it)
      - protein_id, gene_symbol, site_position, motif
      - matched_protein columns are produced downstream in step_normalize
      - sample/ref columns named p{N}_{channel}_sn_sum (rename plex{N}_*)

    The pY workbook ships per-plex metadata columns (plex1_sequence,
    plex1_site_id, ...). Those are dropped — only the sn_sum quant columns
    are retained on the sample side.
    """
    cfg = _resolve_track(track_cfg)
    path = cfg["input_file"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Phospho-track {cfg['name']} input not found: {path}"
        )
    df = pd.read_excel(path, header=1)

    # Rename sample columns: plex{N}_* → p{N}_* (only when prefix differs).
    if cfg["column_prefix"] == "plex":
        rename_map = {}
        for c in df.columns:
            if isinstance(c, str) and c.startswith("plex") and c.endswith("_sn_sum"):
                rest = c[len("plex"):]  # e.g. "1_126c_sn_sum"
                rename_map[c] = "p" + rest
        df = df.rename(columns=rename_map)
        # Drop per-plex non-quant metadata columns (plex1_sequence, plex1_site_id, …)
        drop_cols = [c for c in df.columns
                     if isinstance(c, str)
                     and c.startswith("plex")
                     and not c.endswith("_sn_sum")]
        df = df.drop(columns=drop_cols)

    # Synthesize site_id when the source workbook lacks one.
    if cfg["site_id_source"] == "synthesize" or "site_id" not in df.columns:
        if "protein_id" in df.columns and "site_position" in df.columns:
            df["site_id"] = (
                df["protein_id"].astype(str) + "_" + df["site_position"].astype(str)
            )
        else:
            raise ValueError(
                f"Track {cfg['name']}: cannot synthesize site_id "
                f"(need protein_id and site_position columns)"
            )

    # Residue purity check on motif central residue.
    if "motif" in df.columns:
        # Motif is typically a 15-mer with the modified residue at the center.
        # Take the middle character; tolerate variable-length motifs by
        # picking the index closest to the center.
        def _central(m):
            if not isinstance(m, str) or len(m) == 0:
                return ""
            return m[len(m) // 2].upper()
        central = df["motif"].fillna("").map(_central)
        expected = set(cfg["residue"])
        non_purity = (~central.isin(expected | {""})).sum()
        purity = 1.0 - non_purity / max(len(df), 1)
        if purity < 0.99:
            print(f"  WARNING: track {cfg['name']} motif central-residue "
                  f"purity = {purity*100:.1f}% (expected {cfg['residue']})")
        else:
            print(f"  Track {cfg['name']} motif residue purity: "
                  f"{purity*100:.1f}% {cfg['residue']}")
    return df


def _track_output(filename, track_cfg):
    """Compose an output path with the track's suffix appended before the extension."""
    cfg = _resolve_track(track_cfg)
    suffix = cfg["output_suffix"]
    if not suffix:
        return os.path.join(OUTPUT_DIR, filename)
    base, ext = os.path.splitext(filename)
    return os.path.join(OUTPUT_DIR, f"{base}{suffix}{ext}")


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
    mat = quant_df.values.astype(float).copy()  # proteins x samples
    mat[mat <= 0] = np.nan
    with np.errstate(divide="ignore"):
        mat = np.log2(mat)
    n_imputed = np.sum(~np.isfinite(mat))
    mat = config.minprob_impute(mat)
    print(f"  MinProb imputed {n_imputed} missing values for PCA")
    # Remove proteins with zero variance after imputation
    var = np.var(mat, axis=1)
    mat = mat[var > 0]
    if mat.shape[0] < 10:
        print(f"  WARNING: only {mat.shape[0]} proteins with variance for PCA")
        return None

    pca = PCA(n_components=min(10, mat.shape[1], mat.shape[0]))
    coords = pca.fit_transform(mat.T)  # samples x components
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
            "n_proteins": mat.shape[0]}


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

def step_normalize(track="st"):
    """Stage 1: IRS normalization and stoichiometry computation.

    Parameters
    ----------
    track : str or dict, default "st"
        Phospho-track name from config.PHOSPHO_TRACKS ("st" for the legacy
        IMAC Ser/Thr track, "py" for tyrosine). Output filenames receive
        the track's suffix; total proteome processing is shared across
        tracks but recomputed each call.
    """
    _ensure_output_dir()
    track_cfg = _resolve_track(track)
    print(f"\n=== Stage 1: Cross-Plex Normalization + Stoichiometry "
          f"(track={track_cfg['name']}/{track_cfg['label']}) ===\n")

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
    print(f"\n--- 1.2 Phospho sitequant normalization "
          f"(track={track_cfg['name']}) ---")
    sq = _load_phospho_track(track_cfg)
    print(f"  Loaded {len(sq)} phosphosites from {track_cfg['input_file']}")

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

    # Compute stoichiometry matrix (vectorized)
    stoich_matrix = np.full((n_sites, len(bio_cols)), np.nan)
    site_matched = np.zeros(n_sites, dtype=bool)
    site_protein_gene = [""] * n_sites

    # Build per-site protein row index (-1 = unmatched)
    tp_row_for_site = np.full(n_sites, -1, dtype=int)
    for i in range(n_sites):
        gene_upper = sq_genes.iloc[i]
        if gene_upper in gene_to_tp_idx:
            tp_row_for_site[i] = gene_to_tp_idx[gene_upper][0]
            site_matched[i] = True
            site_protein_gene[i] = gene_upper
    n_matched = int(site_matched.sum())

    # Vectorized stoichiometry for matched sites
    matched_idx = np.where(site_matched)[0]
    if len(matched_idx) > 0 and ph_col_to_tp_col_idx:
        ph_js = np.array(list(ph_col_to_tp_col_idx.keys()))
        tp_js = np.array(list(ph_col_to_tp_col_idx.values()))

        # Extract phospho and proteome sub-matrices for matched sites
        ph_vals = sq_norm_vals[np.ix_(matched_idx, ph_js)]       # (n_matched, n_pairs)
        tp_rows = tp_row_for_site[matched_idx]
        tp_vals = tp_norm_vals[tp_rows][:, tp_js]                 # (n_matched, n_pairs)

        valid = (ph_vals > 0) & (tp_vals > 0) & np.isfinite(ph_vals) & np.isfinite(tp_vals)
        with np.errstate(divide="ignore", invalid="ignore"):
            stoich_vals = np.where(valid, np.log2(ph_vals) - np.log2(tp_vals), np.nan)

        # Scatter results into output matrix
        for k, tp_j in enumerate(tp_js):
            stoich_matrix[matched_idx, tp_j] = stoich_vals[:, k]

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
                "gene": gene, "site_id": str(site_id),
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
    stoich_path = _track_output("stoichiometry_matrix.csv", track_cfg)
    stoich_df.to_csv(stoich_path, index=False)
    print(f"  Saved {stoich_path} ({stoich_df.shape})")

    raw_path = _track_output("raw_phospho_normalized.csv", track_cfg)
    raw_phospho_df.to_csv(raw_path, index=False)
    print(f"  Saved {raw_path}")

    qc_path = _track_output("stoichiometry_qc.csv", track_cfg)
    qc_df.to_csv(qc_path, index=False)
    print(f"  Saved {qc_path}")

    norm_summary = {
        "track": track_cfg["name"],
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
    norm_path = _track_output("normalization_summary.json", track_cfg)
    with open(norm_path, "w") as f:
        json.dump(norm_summary, f, indent=2)
    print(f"  Saved {norm_path}")

    print("\n  Stage 1 complete.")
    return stoich_df, raw_phospho_df


# ===========================================================================
# Stage 2: OLS site-level models + kinase enrichment
# ===========================================================================

def _build_design_matrix(mapping, bio_cols):
    """Build the factorial OLS design matrix.

    males_only mode:  N x 10 (const, App, Tau, Int, time_4mo, time_6mo,
                               App_x_time4, App_x_time6, Tau_x_time4, Tau_x_time6)
    full_cohort mode: N x 11 (adds 'female' column)

    Row order matches bio_cols.
    """
    # Align mapping to bio_cols order
    meta = mapping.set_index("column_name").loc[bio_cols].reset_index()

    X = pd.DataFrame(index=range(len(bio_cols)))
    X["const"] = 1.0
    for factor, val in [("App", None), ("Tau", None), ("Int", None)]:
        X[factor] = meta["genotype"].map(
            lambda g, f=factor: GENOTYPE_CODING[g][f]).astype(float)

    # Only include sex covariate when both sexes are present
    if config.ANALYSIS_MODE != "males_only":
        X["female"] = (meta["sex"] == "F").astype(float)

    X["time_4mo"] = (meta["timepoint"] == "4mo").astype(float)
    X["time_6mo"] = (meta["timepoint"] == "6mo").astype(float)

    # Disease × timepoint interactions (time-resolved contrasts)
    X["App_x_time4"] = X["App"] * X["time_4mo"]
    X["App_x_time6"] = X["App"] * X["time_6mo"]
    X["Tau_x_time4"] = X["Tau"] * X["time_4mo"]
    X["Tau_x_time6"] = X["Tau"] * X["time_6mo"]

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


def _winsorize_lfc(lfc_array, pct=None):
    """Winsorize LFC values at the given percentile to limit outlier influence.

    Returns (clipped_array, outlier_mask, lower_bound, upper_bound).
    Sites beyond the bounds are clipped, not removed.
    """
    if pct is None:
        pct = config.MEA_WINSORIZE_PERCENTILE
    lower = np.nanpercentile(lfc_array, pct)
    upper = np.nanpercentile(lfc_array, 100 - pct)
    outlier_mask = (lfc_array < lower) | (lfc_array > upper)
    clipped = np.clip(lfc_array, lower, upper)
    return clipped, outlier_mask, lower, upper


def _run_mea(motif_series, results_by_contrast, lfc_key,
             site_ids=None, gene_symbols=None, track="st"):
    """Run MEA (GSEA-based) kinase enrichment across contrasts.

    Preprocessing before GSEA ranking (applied per contrast):

    1. **Median-centering** — Subtract the median LFC from all sites so
       the ranked list is centered at zero.  Without this, a global shift
       in stoichiometry (e.g., net increase in phosphorylation at a given
       timepoint) propagates into every kinase substrate set, making NES
       sign reflect the background shift rather than kinase-specific
       activity.  Centering forces GSEA to test whether a kinase's
       substrates are *specifically* enriched above/below the global
       trend, which is the biologically meaningful question.

    2. **Winsorization** — Clip the centered LFCs at the 1st/99th
       percentile to prevent extreme outlier sites from inflating GSEA
       enrichment scores.  Clipped sites are logged to
       ``winsorized_sites.csv``.

    The median offset removed in step 1 is recorded in
    ``mea_global_shift.csv`` for transparency.
    """
    from kinase_library import RankedPhosData, create_kin_sub_sets
    from kinase_library.utils._global_vars import kl_method_comp_direction_dict

    track_cfg = _resolve_track(track)
    enrichment_results = {}
    outlier_records = []
    shift_records = []
    substrate_records = []
    kl_comp_direction = kl_method_comp_direction_dict[config.KL_METHOD]
    for contrast_name, res in results_by_contrast.items():
        raw_lfc = res[lfc_key].copy()

        # ── Step 1: Median-center to remove global directional shift ──
        median_shift = float(np.nanmedian(raw_lfc))
        centered_lfc = raw_lfc - median_shift
        shift_records.append({
            "contrast": contrast_name,
            "median_shift": median_shift,
            "mean_before": float(np.nanmean(raw_lfc)),
            "pct_pos_before": float(np.nanmean(raw_lfc > 0) * 100),
            "pct_pos_after": float(np.nanmean(centered_lfc > 0) * 100),
        })
        if abs(median_shift) > 0.001:
            print(f"  {contrast_name}: centered by {median_shift:+.4f} "
                  f"(pos sites: {shift_records[-1]['pct_pos_before']:.0f}% "
                  f"-> {shift_records[-1]['pct_pos_after']:.0f}%)")

        # ── Step 2: Winsorize to limit outlier influence on ranking ──
        clipped_lfc, outlier_mask, lo, hi = _winsorize_lfc(centered_lfc)
        n_clipped = np.nansum(outlier_mask)
        if n_clipped > 0:
            print(f"  {contrast_name}: winsorized {int(n_clipped)} sites "
                  f"to [{lo:.3f}, {hi:.3f}]")
            # Record outliers for the diagnostic table
            idxs = np.where(outlier_mask)[0]
            for idx in idxs:
                rec = {"contrast": contrast_name,
                       "original_lfc": raw_lfc[idx],
                       "clipped_lfc": clipped_lfc[idx],
                       "lower_bound": lo, "upper_bound": hi}
                if site_ids is not None:
                    rec["site_id"] = site_ids[idx]
                if gene_symbols is not None:
                    rec["gene_symbol"] = gene_symbols[idx]
                outlier_records.append(rec)

        enrich_df = pd.DataFrame({
            "motif": motif_series.values,
            "log2_fold_change": clipped_lfc,
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
            kin_type=track_cfg["kin_type"],
            kl_method=config.KL_METHOD,
            kl_thresh=config.KL_THRESH,
            permutation_num=config.MEA_PERMUTATION_NUM,
            seed=config.MEA_SEED,
        )
        er = result.enrichment_results.copy()
        er.index.name = "kinase"
        er = er.reset_index()
        er["contrast"] = contrast_name
        er["residue_type"] = track_cfg["residue"]
        er["track"] = track_cfg["name"]
        enrichment_results[contrast_name] = er
        n_sig = (er["FDR"] < config.MEA_FDR_THRESH).sum()
        print(f"  {contrast_name}: {len(er)} kinases tested, "
              f"{n_sig} significant (FDR<{config.MEA_FDR_THRESH})")

        # Substrate sets: kinase -> motifs that score past KL_THRESH against the
        # contrast's prerank universe. This is what GSEA actually walks for each
        # kinase; persisting it lets the unified viewer show kinase-conditional
        # MEA preparation without back-applying the leading-edge result.
        kin_sub_sets = create_kin_sub_sets(
            rpd.data_kl_values, threshold=config.KL_THRESH,
            comp_direction=kl_comp_direction,
        )
        for kinase_name, motifs in kin_sub_sets.items():
            for motif in motifs:
                substrate_records.append({
                    "kinase": kinase_name,
                    "contrast": contrast_name,
                    "motif": motif,
                    "residue_type": track_cfg["residue"],
                    "track": track_cfg["name"],
                })

    # Save winsorized-site log
    if outlier_records:
        outlier_df = pd.DataFrame(outlier_records)
        outlier_path = _track_output("winsorized_sites.csv", track_cfg)
        outlier_df.to_csv(outlier_path, index=False)
        print(f"  Saved {outlier_path} ({len(outlier_df)} clipped sites)")

    # Save global-shift log for transparency
    if shift_records:
        shift_df = pd.DataFrame(shift_records)
        shift_path = _track_output("mea_global_shift.csv", track_cfg)
        shift_df.to_csv(shift_path, index=False)
        print(f"  Saved {shift_path} (median offsets removed per contrast)")

    # Save kinase -> substrate-set membership per contrast for audit traceability.
    if substrate_records:
        subs_df = pd.DataFrame(substrate_records)
        subs_path = _track_output("mea_substrate_sets.csv", track_cfg)
        subs_df.to_csv(subs_path, index=False)
        print(f"  Saved {subs_path} ({len(subs_df):,} kinase-motif rows "
              f"across {subs_df['kinase'].nunique()} kinases)")

    if enrichment_results:
        return pd.concat(enrichment_results.values(), ignore_index=True)
    return pd.DataFrame()


def _prepare_raw_ols(mapping, bio_cols, raw_df):
    """Shared OLS preparation for raw phospho data: log2-transform + run OLS."""
    X = _build_design_matrix(mapping, bio_cols)
    X_np = X.values
    param_names = list(X.columns)
    raw_vals = raw_df[bio_cols].values.copy()
    raw_vals[raw_vals <= 0] = np.nan
    with np.errstate(divide="ignore"):
        Y_raw = np.log2(raw_vals)
    betas_r, pvals_r, nobs_r = _run_ols_all_sites(Y_raw, X_np)
    return X, X_np, param_names, Y_raw, betas_r, pvals_r, nobs_r


def step_enrich(track="st"):
    """Stage 2: OLS models + kinase enrichment + classification.

    Parameters
    ----------
    track : str or dict, default "st"
        Phospho-track name from config.PHOSPHO_TRACKS. Reads track-suffixed
        Stage 1 outputs and runs MEA against the track's substrate library
        (ser_thr or tyrosine).
    """
    _ensure_output_dir()
    track_cfg = _resolve_track(track)
    print(f"\n=== Stage 2: OLS Models + Kinase Enrichment "
          f"({config.ANALYSIS_MODE}, track={track_cfg['name']}) ===\n")

    # Load data — filter samples for outlier exclusion + sex subsetting
    mapping_full = load_sample_mapping()
    mapping = _filter_samples(mapping_full)
    bio_cols = mapping["column_name"].tolist()

    stoich_path = _track_output("stoichiometry_matrix.csv", track_cfg)
    raw_path = _track_output("raw_phospho_normalized.csv", track_cfg)
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
    param_names = list(X.columns)
    # const, App, Tau, Int, female, time_4mo, time_6mo,
    # App_x_time4, App_x_time6, Tau_x_time4, Tau_x_time6

    # Compute effective LFC and p-values for each contrast
    print("\n--- 2.3 Computing contrast LFCs and running kinase enrichment ---")
    XtX_inv = np.linalg.inv(X_np.T @ X_np)
    results_by_contrast = {}
    for contrast_name, coefs in CONTRAST_COEFS.items():
        # Build contrast vector from coefficient dict
        c_vec = np.zeros(len(param_names))
        for param, weight in coefs.items():
            c_vec[param_names.index(param)] = weight

        # Stoichiometry: effective LFC = c' @ beta for each site
        lfc_s = betas_s @ c_vec

        # Var(c'beta) = c' (X'X)^{-1} c * sigma2
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
        lfc_r = betas_r @ c_vec
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

    print(f"\n--- 2.4 Running MEA kinase enrichment "
          f"(stoichiometry, kin_type={track_cfg['kin_type']}) ---")
    mea_stoich = _run_mea(
        stoich_df["motif"], results_by_contrast, "stoich_lfc",
        site_ids=stoich_df["site_id"].values,
        gene_symbols=stoich_df["gene_symbol"].values,
        track=track_cfg)
    mea_path = _track_output("mea_stoichiometry.csv", track_cfg)
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
    site_path = _track_output("site_level_ols.csv", track_cfg)
    site_results.to_csv(site_path, index=False)
    print(f"  Saved {site_path} ({len(site_results)} rows)")

    print(f"\n  Stage 2 complete (track={track_cfg['name']}).")


# ===========================================================================
# Stage 3: Unified cell-type attribution (SEA-AD concordance + WMB expression)
# ===========================================================================

def _compute_effective_concordance(nes, sea_ad_lfc, song_lfc):
    """Compute weighted concordance from Song (primary) and SEA-AD (secondary).

    Both references are evidence; Song is weighted significantly more heavily
    (same-species, same-cohort) than SEA-AD (cross-species human proxy).
    When both are available, they are combined with configurable weights
    (default 3:1). Neither has absolute veto power.

    Returns (effective_concordance, concordance_source).
    """
    w_song = config.SONG_CONCORDANCE_WEIGHT
    w_sea_ad = config.SEA_AD_CONCORDANCE_WEIGHT

    song_cs = (np.sign(nes) * song_lfc
               if np.isfinite(song_lfc) else np.nan)
    sea_ad_cs = (np.sign(nes) * sea_ad_lfc
                 if np.isfinite(sea_ad_lfc) else np.nan)

    has_song = np.isfinite(song_cs)
    has_sea_ad = np.isfinite(sea_ad_cs)

    if has_song and has_sea_ad:
        eff = (w_song * song_cs + w_sea_ad * sea_ad_cs) / (w_song + w_sea_ad)
        source = "both"
    elif has_song:
        eff = song_cs
        source = "song"
    elif has_sea_ad:
        eff = sea_ad_cs
        source = "sea_ad"
    else:
        eff = 0.0
        source = "none"

    return eff, source


def _assign_confidence_and_basis(effective_concordance, wmb_specificity,
                                 sea_ad_lfc, song_lfc, concordance_source):
    """Assign confidence tier and evidence basis.

    Weighted concordance model:
    - Song is weighted 3× vs SEA-AD 1× (configurable in config.py)
    - Song-supported concordance (source "song" or "both") can reach high
    - SEA-AD-only concordance is capped at moderate (cross-species proxy)
    - Neither reference has absolute veto power
    """
    if effective_concordance <= 0:
        return "none", "weak"

    has_wmb = wmb_specificity >= config.SPECIFICITY_LOW
    has_wmb_high = wmb_specificity >= config.SPECIFICITY_HIGH
    lfc_val = sea_ad_lfc if np.isfinite(sea_ad_lfc) else 0.0
    has_sea_ad = abs(lfc_val) > config.SEA_AD_LFC_MIN
    song_val = song_lfc if np.isfinite(song_lfc) else 0.0
    has_song = abs(song_val) > config.SONG_LFC_MIN

    # Song contributed to concordance → high eligible
    song_contributed = concordance_source in ("song", "both")

    # Confidence tier
    if song_contributed and has_wmb_high and (has_song or has_sea_ad):
        conf = "high"
    elif song_contributed and (has_wmb or has_sea_ad or has_song):
        conf = "moderate"
    elif not song_contributed:
        # SEA-AD-only concordance — capped at moderate
        conf = "moderate" if (has_wmb or has_sea_ad) else "low"
    else:
        conf = "low"

    # Evidence basis classification
    if has_wmb and has_sea_ad and has_song:
        basis = "three_way"
    elif has_wmb and has_song:
        basis = "within_cohort"
    elif has_wmb and has_sea_ad:
        basis = "cross_species"
    elif has_wmb:
        basis = "mouse_expression_only"
    elif has_song:
        basis = "song_only"
    elif has_sea_ad:
        basis = "human_concordance_only"
    else:
        basis = "weak"

    return conf, basis


def _assign_confidence_and_basis_vectorized(unified):
    """Vectorized version of _assign_confidence_and_basis for DataFrames."""
    eff = unified["effective_concordance"]
    wmb = unified["wmb_specificity"]
    sea_lfc = unified["sea_ad_lfc"].fillna(0.0)
    song_lfc_col = (unified["song_lfc"].fillna(0.0)
                    if "song_lfc" in unified.columns
                    else pd.Series(0.0, index=unified.index))
    cs_source = unified["concordance_source"]

    has_wmb = wmb >= config.SPECIFICITY_LOW
    has_wmb_high = wmb >= config.SPECIFICITY_HIGH
    has_sea_ad = sea_lfc.abs() > config.SEA_AD_LFC_MIN
    has_song = song_lfc_col.abs() > config.SONG_LFC_MIN
    song_contributed = cs_source.isin(("song", "both"))
    positive_eff = eff > 0

    # Confidence: start from "none", layer conditions (last match wins)
    conf = pd.Series("none", index=unified.index)
    mask_low = positive_eff & ~song_contributed & ~(has_wmb | has_sea_ad)
    mask_mod_sea_ad = positive_eff & ~song_contributed & (has_wmb | has_sea_ad)
    mask_mod_song = positive_eff & song_contributed & (has_wmb | has_sea_ad | has_song)
    mask_high = positive_eff & song_contributed & has_wmb_high & (has_song | has_sea_ad)
    conf[mask_low] = "low"
    conf[mask_mod_sea_ad] = "moderate"
    conf[mask_mod_song] = "moderate"
    conf[mask_high] = "high"

    # Evidence basis
    basis = pd.Series("weak", index=unified.index)
    basis[positive_eff & has_sea_ad & ~has_wmb & ~has_song] = "human_concordance_only"
    basis[positive_eff & has_song & ~has_wmb & ~has_sea_ad] = "song_only"
    basis[positive_eff & has_wmb & ~has_sea_ad & ~has_song] = "mouse_expression_only"
    basis[positive_eff & has_wmb & has_sea_ad & ~has_song] = "cross_species"
    basis[positive_eff & has_wmb & has_song & ~has_sea_ad] = "within_cohort"
    basis[positive_eff & has_wmb & has_sea_ad & has_song] = "three_way"

    return conf, basis


def step_attribute():
    """Stage 3: Unified cell-type attribution (SEA-AD concordance + WMB expression).

    Loads MEA results from every available phospho track (S/T + pY when both
    are present) and concatenates them into a single significant-kinase
    table. Each row carries `residue_type` and `track` columns so downstream
    consumers can stratify by Tyr vs Ser/Thr biology. If only the legacy
    S/T track exists, behavior is unchanged.
    """
    _ensure_output_dir()
    print("\n=== Stage 3: Unified Cell-Type Attribution ===\n")
    # 3a. Load MEA results from every track that produced an output and
    # concatenate. Stage 3 is residue-agnostic at the gene level — SEA-AD,
    # WMB and Song lookups all key on gene_symbol — so combining tracks
    # before attribution gives a single coherent table downstream.
    mea_frames = []
    for track_name, track_cfg in config.PHOSPHO_TRACKS.items():
        mea_path = _track_output("mea_stoichiometry.csv", track_cfg)
        if not os.path.exists(mea_path):
            print(f"  Track {track_name}: {mea_path} not present, skipping")
            continue
        df = pd.read_csv(mea_path)
        # Backfill residue_type / track columns for legacy outputs that
        # predate the pY refactor.
        if "residue_type" not in df.columns:
            df["residue_type"] = track_cfg["residue"]
        if "track" not in df.columns:
            df["track"] = track_cfg["name"]
        mea_frames.append(df)
        print(f"  Track {track_name}: loaded {len(df)} rows from {mea_path}")
    if not mea_frames:
        raise FileNotFoundError(
            f"No MEA outputs found under {OUTPUT_DIR}; run --enrich first."
        )
    mea = pd.concat(mea_frames, ignore_index=True, sort=False)
    # Drop the MEA-significance gate: build attribution for every
    # (kinase × contrast) so the unified table can serve as the full
    # evidence-grid for downstream views. Confidence tiers in the output
    # still reflect MEA significance + concordance + WMB support, so
    # readers that care about the gate filter on `combined_confidence`.
    n_sig = (mea["FDR"] < config.MEA_FDR_THRESH).sum()
    print(f"  MEA results (combined): {len(mea)} total, {n_sig} significant "
          f"(FDR<{config.MEA_FDR_THRESH}) — building attribution from full grid")
    sig = mea.copy()
    if "residue_type" in sig.columns:
        residue_breakdown = sig["residue_type"].value_counts().to_dict()
        print(f"  By residue_type: {residue_breakdown}")
    if len(sig) == 0:
        print("  No MEA rows found. Stage 3 complete.")
        return

    # 3b. Map kinases to genes
    k2g = pd.read_csv(config.MAPPING_CACHE_FILE)
    kinase_to_gene = dict(zip(k2g["kinase_abbreviation"], k2g["gene_symbol"]))
    sig["gene_symbol"] = sig["kinase"].map(
        lambda k: kinase_to_gene.get(k, k))

    # 3c. SEA-AD concordance (per kinase × cell type)
    # Load pathway-matched effect sizes: App→early CPS (amyloid-dominant),
    # Tau→late CPS (tau-dominant), ApTt→full CPS (combined pathology).
    sea_ad_rows = []
    supertype_rows = []  # per-(gene, stratum, supertype) audit artifact
    _supertype_emitted = set()  # (stratum, gene_idx) pairs already emitted
    try:
        import anndata as ad

        # Map each contrast to its pathway-matched stratum
        contrast_to_stratum = {}
        for contrast in sig["contrast"].unique():
            pathway = contrast.split("_")[0]
            if pathway not in config.SEA_AD_PATHWAY_MAP:
                raise ValueError(
                    f"Unknown pathway prefix '{pathway}' in contrast "
                    f"'{contrast}'. Expected one of "
                    f"{list(config.SEA_AD_PATHWAY_MAP)}")
            contrast_to_stratum[contrast] = config.SEA_AD_PATHWAY_MAP[pathway]

        # Load each required stratum
        needed_strata = set(contrast_to_stratum.values())
        adata_by_stratum = {}
        for stratum in needed_strata:
            path = config.SEA_AD_EFFECT_SIZES[stratum]
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            adata_by_stratum[stratum] = ad.read_h5ad(path)
        strata_label = ", ".join(sorted(needed_strata))
        print(f"  Loading SEA-AD effect sizes ({strata_label})...")

        # All strata share the same gene/supertype axes — use any for indexing
        ref_adata = next(iter(adata_by_stratum.values()))
        sea_ad_genes_upper = {g.upper(): g for g in ref_adata.obs_names}
        supertypes = list(ref_adata.var_names)
        st_to_subclass = dict(zip(ref_adata.var_names, ref_adata.var["Subclass"]))
        gene_to_idx = {g: i for i, g in enumerate(ref_adata.obs_names)}

        # Load SEA-AD subclass → WMB class mapping (≈ 24 → 9)
        if not os.path.exists(config.SEAAD_TO_WMB_CLASS_FILE):
            raise FileNotFoundError(
                f"SEA-AD → WMB class mapping not found at "
                f"{config.SEAAD_TO_WMB_CLASS_FILE}. Generate via Phase 1c."
            )
        seaad_to_class_df = pd.read_csv(config.SEAAD_TO_WMB_CLASS_FILE)
        seaad_to_class = dict(
            zip(seaad_to_class_df["seaad_subclass"],
                seaad_to_class_df["wmb_class_label"])
        )
        # Build supertype → WMB class (composed mapping)
        st_to_wmb_class = {
            st: seaad_to_class.get(sc) for st, sc in st_to_subclass.items()
        }
        n_st_mapped = sum(1 for v in st_to_wmb_class.values() if v is not None)
        print(f"  SEA-AD supertypes mapped to WMB class: "
              f"{n_st_mapped}/{len(supertypes)}")

        def _class_lfcs(adata, gene_idx):
            """Aggregate supertype effects to WMB class-level median LFCs.

            Uses median over all contributing supertypes (option B from the
            taxonomy migration plan — no precision loss from median-of-medians).
            Returns (class_lfcs, class_counts) keyed on WMB class label.
            """
            effects = adata.X[gene_idx, :]
            if hasattr(effects, "toarray"):
                effects = effects.toarray().flatten()
            else:
                effects = np.asarray(effects).flatten()
            cls_vals, cls_counts = {}, {}
            for i, st in enumerate(supertypes):
                wmb_class = st_to_wmb_class.get(st)
                if wmb_class is None:
                    continue
                val = effects[i]
                if np.isfinite(val):
                    cls_vals.setdefault(wmb_class, []).append(val)
                    cls_counts[wmb_class] = cls_counts.get(wmb_class, 0) + 1
            return (
                {c: float(np.median(v)) for c, v in cls_vals.items()},
                cls_counts,
            )

        # Cache (stratum, gene_idx) → (sc_lfcs, sc_counts) to avoid
        # recomputing for the same gene across multiple contrasts
        _lfc_cache = {}

        for _, row in sig.iterrows():
            kinase = row["kinase"]
            contrast = row["contrast"]
            nes = row["NES"]
            fdr = row["FDR"]
            gene = row["gene_symbol"]
            gene_upper = gene.upper() if isinstance(gene, str) else ""

            if gene_upper not in sea_ad_genes_upper:
                continue

            gene_idx = gene_to_idx[sea_ad_genes_upper[gene_upper]]
            stratum = contrast_to_stratum[contrast]

            cache_key = (stratum, gene_idx)
            if cache_key not in _lfc_cache:
                _lfc_cache[cache_key] = _class_lfcs(
                    adata_by_stratum[stratum], gene_idx)
            cls_lfcs, cls_counts = _lfc_cache[cache_key]

            # Emit per-supertype rows for this (gene, stratum) once.
            # The supertype-level values are independent of contrast and NES;
            # one (gene, stratum, supertype) tuple per unique combination.
            if cache_key not in _supertype_emitted:
                _supertype_emitted.add(cache_key)
                effects = adata_by_stratum[stratum].X[gene_idx, :]
                if hasattr(effects, "toarray"):
                    effects = effects.toarray().flatten()
                else:
                    effects = np.asarray(effects).flatten()
                for i, st in enumerate(supertypes):
                    val = effects[i]
                    if not np.isfinite(val):
                        continue
                    supertype_rows.append({
                        "gene_symbol": gene,
                        "stratum": stratum,
                        "supertype": st,
                        "subclass": st_to_subclass[st],
                        "supertype_lfc": float(val),
                    })

            residue_type = row.get("residue_type", "ST")
            track_name = row.get("track", "st")
            for wmb_class, median_lfc in cls_lfcs.items():
                concordance = np.sign(nes) * median_lfc
                sea_ad_rows.append({
                    "kinase": kinase,
                    "gene_symbol": gene,
                    "contrast": contrast,
                    "NES": nes,
                    "FDR": fdr,
                    "residue_type": residue_type,
                    "track": track_name,
                    "cell_type": wmb_class,
                    "sea_ad_lfc": median_lfc,
                    "sea_ad_n_supertypes": cls_counts[wmb_class],
                    "sea_ad_stratum": stratum,
                    "concordance_score": concordance,
                })

        print(f"  SEA-AD concordance: {len(sea_ad_rows)} "
              f"(kinase, contrast, cell_type) rows")

    except (ImportError, FileNotFoundError) as e:
        print(f"  SEA-AD not available ({e}), skipping concordance")

    sea_ad_df = pd.DataFrame(sea_ad_rows)

    # Audit artifact: per-(gene, stratum, supertype) LFCs.
    # Used by the unified viewer Attribution drawer to render a supertype
    # heatmap grouped by subclass.
    if supertype_rows:
        supertype_df = pd.DataFrame(supertype_rows)
        supertype_path = os.path.join(OUTPUT_DIR, "sea_ad_supertype_lfc.csv")
        supertype_df.to_csv(supertype_path, index=False)
        print(f"  SEA-AD supertype LFCs: {len(supertype_df)} rows → "
              f"{supertype_path}")

    # 3d. WMB expression specificity (per kinase × subclass)
    # Specificity is a relative ratio (cell-type mean log2 / sum across 24
    # subclasses), so it can flag genes as "specific" while their absolute
    # expression is in the noise floor — common with mitotic kinases in OPC.
    # We carry the absolute expression columns through to the attribution
    # output so downstream consumers can audit this.
    wmb_spec = {}
    wmb_mean_log2 = {}
    wmb_frac_cells = {}
    wmb_binary = {}
    if os.path.exists(WMB_EXPRESSION_FILE):
        wmb = pd.read_csv(WMB_EXPRESSION_FILE)
        wmb_key = list(zip(wmb["gene_symbol"].str.upper(), wmb["cell_type"]))
        wmb["_key"] = wmb_key
        wmb_max_idx = wmb.groupby("_key")["specificity_score"].idxmax()
        wmb_top = wmb.loc[wmb_max_idx]
        wmb_spec = dict(zip(wmb_top["_key"], wmb_top["specificity_score"]))
        wmb_mean_log2 = dict(zip(wmb_top["_key"], wmb_top["mean_log2_expression"]))
        wmb_frac_cells = dict(zip(wmb_top["_key"], wmb_top["fraction_cells_expressing"]))
        wmb_binary = dict(zip(wmb_top["_key"], wmb_top["binary_expressed"]))
        print(f"  WMB specificity: {len(wmb_spec)} (gene, cell_type) pairs loaded")
    else:
        print(f"  WMB expression file not found at {WMB_EXPRESSION_FILE}")

    # 3d′. Song within-cohort evidence (specificity + concordance)
    song_spec = {}
    if os.path.exists(config.SONG_EXPRESSION_FILE):
        song_sp = pd.read_csv(config.SONG_EXPRESSION_FILE)
        song_sp_grouped = song_sp.groupby(
            [song_sp["gene_symbol"].str.upper(), "cell_type"]
        )["specificity_score"].max()
        song_spec = song_sp_grouped.to_dict()
        print(f"  Song specificity: {len(song_spec)} (gene, cell_type) pairs loaded")

    song_conc = {}
    song_conc_pval = {}
    song_conc_fdr = {}
    if os.path.exists(config.SONG_CONCORDANCE_FILE):
        song_cd = pd.read_csv(config.SONG_CONCORDANCE_FILE)
        # snRNA OLS now emits 9 time-resolved contrasts, matching the bulk
        # pipeline. Backwards compatible with the legacy 3-pathway schema.
        contrast_col = "contrast" if "contrast" in song_cd.columns else "pathway"
        song_cd["_key"] = list(zip(
            song_cd["gene_symbol"].str.upper(),
            song_cd["cell_type"],
            song_cd[contrast_col],
        ))
        song_conc = dict(zip(song_cd["_key"], song_cd["song_lfc"]))
        if "song_pval" in song_cd.columns:
            song_conc_pval = dict(zip(song_cd["_key"], song_cd["song_pval"]))
        if "song_fdr" in song_cd.columns:
            song_conc_fdr = dict(zip(song_cd["_key"], song_cd["song_fdr"]))
        print(f"  Song concordance: {len(song_conc)} (gene, cell_type, "
              f"{contrast_col}) entries loaded")
        _song_key_is_contrast = (contrast_col == "contrast")

    # 3e. Combine into unified attribution table.
    # Build a sig × WMB_CLASSES base so every (kinase, contrast) gets a row
    # for every WMB class. SEA-AD evidence is left-joined where it exists
    # (≈ 9 of 34 classes); Song concordance + WMB specificity always join.
    print(f"  Building unified base: {len(sig)} sig rows × "
          f"{len(config.WMB_CLASSES)} WMB classes")
    base_rows = []
    for _, row in sig.iterrows():
        gene_upper = row["gene_symbol"].upper() if isinstance(
            row["gene_symbol"], str) else ""
        is_sig = bool(np.isfinite(row["FDR"]) and row["FDR"] < config.MEA_FDR_THRESH)
        for cell_type in config.WMB_CLASSES:
            base_rows.append({
                "kinase": row["kinase"],
                "gene_symbol": row["gene_symbol"],
                "_gene_upper": gene_upper,
                "contrast": row["contrast"],
                "NES": row["NES"],
                "FDR": row["FDR"],
                "mea_significant": is_sig,
                "residue_type": row.get("residue_type", "ST"),
                "track": row.get("track", "st"),
                "cell_type": cell_type,
            })
    unified = pd.DataFrame(base_rows)

    # Left-join SEA-AD evidence (only present for SEA-AD-covered classes)
    if len(sea_ad_df) > 0:
        sea_ad_join = sea_ad_df[[
            "kinase", "contrast", "cell_type",
            "sea_ad_lfc", "sea_ad_n_supertypes", "sea_ad_stratum",
            "concordance_score",
        ]].copy()
        unified = unified.merge(
            sea_ad_join, on=["kinase", "contrast", "cell_type"], how="left"
        )
    else:
        unified["sea_ad_lfc"] = np.nan
        unified["sea_ad_n_supertypes"] = 0
        unified["sea_ad_stratum"] = ""
        unified["concordance_score"] = 0.0

    # WMB specificity + raw absolute expression
    keys = list(zip(unified["_gene_upper"], unified["cell_type"]))
    unified["wmb_specificity"] = [wmb_spec.get(k, 0.0) for k in keys]
    unified["wmb_mean_log2_expression"] = [wmb_mean_log2.get(k, np.nan) for k in keys]
    unified["wmb_fraction_cells_expressing"] = [wmb_frac_cells.get(k, np.nan) for k in keys]
    unified["wmb_binary_expressed"] = [bool(wmb_binary.get(k, False)) for k in keys]

    # Song within-cohort evidence — keyed on contrast when the snRNA OLS
    # produced per-contrast rows; falls back to pathway prefix for the legacy
    # 3-pathway schema.
    unified["song_specificity"] = [song_spec.get(k, np.nan) for k in keys]
    if locals().get("_song_key_is_contrast", False):
        song_keys = list(zip(unified["_gene_upper"], unified["cell_type"],
                             unified["contrast"]))
    else:
        pathways = unified["contrast"].str.split("_").str[0]
        song_keys = list(zip(unified["_gene_upper"], unified["cell_type"], pathways))
    unified["song_lfc"] = [song_conc.get(k, np.nan) for k in song_keys]
    # Per-row p-value/FDR from the snRNA OLS — surfaced for downstream
    # interpretability. With n≈15 males and a 10-param design, BH FDR within
    # (cell_type × contrast) strata saturates near 1.0; uncorrected p<0.05 is
    # the most useful directional flag the dataset can support, and consumers
    # should treat it as such (see snrna_integration.step_concordance).
    unified["song_pval"] = [song_conc_pval.get(k, np.nan) for k in song_keys]
    unified["song_fdr"]  = [song_conc_fdr.get(k, np.nan) for k in song_keys]
    unified["song_concordance_score"] = np.where(
        np.isfinite(unified["song_lfc"]),
        np.sign(unified["NES"]) * unified["song_lfc"],
        np.nan,
    )

    # Weighted concordance: Song (3×) + SEA-AD (1×), neither has veto
    w_song = config.SONG_CONCORDANCE_WEIGHT
    w_sea_ad = config.SEA_AD_CONCORDANCE_WEIGHT
    song_cs = unified["song_concordance_score"]
    sea_ad_cs = unified["concordance_score"]
    has_song_v = np.isfinite(song_cs)
    has_sea_ad_v = np.isfinite(sea_ad_cs) & (sea_ad_cs != 0)
    has_both = has_song_v & has_sea_ad_v
    has_song_only = has_song_v & ~has_sea_ad_v
    has_sea_ad_only = ~has_song_v & has_sea_ad_v
    unified["effective_concordance"] = np.select(
        [has_both, has_song_only, has_sea_ad_only],
        [(w_song * song_cs + w_sea_ad * sea_ad_cs) / (w_song + w_sea_ad),
         song_cs, sea_ad_cs],
        default=0.0,
    )
    unified["concordance_source"] = np.select(
        [has_both, has_song_only, has_sea_ad_only],
        ["both", "song", "sea_ad"],
        default="none",
    )

    unified["combined_score"] = (
        unified["effective_concordance"] * (0.5 + unified["wmb_specificity"]))
    conf, basis = _assign_confidence_and_basis_vectorized(unified)
    # Confidence labels (high/moderate/low) require MEA significance —
    # they describe how well evidence supports an *enrichment hit*. Rows
    # below the MEA FDR gate are kept in the table for visibility but
    # carry confidence "none" so downstream filters preserve their meaning.
    not_mea_sig = ~unified["mea_significant"].astype(bool)
    conf = conf.where(~not_mea_sig, "none")
    basis = basis.where(~not_mea_sig, "weak")
    unified["combined_confidence"] = conf
    unified["evidence_basis"] = basis

    # Drop helper column before saving
    unified = unified.drop(columns=["_gene_upper"], errors="ignore")

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
        print(f"\n  By cell type (WMB class):")
        for ct, cnt in attributed[
                "cell_type"].value_counts().items():
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
    """Optional: Run raw phospho MEA and classify abundance/activity/both.

    Runs raw-phospho MEA for every track (ST + pY) so the audit chain has a
    raw-phospho counterpart for every kinase that has a stoichiometry score.
    Mechanism annotation is computed per track and concatenated.
    """
    _ensure_output_dir()
    print(f"\n=== Mechanism Annotation ({config.ANALYSIS_MODE}) ===\n")

    mapping_full = load_sample_mapping()
    mapping = _filter_samples(mapping_full)
    bio_cols = mapping["column_name"].tolist()

    tracks = [_resolve_track(t) for t in ("st", "py")]
    raw_mea_by_track = {}
    for track_cfg in tracks:
        raw_path = _track_output("raw_phospho_normalized.csv", track_cfg)
        if not os.path.exists(raw_path):
            print(f"  [{track_cfg['name']}] {raw_path} missing; skip raw MEA "
                  "(run --normalize for this track first).")
            continue
        raw_df = pd.read_csv(raw_path)
        if raw_df.empty:
            print(f"  [{track_cfg['name']}] raw normalized file empty; skip.")
            continue
        print(f"  [{track_cfg['name']}] OLS on raw phospho ({len(raw_df)} sites)...")
        X, X_np, param_names, Y_raw, betas_r, pvals_r, nobs_r = _prepare_raw_ols(
            mapping, bio_cols, raw_df)
        results_by_contrast = {}
        for contrast_name, coefs in CONTRAST_COEFS.items():
            c_vec = np.zeros(len(param_names))
            for param, weight in coefs.items():
                c_vec[param_names.index(param)] = weight
            results_by_contrast[contrast_name] = {"raw_lfc": betas_r @ c_vec}
        print(f"  [{track_cfg['name']}] running MEA on raw phospho...")
        mea_raw = _run_mea(raw_df["motif"], results_by_contrast, "raw_lfc",
                           site_ids=raw_df["site_id"].values,
                           gene_symbols=raw_df["gene_symbol"].values,
                           track=track_cfg["name"])
        mea_raw_path = _track_output("mea_raw_phospho.csv", track_cfg)
        mea_raw.to_csv(mea_raw_path, index=False)
        print(f"  Saved {mea_raw_path} ({len(mea_raw)} rows)")
        raw_mea_by_track[track_cfg["name"]] = mea_raw

    if not raw_mea_by_track:
        print("\n  No raw-phospho tracks were processed; skipping mechanism table.")
        return

    # Concatenate ST + pY for downstream mechanism classification.
    mea_raw = pd.concat(list(raw_mea_by_track.values()), ignore_index=True)

    # Load stoichiometry MEA from every track (ST + pY) so tyrosine kinases
    # are included in the mechanism table.
    stoich_paths = [_track_output("mea_stoichiometry.csv", t) for t in tracks]
    stoich_paths = [p for p in stoich_paths if os.path.exists(p)]
    if not stoich_paths:
        raise FileNotFoundError("No mea_stoichiometry*.csv found. Run --enrich first.")
    mea_stoich = pd.concat([pd.read_csv(p) for p in stoich_paths], ignore_index=True)

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
    parser.add_argument(
        "--track", choices=["st", "py", "both"], default="both",
        help="Phospho track: 'st' (Ser/Thr via IMAC, legacy), 'py' (Tyr), "
             "or 'both' (default; runs Stage 1+2 for each track sequentially). "
             "Stage 3 (--attribute) currently consumes the 'st' track only "
             "until Phase 5 lands.")

    args = parser.parse_args()

    # Resolve which tracks to iterate over for stages that are track-specific.
    if args.track == "both":
        tracks = ["st", "py"]
    else:
        tracks = [args.track]

    if args.normalize or args.run:
        for t in tracks:
            step_normalize(track=t)
    if args.enrich or args.run:
        for t in tracks:
            step_enrich(track=t)
    if args.attribute or args.run:
        # Stage 3 still operates on the S/T MEA file by default.
        # Phase 5 will combine tracks before attribution.
        step_attribute()
    if args.mechanism_annotation:
        step_mechanism_annotation()
    if args.summary:
        print_summary()

    if args.run:
        print_summary()


if __name__ == "__main__":
    main()
