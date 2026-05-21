#!/usr/bin/env python3
"""Stage 1 of kinase attribution: cross-plex IRS normalization + stoichiometry.

Inputs:
  data/datasets/song/primary/proteomics/
    song2024_tmttotal_protein_quant_merged_labeled (2).xlsx
  data/datasets/song/primary/phospho/
    song_IMAC_sitequant_merged_labeled (2).xlsx  (or pY workbook for track=py)
  outputs/reports/data_ingest/sample_mapping.csv
  alz/config.py

Outputs (under outputs/reports/kinase_attribution/, track-suffixed):
  stoichiometry_matrix.csv
  raw_phospho_normalized.csv
  stoichiometry_qc.csv
  normalization_summary.json
  pca_plots/{tp_raw,tp_norm}_by_{plex,genotype,sex,timepoint}.png
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.KINASE_ATTRIBUTION_OUTPUT_DIR
DATA_INGEST_DIR = config.DATA_INGEST_OUTPUT_DIR

TOTAL_PROTEOME_FILE = os.path.join(
    config.SONG_PRIMARY_PROTEOMICS_DIR,
    "song2024_tmttotal_protein_quant_merged_labeled (2).xlsx",
)
REF_CHANNEL = config.TMT_REF_CHANNEL

# Genes for stoichiometry QC spot-checks
QC_GENES = ["Mapt", "Gsk3b", "Akt1", "Mapk1", "Camk2a"]


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
    parts = col.split("_", 1)
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
    """Load a phospho-track xlsx and normalize it to the canonical IMAC schema."""
    cfg = _resolve_track(track_cfg)
    path = cfg["input_file"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Phospho-track {cfg['name']} input not found: {path}"
        )
    df = pd.read_excel(path, header=1)
    return _postprocess_phospho_df(df, cfg)


def _postprocess_phospho_df(df, track_cfg):
    """Canonicalise a pre-loaded phospho-track DataFrame to the IMAC schema.

    Split out so Kedro's ExcelDataset can do the read while this helper does
    the rename / site-id synthesis / residue-purity check that the legacy
    script bundled with the disk read.
    """
    cfg = _resolve_track(track_cfg)
    df = df.copy()

    # Rename sample columns: plex{N}_* → p{N}_* (only when prefix differs).
    if cfg["column_prefix"] == "plex":
        rename_map = {}
        for c in df.columns:
            if isinstance(c, str) and c.startswith("plex") and c.endswith("_sn_sum"):
                rest = c[len("plex"):]
                rename_map[c] = "p" + rest
        df = df.rename(columns=rename_map)
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
    """Internal Reference Scaling normalization."""
    ref_mat = pd.DataFrame(
        {p: quant_df[col] for p, col in ref_cols.items() if col in quant_df.columns}
    )
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
    mat = quant_df.values.astype(float).copy()
    mat[mat <= 0] = np.nan
    with np.errstate(divide="ignore"):
        mat = np.log2(mat)
    n_imputed = np.sum(~np.isfinite(mat))
    mat = config.minprob_impute(mat)
    print(f"  MinProb imputed {n_imputed} missing values for PCA")
    var = np.var(mat, axis=1)
    mat = mat[var > 0]
    if mat.shape[0] < 10:
        print(f"  WARNING: only {mat.shape[0]} proteins with variance for PCA")
        return None

    pca = PCA(n_components=min(10, mat.shape[1], mat.shape[0]))
    coords = pca.fit_transform(mat.T)
    var_exp = pca.explained_variance_ratio_ * 100

    col_order = quant_df.columns.tolist()
    meta = mapping.set_index("column_name").loc[col_order].reset_index()

    for factor, col_name in [("plex", "plex"), ("genotype", "genotype"),
                              ("sex", "sex"), ("timepoint", "timepoint")]:
        fig, ax = plt.subplots(figsize=(8, 6))
        groups = meta[col_name].unique()
        cmap = matplotlib.colormaps["tab10"].resampled(len(groups))
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


# ===========================================================================
# Stage 1: Cross-plex normalization + stoichiometry
# ===========================================================================

def step_normalize(track, mapping, tp, sq):
    """Stage 1: IRS normalization and stoichiometry computation.

    Pure-ish: takes pre-loaded inputs (sample mapping, total-proteome and
    phospho-sitequant DataFrames) and returns ``(stoich_df, raw_phospho_df,
    qc_df, summary_dict)``. PCA plots remain side effects (PNG writes) since
    they're ancillary diagnostics, not catalog-managed outputs. The CLI
    entrypoint loads inputs from disk; the Kedro nodes route catalog-loaded
    DataFrames into this function.
    """
    _ensure_output_dir()
    track_cfg = _resolve_track(track)
    print(f"\n=== Stage 1: Cross-Plex Normalization + Stoichiometry "
          f"(track={track_cfg['name']}/{track_cfg['label']}) ===\n")

    # --- 1.0 Sample mapping ---
    sample_to_plex = dict(zip(mapping["column_name"], mapping["plex"]))
    bio_cols = mapping["column_name"].tolist()
    print(f"  {len(bio_cols)} biological samples across "
          f"{mapping['plex'].nunique()} plexes")

    # --- 1.1 IRS-normalize total proteome ---
    print("\n--- 1.1 Total proteome normalization ---")
    tp_gene = tp["Gene Symbol"].copy()

    ref_cols_tp = {}
    for plex in sorted(mapping["plex"].unique()):
        rc = _proteome_ref_col(plex)
        if rc in tp.columns:
            ref_cols_tp[plex] = rc
        else:
            print(f"  WARNING: reference column {rc} not found in total proteome")

    all_tp_cols = bio_cols + list(ref_cols_tp.values())
    tp_quant_raw = tp[[c for c in all_tp_cols if c in tp.columns]].copy()
    tp_quant_raw = tp_quant_raw.apply(pd.to_numeric, errors="coerce")

    print("  PCA before normalization...")
    tp_bio_raw = tp_quant_raw[bio_cols]
    pca_before = _run_pca_and_plot(tp_bio_raw, mapping,
                                    "Total Proteome (raw)",
                                    f"tp_raw{track_cfg['output_suffix']}")

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

    print("  PCA after normalization...")
    pca_after = _run_pca_and_plot(tp_norm, mapping,
                                   f"Total Proteome ({norm_method})",
                                   f"tp_norm{track_cfg['output_suffix']}")

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

    # --- 1.2 IRS-normalize phospho sitequant ---
    print(f"\n--- 1.2 Phospho sitequant normalization "
          f"(track={track_cfg['name']}) ---")
    print(f"  {len(sq)} phosphosites in input")

    phospho_bio_cols = [_proteome_to_phospho_col(c) for c in bio_cols]
    missing_pcols = [c for c in phospho_bio_cols if c not in sq.columns]
    if missing_pcols:
        print(f"  WARNING: {len(missing_pcols)} phospho columns not found: "
              f"{missing_pcols[:3]}")
    phospho_bio_cols = [c for c in phospho_bio_cols if c in sq.columns]

    ref_cols_ph = {}
    for plex in sorted(mapping["plex"].unique()):
        rc = _phospho_ref_col(plex)
        if rc in sq.columns:
            ref_cols_ph[plex] = rc

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

    sq_genes = sq["gene_symbol"].fillna("").astype(str).str.upper()
    tp_genes = tp_gene.fillna("").astype(str).str.upper()

    gene_to_tp_idx = {}
    for idx, g in enumerate(tp_genes):
        if g and g != "0":
            gene_to_tp_idx.setdefault(g, []).append(idx)

    n_sites = len(sq)
    n_matched = 0
    ph_to_tp_col = {}
    for tp_col in bio_cols:
        ph_col = _proteome_to_phospho_col(tp_col)
        if ph_col in phospho_bio_cols:
            ph_to_tp_col[ph_col] = tp_col

    tp_norm_vals = tp_norm.values
    sq_norm_vals = sq_norm.values

    ph_col_to_tp_col_idx = {}
    for j, ph_col in enumerate(phospho_bio_cols):
        tp_col = ph_to_tp_col.get(ph_col)
        if tp_col and tp_col in bio_cols:
            ph_col_to_tp_col_idx[j] = bio_cols.index(tp_col)

    stoich_matrix = np.full((n_sites, len(bio_cols)), np.nan)
    site_matched = np.zeros(n_sites, dtype=bool)
    site_protein_gene = [""] * n_sites

    tp_row_for_site = np.full(n_sites, -1, dtype=int)
    for i in range(n_sites):
        gene_upper = sq_genes.iloc[i]
        if gene_upper in gene_to_tp_idx:
            tp_row_for_site[i] = gene_to_tp_idx[gene_upper][0]
            site_matched[i] = True
            site_protein_gene[i] = gene_upper
    n_matched = int(site_matched.sum())

    matched_idx = np.where(site_matched)[0]
    if len(matched_idx) > 0 and ph_col_to_tp_col_idx:
        ph_js = np.array(list(ph_col_to_tp_col_idx.keys()))
        tp_js = np.array(list(ph_col_to_tp_col_idx.values()))

        ph_vals = sq_norm_vals[np.ix_(matched_idx, ph_js)]
        tp_rows = tp_row_for_site[matched_idx]
        tp_vals = tp_norm_vals[tp_rows][:, tp_js]

        valid = (ph_vals > 0) & (tp_vals > 0) & np.isfinite(ph_vals) & np.isfinite(tp_vals)
        with np.errstate(divide="ignore", invalid="ignore"):
            stoich_vals = np.where(valid, np.log2(ph_vals) - np.log2(tp_vals), np.nan)

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

    stoich_df = pd.DataFrame(stoich_matrix, columns=bio_cols)
    stoich_df.insert(0, "site_id", sq["site_id"].values)
    stoich_df.insert(1, "gene_symbol", sq["gene_symbol"].values)
    stoich_df.insert(2, "motif", sq["motif"].values)
    stoich_df.insert(3, "matched_protein", site_matched)

    raw_phospho_df = pd.DataFrame(sq_norm_vals, columns=phospho_bio_cols)
    rename_map = {ph: ph_to_tp_col[ph] for ph in phospho_bio_cols
                  if ph in ph_to_tp_col}
    raw_phospho_df = raw_phospho_df.rename(columns=rename_map)
    raw_phospho_df.insert(0, "site_id", sq["site_id"].values)
    raw_phospho_df.insert(1, "gene_symbol", sq["gene_symbol"].values)
    raw_phospho_df.insert(2, "motif", sq["motif"].values)

    total_proteome_df = tp_norm.copy()
    total_proteome_df.insert(0, "gene_symbol", tp_gene.values)
    if "Protein Id" in tp.columns:
        total_proteome_df.insert(1, "protein_id", tp["Protein Id"].values)
    elif "protein_id" in tp.columns:
        total_proteome_df.insert(1, "protein_id", tp["protein_id"].values)

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
        site_idx = np.where(mask)[0][0]
        site_id = sq["site_id"].iloc[site_idx]
        site_pos = sq.get("site_position", sq.get("site_pos", pd.Series()))
        pos_str = site_pos.iloc[site_idx] if len(site_pos) > site_idx else "?"

        for geno in ["WTyp", "AppP", "Ttau", "ApTt"]:
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

    # Normalize always uses all 72 samples; analysis_mode is stamped only so
    # downstream consumers can confirm the cohort intended for OLS/MEA agrees
    # with the normalize-time view.
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
        **config.provenance_stamp(),
    }
    return stoich_df, raw_phospho_df, total_proteome_df, qc_df, norm_summary


# ===========================================================================
# CLI
# ===========================================================================

def main():
    """CLI shim: delegates to `kedro run --pipeline=normalize`."""
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project

    bootstrap_project(Path(__file__).resolve().parent.parent)
    with KedroSession.create() as session:
        session.run(pipeline_name="normalize")


if __name__ == "__main__":
    main()
