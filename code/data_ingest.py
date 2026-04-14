#!/usr/bin/env python3
# data_ingest — Song proteomics pipeline entry point
"""Primary data ingestion and characterization for the Song proteomics pipeline.

Reads the raw Song 72-animal TMT proteomics workbooks and produces the
structured tables that the downstream analysis (stoichiometry, attribution)
consumes.  This is the first stage of the live pipeline.

Inputs (all under data/incytr_collections/song/):
  primary/proteomics/Sample_list_72mice (1).xlsx
      TMT plex layout — maps 72 animals to 6 plexes × 10 channels each.
  primary/proteomics/song2024_tmttotal_protein_quant_merged_labeled (2).xlsx
      Total proteome quantitation — ~5,000 proteins × 72 animals.
  primary/proteomics/song_IMAC_sitequant_merged_labeled (2).xlsx
      IMAC phospho site-level quantitation — ~4,000 S/T sites × 6 plexes.
  primary/proteomics/song_IMAC_compositeSites_merged_labeled (2).xlsx
      IMAC phospho composite sites — ~3,000 grouped S/T sites.
  method_records/aobs_desp_standardized/inputs/A_obs_fractions.tsv
      snRNA-seq cell-type composition fractions (24 groups × 10 cell types).
  transcriptomics/snrna_sample_manifest.csv  (optional)
      Manifest linking mouse IDs to snRNA-seq sample IDs.

Outputs (all under outputs/reports/data_ingest/):
  sample_mapping.csv
      72-row table: plex, channel, column name, mouse ID, sex, timepoint,
      genotype, replicate, snRNA-seq linkage, phospho group ID.
  phospho_protein_matching.csv
      Per-site match of IMAC phosphosites to total proteome parent proteins.
  matching_summary.json
      Summary statistics for phosphosite-to-protein matching.
  datadriven_marker_assessment.csv
      Data-driven cell-type marker assessment from WMB atlas expression:
      per-gene specificity scores, correlation with snRNA-seq composition,
      Storey q-values.  Requires wmb_expression.py --proteome output.
  data_quality.json
      Missingness, batch-effect, and PCA summary statistics.
  pca_plots/pca_by_{plex,genotype,sex,timepoint}.png
      PCA scatter plots colored by experimental factors.

Steps (run individually or all at once with --run):
  §1 --mapping        TMT channel-to-animal sample mapping
  §2 --phospho-match  Phosphosite-to-protein matching
  §3 --markers        Cell-type marker protein assessment (WMB atlas, data-driven)
  §4 --quality        Data quality (missingness, batch effects, PCA)
  §5 --outliers       Statistical outlier detection (within-group z-scores)
"""

import argparse
import json
import os
import re


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.DATA_INGEST_OUTPUT_DIR

TOTAL_PROTEOME_FILE = os.path.join(
    config.SONG_PRIMARY_PROTEOMICS_DIR,
    "song2024_tmttotal_protein_quant_merged_labeled (2).xlsx",
)
SAMPLE_LIST_FILE = os.path.join(
    config.SONG_PRIMARY_PROTEOMICS_DIR,
    "Sample_list_72mice (1).xlsx",
)
IMAC_COMPOSITE_FILE = os.path.join(
    config.SONG_PRIMARY_PROTEOMICS_DIR,
    "song_IMAC_compositeSites_merged_labeled (2).xlsx",
)
IMAC_SITEQUANT_FILE = os.path.join(
    config.SONG_PRIMARY_PROTEOMICS_DIR,
    "song_IMAC_sitequant_merged_labeled (2).xlsx",
)
# TMT genotype labels → SAP canonical condition names
GENOTYPE_TO_SAP = {"WT": "WTyp", "APP": "AppP", "T22": "Ttau", "T22/APP": "ApTt"}
SEX_TO_SAP = {"M": "ma", "F": "fe"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _discover_snrna_samples():
    """Load the localized manifest of snRNA-seq-linked sample IDs.

    Returns a dict mapping normalized mouse_id (e.g. 'C198') to the full
    sample string (e.g. 'C198_ma_2mo_WTyp').
    """
    path = config.SONG_SNRNA_SAMPLE_MANIFEST
    if not os.path.exists(path):
        return {}
    manifest = pd.read_csv(path)
    if not {"sample_id", "mouse_id"}.issubset(manifest.columns):
        raise ValueError(
            f"snRNA manifest at {path} must contain sample_id and mouse_id columns"
        )
    manifest = manifest.dropna(subset=["sample_id", "mouse_id"]).copy()
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    manifest["mouse_id"] = manifest["mouse_id"].astype(str)
    return dict(zip(manifest["mouse_id"], manifest["sample_id"]))


def _normalize_mouse_id(raw_id):
    """Normalize mouse ID from TMT format to snRNA format.

    TMT uses e.g. 'C198(L)', 'E49( R)', 'D92 (L)' — strip parenthetical,
    then zero-pad single-digit numbers to match snRNA's 'D092', 'E049'.
    """
    # Strip parenthetical suffix and whitespace
    clean = re.sub(r"\s*\(.*?\)\s*", "", raw_id).strip()
    # Split letter prefix from numeric part
    m = re.match(r"^([A-Z])(\d+)$", clean)
    if not m:
        return clean
    letter, num = m.group(1), m.group(2)
    # Zero-pad to 3 digits to match snRNA convention (D92 → D092)
    return f"{letter}{int(num):03d}"


def _parse_animal_id(animal_str):
    """Parse a TMT animal ID string like '1_C198(L)_M_2mo_WT'.

    Returns dict with keys: sample_num, mouse_id_raw, mouse_id, sex,
    timepoint, genotype.  Returns None if the string doesn't match.
    """
    pat = re.compile(
        r"^(\d+)_(.+?)_(M|F)_(2mo|4mo|6mo)_(WT|T22|APP|T22/APP)$"
    )
    m = pat.match(animal_str)
    if not m:
        return None
    return {
        "sample_num": int(m.group(1)),
        "mouse_id_raw": m.group(2),
        "mouse_id": _normalize_mouse_id(m.group(2)),
        "sex": m.group(3),
        "timepoint": m.group(4),
        "genotype": m.group(5),
    }


def _plex_channel_to_colname(plex, channel):
    """Convert (plex_int, channel_str) to total proteome column name.

    E.g. (1, '128N') → 'plex1_128n_sn_mean'
    """
    return f"plex{plex}_{channel.lower()}_sn_mean"


def load_sample_mapping():
    """Load the cached sample mapping CSV (output of --mapping step)."""
    path = os.path.join(OUTPUT_DIR, "sample_mapping.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sample mapping not found at {path}. Run --mapping first."
        )
    return pd.read_csv(path)


def load_total_proteome():
    """Load total proteome data and return (meta_df, quant_df, mapping_df).

    meta_df: protein metadata (protein_id, Gene Symbol, etc.)
    quant_df: 72 biological sample columns, indexed by Gene Symbol
    mapping_df: sample mapping from --mapping step
    """
    print("  Loading total proteome Excel...")
    tp = pd.read_excel(TOTAL_PROTEOME_FILE, header=1)
    mapping = load_sample_mapping()

    bio_cols = mapping["column_name"].tolist()
    missing_cols = [c for c in bio_cols if c not in tp.columns]
    if missing_cols:
        raise ValueError(
            f"{len(missing_cols)} expected columns not found: {missing_cols[:5]}"
        )

    meta = tp[["protein_id", "Gene Symbol", "geneID", "transcriptID",
               "ProtDescription"]].copy()
    quant = tp[bio_cols].copy()
    # Convert to numeric, coercing errors
    quant = quant.apply(pd.to_numeric, errors="coerce")

    return meta, quant, mapping


# ===========================================================================
# §1  Sample Mapping
# ===========================================================================


def step_sample_mapping():
    """Parse TMT plex layout, map channels to animals, cross-ref snRNA-seq."""
    _ensure_output_dir()
    print("§1: Building sample mapping...")

    # 1. Parse TMT plex layout
    layout = pd.read_excel(SAMPLE_LIST_FILE, sheet_name="TMT plex layout")
    # Drop blank separator rows
    layout = layout.dropna(subset=["Plex"]).copy()
    layout["Plex"] = layout["Plex"].astype(int)

    # 2. Filter to biological samples (exclude pools, empty, na)
    exclude_keywords = {"Large Pool", "na", "Ref_Pool", "WT_M", "WT_F",
                        "DKO_M", "DKO_F"}
    bio = layout[~layout["Animal #"].isin(exclude_keywords)].copy()
    print(f"  Biological samples in TMT layout: {len(bio)}")

    # 3. Parse animal IDs
    records = []
    for _, row in bio.iterrows():
        plex = int(row["Plex"])
        channel = str(row["Channel"])
        animal_str = str(row["Animal #"])
        parsed = _parse_animal_id(animal_str)
        if parsed is None:
            print(f"  WARNING: could not parse animal ID: '{animal_str}'")
            continue
        colname = _plex_channel_to_colname(plex, channel)
        records.append({
            "plex": plex,
            "channel": channel,
            "column_name": colname,
            "animal_id": animal_str,
            "mouse_id": parsed["mouse_id"],
            "mouse_id_raw": parsed["mouse_id_raw"],
            "sex": parsed["sex"],
            "timepoint": parsed["timepoint"],
            "genotype": parsed["genotype"],
            "sample_num": parsed["sample_num"],
        })

    df = pd.DataFrame(records)
    print(f"  Parsed {len(df)} biological samples")

    # 4. Verify against total proteome columns
    print("  Verifying columns exist in total proteome...")
    tp = pd.read_excel(TOTAL_PROTEOME_FILE, header=1, nrows=0)
    tp_cols = set(tp.columns)
    df["column_exists"] = df["column_name"].isin(tp_cols)
    n_missing = (~df["column_exists"]).sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} columns not found in total proteome")
        print(df[~df["column_exists"]][["animal_id", "column_name"]])
    else:
        print("  All 72 columns verified in total proteome file")

    # 5. Assign replicate numbers (within sex × timepoint × genotype groups)
    df = df.sort_values(["sex", "timepoint", "genotype", "sample_num"])
    df["replicate"] = df.groupby(["sex", "timepoint", "genotype"]).cumcount() + 1

    # 6. Cross-reference with snRNA-seq
    snrna_samples = _discover_snrna_samples()
    df["has_snrna_seq"] = df["mouse_id"].map(
        lambda mid: mid in snrna_samples
    )
    df["snrna_sample_id"] = df["mouse_id"].map(
        lambda mid: snrna_samples.get(mid, "")
    )
    n_snrna = df["has_snrna_seq"].sum()
    print(f"  Animals with snRNA-seq: {n_snrna} / {len(df)}")

    # 7. Build phospho group ID (matching A_obs sample_id format)
    df["phospho_group_id"] = df.apply(
        lambda r: (
            f"{SEX_TO_SAP[r['sex']]}_{r['timepoint']}"
            f"_{GENOTYPE_TO_SAP[r['genotype']]}"
        ),
        axis=1,
    )

    # 8. Verify design balance
    design = df.groupby(["sex", "timepoint", "genotype"]).size()
    print(f"\n  Design balance (sex × timepoint × genotype → n replicates):")
    for (sex, tp, geno), n in design.items():
        print(f"    {sex} × {tp} × {geno}: n={n}")
    if (design != 3).any():
        print("  WARNING: not all groups have exactly 3 replicates")
    else:
        print("  All 24 groups have exactly 3 replicates (72 animals total)")

    # 9. Save
    out_cols = ["plex", "channel", "column_name", "animal_id", "mouse_id",
                "sex", "timepoint", "genotype", "replicate", "has_snrna_seq",
                "snrna_sample_id", "phospho_group_id"]
    df = df[out_cols].reset_index(drop=True)
    out_path = os.path.join(OUTPUT_DIR, "sample_mapping.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Summary
    print(f"\n  Summary:")
    print(f"    Total animals: {len(df)}")
    print(f"    With snRNA-seq: {n_snrna}")
    print(f"    Without snRNA-seq: {len(df) - n_snrna}")
    print(f"    Unique phospho groups: {df['phospho_group_id'].nunique()}")
    return df


# ===========================================================================
# §2  Phosphosite-to-Protein Matching
# ===========================================================================


def step_phospho_match():
    """Match phosphosite parent proteins to the total proteome."""
    _ensure_output_dir()
    print("§2: Phosphosite-to-protein matching...")

    # 1. Load total proteome gene symbols
    print("  Loading total proteome...")
    tp = pd.read_excel(TOTAL_PROTEOME_FILE, header=1)
    tp_genes_raw = tp["Gene Symbol"].copy()
    # Filter out non-gene entries (Gene Symbol == 0 for custom/contaminant)
    tp_genes_raw = tp_genes_raw[tp_genes_raw != 0].astype(str)
    tp_genes_upper = set(tp_genes_raw.str.upper())
    print(f"  Total proteome: {len(tp)} proteins, "
          f"{len(tp_genes_upper)} unique gene symbols")

    # 2. Load IMAC sitequant phospho data (primary Excel)
    print("  Loading IMAC sitequant phospho data...")
    phospho = pd.read_excel(IMAC_SITEQUANT_FILE, header=1)
    phospho["gene_symbol_upper"] = (
        phospho["gene_symbol"].fillna("").astype(str).str.upper()
    )

    n_sites = len(phospho)
    n_unique_genes = phospho.loc[
        phospho["gene_symbol_upper"] != "", "gene_symbol_upper"
    ].nunique()
    print(f"  Phospho sites: {n_sites}")
    print(f"  Unique parent proteins (gene symbols): {n_unique_genes}")

    # 3. Match
    phospho["matched_in_total_proteome"] = phospho["gene_symbol_upper"].isin(
        tp_genes_upper
    )

    n_matched_sites = phospho["matched_in_total_proteome"].sum()
    matched_genes = phospho.loc[
        phospho["matched_in_total_proteome"], "gene_symbol_upper"
    ].nunique()
    unmatched_genes = phospho.loc[
        ~phospho["matched_in_total_proteome"], "gene_symbol_upper"
    ].nunique()

    print(f"\n  Matching results (IMAC sitequant):")
    print(f"    Sites with matched parent protein: "
          f"{n_matched_sites}/{n_sites} ({100*n_matched_sites/n_sites:.1f}%)")
    print(f"    Unique parent proteins matched: "
          f"{matched_genes}/{n_unique_genes} "
          f"({100*matched_genes/n_unique_genes:.1f}%)")
    print(f"    Unmatched parent proteins: {unmatched_genes}")

    # Sites per matched protein
    sites_per_prot = phospho.loc[
        phospho["matched_in_total_proteome"]
    ].groupby("gene_symbol_upper").size()
    print(f"\n  Sites per matched protein:")
    print(f"    Mean: {sites_per_prot.mean():.1f}")
    print(f"    Median: {sites_per_prot.median():.0f}")
    print(f"    Range: {sites_per_prot.min()}-{sites_per_prot.max()}")

    # 4. Save per-site matching
    out_df = phospho[["site_id", "protein_id", "gene_symbol",
                       "matched_in_total_proteome"]].copy()
    out_path = os.path.join(OUTPUT_DIR, "phospho_protein_matching.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # 5. Also check composite sites
    summary = {
        "imac_sitequant": {
            "total_sites": int(n_sites),
            "unique_parent_proteins": int(n_unique_genes),
            "matched_sites": int(n_matched_sites),
            "matched_sites_pct": round(100 * n_matched_sites / n_sites, 1),
            "matched_proteins": int(matched_genes),
            "matched_proteins_pct": round(
                100 * matched_genes / n_unique_genes, 1
            ),
            "unmatched_proteins": int(unmatched_genes),
            "sites_per_matched_protein_mean": round(
                sites_per_prot.mean(), 1
            ),
            "sites_per_matched_protein_median": int(sites_per_prot.median()),
        },
    }

    # Composite sites matching
    print("\n  Checking IMAC composite sites...")
    try:
        comp = pd.read_excel(IMAC_COMPOSITE_FILE, header=1)
        comp_genes = comp["gene_symbol"].dropna().astype(str).str.upper()
        n_comp = len(comp)
        n_comp_genes = comp_genes.nunique()
        comp_matched_genes = comp_genes.isin(tp_genes_upper).groupby(
            comp_genes
        ).first()
        n_comp_matched = comp_matched_genes.sum()
        comp_matched_sites = comp_genes.isin(tp_genes_upper).sum()
        print(f"    Composite sites: {n_comp}")
        print(f"    Unique genes: {n_comp_genes}")
        print(f"    Sites with matched protein: "
              f"{comp_matched_sites}/{n_comp} "
              f"({100*comp_matched_sites/n_comp:.1f}%)")
        print(f"    Genes matched: {n_comp_matched}/{n_comp_genes} "
              f"({100*n_comp_matched/n_comp_genes:.1f}%)")
        summary["imac_composite"] = {
            "total_sites": int(n_comp),
            "unique_parent_proteins": int(n_comp_genes),
            "matched_sites": int(comp_matched_sites),
            "matched_sites_pct": round(100 * comp_matched_sites / n_comp, 1),
            "matched_proteins": int(n_comp_matched),
            "matched_proteins_pct": round(
                100 * n_comp_matched / n_comp_genes, 1
            ),
        }
    except Exception as e:
        print(f"    Could not read composite sites: {e}")

    # Site-level quant matching (reuse already-loaded phospho DataFrame)
    print("\n  Checking IMAC site-level quant...")
    try:
        sq = phospho
        sq_genes = sq["gene_symbol"].dropna().astype(str).str.upper()
        n_sq = len(sq)
        n_sq_genes = sq_genes.nunique()
        sq_matched_sites = sq_genes.isin(tp_genes_upper).sum()
        sq_matched_genes_n = sq_genes[sq_genes.isin(tp_genes_upper)].nunique()
        print(f"    Site-level sites: {n_sq}")
        print(f"    Unique genes: {n_sq_genes}")
        print(f"    Sites with matched protein: "
              f"{sq_matched_sites}/{n_sq} "
              f"({100*sq_matched_sites/n_sq:.1f}%)")
        print(f"    Genes matched: {sq_matched_genes_n}/{n_sq_genes} "
              f"({100*sq_matched_genes_n/n_sq_genes:.1f}%)")
        summary["imac_sitequant"] = {
            "total_sites": int(n_sq),
            "unique_parent_proteins": int(n_sq_genes),
            "matched_sites": int(sq_matched_sites),
            "matched_sites_pct": round(100 * sq_matched_sites / n_sq, 1),
            "matched_proteins": int(sq_matched_genes_n),
            "matched_proteins_pct": round(
                100 * sq_matched_genes_n / n_sq_genes, 1
            ),
        }
    except Exception as e:
        print(f"    Could not read site-level quant: {e}")

    # Total proteome stats
    summary["total_proteome"] = {
        "total_proteins": int(len(tp)),
        "unique_gene_symbols": int(len(tp_genes_upper)),
    }

    # Export gene list for proteome-wide WMB expression
    gene_list_path = os.path.join(OUTPUT_DIR, "total_proteome_genes.txt")
    with open(gene_list_path, "w") as f:
        for g in sorted(tp_genes_upper):
            f.write(g + "\n")
    print(f"\n  Saved: {gene_list_path} ({len(tp_genes_upper)} genes)")

    sum_path = os.path.join(OUTPUT_DIR, "matching_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {sum_path}")


# ===========================================================================
# §3  Cell-Type Marker Protein Assessment (WMB atlas, data-driven)
# ===========================================================================


def _compute_subclass_fractions():
    """Recompute snRNA composition fractions at SEA-AD subclass resolution.

    Reads per-cluster-label cell counts from yuyu_clustersize.csv (46 labels ×
    24 sample groups), maps labels to SEA-AD subclasses via
    config.SNRNA_CLUSTER_TAXONOMY, sums counts per subclass per group, and
    returns fractions (each group sums to 1).

    Returns
    -------
    pd.DataFrame
        Index: sample group IDs (24 rows, e.g., "fe_2mo_AppP").
        Columns: subclass names that have nonzero counts (SEA-AD subclasses +
        researcher categories like "Generic_excitatory", "Medium spiny neurons").
    """
    raw = pd.read_csv(config.CLUSTERSIZE_FILE, index_col=0)
    # raw: rows = cluster labels, columns = 24 sample groups

    # Map each row label → sea_ad_subclass
    taxonomy = config.SNRNA_CLUSTER_TAXONOMY
    label_to_subclass = {}
    for label in raw.index:
        if label.startswith("cluster-"):
            label_to_subclass[label] = "Other"
        elif label in taxonomy:
            label_to_subclass[label] = taxonomy[label]["sea_ad_subclass"]
        else:
            raise ValueError(
                f"SNRNA_CLUSTER_TAXONOMY is missing label: {label!r}. "
                "Update config.SNRNA_CLUSTER_TAXONOMY to include this cluster."
            )

    raw["_subclass"] = raw.index.map(label_to_subclass)
    grouped = raw.groupby("_subclass").sum()

    # Convert counts → fractions (each sample group sums to 1)
    fractions = grouped.div(grouped.sum(axis=0), axis=1).T
    fractions.index.name = "sample_id"

    # Drop columns that are all zero (no cells in any group)
    fractions = fractions.loc[:, (fractions > 0).any(axis=0)]

    n_subclasses = len(fractions.columns)
    n_from_sea_ad = sum(1 for c in fractions.columns if c in config.SEA_AD_SUBCLASSES)
    print(f"  Subclass fractions: {n_subclasses} categories "
          f"({n_from_sea_ad} SEA-AD subclasses, "
          f"{n_subclasses - n_from_sea_ad} researcher categories)")
    return fractions


def step_markers():
    """Identify cell-type markers from WMB atlas expression, test against proteome.

    Uses the proteome-wide WMB expression matrix (from wmb_expression.py --proteome)
    to rank genes by cell-type specificity, then correlates the top markers with
    snRNA-seq composition fractions.
    """
    _ensure_output_dir()
    print("§3: Cell-type marker protein assessment (WMB atlas)...")

    wmb_path = config.WMB_PROTEOME_EXPRESSION_FILE
    if not os.path.exists(wmb_path):
        raise FileNotFoundError(
            f"WMB proteome expression not found at {wmb_path}. "
            "Run: python code/wmb_expression.py --proteome"
        )

    wmb = pd.read_csv(wmb_path)
    print(f"  WMB proteome expression: {len(wmb)} rows")

    meta, quant, mapping = load_total_proteome()
    gene_upper = meta["Gene Symbol"].astype(str).str.upper()

    # Recompute snRNA composition fractions at subclass resolution
    subclass_fracs = _compute_subclass_fractions()
    available_frac_cols = set(subclass_fracs.columns)

    # Map each WMB cell type → composition column.  For SEA-AD subclasses with
    # cluster data the column is the subclass name itself; for subclasses with
    # no snRNA clusters mapped (e.g., Pvalb, Sst, Chandelier) → None.
    subclass_to_frac_col = {}
    for sc in config.SEA_AD_SUBCLASSES:
        subclass_to_frac_col[sc] = sc if sc in available_frac_cols else None

    # Pre-compute lookups used in the inner loop
    col_to_idx = {col: i for i, col in enumerate(quant.columns)}
    gene_to_row = {}
    for i, g in enumerate(gene_upper):
        gene_to_row.setdefault(g, i)  # first occurrence wins

    # Pre-compute snRNA animal → (quant column index, sample group ID) pairs
    snrna_animals = mapping[mapping["has_snrna_seq"]].copy()
    snrna_pairs = []  # [(col_idx, group_id), ...]
    for _, srow in snrna_animals.iterrows():
        col = srow["column_name"]
        gid = srow["phospho_group_id"]
        if col in col_to_idx and gid in subclass_fracs.index:
            snrna_pairs.append((col_to_idx[col], gid))

    records = []
    cell_types = wmb["cell_type"].unique()

    for ct in sorted(cell_types):
        ct_df = wmb[wmb["cell_type"] == ct].copy()
        # Only consider genes that are expressed (mean > 1, fraction > 10%)
        ct_expr = ct_df[ct_df["binary_expressed"]].copy()
        ct_expr = ct_expr.sort_values("specificity_score", ascending=False)

        frac_col = subclass_to_frac_col.get(ct)
        has_composition = frac_col is not None

        # Pre-extract composition values for this cell type (constant across genes)
        if has_composition and snrna_pairs:
            frac_dict = subclass_fracs[frac_col].to_dict()
            comp_by_pair = [(ci, frac_dict[gid])
                           for ci, gid in snrna_pairs
                           if not np.isnan(frac_dict.get(gid, np.nan))]
        else:
            comp_by_pair = []

        for rank, (_, row) in enumerate(ct_expr.iterrows(), 1):
            gene_human = row["gene_symbol_human"]
            gene_mouse = row["gene_symbol_mouse"]

            row_idx = gene_to_row.get(gene_human)
            found = row_idx is not None

            rec = {
                "cell_type": ct,
                "rank_in_cell_type": rank,
                "gene_symbol": gene_mouse,
                "gene_symbol_human": gene_human,
                "specificity_score": row["specificity_score"],
                "wmb_mean_expression": row["mean_log2_expression"],
                "wmb_fraction_expressing": row["fraction_cells_expressing"],
                "found_in_total_proteome": found,
                "mean_intensity": np.nan,
                "correlation_with_snrna_composition": np.nan,
                "correlation_pvalue": np.nan,
                "composition_resolution": "subclass" if has_composition else "none",
            }

            if found:
                intensities = quant.iloc[row_idx].values.astype(float)
                rec["mean_intensity"] = float(np.nanmean(intensities))

                # Correlation with snRNA-seq composition at subclass resolution
                if comp_by_pair:
                    prot_vals = []
                    comp_vals = []
                    for col_idx, cv in comp_by_pair:
                        pv = intensities[col_idx]
                        if not np.isnan(pv):
                            prot_vals.append(pv)
                            comp_vals.append(cv)
                    if len(prot_vals) >= 5:
                        r, p = stats.pearsonr(prot_vals, comp_vals)
                        rec["correlation_with_snrna_composition"] = float(r)
                        rec["correlation_pvalue"] = float(p)

            records.append(rec)

    df = pd.DataFrame(records)

    # Direction label based on correlation sign
    df["direction"] = np.where(
        df["correlation_with_snrna_composition"].isna(), "",
        np.where(df["correlation_with_snrna_composition"] > 0, "positive", "negative")
    )

    # Storey's q-value correction per cell type
    df["qvalue"] = np.nan
    df["pi0"] = np.nan
    for ct in sorted(cell_types):
        ct_mask = (df["cell_type"] == ct) & df["correlation_pvalue"].notna()
        pvals = df.loc[ct_mask, "correlation_pvalue"].values
        if len(pvals) < 10:
            continue
        # Storey's procedure: BH with estimated pi0
        # statsmodels doesn't have native Storey's, so we implement pi0 estimation
        # and apply it as an adjusted BH
        lambdas = np.arange(0.05, 0.95, 0.05)
        pi0_estimates = np.array([np.mean(pvals > lam) / (1 - lam) for lam in lambdas])
        # Natural cubic spline fit at lambda=max — use conservative estimate
        pi0_hat = min(float(pi0_estimates[-1]), 1.0)
        pi0_hat = max(pi0_hat, 1.0 / len(pvals))  # floor

        # BH procedure then scale by pi0
        n_tests = len(pvals)
        sorted_idx = np.argsort(pvals)
        sorted_pvals = pvals[sorted_idx]
        # q_i = min(p_i * n * pi0 / rank, q_{i+1})
        qvals = np.zeros(n_tests)
        ranks = np.arange(1, n_tests + 1)
        raw_q = sorted_pvals * n_tests * pi0_hat / ranks
        # Enforce monotonicity (from largest to smallest)
        qvals[sorted_idx[-1]] = min(raw_q[-1], 1.0)
        for i in range(n_tests - 2, -1, -1):
            qvals[sorted_idx[i]] = min(raw_q[i], qvals[sorted_idx[i + 1]])
        qvals = np.clip(qvals, 0, 1)

        df.loc[ct_mask, "qvalue"] = qvals
        df.loc[ct_mask, "pi0"] = pi0_hat
        print(f"  {ct}: π₀={pi0_hat:.3f} ({len(pvals)} tests)")

    # Add tissue category for researcher-level reporting
    df["tissue_category"] = df["cell_type"].map(config.SUBCLASS_TO_TISSUE_CATEGORY)

    out_path = os.path.join(OUTPUT_DIR, "datadriven_marker_assessment.csv")
    df.to_csv(out_path, index=False)
    n_with_comp = (df["composition_resolution"] == "subclass").sum()
    print(f"\n  Saved: {out_path} ({len(df)} rows, "
          f"{n_with_comp} with subclass-resolution composition)")

    # Summary: top markers per cell type
    for ct in sorted(cell_types):
        ct_df = df[df["cell_type"] == ct]
        n_total = len(ct_df)
        n_in_proteome = ct_df["found_in_total_proteome"].sum()
        has_q = ct_df["qvalue"].notna()
        sig_raw = ct_df[ct_df["correlation_pvalue"] < 0.05]
        sig_q = ct_df[ct_df["qvalue"] < 0.10]
        pi0 = ct_df["pi0"].dropna().iloc[0] if has_q.any() else np.nan

        comp_res = ct_df["composition_resolution"].iloc[0] if len(ct_df) > 0 else "none"
        tissue = ct_df["tissue_category"].iloc[0] if len(ct_df) > 0 else "?"
        print(f"\n  {ct} [{tissue}] (π₀={pi0:.3f}, composition={comp_res}):")
        print(f"    Expressed+specific: {n_total}, in proteome: {n_in_proteome}")
        print(f"    Raw p<0.05: {len(sig_raw)}, Storey q<0.10: {len(sig_q)}")

        if len(sig_q) > 0:
            n_pos = len(sig_q[sig_q["direction"] == "positive"])
            n_neg = len(sig_q[sig_q["direction"] == "negative"])
            print(f"    Direction: {n_pos} positive, {n_neg} negative")

            for direction in ["positive", "negative"]:
                dir_df = sig_q[sig_q["direction"] == direction].sort_values("qvalue")
                if len(dir_df) == 0:
                    continue
                print(f"    Top {direction} markers (by q-value):")
                for _, r in dir_df.head(5).iterrows():
                    print(f"      {r['gene_symbol']:12s} spec={r['specificity_score']:.3f}, "
                          f"r={r['correlation_with_snrna_composition']:+.3f}, "
                          f"p={r['correlation_pvalue']:.4f}, q={r['qvalue']:.4f}")

        # Show top-5 by specificity regardless
        top5 = ct_df.head(5)
        print(f"    Top-5 by specificity:")
        for _, r in top5.iterrows():
            found_str = "YES" if r["found_in_total_proteome"] else "no"
            corr_str = (f"r={r['correlation_with_snrna_composition']:+.3f}"
                       if not np.isnan(r["correlation_with_snrna_composition"])
                       else "N/A")
            q_str = (f"q={r['qvalue']:.3f}"
                    if not np.isnan(r["qvalue"]) else "")
            print(f"      {r['gene_symbol']:12s} spec={r['specificity_score']:.3f}, "
                  f"proteome={found_str}, corr={corr_str} {q_str}")


# ===========================================================================
# §4  Data Quality Assessment
# ===========================================================================


def step_quality():
    """Assess data quality: missingness, batch effects, PCA."""
    _ensure_output_dir()
    print("§4: Data quality assessment...")

    meta, quant, mapping = load_total_proteome()
    n_proteins, n_samples = quant.shape
    print(f"  Matrix: {n_proteins} proteins × {n_samples} samples")

    # 1. Missingness (report NA and zero separately)
    n_na = quant.isna().sum().sum()
    n_zero = ((quant == 0) & quant.notna()).sum().sum()
    total_cells = n_proteins * n_samples
    frac_na = n_na / total_cells
    frac_zero = n_zero / total_cells
    frac_missing = (n_na + n_zero) / total_cells
    print(f"\n  Missingness:")
    print(f"    NA: {n_na}/{total_cells} ({100*frac_na:.1f}%)")
    print(f"    Zero: {n_zero}/{total_cells} ({100*frac_zero:.1f}%)")
    print(f"    Total missing (NA + zero): {n_na + n_zero}/{total_cells} "
          f"({100*frac_missing:.1f}%)")

    # Combined mask for plex-level and structured missingness reporting
    is_missing = quant.isna() | (quant == 0)

    # Pre-compute plex->column-index mapping once
    col_to_idx = {col: i for i, col in enumerate(quant.columns)}
    plex_indices = {}
    for plex in sorted(mapping["plex"].unique()):
        plex_cols = mapping[mapping["plex"] == plex]["column_name"].tolist()
        plex_indices[int(plex)] = [col_to_idx[c] for c in plex_cols
                                   if c in col_to_idx]

    plex_missing = {}
    for plex, plex_idx in plex_indices.items():
        if plex_idx:
            plex_miss = is_missing.iloc[:, plex_idx].sum().sum()
            plex_total = n_proteins * len(plex_idx)
            plex_frac = plex_miss / plex_total
            plex_missing[plex] = {
                "n_samples": len(plex_idx),
                "missing_fraction": round(float(plex_frac), 4),
            }
            print(f"    Plex {plex}: {100*plex_frac:.1f}% missing "
                  f"({len(plex_idx)} samples)")

    is_missing_arr = is_missing.values
    n_plex_structured = 0
    for i in range(n_proteins):
        for plex_idx in plex_indices.values():
            if plex_idx and all(is_missing_arr[i, j] for j in plex_idx):
                n_plex_structured += 1
                break
    print(f"    Proteins missing from at least one full plex: "
          f"{n_plex_structured}")

    print(f"\n  Batch effects:")
    plex_medians = {}
    for plex, plex_idx in plex_indices.items():
        vals = quant.iloc[:, plex_idx].values.flatten()
        vals = vals[~np.isnan(vals)]
        vals = vals[vals > 0]
        med = float(np.median(vals)) if len(vals) > 0 else np.nan
        plex_medians[plex] = med
        print(f"    Plex {plex} median intensity: {med:.0f}")

    print(f"\n  Computing PCA...")
    mat = quant.values.astype(float).copy()
    mat[mat <= 0] = np.nan
    with np.errstate(divide="ignore"):
        log_mat = np.log2(mat)
    n_imputed = np.sum(~np.isfinite(log_mat))
    log_mat = config.minprob_impute(log_mat)
    print(f"  MinProb imputed {n_imputed} missing values for PCA")

    log_mat_centered = log_mat.copy()
    for plex_idx in plex_indices.values():
        if plex_idx:
            plex_median = np.median(log_mat_centered[:, plex_idx], axis=1,
                                     keepdims=True)
            log_mat_centered[:, plex_idx] -= plex_median

    # Transpose: samples × proteins
    X = log_mat_centered.T
    # Remove proteins with zero variance
    var = np.var(X, axis=0)
    X = X[:, var > 0]
    print(f"  Proteins with non-zero variance: {X.shape[1]}/{n_proteins}")

    pca = PCA(n_components=min(10, X.shape[0], X.shape[1]))
    scores = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_

    print(f"  Variance explained:")
    for i, v in enumerate(var_explained[:5]):
        print(f"    PC{i+1}: {100*v:.1f}%")

    # 4. PCA plots
    pca_dir = os.path.join(OUTPUT_DIR, "pca_plots")
    os.makedirs(pca_dir, exist_ok=True)

    color_factors = {
        "plex": mapping["plex"].astype(str).values,
        "genotype": mapping["genotype"].values,
        "sex": mapping["sex"].values,
        "timepoint": mapping["timepoint"].values,
    }

    for factor_name, factor_vals in color_factors.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_vals = sorted(set(factor_vals))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vals)))
        for i, val in enumerate(unique_vals):
            mask = np.array(factor_vals) == val
            ax.scatter(
                scores[mask, 0], scores[mask, 1],
                c=[colors[i]], label=val, s=40, alpha=0.7, edgecolors="k",
                linewidths=0.5,
            )
        ax.set_xlabel(
            f"PC1 ({100*var_explained[0]:.1f}% variance)"
        )
        ax.set_ylabel(
            f"PC2 ({100*var_explained[1]:.1f}% variance)"
        )
        ax.set_title(
            f"Total Proteome PCA — colored by {factor_name}"
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(
            os.path.join(pca_dir, f"pca_by_{factor_name}.png"), dpi=150
        )
        plt.close(fig)
        print(f"  Saved: pca_plots/pca_by_{factor_name}.png")

    # 5. Save quality summary
    quality = {
        "matrix_shape": {"proteins": n_proteins, "samples": n_samples},
        "missingness": {
            "na_fraction": round(float(frac_na), 4),
            "zero_fraction": round(float(frac_zero), 4),
            "total_fraction": round(float(frac_missing), 4),
            "by_plex": plex_missing,
            "proteins_missing_full_plex": n_plex_structured,
        },
        "batch_effects": {
            "plex_median_intensities": {
                str(k): round(v, 1) for k, v in plex_medians.items()
            },
        },
        "pca": {
            "variance_explained": [
                round(float(v), 4) for v in var_explained[:10]
            ],
        },
    }

    qual_path = os.path.join(OUTPUT_DIR, "data_quality.json")
    with open(qual_path, "w") as f:
        json.dump(quality, f, indent=2)
    print(f"\n  Saved: {qual_path}")


# ===========================================================================
# §5  Sample outlier detection
# ===========================================================================


def step_outliers():
    """§5: Detect statistical outliers using within-group z-scores.

    For each animal, computes the mean stoichiometry across all non-missing
    sites, then z-scores within its genotype×timepoint group (6 animals each).
    Animals with |z| > config.OUTLIER_ZSCORE_THRESH are flagged for exclusion.

    Requires kinase_attribution.py --normalize to have been run first (needs
    stoichiometry_matrix.csv).  Falls back to total-proteome PCA if unavailable.

    Outputs:
      sample_exclusions.csv — per-animal metrics and exclusion flags
      pca_plots/outlier_diagnostic.png — PCA + strip-plot diagnostic
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n=== §5: Sample Outlier Detection ===\n")

    # Load sample mapping
    sm_path = os.path.join(OUTPUT_DIR, "sample_mapping.csv")
    if not os.path.exists(sm_path):
        raise FileNotFoundError(
            f"{sm_path} not found. Run --mapping first.")
    mapping = pd.read_csv(sm_path)
    bio_cols = mapping["column_name"].tolist()

    # Try stoichiometry matrix first; fall back to total proteome
    stoich_path = os.path.join(
        config.KINASE_ATTRIBUTION_OUTPUT_DIR, "stoichiometry_matrix.csv")
    if os.path.exists(stoich_path):
        print(f"  Using stoichiometry matrix: {stoich_path}")
        mat_df = pd.read_csv(stoich_path)
        # Extract only biological sample columns
        vals = mat_df[bio_cols].values.astype(float)  # (sites, 72)
        source = "stoichiometry"
    else:
        print("  Stoichiometry matrix not found; falling back to total proteome")
        quant = pd.read_excel(
            config.TOTAL_PROTEOME_FILE, sheet_name="Protein quant")
        vals = quant[bio_cols].values.astype(float)
        vals[vals <= 0] = np.nan
        with np.errstate(divide="ignore"):
            vals = np.log2(vals)
        source = "total_proteome_log2"

    # Per-animal mean across all non-missing sites
    animal_means = np.nanmean(vals, axis=0)  # (72,)

    # Within-group robust z-scores (genotype × timepoint, 12 groups of 6)
    # Uses median and MAD instead of mean and std to prevent the outlier
    # from inflating its own group's spread (critical at n=6 per group).
    mapping["mean_stoich"] = animal_means
    mapping["group"] = mapping["genotype"] + "_" + mapping["timepoint"]

    def _robust_stats(s):
        med = s.median()
        mad = np.median(np.abs(s - med))
        # Scale MAD to be consistent with std for normal data
        mad_scaled = mad * 1.4826
        return pd.Series({"median": med, "mad_scaled": mad_scaled})

    group_stats = mapping.groupby("group")["mean_stoich"].apply(_robust_stats)
    group_stats = group_stats.unstack()
    mapping["group_mean"] = mapping["group"].map(group_stats["median"])
    mapping["group_std"] = mapping["group"].map(group_stats["mad_scaled"])
    # Guard against zero MAD (all values identical)
    mapping["z_score"] = np.where(
        mapping["group_std"] > 0,
        (mapping["mean_stoich"] - mapping["group_mean"]) / mapping["group_std"],
        0.0,
    )

    # Flag outliers
    thresh = config.OUTLIER_ZSCORE_THRESH
    mapping["excluded"] = mapping["z_score"].abs() > thresh
    mapping["reason"] = ""
    mapping.loc[mapping["excluded"], "reason"] = (
        f"within-group robust |z| > {thresh:.1f} on mean {source}"
    )

    n_excluded = mapping["excluded"].sum()
    print(f"  Threshold: |z| > {thresh}")
    print(f"  Flagged: {n_excluded} of {len(mapping)} animals")

    if n_excluded > 0:
        for _, row in mapping[mapping["excluded"]].iterrows():
            print(f"    {row['mouse_id']} ({row['genotype']} {row['timepoint']} "
                  f"{row['sex']}): z={row['z_score']:.2f}")

    # Save exclusions table
    out_cols = ["mouse_id", "animal_id", "sex", "timepoint", "genotype", "plex",
                "mean_stoich", "group_mean", "group_std", "z_score",
                "excluded", "reason"]
    # animal_id may be named differently — use column_name as fallback
    if "animal_id" not in mapping.columns:
        mapping["animal_id"] = mapping["column_name"]
    excl_path = os.path.join(OUTPUT_DIR, "sample_exclusions.csv")
    mapping[out_cols].to_csv(excl_path, index=False)
    print(f"\n  Saved: {excl_path}")

    # --- Diagnostic plot ---
    pca_dir = os.path.join(OUTPUT_DIR, "pca_plots")
    os.makedirs(pca_dir, exist_ok=True)

    # PCA for scatter panel (reuse total proteome approach)
    pca_vals = vals.copy()
    pca_vals[~np.isfinite(pca_vals)] = np.nan
    pca_vals = config.minprob_impute(pca_vals)
    X_pca = pca_vals.T  # (72, sites)
    var = np.var(X_pca, axis=0)
    X_pca = X_pca[:, var > 0]
    pca = PCA(n_components=min(5, X_pca.shape[0], X_pca.shape[1]))
    scores = pca.fit_transform(X_pca)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: PCA scatter with outliers highlighted
    ax = axes[0]
    outlier_mask = mapping["excluded"].values
    ax.scatter(scores[~outlier_mask, 0], scores[~outlier_mask, 1],
               c="steelblue", s=40, alpha=0.6, edgecolors="k", linewidths=0.5,
               label="Retained")
    if outlier_mask.any():
        ax.scatter(scores[outlier_mask, 0], scores[outlier_mask, 1],
                   c="red", s=80, marker="X", edgecolors="k", linewidths=0.5,
                   label="Excluded", zorder=5)
        for idx in np.where(outlier_mask)[0]:
            ax.annotate(mapping.iloc[idx]["mouse_id"],
                        (scores[idx, 0], scores[idx, 1]),
                        fontsize=8, ha="left", va="bottom")
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    ax.set_title("PCA — Outlier Detection")
    ax.legend()

    # Panel 2: within-group z-score strip plot
    ax = axes[1]
    groups = sorted(mapping["group"].unique())
    for i, grp in enumerate(groups):
        grp_mask = mapping["group"] == grp
        z_vals = mapping.loc[grp_mask, "z_score"].values
        excl = mapping.loc[grp_mask, "excluded"].values
        ax.scatter(np.full_like(z_vals, i), z_vals,
                   c=["red" if e else "steelblue" for e in excl],
                   s=40, edgecolors="k", linewidths=0.5, zorder=3)
    ax.axhline(thresh, color="red", linestyle="--", alpha=0.5, label=f"±{thresh}")
    ax.axhline(-thresh, color="red", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Within-group z-score")
    ax.set_title("Per-Animal Z-Scores by Group")
    ax.legend()

    fig.tight_layout()
    diag_path = os.path.join(pca_dir, "outlier_diagnostic.png")
    fig.savefig(diag_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {diag_path}")


# ===========================================================================
# Summary
# ===========================================================================


def print_summary():
    """Print cached results from all steps."""
    print("=" * 70)
    print("Data Ingestion — Summary")
    print("=" * 70)

    # Sample mapping
    sm_path = os.path.join(OUTPUT_DIR, "sample_mapping.csv")
    if os.path.exists(sm_path):
        sm = pd.read_csv(sm_path)
        print(f"\n§1 Sample Mapping:")
        print(f"  Animals: {len(sm)}")
        print(f"  With snRNA-seq: {sm['has_snrna_seq'].sum()}")
        print(f"  Plexes: {sm['plex'].nunique()}")
        print(f"  Unique groups: {sm['phospho_group_id'].nunique()}")
    else:
        print(f"\n§1 Sample Mapping: not run yet")

    # Phospho matching
    ms_path = os.path.join(OUTPUT_DIR, "matching_summary.json")
    if os.path.exists(ms_path):
        with open(ms_path) as f:
            ms = json.load(f)
        print(f"\n§2 Phosphosite-to-Protein Matching:")
        im = ms.get("imac_sitequant", ms.get("imac_median", {}))
        print(f"  IMAC sitequant: {im.get('matched_sites', '?')}/"
              f"{im.get('total_sites', '?')} sites matched "
              f"({im.get('matched_sites_pct', '?')}%)")
        print(f"  Parent proteins: {im.get('matched_proteins', '?')}/"
              f"{im.get('unique_parent_proteins', '?')} matched "
              f"({im.get('matched_proteins_pct', '?')}%)")
        if "imac_composite" in ms:
            ic = ms["imac_composite"]
            print(f"  Composite sites: {ic.get('matched_sites', '?')}/"
                  f"{ic.get('total_sites', '?')} matched "
                  f"({ic.get('matched_sites_pct', '?')}%)")
        if "imac_sitequant" in ms:
            sq = ms["imac_sitequant"]
            print(f"  Site-level quant: {sq.get('matched_sites', '?')}/"
                  f"{sq.get('total_sites', '?')} matched "
                  f"({sq.get('matched_sites_pct', '?')}%)")
    else:
        print(f"\n§2 Phosphosite-to-Protein Matching: not run yet")

    # Markers (data-driven, WMB atlas)
    mk_path = os.path.join(OUTPUT_DIR, "datadriven_marker_assessment.csv")
    if os.path.exists(mk_path):
        mk = pd.read_csv(mk_path)
        print(f"\n§3 Cell-Type Marker Proteins (WMB atlas, data-driven):")
        for ct in sorted(mk["cell_type"].unique()):
            ct_df = mk[mk["cell_type"] == ct]
            n_in_proteome = ct_df["found_in_total_proteome"].sum()
            sig_q = ct_df[ct_df["qvalue"] < 0.10]
            print(f"  {ct}: {len(ct_df)} genes, {n_in_proteome} in proteome, "
                  f"{len(sig_q)} significant (q<0.10)")
    else:
        print(f"\n§3 Cell-Type Marker Proteins: not run yet")

    # Data quality
    dq_path = os.path.join(OUTPUT_DIR, "data_quality.json")
    if os.path.exists(dq_path):
        with open(dq_path) as f:
            dq = json.load(f)
        print(f"\n§4 Data Quality:")
        m = dq.get("missingness", {})
        print(f"  Missingness: {100*m.get('total_fraction', 0):.1f}%")
        pca_ve = dq.get("pca", {}).get("variance_explained", [])
        if pca_ve:
            print(f"  PCA variance: PC1={100*pca_ve[0]:.1f}%, "
                  f"PC2={100*pca_ve[1]:.1f}%")
        be = dq.get("batch_effects", {}).get("plex_median_intensities", {})
        if be:
            vals = list(be.values())
            print(f"  Plex median range: {min(vals):.0f} – {max(vals):.0f}")
    else:
        print(f"\n§4 Data Quality: not run yet")

    print()


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Data ingestion and characterization for the Song proteomics pipeline",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--mapping", action="store_true",
        help="§1: TMT channel-to-animal sample mapping",
    )
    group.add_argument(
        "--phospho-match", action="store_true",
        help="§2: Phosphosite-to-protein matching",
    )
    group.add_argument(
        "--markers", action="store_true",
        help="§3: Cell-type marker protein assessment (WMB atlas, data-driven)",
    )
    group.add_argument(
        "--quality", action="store_true",
        help="§4: Data quality assessment (PCA, batch effects, missingness)",
    )
    group.add_argument(
        "--outliers", action="store_true",
        help="§5: Statistical outlier detection (within-group z-scores)",
    )
    group.add_argument(
        "--run", action="store_true",
        help="Run all steps in order",
    )
    group.add_argument(
        "--summary", action="store_true",
        help="Print cached results",
    )

    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if args.mapping or args.run:
        step_sample_mapping()

    if args.phospho_match or args.run:
        step_phospho_match()

    if args.markers or args.run:
        step_markers()

    if args.quality or args.run:
        step_quality()

    if args.outliers or args.run:
        step_outliers()

    if args.run:
        print("\n" + "=" * 70)
        print_summary()


if __name__ == "__main__":
    main()
