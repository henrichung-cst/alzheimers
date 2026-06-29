"""Song (mouse AD) unified-viewer slice builders."""

from __future__ import annotations

import glob
import gzip
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alz.shared import config
from alz.viewer.paths import (
    DECOMP_OLS_PARQUET,
    EDGE_SLICES_DECOMP_OLS_DIR,
    EDGE_SLICES_INCYTR_PATHWAYS_DIR,
    EDGE_SLICES_SONG_CONCORDANCE_DIR,
    INCYTR_PAIR_MODE_OUTPUTS_DIR,
    KINASE_INCYTR_BRIDGE_DIR,
    SCHEMA_VERSION,
    UNIFIED_VIEWER_DIR,
)
from alz.viewer.shared.incytr_index import (
    _INCYTR_FC_COLS,
    _INCYTR_FC_METRICS,
    _INCYTR_GENE_NODE_INDEX_FILENAME,
    _INCYTR_INDEX_FILENAME,
    _INCYTR_LABEL_COLS,
    _INCYTR_LABEL_NODES,
    _INCYTR_LABEL_SRC,
    _INCYTR_LABEL_VOCAB,
    _INCYTR_PATHWAY_ABS_PDS,
    _INCYTR_PATHWAY_PVALUES,
    _INCYTR_SCORE_COLS,
    _SIGN_VEC_LABELS,
    _idx_label_bits,
    _idx_traj_bits,
)
from alz.viewer.shared.cohort_slice import CohortViewerSlice, EdgeSliceContribution
from alz.viewer.shared.payload_helpers import (
    _INCYTR_FC_NODES,
    _build_incytr_gene_node_index,
    _configure_duckdb_tempdir,
    _sanitize,
    _write_gene_node_index_shard,
)

if TYPE_CHECKING:
    from alz.build_unified_viewer import UnifiedData

TISSUE_CATEGORIES = {
    "Excitatory": ["01 IT-ET Glut", "02 NP-CT-L6b Glut", "03 OB-CR Glut",
                   "04 DG-IMN Glut"],
    "Inhibitory": ["05 OB-IMN GABA", "06 CTX-CGE GABA", "07 CTX-MGE GABA",
                   "08 CNU-MGE GABA", "09 CNU-LGE GABA"],
    "Subcortical": ["10 LSX GABA", "11 CNU-HYa GABA", "12 HY GABA",
                    "13 CNU-HYa Glut", "14 HY Glut", "15 HY Gnrh1 Glut",
                    "16 HY MM Glut", "17 MH-LH Glut", "18 TH Glut"],
    "Brainstem": ["19 MB Glut", "20 MB GABA", "21 MB Dopa", "22 MB-HB Sero",
                  "23 P Glut", "24 MY Glut", "25 Pineal Glut",
                  "26 P GABA", "27 MY GABA"],
    "Cerebellum": ["28 CB GABA", "29 CB Glut"],
    "Non-neuronal": ["30 Astro-Epen", "31 OPC-Oligo", "32 OEC",
                     "33 Vascular", "34 Immune"],
}
RECEIVER_TO_TISSUE = {r: t for t, rs in TISSUE_CATEGORIES.items() for r in rs}


from alz.viewer.shared.build_cache import (  # noqa: E402
    _BUILD_CACHE_SCHEMA_VERSION,
    _VIEWER_BUILD_CACHE_DIR,
    _sha256_file,
    _file_fingerprint,
    _input_signature,
    _build_cache_path,
    _load_build_cache,
    _write_build_cache,
)

_SONG_MECHANISM_COLUMNS = [
    "cohort",
    "track",
    "contrast",
    "kinase",
    "stoich_NES",
    "stoich_FDR",
    "raw_NES",
    "raw_FDR",
    "stoich_significant",
    "raw_significant",
    "sign_relation",
    "mechanism_call",
    "skip_reason",
]


def _build_kinases_slice(data: UnifiedData, secretome_map: dict | None = None) -> dict:
    """Columnar kinases table. IDs follow edge_metadata['kinases'] ordering."""
    kinases = data.edge_metadata["kinases"]
    kid = {k: i for i, k in enumerate(kinases)}

    ka = data.kinase_activity.set_index("kinase")
    hyp = data.kinase_hypothesis.set_index("kinase")
    contrasts = data.edge_metadata["contrasts"]

    # Pair-mode backbone/path participation from B4's bridge (kinase_participation.csv):
    # n_backbones = distinct (Sender,Receiver,Receptor,EM) spines the kinase acts on;
    # n_paths = distinct full pathways. Kinases absent from the table get NULL counts.
    part_path = os.path.join(KINASE_INCYTR_BRIDGE_DIR, "song", "kinase_participation.csv")
    n_backbones_by_kin: dict[str, int] = {}
    n_paths_by_kin: dict[str, int] = {}
    if os.path.exists(part_path):
        part = pd.read_csv(part_path)
        n_backbones_by_kin = dict(zip(part["kinase"], part["n_backbones"]))
        n_paths_by_kin = dict(zip(part["kinase"], part["n_paths"]))

    cols: dict[str, list] = {
        "id": [], "name": [], "gene_symbol": [],
        "residue_type": [],
        "secretome_location": [],
        "top_celltype_1": [], "top_celltype_2": [], "top_celltype_3": [],
        "top_celltype_1_sea_ad_lfc": [],
        "top_celltype_1_song_lfc": [],
        "n_celltype_candidates": [],
        "n_backbones": [], "n_paths": [],
    }
    # Per-genotype scalars: peak_NES_{g}, peak_contrast_{g}, n_sig_{g}, trajectory_{g}
    for g in config.DISEASE_GROUPS:
        cols[f"peak_NES_{g}"] = []
        cols[f"peak_contrast_{g}"] = []
        cols[f"n_sig_{g}"] = []
        cols[f"trajectory_{g}"] = []
    for c in contrasts:
        cols[f"NES_{c}"] = []
        cols[f"FDR_{c}"] = []

    for k in kinases:
        cols["id"].append(kid[k])
        cols["name"].append(k)
        ka_row = ka.loc[k] if k in ka.index else None
        hyp_row = hyp.loc[k] if k in hyp.index else None

        def _get(r, col, default=None):
            if r is None or col not in r.index:
                return default
            v = r[col]
            return default if pd.isna(v) else v

        cols["gene_symbol"].append(_get(ka_row, "gene_symbol", ""))
        cols["residue_type"].append(_get(ka_row, "residue_type", "ST"))
        gene_up = (cols["gene_symbol"][-1] or "").upper()
        cols["secretome_location"].append(secretome_map.get(gene_up, "") if secretome_map else "")
        for g in config.DISEASE_GROUPS:
            cols[f"peak_NES_{g}"].append(_get(ka_row, f"peak_NES_{g}"))
            cols[f"peak_contrast_{g}"].append(_get(ka_row, f"peak_contrast_{g}", ""))
            cols[f"n_sig_{g}"].append(_get(ka_row, f"n_sig_{g}", 0))
            cols[f"trajectory_{g}"].append(_get(ka_row, f"trajectory_{g}", ""))
        cols["top_celltype_1"].append(_get(hyp_row, "top_celltype_1", ""))
        cols["top_celltype_2"].append(_get(hyp_row, "top_celltype_2", ""))
        cols["top_celltype_3"].append(_get(hyp_row, "top_celltype_3", ""))
        cols["top_celltype_1_sea_ad_lfc"].append(_get(hyp_row, "top_celltype_1_sea_ad_lfc"))
        cols["top_celltype_1_song_lfc"].append(_get(hyp_row, "top_celltype_1_song_lfc"))
        cols["n_celltype_candidates"].append(_get(hyp_row, "n_celltype_candidates", 0))
        nb = n_backbones_by_kin.get(k)
        npaths = n_paths_by_kin.get(k)
        cols["n_backbones"].append(int(nb) if pd.notna(nb) else None)
        cols["n_paths"].append(int(npaths) if pd.notna(npaths) else None)
        for c in contrasts:
            cols[f"NES_{c}"].append(_get(ka_row, f"{c}_NES"))
            cols[f"FDR_{c}"].append(_get(ka_row, f"{c}_FDR"))
    return cols

def _build_celltypes_slice(data: UnifiedData) -> dict:
    celltypes = data.edge_metadata["celltypes"]
    return {
        "id": list(range(len(celltypes))),
        "name": list(celltypes),
        "tissue_category": [RECEIVER_TO_TISSUE.get(c, "Other") for c in celltypes],
    }

def _as_single_context_block(block: dict | None, context_id: str) -> dict | None:
    """Wrap a single-context block in the schema-v2 by_context shape."""
    if block is None:
        return None
    return {"by_context": {context_id: block}}

def _build_subclass_breakdown(kid: dict[str, int]) -> dict:
    """Per-kinase subclass composition tooltips for verdict-table rows.

    For each (kinase, WMB class) where the class spans ≥2 WMB subclasses with
    detectable expression, returns the top-3 contributing subclasses ranked by
    mean log2 expression. Lets a viewer user see what subclass-level structure
    is collapsed behind a class-level call (e.g., "07 CTX-MGE GABA" → Pvalb +
    Sst + Chandelier).
    """
    sub_path = config.WMB_EXPRESSION_SUBCLASS_FILE
    map_path = config.WMB_SUBCLASS_TO_CLASS_FILE
    if not (os.path.exists(sub_path) and os.path.exists(map_path)):
        print(f"  subclass_breakdown: skipped (missing {sub_path} or {map_path})",
              flush=True)
        return {}
    sub = pd.read_csv(sub_path)
    sc2cls = pd.read_csv(map_path)
    sub = sub.merge(sc2cls, left_on="wmb_subclass", right_on="subclass", how="left")
    sub = sub[sub["class"].notna() & sub["kinase_id"].isin(kid)]
    # Keep only subclasses with detectable expression
    sub = sub[sub["mean_log2_expression"] > 0.5]
    if len(sub) == 0:
        return {}
    sub = sub.sort_values(["kinase_id", "class", "mean_log2_expression"],
                           ascending=[True, True, False])
    out: dict[str, dict[str, str]] = {}
    for (kin, cls), g in sub.groupby(["kinase_id", "class"], sort=False):
        if len(g) < 2:
            continue
        top = g.head(3)
        parts = [
            f"{r['wmb_subclass']} (mean={r['mean_log2_expression']:.2f}, "
            f"frac={r['fraction_cells_expressing']:.2f})"
            for _, r in top.iterrows()
        ]
        n_more = len(g) - len(top)
        text = "; ".join(parts)
        if n_more > 0:
            text += f"; +{n_more} more"
        out.setdefault(str(kid[kin]), {})[str(cls)] = text
    print(f"  subclass_breakdown: {len(out)} kinases × "
          f"{sum(len(v) for v in out.values())} (kinase,class) tooltips",
          flush=True)
    return out

_AGREEMENT_STATE_CODES = {
    "neither_sig": 0,
    "agree":       1,
    "mixed":       2,
    "disagree":    3,
    "bulk_only":   4,
    "decomp_only": 5,
}


def _build_agreement_index(
    mea: pd.DataFrame,
    decomp: pd.DataFrame,
    kid: dict,
    contrast_to_id: dict,
    fdr_thresh: float,
) -> dict:
    """Per-(kinase, contrast) agreement state between bulk MEA and per-cell decomp MEA.

    For each (kinase, contrast) where bulk and/or decomp data exist, classify:
      - agree:       bulk sig, ≥1 cell sig, all sig cells match bulk sign
      - mixed:       bulk sig, sig cells split (some match, some oppose)
      - disagree:    bulk sig, sig cells all oppose bulk sign
      - bulk_only:   bulk sig, no cell sig
      - decomp_only: bulk insig, ≥1 cell sig
      - neither_sig: bulk insig, no cell sig (NOT emitted; absence == this state)

    Also reports the top decomp cell (largest |NES| among that kinase×contrast's
    decomp rows) for the scatter plot.
    """
    if mea.empty or decomp.empty:
        return {"kinase_id": [], "contrast_id": [], "state": [],
                "bulk_nes": [], "bulk_fdr": [],
                "top_cell": [], "top_cell_nes": [], "top_cell_fdr": [],
                "n_cells_match": [], "n_cells_oppose": []}

    b = mea[mea["kinase"].isin(kid) & mea["contrast"].isin(contrast_to_id)].copy()
    b = b.rename(columns={"NES": "bulk_NES", "FDR": "bulk_FDR"})
    b["bulk_sig"] = b["bulk_FDR"] < fdr_thresh
    b["bulk_dir"] = np.sign(b["bulk_NES"])

    d = decomp[decomp["kinase"].isin(kid) & decomp["contrast"].isin(contrast_to_id)].copy()
    d["dec_sig"] = d["FDR"] < fdr_thresh
    d["dec_dir"] = np.sign(d["NES"])
    d["abs_nes"] = d["NES"].abs()

    # Outer join so bulk-only and decomp-only rows are kept.
    m = d.merge(b[["kinase", "contrast", "bulk_NES", "bulk_FDR", "bulk_sig", "bulk_dir"]],
                on=["kinase", "contrast"], how="outer")
    # Decomp side may be NaN for bulk-only (kinase, contrast); fill safe defaults.
    m["dec_sig"] = m["dec_sig"].fillna(False)
    m["dec_dir"] = m["dec_dir"].fillna(0)
    m["bulk_sig"] = m["bulk_sig"].fillna(False)
    m["bulk_dir"] = m["bulk_dir"].fillna(0)

    # For sign comparisons, only count cells where bulk_dir != 0.
    m["match"] = m["dec_sig"] & (m["bulk_dir"] != 0) & (m["dec_dir"] == m["bulk_dir"])
    m["oppose"] = m["dec_sig"] & (m["bulk_dir"] != 0) & (m["dec_dir"] == -m["bulk_dir"])

    # Top decomp cell by |NES| per (kinase, contrast).
    has_dec = m["wmb_class"].notna()
    top_idx = m[has_dec].groupby(["kinase", "contrast"])["abs_nes"].idxmax()
    top = m.loc[top_idx, ["kinase", "contrast", "wmb_class", "NES", "FDR"]].rename(
        columns={"wmb_class": "top_cell", "NES": "top_cell_nes", "FDR": "top_cell_fdr"}
    )

    agg = m.groupby(["kinase", "contrast"]).agg(
        bulk_NES=("bulk_NES", "first"),
        bulk_FDR=("bulk_FDR", "first"),
        bulk_sig=("bulk_sig", "first"),
        n_match=("match", "sum"),
        n_oppose=("oppose", "sum"),
        n_dec_sig=("dec_sig", "sum"),
    ).reset_index()
    agg = agg.merge(top, on=["kinase", "contrast"], how="left")

    def _state(r):
        if r.bulk_sig:
            if r.n_dec_sig == 0:
                return "bulk_only"
            if r.n_match > 0 and r.n_oppose == 0:
                return "agree"
            if r.n_match == 0 and r.n_oppose > 0:
                return "disagree"
            return "mixed"
        if r.n_dec_sig > 0:
            return "decomp_only"
        return "neither_sig"

    agg["state"] = agg.apply(_state, axis=1)
    # Drop neither_sig — absence in lookup table == that state.
    agg = agg[agg["state"] != "neither_sig"].reset_index(drop=True)

    print(f"  agreement_index: {len(agg):,} (kinase, contrast) cells "
          f"(states: {agg['state'].value_counts().to_dict()})", flush=True)

    state_codes = agg["state"].map(_AGREEMENT_STATE_CODES).astype("uint8").tolist()
    return {
        "kinase_id":   agg["kinase"].map(kid).astype("uint16").tolist(),
        "contrast_id": agg["contrast"].map(contrast_to_id).astype("uint8").tolist(),
        "state":       state_codes,
        "bulk_nes":    agg["bulk_NES"].astype(float).round(4).tolist(),
        "bulk_fdr":    agg["bulk_FDR"].astype(float).round(4).tolist(),
        "top_cell":    agg["top_cell"].fillna("").astype(str).tolist(),
        "top_cell_nes": agg["top_cell_nes"].astype(float).round(4).tolist(),
        "top_cell_fdr": agg["top_cell_fdr"].astype(float).round(4).tolist(),
        "n_cells_match":  agg["n_match"].astype(int).tolist(),
        "n_cells_oppose": agg["n_oppose"].astype(int).tolist(),
        "_state_codes": _AGREEMENT_STATE_CODES,
    }

def _norm_motif(s: str) -> str:
    return str(s or "").strip("_").upper()

def _to_float32_estimable(s: pd.Series) -> np.ndarray:
    """Cast to float32, mapping finite values outside float32 range to NaN.

    Degenerate OLS fits (near-singular design) yield astronomically large
    standard errors (|se| up to ~2e41). A blind float32 cast overflows these to
    +/-inf (RuntimeWarning) and misrepresents an unidentifiable coefficient as a
    finite-but-huge bound. NaN ("not estimable") is the honest representation and
    silences the spurious warning. Values already within range cast unchanged.
    """
    a = s.to_numpy(dtype="float64", copy=True)
    a[np.isfinite(a) & (np.abs(a) > np.finfo(np.float32).max)] = np.nan
    return a.astype("float32")

def _write_decomp_ols_slices(kid: dict, contrast_to_id: dict) -> dict:
    """Per-kinase shard of per-cell-type OLS at substrate sites.

    Reads `outputs/reports/decomposition/{spine}/per_animal/site_level_ols.parquet`
    (per-cell_type × per-contrast × per-track site rows in Levy-t5
    vocabulary), filters each kinase to its substrate-set motifs (across
    all contrasts/tracks), and writes one parquet per kinase to
    `edge_slices/decomp_ols/{kid:03d}.parquet`.

    The drawer in the Attribution tab fetches one shard on demand and
    filters client-side by current contrast + cell_type to populate the
    substrate-level evidence table for the per-cell pseudo-deconv NES.
    """
    if not os.path.exists(DECOMP_OLS_PARQUET):
        print(f"  (warn) decomp OLS parquet missing: {DECOMP_OLS_PARQUET}; "
              f"skipping decomp_ols slice generation", flush=True)
        return {"slice_count": 0, "present_kinase_ids": [], "filename_template":
                "{kinase_id:03d}.parquet"}

    # Substrate sets — st + py tracks. Both files share schema.
    ss_paths = [
        os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "mea_substrate_sets.csv"),
        os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "mea_substrate_sets_pY.csv"),
    ]
    cols = ["site_id", "gene_symbol", "motif", "cell_type",
            "contrast", "lfc", "se", "pval", "track"]
    signature = _input_signature(
        "decomp_ols",
        [__file__, DECOMP_OLS_PARQUET, *ss_paths],
        {
            "builder_version": 1,
            "schema_version": SCHEMA_VERSION,
            "spine": config.CLUSTER_SPINE_NAME,
            "kinases": sorted((str(k), int(v)) for k, v in kid.items()),
            "contrast_to_id": sorted((str(k), int(v)) for k, v in contrast_to_id.items()),
            "source_columns": cols,
        },
    )
    cached = _load_build_cache(
        "decomp_ols", signature, EDGE_SLICES_DECOMP_OLS_DIR,
    )
    if cached is not None:
        return cached

    # Wipe-and-recreate so an aborted previous run can't leave partial
    # shards alongside the new ones (mismatched slice_count vs.
    # present_kinase_ids in index.json).
    shutil.rmtree(EDGE_SLICES_DECOMP_OLS_DIR, ignore_errors=True)
    os.makedirs(EDGE_SLICES_DECOMP_OLS_DIR, exist_ok=True)

    ss_frames = []
    for p in ss_paths:
        if os.path.exists(p):
            ss_frames.append(pd.read_csv(p, usecols=["kinase", "motif", "track"]))
    if not ss_frames:
        print(f"  (warn) substrate-set tables not found under "
              f"{config.KINASE_ATTRIBUTION_OUTPUT_DIR}; "
              f"skipping decomp_ols slice generation", flush=True)
        return {"slice_count": 0, "present_kinase_ids": [], "filename_template":
                "{kinase_id:03d}.parquet"}
    ss = pd.concat(ss_frames, ignore_index=True)
    ss["motif_norm"] = ss["motif"].map(_norm_motif)
    ss = ss[ss["kinase"].isin(kid)]
    # kinase -> set of (motif_norm, track) substrate keys
    kinase_subs: dict[str, set] = {}
    for k, g in ss.groupby("kinase"):
        kinase_subs[k] = set(zip(g["motif_norm"], g["track"]))
    print(f"  decomp_ols: {len(kinase_subs)} kinases with substrate sets", flush=True)

    print(f"  decomp_ols: loading {DECOMP_OLS_PARQUET} "
          f"({os.path.getsize(DECOMP_OLS_PARQUET) / 1e6:.1f} MB)", flush=True)
    pcdf = pq.read_table(DECOMP_OLS_PARQUET, columns=cols).to_pandas()
    pcdf = pcdf[pcdf["contrast"].isin(contrast_to_id)].copy()
    pcdf["motif_norm"] = pcdf["motif"].astype(str).map(_norm_motif)
    pcdf["contrast_id"] = pcdf["contrast"].map(contrast_to_id).astype("uint8")
    pcdf = pcdf.drop(columns=["contrast"])
    print(f"  decomp_ols: {len(pcdf):,} per-cell rows after contrast filter", flush=True)

    # Index by (motif_norm, track) for fast per-kinase slicing.
    pc_index = pcdf.set_index(["motif_norm", "track"], drop=False).sort_index()

    template = "{kinase_id:03d}.parquet"
    present = []
    total_rows = 0
    for k, kid_int in kid.items():
        keys = kinase_subs.get(k)
        if not keys:
            continue
        # Build a small DataFrame of (motif_norm, track) selectors and join.
        sel_keys = list(keys)
        try:
            sub = pc_index.loc[sel_keys]
        except KeyError:
            sub = pc_index.loc[pc_index.index.intersection(sel_keys)]
        if isinstance(sub, pd.Series):
            continue
        if sub.empty:
            continue
        out = sub.reset_index(drop=True)[
            ["contrast_id", "cell_type", "site_id", "gene_symbol",
             "motif", "lfc", "se", "pval", "track"]
        ].copy()
        out["lfc"] = _to_float32_estimable(out["lfc"])
        out["se"] = _to_float32_estimable(out["se"])
        out["pval"] = _to_float32_estimable(out["pval"])
        path = os.path.join(EDGE_SLICES_DECOMP_OLS_DIR,
                            template.format(kinase_id=int(kid_int)))
        pq.write_table(pa.Table.from_pandas(out, preserve_index=False), path,
                       compression="zstd")
        present.append(int(kid_int))
        total_rows += len(out)

    present.sort()
    index = {
        "schema_version": SCHEMA_VERSION,
        "slice_count": len(present),
        "present_kinase_ids": present,
        "filename_template": template,
        "n_total_rows": total_rows,
    }
    with open(os.path.join(EDGE_SLICES_DECOMP_OLS_DIR, "index.json"), "w") as f:
        json.dump(index, f)
    output_files = [
        "index.json",
        *[
            template.format(kinase_id=int(kid_int))
            for kid_int in present
        ],
    ]
    _write_build_cache(
        "decomp_ols",
        signature,
        EDGE_SLICES_DECOMP_OLS_DIR,
        output_files,
        index,
    )
    print(f"  decomp_ols: wrote {len(present)} shards "
          f"({total_rows:,} total rows)", flush=True)
    return index

def _write_song_concordance_slices(genes_of_interest: set[str]) -> dict:
    """Per-gene shards of `song_concordance.csv`.

    The full file is ~210 MB; the Attribution drawer only ever filters to a
    single gene, so the JS fetches one shard on demand instead.
    """
    src = config.SONG_CONCORDANCE_FILE
    if not os.path.exists(src):
        print(f"  (warn) song_concordance source missing: {src}; skipping",
              flush=True)
        return {"slice_count": 0, "present_genes": [],
                "filename_template": "{gene}.parquet"}

    cols = ["gene_symbol", "cell_type", "contrast",
            "song_lfc", "song_se", "song_pval", "song_fdr", "n_animals"]
    gset = {str(g).upper() for g in genes_of_interest if g}
    signature = _input_signature(
        "song_concordance",
        [__file__, src],
        {
            "builder_version": 1,
            "schema_version": SCHEMA_VERSION,
            "genes_of_interest": sorted(gset),
            "source_columns": cols,
        },
    )
    cached = _load_build_cache(
        "song_concordance", signature, EDGE_SLICES_SONG_CONCORDANCE_DIR,
    )
    if cached is not None:
        return cached

    shutil.rmtree(EDGE_SLICES_SONG_CONCORDANCE_DIR, ignore_errors=True)
    os.makedirs(EDGE_SLICES_SONG_CONCORDANCE_DIR, exist_ok=True)

    print(f"  song_concordance: loading {src} "
          f"({os.path.getsize(src) / 1e6:.1f} MB)", flush=True)
    df = pd.read_csv(src, usecols=lambda c: c in cols)
    df["gene_upper"] = df["gene_symbol"].astype(str).str.upper()
    if gset:
        df = df[df["gene_upper"].isin(gset)]
    print(f"  song_concordance: {len(df):,} rows after gene filter "
          f"({df['gene_upper'].nunique()} genes)", flush=True)

    float_cols = [c for c in ("song_lfc", "song_se", "song_pval", "song_fdr")
                  if c in df.columns]
    has_n_animals = "n_animals" in df.columns
    # Skip names that would need URL escaping on the client.
    def _safe(g: str) -> bool:
        return bool(g) and all(c.isalnum() or c in ("-", "_") for c in g)

    present = []
    total_rows = 0
    template = "{gene}.parquet"
    for gene_upper, g in df.groupby("gene_upper", sort=False):
        if not _safe(gene_upper):
            continue
        out = g.drop(columns=["gene_upper"])
        for c in float_cols:
            out[c] = out[c].astype("float32")
        if has_n_animals:
            out["n_animals"] = out["n_animals"].fillna(0).astype("int16")
        path = os.path.join(EDGE_SLICES_SONG_CONCORDANCE_DIR,
                            template.format(gene=gene_upper))
        pq.write_table(pa.Table.from_pandas(out, preserve_index=False), path,
                       compression="zstd")
        present.append(gene_upper)
        total_rows += len(out)

    present.sort()
    index = {
        "schema_version": SCHEMA_VERSION,
        "slice_count": len(present),
        "present_genes": present,
        "filename_template": template,
        "n_total_rows": total_rows,
    }
    with open(os.path.join(EDGE_SLICES_SONG_CONCORDANCE_DIR, "index.json"), "w") as f:
        json.dump(index, f)
    output_files = [
        "index.json",
        *[template.format(gene=g) for g in present],
    ]
    _write_build_cache(
        "song_concordance",
        signature,
        EDGE_SLICES_SONG_CONCORDANCE_DIR,
        output_files,
        index,
    )
    print(f"  song_concordance: wrote {len(present)} shards "
          f"({total_rows:,} total rows)", flush=True)
    return index

_INCYTR_CONTRASTS = tuple(config.CONTRAST_COEFS.keys())

# ---------------------------------------------------------------------------
# CR-04: trajectory_index + recur_index computation
# ---------------------------------------------------------------------------
# Trajectory is a property of the path's raw PDS sign vector — NOT of any
# significance gate. A (path, disease) earns a label only when the Incytr
# pipeline produced a row at all three timepoints (2/4/6 mo); incomplete
# paths get no label (rendered as "—" in the viewer). Sign chars are 'u'
# (PDS > 0) or 'd' (PDS < 0) — no 'f', no hardcoded flat threshold. The
# viewer's pvalue/|PDS| sliders filter visible rows but do not redefine
# the trajectory.
_TRAJ_TIMEPOINTS = ("2mo", "4mo", "6mo")
_TRAJ_VALID_DISEASES = {"App", "Tau", "ApTt"}


def _annotate_trajectory_columns(
    df: "pd.DataFrame",
    source_label: str = "factorial",
) -> "tuple[pd.DataFrame, dict, dict]":
    """Add traj_labels and sign_vec columns to the long-form shard DataFrame.

    Delegates to the shared ``annotate_trajectory_columns`` helper parameterised
    on Song/AD's timepoints (2mo, 4mo, 6mo) and disease labels (App, Tau, ApTt).
    """
    from alz.viewer.shared.trajectory import annotate_trajectory_columns
    return annotate_trajectory_columns(
        df,
        timepoints=_TRAJ_TIMEPOINTS,
        valid_diseases=_TRAJ_VALID_DISEASES,
        source_label=source_label,
    )

_INCYTR_LOW_SIGNAL_MEDIAN_N_THRESHOLD = 3



def _incytr_celltype_qc_counts_path() -> str:
    return os.path.join(
        config.REPO_ROOT,
        "outputs",
        "reports",
        "snrna_integration",
        "pseudobulk_cell_counts.csv",
    )

def _read_incytr_celltype_qc(celltypes: list[str]) -> dict:
    """Cell-count QC metadata for Song/AD Incytr interpretation.

    The canonical Incytr rows remain unchanged. This metadata supports a
    viewer-side sensitivity filter that removes cell-cell interactions where
    either endpoint has median male pseudobulk n_cells <= 3.
    """
    counts_path = _incytr_celltype_qc_counts_path()
    threshold = _INCYTR_LOW_SIGNAL_MEDIAN_N_THRESHOLD
    out = {
        "source": os.path.relpath(counts_path, config.REPO_ROOT),
        "sample_scope": "samples whose id contains '_ma_'",
        "low_signal_rule": f"median_n <= {threshold}",
        "low_signal_median_n_threshold": threshold,
        "by_celltype": {},
        "low_signal_celltypes": [],
    }
    if not os.path.exists(counts_path):
        print(f"  (warn) Incytr celltype QC counts not found: {counts_path}",
              flush=True)
        return out

    counts = pd.read_csv(counts_path)
    required = {"sample", "cell_type", "n_cells"}
    if not required.issubset(counts.columns):
        print(f"  (warn) Incytr celltype QC counts missing columns: "
              f"{sorted(required - set(counts.columns))}", flush=True)
        return out

    counts = counts[counts["sample"].astype(str).str.contains("_ma_", regex=False)]
    counts["n_cells"] = pd.to_numeric(counts["n_cells"], errors="coerce")
    counts = counts.dropna(subset=["cell_type", "n_cells"])
    stats = counts.groupby("cell_type", sort=False)["n_cells"].agg(
        median_n="median",
        mean_n="mean",
        min_n="min",
        total_n="sum",
        n_samples="count",
    )
    cell_set = sorted(set(celltypes))
    low: list[str] = []
    by_celltype: dict[str, dict] = {}
    for ct in cell_set:
        if ct in stats.index:
            row = stats.loc[ct]
            median_n = float(row["median_n"])
            is_low = median_n <= threshold
            rec = {
                "median_n": median_n,
                "mean_n": float(row["mean_n"]),
                "min_n": int(row["min_n"]),
                "total_n": int(row["total_n"]),
                "n_samples": int(row["n_samples"]),
                "low_signal_median_le_3": bool(is_low),
            }
            if is_low:
                low.append(ct)
        else:
            rec = {
                "median_n": None,
                "mean_n": None,
                "min_n": None,
                "total_n": 0,
                "n_samples": 0,
                "low_signal_median_le_3": False,
            }
        by_celltype[ct] = rec
    out["by_celltype"] = by_celltype
    out["low_signal_celltypes"] = sorted(low)
    print(f"  incytr celltype_qc: {len(low)} low-signal endpoint(s) "
          f"at median_n <= {threshold}", flush=True)
    return out


def _incytr_sanitize(name: str) -> str:
    """Match the upstream sanitize in alz/integration/load.R:sanitize_celltype."""
    return name.replace("/", "-").replace(" ", "_")

_INCYTR_PAIR_GENO_NORMALIZE = {
    "AppP": "App", "App": "App",
    "Ttau": "Tau", "Tau": "Tau", "TtauP": "Tau",
    "ApTt": "ApTt", "AppPTtau": "ApTt", "AppTtau": "ApTt",
}


def _pair_mode_contrast_from_filename(fname: str) -> str | None:
    """`ma_2mo_AppP_ma_2mo_WTyp_incytr_output.parquet` → 'App_2mo'."""
    m = re.match(
        r"ma_(\d+)mo_([A-Za-z]+)_ma_\1mo_WTyp_incytr_output\.parquet$", fname
    )
    if not m:
        return None
    age, geno_token = m.group(1), m.group(2)
    geno = _INCYTR_PAIR_GENO_NORMALIZE.get(geno_token)
    if geno is None:
        print(f"    (warn) unknown geno token '{geno_token}' in {fname}; skipping",
              flush=True)
        return None
    return f"{geno}_{age}mo"

def _write_incytr_pair_pathways() -> dict | None:
    """Shard the pair-mode Incytr output by (sender, receiver) — unfiltered.

    Reads `outputs/reports/incytr_pair_mode/wide/*.parquet` (one file per
    contrast, Levy-t5 spine, 31² = 961 sender×receiver pairs) and emits one
    parquet per pair under `edge_slices/incytr_pathways/` plus the
    `incytr_pathways` payload block. No build-time significance gate; the
    UI thresholds live.
    """
    input_dir = os.environ.get(
        "INCYTR_PAIR_MODE_INPUT_DIR",
        os.path.join(config.REPO_ROOT, "outputs", "reports", "incytr_pair_mode", "wide"),
    )
    if not os.path.isdir(input_dir):
        print(f"  (warn) pair-mode input dir not found: {input_dir}; "
              f"skipping incytr_pathways", flush=True)
        return None

    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*_incytr_output.parquet")))
    if not parquet_files:
        print(f"  (warn) no pair-mode parquets in {input_dir}; "
              f"skipping incytr_pathways", flush=True)
        return None

    # Pair each file with its contrast label. Drop files we can't parse.
    file_to_contrast: list[tuple[str, str]] = []
    for fpath in parquet_files:
        contrast = _pair_mode_contrast_from_filename(os.path.basename(fpath))
        if contrast is not None:
            file_to_contrast.append((fpath, contrast))
    if not file_to_contrast:
        print(f"  (warn) no parseable pair-mode parquets in {input_dir}; "
              f"skipping incytr_pathways", flush=True)
        return None

    # Contrast list: subset of the canonical 9, in canonical order.
    present_contrasts = [c for c in _INCYTR_CONTRASTS
                         if c in {c2 for _, c2 in file_to_contrast}]
    contrast_to_idx = {c: i for i, c in enumerate(present_contrasts)}
    signature = _input_signature(
        "incytr_pathways",
        [__file__, *[f for f, _ in file_to_contrast], _incytr_celltype_qc_counts_path()],
        {
            "builder_version": 2,  # 2: global filter-index replaces top_instances
            "schema_version": SCHEMA_VERSION,
            "input_dir": os.path.relpath(input_dir, config.REPO_ROOT),
            "file_to_contrast": [
                (os.path.relpath(f, config.REPO_ROOT), c)
                for f, c in file_to_contrast
            ],
            "present_contrasts": list(present_contrasts),
            "pvalue_thresholds": list(_INCYTR_PATHWAY_PVALUES),
            "abs_pds_thresholds": list(_INCYTR_PATHWAY_ABS_PDS),
            "global_index_file": _INCYTR_INDEX_FILENAME,
            "score_columns": list(_INCYTR_SCORE_COLS),
            "fc_columns": list(_INCYTR_FC_COLS),
            "label_columns": list(_INCYTR_LABEL_COLS),
            "label_vocab": list(_INCYTR_LABEL_VOCAB),
            "low_signal_median_n_threshold": _INCYTR_LOW_SIGNAL_MEDIAN_N_THRESHOLD,
            "trajectory_timepoints": list(_TRAJ_TIMEPOINTS),
            "trajectory_labels": list(_SIGN_VEC_LABELS),
        },
    )
    cached = _load_build_cache(
        "incytr_pathways", signature, EDGE_SLICES_INCYTR_PATHWAYS_DIR,
    )
    if cached is not None:
        return cached

    print(f"  incytr_pair_pathways: {len(file_to_contrast)} parquet(s); "
          f"contrasts = {present_contrasts}", flush=True)

    import duckdb

    con = duckdb.connect()
    con.execute("PRAGMA threads=8; PRAGMA memory_limit='12GB';")
    _configure_duckdb_tempdir(con)

    # Detect optional path-level direction-flag columns once.
    sample_schema = pq.read_schema(file_to_contrast[0][0])
    src_cols = {f.name for f in sample_schema}
    dir_flag_cols = [c for c in ("pr_up", "pr_down", "ps_up", "ps_down",
                                  "py_up", "py_down")
                     if c in src_cols]
    # aFC is byte-identical to log2FC in every row (verified across all 9
    # contrasts) — drop the duplicate to save ~17% of shard storage. log2FC
    # is the canonical path-level fold-change column.
    extra_path_cols = [c for c in ("log2FC",) if c in src_cols]
    if not dir_flag_cols:
        print(f"    (warn) no direction-flag columns; downstream UI badges "
              f"will be empty", flush=True)

    # Build per-file SELECTs unioning into a single src table. Use disease-arm
    # p-value (`p_value_ma_<age>mo_<geno>`) as the `pvalue` proxy — pair-mode
    # has no factorial-OLS pooled p-value.
    selects = []
    has_pvalue = False  # nboot=0 outputs omit p_value_* entirely.
    for fpath, contrast in file_to_contrast:
        sch = pq.read_schema(fpath)
        names = {f.name for f in sch}
        pcol_disease = None
        for n in names:
            if n.startswith("p_value_") and not n.endswith("_WTyp"):
                pcol_disease = n
                has_pvalue = True
                break
        if pcol_disease is None:
            print(f"    (warn) no disease-arm p_value col in "
                  f"{os.path.basename(fpath)}; using NULL", flush=True)
            pcol_clause = "CAST(NULL AS DOUBLE)"
        else:
            pcol_clause = f'CAST("{pcol_disease}" AS DOUBLE)'

        # SiK_score is the lone two-condition score: the wide driver emits
        # `SiK_score_<disease>` and `SiK_score_<WTyp>` (never a bare
        # `SiK_score`), so collapse to the disease arm here — same disease-arm
        # selection as pvalue above, and matching pair_to_receiver_cache.py.
        sik_disease = next(
            (n for n in names
             if n.startswith("SiK_score_") and not n.endswith("_WTyp")),
            None,
        )
        if sik_disease is None:
            print(f"    (warn) no disease-arm SiK_score col in "
                  f"{os.path.basename(fpath)}; using NULL", flush=True)
            sik_clause = "CAST(NULL AS DOUBLE) AS SiK_score"
        else:
            sik_clause = f'CAST("{sik_disease}" AS DOUBLE) AS SiK_score'

        dir_clauses = ",\n          ".join(
            f"CAST({c} AS DOUBLE) AS {c}" for c in dir_flag_cols
        )
        path_clauses = ",\n          ".join(
            f"CAST({c} AS DOUBLE) AS {c}" for c in extra_path_cols
        )
        # SiK_score handled via sik_clause (two-condition collapse); the rest
        # are single-name and emitted verbatim or NULL-filled.
        generic_scores = [c for c in _INCYTR_SCORE_COLS if c != "SiK_score"]
        score_clauses = ",\n          ".join(
            f"CAST({c} AS DOUBLE) AS {c}" for c in generic_scores
            if c in names
        )
        # Stable column order; missing scores are NULL.
        missing_scores = [c for c in generic_scores if c not in names]
        missing_score_clauses = ",\n          ".join(
            f"CAST(NULL AS DOUBLE) AS {c}" for c in missing_scores
        )
        # Per-node FC columns (Ligand_sclog2FC, …): NULL-fill if absent in
        # this driver's output. Names match _INCYTR_FC_COLS exactly.
        fc_clauses = ",\n          ".join(
            (f'CAST("{c}" AS DOUBLE) AS "{c}"' if c in names
             else f'CAST(NULL AS DOUBLE) AS "{c}"')
            for c in _INCYTR_FC_COLS
        )
        # Per-node evidence labels: source columns use dot notation
        # ("Ligand.label"), aliased to underscore ("Ligand_label") in the
        # output shard. NULL-fill if upstream driver omitted them.
        label_clauses = ",\n          ".join(
            (f'CAST("{src}" AS VARCHAR) AS "{dst}"' if src in names
             else f'CAST(NULL AS VARCHAR) AS "{dst}"')
            for src, dst in zip(_INCYTR_LABEL_SRC, _INCYTR_LABEL_COLS)
        )
        clauses = [score_clauses, missing_score_clauses, sik_clause,
                   dir_clauses, path_clauses, fc_clauses, label_clauses]
        extra_select = ",\n          ".join(c for c in clauses if c)

        # No |PDS| pre-filter here: significance is now applied upstream in the
        # driver pipeline (alz/incytr_pair/filter_significant_paths.py:
        # SigProb>0.1 AND |PDS|>=0.2), so the inputs are already the significant
        # set. For nboot=100 inputs we still drop clearly-no-signal rows by
        # disease-arm pvalue > 0.75 (coarse no-signal cut, not a ranking gate);
        # nboot=0 has no pvalue column so nothing is dropped here.
        where_clause = (
            f"WHERE {pcol_clause} IS NULL OR {pcol_clause} <= 0.75"
            if pcol_disease is not None else ""
        )
        selects.append(f"""
        SELECT
          "Sender.group"   AS sender,
          "Receiver.group" AS receiver,
          Path, Ligand, Receptor, EM, Target,
          '{contrast}'      AS contrast,
          {pcol_clause}     AS pvalue,
          CAST(PDS AS DOUBLE) AS PDS,
          {extra_select}
        FROM read_parquet('{fpath}')
        {where_clause}
        """)
    union_sql = "\nUNION ALL\n".join(selects)
    # VIEW (not TEMP TABLE) — DuckDB re-reads parquet on demand for each query,
    # avoiding a large in-memory materialization that OOMs the box. Significance
    # filtering happens upstream (SigProb>0.1 AND |PDS|>=0.2); the only per-file
    # WHERE retained here is the nboot=100 disease-arm pvalue <= 0.75 coarse
    # no-signal cut (pair-mode pvalue is untrustworthy for ranking).
    con.execute(f"CREATE VIEW src AS {union_sql}")
    n_src = con.execute("SELECT COUNT(*) FROM src").fetchone()[0]
    print(f"  incytr_pair_pathways: loaded {n_src:,} rows across "
          f"{len(file_to_contrast)} contrast(s) (upstream-filtered SigProb>0.1 & |PDS|>=0.2)",
          flush=True)

    # Sender/receiver canonical lists from the data itself (Levy-t5, already
    # sanitized — no display↔sanitized indirection).
    senders_canonical = sorted({r[0] for r in con.execute(
        "SELECT DISTINCT sender FROM src").fetchall()})
    receivers_canonical = sorted({r[0] for r in con.execute(
        "SELECT DISTINCT receiver FROM src").fetchall()})
    sender_to_idx = {s: i for i, s in enumerate(senders_canonical)}
    receiver_to_idx = {r: i for i, r in enumerate(receivers_canonical)}
    n_s, n_r, n_c = len(senders_canonical), len(receivers_canonical), len(present_contrasts)
    print(f"    senders={n_s}, receivers={n_r}, contrasts={n_c} "
          f"(pair count={n_s * n_r})", flush=True)
    celltype_qc = _read_incytr_celltype_qc(
        sorted(set(senders_canonical) | set(receivers_canonical))
    )
    low_signal_celltypes = set(celltype_qc.get("low_signal_celltypes") or [])
    if low_signal_celltypes:
        con.register(
            "low_signal_celltypes",
            pd.DataFrame({"cell_type": sorted(low_signal_celltypes)}),
        )

    # --- heatmap_counts cube (same shape contract as factorial) ----------
    n_thr = len(_INCYTR_PATHWAY_PVALUES)
    n_ap = len(_INCYTR_PATHWAY_ABS_PDS)
    # nboot=0 has no pvalue: count every row (|PDS| does the gating) instead of
    # gating on a NULL column, which would zero out the whole cube. nboot=100
    # keeps the pvalue gate unchanged.
    pval_filter = (lambda tp: f"pvalue < {tp}") if has_pvalue else (lambda tp: "TRUE")
    pval_where = "WHERE pvalue IS NOT NULL" if has_pvalue else ""
    hm_thr_clauses_list = []
    for ip, tp in enumerate(_INCYTR_PATHWAY_PVALUES):
        for iap, tap in enumerate(_INCYTR_PATHWAY_ABS_PDS):
            hm_thr_clauses_list.append(
                f"COUNT(*) FILTER (WHERE {pval_filter(tp)} "
                f"AND COALESCE(ABS(PDS), 0) >= {tap}) AS c_{ip}_{iap}"
            )
    hm_thr_clauses = ", ".join(hm_thr_clauses_list)
    hm_rows = con.execute(f"""
        SELECT sender, receiver, contrast, {hm_thr_clauses}
        FROM src
        {pval_where}
        GROUP BY sender, receiver, contrast
    """).fetchall()
    grid = np.zeros((n_s, n_r, n_c, n_thr, n_ap), dtype=np.uint32)
    for row in hm_rows:
        s_raw, r_raw, c = row[0], row[1], row[2]
        if s_raw not in sender_to_idx or r_raw not in receiver_to_idx:
            continue
        if c not in contrast_to_idx:
            continue
        s_i, r_i, c_i = sender_to_idx[s_raw], receiver_to_idx[r_raw], contrast_to_idx[c]
        offset = 3
        for ip in range(n_thr):
            for iap in range(n_ap):
                grid[s_i, r_i, c_i, ip, iap] = int(row[offset])
                offset += 1
    totals = np.zeros((n_thr, n_ap), dtype=np.uint64)
    for ip in range(n_thr):
        for iap in range(n_ap):
            totals[ip, iap] = int(grid[:, :, :, ip, iap].sum())
    heatmap_counts = {
        "thresholds": list(_INCYTR_PATHWAY_PVALUES),
        "abs_pds_thresholds": list(_INCYTR_PATHWAY_ABS_PDS),
        "shape": [n_s, n_r, n_c, n_thr, n_ap],
        "counts": grid.flatten().tolist(),
        "total_by_threshold": totals.tolist(),
    }
    hm_signed_rows = con.execute(f"""
        SELECT sender, receiver, contrast,
               CASE WHEN PDS > 0 THEN 2
                    WHEN PDS < 0 THEN 0
                    ELSE 1 END AS s,
               {hm_thr_clauses}
        FROM src
        {pval_where}
        GROUP BY sender, receiver, contrast, s
    """).fetchall()
    signed_grid = np.zeros((n_s, n_r, n_c, 3, n_thr, n_ap), dtype=np.uint32)
    for row in hm_signed_rows:
        s_raw, r_raw, c, sign_i = row[0], row[1], row[2], int(row[3])
        if s_raw not in sender_to_idx or r_raw not in receiver_to_idx:
            continue
        if c not in contrast_to_idx:
            continue
        s_i, r_i, c_i = sender_to_idx[s_raw], receiver_to_idx[r_raw], contrast_to_idx[c]
        offset = 4
        for ip in range(n_thr):
            for iap in range(n_ap):
                signed_grid[s_i, r_i, c_i, sign_i, ip, iap] = int(row[offset])
                offset += 1
    signed_totals = np.zeros((3, n_thr, n_ap), dtype=np.uint64)
    for sign_i in range(3):
        for ip in range(n_thr):
            for iap in range(n_ap):
                signed_totals[sign_i, ip, iap] = int(signed_grid[:, :, :, sign_i, ip, iap].sum())
    heatmap_counts_signed = {
        "thresholds": list(_INCYTR_PATHWAY_PVALUES),
        "abs_pds_thresholds": list(_INCYTR_PATHWAY_ABS_PDS),
        "shape": [n_s, n_r, n_c, 3, n_thr, n_ap],
        "counts": signed_grid.flatten().tolist(),
        "total_by_sign_threshold": signed_totals.tolist(),
        "sign_source": "PDS",
    }
    p_005_idx = _INCYTR_PATHWAY_PVALUES.index(0.05)
    ap_zero_idx = _INCYTR_PATHWAY_ABS_PDS.index(0.0)
    ap_001_idx = _INCYTR_PATHWAY_ABS_PDS.index(0.01)
    print(f"    heatmap_counts: total at pvalue<0.05 & |PDS|>=0    = "
          f"{int(totals[p_005_idx, ap_zero_idx]):>9,}; "
          f"at pvalue<0.05 & |PDS|>=0.01 = {int(totals[p_005_idx, ap_001_idx]):>9,}",
          flush=True)

    # --- pathway_counts cube ---------------------------------------------
    def _build_pathway_counts(where_extra: str = "") -> dict:
        thr_clauses = []
        for ip, tp in enumerate(_INCYTR_PATHWAY_PVALUES):
            for iap, tap in enumerate(_INCYTR_PATHWAY_ABS_PDS):
                thr_clauses.append(
                    f"COUNT(*) FILTER (WHERE {pval_filter(tp)} "
                    f"AND COALESCE(ABS(PDS), 0) >= {tap}) AS c_{ip}_{iap}"
                )
        where_parts = []
        if has_pvalue:
            where_parts.append("pvalue IS NOT NULL")
        if where_extra:
            where_parts.append(where_extra)
        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        pathway_rows = con.execute(f"""
            SELECT contrast,
                   CASE WHEN PDS > 0 THEN 2
                        WHEN PDS < 0 THEN 0
                        ELSE 1 END AS s,
                   {", ".join(thr_clauses)}
            FROM src
            {where_clause}
            GROUP BY contrast, s
        """).fetchall()
        pathway_arr = np.zeros((n_c, 3, n_thr, n_ap), dtype=np.uint32)
        for row in pathway_rows:
            contrast, s_idx = row[0], int(row[1])
            if contrast not in contrast_to_idx:
                continue
            c_idx = contrast_to_idx[contrast]
            for ip in range(n_thr):
                for iap in range(n_ap):
                    pathway_arr[c_idx, s_idx, ip, iap] = int(row[2 + ip * n_ap + iap])
        return {
            "thresholds": list(_INCYTR_PATHWAY_PVALUES),
            "abs_pds_thresholds": list(_INCYTR_PATHWAY_ABS_PDS),
            "contrasts": list(present_contrasts),
            "counts": pathway_arr.flatten().tolist(),
            "shape": [n_c, 3, n_thr, n_ap],
            "sign_source": "PDS",
        }

    pathway_counts = _build_pathway_counts()
    low_signal_where = (
        "sender NOT IN (SELECT cell_type FROM low_signal_celltypes) "
        "AND receiver NOT IN (SELECT cell_type FROM low_signal_celltypes)"
        if low_signal_celltypes else ""
    )
    pathway_counts_low_signal_excluded = (
        _build_pathway_counts(low_signal_where) if low_signal_where else None
    )

    src_cols_pair = set(
        con.execute("DESCRIBE SELECT * FROM src LIMIT 0").fetchdf()["column_name"]
    )

    # --- global filter-index encoders --------------------------------------
    # Encoded per-pair inside _flush (full precision, before the float16
    # downcast), accumulated, then concatenated + globally |PDS|-sorted after
    # the streaming pass into one packed binary. Replaces the former top-5000
    # `top_instances` pre-cap. Column layout (little-endian, length N): columns
    # are ordered wide→narrow (f4, then u2, then u1) so each column's byte
    # offset is a multiple of its element size — the viewer maps each as a
    # zero-copy TypedArray view, which REQUIRES aligned offsets.
    INCYTR_INDEX_COLUMNS = (
        [("PDS", "f4"), ("pvalue", "f4")]
        + [(sc, "u2") for sc in _INCYTR_SCORE_COLS]   # float16 bit-patterns
        + [("ligandId", "u2"), ("receptorId", "u2"),
           ("emId", "u2"), ("targetId", "u2")]
        + [("senderId", "u1"), ("receiverId", "u1"), ("contrastId", "u1"),
           ("labelBits", "u1"), ("trajBits", "u1")]
    )
    idx_gene_to_id: dict[str, int] = {}
    idx_gene_vocab: list[str] = []
    idx_chunks: list[dict] = []

    def _idx_gene_ids(series) -> np.ndarray:
        cat = series.astype(str)
        for g in cat.unique():
            if g not in idx_gene_to_id:
                idx_gene_to_id[g] = len(idx_gene_vocab)
                idx_gene_vocab.append(g)
        return cat.map(idx_gene_to_id).to_numpy(dtype="<u2")

    def _accumulate_index(s_name: str, r_name: str, frame) -> None:
        n = len(frame)
        if n == 0:
            return
        chunk = {
            "senderId": np.full(n, sender_to_idx[s_name], dtype="<u1"),
            "receiverId": np.full(n, receiver_to_idx[r_name], dtype="<u1"),
            "contrastId": frame["contrast"].map(contrast_to_idx).to_numpy(dtype="<u1"),
            "ligandId": _idx_gene_ids(frame["Ligand"]),
            "receptorId": _idx_gene_ids(frame["Receptor"]),
            "emId": _idx_gene_ids(frame["EM"]),
            "targetId": _idx_gene_ids(frame["Target"]),
            "labelBits": _idx_label_bits(frame),
            "trajBits": _idx_traj_bits(frame["traj_labels"]),
            "PDS": frame["PDS"].to_numpy(dtype="<f4"),
            "pvalue": frame["pvalue"].to_numpy(dtype="<f4"),
        }
        for sc in _INCYTR_SCORE_COLS:
            chunk[sc] = (frame[sc].to_numpy(dtype="float16").view("<u2")
                         if sc in frame.columns else np.zeros(n, dtype="<u2"))
        idx_chunks.append(chunk)

    gene_node_index = _build_incytr_gene_node_index(con)
    print(
        f"    gene_node_index: {len(gene_node_index['gene_id']):,} "
        f"gene-role-pair entries; {len(gene_node_index['genes']):,} genes",
        flush=True,
    )

    # --- shard the long table per (sender, receiver) ---------------------
    # Wipe-and-recreate to avoid mixing old/new shards on an aborted run.
    shutil.rmtree(EDGE_SLICES_INCYTR_PATHWAYS_DIR, ignore_errors=True)
    os.makedirs(EDGE_SLICES_INCYTR_PATHWAYS_DIR, exist_ok=True)

    # Materialize once; group + write. Pair-mode now supplies per-node FC
    # (driver Cal_scFC, written inline) and labels (driver DEG/prG assignment);
    # fall back to NULL if the source parquet was produced by an older driver.
    fc_select = [
        f'"{c}"' if c in src_cols_pair
        else f'CAST(NULL AS DOUBLE) AS "{c}"'
        for c in _INCYTR_FC_COLS
    ]
    # The union SELECT above already aliased `<Node>.label` → `<Node>_label`,
    # so the view exposes the underscore form. Reference the dst name.
    label_select = [
        f'"{dst}"' if dst in src_cols_pair
        else f'CAST(NULL AS VARCHAR) AS "{dst}"'
        for dst in _INCYTR_LABEL_COLS
    ]
    # Per-pair shard columns: sender/receiver are constant within a shard so
    # they're omitted (matches the factorial path's contract). `Path` is also
    # omitted — it's just `Ligand|Receptor|EM|Target` concatenated and is
    # reconstructed client-side in incytr_pathways.js (~10% disk savings).
    shard_select_cols = (
        ["Ligand", "Receptor", "EM", "Target",
         "contrast", "pvalue", "PDS"]
        + list(_INCYTR_SCORE_COLS)
        + dir_flag_cols
        + extra_path_cols
        + fc_select
        + label_select
    )
    # pvalue stays float32 to preserve dynamic range for small values
    # (float16's smallest normal is ~6e-8). All other floats compress to
    # float16 — ~3 decimal digits of precision, matching the UI display.
    float32_cols = ["pvalue"]
    float16_cols = (["PDS"]
                    + list(_INCYTR_SCORE_COLS)
                    + list(_INCYTR_FC_COLS)
                    + dir_flag_cols + extra_path_cols)
    float_cols = float32_cols + float16_cols  # for BSS encoding selection

    present_pairs: list[list[str]] = []
    pair_row_counts: dict[str, int] = {}
    total_rows = 0
    max_shard_bytes = 0
    max_shard_name = ""
    # CR-04: trajectory annotation runs per-pair inside _flush. Per-pair
    # path_strings (sender||receiver||Path) are globally unique, so per-pair
    # recur dicts and traj label counts compose by simple union / sum.
    recur_index: dict = {}
    traj_summary: dict = {}

    # Single streaming pass over the union view, ordered by (sender, receiver)
    # so each pair's rows arrive contiguously. Previous implementation issued
    # one DuckDB query per (sender, receiver) — 961 queries × 9 parquet rescans
    # each = ~8.6k parquet reads. This pass reads each parquet once.
    def _flush(key: tuple[str, str], frames: list[pd.DataFrame]) -> None:
        nonlocal total_rows, max_shard_bytes, max_shard_name
        if not frames:
            return
        sub = pd.concat(frames, ignore_index=True, copy=False)
        for col in _INCYTR_LABEL_COLS:
            if col in sub.columns:
                sub[col] = pd.Categorical(sub[col], categories=_INCYTR_LABEL_VOCAB)
        # CR-04: annotate traj_labels + sign_vec. Add sender/receiver/Path as
        # temp columns (constant within a shard) so the annotation function
        # can key on (sender, receiver, Path); drop sender/receiver after
        # annotation (Path is rebuilt client-side, so also dropped).
        s_key, r_key = key
        sub["sender"] = s_key
        sub["receiver"] = r_key
        sub["Path"] = (sub["Ligand"].astype(str) + "|"
                       + sub["Receptor"].astype(str) + "|"
                       + sub["EM"].astype(str) + "|"
                       + sub["Target"].astype(str))
        sub, pair_recur, pair_traj = _annotate_trajectory_columns(
            sub, source_label="pair_mode",
        )
        recur_index.update(pair_recur)
        for label, count in pair_traj.items():
            traj_summary[label] = traj_summary.get(label, 0) + int(count)
        # Encode this pair into the global filter-index BEFORE the float16
        # downcast (PDS/pvalue captured at full precision; ranking is locked by
        # the build-time argsort below, so the stored f16 scores are display-
        # only and cannot reorder ranks).
        _accumulate_index(s_key, r_key, sub)
        sub = sub.drop(columns=["sender", "receiver", "Path"])
        # Float dtype compression runs after annotation so traj_labels
        # (string) and sign_vec (str) are unaffected.
        for col in float32_cols:
            if col in sub.columns:
                sub[col] = sub[col].astype("float32")
        for col in float16_cols:
            if col in sub.columns:
                sub[col] = sub[col].astype("float16")
        # Sort by (Ligand, Receptor, EM, Target, contrast) so the path-string
        # columns form long RLE runs in the parquet dictionary encoding.
        # 4,130 distinct paths × 9 contrasts repeated → 5-10× compression on
        # the string columns vs the previous (contrast, pvalue) sort, which
        # scattered identical paths across the file.
        path_sort_cols = [c for c in ("Ligand", "Receptor", "EM", "Target", "contrast")
                          if c in sub.columns]
        if path_sort_cols:
            sub = sub.sort_values(path_sort_cols, kind="stable",
                                  na_position="last").reset_index(drop=True)
        s, r = key
        fname = f"{_incytr_sanitize(s)}__{_incytr_sanitize(r)}.parquet"
        path = os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, fname)
        # byte_stream_split on float columns improves zstd compression of
        # float bit-patterns by ~15-25% (parquet docs §encoding). pyarrow
        # refuses column_encoding when use_dictionary=True, so pass
        # use_dictionary as an explicit list of non-float columns (which keeps
        # RLE_DICTIONARY on string/categorical columns) and BYTE_STREAM_SPLIT
        # on the float columns via column_encoding.
        present_floats = [c for c in float_cols if c in sub.columns]
        bss_cols = {c: "BYTE_STREAM_SPLIT" for c in present_floats}
        dict_cols = [c for c in sub.columns if c not in bss_cols]
        pq.write_table(
            pa.Table.from_pandas(sub, preserve_index=False),
            path, compression="zstd",
            column_encoding=bss_cols if bss_cols else None,
            use_dictionary=dict_cols if bss_cols else True,
        )
        present_pairs.append([s, r])
        pair_row_counts[fname] = len(sub)
        total_rows += len(sub)
        sz = os.path.getsize(path)
        if sz > max_shard_bytes:
            max_shard_bytes = sz
            max_shard_name = fname

    # Per-sender streaming via Arrow record batches. One query per sender (31
    # total) — not one global sort (OOMs DuckDB) nor one query per
    # (sender, receiver) pair (961 queries × 9 parquet rescans). The
    # ORDER BY receiver is a blocking sort that spills to temp under
    # memory_limit, then streams sorted batches; we accumulate one
    # receiver-run at a time and flush at each receiver boundary. This bounds
    # peak RAM to a single (sender, receiver) shard's rows rather than a whole
    # sender — the largest sender (Cholinergic, 11.6M rows at nboot=0) would be
    # ~8 GB if materialized with fetchdf(), which risks OOM on the shared box.
    stream_cols = ["receiver"] + shard_select_cols
    for s in senders_canonical:
        reader = con.execute(
            f"""SELECT {', '.join(stream_cols)}
                FROM src
                WHERE sender = ?
                ORDER BY receiver""",
            [s],
        ).fetch_record_batch(1_000_000)
        cur_receiver: str | None = None
        buf: list[pd.DataFrame] = []
        for batch in reader:
            bdf = batch.to_pandas()
            receivers = bdf["receiver"].to_numpy()
            # Run-length partition by receiver within this batch; receivers are
            # globally contiguous because of the ORDER BY, so a run may span
            # batch boundaries (handled by carrying cur_receiver across batches).
            starts = [0]
            for i in range(1, len(receivers)):
                if receivers[i] != receivers[i - 1]:
                    starts.append(i)
            starts.append(len(receivers))
            for j in range(len(starts) - 1):
                a, b = starts[j], starts[j + 1]
                r = receivers[a]
                seg = bdf.iloc[a:b].drop(columns=["receiver"])
                if cur_receiver is None:
                    cur_receiver = r
                elif r != cur_receiver:
                    _flush((s, cur_receiver), buf)
                    buf = []
                    cur_receiver = r
                buf.append(seg)
        if buf and cur_receiver is not None:
            _flush((s, cur_receiver), buf)
    con.close()

    index = {
        "schema_version": SCHEMA_VERSION,
        "filename_template": "{sender}__{receiver}.parquet",
        "sanitize_rule": "replace('/', '-'); replace(' ', '_')",
        "present": sorted(present_pairs),
        "n_total_rows": total_rows,
        "pair_row_counts": pair_row_counts,
    }
    with open(os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, "index.json"), "w") as f:
        json.dump(index, f)

    total_bytes = sum(
        os.path.getsize(os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, fn))
        for fn in os.listdir(EDGE_SLICES_INCYTR_PATHWAYS_DIR)
        if fn.endswith(".parquet")
    )
    print(f"  incytr_pair_pathways: wrote {len(present_pairs)} shards "
          f"({total_rows:,} rows; {total_bytes/1e6:.1f} MB total; "
          f"max {max_shard_bytes/1e6:.2f} MB → {max_shard_name})", flush=True)

    # --- global filter-index: concat → |PDS|-sort → packed binary ----------
    # Rows are reordered by ABS(PDS) DESC so the row position IS the global
    # rank; the viewer streams the columns into TypedArrays and runs
    # filter → rank → slice(N) over the complete universe (no pre-cap).
    assert sys.byteorder == "little", "global index assumes little-endian"
    global_index = None
    if idx_chunks:
        cols = {name: np.concatenate([c[name] for c in idx_chunks])
                for name, _dt in INCYTR_INDEX_COLUMNS}
        idx_chunks.clear()
        n_idx = int(len(cols["PDS"]))
        perm = np.argsort(-np.abs(cols["PDS"]), kind="stable")
        buf = bytearray()
        columns_manifest = []
        for name, dt in INCYTR_INDEX_COLUMNS:
            arr = np.ascontiguousarray(cols[name][perm], dtype=np.dtype("<" + dt[0] + dt[1]))
            columns_manifest.append({"name": name, "type": dt, "bytes": int(arr.nbytes)})
            buf += arr.tobytes()
        del cols
        raw_bin = bytes(buf)
        gz_bin = gzip.compress(raw_bin, compresslevel=6)
        with open(os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR,
                               _INCYTR_INDEX_FILENAME), "wb") as f:
            f.write(gz_bin)
        global_index = {
            "url": f"edge_slices/incytr_pathways/{_INCYTR_INDEX_FILENAME}",
            "nrows": n_idx,
            "rank_by": "abs(PDS)",
            "byteorder": "little",
            "sender_vocab": senders_canonical,
            "receiver_vocab": receivers_canonical,
            "contrast_vocab": list(present_contrasts),
            "gene_vocab": idx_gene_vocab,
            "traj_label_vocab": list(_SIGN_VEC_LABELS),
            "label_states": ["", *_INCYTR_LABEL_VOCAB],   # code 0/1/2 → ""/DEG/prG
            "label_nodes": list(_INCYTR_LABEL_NODES),
            "score_columns": list(_INCYTR_SCORE_COLS),
            "columns": columns_manifest,
            "raw_bytes": len(raw_bin),
            "gzip_bytes": len(gz_bin),
        }
        print(f"  incytr global_index: {n_idx:,} rows × {len(columns_manifest)} cols, "
              f"{len(idx_gene_vocab):,} genes; {len(raw_bin)/1e6:.1f} MB raw → "
              f"{len(gz_bin)/1e6:.1f} MB gz", flush=True)

    payload_block = {
        "schema_version": SCHEMA_VERSION,
        "version": 3,           # CR-04: v3 = multi-label traj_labels (semicolon-joined)
        "source": f"pair_mode ({os.path.relpath(input_dir, config.REPO_ROOT)})",
        "source_mode": "pair_mode",
        "contrasts": list(present_contrasts),
        "senders": senders_canonical,
        "receivers": receivers_canonical,
        "empty_deg_celltypes": [],
        "celltype_qc": celltype_qc,
        "low_signal_celltypes": list(celltype_qc.get("low_signal_celltypes") or []),
        "heatmap_counts": heatmap_counts,
        "heatmap_counts_signed": heatmap_counts_signed,
        "pathway_counts": pathway_counts,
        "pathway_counts_low_signal_excluded": pathway_counts_low_signal_excluded,
        "slice_index": index,
        "score_columns": list(_INCYTR_SCORE_COLS),
        "label_columns": list(_INCYTR_LABEL_COLS),
        "label_nodes": list(_INCYTR_LABEL_NODES),
        "label_vocab": list(_INCYTR_LABEL_VOCAB),
        "direction_flag_columns": list(dir_flag_cols),
        "path_metric_columns": list(extra_path_cols),
        "global_index": global_index,
        "gene_node_index_shard": _write_gene_node_index_shard(
            gene_node_index, EDGE_SLICES_INCYTR_PATHWAYS_DIR,
            _INCYTR_GENE_NODE_INDEX_FILENAME,
        ),
        # CR-04: traj_labels/sign_vec live in shard rows; summary inline.
        "trajectory_summary": traj_summary,
    }
    _write_build_cache(
        "incytr_pathways",
        signature,
        EDGE_SLICES_INCYTR_PATHWAYS_DIR,
        ["index.json", _INCYTR_INDEX_FILENAME,
         _INCYTR_GENE_NODE_INDEX_FILENAME, *sorted(pair_row_counts)],
        payload_block,
    )
    return payload_block

def _build_kinase_celltype_evidence(data: "UnifiedData", kid: dict) -> dict:
    ev = data.celltype_evidence[
        data.celltype_evidence["kinase"].isin(kid)
    ].copy()
    ev["kinase_id"] = ev["kinase"].map(kid).astype("uint16")
    for _col, _default in [
        ("song_detected", False),
        ("song_concentration", float("nan")),
        ("song_concentration_of_total", float("nan")),
        ("song_concentration_tier", 0),
        ("song_effective_n", float("nan")),
        ("song_top_celltype", ""),
        ("song_top_concentration", float("nan")),
        ("confidence_tier", "none"),
        ("confidence_basis", ""),
        ("song_direction_support", False),
        ("human_location_tier", "none"),
        ("decomp_agrees_bulk", False),
        ("wmb_detected", False),
        ("wmb_concentration", float("nan")),
        ("wmb_concentration_tier", 0),
        ("wmb_fraction_cells_expressing", float("nan")),
        ("seaad_location_score", float("nan")),
        ("hbca_location_score", float("nan")),
        ("human_location_score", float("nan")),
        ("decomp_nes", float("nan")),
        ("decomp_fdr", float("nan")),
    ]:
        if _col not in ev.columns:
            ev[_col] = _default
    return {
        "kinase_id":  ev["kinase_id"].tolist(),
        "cell_type":  ev["cell_type"].tolist(),
        "confidence_tier": ev["confidence_tier"].astype(str).tolist(),
        "confidence_basis": ev["confidence_basis"].fillna("").astype(str).tolist(),
        "song_direction_support": [bool(v) for v in ev["song_direction_support"].fillna(False)],
        "human_location_tier": ev["human_location_tier"].fillna("none").astype(str).tolist(),
        "decomp_agrees_bulk": [bool(v) for v in ev["decomp_agrees_bulk"].fillna(False)],
        "song_detected": [bool(v) for v in ev["song_detected"].fillna(False)],
        "song_concentration": ev["song_concentration"].astype(float).round(3).tolist(),
        "song_concentration_of_total": ev["song_concentration_of_total"].astype(float).round(3).tolist(),
        "song_concentration_tier": ev["song_concentration_tier"].fillna(0).astype(int).tolist(),
        "song_effective_n": ev["song_effective_n"].astype(float).round(2).tolist(),
        "song_top_celltype": ev["song_top_celltype"].fillna("").astype(str).tolist(),
        "song_top_concentration": ev["song_top_concentration"].astype(float).round(3).tolist(),
        "wmb_detected": [bool(v) for v in ev["wmb_detected"].fillna(False)],
        "wmb_concentration": ev["wmb_concentration"].astype(float).round(3).tolist(),
        "wmb_concentration_tier": ev["wmb_concentration_tier"].fillna(0).astype(int).tolist(),
        "wmb_fraction_cells_expressing": ev["wmb_fraction_cells_expressing"].astype(float).round(3).tolist(),
        "sea_ad_lfc": ev["sea_ad_lfc"].astype(float).round(3).tolist(),
        "song_lfc":   ev["song_lfc"].astype(float).round(3).tolist(),
        "seaad_location_score": ev["seaad_location_score"].astype(float).round(3).tolist(),
        "hbca_location_score": ev["hbca_location_score"].astype(float).round(3).tolist(),
        "human_location_score": ev["human_location_score"].astype(float).round(3).tolist(),
        "decomp_nes": ev["decomp_nes"].astype(float).round(3).tolist(),
        "decomp_fdr": ev["decomp_fdr"].astype(float).round(3).tolist(),
        "concordance_direction": ev["concordance_direction"].fillna("").astype(str).tolist(),
    }


def _build_attribution_index(data: "UnifiedData", kid: dict, contrast_to_id: dict) -> dict:
    # Use full table if available, fall back to high+moderate subset.
    ua_src = data.unified_attribution_full if len(data.unified_attribution_full) > 0 \
        else data.unified_attribution
    ua = ua_src[ua_src["kinase"].isin(kid)
                & ua_src["contrast"].isin(contrast_to_id)].copy()
    # Ensure expected columns exist if building from the attributed subset.
    for _col, _default in [
        ("confidence_tier", "none"),
        ("confidence_basis", ""),
        ("specificity_unit", ""),
        ("specificity_unit_label", ""),
        ("specificity_celltype", ""),
        ("specificity_collapsed", False),
        ("direction_tier", "none"),
        ("direction_basis", ""),
        ("song_direction_support", False),
        ("human_location_tier", "none"),
        ("decomp_agrees_bulk", False),
        ("wmb_detected", False),
        ("wmb_concentration", float("nan")),
        ("wmb_concentration_tier", 0),
        ("wmb_mean_log2_expression", float("nan")),
        ("wmb_fraction_cells_expressing", float("nan")),
        ("wmb_binary_expressed", False),
        ("sea_ad_lfc", float("nan")),
        ("seaad_location_score", float("nan")),
        ("hbca_location_score", float("nan")),
        ("human_location_score", float("nan")),
        ("decomp_nes", float("nan")),
        ("decomp_fdr", float("nan")),
        ("song_lfc", float("nan")),
        ("song_pval", float("nan")),
        ("song_fdr", float("nan")),
        ("song_detected", False),
        ("song_concentration", float("nan")),
        ("song_concentration_of_total", float("nan")),
        ("song_concentration_tier", 0),
        ("song_fraction_cells_expressing", float("nan")),
        ("song_effective_n", float("nan")),
        ("song_unit_effective_n", float("nan")),
        ("song_top_celltype", ""),
        ("song_top_concentration", float("nan")),
        ("concordance_source", ""),
        ("NES", float("nan")),
        ("FDR", float("nan")),
    ]:
        if _col not in ua.columns:
            ua[_col] = _default
    attribution_index = {
        "kinase_id":   ua["kinase"].map(kid).astype("uint16").tolist(),
        "contrast_id": ua["contrast"].map(contrast_to_id).astype("uint8").tolist(),
        "cell_type":   ua["cell_type"].astype(str).tolist(),
        "confidence_tier": ua["confidence_tier"].astype(str).tolist(),
        "confidence_basis": ua["confidence_basis"].fillna("").astype(str).tolist(),
        "specificity_unit": ua["specificity_unit"].fillna("").astype(str).tolist(),
        "specificity_unit_label": ua["specificity_unit_label"].fillna("").astype(str).tolist(),
        "specificity_celltype": ua["specificity_celltype"].fillna("").astype(str).tolist(),
        "specificity_collapsed": [bool(v) for v in ua["specificity_collapsed"].fillna(False)],
        "direction_tier": ua["direction_tier"].fillna("none").astype(str).tolist(),
        "direction_basis": ua["direction_basis"].fillna("").astype(str).tolist(),
        "song_direction_support": [bool(v) for v in ua["song_direction_support"].fillna(False)],
        "human_location_tier": ua["human_location_tier"].fillna("none").astype(str).tolist(),
        "decomp_agrees_bulk": [bool(v) for v in ua["decomp_agrees_bulk"].fillna(False)],
        "wmb_detected": [bool(v) for v in ua["wmb_detected"].fillna(False)],
        "wmb_concentration": ua["wmb_concentration"].astype(float).round(4).tolist(),
        "wmb_concentration_tier": ua["wmb_concentration_tier"].fillna(0).astype(int).tolist(),
        "wmb_mean_log2_expression": ua["wmb_mean_log2_expression"].astype(float).round(3).tolist(),
        "wmb_fraction_cells_expressing": ua["wmb_fraction_cells_expressing"].astype(float).round(3).tolist(),
        "wmb_binary_expressed": [bool(v) for v in ua["wmb_binary_expressed"].fillna(False)],
        "song_detected": [bool(v) for v in ua["song_detected"].fillna(False)],
        "song_concentration": ua["song_concentration"].astype(float).round(4).tolist(),
        "song_concentration_of_total": ua["song_concentration_of_total"].astype(float).round(4).tolist(),
        "song_concentration_tier": ua["song_concentration_tier"].fillna(0).astype(int).tolist(),
        "song_fraction_cells_expressing": ua["song_fraction_cells_expressing"].astype(float).round(3).tolist(),
        "song_effective_n": ua["song_effective_n"].astype(float).round(2).tolist(),
        "song_unit_effective_n": ua["song_unit_effective_n"].astype(float).round(2).tolist(),
        "song_top_celltype": ua["song_top_celltype"].fillna("").astype(str).tolist(),
        "song_top_concentration": ua["song_top_concentration"].astype(float).round(4).tolist(),
        "sea_ad_lfc": ua["sea_ad_lfc"].astype(float).round(4).tolist(),
        "seaad_location_score": ua["seaad_location_score"].astype(float).round(4).tolist(),
        "hbca_location_score": ua["hbca_location_score"].astype(float).round(4).tolist(),
        "human_location_score": ua["human_location_score"].astype(float).round(4).tolist(),
        "decomp_nes": ua["decomp_nes"].astype(float).round(4).tolist(),
        "decomp_fdr": ua["decomp_fdr"].astype(float).round(4).tolist(),
        "song_lfc": ua["song_lfc"].astype(float).round(4).tolist(),
        "song_pval": ua["song_pval"].astype(float).round(4).tolist(),
        "song_fdr": ua["song_fdr"].astype(float).round(4).tolist(),
        "concordance_source": ua["concordance_source"].fillna("").astype(str).tolist(),
        "nes": ua["NES"].astype(float).round(4).tolist(),
        "fdr": ua["FDR"].astype(float).round(4).tolist(),
    }
    print(f"  attribution_index: {len(ua):,} rows "
          f"({ua['confidence_tier'].value_counts().to_dict()})",
          flush=True)
    return attribution_index


def _cluster_to_seaad_subclass() -> dict:
    """Song cluster → sorted SEA-AD subclasses it maps to (via its supertypes).

    Lets the audit SEA-AD heatmap outline the subclass row matching the clicked
    Song cluster — the two vocabularies differ, so a string match would silently
    fail. Authoritative supertype→subclass comes from sea_ad_supertype_lfc.csv
    (only the two label columns are read). Returns {} cleanly if absent."""
    sup_map = config.load_cluster_to_seaad_supertype_map()
    lfc_path = os.path.join(
        config.KINASE_ATTRIBUTION_OUTPUT_DIR, "sea_ad_supertype_lfc.csv")
    if not os.path.exists(lfc_path):
        return {}
    st_df = pd.read_csv(
        lfc_path, usecols=["supertype", "subclass"]).drop_duplicates()
    st_to_sc = dict(zip(st_df["supertype"], st_df["subclass"]))
    out: dict[str, list[str]] = {}
    for cl, entries in sup_map.items():
        scs = sorted({st_to_sc[st] for st, _w in entries if st in st_to_sc})
        if scs:
            out[cl] = scs
    return out


def _build_specificity_units() -> dict:
    """Static cluster→unit grouping for the confidence pill (see
    config.load_specificity_unit_map). Lets the viewer render a collapsed unit
    as an expandable parent over its child Song clusters. Also carries the
    cluster→WMB-class and cluster→SEA-AD-subclass crosswalks the audit detail
    uses to outline the matching reference row (the vocabularies differ)."""
    m = config.load_specificity_unit_map()
    cluster_to_unit = {cl: info["unit"] for cl, info in m.items()}
    cluster_to_wmb_class = {cl: info["wmb_class"] for cl, info in m.items()}
    units: dict[str, dict] = {}
    for cl, info in m.items():
        units.setdefault(info["unit"], {
            "label": info["label"],
            "collapsed": bool(info["collapsed"]),
            "children": info["children"],
        })
    return {
        "cluster_to_unit": cluster_to_unit,
        "cluster_to_wmb_class": cluster_to_wmb_class,
        "cluster_to_seaad_subclass": _cluster_to_seaad_subclass(),
        "units": units,
    }


def _build_mechanism_attribution_index() -> dict | None:
    path = os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "mechanism_attribution.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty or "mechanism_call" not in df.columns or "mechanism_score" in df.columns:
        return None
    rows = df[[c for c in _SONG_MECHANISM_COLUMNS if c in df.columns]].to_dict(orient="records")
    if not rows:
        return None
    return {
        "schema_version": 1,
        "by_context": {
            "song_ad": {
                "rows": rows,
                "source_files": [os.path.relpath(path, UNIFIED_VIEWER_DIR)],
            }
        },
    }


def _build_decomposition_index(data: "UnifiedData", kid: dict, contrast_to_id: dict) -> dict:
    decomp = data.decomposition
    decomp = decomp[decomp["kinase"].isin(kid)
                    & decomp["contrast"].isin(contrast_to_id)].copy()
    decomposition_index = {
        "kinase_id":   decomp["kinase"].map(kid).astype("uint16").tolist(),
        "contrast_id": decomp["contrast"].map(contrast_to_id).astype("uint8").tolist(),
        "cell_type":   decomp["wmb_class"].astype(str).tolist(),
        "decomp_nes":  decomp["NES"].astype(float).round(4).tolist(),
        "decomp_fdr":  decomp["FDR"].astype(float).round(4).tolist(),
    }
    print(f"  decomposition_index: {len(decomp):,} rows", flush=True)
    return decomposition_index


@dataclass(frozen=True)
class SongBuild:
    slice: CohortViewerSlice
    incytr_present: bool
    decomp_ols_slice_count: int
    song_concordance_present_genes: list


def build_song_viewer_slice(data: "UnifiedData", secretome_path: str | None = None) -> SongBuild:
    """Build the song (mouse AD) cohort slice + the meta-capability scalars
    build_payload needs. Reproduces build_payload's inline song-construction
    exactly; meta-building stays in build_payload."""
    secretome_map: dict | None = None
    if secretome_path and os.path.exists(secretome_path):
        sec_df = pd.read_csv(secretome_path, sep="\t")
        secretome_map = {
            str(row["Gene"]).upper(): str(row["Secretome location"])
            for _, row in sec_df.iterrows()
            if pd.notna(row.get("Gene")) and pd.notna(row.get("Secretome location"))
        }
        print(f"  secretome: loaded {len(secretome_map)} genes from {secretome_path}", flush=True)
    kinases_slice = _build_kinases_slice(data, secretome_map=secretome_map)
    celltypes_slice = _build_celltypes_slice(data)

    contrasts = data.edge_metadata["contrasts"]

    kid = {k: i for i, k in enumerate(data.edge_metadata["kinases"])}
    contrast_to_id = {c: i for i, c in enumerate(contrasts)}

    decomp_ols_slice_index = _write_decomp_ols_slices(kid, contrast_to_id)

    _kinase_genes = set(data.kinase_activity["gene_symbol"].dropna().astype(str))
    _kinase_genes |= set(data.edge_metadata["kinases"])
    song_concordance_slice_index = _write_song_concordance_slices(_kinase_genes)

    incytr_pathways_block = _write_incytr_pair_pathways()
    context_id = "song_ad"

    kinase_celltype_evidence = _build_kinase_celltype_evidence(data, kid)
    attribution_index = _build_attribution_index(data, kid, contrast_to_id)
    mechanism_attribution = _build_mechanism_attribution_index()
    decomposition_index = _build_decomposition_index(data, kid, contrast_to_id)

    agreement_index = _build_agreement_index(
        data.mea_stoichiometry, data.decomposition,
        kid, contrast_to_id, config.MEA_FDR_THRESH,
    )

    subclass_breakdown = _build_subclass_breakdown(kid)

    slice_ = CohortViewerSlice(
        cohort_id="song",
        context_ids=("song_ad",),
        owned_sections={
            "kinases": _as_single_context_block(kinases_slice, context_id),
            "celltypes": _as_single_context_block(celltypes_slice, context_id),
            "kinase_celltype_evidence": kinase_celltype_evidence,
            "attribution_index": attribution_index,
            "specificity_units": _build_specificity_units(),
            "decomposition_index": decomposition_index,
            "agreement_index": agreement_index,
            "subclass_breakdown": subclass_breakdown,
            "incytr_pathways": _as_single_context_block(incytr_pathways_block, context_id),
            **({"mechanism_attribution": mechanism_attribution}
               if mechanism_attribution is not None else {}),
        },
        edge_slice_ref=(
            EdgeSliceContribution("decomp_ols", {
                "decomp_ols_url": "edge_slices/decomp_ols/",
                "decomp_ols_index": "edge_slices/decomp_ols/index.json",
                "n_decomp_ols_slices": decomp_ols_slice_index.get("slice_count", 0),
                "present_decomp_ols_kinase_ids": decomp_ols_slice_index.get("present_kinase_ids", []),
            }),
            EdgeSliceContribution("incytr_pathways", {
                "incytr_pathways_url": "edge_slices/incytr_pathways/",
                "incytr_pathways_index": "edge_slices/incytr_pathways/index.json",
            }),
            EdgeSliceContribution("song_concordance", {
                "song_concordance_url": "edge_slices/song_concordance/",
                "song_concordance_index": "edge_slices/song_concordance/index.json",
                "present_song_concordance_genes": song_concordance_slice_index.get("present_genes", []),
            }),
        ),
        kinase_names=tuple(kinases_slice.get("name", [])),
        provenance={"cohort": "song_ad"},
    )
    return SongBuild(
        slice=slice_,
        incytr_present=incytr_pathways_block is not None,
        decomp_ols_slice_count=decomp_ols_slice_index.get("slice_count", 0),
        song_concordance_present_genes=song_concordance_slice_index.get("present_genes", []),
    )


def _read_empty_deg_celltypes() -> list[str]:
    """Read the list of cell types with no DEGs from the upstream MANIFEST.

    `compute_seed_lists.R` writes `deg_cell_type_status` into
    `<INCYTR_PAIR_MODE_OUTPUTS_DIR>/MANIFEST.json`. Cell types whose status
    indicates an empty DEG set are surfaced in the heatmap as hatched cells
    (visually distinct from "0 candidates pass the gate"). Returns `[]` if
    the manifest is absent or doesn't carry the field — the heatmap will
    just not render the hatched overlay in that case.
    """
    manifest_path = os.path.join(INCYTR_PAIR_MODE_OUTPUTS_DIR, "MANIFEST.json")
    if not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    status = manifest.get("deg_cell_type_status") or {}
    empty: list[str] = []
    for ct, info in status.items():
        if isinstance(info, dict):
            n = info.get("n_degs") or info.get("n_DEGs") or 0
            state = (info.get("status") or "").lower()
            if (isinstance(n, (int, float)) and n == 0) or state in {"empty", "no_degs"}:
                empty.append(ct)
        elif isinstance(info, str) and info.lower() in {"empty", "no_degs"}:
            empty.append(ct)
    return sorted(empty)
