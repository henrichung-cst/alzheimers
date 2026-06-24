"""Measurement-trace and omics-trace shard writers for the T-cell viewer."""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alz.shared import config  # noqa: E402
from alz.viewer.shared.payload_helpers import _sanitize  # noqa: E402
from alz.tcell_viewer.paths import (  # noqa: E402
    AUDIT_PREVIEW_ROWS,
    AUDIT_SOURCES_DIR,
    EDGE_SLICES_INCYTR_PATHWAYS_DIR,
    KINASE_ATTRIBUTION_TCELLS_DIR,
    TCELLS_INCYTR_INPUTS_DIR,
    UNIFIED_VIEWER_DIR,
)
from alz.tcell_viewer.common import DONORS, _incytr_sanitize  # noqa: E402

# ---------------------------------------------------------------------------
# Module-local constants
# ---------------------------------------------------------------------------

_TCELL_DAY_COL_RE = re.compile(r"^D\d+_d\d+$")

_MEASUREMENT_TRACE_COLUMNS = [
    "site_id", "gene_symbol", "motif", "protein_gene", "matched_protein",
    "log2_irs_phospho", "log2_irs_protein", "stoichiometry",
]

_MEASUREMENT_TRACE_COL_META = [
    {"raw": "site_id", "label": "Site ID",
     "definition": "Phosphosite identifier.", "format": "text"},
    {"raw": "gene_symbol", "label": "Gene",
     "definition": "Gene symbol of the phosphosite.", "format": "text"},
    {"raw": "motif", "label": "Motif",
     "definition": "Peptide motif centered on the phosphorylated residue.",
     "format": "text"},
    {"raw": "protein_gene", "label": "Protein gene",
     "definition": "Gene whose total-proteome abundance was matched as the "
                   "parent protein (blank when the gene is absent from the "
                   "total proteome).", "format": "text"},
    {"raw": "matched_protein", "label": "Matched?",
     "definition": "True when a parent-protein abundance was found in the total "
                   "proteome.", "format": "text"},
    {"raw": "log2_irs_phospho", "label": "log2 phospho",
     "definition": "Normalized phospho abundance for this day (log2; ForPerseus "
                   "per-run median-centered, as stored in "
                   "raw_phospho_normalized.csv).", "format": "float"},
    {"raw": "log2_irs_protein", "label": "log2 protein",
     "definition": "Gene-matched parent-protein abundance for this day (log2; "
                   "from total_proteome_normalized.csv).", "format": "float"},
    {"raw": "stoichiometry", "label": "Stoichiometry",
     "definition": "Median-centered log2(phospho) − log2(protein) for this day; "
                   "the value fed to MEA (stoichiometry_matrix.csv).",
     "format": "float"},
]

_TCELL_OMICS_LAYERS = (
    ("protein", "pr_deconvoluted.csv", ("gene_symbol",)),
    ("phospho_ps", "ps_deconvoluted.csv", ("site_id", "gene_symbol", "motif")),
    ("phospho_py", "py_deconvoluted.csv", ("site_id", "gene_symbol", "motif")),
)

_TCELL_OMICS_SHARD_COLS = [
    "layer", "gene_symbol", "site_id", "motif", "animal_id", "group",
    "sex", "timepoint", "genotype", "value", "log2_value",
]


# ---------------------------------------------------------------------------
# Measurement trace
# ---------------------------------------------------------------------------

def _build_tcell_measurement_trace() -> dict | None:
    """Per-day measurement-trace manifest for donor1 (ST + pY tracks).

    Returns None when donor1's normalized matrices are absent (then the audit
    drawer's Measurement Trace tab degrades; donor2 never reaches it — no IMAC).
    """
    donor = "donor1"
    root = os.path.join(KINASE_ATTRIBUTION_TCELLS_DIR, donor)
    protein_path = os.path.join(root, "total_proteome_normalized.csv")
    if not os.path.exists(protein_path):
        return None
    protein = pd.read_csv(protein_path)
    protein_by_gene = protein.drop_duplicates("gene_symbol").set_index("gene_symbol")

    out_root = os.path.join(AUDIT_SOURCES_DIR, "measurement_trace")
    tracks_index: dict[str, dict] = {}
    first_preview: list[dict] = []

    for residue_label, ph_name, st_name, subdir in (
        ("ST", "raw_phospho_normalized.csv", "stoichiometry_matrix.csv", ""),
        ("Y", "raw_phospho_normalized_pY.csv", "stoichiometry_matrix_pY.csv", "py"),
    ):
        ph_path = os.path.join(root, ph_name)
        st_path = os.path.join(root, st_name)
        if not (os.path.exists(ph_path) and os.path.exists(st_path)):
            continue
        phospho = pd.read_csv(ph_path)
        stoich = pd.read_csv(st_path).drop_duplicates("site_id").set_index("site_id")
        day_cols = [c for c in phospho.columns if _TCELL_DAY_COL_RE.match(c)]
        samples = [c for c in day_cols if c in protein.columns]
        if not samples:
            continue

        genes = phospho["gene_symbol"].fillna("").astype(str)
        gene_in_prot = genes.isin(protein_by_gene.index)
        prot_aligned = protein_by_gene.reindex(genes.values)  # row-aligned to phospho
        base = pd.DataFrame({
            "site_id": phospho["site_id"].values,
            "gene_symbol": phospho["gene_symbol"].values,
            "motif": phospho["motif"].values,
            "protein_gene": np.where(gene_in_prot.values, genes.values, ""),
            "matched_protein": gene_in_prot.values,
        })

        track_subdir = os.path.join(out_root, subdir) if subdir else out_root
        os.makedirs(track_subdir, exist_ok=True)
        sample_files: dict[str, str] = {}
        preview: list[dict] = []
        for sample in samples:
            trace = base.copy()
            trace["log2_irs_phospho"] = phospho[sample].values
            trace["log2_irs_protein"] = (
                prot_aligned[sample].values if sample in prot_aligned.columns
                else np.nan)
            trace["stoichiometry"] = (
                stoich[sample].reindex(phospho["site_id"].values).values
                if sample in stoich.columns else np.nan)
            trace = trace[_MEASUREMENT_TRACE_COLUMNS]
            dest = os.path.join(track_subdir, f"{sample}.csv")
            trace.to_csv(dest, index=False)
            sample_files[sample] = os.path.relpath(dest, UNIFIED_VIEWER_DIR)
            if not preview:
                head = trace.head(AUDIT_PREVIEW_ROWS)
                preview = _sanitize(
                    head.where(pd.notna(head), None).to_dict("records"))

        tracks_index[residue_label] = {
            "residue": residue_label,
            "track": residue_label,
            "normalization_method": "ForPerseus (log2 + per-run median-centered)",
            "row_count_per_sample": int(len(base)),
            "matched_site_count": int(gene_in_prot.sum()),
            "unmatched_site_count": int((~gene_in_prot).sum()),
            "sample_count": len(sample_files),
            "sample_files": sample_files,
            "preview": preview,
        }
        if not first_preview:
            first_preview = preview

    if not tracks_index:
        return None
    st_block = tracks_index.get("ST", {})
    return {
        "label": "Measurement trace",
        "normalization_method": "ForPerseus (log2 + per-run median-centered); "
                                "no pre-normalization raw channel for the T-cell "
                                "bulk track, so the trace starts at log2.",
        "default_track": "ST",
        "tracks": tracks_index,
        "row_count_per_sample": st_block.get("row_count_per_sample", 0),
        "matched_site_count": st_block.get("matched_site_count", 0),
        "unmatched_site_count": st_block.get("unmatched_site_count", 0),
        "sample_count": st_block.get("sample_count", 0),
        "columns": _MEASUREMENT_TRACE_COL_META,
        "preview": st_block.get("preview", first_preview),
        "sample_files": st_block.get("sample_files", {}),
        "relative_path": os.path.relpath(out_root, UNIFIED_VIEWER_DIR),
        "source_path": "derived from donor1 raw_phospho_normalized + "
                       "total_proteome_normalized + stoichiometry_matrix",
    }


# ---------------------------------------------------------------------------
# Transcript trace
# ---------------------------------------------------------------------------

def _write_tcell_transcript_trace() -> dict:
    """Generate per-donor, per-cluster transcript pseudobulk parquets.

    Returns {by_context: {<donor-context>: {clusters, relative_path}}, ...}.
    Donor scoping disambiguates clusters whose names appear in both donors
    (e.g. CD4Naive) but whose pseudobulk values differ.
    """
    rel_path = "audit_sources/transcript_trace"
    out_dir_base = os.path.join(UNIFIED_VIEWER_DIR, rel_path)

    by_context: dict[str, dict] = {}
    for donor in DONORS:
        agg_path = os.path.join(
            TCELLS_INCYTR_INPUTS_DIR, donor, "scrna", "aggexp_data.csv"
        )
        donor_rel = f"{rel_path}/{donor}"
        donor_out = os.path.join(UNIFIED_VIEWER_DIR, donor_rel)
        os.makedirs(donor_out, exist_ok=True)
        if not os.path.exists(agg_path):
            print(f"  ({donor}) no aggexp_data.csv; skipping transcript_trace",
                  flush=True)
            by_context[donor] = {"clusters": [], "relative_path": donor_rel}
            continue
        df = pd.read_csv(agg_path)
        if "gene" not in df.columns:
            print(f"  ({donor}) aggexp_data.csv missing `gene` column; skip",
                  flush=True)
            by_context[donor] = {"clusters": [], "relative_path": donor_rel}
            continue
        col_split: dict[str, list[tuple[str, str]]] = {}
        for c in df.columns:
            if c == "gene" or "__" not in c:
                continue
            cluster, day = c.rsplit("__", 1)
            col_split.setdefault(cluster, []).append((day, c))
        donor_slugs: list[str] = []
        for cluster, pairs in col_split.items():
            frames = []
            for day, col in pairs:
                sub = df[["gene", col]].rename(columns={col: "value"}).copy()
                sub["group"] = day
                frames.append(sub[["gene", "group", "value"]])
            long_df = pd.concat(frames, ignore_index=True)
            long_df = long_df.dropna(subset=["gene"])
            long_df = long_df[long_df["gene"].astype(str).str.len() > 0]
            slug = cluster  # already alphanumeric from extract pipeline
            out_path = os.path.join(donor_out, f"{slug}.parquet")
            pq.write_table(pa.Table.from_pandas(long_df, preserve_index=False),
                           out_path, compression="zstd")
            donor_slugs.append(slug)
        by_context[donor] = {
            "clusters": sorted(donor_slugs),
            "relative_path": donor_rel,
        }
        print(f"  ({donor}) wrote {len(donor_slugs)} transcript_trace shard(s)",
              flush=True)
    return {"by_context": by_context}


# ---------------------------------------------------------------------------
# Omics trace
# ---------------------------------------------------------------------------

def _parse_tcell_deconv_col(col: str) -> tuple[str, str] | None:
    if "_" not in col:
        return None
    group, cluster = col.split("_", 1)
    if not re.fullmatch(r"d\d+", group) or not cluster:
        return None
    return group, cluster


def _tcell_evidence_genes_by_cluster(donor: str) -> dict[str, set[str]]:
    """Return routed Incytr evidence genes per T-cell cluster for one donor."""
    index_path = os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, "index.json")
    if not os.path.exists(index_path):
        return {}
    with open(index_path) as f:
        idx = json.load(f)
    donor_idx = (idx.get("by_context") or {}).get(donor) or {}
    present = donor_idx.get("present") or []
    out: dict[str, set[str]] = {}

    def add(cluster: str, gene: object) -> None:
        if gene is None or pd.isna(gene):
            return
        gene_s = str(gene)
        if not gene_s:
            return
        out.setdefault(str(cluster), set()).add(gene_s)

    for pair in present:
        if len(pair) < 2:
            continue
        sender, receiver = str(pair[0]), str(pair[1])
        fname = (
            f"{donor}__{_incytr_sanitize(sender)}__"
            f"{_incytr_sanitize(receiver)}.parquet"
        )
        fpath = os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"{donor} incytr_pathways shard missing while building "
                f"omics_trace: {fpath}"
            )
        df = pq.read_table(
            fpath, columns=["Ligand", "Receptor", "EM", "Target"]
        ).to_pandas()
        for gene in df["Ligand"].dropna().unique():
            add(sender, gene)
        for col in ("Receptor", "EM", "Target"):
            for gene in df[col].dropna().unique():
                add(receiver, gene)
    return out


def _write_tcell_omics_trace() -> dict:
    """Generate per-donor, per-cluster protein/phospho evidence parquets."""
    import shutil
    rel_path = "audit_sources/omics_trace"
    by_context: dict[str, dict] = {}

    for donor in DONORS:
        evidence_genes = _tcell_evidence_genes_by_cluster(donor)
        n_cluster_genes = sum(len(v) for v in evidence_genes.values())
        if n_cluster_genes:
            print(
                f"  ({donor}) omics_trace routed evidence genes: "
                f"{n_cluster_genes:,} cluster-gene pairs",
                flush=True,
            )
        donor_rel = f"{rel_path}/{donor}"
        donor_out = os.path.join(UNIFIED_VIEWER_DIR, donor_rel)
        shutil.rmtree(donor_out, ignore_errors=True)
        os.makedirs(donor_out, exist_ok=True)

        layer_frames: list[tuple[str, pd.DataFrame, tuple[str, ...], dict[str, list[tuple[str, str]]]]] = []
        layer_names: list[str] = []
        clusters: set[str] = set()
        source_files: dict[str, str] = {}

        for layer, fname, key_cols in _TCELL_OMICS_LAYERS:
            src = os.path.join(TCELLS_INCYTR_INPUTS_DIR, donor, fname)
            if not os.path.exists(src):
                continue
            df = pd.read_csv(src)
            missing = [c for c in key_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"{donor} {fname} missing required column(s): {missing}"
                )
            by_cluster: dict[str, list[tuple[str, str]]] = {}
            key_set = set(key_cols)
            for c in df.columns:
                if c in key_set:
                    continue
                parsed = _parse_tcell_deconv_col(c)
                if parsed is None:
                    continue
                group, cluster = parsed
                by_cluster.setdefault(cluster, []).append((group, c))
                clusters.add(cluster)
            if not by_cluster:
                continue
            layer_frames.append((layer, df, key_cols, by_cluster))
            layer_names.append(layer)
            source_files[layer] = os.path.relpath(src, config.REPO_ROOT)

        shards_written: dict[str, str] = {}
        for cluster in sorted(clusters):
            frames: list[pd.DataFrame] = []
            for layer, df, key_cols, by_cluster in layer_frames:
                pairs = by_cluster.get(cluster) or []
                if not pairs:
                    continue
                base = df[list(key_cols)].copy()
                allowed_genes = evidence_genes.get(cluster)
                if allowed_genes:
                    base = base[
                        base["gene_symbol"].astype(str).isin(allowed_genes)
                    ].copy()
                if base.empty:
                    continue
                if "site_id" not in base.columns:
                    base["site_id"] = None
                if "motif" not in base.columns:
                    base["motif"] = None
                for group, col in pairs:
                    sub = base[["site_id", "gene_symbol", "motif"]].copy()
                    sub["layer"] = layer
                    sub["animal_id"] = f"{donor}_{group}_{cluster}"
                    sub["group"] = group
                    sub["sex"] = None
                    sub["timepoint"] = group
                    sub["genotype"] = None
                    sub["value"] = pd.to_numeric(df[col], errors="coerce")
                    sub = sub.dropna(subset=["value"])
                    sub["log2_value"] = np.where(
                        sub["value"] > 0, np.log2(sub["value"]), np.nan
                    )
                    frames.append(sub[_TCELL_OMICS_SHARD_COLS])
            if not frames:
                continue
            out = pd.concat(frames, ignore_index=True)
            slug = _incytr_sanitize(cluster)
            out_path = os.path.join(donor_out, f"{slug}.parquet")
            pq.write_table(
                pa.Table.from_pandas(out, preserve_index=False),
                out_path,
                compression="zstd",
            )
            shards_written[cluster] = os.path.relpath(out_path, UNIFIED_VIEWER_DIR)

        by_context[donor] = {
            "omics_schema_version": 1,
            "relative_path": donor_rel,
            "clusters": sorted(shards_written.keys()),
            "layers": layer_names,
            "filename_template": "{cluster}.parquet",
            "sanitize_rule": "replace('/', '-'); replace(' ', '_'); replace('.', '')",
            "source_files": source_files,
            "shard_files": shards_written,
            "n_shards": len(shards_written),
            "gene_scope": (
                "routed_incytr_pathway_evidence_genes"
                if n_cluster_genes else "all_deconvoluted_genes"
            ),
            "n_routed_cluster_gene_pairs": n_cluster_genes,
            "n_libraries_per_arm": 1,
            "note": (
                "T-cell values are deconvoluted cluster abundance estimates; "
                "one value is available per donor/day/cluster arm."
            ),
        }
        print(f"  ({donor}) wrote {len(shards_written)} omics_trace shard(s)",
              flush=True)

    return {"by_context": by_context}
