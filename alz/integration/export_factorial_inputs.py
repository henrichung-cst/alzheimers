"""Export Song snRNA-seq h5ad to a labeled fixture for the upstream Incytr package.

The fixture is a curation-only deliverable: this repo prepares inputs and
labels them; performance optimization (Hill cutoffs, max_join_rows, candidate
construction strategy) belongs in ../incytr, not here.

Layout written to OUT_DIR:
  expression_matrix.mtx            genes x cells, sparse
  expression_genes.csv             gene symbol per row
  expression_barcodes.csv          barcode per cell
  expression_metadata.csv          per-cell labels, animal_id, genotype, timepoint
  animal_metadata.csv              per-animal design matrix matching kinase_enrich.py OLS
  MANIFEST.json                    provenance, filter, dimensions, vocab, contrasts
  README.md                        consumer-facing description
  subset_astro_microglia/          same 5-file shape, two Levy-19 clusters (fast iteration)
    expression_matrix.mtx
    expression_genes.csv
    expression_barcodes.csv
    expression_metadata.csv
    animal_metadata.csv
    MANIFEST.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "alz"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as main_config  # noqa: E402
import config_integration as icfg  # noqa: E402

# Two Levy-19 clusters used for the fast iteration subset bundle. Picked for
# wide coverage and non-neuronal class diversity so upstream Incytr profiling
# work hits both glial and excitatory paths. Re-confirm post-export if cell
# counts shift.
SUBSET_NAME = "subset_astro_microglia"
SUBSET_LABELS = ("Astrocytes", "Microglia")

# Source kldata.csv (incytr-schema, mouse-mapped) bundled into the
# fixture so downstream Integr_kinasedata has its inputs alongside the
# transcript inputs. The kinase library is a chemical/biochemical
# substrate map (PSPA motif specificity), not study-specific, so the
# 5xFAD-derived file applies to Song.
KLDATA_SOURCE_REL = os.path.join(
    "data", "datasets", "5xFAD", "kinase", "kldata_pspy.csv"
)


def build_design_row(genotype: str, timepoint: str) -> dict:
    geno = icfg.MUTANT_TO_DESIGN[genotype]
    time = icfg.TIMEPOINT_TO_DESIGN[timepoint]
    row = {"const": 1}
    row.update(geno)
    row.update(time)
    row["App_x_time4"] = geno["App"] * time["time_4mo"]
    row["App_x_time6"] = geno["App"] * time["time_6mo"]
    row["Tau_x_time4"] = geno["Tau"] * time["time_4mo"]
    row["Tau_x_time6"] = geno["Tau"] * time["time_6mo"]
    return row


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _long_to_per_cluster_wide(
    long: pd.DataFrame, value_col: str, gene_key: str,
) -> dict[str, pd.DataFrame]:
    """Pivot Stage 6 long parquet to {cluster: gene × animal wide matrix}.

    For phospho input, ``gene_key`` is collapsed by sum(min_count=1) to match
    the legacy site→gene aggregator before pivoting; protein input is already
    keyed at gene level.
    """
    if "site_id" in long.columns:
        long = (
            long.groupby([gene_key, "cluster", "animal_id"], as_index=False)[value_col]
            .sum(min_count=1)
        )
    out: dict[str, pd.DataFrame] = {}
    for cluster, sub in long.groupby("cluster"):
        wide = sub.pivot_table(
            index=gene_key, columns="animal_id", values=value_col, aggfunc="first",
        )
        wide = wide.sort_index().sort_index(axis=1)
        out[cluster] = wide
    return out


def _sanitize_cluster(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


def write_per_cluster_bundle(
    out_dir: str,
    kept_animal_ids: list[str],
) -> dict[str, dict]:
    """Write per-cluster pr/ps parquet bundles under per_cluster/{layer}/.

    Sources Stage 6 outputs at
    ``outputs/reports/decomposition/{spine}/{protein,phospho}_per_cluster.parquet``.
    Restricts animal columns to ``kept_animal_ids`` so design rownames align.
    The py layer is written when a track-suffixed Stage 6 parquet
    (``phospho_per_cluster_pY.parquet``) is present; absent otherwise (Stage 6
    pY extension is Step 13 work).

    Returns a per-layer summary dict for the manifest.
    """
    summary: dict[str, dict] = {}
    layer_specs = [
        ("pr", icfg.PROTEIN_PER_CLUSTER_FILE, "gene_symbol"),
        ("ps", icfg.PHOSPHO_PER_CLUSTER_FILE, "gene_symbol"),
        (
            "py",
            os.path.join(icfg.DECOMPOSITION_DIR, "phospho_per_cluster_pY.parquet"),
            "gene_symbol",
        ),
    ]
    kept_set = set(kept_animal_ids)
    for layer, src, gene_key in layer_specs:
        if not os.path.exists(src):
            print(f"    [{layer}] skipped: {src} not found")
            summary[layer] = {"source": src, "status": "missing"}
            continue
        long = pd.read_parquet(src)
        wide_by_cluster = _long_to_per_cluster_wide(long, "value", gene_key)
        layer_dir = os.path.join(out_dir, "per_cluster", layer)
        os.makedirs(layer_dir, exist_ok=True)
        clusters_written = 0
        n_genes_total = 0
        for cluster, wide in wide_by_cluster.items():
            cols = [c for c in wide.columns if c in kept_set]
            sub = wide.reindex(columns=kept_animal_ids)
            sub = sub.loc[sub.notna().any(axis=1)]
            if sub.empty:
                continue
            fname = f"{_sanitize_cluster(cluster)}.parquet"
            sub.reset_index().to_parquet(
                os.path.join(layer_dir, fname), index=False,
            )
            clusters_written += 1
            n_genes_total = max(n_genes_total, sub.shape[0])
        print(
            f"    [{layer}] per_cluster/{layer}/: {clusters_written} clusters, "
            f"≤{n_genes_total} genes per cluster"
        )
        summary[layer] = {
            "source": os.path.relpath(src, REPO_ROOT),
            "n_clusters": clusters_written,
            "max_n_genes": int(n_genes_total),
        }
    return summary


def write_seedlist_pr_matrix(
    out_dir: str,
    kept_animal_ids: list[str],
) -> str | None:
    """Write a flat bulk pr_matrix.csv (genes × animals) for compute_seed_lists.R.

    Seed-list DEP estimation (limma on bulk proteomics) is upstream of the
    per-cluster decomposition; it should stay flat. Source is the same
    ``total_proteome_normalized.csv`` that Stage 1 emits; columns are TMT
    ``column_name`` values that we map to transcript-side animal_ids via
    ``sample_mapping.csv``.
    """
    src = os.path.join(
        REPO_ROOT, "outputs/reports/kinase_attribution/total_proteome_normalized.csv"
    )
    smap_path = os.path.join(REPO_ROOT, "outputs/reports/data_ingest/sample_mapping.csv")
    if not (os.path.exists(src) and os.path.exists(smap_path)):
        print(f"    [seed-list pr_matrix.csv] skipped (missing {src} or {smap_path})")
        return None
    tp = pd.read_csv(src)
    smap = pd.read_csv(smap_path)
    col_to_animal = dict(zip(smap["column_name"], smap["animal_id"]))
    meta_cols = [c for c in ("gene_symbol", "protein_id") if c in tp.columns]
    val_cols = [c for c in tp.columns if c not in meta_cols and c in col_to_animal]
    flat = tp[["gene_symbol"] + val_cols].dropna(subset=["gene_symbol"]).copy()
    flat = flat.rename(columns={c: col_to_animal[c] for c in val_cols})
    # one row per gene (first occurrence; total proteome ≈ unique)
    flat = flat.drop_duplicates(subset=["gene_symbol"], keep="first")
    flat = flat.set_index("gene_symbol")
    flat = flat.reindex(columns=kept_animal_ids)
    path = os.path.join(out_dir, "pr_matrix.csv")
    flat.to_csv(path, index_label="gene")
    print(f"    wrote {path}  ({flat.shape[0]} genes x {flat.shape[1]} animals; "
          f"seed-list limma input)")
    return path


def write_pseudobulk_counts(
    out_dir: str,
    adata,
    meta: pd.DataFrame,
) -> dict:
    """Write per-(animal × cell type) pseudobulk raw counts for DESeq2.

    Uses ``adata.layers["counts"]`` (integer UMIs); the fixture's
    ``expression_matrix.mtx`` writes ``adata.X`` which is normalized.
    Pre-intersect transcript set so DEG estimation gets all male
    transcript animals, not just the 4-layer omics intersect.
    """
    if "counts" not in adata.layers:
        raise SystemExit(
            "Source h5ad has no 'counts' layer; cannot pseudobulk raw "
            "UMIs for DESeq2. Refusing to write seed-list inputs."
        )
    counts = adata.layers["counts"]
    if not scipy.sparse.issparse(counts):
        counts = scipy.sparse.csr_matrix(counts)
    # Snap float-typed counts to int (some pipelines store UMIs as float32)
    # so the selector @ counts matmul stays in integer space; saves a
    # transient float64 copy of the entire ~63K x 30K matrix.
    if not np.issubdtype(counts.dtype, np.integer):
        counts = counts.copy()
        counts.data = np.rint(counts.data).astype(np.int32)

    group_keys = (
        meta["animal_id"].astype(str) + "__" + meta["labels"].astype(str)
    )
    cat = pd.Categorical(group_keys)
    n_groups = len(cat.categories)
    n_cells = adata.n_obs
    selector = scipy.sparse.csr_matrix(
        (np.ones(n_cells, dtype=np.int32), (cat.codes, np.arange(n_cells))),
        shape=(n_groups, n_cells),
    )
    pseudobulk = (selector @ counts).tocsr()
    pseudobulk.eliminate_zeros()

    grouped = (
        meta.assign(pseudosample=group_keys.values)
        .groupby("pseudosample", sort=False)
        .agg(
            animal_id=("animal_id", "first"),
            genotype=("genotype", "first"),
            timepoint=("timepoint", "first"),
            celltype=("labels", "first"),
            n_cells=("animal_id", "size"),
        )
        .loc[list(cat.categories)]
        .reset_index()
    )
    pb_meta = grouped[
        ["pseudosample", "animal_id", "celltype", "genotype", "timepoint", "n_cells"]
    ]

    mtx_path = os.path.join(out_dir, "pseudobulk_counts.mtx")
    samples_path = os.path.join(out_dir, "pseudobulk_pseudosamples.csv")
    genes_path = os.path.join(out_dir, "pseudobulk_genes.csv")
    meta_path = os.path.join(out_dir, "pseudobulk_metadata.csv")

    scipy.io.mmwrite(mtx_path, pseudobulk.T, field="integer", symmetry="general")
    pd.DataFrame({"pseudosample": list(cat.categories)}).to_csv(samples_path, index=False)
    pd.DataFrame({"gene": adata.var_names}).to_csv(genes_path, index=False)
    pb_meta.to_csv(meta_path, index=False)

    for p in (mtx_path, samples_path, genes_path, meta_path):
        print(f"    wrote {p}")
    return {
        "n_pseudosamples": int(n_groups),
        "n_genes": int(adata.n_vars),
        "n_celltypes": int(pb_meta["celltype"].nunique()),
        "n_animals": int(pb_meta["animal_id"].nunique()),
    }


def write_kldata(out_dir: str) -> str:
    """Copy the bundled kinase library into the fixture."""
    import shutil
    src = os.path.join(REPO_ROOT, KLDATA_SOURCE_REL)
    if not os.path.isfile(src):
        raise FileNotFoundError(
            f"Bundled kldata source not found: {src}. "
            f"Cannot ship factorial fixture without kinase library."
        )
    dst = os.path.join(out_dir, "kldata.csv")
    shutil.copy2(src, dst)
    print(f"    wrote {dst}  (copied from {KLDATA_SOURCE_REL})")
    return dst


def assert_factorial_estimability(animal_meta: pd.DataFrame) -> None:
    """Pre-write asserts: rank-full + all 9 contrasts c'(X'X)^-1 nonzero."""
    X = animal_meta[icfg.DESIGN_COLUMNS].values.astype(float)
    rank = int(np.linalg.matrix_rank(X))
    if rank != len(icfg.DESIGN_COLUMNS):
        raise SystemExit(
            f"Design rank {rank} != {len(icfg.DESIGN_COLUMNS)} on intersect; "
            f"refusing to write."
        )
    XtX_inv = np.linalg.pinv(X.T @ X)
    for name, c in icfg.FACTORIAL_CONTRASTS.items():
        c_arr = np.asarray(c, dtype=float)
        var_factor = float(c_arr @ XtX_inv @ c_arr)
        if not np.isfinite(var_factor) or var_factor <= 0:
            raise SystemExit(
                f"Contrast {name} is not estimable on intersect "
                f"(c'(X'X)^-1 c = {var_factor})"
            )


def write_bundle(out_dir: str, adata, meta: pd.DataFrame, animal_meta: pd.DataFrame) -> None:
    os.makedirs(out_dir, exist_ok=True)

    X_t = adata.X.T
    if not scipy.sparse.issparse(X_t):
        X_t = scipy.sparse.csr_matrix(X_t)

    mtx_path = os.path.join(out_dir, "expression_matrix.mtx")
    genes_path = os.path.join(out_dir, "expression_genes.csv")
    barcodes_path = os.path.join(out_dir, "expression_barcodes.csv")
    meta_path = os.path.join(out_dir, "expression_metadata.csv")
    animal_path = os.path.join(out_dir, "animal_metadata.csv")

    scipy.io.mmwrite(mtx_path, X_t)
    pd.DataFrame({"gene": adata.var_names}).to_csv(genes_path, index=False)
    pd.DataFrame({"barcode": adata.obs_names}).to_csv(barcodes_path, index=False)
    meta.to_csv(meta_path, index=True)
    animal_meta.to_csv(animal_path, index=False)

    for p in (mtx_path, genes_path, barcodes_path, meta_path, animal_path):
        print(f"    wrote {p}")


def build_manifest(
    *,
    scope: str,
    adata,
    meta: pd.DataFrame,
    animal_meta: pd.DataFrame,
    h5ad_path: str,
    parent_subset_of: str | None = None,
) -> dict:
    label_counts = meta["labels"].value_counts().sort_values(ascending=False).to_dict()
    counts = (
        meta.groupby(["genotype", "timepoint"], observed=True)
        .agg(n_cells=("animal_id", "size"), n_animals=("animal_id", "nunique"))
        .reset_index()
    )

    manifest: dict = {
        "fixture_kind": "incytr_factorial_input",
        "scope": scope,
        "purpose": (
            "Frozen labeled fixture for upstream Incytr factorial path "
            "construction/scoring. This repo curates; optimization (Hill cutoffs, "
            "max_join_rows, candidate-construction strategy) belongs in ../incytr."
        ),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "generator": "alz/integration/export_factorial_inputs.py",
        "alz_repo_git_sha": git_sha(),
        "source": {
            "h5ad_path": h5ad_path,
            "subclass_prob_min": float(main_config.SONG_MIN_SUBCLASS_PROB),
            "cluster_spine": icfg.CLUSTER_SPINE,
            "cluster_spine_file": os.path.relpath(
                icfg.CLUSTER_SPINE_FILE, REPO_ROOT,
            ),
            "barcode_to_cluster_file": os.path.relpath(
                icfg.BARCODE_TO_CLUSTER_FILE, REPO_ROOT,
            ),
            "celltype_taxonomy": (
                f"Levy-19 strict spine (alz/integration/build_cluster_spine.py); "
                f"in_spine == True in {icfg.CLUSTER_SPINE_FILE}"
            ),
        },
        "filter": {
            "sex": icfg.FACTORIAL_SEX,
            "genotypes": list(icfg.FACTORIAL_GENOTYPES),
            "timepoints": list(icfg.FACTORIAL_TIMEPOINTS),
        },
    }
    if parent_subset_of:
        manifest["parent_subset_of"] = parent_subset_of
        manifest["subset_labels"] = sorted(meta["labels"].unique().tolist())

    manifest["dimensions"] = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_animals": int(animal_meta.shape[0]),
        "n_labels": int(meta["labels"].nunique()),
    }
    manifest["label_vocabulary"] = {str(k): int(v) for k, v in label_counts.items()}
    manifest["per_condition_cell_counts"] = [
        {
            "genotype": row.genotype,
            "timepoint": row.timepoint,
            "n_cells": int(row.n_cells),
            "n_animals": int(row.n_animals),
        }
        for row in counts.itertuples()
    ]
    manifest["design_columns"] = list(icfg.DESIGN_COLUMNS)
    manifest["contrasts"] = {k: list(v) for k, v in icfg.FACTORIAL_CONTRASTS.items()}
    manifest["contrast_conditions"] = {
        k: list(v) for k, v in icfg.FACTORIAL_CONTRAST_CONDITIONS.items()
    }
    manifest["files"] = {
        "expression_matrix.mtx": "Sparse MatrixMarket: rows=genes, cols=cells.",
        "expression_genes.csv": "Single column 'gene'; row order matches matrix rows.",
        "expression_barcodes.csv": (
            "Single column 'barcode'; column order matches matrix cols."
        ),
        "expression_metadata.csv": (
            "Per-cell metadata indexed by barcode. Columns: labels (cell type), "
            "animal_id, genotype, timepoint."
        ),
        "animal_metadata.csv": (
            "Per-animal design matrix. id_cols + DESIGN_COLUMNS. "
            "Use animal_id as design rownames; column order matches "
            "DESIGN_COLUMNS in MANIFEST."
        ),
        "per_cluster/pr/<cluster>.parquet": (
            "Per-cluster bulk proteomics (forward-projected onto Stage 5 "
            "per-cell-rate proportions, linear space). Columns: gene_symbol "
            "(row index) + animal_ids matching animal_metadata.csv. One "
            "parquet file per Levy-19 cluster; load.R assembles "
            "list(data_wide = list(cluster -> matrix)) for upstream "
            "resolve_wide list-dispatch."
        ),
        "per_cluster/ps/<cluster>.parquet": (
            "Per-cluster pSer/pThr phosphoproteomics (Stage 6 projection "
            "of bulk raw_phospho_normalized.csv). Sites collapsed to gene "
            "level by sum(min_count=1) before pivoting. Same column "
            "convention as per_cluster/pr/."
        ),
        "per_cluster/py/<cluster>.parquet": (
            "Per-cluster pTyr phosphoproteomics. Present only when Stage 6 "
            "has been run for the pY track (phospho_per_cluster_pY.parquet)."
        ),
        "pr_matrix.csv": (
            "Flat bulk proteomics (genes × animals) — sidecar for "
            "compute_seed_lists.R only. Sourced from total_proteome_normalized.csv "
            "with TMT column_name → transcript-side animal_id mapping via "
            "sample_mapping.csv. Not consumed by the factorial path."
        ),
        "kldata.csv": (
            "Kinase library substrate map (incytr schema: gene, site_pos, "
            "motif.geneName, Type). Copied from 5xFAD/kinase/kldata_pspy.csv; "
            "the kinase library is a chemical motif map and is not "
            "study-specific."
        ),
        "pseudobulk_counts.mtx": (
            "Sparse MatrixMarket integer counts: rows=genes, cols=pseudosamples. "
            "Per-(animal x cell type) summed UMIs from layers['counts']. "
            "Pre-omics-intersect; uses all male transcript animals so DESeq2 "
            "has the broadest replicate base. Consumed by compute_seed_lists.R."
        ),
        "pseudobulk_pseudosamples.csv": (
            "Single column 'pseudosample'; column order matches "
            "pseudobulk_counts.mtx columns. Format '<animal_id>__<celltype>'."
        ),
        "pseudobulk_genes.csv": (
            "Single column 'gene'; row order matches pseudobulk_counts.mtx "
            "rows. Same vocabulary as expression_genes.csv."
        ),
        "pseudobulk_metadata.csv": (
            "Per-pseudosample metadata: pseudosample, animal_id, celltype, "
            "genotype, timepoint, n_cells. Used by compute_seed_lists.R for "
            "DESeq2 design construction and the >=10-cell filter."
        ),
    }
    manifest["consumer_contract"] = {
        "construct_factorial_paths": (
            "Pass expression (genes x cells dgCMatrix), metadata "
            "(rownames = barcodes), senders/receivers from "
            "label_vocabulary, group.by='labels'. Optimization knobs "
            "(expression_threshold, cutoff_SigProb, max_join_rows, K, N) "
            "are upstream's responsibility."
        ),
        "score_factorial_paths": (
            "Use design = animal_metadata[, DESIGN_COLUMNS] with "
            "rownames = animal_id, contrasts = MANIFEST.contrasts, "
            "animal_id='animal_id', condition_col='condition' "
            "(synthesized in load.R as paste(genotype, timepoint, sep='_'))."
        ),
    }
    manifest["non_goals"] = [
        "No caps or thresholds applied here. The fixture is unfiltered at the "
        "Incytr-construction layer.",
        "No scoring is run from this repo. score_factorial_paths is invoked "
        "downstream of upstream-side optimization.",
        "No backflow. Outputs of upstream scoring write to "
        "outputs/reports/incytr_factorial/, not back into this fixture.",
    ]
    return manifest


def write_manifest(out_dir: str, manifest: dict) -> None:
    path = os.path.join(out_dir, "MANIFEST.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"    wrote {path}")


def write_readme(out_dir: str) -> None:
    path = os.path.join(out_dir, "README.md")
    body = (
        "# Incytr factorial input fixture\n\n"
        "Frozen labeled fixture for the upstream `incytr` R package. This\n"
        "directory is the AD-side curation deliverable; candidate-construction\n"
        "caps, Hill cutoffs, and scoring strategy belong upstream in\n"
        "`~/Projects/work/incytr/`, not here.\n\n"
        "**Transcript inputs:** `expression_matrix.mtx` (sparse, genes × "
        "cells), `expression_genes.csv`, `expression_barcodes.csv`, "
        "`expression_metadata.csv` (per-cell), `animal_metadata.csv` "
        "(per-animal design).\n\n"
        "**Per-cluster omics inputs (multi-omic factorial bridge, "
        "Levy-19 spine):** `per_cluster/pr/<cluster>.parquet` (forward-"
        "projected bulk proteomics), `per_cluster/ps/<cluster>.parquet` "
        "(pSer/pThr; sites summed to gene), "
        "`per_cluster/py/<cluster>.parquet` (pTyr; present when Stage 6 "
        "was run for the pY track), `kldata.csv` (kinase library, incytr "
        "schema). `pr_matrix.csv` is a flat bulk sidecar for "
        "`compute_seed_lists.R` only.\n\n"
        "**Seed-list DE inputs:** `pseudobulk_counts.mtx` (genes × "
        "(animal × cell type) raw integer UMIs from "
        "`adata.layers['counts']`), `pseudobulk_pseudosamples.csv`, "
        "`pseudobulk_genes.csv`, `pseudobulk_metadata.csv`. These are "
        "consumed by `compute_seed_lists.R` to produce `deg_lists.json` "
        "(per-cell-type DEGs from DESeq2 on `~ 0 + group`, union over the "
        "9 App/Tau/ApTt × 2/4/6mo contrasts; ≥10 cells per "
        "(animal × cell type), ≥3 reps per group, padj<0.05) and "
        "`prg_list.csv` (full DEP set from limma on `pr_matrix.csv` with "
        "the same design + contrasts). Strict-partition (DEG-first; "
        "prG = DEP \\ DEG) is applied upstream at label-assignment time. "
        "Pseudobulk uses the *pre-omics-intersect* transcript animal set "
        "to maximize replicate count for DESeq2.\n\n"
        "All transcript/proteomics matrices share the same animal column "
        "order, matching `animal_metadata.csv` row order. Animals lacking "
        "coverage in any layer are excluded from the fixture (see "
        "`MANIFEST.multiomic_coverage`).\n\n"
        f"`{SUBSET_NAME}/` — same shape restricted to "
        f"{', '.join(SUBSET_LABELS)} (Levy-19 spine) for fast upstream "
        f"iteration.\n\n"
        "See `MANIFEST.json` for full provenance, filter values, dimensions, "
        "label vocabulary, design columns, contrast vectors, multi-omic "
        "coverage notes, and the consumer contract for "
        "`Incytr::construct_factorial_paths` / `Incytr::score_factorial_paths`.\n\n"
        "Built by `alz/integration/export_factorial_inputs.py`.\n"
    )
    with open(path, "w") as fh:
        fh.write(body)
    print(f"    wrote {path}")


def write_subset(
    *,
    parent_dir: str,
    subset_name: str,
    labels: tuple[str, ...],
    adata,
    meta: pd.DataFrame,
    animal_meta_full: pd.DataFrame,
    h5ad_path: str,
) -> None:
    print(f"\n  Subset bundle '{subset_name}' (labels={list(labels)})")
    cell_mask = meta["labels"].isin(labels)
    if not cell_mask.any():
        print("    no cells matched; skipping subset")
        return
    sub_meta = meta.loc[cell_mask].copy()
    sub_adata = adata[cell_mask.values].copy()

    represented_animals = set(sub_meta["animal_id"].unique())
    sub_animal_meta = animal_meta_full[
        animal_meta_full["animal_id"].isin(represented_animals)
    ].reset_index(drop=True)
    print(
        f"    {sub_adata.n_obs} cells x {sub_adata.n_vars} genes; "
        f"{len(sub_animal_meta)} animals"
    )

    rank = np.linalg.matrix_rank(
        sub_animal_meta[icfg.DESIGN_COLUMNS].values.astype(float)
    )
    print(f"    Subset design matrix rank: {rank} / {len(icfg.DESIGN_COLUMNS)}")
    if rank < len(icfg.DESIGN_COLUMNS):
        print(
            "    NOTE: subset design is rank-deficient (expected — small subset). "
            "Subset is intended for construction-time profiling, not OLS scoring."
        )

    out_dir = os.path.join(parent_dir, subset_name)
    write_bundle(out_dir, sub_adata, sub_meta, sub_animal_meta)
    manifest = build_manifest(
        scope=f"subset:{subset_name}",
        adata=sub_adata,
        meta=sub_meta,
        animal_meta=sub_animal_meta,
        h5ad_path=h5ad_path,
        parent_subset_of=os.path.relpath(parent_dir, os.path.dirname(out_dir)),
    )
    manifest["subset_design_rank"] = int(rank)
    manifest["subset_intended_use"] = (
        "Fast iteration on construct_factorial_paths during upstream "
        "performance work. Not guaranteed to support score_factorial_paths "
        "if the subset's design matrix is rank-deficient."
    )
    write_manifest(out_dir, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=icfg.FACTORIAL_INPUT_DIR)
    parser.add_argument("--h5ad", default=icfg.H5AD_PATH)
    parser.add_argument(
        "--no-subset",
        action="store_true",
        help="Skip the subset_immune_astro bundle.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.h5ad} ...")
    adata = ad.read_h5ad(args.h5ad)
    print(f"  Full dataset: {adata.n_obs} cells x {adata.n_vars} genes")

    sex_geno_time = (
        (adata.obs["sex"] == icfg.FACTORIAL_SEX)
        & (adata.obs["mutant"].isin(icfg.FACTORIAL_GENOTYPES))
        & (adata.obs["age"].isin(icfg.FACTORIAL_TIMEPOINTS))
    )
    print(f"  After sex/genotype/timepoint filter: {int(sex_geno_time.sum())} cells")

    if "subclass_prob" in adata.obs.columns:
        prob_mask = adata.obs["subclass_prob"] >= main_config.SONG_MIN_SUBCLASS_PROB
        print(
            f"  After subclass_prob >= {main_config.SONG_MIN_SUBCLASS_PROB}: "
            f"{int((sex_geno_time & prob_mask).sum())} cells"
        )
    else:
        prob_mask = pd.Series(True, index=adata.obs.index)

    # Levy-19 cluster spine: per-cluster pivot replaces WMB-34 labels.
    # barcode_to_cluster.csv stores seurat_cluster_id; the Cluster ID → New_ID
    # key resolves to the human-readable cluster_name used downstream.
    bc_path = icfg.BARCODE_TO_CLUSTER_FILE
    key_path = os.path.join(
        REPO_ROOT, "data/incytr/v2_46clusters/provenance/kr_cluster_id_key.csv"
    )
    if not (os.path.exists(bc_path) and os.path.exists(key_path)):
        raise SystemExit(
            f"Missing Levy cluster assignments: {bc_path} or {key_path}. "
            "Run alz/integration/extract_cluster_assignments.R first."
        )
    bc_df = pd.read_csv(bc_path)
    key_df = pd.read_csv(key_path)
    name_for_id = dict(zip(key_df["Cluster ID"].astype(int), key_df["New_ID"].astype(str)))
    bc_df["cluster_name"] = bc_df["seurat_cluster_id"].map(name_for_id)
    if bc_df["cluster_name"].isna().any():
        raise SystemExit("seurat_cluster_id missing from kr_cluster_id_key.csv")
    barcode_to_name = dict(zip(bc_df["barcode"].astype(str), bc_df["cluster_name"]))

    spine_names = icfg.load_cluster_spine()
    spine_set = set(spine_names)
    print(f"  Levy-{len(spine_names)} cluster spine: {spine_names}")

    pre_map = sex_geno_time & prob_mask
    bc_labels = adata.obs_names.astype(str).map(barcode_to_name)
    unmapped = int((pre_map & bc_labels.isna()).sum())
    if unmapped > 0:
        print(f"  Dropping {unmapped} cells without a barcode→cluster assignment")

    in_spine = bc_labels.isin(spine_set)
    off_spine = int((pre_map & bc_labels.notna() & ~in_spine).sum())
    if off_spine > 0:
        print(f"  Dropping {off_spine} cells in off-spine clusters (27/46 rejected)")

    keep = pre_map & bc_labels.notna() & in_spine
    adata = adata[keep].copy()
    cluster_labels = pd.Series(
        adata.obs_names.astype(str).map(barcode_to_name).values,
        index=adata.obs.index,
    )
    print(f"  After Levy-{len(spine_names)} spine filter: {adata.n_obs} cells")

    meta = pd.DataFrame(index=adata.obs.index)
    meta["labels"] = cluster_labels.values
    meta["animal_id"] = adata.obs["sample"].values
    meta["genotype"] = adata.obs["mutant"].values
    meta["timepoint"] = adata.obs["age"].values

    animals = meta[["animal_id", "genotype", "timepoint"]].drop_duplicates()
    print(f"\n  Animals: {len(animals)}")
    for geno in icfg.FACTORIAL_GENOTYPES:
        for tp in icfg.FACTORIAL_TIMEPOINTS:
            n_cells = ((meta["genotype"] == geno) & (meta["timepoint"] == tp)).sum()
            n_animals = animals[
                (animals["genotype"] == geno) & (animals["timepoint"] == tp)
            ].shape[0]
            print(f"    {geno:>4s} x {tp}: {n_cells:>5d} cells, {n_animals} animals")

    animal_meta_rows = []
    for _, row in animals.iterrows():
        design_row = build_design_row(row["genotype"], row["timepoint"])
        design_row["animal_id"] = row["animal_id"]
        design_row["genotype"] = row["genotype"]
        design_row["timepoint"] = row["timepoint"]
        animal_meta_rows.append(design_row)

    animal_meta = pd.DataFrame(animal_meta_rows)
    id_cols = ["animal_id", "genotype", "timepoint"]
    animal_meta = animal_meta[id_cols + icfg.DESIGN_COLUMNS]
    animal_meta = animal_meta.sort_values("animal_id").reset_index(drop=True)

    rank = np.linalg.matrix_rank(animal_meta[icfg.DESIGN_COLUMNS].values.astype(float))
    print(
        f"\n  Design matrix (pre-intersect): {len(animal_meta)} animals x "
        f"{len(icfg.DESIGN_COLUMNS)} parameters, rank {rank}"
    )
    if rank < len(icfg.DESIGN_COLUMNS):
        raise SystemExit("Design matrix is rank-deficient; refusing to write outputs.")

    # Per-cluster omics: source is Stage 6 (alz/decomposition/build_celltype_decomposition.py)
    # which projects bulk phospho + protein onto Stage 5 per-cell-rate proportions.
    # The transcript animal vocabulary is authoritative; Stage 6 already keys
    # on transcript-side animal_id, so we keep the full transcript set and let
    # per-cluster parquet columns drop animals absent from the decomposition.
    kept_ids = animal_meta["animal_id"].tolist()
    animal_meta_intersect = animal_meta.copy().reset_index(drop=True)
    assert_factorial_estimability(animal_meta_intersect)
    print(
        f"  Asserts: rank-full {len(icfg.DESIGN_COLUMNS)}, "
        f"all {len(icfg.FACTORIAL_CONTRASTS)} contrasts estimable across "
        f"{len(kept_ids)} transcript animals."
    )

    adata_intersect = adata
    meta_intersect = meta

    print(f"\n  Full bundle -> {args.out_dir}")
    write_bundle(args.out_dir, adata_intersect, meta_intersect, animal_meta_intersect)
    print("\n  Per-cluster parquet bundles (Stage 6 source):")
    per_cluster_summary = write_per_cluster_bundle(args.out_dir, kept_ids)
    write_seedlist_pr_matrix(args.out_dir, kept_ids)
    write_kldata(args.out_dir)

    print("\n  Pseudobulk raw counts (pre-intersect, all male transcript animals)...")
    pseudobulk_summary = write_pseudobulk_counts(args.out_dir, adata, meta)
    print(
        f"    pseudobulk: {pseudobulk_summary['n_pseudosamples']} samples "
        f"({pseudobulk_summary['n_animals']} animals x "
        f"{pseudobulk_summary['n_celltypes']} cell types)"
    )

    manifest = build_manifest(
        scope="full",
        adata=adata_intersect,
        meta=meta_intersect,
        animal_meta=animal_meta_intersect,
        h5ad_path=args.h5ad,
    )
    manifest["multiomic_coverage"] = {
        "n_animals_transcript": int(len(kept_ids)),
        "per_cluster_summary": per_cluster_summary,
        "site_to_gene_collapse": "sum",
        "id_keying": (
            "transcript-side animal_id is authoritative. Stage 6 "
            "(alz/decomposition/build_celltype_decomposition.py) projects "
            "bulk values onto Stage 5 proportions keyed on the same animal_id "
            "vocabulary, so no canonicalization step runs here."
        ),
        "kldata_source": KLDATA_SOURCE_REL,
        "py_layer_status": (
            "Per-cluster pY parquet is written only when Stage 6 has been "
            "run with --track py; extension is Step 13 work."
        ),
    }
    write_manifest(args.out_dir, manifest)
    write_readme(args.out_dir)

    if not args.no_subset:
        write_subset(
            parent_dir=args.out_dir,
            subset_name=SUBSET_NAME,
            labels=SUBSET_LABELS,
            adata=adata_intersect,
            meta=meta_intersect,
            animal_meta_full=animal_meta_intersect,
            h5ad_path=args.h5ad,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
