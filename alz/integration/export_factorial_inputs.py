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
  subset_microglia_astrocyte/      same 5-file shape, two cell types only (fast iteration)
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
import omics_loaders  # noqa: E402

SUBSET_NAME = "subset_microglia_astrocyte"
SUBSET_LABELS = ("Microglia-PVM", "Astrocyte")

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


def write_omics_bundle(
    out_dir: str,
    rekeyed_omics: dict[str, pd.DataFrame],
) -> dict[str, str]:
    """Write pr/ps/py matrices as CSV (genes × animals). Returns paths."""
    written: dict[str, str] = {}
    for layer, df in rekeyed_omics.items():
        path = os.path.join(out_dir, f"{layer}_matrix.csv")
        df.to_csv(path, index_label="gene")
        written[layer] = path
        print(f"    wrote {path}  ({df.shape[0]} genes x {df.shape[1]} animals)")
    return written


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


def assert_omics_alignment(
    rekeyed_omics: dict[str, pd.DataFrame],
    expected_animal_ids: list[str],
) -> None:
    """Pre-write asserts: every layer aligned to the intersect animal set."""
    for layer, df in rekeyed_omics.items():
        if list(df.columns) != expected_animal_ids:
            raise SystemExit(
                f"{layer}_matrix columns do not match expected animal order: "
                f"got {list(df.columns)}, expected {expected_animal_ids}"
            )
        all_nan_cols = [c for c in df.columns if df[c].isna().all()]
        if all_nan_cols:
            raise SystemExit(
                f"{layer}_matrix has all-NaN columns: {all_nan_cols}"
            )


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
            "wmb_class_map_source": "alz.config.load_song_to_wmb_class_map()",
            "wmb_class_map_file": main_config.WMB_SUBCLASS_TO_CLASS_FILE,
            "wmb_class_map_n_entries": len(main_config.load_song_to_wmb_class_map()),
            "celltype_taxonomy": "WMB 34-class (alz.config.WMB_CLASSES)",
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
        "pr_matrix.csv": (
            "Bulk proteomics: genes x animals; first column 'gene', other "
            "columns are animal_ids matching animal_metadata.csv rows. "
            "Duplicate gene rows collapsed by mean across protein isoforms."
        ),
        "ps_matrix.csv": (
            "Bulk pSer/pThr phosphoproteomics: genes x animals; sites "
            "collapsed to gene level by sum across (gene, animal). "
            "Same column convention as pr_matrix.csv."
        ),
        "py_matrix.csv": (
            "Bulk pTyr phosphoproteomics: genes x animals; sites collapsed "
            "to gene level by sum across (gene, animal). Same column "
            "convention as pr_matrix.csv."
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
        "**Bulk omics inputs (multi-omic factorial bridge):** "
        "`pr_matrix.csv` (proteomics), `ps_matrix.csv` (pSer/pThr; sites "
        "summed to gene), `py_matrix.csv` (pTyr; sites summed to gene), "
        "`kldata.csv` (kinase library, incytr schema).\n\n"
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
        f"{', '.join(SUBSET_LABELS)} for fast upstream iteration.\n\n"
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
        help="Skip the subset_microglia_astrocyte bundle.",
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

    song_to_wmb = main_config.load_song_to_wmb_class_map()
    mapped = adata.obs["subclass_name"].map(song_to_wmb)
    pre_map = sex_geno_time & prob_mask
    unmapped = int((pre_map & mapped.isna()).sum())
    if unmapped > 0:
        unmapped_names = sorted(
            adata.obs.loc[pre_map & mapped.isna(), "subclass_name"].unique().tolist()
        )
        print(
            f"  Dropping {unmapped} cells with unmapped Allen subclass_name "
            f"(no entry in SONG_TO_WMB_CLASS_MAP): {unmapped_names}"
        )

    keep = pre_map & mapped.notna()
    adata = adata[keep].copy()
    adata.obs["wmb_class"] = mapped[keep].values
    print(f"  After WMB-class mapping: {adata.n_obs} cells")

    meta = pd.DataFrame(index=adata.obs.index)
    meta["labels"] = adata.obs["wmb_class"].values
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

    print("\n  Loading bulk omics (pr, ps, py)...")
    omics_raw = omics_loaders.load_omics_matrices()
    for layer, df in omics_raw.items():
        print(f"    {layer}: {df.shape[0]} genes x {df.shape[1]} animals (pre-intersect)")

    transcript_map = omics_loaders.transcript_animal_canon_map(animal_meta)
    rekeyed_omics, kept_ids, dropped_ids = omics_loaders.intersect_and_rekey(
        omics_raw, transcript_map
    )
    print(
        f"\n  4-layer intersect: kept {len(kept_ids)} animals, "
        f"dropped {len(dropped_ids)} transcript animals ({dropped_ids})"
    )

    intersect_mask = animal_meta["animal_id"].isin(kept_ids)
    animal_meta_intersect = animal_meta.loc[intersect_mask].reset_index(drop=True)
    dropped_animals = animal_meta.loc[~intersect_mask, "animal_id"].tolist()

    assert_factorial_estimability(animal_meta_intersect)
    assert_omics_alignment(rekeyed_omics, kept_ids)
    print(
        f"  Asserts: rank-full {len(icfg.DESIGN_COLUMNS)}, "
        f"all {len(icfg.FACTORIAL_CONTRASTS)} contrasts estimable, "
        f"all 3 omics layers aligned to {len(kept_ids)} animals."
    )

    cell_intersect_mask = meta["animal_id"].isin(kept_ids)
    adata_intersect = adata[cell_intersect_mask.values].copy()
    meta_intersect = meta.loc[cell_intersect_mask].copy()
    print(
        f"  Transcript intersect: {adata_intersect.n_obs} cells "
        f"(dropped {adata.n_obs - adata_intersect.n_obs} cells from {dropped_animals})"
    )

    print(f"\n  Full bundle -> {args.out_dir}")
    write_bundle(args.out_dir, adata_intersect, meta_intersect, animal_meta_intersect)
    write_omics_bundle(args.out_dir, rekeyed_omics)
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
        "n_animals_full_4layer": int(len(kept_ids)),
        "dropped_animals": dropped_animals,
        "dropped_reason": "missing from proteomics/phospho coverage (Song dataset)",
        "site_to_gene_collapse": "sum",
        "id_canonicalization": (
            "strip leading zeros after letter prefix (transcript D092 ↔ "
            "proteomics D92); canonical form is the unpadded letter+number."
        ),
        "aptt_4mo_caveat": (
            "After dropping E137 the ApTt x 4mo cell becomes n=0. "
            "The ApTt_4mo contrast remains mathematically estimable via "
            "the additive interaction model but is model-extrapolated, "
            "with inflated SE relative to ApTt_2mo / ApTt_6mo. See "
            "docs/incytr_proposals/factorial_multiomic_coverage_report.md "
            "(incytr repo) for details."
        ),
        "kldata_source": KLDATA_SOURCE_REL,
        "future_workstream": (
            "Integrating proteomics/phospho coverage from animals without "
            "matched transcriptomics is a future model variant (per-layer "
            "designs, NA-tolerant OLS, or hierarchical model with "
            "transcript-imputed nodes). Out of scope here."
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
