"""Export factorial proteome/PTM evidence for Incytr pathway support.

The factorial R wrapper owns pathway enumeration and transcriptomic TPDS. This
adapter prepares the non-RNA evidence streams in one stable, contrast-long CSV
keyed by (contrast, cell_type, gene_symbol), so the R layer can attach
native-style node support without parsing upstream proteomics formats.

Baseline mode is independent of kinase enrichment and of a fixed SEA-AD/WMB
cell-type mapping. It reads the cell-type vocabulary exported for factorial
Incytr, uses exact matching when the omics table has columns in that same
vocabulary, and otherwise falls back to a condition-level global mean across
available omics columns. Set ENABLE_CELLTYPE_MAPPING=1 to opt into the legacy
SEA-AD-to-WMB compatibility mapping.
"""

import os
from functools import reduce

import numpy as np
import pandas as pd

from common import ensure_intermediates_dir
import config_integration as icfg


GENOTYPE_TO_SOURCE = {
    "App": "AppP",
    "Tau": "Ttau",
    "ApTt": "ApTt",
}


def _load_factorial_cell_types():
    meta_path = os.path.join(icfg.FACTORIAL_DIR, "expression_metadata.csv")
    meta = pd.read_csv(meta_path)
    if "labels" not in meta.columns:
        raise ValueError(f"{meta_path} is missing required 'labels' column")
    return sorted(meta["labels"].dropna().unique())


def _load_optional_cell_type_map(cell_types):
    if os.environ.get("ENABLE_CELLTYPE_MAPPING", "0") != "1":
        return {ct: ct for ct in cell_types}, "exact_or_global"
    if not os.path.exists(icfg.SEAAD_TO_WMB_CLASS_CSV):
        print(f"  Mapping file missing: {icfg.SEAAD_TO_WMB_CLASS_CSV}")
        return {ct: ct for ct in cell_types}, "exact_or_global"

    mapping = pd.read_csv(icfg.SEAAD_TO_WMB_CLASS_CSV)
    mapping = mapping[["seaad_subclass", "wmb_class_label"]].dropna()
    mapping = mapping.drop_duplicates("seaad_subclass")
    mapping = mapping[mapping["seaad_subclass"].isin(set(cell_types))].copy()
    out = {ct: ct for ct in cell_types}
    out.update(dict(zip(mapping["seaad_subclass"], mapping["wmb_class_label"])))
    return out, "seaad_subclass_to_wmb_class"


def _contrast_parts(contrast):
    geno, timepoint = contrast.rsplit("_", 1)
    return geno, GENOTYPE_TO_SOURCE[geno], timepoint


def _safe_log2_ratio(disease, control, pseudocount=1e-6):
    disease = pd.to_numeric(disease, errors="coerce")
    control = pd.to_numeric(control, errors="coerce")
    disease = disease.where(disease > 0)
    control = control.where(control > 0)
    return np.log2((disease + pseudocount) / (control + pseudocount))


def _condition_col(sex, timepoint, genotype, cell_type):
    return f"{sex}_{timepoint}_{genotype}_{cell_type}"


def _global_condition_values(df, sex, timepoint, genotype):
    prefix = f"{sex}_{timepoint}_{genotype}_"
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return None
    return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def _condition_values(df, sex, timepoint, genotype, evidence_cell_type):
    exact_col = _condition_col(sex, timepoint, genotype, evidence_cell_type)
    if exact_col in df.columns:
        return pd.to_numeric(df[exact_col], errors="coerce"), "cell_type_exact"
    vals = _global_condition_values(df, sex, timepoint, genotype)
    if vals is None:
        return None, "missing"
    return vals, "condition_global_mean"


def _proteome_evidence(cell_type_map, mapping_source):
    if not os.path.exists(icfg.PR_WMB_DECOMPOSITION_CSV):
        print(f"  Proteome decomposition missing: {icfg.PR_WMB_DECOMPOSITION_CSV}")
        return pd.DataFrame()

    pr = pd.read_csv(icfg.PR_WMB_DECOMPOSITION_CSV)
    rows = []
    for cell_type, evidence_cell_type in cell_type_map.items():
        for contrast in icfg.FACTORIAL_CONTRASTS:
            _, source_geno, timepoint = _contrast_parts(contrast)
            wt_vals, wt_source = _condition_values(
                pr, icfg.FACTORIAL_SEX, timepoint, "WTyp", evidence_cell_type
            )
            dis_vals, dis_source = _condition_values(
                pr, icfg.FACTORIAL_SEX, timepoint, source_geno, evidence_cell_type
            )
            if wt_vals is None or dis_vals is None:
                continue
            lfc = _safe_log2_ratio(dis_vals, wt_vals)
            tmp = pd.DataFrame({
                "contrast": contrast,
                "cell_type": cell_type,
                "pr_evidence_cell_type": evidence_cell_type,
                "gene_symbol": pr["gene_symbol"],
                "pr_log2FC": lfc,
                "pr_aFC": lfc,
                "pr_source": "wmb_decomposition",
                "pr_assignment": (
                    wt_source if wt_source == dis_source else "mixed"
                ),
                "pr_cell_type_mapping_source": mapping_source,
            })
            tmp = tmp.dropna(subset=["gene_symbol", "pr_log2FC"])
            if tmp.empty:
                continue
            tmp["_abs_lfc"] = tmp["pr_log2FC"].abs()
            idx = tmp.groupby(
                ["contrast", "cell_type", "gene_symbol"], sort=False
            )["_abs_lfc"].idxmax()
            rows.append(tmp.loc[idx].drop(columns=["_abs_lfc"]))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out


def _load_site_fdr(path, track):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["site_id", "contrast", f"{track}_site_fdr"])
    site = pd.read_csv(path, dtype={"site_id": str})
    rows = []
    for contrast in icfg.FACTORIAL_CONTRASTS:
        fdr_col = f"stoich_fdr_{contrast}"
        if fdr_col not in site.columns:
            continue
        rows.append(pd.DataFrame({
            "site_id": site["site_id"].astype(str),
            "contrast": contrast,
            f"{track}_site_fdr": pd.to_numeric(site[fdr_col], errors="coerce"),
        }))
    if not rows:
        return pd.DataFrame(columns=["site_id", "contrast", f"{track}_site_fdr"])
    return pd.concat(rows, ignore_index=True)


def _ptm_evidence(path, track, cell_type_map, mapping_source, site_fdr_path):
    if not os.path.exists(path):
        print(f"  {track} decomposition missing: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, dtype={"site_id": str})
    if "site_id" not in df.columns:
        df["site_id"] = (
            df["protein_id"].astype(str) + "_" + df["site_position"].astype(str)
        )

    site_fdr = _load_site_fdr(site_fdr_path, track)
    rows = []
    for cell_type, evidence_cell_type in cell_type_map.items():
        for contrast in icfg.FACTORIAL_CONTRASTS:
            _, source_geno, timepoint = _contrast_parts(contrast)
            wt_vals, wt_source = _condition_values(
                df, icfg.FACTORIAL_SEX, timepoint, "WTyp", evidence_cell_type
            )
            dis_vals, dis_source = _condition_values(
                df, icfg.FACTORIAL_SEX, timepoint, source_geno, evidence_cell_type
            )
            if wt_vals is None or dis_vals is None:
                continue
            lfc = _safe_log2_ratio(dis_vals, wt_vals)
            tmp = pd.DataFrame({
                "contrast": contrast,
                "cell_type": cell_type,
                f"{track}_evidence_cell_type": evidence_cell_type,
                "gene_symbol": df["gene_symbol"],
                f"{track}_log2FC": lfc,
                f"{track}_aFC": lfc,
                f"{track}_site_id": df["site_id"].astype(str),
                f"{track}_site_position": df.get("site_position", pd.NA),
                f"{track}_source": "wmb_decomposition",
                f"{track}_assignment": (
                    wt_source if wt_source == dis_source else "mixed"
                ),
                f"{track}_cell_type_mapping_source": mapping_source,
            })
            tmp = tmp.dropna(subset=["gene_symbol", f"{track}_log2FC"])
            if tmp.empty:
                continue

            # Collapse multiple sites per gene by the largest absolute effect.
            tmp["_abs_lfc"] = tmp[f"{track}_log2FC"].abs()
            idx = tmp.groupby(
                ["contrast", "cell_type", "gene_symbol"], sort=False
            )["_abs_lfc"].idxmax()
            rows.append(tmp.loc[idx].drop(columns=["_abs_lfc"]))

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    if not site_fdr.empty:
        out = out.merge(
            site_fdr,
            left_on=[f"{track}_site_id", "contrast"],
            right_on=["site_id", "contrast"],
            how="left",
        ).drop(columns=["site_id"])
    else:
        out[f"{track}_site_fdr"] = np.nan
    return out


def _merge_frames(frames):
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    keys = ["contrast", "cell_type", "gene_symbol"]
    return reduce(lambda left, right: left.merge(right, on=keys, how="outer"), frames)


def main():
    ensure_intermediates_dir()
    os.makedirs(icfg.FACTORIAL_DIR, exist_ok=True)

    cell_types = _load_factorial_cell_types()
    cell_type_map, mapping_source = _load_optional_cell_type_map(cell_types)
    n_mapped = sum(1 for ct, ev in cell_type_map.items() if ct != ev)
    print(f"Factorial cell types: {len(cell_types)}")
    print(f"Cell-type mapping mode: {mapping_source} ({n_mapped} mapped)")

    pr = _proteome_evidence(cell_type_map, mapping_source)
    print(f"  proteome rows: {len(pr):,}")

    ps = _ptm_evidence(
        icfg.PS_WMB_DECOMPOSITION_CSV,
        "ps",
        cell_type_map,
        mapping_source,
        icfg.SITE_LEVEL_OLS_CSV,
    )
    print(f"  S/T PTM rows: {len(ps):,}")

    py = _ptm_evidence(
        icfg.PY_WMB_DECOMPOSITION_CSV,
        "py",
        cell_type_map,
        mapping_source,
        icfg.SITE_LEVEL_OLS_PY_CSV,
    )
    print(f"  pY PTM rows: {len(py):,}")

    out = _merge_frames([pr, ps, py])
    if out.empty:
        print("WARNING: no multiomics evidence rows produced")
    else:
        out = out.sort_values(["contrast", "cell_type", "gene_symbol"])

    out_path = os.path.join(icfg.FACTORIAL_DIR, "multiomics_evidence.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
