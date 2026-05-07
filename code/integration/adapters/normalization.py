"""Normalized Incytr storage helpers.

This module centralizes the schema-version-2 layout described in
``pipeline_notes/incytr_storage_normalization_plan.md``.  Producers can use it
additively while legacy CSV/parquet outputs are still present.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

import config_integration as icfg


CONCORDANCE_ENUM = {
    "none": 0,
    "concordant": 1,
    "discordant": 2,
    "mixed": 3,
}

EVIDENCE_ENUM = {
    "expression-confirmed": 0,
    "kinase-imputed": 1,
    "mixed": 2,
}

IMPUTED_NODE_BITS = {
    "Receptor": 1,
    "EM": 2,
    "Target": 4,
}


@dataclass(frozen=True)
class NormalizedPaths:
    universe_id: str
    scoring_id: str
    config_id: str

    @property
    def universe_dir(self) -> str:
        return os.path.join(icfg.UNIVERSE_BASE, self.universe_id)

    @property
    def scoring_dir(self) -> str:
        return os.path.join(icfg.SCORING_BASE, self.scoring_id)

    @property
    def config_dir(self) -> str:
        return os.path.join(icfg.CONFIG_BASE, self.config_id)


def resolve_paths(*, universe_id=None, scoring_id=None, config_id=None) -> NormalizedPaths:
    universe_id = universe_id or icfg.resolve_universe_id()
    scoring_id = scoring_id or icfg.resolve_scoring_id({"universe_id": universe_id})
    config_id = config_id or icfg.resolve_config_id({
        "universe_id": universe_id,
        "scoring_id": scoring_id,
    })
    return NormalizedPaths(universe_id, scoring_id, config_id)


def write_manifest(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = {
        "schema_version": icfg.NORMALIZED_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def parse_path_series(paths: pd.Series) -> pd.DataFrame:
    parts = paths.astype("string").str.split("*", expand=True)
    if parts.shape[1] != 4:
        raise ValueError("Expected pathway strings in Ligand*Receptor*EM*Target format")
    parts.columns = ["Ligand", "Receptor", "EM", "Target"]
    return parts


def _id_table(values: Iterable[str], id_col: str, value_col: str) -> pd.DataFrame:
    vals = sorted({str(v) for v in values if pd.notna(v) and str(v) != ""})
    dtype = "uint8" if len(vals) <= np.iinfo(np.uint8).max else "uint32"
    return pd.DataFrame({
        id_col: np.arange(len(vals), dtype=dtype),
        value_col: vals,
    })


def imputed_nodes_mask(value) -> int:
    if pd.isna(value) or not str(value):
        return 0
    mask = 0
    for item in str(value).replace(",", ";").split(";"):
        item = item.strip()
        mask |= IMPUTED_NODE_BITS.get(item, 0)
    return mask


def build_universe_tables_from_legacy(
    *,
    pair_dirs: list[tuple[str, str]],
    backbones: pd.DataFrame,
    kinases: Iterable[str],
    celltypes: Iterable[str],
    contrasts: Iterable[str],
    split_pair,
    output_dir: str,
    provenance: pd.DataFrame | None = None,
) -> dict:
    """Build normalized universe dimension tables from legacy pair outputs."""
    os.makedirs(output_dir, exist_ok=True)

    cells = _id_table(celltypes, "cell_id", "name")
    contrasts_df = _id_table(contrasts, "contrast_id", "name")
    kinases_df = _id_table(kinases, "kinase_id", "symbol")
    cell_to_id = dict(zip(cells["name"], cells["cell_id"]))

    pair_rows = []
    path_parts = []
    for pair_id, (pair_name, pair_dir) in enumerate(pair_dirs):
        sender, receiver = split_pair(pair_name)
        pair_rows.append({
            "pair_id": pair_id,
            "sender_id": int(cell_to_id[sender]),
            "receiver_id": int(cell_to_id[receiver]),
            "name": pair_name,
        })

        score_path = os.path.join(pair_dir, "kinase_support_scores.csv")
        used_paths = pd.read_csv(score_path, usecols=["Path"])["Path"].drop_duplicates()
        parts = parse_path_series(used_paths)
        parts["Path"] = used_paths.astype("string").to_numpy()
        parts["sender_id"] = int(cell_to_id[sender])
        parts["receiver"] = receiver
        path_parts.append(parts)

    pair_dim = pd.DataFrame(pair_rows).astype({
        "pair_id": "uint16",
        "sender_id": "uint8",
        "receiver_id": "uint8",
    })

    backbone_out = backbones.copy()
    if "backbone_id" not in backbone_out.columns:
        backbone_out = backbone_out.sort_values(["receiver", "Receptor", "EM", "Target"]).reset_index(drop=True)
        backbone_out["backbone_id"] = np.arange(len(backbone_out), dtype=np.uint32)

    if provenance is not None and not provenance.empty:
        prov_cols = [
            "receiver", "Receptor", "EM", "Target", "pathway_evidence_backbone",
            "n_expression_confirmed", "n_kinase_imputed", "imputed_nodes_union",
        ]
        backbone_out = backbone_out.merge(
            provenance[prov_cols].drop_duplicates(),
            on=["receiver", "Receptor", "EM", "Target"],
            how="left",
        )
    else:
        backbone_out["pathway_evidence_backbone"] = "expression-confirmed"
        backbone_out["n_expression_confirmed"] = 0
        backbone_out["n_kinase_imputed"] = 0
        backbone_out["imputed_nodes_union"] = ""

    gene_values = set()
    for col in ["Receptor", "EM", "Target"]:
        gene_values.update(backbone_out[col].dropna().astype(str))
    if path_parts:
        all_path_parts = pd.concat(path_parts, ignore_index=True).drop_duplicates()
        gene_values.update(all_path_parts["Ligand"].dropna().astype(str))
    else:
        all_path_parts = pd.DataFrame(columns=["Path", "Ligand", "Receptor", "EM", "Target", "sender_id", "receiver"])
    genes = _id_table(gene_values, "gene_id", "symbol")
    gene_to_id = dict(zip(genes["symbol"], genes["gene_id"]))

    backbones_norm = pd.DataFrame({
        "backbone_id": backbone_out["backbone_id"].astype("uint32"),
        "receiver_id": backbone_out["receiver"].map(cell_to_id).astype("uint8"),
        "receptor_gene_id": backbone_out["Receptor"].map(gene_to_id).astype("uint32"),
        "em_gene_id": backbone_out["EM"].map(gene_to_id).astype("uint32"),
        "target_gene_id": backbone_out["Target"].map(gene_to_id).astype("uint32"),
        "pathway_evidence": backbone_out["pathway_evidence_backbone"].fillna("expression-confirmed").map(EVIDENCE_ENUM).astype("uint8"),
        "imputed_nodes_mask": backbone_out["imputed_nodes_union"].map(imputed_nodes_mask).astype("uint8"),
        "n_expression_confirmed": backbone_out["n_expression_confirmed"].fillna(0).astype("uint16"),
        "n_kinase_imputed": backbone_out["n_kinase_imputed"].fillna(0).astype("uint16"),
    })

    bb_key = backbone_out[["receiver", "Receptor", "EM", "Target", "backbone_id"]]
    pathways_raw = all_path_parts.merge(
        bb_key, on=["receiver", "Receptor", "EM", "Target"], how="left",
        validate="many_to_one",
    )
    if pathways_raw["backbone_id"].isna().any():
        missing = pathways_raw[pathways_raw["backbone_id"].isna()].head(3)
        raise ValueError(f"Unresolved pathway backbones: {missing.to_dict('records')}")
    pathways_raw = pathways_raw.drop_duplicates(["Path", "sender_id"]).sort_values(["Path", "sender_id"])
    pathways_raw["path_id"] = np.arange(len(pathways_raw), dtype=np.uint32)
    pathways = pd.DataFrame({
        "path_id": pathways_raw["path_id"].astype("uint32"),
        "ligand_gene_id": pathways_raw["Ligand"].map(gene_to_id).astype("uint32"),
        "backbone_id": pathways_raw["backbone_id"].astype("uint32"),
        "sender_id": pathways_raw["sender_id"].astype("uint8"),
        "path": pathways_raw["Path"].astype("string"),
    })

    cells.to_parquet(os.path.join(output_dir, "cells.parquet"), index=False)
    contrasts_df.to_parquet(os.path.join(output_dir, "contrasts.parquet"), index=False)
    kinases_df.to_parquet(os.path.join(output_dir, "kinases.parquet"), index=False)
    genes.to_parquet(os.path.join(output_dir, "genes.parquet"), index=False)
    pair_dim.to_parquet(os.path.join(output_dir, "pair_dim.parquet"), index=False)
    backbones_norm.to_parquet(os.path.join(output_dir, "backbones.parquet"), index=False)
    pathways.to_parquet(os.path.join(output_dir, "pathways.parquet"), index=False)

    write_manifest(os.path.join(output_dir, "manifest.json"), {
        "universe_id": os.path.basename(output_dir),
        "n_genes": int(len(genes)),
        "n_cells": int(len(cells)),
        "n_contrasts": int(len(contrasts_df)),
        "n_kinases": int(len(kinases_df)),
        "n_pairs": int(len(pair_dim)),
        "n_backbones": int(len(backbones_norm)),
        "n_pathways": int(len(pathways)),
    })
    return {
        "genes": genes,
        "cells": cells,
        "contrasts": contrasts_df,
        "kinases": kinases_df,
        "pair_dim": pair_dim,
        "backbones": backbones_norm,
        "pathways": pathways,
    }


def pathway_lookup(universe_dir: str, sender_id: int | None = None) -> pd.DataFrame:
    out = pd.read_parquet(
        os.path.join(universe_dir, "pathways.parquet"),
        columns=["path_id", "path", "sender_id"],
    )
    if sender_id is not None:
        out = out[out["sender_id"] == sender_id].copy()
    return out[["path_id", "path"]]


def write_routes_from_legacy(pair_dirs: list[tuple[str, str]], universe_dir: str) -> int:
    """Normalize legacy per-pair kinase_routes.parquet files into routes/{pair_id}.parquet."""
    routes_dir = os.path.join(universe_dir, "routes")
    os.makedirs(routes_dir, exist_ok=True)
    pathways = pd.read_parquet(
        os.path.join(universe_dir, "pathways.parquet"),
        columns=["path_id", "path", "sender_id"],
    )
    contrasts = pd.read_parquet(os.path.join(universe_dir, "contrasts.parquet"))
    contrast_to_id = dict(zip(contrasts["name"], contrasts["contrast_id"]))
    kinases = pd.read_parquet(os.path.join(universe_dir, "kinases.parquet"))
    kinase_to_id = dict(zip(kinases["symbol"], kinases["kinase_id"]))
    pair_dim = pd.read_parquet(os.path.join(universe_dir, "pair_dim.parquet"))
    name_to_pair_id = dict(zip(pair_dim["name"], pair_dim["pair_id"]))

    total_rows = 0
    for pair_name, pair_dir in pair_dirs:
        src = os.path.join(pair_dir, "kinase_routes.parquet")
        if not os.path.exists(src):
            continue
        routes = pd.read_parquet(src)
        sender_id = int(pair_dim.loc[pair_dim["name"] == pair_name, "sender_id"].iloc[0])
        path_subset = pathways[pathways["sender_id"] == sender_id]
        path_to_id = dict(zip(path_subset["path"], path_subset["path_id"]))
        path_id = routes["Path"].map(path_to_id)
        contrast_id = routes["contrast"].map(contrast_to_id)
        kinase_id = routes["kinase"].map(kinase_to_id)
        if path_id.isna().any() or contrast_id.isna().any() or kinase_id.isna().any():
            raise ValueError(f"Unresolved ids in {src}")
        out = pd.DataFrame({
            "pair_id": np.full(len(routes), int(name_to_pair_id[pair_name]), dtype=np.uint16),
            "path_id": path_id.astype("uint32"),
            "contrast_id": contrast_id.astype("uint8"),
            "kinase_id": kinase_id.astype("uint16"),
            "support_contribution": routes["mea_support_contribution"].astype("float32"),
            "nes_sign": routes["nes_sign"].astype("int8"),
        })
        out.to_parquet(os.path.join(routes_dir, f"{int(name_to_pair_id[pair_name])}.parquet"),
                       index=False)
        total_rows += len(out)
    return total_rows


def normalized_scores_for_pair(scores_df: pd.DataFrame, path_lookup: pd.DataFrame, contrast_id: int) -> pd.DataFrame:
    out = scores_df.merge(path_lookup, left_on="Path", right_on="path", how="left", validate="many_to_one")
    if out["path_id"].isna().any():
        missing = out.loc[out["path_id"].isna(), "Path"].head(5).tolist()
        raise ValueError(f"Missing path_id for pathway scores: {missing}")
    return pd.DataFrame({
        "path_id": out["path_id"].astype("uint32"),
        "contrast_id": np.full(len(out), contrast_id, dtype=np.uint8),
        "support_score": out["mea_kinase_support_score"].astype("float32"),
        "support_sum": out["mea_kinase_support_sum"].astype("float32"),
        "n_distinct_kinases": out["mea_n_distinct_kinases"].astype("uint16"),
        "concordance_flag": out["mea_concordance_flag"].map(CONCORDANCE_ENUM).fillna(0).astype("uint8"),
        "tpds": out["TPDS"].astype("float32"),
    })


def write_pathway_scores_for_pair(
    scores_df: pd.DataFrame,
    *,
    universe_dir: str,
    scoring_dir: str,
    pair_name: str,
    contrast_name: str,
) -> str:
    """Write one normalized pathway score partition for a legacy pair score frame."""
    pair_dim = pd.read_parquet(os.path.join(universe_dir, "pair_dim.parquet"))
    pair_ids = pair_dim.loc[pair_dim["name"] == pair_name, "pair_id"]
    if pair_ids.empty:
        raise ValueError(f"Pair {pair_name!r} is absent from normalized universe")
    sender_id = int(pair_dim.loc[pair_dim["name"] == pair_name, "sender_id"].iloc[0])
    lookup = pathway_lookup(universe_dir, sender_id=sender_id)
    contrasts = pd.read_parquet(os.path.join(universe_dir, "contrasts.parquet"))
    contrast_to_id = dict(zip(contrasts["name"], contrasts["contrast_id"]))
    if contrast_name not in contrast_to_id:
        raise ValueError(f"Contrast {contrast_name!r} is absent from normalized universe")

    out = normalized_scores_for_pair(
        scores_df,
        lookup,
        int(contrast_to_id[contrast_name]),
    )
    pair_id = int(pair_ids.iloc[0])
    out_dir = os.path.join(scoring_dir, "pathway_scores.parquet", f"pair_id={pair_id}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "part-0.parquet")
    out.to_parquet(out_path, index=False)
    write_manifest(os.path.join(scoring_dir, "manifest.json"), {
        "scoring_id": os.path.basename(scoring_dir),
        "universe_id": os.path.basename(universe_dir),
        "contrast_name": contrast_name,
    })
    return out_path


def write_factorial_pathway_scores_for_pair(
    scores_df: pd.DataFrame,
    *,
    universe_dir: str,
    scoring_dir: str,
    pair_name: str,
    contrasts: Iterable[str],
) -> str:
    """Write one normalized pathway score partition from a wide factorial score frame."""
    pair_dim = pd.read_parquet(os.path.join(universe_dir, "pair_dim.parquet"))
    pair_ids = pair_dim.loc[pair_dim["name"] == pair_name, "pair_id"]
    if pair_ids.empty:
        raise ValueError(f"Pair {pair_name!r} is absent from normalized universe")
    sender_id = int(pair_dim.loc[pair_dim["name"] == pair_name, "sender_id"].iloc[0])
    lookup = pathway_lookup(universe_dir, sender_id=sender_id)
    contrasts_df = pd.read_parquet(os.path.join(universe_dir, "contrasts.parquet"))
    contrast_to_id = dict(zip(contrasts_df["name"], contrasts_df["contrast_id"]))

    joined = scores_df.merge(lookup, left_on="Path", right_on="path", how="left", validate="many_to_one")
    if joined["path_id"].isna().any():
        missing = joined.loc[joined["path_id"].isna(), "Path"].head(5).tolist()
        raise ValueError(f"Missing path_id for factorial pathway scores: {missing}")

    parts = []
    for contrast in contrasts:
        if contrast not in contrast_to_id:
            raise ValueError(f"Contrast {contrast!r} is absent from normalized universe")
        parts.append(pd.DataFrame({
            "path_id": joined["path_id"].astype("uint32"),
            "contrast_id": np.full(len(joined), int(contrast_to_id[contrast]), dtype=np.uint8),
            "support_score": joined[f"mea_kinase_support_score_{contrast}"].astype("float32"),
            "support_sum": joined[f"mea_kinase_support_sum_{contrast}"].astype("float32"),
            "n_distinct_kinases": joined[f"mea_n_distinct_kinases_{contrast}"].astype("uint16"),
            "concordance_flag": joined[f"mea_concordance_flag_{contrast}"].map(CONCORDANCE_ENUM).fillna(0).astype("uint8"),
            "tpds": joined[f"TPDS_{contrast}"].astype("float32"),
        }))
    out = pd.concat(parts, ignore_index=True)
    pair_id = int(pair_ids.iloc[0])
    out_dir = os.path.join(scoring_dir, "pathway_scores.parquet", f"pair_id={pair_id}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "part-0.parquet")
    out.to_parquet(out_path, index=False)
    write_manifest(os.path.join(scoring_dir, "manifest.json"), {
        "scoring_id": os.path.basename(scoring_dir),
        "universe_id": os.path.basename(universe_dir),
        "contrasts": list(contrasts),
    })
    return out_path


def write_config_tables_from_legacy(
    *,
    aggregation_dir: str,
    universe_dir: str,
    config_dir: str,
    pvalue_threshold: float,
) -> dict:
    """Convert legacy aggregation CSVs into normalized config parquet tables."""
    os.makedirs(config_dir, exist_ok=True)
    cells = pd.read_parquet(os.path.join(universe_dir, "cells.parquet"))
    genes = pd.read_parquet(os.path.join(universe_dir, "genes.parquet"))
    contrasts = pd.read_parquet(os.path.join(universe_dir, "contrasts.parquet"))
    backbones = pd.read_parquet(os.path.join(universe_dir, "backbones.parquet"))
    cell_to_id = dict(zip(cells["name"], cells["cell_id"]))
    gene_to_symbol = dict(zip(genes["gene_id"], genes["symbol"]))
    gene_to_id = dict(zip(genes["symbol"], genes["gene_id"]))
    contrast_to_id = dict(zip(contrasts["name"], contrasts["contrast_id"]))

    bb_key = pd.DataFrame({
        "backbone_id": backbones["backbone_id"],
        "receiver": backbones["receiver_id"].map(dict(zip(cells["cell_id"], cells["name"]))),
        "Receptor": backbones["receptor_gene_id"].map(gene_to_symbol),
        "EM": backbones["em_gene_id"].map(gene_to_symbol),
        "Target": backbones["target_gene_id"].map(gene_to_symbol),
    })

    outputs = {}
    recurrence_path = os.path.join(aggregation_dir, "backbone_recurrence_by_contrast.csv")
    if os.path.exists(recurrence_path):
        recurrence = pd.read_csv(recurrence_path)
        recurrence = recurrence.merge(
            bb_key, on=["receiver", "Receptor", "EM", "Target"],
            how="left", validate="many_to_one",
        )
        if recurrence["backbone_id"].isna().any():
            raise ValueError("Unresolved backbone_id while normalizing backbone recurrence")
        recurrence["contrast_id"] = recurrence["contrast"].map(contrast_to_id).astype("uint8")

        sender_rows = []
        for row in recurrence.itertuples(index=False):
            significant = set()
            sig_value = getattr(row, "significant_sender_list", "")
            if not pd.isna(sig_value):
                significant = {s.strip() for s in str(sig_value).split(",") if s.strip()}
            for sender in str(getattr(row, "sender_list", "")).split(","):
                sender = sender.strip()
                if sender:
                    sender_rows.append({
                        "backbone_id": int(row.backbone_id),
                        "contrast_id": int(row.contrast_id),
                        "sender_id": int(cell_to_id[sender]),
                        "is_significant": sender in significant,
                    })

        keep = [
            "backbone_id", "contrast_id", "n_senders", "n_senders_significant",
            "mean_tpds", "max_abs_tpds", "tpds_pvalue",
        ]
        backbones_by_contrast = recurrence[keep].copy()

        perm_path = os.path.join(aggregation_dir, "backbone_permutation_pvalues_by_contrast.csv")
        if os.path.exists(perm_path):
            perm = pd.read_csv(perm_path)
            perm = perm.merge(
                bb_key, on=["receiver", "Receptor", "EM", "Target"],
                how="left", validate="many_to_one",
            )
            perm["contrast_id"] = perm["contrast"].map(contrast_to_id).astype("uint8")
            metric_cols = [
                c for c in perm.columns
                if c not in {"contrast", "receiver", "Receptor", "EM", "Target"}
                and c not in set(bb_key.columns)
            ]
            backbones_by_contrast = backbones_by_contrast.merge(
                perm[["backbone_id", "contrast_id"] + metric_cols],
                on=["backbone_id", "contrast_id"],
                how="left",
            )

        out_path = os.path.join(config_dir, "backbones_by_contrast.parquet")
        backbones_by_contrast.to_parquet(out_path, index=False)
        outputs["backbones_by_contrast"] = out_path

        senders = pd.DataFrame(sender_rows, columns=[
            "backbone_id", "contrast_id", "sender_id", "is_significant",
        ])
        senders_path = os.path.join(config_dir, "backbone_senders.parquet")
        senders.to_parquet(senders_path, index=False)
        outputs["backbone_senders"] = senders_path

    target_path = os.path.join(aggregation_dir, "target_convergence_by_contrast.csv")
    if os.path.exists(target_path):
        target = pd.read_csv(target_path)
        target["contrast_id"] = target["contrast"].map(contrast_to_id).astype("uint8")
        target["receiver_id"] = target["receiver"].map(cell_to_id).astype("uint8")
        target["target_gene_id"] = target["Target"].map(gene_to_id).astype("uint32")

        target_sender_rows = []
        for row in target.itertuples(index=False):
            senders = getattr(row, "significant_senders", "")
            if pd.isna(senders):
                continue
            for sender in str(senders).split(","):
                sender = sender.strip()
                if sender:
                    target_sender_rows.append({
                        "contrast_id": int(row.contrast_id),
                        "receiver_id": int(row.receiver_id),
                        "target_gene_id": int(row.target_gene_id),
                        "sender_id": int(cell_to_id[sender]),
                        "is_significant": True,
                    })

        target_out = target[[
            "contrast_id", "receiver_id", "target_gene_id",
            "n_senders", "n_senders_significant", "mean_tpds",
        ]]
        target_out_path = os.path.join(config_dir, "target_convergence.parquet")
        target_out.to_parquet(target_out_path, index=False)
        outputs["target_convergence"] = target_out_path

        target_senders = pd.DataFrame(target_sender_rows, columns=[
            "contrast_id", "receiver_id", "target_gene_id", "sender_id",
            "is_significant",
        ])
        target_senders_path = os.path.join(config_dir, "target_convergence_senders.parquet")
        target_senders.to_parquet(target_senders_path, index=False)
        outputs["target_convergence_senders"] = target_senders_path

    write_manifest(os.path.join(config_dir, "manifest.json"), {
        "config_id": os.path.basename(config_dir),
        "universe_id": os.path.basename(universe_dir),
        "pvalue_threshold": pvalue_threshold,
        "outputs": sorted(outputs),
    })
    return outputs


def open_edges_view(universe_id=None, scoring_id=None):
    """Open a DuckDB connection with normalized edge-view relations registered."""
    paths = resolve_paths(universe_id=universe_id, scoring_id=scoring_id)
    con = duckdb.connect()
    udir = paths.universe_dir
    sdir = paths.scoring_dir
    con.execute(f"CREATE VIEW pathways AS SELECT * FROM read_parquet('{os.path.join(udir, 'pathways.parquet')}')")
    con.execute(f"CREATE VIEW backbones AS SELECT * FROM read_parquet('{os.path.join(udir, 'backbones.parquet')}')")
    con.execute(f"CREATE VIEW pair_dim AS SELECT * FROM read_parquet('{os.path.join(udir, 'pair_dim.parquet')}')")
    con.execute(f"CREATE VIEW routes AS SELECT * FROM read_parquet('{os.path.join(udir, 'routes', '*.parquet')}')")
    scores_glob = os.path.join(sdir, "pathway_scores.parquet", "**", "*.parquet")
    con.execute(f"CREATE VIEW pathway_scores AS SELECT * FROM read_parquet('{scores_glob}')")
    con.execute("""
        CREATE VIEW kinase_backbone_edges AS
        SELECT r.kinase_id,
               p.backbone_id,
               r.contrast_id,
               pd.sender_id,
               b.receiver_id,
               r.support_contribution
        FROM routes r
        JOIN pathways p USING (path_id)
        JOIN pair_dim pd USING (pair_id)
        JOIN backbones b USING (backbone_id)
    """)
    return con
