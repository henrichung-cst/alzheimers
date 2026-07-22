"""Payload helpers shared by the unified and t-cell viewer builders (dedup of drifted copies, phase 5D)."""
from __future__ import annotations

import gzip
import json
import os
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_INCYTR_FC_NODES = ("Ligand", "Receptor", "EM", "Target")

_INCYTR_SIDECHAIN_INTERACTOME_COLUMNS = (
    "source_gene", "target_gene", "provenance", "in_vivo_refs", "in_vitro_refs",
    "n_motif_contrasts", "motif_contrasts",
)
_INCYTR_SIDECHAIN_TERMINAL_COLUMNS = (
    "kinase", "source_gene", "target_gene", "role", "contrast", "owning_cluster", "celltype_match",
    "provenance", "best_abs_pds",
    "best_abs_nes", "signed_nes", "best_fdr", "n_sites",
    "sites", "n_significant_concordant", "edge_delta",
)
_INCYTR_SIDECHAIN_INDEX_FILENAME = "sidechains_index.json"

_KINASE_LIBRARY_MOTIF_ALIASES = {
    # Kinase Library may expose activin receptor-like kinase aliases by either
    # receptor shorthand or HGNC symbol, depending on the package data table.
    "ALK1": "ACVRL1",
    "ALK2": "ACVR1",
    "ALK4": "ACVR1B",
    "ALK7": "ACVR1C",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sanitize(obj: Any, decimals: int = 4):
    """JSON-safe: NaN/Inf -> None, numpy -> native, floats rounded."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _sanitize(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, decimals) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist(), decimals)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return None
        return round(x, decimals)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NA:
        return None
    return obj


def _configure_duckdb_tempdir(con) -> None:
    d = os.environ.get("DUCKDB_TEMP_DIR", os.path.expanduser("~/.cache/duckdb"))
    os.makedirs(d, exist_ok=True)
    con.execute(f"SET temp_directory='{d}';")
    # Cap spill so a large sort (e.g. the per-sender ORDER BY over the 181M-row
    # nboot=0 pair-mode set) fails fast instead of filling the shared disk.
    con.execute("SET max_temp_directory_size='40GiB';")


def _json_clean_value(v, digits: int | None = None):
    """JSON-safe scalar conversion for compact viewer payload rows."""
    if v is None:
        return None
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        if not np.isfinite(x):
            return None
        return round(x, digits) if digits is not None else x
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return str(v)


def _build_incytr_gene_node_index(con) -> dict:
    """Compact exact gene-symbol index over Ligand/Receptor/EM/Target."""
    roles = list(_INCYTR_FC_NODES)
    df = con.execute("""
        WITH long AS (
          SELECT Ligand AS gene, 'Ligand' AS role, sender, receiver,
                 pvalue, PDS
          FROM src WHERE Ligand IS NOT NULL AND Ligand <> ''
          UNION ALL
          SELECT Receptor AS gene, 'Receptor' AS role, sender, receiver,
                 pvalue, PDS
          FROM src WHERE Receptor IS NOT NULL AND Receptor <> ''
          UNION ALL
          SELECT EM AS gene, 'EM' AS role, sender, receiver, pvalue, PDS
          FROM src WHERE EM IS NOT NULL AND EM <> ''
          UNION ALL
          SELECT Target AS gene, 'Target' AS role, sender, receiver,
                 pvalue, PDS
          FROM src WHERE Target IS NOT NULL AND Target <> ''
        )
        SELECT gene, role, sender, receiver,
               COUNT(*)::INTEGER AS n_rows,
               MAX(ABS(PDS)) AS best_abs_pds,
               arg_max(PDS, ABS(PDS)) AS best_pds,
               MIN(pvalue) AS best_pvalue
        FROM long
        GROUP BY gene, role, sender, receiver
        ORDER BY gene, role, sender, receiver
    """).fetchdf()
    if df.empty:
        return {
            "schema_version": 1,
            "index_type": "gene_node_pair_summary",
            "match_columns": roles,
            "genes": [],
            "roles": roles,
            "senders": [],
            "receivers": [],
            "gene_id": [],
            "role_id": [],
            "sender_id": [],
            "receiver_id": [],
            "n_rows": [],
            "best_abs_pds": [],
            "best_pds": [],
            "best_pvalue": [],
        }

    genes = sorted(df["gene"].dropna().astype(str).unique().tolist())
    senders = sorted(df["sender"].dropna().astype(str).unique().tolist())
    receivers = sorted(df["receiver"].dropna().astype(str).unique().tolist())
    gene_to_id = {v: i for i, v in enumerate(genes)}
    role_to_id = {v: i for i, v in enumerate(roles)}
    sender_to_id = {v: i for i, v in enumerate(senders)}
    receiver_to_id = {v: i for i, v in enumerate(receivers)}

    return {
        "schema_version": 1,
        "index_type": "gene_node_pair_summary",
        "match_columns": roles,
        "genes": genes,
        "roles": roles,
        "senders": senders,
        "receivers": receivers,
        "gene_id": [gene_to_id[str(v)] for v in df["gene"]],
        "role_id": [role_to_id[str(v)] for v in df["role"]],
        "sender_id": [sender_to_id[str(v)] for v in df["sender"]],
        "receiver_id": [receiver_to_id[str(v)] for v in df["receiver"]],
        "n_rows": [int(v) for v in df["n_rows"]],
        "best_abs_pds": [_json_clean_value(v, 4) for v in df["best_abs_pds"]],
        "best_pds": [_json_clean_value(v, 4) for v in df["best_pds"]],
        "best_pvalue": [_json_clean_value(v, 6) for v in df["best_pvalue"]],
    }


def _write_gene_node_index_shard(gene_node_index: dict, out_dir: str,
                                 filename: str,
                                 url_prefix: str = "edge_slices/incytr_pathways/") -> str:
    """Write the compact gene→node-pair index as a gzipped sidecar; return its
    viewer-relative URL.

    Audit P5: this ~15 MB-class structure is only consumed by pair-mode gene
    search in the Incytr Pathways tab, so it is fetched on demand rather than
    inlined in the payload and parsed at startup. Mirrors the `global_index`
    sidecar convention (same `edge_slices/incytr_pathways/` dir, atomic swap).
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    tmp_path = f"{out_path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    raw = json.dumps(gene_node_index, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    with gzip.open(tmp_path, "wb", compresslevel=6) as f:
        f.write(raw)
    os.replace(tmp_path, out_path)
    return f"{url_prefix}{filename}"


def _sidechain_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> dict:
    """Encode a small, fixed-schema edge table column-wise for a lazy shard."""
    return {
        column: [_json_clean_value(value) for value in df[column].tolist()]
        for column in columns
    }


def _write_incytr_sidechain_slice(
    source_dir: str,
    out_dir: str,
    filename: str,
    context_id: str,
    url_prefix: str = "edge_slices/incytr_pathways/",
    contrast_transform: Callable[[str], str] | None = None,
) -> dict:
    """Write one cohort-native kinase-sidechain lazy shard.

    ``kinase_kinase_edges.py`` has already reduced the motif and PSP inputs to
    kinome-bounded ``interactome.csv`` and ``terminal_edges.csv`` files.  This
    helper copies their exact row sets into a compact columnar JSON sidecar; it
    never opens the multi-gigabyte Incytr wide shards or derives pathway rows.

    ``contrast_transform`` rewrites each terminal-edge ``contrast`` value so it
    matches the consuming pathways-table row vocabulary.  The sidechain tab
    joins terminal edges to a selected row on exact ``contrast`` equality; the
    t-cell cohort's backend contrast (``D1_d13_vs_d2``) differs from its
    pathways-row contrast (``d13_d2``), so it passes a normalizer.  song/5xFAD
    share one vocabulary and pass ``None``.
    """
    interactome_path = os.path.join(source_dir, "interactome.csv")
    terminal_path = os.path.join(source_dir, "terminal_edges.csv")
    missing = [path for path in (interactome_path, terminal_path)
               if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "missing kinase-sidechain backend artifact: " + ", ".join(missing)
        )

    interactome = pd.read_csv(interactome_path, keep_default_na=False)
    terminal = pd.read_csv(terminal_path, keep_default_na=False)
    if tuple(interactome.columns) != _INCYTR_SIDECHAIN_INTERACTOME_COLUMNS:
        raise ValueError(
            f"{interactome_path}: unexpected interactome schema "
            f"{list(interactome.columns)!r}"
        )
    if tuple(terminal.columns) != _INCYTR_SIDECHAIN_TERMINAL_COLUMNS:
        raise ValueError(
            f"{terminal_path}: unexpected terminal-edge schema "
            f"{list(terminal.columns)!r}"
        )
    if terminal["provenance"].eq("psp").any():
        raise ValueError(
            f"{terminal_path}: terminal edges must be motif-anchored, not psp-only"
        )
    if contrast_transform is not None:
        terminal["contrast"] = terminal["contrast"].map(contrast_transform)

    nodes = set(interactome["source_gene"].dropna())
    nodes.update(interactome["target_gene"].dropna())
    payload = {
        "schema_version": 1,
        "slice_type": "incytr_kinase_sidechains",
        "context_id": context_id,
        "interactome_edge_count": int(len(interactome)),
        "interactome_node_count": int(len(nodes)),
        "terminal_edge_count": int(len(terminal)),
        "interactome": _sidechain_columns(
            interactome, _INCYTR_SIDECHAIN_INTERACTOME_COLUMNS
        ),
        "terminal_edges": _sidechain_columns(
            terminal, _INCYTR_SIDECHAIN_TERMINAL_COLUMNS
        ),
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    tmp_path = f"{out_path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(tmp_path, "wb", compresslevel=6) as f:
        f.write(raw)
    os.replace(tmp_path, out_path)
    entry = {
        "url": f"{url_prefix}{filename}",
        "interactome_edge_count": payload["interactome_edge_count"],
        "interactome_node_count": payload["interactome_node_count"],
        "terminal_edge_count": payload["terminal_edge_count"],
    }
    index_path = os.path.join(out_dir, _INCYTR_SIDECHAIN_INDEX_FILENAME)
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {
            "schema_version": 1,
            "slice_type": "incytr_kinase_sidechains",
            "by_context": {},
        }
    if (
        index.get("schema_version") != 1
        or index.get("slice_type") != "incytr_kinase_sidechains"
        or not isinstance(index.get("by_context"), dict)
    ):
        raise ValueError(f"{index_path}: unexpected sidechain-index schema")
    index["by_context"][context_id] = entry
    tmp_index_path = f"{index_path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    with open(tmp_index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_index_path, index_path)
    return entry


def _reset_incytr_sidechain_slices(out_dir: str) -> None:
    """Clear one viewer's sidechain index and shards before a fresh build."""
    if not os.path.isdir(out_dir):
        return
    for filename in os.listdir(out_dir):
        if (
            filename == _INCYTR_SIDECHAIN_INDEX_FILENAME
            or (filename.startswith("sidechains__") and filename.endswith(".json.gz"))
        ):
            os.remove(os.path.join(out_dir, filename))


def _build_kinase_motifs(kinase_names: list[str]) -> dict[str, dict]:
    """Build the global motif lookup consumed by the shared sequence-logo widget.

    Output is keyed by kinase name. Each entry:
        {
          "kin_type": "ser_thr" | "tyrosine",
          "positions": [-5, -4, ..., 4]  (or +5 for tyrosine),
          "amino_acids": [..., 23 entries ...],
          "matrix":  [[...], ...]  (n_aa x n_positions, normalized probs),
          "st_fav":  {"S": float, "T": float} | null,
        }
    Alias-aware: ALK1/2/4/7 shorthand names are resolved via
    _KINASE_LIBRARY_MOTIF_ALIASES before lookup.
    Sequence-logo widget on the viewer side scales letter heights by
    information content (log2(20) - entropy) at each position.
    """
    try:
        import kinase_library as kl
    except ImportError as e:
        print(f"  (warn) kinase_motifs unavailable: {e}", flush=True)
        return {}

    out: dict[str, dict] = {}
    skipped: list[str] = []
    for name in sorted({str(k) for k in kinase_names if str(k)}):
        source_name = name
        try:
            mat = kl.get_matrix(source_name, mat_type="norm")
            kin_type = kl.get_kinase_type(source_name)
        except Exception:
            alias = _KINASE_LIBRARY_MOTIF_ALIASES.get(name)
            if not alias:
                skipped.append(name)
                continue
            source_name = alias
            try:
                mat = kl.get_matrix(source_name, mat_type="norm")
                kin_type = kl.get_kinase_type(source_name)
            except Exception as e:
                skipped.append(f"{name}->{source_name} ({e})")
                continue

        st_fav: dict[str, float] | None = None
        if kin_type == "ser_thr":
            try:
                sf = kl.get_st_fav(source_name)
                st_fav = {
                    "S": float(sf.loc[source_name, "S"]),
                    "T": float(sf.loc[source_name, "T"]),
                }
            except Exception:
                st_fav = None

        out[name] = {
            "kin_type": kin_type,
            "positions": [int(c) for c in mat.columns],
            "amino_acids": [str(a) for a in mat.index],
            "matrix": [[round(float(v), 4) for v in row] for row in mat.values],
            "st_fav": st_fav,
        }

    if skipped:
        print(f"  kinase_motifs: skipped {len(skipped)} kinases "
              f"(first 3: {skipped[:3]})", flush=True)
    print(f"  kinase_motifs: emitted PSSM for {len(out):,} kinases", flush=True)
    return out
