"""Mukesh / NBB Human AD ingest — single-use converter to Song-shaped artifacts.

See ``docs/mukesh_human_ingest_plan.md`` for full spec. This module implements
Phase A (UniProt canonical-isoform cache) and Phase B (diagnostic pass); Phase
C (reshape) is stubbed.

Phase A:  python -m alz.cohorts.mukesh.ingest --uniprot-cache
Phase B:  python -m alz.cohorts.mukesh.ingest --diagnose
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import logging
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Iterable

import requests

from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.shared import config

LOG = logging.getLogger("ingest_mukesh")

MUKESH_DIR = os.path.join(config.REPO_ROOT, "data", "datasets", "mukesh")
PROTEIN_REPORT = os.path.join(
    MUKESH_DIR,
    "proteomics",
    "MK_Astral_DIA-45min-NBB-AD-CTRL-InsolDigest-Re-prePTMScan-STY-AcK-KGG-LFS_Protein-Report.csv",
)
IMAC_REPORT = os.path.join(
    MUKESH_DIR,
    "phospho",
    "IMAC",
    "MK-NBB-AD-Control-Tissue-IMAC-DIA-45min-LFS-allTauIsoform--v1.0_Peptide-Report-Norm.csv",
)
PY_REPORT = os.path.join(
    MUKESH_DIR,
    "phospho",
    "pY",
    "20240711_104735_051624_MK_Astral_DIA-45min-NBB-AD-CTRL-pYenrichment-HBS_Peptide-Report.csv",
)
INPUT_CSVS = [PROTEIN_REPORT, IMAC_REPORT, PY_REPORT]

CACHE_DIR = os.path.join(MUKESH_DIR, "analysis_cache", "uniprot")
CANONICAL_MAP = os.path.join(CACHE_DIR, "canonical_map.json")
UNRESOLVED_CSV = os.path.join(CACHE_DIR, "unresolved_genes.csv")
FETCH_LOG = os.path.join(CACHE_DIR, "fetch.log")

HUMAN_DATA_INGEST_DIR = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "data_ingest_human"
)
HUMAN_KINASE_DIR = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "kinase_attribution_human"
)
SYNTHESIS_AUDIT_CSV = os.path.join(HUMAN_DATA_INGEST_DIR, "synthesis_audit.csv")
SYNTHESIS_AUDIT_SUMMARY = os.path.join(
    HUMAN_DATA_INGEST_DIR, "synthesis_audit_summary.txt"
)

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
SEARCH_FIELDS = "accession,sequence,cc_alternative_products,gene_primary,protein_name"
MAX_WORKERS = 8
RETRY_DELAYS = (1.0, 4.0, 16.0)
SANITY_CONTROLS = {
    "MAPT": "P10636",
    "APP": "P05067",
    "SNCA": "P37840",
    "GAPDH": "P04406",
    "ACTB": "P60709",
}


def _collect_gene_symbols(paths: Iterable[str]) -> set[str]:
    """Stream `PG.Genes` from each CSV; tolerate semicolon-joined multi-gene entries."""
    genes: set[str] = set()
    for path in paths:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            if "PG.Genes" not in (reader.fieldnames or []):
                LOG.warning("PG.Genes not in %s; skipping", path)
                continue
            for row in reader:
                cell = (row.get("PG.Genes") or "").strip()
                if not cell:
                    continue
                for g in cell.replace(",", ";").split(";"):
                    g = g.strip()
                    if g:
                        genes.add(g)
    return genes


def _request_with_retry(session: requests.Session, method: str, url: str, **kwargs):
    last_exc = None
    for attempt, delay in enumerate(RETRY_DELAYS, start=1):
        try:
            resp = session.request(method, url, timeout=30, **kwargs)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code} {resp.reason}", response=resp)
            return resp
        except (requests.RequestException, requests.HTTPError) as exc:
            last_exc = exc
            if attempt == len(RETRY_DELAYS):
                break
            sleep_for = delay + random.uniform(0, 0.5)
            LOG.debug("retry %d after %.1fs: %s", attempt, sleep_for, exc)
            time.sleep(sleep_for)
    raise last_exc  # type: ignore[misc]


def _parse_isoforms(cc_text: str) -> list[dict]:
    """Parse `cc_alternative_products` free-text into a list of isoform accessions.

    The UniProt JSON returns this either as a string or a structured comment.
    We accept both.
    """
    isoforms: list[dict] = []
    if not cc_text:
        return isoforms
    # Structured form: handled by caller; this helper handles plain strings.
    # Look for "IsoId=XXXX-N" tokens.
    import re

    for m in re.finditer(r"IsoId=([A-Z0-9\-]+)", cc_text):
        isoforms.append({"accession": m.group(1)})
    return isoforms


def _extract_isoform_accessions(entry: dict) -> list[str]:
    """Pull isoform accessions from a UniProt JSON entry's `comments` block."""
    accs: list[str] = []
    for comment in entry.get("comments", []) or []:
        if comment.get("commentType") != "ALTERNATIVE PRODUCTS":
            continue
        for iso in comment.get("isoforms", []) or []:
            for a in iso.get("isoformIds", []) or []:
                accs.append(a)
    return accs


def _fetch_fasta_sequence(session: requests.Session, acc: str) -> str | None:
    try:
        resp = _request_with_retry(session, "GET", UNIPROT_FASTA.format(acc=acc))
    except requests.RequestException:
        return None
    if resp.status_code != 200 or not resp.text:
        return None
    lines = resp.text.splitlines()
    seq = "".join(l for l in lines if not l.startswith(">"))
    return seq or None


def _resolve_gene(
    session: requests.Session,
    gene: str,
    observed_accessions: set[str],
) -> tuple[str, dict | None, str | None]:
    """Resolve a single gene symbol against UniProt Swiss-Prot human.

    Returns (gene, entry_dict_or_None, error_reason_or_None).
    """
    params = {
        "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
        "fields": SEARCH_FIELDS,
        "format": "json",
        "size": 10,
    }
    try:
        resp = _request_with_retry(session, "GET", UNIPROT_SEARCH, params=params)
    except requests.RequestException as exc:
        return gene, None, f"network_error: {exc}"
    if resp.status_code != 200:
        return gene, None, f"http_{resp.status_code}"
    results = resp.json().get("results", []) or []
    if not results:
        return gene, None, "no_reviewed_human_hit"

    chosen = None
    ambiguous = False
    if len(results) == 1:
        chosen = results[0]
    else:
        # Prefer hits whose accession appears in observed PG.ProteinGroups.
        hits_in_obs = [
            r for r in results if r.get("primaryAccession") in observed_accessions
        ]
        if len(hits_in_obs) == 1:
            chosen = hits_in_obs[0]
        elif len(hits_in_obs) > 1:
            chosen = hits_in_obs[0]
            ambiguous = True
        else:
            primary = [
                r
                for r in results
                if (r.get("genes") or [{}])[0].get("geneName", {}).get("value") == gene
            ]
            if len(primary) == 1:
                chosen = primary[0]
            elif primary:
                chosen = primary[0]
                ambiguous = True
            else:
                chosen = results[0]
                ambiguous = True

    canonical_acc = chosen.get("primaryAccession")
    seq_obj = chosen.get("sequence") or {}
    canonical_seq = seq_obj.get("value") or ""
    canonical_len = seq_obj.get("length") or len(canonical_seq)
    protein_name = (
        (chosen.get("proteinDescription") or {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value")
    )

    isoform_accs = _extract_isoform_accessions(chosen)
    isoforms: list[dict] = []
    canonical_iso_id = f"{canonical_acc}-1"
    if not isoform_accs:
        isoforms.append(
            {
                "accession": canonical_iso_id,
                "length": canonical_len,
                "is_canonical": True,
                "sequence": canonical_seq,
            }
        )
    else:
        for a in isoform_accs:
            if a == canonical_iso_id or a == canonical_acc:
                isoforms.append(
                    {
                        "accession": a,
                        "length": canonical_len,
                        "is_canonical": True,
                        "sequence": canonical_seq,
                    }
                )
            else:
                seq = _fetch_fasta_sequence(session, a)
                if seq is None:
                    isoforms.append(
                        {
                            "accession": a,
                            "length": None,
                            "is_canonical": False,
                            "sequence": None,
                            "fetch_error": True,
                        }
                    )
                else:
                    isoforms.append(
                        {
                            "accession": a,
                            "length": len(seq),
                            "is_canonical": False,
                            "sequence": seq,
                        }
                    )

    entry = {
        "canonical_accession": canonical_acc,
        "canonical_sequence": canonical_seq,
        "canonical_length": canonical_len,
        "protein_name": protein_name,
        "isoforms": isoforms,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ambiguous_canonical": ambiguous,
    }
    return gene, entry, None


def _collect_observed_accessions(paths: Iterable[str]) -> dict[str, set[str]]:
    """Per-gene set of accessions observed in `PG.ProteinGroups`."""
    obs: dict[str, set[str]] = {}
    for path in paths:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            if "PG.Genes" not in fields or "PG.ProteinGroups" not in fields:
                continue
            for row in reader:
                genes_cell = (row.get("PG.Genes") or "").strip()
                accs_cell = (row.get("PG.ProteinGroups") or "").strip()
                if not genes_cell or not accs_cell:
                    continue
                accs = {a.strip() for a in accs_cell.replace(",", ";").split(";") if a.strip()}
                for g in genes_cell.replace(",", ";").split(";"):
                    g = g.strip()
                    if not g:
                        continue
                    obs.setdefault(g, set()).update(accs)
    return obs


def _atomic_write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_canonical_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _load_cache() -> dict:
    if os.path.exists(CANONICAL_MAP):
        with open(CANONICAL_MAP) as fh:
            return json.load(fh)
    return {}


def _write_unresolved(rows: list[tuple[str, str]]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(UNRESOLVED_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gene", "reason"])
        w.writerows(rows)


def run_uniprot_cache(force_refresh_genes: list[str] | None = None) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(FETCH_LOG, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    LOG.info("collecting gene symbols from %d input CSVs", len(INPUT_CSVS))
    genes = _collect_gene_symbols(INPUT_CSVS)
    LOG.info("found %d unique gene symbols", len(genes))

    cache = _load_cache()
    if force_refresh_genes:
        for g in force_refresh_genes:
            cache.pop(g, None)

    to_fetch = sorted(genes - set(cache.keys()))
    LOG.info("cache hits=%d, fetching=%d", len(cache), len(to_fetch))

    if not to_fetch:
        LOG.info("cache up to date; running sanity checks")
        _sanity_check(cache)
        return

    LOG.info("collecting observed PG.ProteinGroups per gene (for ambiguous resolution)")
    observed = _collect_observed_accessions(INPUT_CSVS)

    unresolved: list[tuple[str, str]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "alz-ingest-mukesh/1.0"})

    completed = 0
    flush_every = 250
    last_flush = 0
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(_resolve_gene, session, g, observed.get(g, set())): g
            for g in to_fetch
        }
        for fut in cf.as_completed(futures):
            gene = futures[fut]
            try:
                _, entry, err = fut.result()
            except Exception as exc:  # defensive: should be caught inside _resolve_gene
                err = f"unexpected_error: {exc}"
                entry = None
            if entry is not None:
                cache[gene] = entry
            else:
                unresolved.append((gene, err or "unknown"))
                LOG.warning("unresolved %s: %s", gene, err)
            completed += 1
            if completed - last_flush >= flush_every:
                _atomic_write_json(CANONICAL_MAP, cache)
                last_flush = completed
                LOG.info("progress %d/%d (cache flushed)", completed, len(to_fetch))

    _atomic_write_json(CANONICAL_MAP, cache)
    _write_unresolved(unresolved)
    LOG.info(
        "done: resolved=%d, unresolved=%d, total_cache_entries=%d",
        len(to_fetch) - len(unresolved),
        len(unresolved),
        len(cache),
    )
    _sanity_check(cache)
    _coverage_check(cache, genes)


def _sanity_check(cache: dict) -> None:
    misses = []
    for gene, expected_acc in SANITY_CONTROLS.items():
        entry = cache.get(gene)
        if entry is None:
            misses.append((gene, expected_acc, None))
            continue
        actual = entry.get("canonical_accession")
        if actual != expected_acc:
            misses.append((gene, expected_acc, actual))
    if misses:
        for gene, exp, act in misses:
            LOG.warning("sanity miss: %s expected=%s got=%s", gene, exp, act)
    else:
        LOG.info("sanity controls OK: %s", ", ".join(SANITY_CONTROLS))


def _coverage_check(cache: dict, all_genes: set[str]) -> None:
    if not all_genes:
        return
    resolved = sum(1 for g in all_genes if g in cache)
    pct = 100.0 * resolved / len(all_genes)
    LOG.info("coverage: %d/%d (%.1f%%) genes resolved", resolved, len(all_genes), pct)
    if pct < 98.0:
        LOG.warning("coverage below 98%% threshold (%.1f%%)", pct)


# ---------------------------------------------------------------------------
# Phase B — diagnostic pass
# ---------------------------------------------------------------------------

import re

_MOD_RE = re.compile(r"\[([^\]]+)\]")
_PHOSPHO_LABEL = "Phospho (STY)"
_PTM_TOKEN_RE = re.compile(r"([A-Z])(\d+)")


def _parse_modified_sequence(modseq: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Parse a Spectronaut `EG.ModifiedSequence` string.

    Returns (stripped_sequence, mods) where `mods` is a list of
    `(peptide_position_1_indexed, residue, mod_label)` tuples — one per `[...]`
    annotation, attached to the residue immediately preceding the bracket.
    Leading/trailing `_` delimiters are stripped; non-residue characters inside
    the peptide body (other than bracketed modifications) are ignored.
    """
    s = modseq.strip()
    if s.startswith("_"):
        s = s[1:]
    if s.endswith("_"):
        s = s[:-1]
    stripped_parts: list[str] = []
    mods: list[tuple[int, str, str]] = []
    i = 0
    pos = 0  # last residue position (1-indexed) consumed
    while i < len(s):
        c = s[i]
        if c == "[":
            j = s.find("]", i)
            if j < 0:
                break
            label = s[i + 1 : j]
            if pos > 0:
                residue = stripped_parts[-1]
                mods.append((pos, residue, label))
            i = j + 1
        elif c.isalpha():
            stripped_parts.append(c)
            pos += 1
            i += 1
        else:
            i += 1
    return "".join(stripped_parts), mods


def _parse_ptm_locations(text: str) -> list[tuple[str, int]]:
    """Parse `EG.ProteinPTMLocations` like `(S55, T65)` into [(residue, pos), ...]."""
    if not text or text == "?":
        return []
    out: list[tuple[str, int]] = []
    for m in _PTM_TOKEN_RE.finditer(text):
        residue, pos = m.group(1), int(m.group(2))
        out.append((residue, pos))
    return out


def _find_all_occurrences(needle: str, haystack: str) -> list[int]:
    """Return all 1-indexed start positions of `needle` in `haystack`."""
    if not needle or not haystack:
        return []
    starts: list[int] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        starts.append(idx + 1)
        start = idx + 1
    return starts


AUDIT_COLUMNS = [
    "track",
    "source_row_idx",
    "gene",
    "pg_protein_groups",
    "pep_stripped_sequence",
    "eg_modified_sequence",
    "eg_protein_ptm_locations",
    "canonical_accession",
    "canonical_unresolved",
    "canonical_match_count",
    "canonical_match_starts",
    "isoform_match_acc",
    "isoform_specific",
    "peptide_not_in_any",
    "peptide_phospho_position",
    "absolute_position",
    "computed_residue",
    "marker_residue",
    "residue_matches",
    "spectronaut_position",
    "spectronaut_agrees",
    "phospho_count_in_peptide",
    "multi_phospho",
    "site_id_proposed",
]


def _diagnose_track(
    csv_path: str,
    track_name: str,
    cache: dict,
    writer: csv.DictWriter,
    counters: dict,
) -> None:
    """Stream one peptide report; emit audit rows + update counters."""
    is_py = track_name == "pY"
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for source_row_idx, row in enumerate(reader):
            gene = (row.get("PG.Genes") or "").strip()
            pg_groups = (row.get("PG.ProteinGroups") or "").strip()
            modseq = (row.get("EG.ModifiedSequence") or "").strip()
            stripped = (row.get("PEP.StrippedSequence") or "").strip()
            ptm_locs_raw = (row.get("EG.ProteinPTMLocations") or "").strip()

            # Parse all mods, filter to phospho only.
            _, all_mods = _parse_modified_sequence(modseq)
            phospho_mods = [m for m in all_mods if m[2] == _PHOSPHO_LABEL]
            counters["rows_seen"][track_name] += 1
            if not phospho_mods:
                counters["rows_no_phospho"][track_name] += 1
                if is_py:
                    counters["rows_dropped_py_nonphospho"] += 1
                continue

            # Locked drop: in pY file, drop non-Y phosphos. If none remain, skip row.
            if is_py:
                kept = [m for m in phospho_mods if m[1] == "Y"]
                dropped = len(phospho_mods) - len(kept)
                counters["py_nonY_markers_dropped"] += dropped
                phospho_mods = kept
                if not phospho_mods:
                    counters["rows_dropped_py_nonY"] += 1
                    continue

            # First gene only — multi-gene cells fall through with the first.
            primary_gene = gene.split(";")[0].strip() if gene else ""
            entry = cache.get(primary_gene)
            canonical_unresolved = entry is None
            if canonical_unresolved:
                counters["rows_canonical_unresolved"][track_name] += 1
                # Fallback accession: first listed in PG.ProteinGroups.
                fallback_acc = (
                    pg_groups.split(";")[0].split(",")[0].strip() if pg_groups else ""
                )
                # Emit one stub audit row per marker.
                for pep_pos, marker_residue, _ in phospho_mods:
                    writer.writerow(
                        {
                            "track": track_name,
                            "source_row_idx": source_row_idx,
                            "gene": gene,
                            "pg_protein_groups": pg_groups,
                            "pep_stripped_sequence": stripped,
                            "eg_modified_sequence": modseq,
                            "eg_protein_ptm_locations": "" if is_py else ptm_locs_raw,
                            "canonical_accession": fallback_acc,
                            "canonical_unresolved": True,
                            "canonical_match_count": "",
                            "canonical_match_starts": "",
                            "isoform_match_acc": "",
                            "isoform_specific": "",
                            "peptide_not_in_any": "",
                            "peptide_phospho_position": pep_pos,
                            "absolute_position": "",
                            "computed_residue": "",
                            "marker_residue": marker_residue,
                            "residue_matches": "",
                            "spectronaut_position": "",
                            "spectronaut_agrees": "n_a" if is_py else "",
                            "phospho_count_in_peptide": len(phospho_mods),
                            "multi_phospho": len(phospho_mods) > 1,
                            "site_id_proposed": "",
                        }
                    )
                    counters["audit_rows"][track_name] += 1
                continue

            canonical_acc = entry["canonical_accession"]
            canonical_seq = entry["canonical_sequence"]

            # Peptide containment in canonical sequence.
            canonical_starts = _find_all_occurrences(stripped, canonical_seq)
            match_count = len(canonical_starts)

            isoform_match_acc = ""
            isoform_specific = False
            peptide_not_in_any = False
            chosen_match_start: int | None = None

            if match_count == 1:
                chosen_match_start = canonical_starts[0]
            elif match_count == 0:
                # Try non-canonical isoforms in descending length order.
                isoforms = entry.get("isoforms") or []
                non_canon = [
                    iso
                    for iso in isoforms
                    if not iso.get("is_canonical") and iso.get("sequence")
                ]
                non_canon.sort(key=lambda iso: iso.get("length") or 0, reverse=True)
                for iso in non_canon:
                    iso_starts = _find_all_occurrences(stripped, iso["sequence"])
                    if len(iso_starts) == 1:
                        isoform_match_acc = iso["accession"]
                        isoform_specific = True
                        chosen_match_start = iso_starts[0]
                        # Replace working sequence for residue lookup
                        canonical_seq = iso["sequence"]
                        canonical_acc = iso["accession"]
                        counters["isoform_specific"][track_name] += 1
                        break
                if chosen_match_start is None:
                    peptide_not_in_any = True
                    counters["peptide_not_in_any"][track_name] += 1
            else:
                # multi-match: do not pick. Record only.
                counters["canonical_multi"][track_name] += 1

            ptm_locs = _parse_ptm_locations(ptm_locs_raw) if not is_py else []
            # Phospho-only entries from ptm_locs, in N-to-C order.
            ptm_phospho_positions = [
                (r, p) for (r, p) in ptm_locs if r in ("S", "T", "Y")
            ]

            for marker_idx, (pep_pos, marker_residue, _) in enumerate(phospho_mods):
                if chosen_match_start is not None:
                    abs_pos = chosen_match_start + pep_pos - 1
                    computed = (
                        canonical_seq[abs_pos - 1]
                        if 1 <= abs_pos <= len(canonical_seq)
                        else ""
                    )
                    residue_matches = computed == marker_residue
                    if not residue_matches:
                        counters["residue_mismatch"][track_name] += 1
                else:
                    abs_pos = None
                    computed = ""
                    residue_matches = ""

                # Spectronaut cross-check (IMAC only).
                spectronaut_position = ""
                spectronaut_agrees = "n_a"
                if not is_py:
                    if marker_idx < len(ptm_phospho_positions):
                        sp_res, sp_pos = ptm_phospho_positions[marker_idx]
                        spectronaut_position = sp_pos
                        if abs_pos is not None:
                            agrees = (sp_res == marker_residue) and (sp_pos == abs_pos)
                            spectronaut_agrees = "true" if agrees else "false"
                            if not agrees:
                                counters["spectronaut_disagree"] += 1
                        else:
                            spectronaut_agrees = "n_a"
                    else:
                        spectronaut_agrees = "n_a"

                site_id_proposed = ""
                if abs_pos is not None and computed in ("S", "T", "Y") and residue_matches:
                    site_id_proposed = f"{canonical_acc}_{computed}{abs_pos}"

                writer.writerow(
                    {
                        "track": track_name,
                        "source_row_idx": source_row_idx,
                        "gene": gene,
                        "pg_protein_groups": pg_groups,
                        "pep_stripped_sequence": stripped,
                        "eg_modified_sequence": modseq,
                        "eg_protein_ptm_locations": "" if is_py else ptm_locs_raw,
                        "canonical_accession": canonical_acc,
                        "canonical_unresolved": False,
                        "canonical_match_count": match_count,
                        "canonical_match_starts": ";".join(str(s) for s in canonical_starts),
                        "isoform_match_acc": isoform_match_acc,
                        "isoform_specific": isoform_specific,
                        "peptide_not_in_any": peptide_not_in_any,
                        "peptide_phospho_position": pep_pos,
                        "absolute_position": "" if abs_pos is None else abs_pos,
                        "computed_residue": computed,
                        "marker_residue": marker_residue,
                        "residue_matches": residue_matches,
                        "spectronaut_position": spectronaut_position,
                        "spectronaut_agrees": spectronaut_agrees,
                        "phospho_count_in_peptide": len(phospho_mods),
                        "multi_phospho": len(phospho_mods) > 1,
                        "site_id_proposed": site_id_proposed,
                    }
                )
                counters["audit_rows"][track_name] += 1
                if match_count == 1:
                    counters["canonical_single"][track_name] += 1
                elif match_count == 0 and not peptide_not_in_any:
                    counters["canonical_zero_isoform_rescue"][track_name] += 1
                if len(phospho_mods) > 1:
                    counters["multi_phospho_markers"][track_name] += 1
                counters["residue_breakdown"][track_name][marker_residue] = (
                    counters["residue_breakdown"][track_name].get(marker_residue, 0) + 1
                )


def run_diagnose() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    if not os.path.exists(CANONICAL_MAP):
        LOG.error("canonical_map.json missing — run --uniprot-cache first")
        sys.exit(2)
    os.makedirs(HUMAN_DATA_INGEST_DIR, exist_ok=True)
    cache = _load_cache()
    LOG.info("loaded canonical_map with %d entries", len(cache))

    counters = {
        "rows_seen": {"IMAC": 0, "pY": 0},
        "rows_no_phospho": {"IMAC": 0, "pY": 0},
        "rows_dropped_py_nonphospho": 0,
        "rows_dropped_py_nonY": 0,
        "py_nonY_markers_dropped": 0,
        "rows_canonical_unresolved": {"IMAC": 0, "pY": 0},
        "audit_rows": {"IMAC": 0, "pY": 0},
        "canonical_single": {"IMAC": 0, "pY": 0},
        "canonical_multi": {"IMAC": 0, "pY": 0},
        "canonical_zero_isoform_rescue": {"IMAC": 0, "pY": 0},
        "isoform_specific": {"IMAC": 0, "pY": 0},
        "peptide_not_in_any": {"IMAC": 0, "pY": 0},
        "residue_mismatch": {"IMAC": 0, "pY": 0},
        "spectronaut_disagree": 0,
        "multi_phospho_markers": {"IMAC": 0, "pY": 0},
        "residue_breakdown": {"IMAC": {}, "pY": {}},
    }

    with open(SYNTHESIS_AUDIT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for path, name in [(IMAC_REPORT, "IMAC"), (PY_REPORT, "pY")]:
            LOG.info("diagnosing %s ...", name)
            _diagnose_track(path, name, cache, writer, counters)

    _write_diagnose_summary(counters)


def _fmt_pct(num: int, denom: int) -> str:
    return f"{100.0*num/denom:.2f}%" if denom else "n/a"


def _write_diagnose_summary(counters: dict) -> None:
    lines: list[str] = []
    lines.append("Phase B — synthesis audit summary")
    lines.append("=" * 60)
    for track in ("IMAC", "pY"):
        seen = counters["rows_seen"][track]
        no_phos = counters["rows_no_phospho"][track]
        audit_rows = counters["audit_rows"][track]
        kept_phos_rows = seen - no_phos
        if track == "pY":
            kept_phos_rows -= counters["rows_dropped_py_nonY"]
        single = counters["canonical_single"][track]
        multi = counters["canonical_multi"][track]
        zero_rescued = counters["canonical_zero_isoform_rescue"][track]
        iso_specific = counters["isoform_specific"][track]
        not_in_any = counters["peptide_not_in_any"][track]
        resid_mismatch = counters["residue_mismatch"][track]
        unresolved = counters["rows_canonical_unresolved"][track]
        multi_phos = counters["multi_phospho_markers"][track]
        res_breakdown = counters["residue_breakdown"][track]
        denom_markers = max(audit_rows, 1)

        lines.append(f"\n[{track}]")
        lines.append(f"  source rows seen:                  {seen}")
        lines.append(f"  rows with no phospho marker:        {no_phos}")
        if track == "pY":
            lines.append(
                f"  rows dropped (locked pY non-phospho):{counters['rows_dropped_py_nonphospho']}"
            )
            lines.append(
                f"  rows dropped (all markers non-Y):    {counters['rows_dropped_py_nonY']}"
            )
            lines.append(
                f"  non-Y markers dropped (locked):       {counters['py_nonY_markers_dropped']}"
            )
        lines.append(f"  rows kept (phospho-bearing):        {kept_phos_rows}")
        lines.append(f"  audit rows emitted (per marker):    {audit_rows}")
        lines.append(
            f"  canonical_match_count == 1:          {single} ({_fmt_pct(single, denom_markers)})"
        )
        lines.append(
            f"  canonical_match_count == 0 (rescue): {zero_rescued} ({_fmt_pct(zero_rescued, denom_markers)})"
        )
        lines.append(
            f"  canonical_match_count >= 2:          {multi} ({_fmt_pct(multi, denom_markers)})"
        )
        lines.append(
            f"  isoform_specific (rescued):          {iso_specific} ({_fmt_pct(iso_specific, denom_markers)})"
        )
        lines.append(
            f"  peptide_not_in_any:                  {not_in_any} ({_fmt_pct(not_in_any, denom_markers)})"
        )
        lines.append(
            f"  residue_mismatch (computed != marker):{resid_mismatch} ({_fmt_pct(resid_mismatch, denom_markers)})"
        )
        lines.append(f"  canonical_unresolved rows:           {unresolved}")
        lines.append(
            f"  multi-phospho markers:                {multi_phos} ({_fmt_pct(multi_phos, denom_markers)})"
        )
        lines.append(
            f"  residue breakdown (marker residue):  {dict(sorted(res_breakdown.items()))}"
        )
    lines.append(
        f"\n  spectronaut_agrees == false (IMAC):  {counters['spectronaut_disagree']}"
    )

    text = "\n".join(lines) + "\n"
    with open(SYNTHESIS_AUDIT_SUMMARY, "w") as fh:
        fh.write(text)
    print(text)


# ---------------------------------------------------------------------------
# Phase C — reshape into Song-shaped artifacts (gated)
# ---------------------------------------------------------------------------

POLICY_FILE = os.path.join(
    config.REPO_ROOT, "docs", "audits", "mukesh_ingest_policies.yml"
)
SAMPLE_MAPPING_CSV = os.path.join(HUMAN_DATA_INGEST_DIR, "sample_mapping.csv")
SAMPLE_EXCLUSIONS_CSV = os.path.join(HUMAN_DATA_INGEST_DIR, "sample_exclusions.csv")
STOICH_MATRIX_CSV = os.path.join(HUMAN_KINASE_DIR, "stoichiometry_matrix.csv")
RAW_PHOSPHO_CSV = os.path.join(HUMAN_KINASE_DIR, "raw_phospho_normalized.csv")
SYNTHESIS_DROPPED_CSV = os.path.join(HUMAN_KINASE_DIR, "synthesis_dropped.csv")
STOICH_DROPPED_CSV = os.path.join(HUMAN_KINASE_DIR, "stoichiometry_dropped.csv")
INGEST_MANIFEST = os.path.join(HUMAN_KINASE_DIR, "ingest_manifest.json")

_SAMPLE_COL_RE = re.compile(r"(?:[_\-])(AD|CTRL)-(\d+)\.raw")
REQUIRED_POLICIES = {
    "canonical_match_count_zero",
    "canonical_match_count_multi",
    "peptide_not_in_any",
    "residue_mismatch",
    "spectronaut_disagreement_imac",
    "canonical_unresolved",
}


def _load_policy() -> dict:
    if not os.path.exists(POLICY_FILE):
        LOG.error("policy file missing: %s", POLICY_FILE)
        sys.exit(2)
    import yaml

    with open(POLICY_FILE) as fh:
        policy = yaml.safe_load(fh) or {}
    if not policy.get("policies_reviewed"):
        LOG.error("policies_reviewed != true in %s — refusing to reshape", POLICY_FILE)
        sys.exit(2)
    edge = policy.get("edge_cases") or {}
    missing = REQUIRED_POLICIES - set(edge.keys())
    if missing:
        LOG.error("policy file missing required edge_cases: %s", sorted(missing))
        sys.exit(2)
    for k in REQUIRED_POLICIES:
        if not (edge[k].get("policy")):
            LOG.error("edge_cases.%s is missing a policy value", k)
            sys.exit(2)
    return policy


def _parse_sample_columns(header: list[str]) -> list[tuple[str, str, str]]:
    """Return [(column_name, sample_id, group), ...] for sample-quant columns."""
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for col in header:
        if "Quantity" not in col:
            continue
        m = _SAMPLE_COL_RE.search(col)
        if not m:
            continue
        group, num = m.group(1), m.group(2)
        sample_id = f"{group}-{num}"
        if sample_id in seen:
            continue  # take first matching column per sample
        seen.add(sample_id)
        out.append((col, sample_id, group))
    return out


def _read_protein_quant(cache: dict) -> tuple[dict[str, dict[str, float]], list[tuple[str, str]]]:
    """Return (acc → {sample_id → log2_quant}, [(sample_id, group), ...] from header)."""
    import math

    with open(PROTEIN_REPORT, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        sample_cols = _parse_sample_columns(fields)
        if not sample_cols:
            raise RuntimeError("no sample columns parsed from protein report header")
        samples = [(s, g) for (_, s, g) in sample_cols]
        out: dict[str, dict[str, float]] = {}
        # Build gene → canonical_accession map for quick lookup.
        gene_to_canon = {
            g: e["canonical_accession"] for g, e in cache.items()
        }
        for row in reader:
            gene_cell = (row.get("PG.Genes") or "").strip()
            primary_gene = gene_cell.split(";")[0].strip() if gene_cell else ""
            accs_cell = (row.get("PG.ProteinAccessions") or "").strip()
            accs = [a.strip() for a in accs_cell.replace(",", ";").split(";") if a.strip()]
            # Key by canonical when the gene is in the UniProt cache (so the
            # protein row joins to the phospho sites which are canonical-keyed).
            # Falls back to the first listed accession only when canonical is
            # unknown. Mukesh's protein report sometimes ships isoform tags
            # (e.g. MAPT → "0N3R") in PG.ProteinAccessions instead of UniProt
            # accessions; collapsing all such rows to canonical for the gene
            # matches the canonical-collapse decision documented in the plan.
            canon = gene_to_canon.get(primary_gene)
            chosen = canon if canon else (accs[0] if accs else None)
            if not chosen:
                continue
            vals: dict[str, float] = {}
            for col, sid, _ in sample_cols:
                v = row.get(col, "")
                if v is None or v == "" or v in ("Filtered", "NaN"):
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    continue
                if fv <= 0:
                    continue
                vals[sid] = math.log2(fv)
            if not vals:
                continue
            # If this canonical accession is already present, keep the max-coverage row.
            prev = out.get(chosen)
            if prev is None or len(vals) > len(prev):
                out[chosen] = vals
        return out, samples


def _read_audit_filtered(policy: dict) -> tuple[list[dict], list[dict]]:
    """Apply policies; return (kept, dropped) audit-row dicts."""
    edge = policy["edge_cases"]
    pol_zero = edge["canonical_match_count_zero"]["policy"]
    pol_multi = edge["canonical_match_count_multi"]["policy"]
    pol_not_in_any = edge["peptide_not_in_any"]["policy"]
    pol_mismatch = edge["residue_mismatch"]["policy"]
    pol_unresolved = edge["canonical_unresolved"]["policy"]

    kept: list[dict] = []
    dropped: list[dict] = []
    with open(SYNTHESIS_AUDIT_CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            reason = None
            if r["canonical_unresolved"] == "True":
                if pol_unresolved == "drop":
                    reason = "canonical_unresolved"
            elif r["peptide_not_in_any"] == "True":
                if pol_not_in_any == "drop":
                    reason = "peptide_not_in_any"
            elif r["residue_matches"] == "False":
                if pol_mismatch == "drop":
                    reason = "residue_mismatch"
            else:
                mc = r["canonical_match_count"]
                mc_int = int(mc) if mc not in ("", None) else -1
                if mc_int == 0:
                    if pol_zero == "drop":
                        reason = "canonical_match_count_zero"
                elif mc_int >= 2:
                    if pol_multi == "drop":
                        reason = "canonical_match_count_multi"
                    # first_match: take first canonical start; site_id_proposed is empty
                    # for multi rows in Phase B; recompute here using first match.
                    elif pol_multi == "first_match":
                        starts = (r.get("canonical_match_starts") or "").split(";")
                        if starts and starts[0]:
                            first = int(starts[0])
                            pep_pos = int(r["peptide_phospho_position"])
                            abs_pos = first + pep_pos - 1
                            r["absolute_position"] = abs_pos
                            r["site_id_proposed"] = (
                                f"{r['canonical_accession']}_{r['marker_residue']}{abs_pos}"
                            )
                        else:
                            reason = "canonical_match_count_multi_no_start"
            if reason:
                r["drop_reason"] = reason
                dropped.append(r)
            else:
                kept.append(r)
    return kept, dropped


def _read_phospho_quant(
    track_name: str,
    quantity_col_keyword: str,
    audit_rows: list[dict],
    samples: list[tuple[str, str]],
) -> dict[str, dict[str, float]]:
    """Aggregate peptide quants to site_id (median of log2 across peptides).

    `audit_rows` is the kept audit slice for this track; rows here drive which
    source rows to read. Multiple audit rows can map to the same source CSV row
    (multi-phospho peptide) — they share the peptide quantity.
    """
    import math
    from collections import defaultdict

    csv_path = IMAC_REPORT if track_name == "IMAC" else PY_REPORT
    # group audit rows by source_row_idx
    by_idx: dict[int, list[dict]] = defaultdict(list)
    for r in audit_rows:
        if r["track"] != track_name:
            continue
        by_idx[int(r["source_row_idx"])].append(r)
    if not by_idx:
        return {}

    site_log2: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        sample_cols = [
            (c, sid)
            for c in fields
            for (_, sid, _) in [next(((cc, ss, gg) for (cc, ss, gg) in _parse_sample_columns([c])), (None, None, None))]
            if quantity_col_keyword in c and sid is not None
        ]
        # _parse_sample_columns dedupes; redo simpler:
        sample_cols = []
        seen: set[str] = set()
        for c in fields:
            if quantity_col_keyword not in c:
                continue
            m = _SAMPLE_COL_RE.search(c)
            if not m:
                continue
            sid = f"{m.group(1)}-{m.group(2)}"
            if sid in seen:
                continue
            seen.add(sid)
            sample_cols.append((c, sid))

        for source_idx, row in enumerate(reader):
            audits = by_idx.get(source_idx)
            if not audits:
                continue
            # Compute log2 quant per sample for this peptide row.
            peptide_log2: dict[str, float] = {}
            for col, sid in sample_cols:
                v = row.get(col, "")
                if v is None or v == "" or v in ("Filtered", "NaN"):
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    continue
                if fv <= 0:
                    continue
                peptide_log2[sid] = math.log2(fv)
            if not peptide_log2:
                continue
            for a in audits:
                site_id = a.get("site_id_proposed") or ""
                if not site_id:
                    continue
                for sid, log2v in peptide_log2.items():
                    site_log2[site_id][sid].append(log2v)

    # Median aggregation across peptides for each (site, sample).
    import statistics

    out: dict[str, dict[str, float]] = {}
    for site_id, sample_lists in site_log2.items():
        out[site_id] = {sid: statistics.median(vals) for sid, vals in sample_lists.items()}
    return out


def _extract_motif(seq: str, pos_1indexed: int, flank: int = 7) -> str:
    """±flank window around `pos_1indexed`; pad with `_` at termini."""
    if not seq or pos_1indexed < 1 or pos_1indexed > len(seq):
        return ""
    center = seq[pos_1indexed - 1]
    left_start = pos_1indexed - 1 - flank
    right_end = pos_1indexed + flank
    left = seq[max(0, left_start):pos_1indexed - 1]
    right = seq[pos_1indexed:min(len(seq), right_end)]
    if left_start < 0:
        left = "_" * (-left_start) + left
    if right_end > len(seq):
        right = right + "_" * (right_end - len(seq))
    return left + center + right


def _robust_zscore_outliers(
    sample_vals: dict[str, list[float]], samples_groups: dict[str, str], thresh: float
) -> dict[str, dict]:
    """Within-group robust z (median/MAD) on per-sample summary; flag |z|>thresh.

    sample_vals: sample_id → list of per-site log2 protein quants.
    Returns: sample_id → {median, mad_z, outlier_flag, group}.
    """
    import statistics
    from collections import defaultdict

    # Per-sample summary statistic: median of log2 protein values.
    summary = {sid: statistics.median(vals) for sid, vals in sample_vals.items() if vals}
    by_group: dict[str, list[str]] = defaultdict(list)
    for sid in summary:
        by_group[samples_groups[sid]].append(sid)
    out: dict[str, dict] = {}
    for g, sids in by_group.items():
        vals = [summary[s] for s in sids]
        med = statistics.median(vals)
        absdev = [abs(v - med) for v in vals]
        mad = statistics.median(absdev)
        scale = 1.4826 * mad if mad > 0 else float("nan")
        for s in sids:
            z = (summary[s] - med) / scale if scale and scale == scale else 0.0
            out[s] = {
                "group": g,
                "median_log2_protein": summary[s],
                "robust_z": z,
                "outlier": abs(z) > thresh,
            }
    return out


def _file_sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_reshape() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    if not os.path.exists(SYNTHESIS_AUDIT_CSV):
        LOG.error("Phase B audit missing — run --diagnose first")
        sys.exit(2)

    policy = _load_policy()
    LOG.info("policy file accepted")
    cache = _load_cache()

    kept, dropped = _read_audit_filtered(policy)
    LOG.info("audit rows: kept=%d dropped=%d", len(kept), len(dropped))

    os.makedirs(HUMAN_DATA_INGEST_DIR, exist_ok=True)
    os.makedirs(HUMAN_KINASE_DIR, exist_ok=True)

    # Write drop sidecar.
    with open(SYNTHESIS_DROPPED_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS + ["drop_reason"])
        w.writeheader()
        for r in dropped:
            w.writerow({k: r.get(k, "") for k in AUDIT_COLUMNS + ["drop_reason"]})

    # Protein quant.
    LOG.info("reading protein report ...")
    protein_quant, samples = _read_protein_quant(cache)
    LOG.info(
        "protein quant: %d accessions × %d samples", len(protein_quant), len(samples)
    )
    sample_ids = [s for s, _ in samples]
    samples_groups = {s: g for s, g in samples}
    with open(SAMPLE_MAPPING_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "group"])
        for s, g in samples:
            w.writerow([s, g])

    # Phospho quant per track.
    LOG.info("aggregating IMAC phospho quants ...")
    imac_q = _read_phospho_quant(
        "IMAC",
        policy["quantity_columns"]["imac_phospho"],
        kept,
        samples,
    )
    LOG.info("IMAC sites: %d", len(imac_q))
    LOG.info("aggregating pY phospho quants ...")
    py_q = _read_phospho_quant(
        "pY",
        policy["quantity_columns"]["py_phospho"],
        kept,
        samples,
    )
    LOG.info("pY sites: %d", len(py_q))

    # Outlier exclusion on protein log2 (gated by policy).
    # NBB cohort runs per-donor MEA (each AD donor compared to mean(CTRL)),
    # so we record robust-z diagnostics but do not drop any samples — every
    # AD donor must reach Stage 2 as its own contrast.
    exclude_outliers = bool(policy.get("sample_outlier_exclusion", True))
    per_sample_vals: dict[str, list[float]] = {sid: [] for sid in sample_ids}
    for acc, smap in protein_quant.items():
        for sid, v in smap.items():
            per_sample_vals[sid].append(v)
    outliers = _robust_zscore_outliers(
        per_sample_vals, samples_groups, config.OUTLIER_ZSCORE_THRESH
    )
    with open(SAMPLE_EXCLUSIONS_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["sample_id", "group", "median_log2_protein", "robust_z", "outlier_flag", "excluded"]
        )
        for sid in sample_ids:
            d = outliers.get(sid, {})
            flag = d.get("outlier", False)
            w.writerow([
                sid,
                d.get("group", ""),
                d.get("median_log2_protein", ""),
                d.get("robust_z", ""),
                flag,
                bool(flag and exclude_outliers),
            ])
    flagged = sorted([s for s, d in outliers.items() if d["outlier"]])
    if flagged and exclude_outliers:
        LOG.warning("outlier samples excluded: %s", flagged)
        excluded = flagged
    else:
        if flagged:
            LOG.info(
                "outlier samples flagged but kept (sample_outlier_exclusion=false): %s",
                flagged,
            )
        excluded = []
    kept_samples = [s for s in sample_ids if s not in excluded]

    # Build site metadata table from kept audit rows.
    # Key by site_id_proposed; first-seen audit row wins for metadata.
    site_meta: dict[str, dict] = {}
    for r in kept:
        site_id = r.get("site_id_proposed")
        if not site_id or site_id in site_meta:
            continue
        acc = r["canonical_accession"]
        try:
            abs_pos = int(r["absolute_position"])
        except (TypeError, ValueError):
            continue
        residue = r["marker_residue"]
        # Sequence to slice for motif: canonical of resolved gene, else isoform.
        if r.get("isoform_specific") == "True" and r.get("isoform_match_acc"):
            seq = None
            primary_gene = (r.get("gene") or "").split(";")[0].strip()
            entry = cache.get(primary_gene)
            if entry:
                for iso in entry.get("isoforms") or []:
                    if iso.get("accession") == r["isoform_match_acc"]:
                        seq = iso.get("sequence")
                        break
        else:
            primary_gene = (r.get("gene") or "").split(";")[0].strip()
            entry = cache.get(primary_gene)
            seq = entry.get("canonical_sequence") if entry else None
        motif = _extract_motif(seq, abs_pos) if seq else ""
        site_meta[site_id] = {
            "site_id": site_id,
            "protein_id": acc,
            "gene_symbol": (r.get("gene") or "").split(";")[0].strip(),
            "site_position": f"{residue}{abs_pos}",
            "motif": motif,
        }

    # Emit phospho and stoichiometry per track.
    stoich_dropped: list[dict] = []

    # Gene → canonical accession lookup for isoform-specific site rescue.
    # PG.Quantity collapses isoforms at the quant step (Mukesh ships one
    # PG row per gene regardless of isoform attribution), so an
    # isoform-specific site (e.g. P10636-8_S113) joins to parent protein
    # quant under the gene's canonical accession (P10636). This preserves
    # the isoform-level site_id while still enabling stoichiometry.
    gene_to_canon: dict[str, str] = {}
    for g, entry in cache.items():
        ca = (entry or {}).get("canonical_accession")
        if ca:
            gene_to_canon[g] = ca

    def _emit(track_name: str, site_quant: dict[str, dict[str, float]]) -> tuple[int, int]:
        suffix = "" if track_name == "IMAC" else "_pY"
        phospho_path = os.path.join(
            HUMAN_KINASE_DIR, f"raw_phospho_normalized{suffix}.csv"
        )
        stoich_path = os.path.join(HUMAN_KINASE_DIR, f"stoichiometry_matrix{suffix}.csv")
        meta_cols = ["site_id", "protein_id", "gene_symbol", "site_position", "motif"]
        cols = meta_cols + kept_samples
        n_phos = 0
        n_stoich = 0
        with open(phospho_path, "w", newline="") as fh_p, open(
            stoich_path, "w", newline=""
        ) as fh_s:
            wp = csv.writer(fh_p)
            ws = csv.writer(fh_s)
            wp.writerow(cols)
            ws.writerow(cols)
            for site_id, qmap in site_quant.items():
                meta = site_meta.get(site_id)
                if not meta:
                    continue
                phospho_row = [meta[c] for c in meta_cols] + [
                    qmap.get(s, "") for s in kept_samples
                ]
                wp.writerow(phospho_row)
                n_phos += 1
                parent = protein_quant.get(meta["protein_id"])
                parent_source = meta["protein_id"]
                if parent is None and "-" in meta["protein_id"]:
                    # Isoform-specific site: fall back to gene's canonical
                    # accession since PG.Quantity is gene-level only.
                    canon = gene_to_canon.get(meta["gene_symbol"])
                    if canon:
                        parent = protein_quant.get(canon)
                        if parent is not None:
                            parent_source = canon
                if parent is None:
                    stoich_dropped.append(
                        {
                            "site_id": site_id,
                            "protein_id": meta["protein_id"],
                            "reason": "parent_protein_not_quantified",
                        }
                    )
                    continue
                stoich_row = [meta[c] for c in meta_cols]
                for s in kept_samples:
                    p = qmap.get(s)
                    pr = parent.get(s)
                    if p is None or pr is None:
                        stoich_row.append("")
                    else:
                        stoich_row.append(p - pr)
                ws.writerow(stoich_row)
                n_stoich += 1
        return n_phos, n_stoich

    nph_imac, ns_imac = _emit("IMAC", imac_q)
    nph_py, ns_py = _emit("pY", py_q)
    LOG.info(
        "emitted IMAC phospho=%d stoich=%d | pY phospho=%d stoich=%d",
        nph_imac, ns_imac, nph_py, ns_py,
    )

    with open(STOICH_DROPPED_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["site_id", "protein_id", "reason"])
        w.writeheader()
        w.writerows(stoich_dropped)

    # Concatenate IMAC + pY into the canonical Song-shaped filenames as well,
    # for downstream tooling that expects single files.
    def _concat(track_files: list[str], out_path: str) -> None:
        wrote_header = False
        with open(out_path, "w", newline="") as outfh:
            w = csv.writer(outfh)
            for p in track_files:
                if not os.path.exists(p):
                    continue
                with open(p, newline="") as infh:
                    r = csv.reader(infh)
                    hdr = next(r)
                    if not wrote_header:
                        w.writerow(hdr)
                        wrote_header = True
                    for row in r:
                        w.writerow(row)

    _concat(
        [
            os.path.join(HUMAN_KINASE_DIR, "stoichiometry_matrix.csv"),
            os.path.join(HUMAN_KINASE_DIR, "stoichiometry_matrix_pY.csv"),
        ],
        os.path.join(HUMAN_KINASE_DIR, "stoichiometry_matrix_all.csv"),
    )
    _concat(
        [
            os.path.join(HUMAN_KINASE_DIR, "raw_phospho_normalized.csv"),
            os.path.join(HUMAN_KINASE_DIR, "raw_phospho_normalized_pY.csv"),
        ],
        os.path.join(HUMAN_KINASE_DIR, "raw_phospho_normalized_all.csv"),
    )

    # Manifest.
    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": [
            {
                "path": os.path.relpath(p, config.REPO_ROOT),
                "size_bytes": os.path.getsize(p),
                "sha256": _file_sha256(p),
            }
            for p in INPUT_CSVS
        ],
        "uniprot_cache_entries": len(cache),
        "policy_file_sha256": _file_sha256(POLICY_FILE),
        "outlier_threshold": config.OUTLIER_ZSCORE_THRESH,
        "samples_total": len(sample_ids),
        "samples_excluded": excluded,
        "samples_kept": kept_samples,
        "counts": {
            "audit_rows_kept": len(kept),
            "audit_rows_dropped": len(dropped),
            "protein_accessions": len(protein_quant),
            "imac_phospho_rows": nph_imac,
            "imac_stoich_rows": ns_imac,
            "py_phospho_rows": nph_py,
            "py_stoich_rows": ns_py,
        },
    }
    with open(INGEST_MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
    LOG.info("manifest written: %s", INGEST_MANIFEST)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uniprot-cache",
        action="store_true",
        help="Phase A: build UniProt canonical-isoform cache.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Phase B: emit synthesis_audit.csv (not yet implemented).",
    )
    parser.add_argument(
        "--reshape",
        action="store_true",
        help="Phase C: emit Song-shaped artifacts (not yet implemented).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of cached outputs.",
    )
    args = parser.parse_args(argv)

    if args.uniprot_cache:
        run_uniprot_cache()
        return 0
    if args.diagnose:
        run_diagnose()
        return 0
    if args.reshape:
        run_reshape()
        return 0
    if args.summary:
        if not os.path.exists(CANONICAL_MAP):
            print("no canonical_map.json — run --uniprot-cache first")
            return 0
        cache = _load_cache()
        print(f"canonical_map.json: {len(cache)} gene entries")
        ambig = sum(1 for v in cache.values() if v.get("ambiguous_canonical"))
        print(f"  ambiguous canonical choices: {ambig}")
        iso_counts = [len(v.get("isoforms") or []) for v in cache.values()]
        if iso_counts:
            print(
                f"  isoforms per gene: min={min(iso_counts)} median={sorted(iso_counts)[len(iso_counts)//2]} max={max(iso_counts)}"
            )
        if os.path.exists(UNRESOLVED_CSV):
            with open(UNRESOLVED_CSV) as fh:
                n = sum(1 for _ in fh) - 1
            print(f"unresolved_genes.csv: {n} entries")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
