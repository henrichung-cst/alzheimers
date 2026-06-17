#!/usr/bin/env python3
"""Report canonical and noncanonical AD Incytr input-root references.

The production AD input contract is intentionally narrow:

    data/derived/incytr_inputs

Other sce4/v2/source/scratch input roots are useful for forensic reproduction
tests, but they should not silently become production defaults. This script
scans repo scripts/docs/configs and reports where those roots appear.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CANONICAL_ROOT = "data/derived/incytr_inputs"

NONCANONICAL_INPUT_ROOTS = (
    "data/incytr_frozen/v2_46clusters/incytr input",
    "data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721",
    "data/derived/incytr_inputs_source_ps_diag",
    "data/derived/_sce4_input_scratch",
)

DEFAULT_SCAN_ROOTS = (
    "CLAUDE.md",
    "pixi.toml",
    "README.md",
    "alz/incytr_pair",
    "alz/runners",
    "docs/plans",
    "docs/integrations",
    "bench",
)

TEXT_EXTENSIONS = {
    ".R",
    ".py",
    ".sh",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}

SKIP_DIRS = {
    ".git",
    ".pixi",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

DIAGNOSTIC_FILE_PATTERNS = (
    "docs/plans/repo_cleanup_targets_2026-06-17.md",
    "docs/integrations/*",
    "bench/*",
    "alz/incytr_pair/README.md",
    "alz/incytr_pair/audit_*.R",
    "alz/incytr_pair/audit_*.py",
    "alz/incytr_pair/forensic_*.R",
    "alz/incytr_pair/verify_*.R",
    "alz/incytr_pair/verify_*.py",
    "alz/incytr_pair/verify_*.sh",
)

PROVENANCE_BUILDER_PATTERNS = (
    "alz/incytr_pair/build_pair_inputs.sh",
    "alz/incytr_pair/build_pair_seurat.R",
    "alz/incytr_pair/export_decomposition_for_pair.py",
    "alz/incytr_pair/extract_sce4_geneuse.R",
)


@dataclass(frozen=True)
class Finding:
    status: str
    path: str
    line: int
    matched_root: str
    evidence: str


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _is_text_candidate(path: Path) -> bool:
    return path.suffix in TEXT_EXTENSIONS or path.name in {"CLAUDE.md", "README"}


def _iter_files(repo: Path, roots: Iterable[str], include_outputs: bool) -> Iterable[Path]:
    for root_name in roots:
        root = repo / root_name
        if not root.exists():
            continue
        if root.is_file():
            if _is_text_candidate(root):
                yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = _rel(Path(dirpath), repo)
            dirnames[:] = [
                d
                for d in dirnames
                if d not in SKIP_DIRS
                and (include_outputs or d != "outputs")
                and (include_outputs or not rel_dir.startswith("outputs"))
            ]
            for filename in filenames:
                path = Path(dirpath) / filename
                if _is_text_candidate(path):
                    yield path


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _classify(rel_path: str, root: str, line_text: str) -> str:
    if root == CANONICAL_ROOT:
        return "OK_CANONICAL"
    if _matches_any(rel_path, PROVENANCE_BUILDER_PATTERNS):
        return "OK_PROVENANCE_BUILDER"
    if _matches_any(rel_path, DIAGNOSTIC_FILE_PATTERNS):
        return "DOCUMENTED_OR_DIAGNOSTIC_REFERENCE"
    if "diagnostic" in line_text.lower() or "forensic" in line_text.lower():
        return "DOCUMENTED_OR_DIAGNOSTIC_REFERENCE"
    return "NONCANONICAL_REFERENCE"


def audit(repo: Path, roots: Iterable[str], include_outputs: bool) -> list[Finding]:
    needles = NONCANONICAL_INPUT_ROOTS + (CANONICAL_ROOT,)
    findings: list[Finding] = []
    for path in sorted(set(_iter_files(repo, roots, include_outputs))):
        rel_path = _rel(path, repo)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            normalized = line.replace("\\ ", " ")
            for root in needles:
                if root not in normalized:
                    continue
                if root == CANONICAL_ROOT and any(
                    noncanonical in normalized for noncanonical in NONCANONICAL_INPUT_ROOTS
                ):
                    continue
                findings.append(
                    Finding(
                        status=_classify(rel_path, root, normalized),
                        path=rel_path,
                        line=line_no,
                        matched_root=root,
                        evidence=normalized.strip()[:220],
                    )
                )
    return findings


def print_table(findings: list[Finding]) -> None:
    if not findings:
        print("No AD Incytr input-root references found.")
        return

    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.status] = counts.get(finding.status, 0) + 1

    print("Incytr input provenance audit")
    print(f"canonical root: {CANONICAL_ROOT}")
    print("summary:")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")
    print()
    print("status\tpath:line\tmatched_root\tevidence")
    for finding in findings:
        print(
            f"{finding.status}\t{finding.path}:{finding.line}\t"
            f"{finding.matched_root}\t{finding.evidence}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=None,
        help="repo root; defaults to git top-level or current working directory",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        dest="scan_roots",
        help=(
            "relative path to scan; may be repeated. Defaults to "
            + ", ".join(DEFAULT_SCAN_ROOTS)
        ),
    )
    parser.add_argument(
        "--include-outputs",
        action="store_true",
        help="also scan output/log directories if they are under a selected scan root",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON findings instead of a tabular report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero if any suspicious noncanonical reference is found",
    )
    args = parser.parse_args()

    if args.root:
        repo = Path(args.root).resolve()
    else:
        try:
            git_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            git_root = "."
        repo = Path(git_root).resolve()

    scan_roots = tuple(args.scan_roots or DEFAULT_SCAN_ROOTS)
    findings = audit(repo, scan_roots, args.include_outputs)
    if args.json:
        print(json.dumps([asdict(finding) for finding in findings], indent=2))
    else:
        print_table(findings)

    has_suspicious = any(f.status == "NONCANONICAL_REFERENCE" for f in findings)
    if args.strict and has_suspicious:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
