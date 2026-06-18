"""Shared MEA runner: the invariant shell around `_run_mea`.

Owns steps 2–5 of the Phase-3 design:
  2. Call the FROZEN `kinase_enrich._run_mea` with unit args.
  3. Write the 4 standard tables (mea_*, mea_global_shift, winsorized_sites,
     mea_substrate_sets) under the unit's naming convention.
  4. Record skip/empty (matrix absent / 0 motifs / empty mea_df) — structured.
  5. Stamp provenance (a per-run manifest written by write_provenance).

The adapter owns everything else: input loading, contrast construction, naming,
and the optional aggregate/provenance shape.

Public API
----------
MeaUnit : dataclass
    Descriptor for one _run_mea invocation.

MeaAdapter : Protocol
    Minimal interface each cohort adapter must satisfy.

MeaRunner : class
    run_unit(unit, adapter) -> RunResult | None
        Calls _run_mea, writes 4 tables, applies skip guard, invokes
        adapter.write_aggregates.  Returns None on skip.
    capture_fingerprint(unit, adapter) -> dict
        Input-equivalence mode: emits a deterministic fingerprint of the
        exact _run_mea kwargs for the unit, without calling _run_mea.
    run_all(adapter) -> list[RunResult]
        Drives all units from adapter.iter_units() through run_unit().
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# MeaUnit descriptor
# ---------------------------------------------------------------------------

@dataclass
class MeaUnit:
    """Descriptor for one _run_mea invocation.

    Attributes
    ----------
    track : str
        Phospho track key — ``"st"`` or ``"py"``.
    kind : str
        Preprocessing kind — ``"stoich"`` or ``"raw"``.
    lfc_key : str
        Key into each contrast's dict (``"stoich_lfc"`` or ``"raw_lfc"``).
    out_dir : str | Path
        Directory where the 4 standard tables are written.
    infix : str
        Filename infix (``""`` for stoich, ``"_raw"`` for raw).
    suffix : str
        Filename suffix (``""`` for ST, ``"_pY"`` for pY).
    motif_series : pd.Series | None
        Site motif strings aligned to the input matrix rows.  Set by
        load_inputs; None before loading.
    site_ids : np.ndarray | None
        Site identifier array aligned to the matrix rows.
    gene_symbols : np.ndarray | None
        Gene symbol array aligned to the matrix rows.
    results_by_contrast : dict[str, dict[str, np.ndarray]] | None
        ``{contrast_name: {lfc_key: np.ndarray}}`` — the per-contrast delta
        vectors.  Set by build_contrasts; None before construction.
    long_table_stem : str
        Filename stem for the MEA long table.  Cohort-specific: Mukesh uses
        ``"mea_perdonor"`` (default); T-cell uses ``"mea_timecourse"``.  The
        other three standard-table stems (``mea_global_shift``,
        ``winsorized_sites``, ``mea_substrate_sets``) are identical across
        cohorts and remain shared constants in the runner.
    meta : dict
        Free-form metadata for the adapter (e.g. ``ad_ids``, ``ctrl_ids``).
        The runner does not interpret this; it is passed to write_aggregates.
    """
    track: str
    kind: str
    lfc_key: str
    out_dir: str | Path
    infix: str
    suffix: str
    motif_series: pd.Series | None = field(default=None, repr=False)
    site_ids: np.ndarray | None = field(default=None, repr=False)
    gene_symbols: np.ndarray | None = field(default=None, repr=False)
    results_by_contrast: dict[str, dict[str, np.ndarray]] | None = field(
        default=None, repr=False
    )
    long_table_stem: str = "mea_perdonor"
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Human-readable label for logs and skip records."""
        return f"{self.track}/{self.kind}"


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    unit: MeaUnit
    mea_df: pd.DataFrame
    shift_df: pd.DataFrame
    wins_df: pd.DataFrame
    substrate_df: pd.DataFrame


# ---------------------------------------------------------------------------
# Skip record
# ---------------------------------------------------------------------------

@dataclass
class SkipRecord:
    unit_label: str
    reason: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Adapter Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class MeaAdapter(Protocol):
    """Minimal interface each cohort MEA adapter must satisfy.

    The runner calls these methods in order:
      1. iter_units()       — enumerate units to run
      2. load_inputs(unit)  — populate motif_series / site_ids / gene_symbols
      3. build_contrasts(unit) — populate results_by_contrast
      4. skip_check(unit)   — OPTIONAL early exit (default: never skip)
      5. run_unit(unit) via MeaRunner — calls _run_mea + writes 4 tables
      6. write_aggregates(result) — OPTIONAL post-unit hook
      7. write_provenance(records) — OPTIONAL manifest
    """

    def iter_units(self) -> Iterable[MeaUnit]:
        """Yield one MeaUnit per _run_mea call."""
        ...

    def load_inputs(self, unit: MeaUnit) -> MeaUnit | None:
        """Populate motif_series, site_ids, gene_symbols on the unit.

        Returns the unit (mutated or replaced).  Returning None signals a
        permanent skip (e.g. missing input file).
        """
        ...

    def build_contrasts(self, unit: MeaUnit) -> MeaUnit:
        """Populate results_by_contrast on the unit.

        Returns the unit (mutated or replaced).
        """
        ...

    def skip_check(self, unit: MeaUnit) -> tuple[bool, str | None]:
        """Return (skip, reason).  Default: (False, None)."""
        ...

    def write_aggregates(self, result: RunResult) -> None:
        """Optional post-unit aggregate step (NES/FDR/recurrence).

        Default no-op.  Mukesh/T-cell delegate to mea_outputs.
        """
        ...

    def write_provenance(self, skips: list[SkipRecord]) -> None:
        """Optional manifest / audit JSON.

        Default no-op.  Cohorts that wrote nothing before keep writing nothing.
        """
        ...


# ---------------------------------------------------------------------------
# Fingerprinting helpers
# ---------------------------------------------------------------------------

def _array_fingerprint(arr: np.ndarray | None, name: str) -> dict:
    """Return shape, dtype, NaN count, first/last 3, and a stable hash."""
    if arr is None:
        return {"name": name, "is_none": True}
    a = np.asarray(arr)
    flat = a.ravel()
    nan_count = int(np.sum(np.isnan(flat))) if np.issubdtype(a.dtype, np.floating) else 0
    first3 = flat[:3].tolist()
    last3 = flat[-3:].tolist()
    # Stable hash: for numeric arrays use tobytes (fast, exact); for object/str
    # arrays join as newline-separated strings (deterministic, encoding-safe).
    if np.issubdtype(a.dtype, np.floating):
        raw = a.astype(np.float64).tobytes()
    elif np.issubdtype(a.dtype, np.integer):
        raw = a.astype(np.int64).tobytes()
    else:
        raw = "\n".join(str(v) for v in flat).encode()
    h = hashlib.sha256(raw).hexdigest()[:16]
    return {
        "name": name,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "nan_count": nan_count,
        "first3": first3,
        "last3": last3,
        "sha256_16": h,
    }


def _series_fingerprint(s: pd.Series | None, name: str) -> dict:
    """Return length, null count, first/last 3, and a stable hash."""
    if s is None:
        return {"name": name, "is_none": True}
    vals = s.values
    length = len(vals)
    null_count = int(pd.isnull(vals).sum())
    first3 = list(vals[:3])
    last3 = list(vals[-3:])
    h = hashlib.sha256(
        "\n".join(str(v) for v in vals).encode()
    ).hexdigest()[:16]
    return {
        "name": name,
        "length": length,
        "null_count": null_count,
        "first3": first3,
        "last3": last3,
        "sha256_16": h,
    }


def _contrast_fingerprint(
    results_by_contrast: dict[str, dict[str, np.ndarray]], lfc_key: str
) -> dict:
    """Fingerprint each contrast's lfc array."""
    out = {}
    for contrast_name, res in sorted(results_by_contrast.items()):
        arr = res.get(lfc_key)
        out[contrast_name] = _array_fingerprint(arr, f"{contrast_name}/{lfc_key}")
    return out


# ---------------------------------------------------------------------------
# MeaRunner
# ---------------------------------------------------------------------------

class MeaRunner:
    """Shared shell: _run_mea + 4-table write + skip guard + provenance.

    Parameters
    ----------
    kinase_enrich_module : module
        The frozen ``alz.bulk_mea.enrich`` module.  Injected so the runner
        can be tested without importing the full kinase_library.
    """

    def __init__(self, kinase_enrich_module: Any) -> None:
        self._enrich = kinase_enrich_module
        self._skips: list[SkipRecord] = []

    # ------------------------------------------------------------------
    # Input-capture / fingerprint mode
    # ------------------------------------------------------------------

    def capture_fingerprint(self, unit: MeaUnit) -> dict:
        """Return a deterministic fingerprint of the _run_mea kwargs.

        Does NOT call _run_mea.  Used to prove input-equivalence between the
        canonical inline path and the runner path without re-running permutations.
        """
        if unit.motif_series is None or unit.results_by_contrast is None:
            raise ValueError(
                f"capture_fingerprint called on unit {unit.label} before "
                "load_inputs/build_contrasts populated it."
            )
        return {
            "unit_label": unit.label,
            "track": unit.track,
            "lfc_key": unit.lfc_key,
            "infix": unit.infix,
            "suffix": unit.suffix,
            "n_contrasts": len(unit.results_by_contrast),
            "contrast_keys": sorted(unit.results_by_contrast.keys()),
            "motif_series": _series_fingerprint(unit.motif_series, "motif_series"),
            "site_ids": _array_fingerprint(unit.site_ids, "site_ids"),
            "gene_symbols": _array_fingerprint(unit.gene_symbols, "gene_symbols"),
            "contrasts": _contrast_fingerprint(
                unit.results_by_contrast, unit.lfc_key
            ),
        }

    # ------------------------------------------------------------------
    # Single-unit execution
    # ------------------------------------------------------------------

    def _record_skip(self, unit: MeaUnit, reason: str) -> None:
        """Append a SkipRecord for ``unit`` and echo it (the standard skip path)."""
        rec = SkipRecord(unit_label=unit.label, reason=reason)
        self._skips.append(rec)
        print(f"  [{unit.label}] SKIP: {rec.reason}")

    def _call_mea_unit(
        self,
        unit: MeaUnit,
        adapter: MeaAdapter,
        *,
        skip_check_fn: Callable[[MeaUnit], tuple[bool, str | None]] | None = None,
    ) -> RunResult | None:
        """Skip-check + _run_mea call only — NO table writes, NO write_aggregates.

        Applies the same skip/empty guards as run_unit and records SkipRecords
        identically.  Returns None on skip; returns a RunResult on success (even
        if mea_df is empty — the empty-mea SkipRecord is recorded but the result
        is still returned so the caller can decide what to write).

        Use this primitive when the caller needs the raw _run_mea outputs before
        assembling its own output files (e.g. the 5xFAD bulk adapter that calls
        _run_mea twice and concatenates before writing).
        """
        # Caller-supplied skip check (public run_unit API) takes precedence over
        # adapter policy.
        if skip_check_fn is not None:
            do_skip, reason = skip_check_fn(unit)
            if do_skip:
                self._record_skip(unit, reason or "skip_check_fn")
                return None

        # Adapter skip check
        do_skip, reason = adapter.skip_check(unit)
        if do_skip:
            self._record_skip(unit, reason or "adapter skip_check")
            return None

        if unit.motif_series is None:
            self._record_skip(unit, "motif_series is None (missing input)")
            return None

        n_motifs = unit.motif_series.notna().sum()
        if n_motifs == 0:
            self._record_skip(unit, "0 non-null motifs in input")
            return None

        if not unit.results_by_contrast:
            self._record_skip(unit, "results_by_contrast is empty")
            return None

        # Call the FROZEN _run_mea
        mea_df, shift_df, wins_df, substrate_df = self._enrich._run_mea(
            motif_series=unit.motif_series,
            results_by_contrast=unit.results_by_contrast,
            lfc_key=unit.lfc_key,
            site_ids=unit.site_ids,
            gene_symbols=unit.gene_symbols,
            track=unit.track,
        )

        # Empty-mea guard (record but do not suppress — caller decides)
        if mea_df.empty:
            rec = SkipRecord(
                unit_label=unit.label,
                reason="empty mea_df from _run_mea",
                detail="n_contrasts=" + str(len(unit.results_by_contrast)),
            )
            self._skips.append(rec)
            print(f"  [{unit.label}] WARNING: empty MEA result")

        return RunResult(
            unit=unit,
            mea_df=mea_df,
            shift_df=shift_df,
            wins_df=wins_df,
            substrate_df=substrate_df,
        )

    def run_unit(
        self,
        unit: MeaUnit,
        adapter: MeaAdapter,
        *,
        skip_check_fn: Callable[[MeaUnit], tuple[bool, str | None]] | None = None,
    ) -> RunResult | None:
        """Execute one unit: skip-check → _run_mea → 4-table write → aggregates.

        Returns None if the unit is skipped.
        """
        result = self._call_mea_unit(
            unit,
            adapter,
            skip_check_fn=skip_check_fn,
        )
        if result is None:
            return None

        mea_df = result.mea_df
        shift_df = result.shift_df
        wins_df = result.wins_df
        substrate_df = result.substrate_df

        # Write 4 standard tables
        out_dir = str(unit.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        infix = unit.infix
        suffix = unit.suffix

        mea_path = os.path.join(out_dir, f"{unit.long_table_stem}{infix}{suffix}.csv")
        mea_df.to_csv(mea_path, index=False)
        print(f"  [{unit.label}] wrote {mea_path}  rows={len(mea_df)}")

        shift_df.to_csv(
            os.path.join(out_dir, f"mea_global_shift{infix}{suffix}.csv"), index=False
        )
        wins_df.to_csv(
            os.path.join(out_dir, f"winsorized_sites{infix}{suffix}.csv"), index=False
        )
        substrate_df.to_csv(
            os.path.join(out_dir, f"mea_substrate_sets{infix}{suffix}.csv"), index=False
        )

        if not mea_df.empty:
            adapter.write_aggregates(result)

        return result

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run_all(self, adapter: MeaAdapter) -> list[RunResult]:
        """Drive all units from adapter.iter_units() through run_unit.

        Returns list of successful RunResult objects (skips excluded).
        """
        results: list[RunResult] = []
        for unit in adapter.iter_units():
            loaded_unit = adapter.load_inputs(unit)
            if loaded_unit is None:
                self._record_skip(unit, "load_inputs returned None")
                continue
            unit = loaded_unit
            unit = adapter.build_contrasts(unit)
            result = self.run_unit(unit, adapter)
            if result is not None:
                results.append(result)
        adapter.write_provenance(self._skips)
        return results

    @property
    def skips(self) -> list[SkipRecord]:
        return list(self._skips)
