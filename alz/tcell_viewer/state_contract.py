"""Canonical evidence-backed state roster for T-cell viewer consumers."""

from __future__ import annotations

import json
import os

from alz.tcell_viewer.paths import TCELLS_INCYTR_INPUTS_DIR


def state_audit_path(donor: str) -> str:
    return os.path.join(
        TCELLS_INCYTR_INPUTS_DIR, donor, "scrna", "state_audit.json"
    )


def load_state_audit(donor: str) -> dict:
    """Load and validate the generated per-cell state audit for one donor."""
    path = state_audit_path(donor)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{donor} state audit is missing: {path}")
    with open(path, encoding="utf-8") as handle:
        audit = json.load(handle)
    if audit.get("donor") != donor:
        raise ValueError(
            f"state audit donor mismatch: expected {donor!r}, "
            f"found {audit.get('donor')!r}"
        )
    totals = audit.get("state_totals")
    if not isinstance(totals, dict) or not totals:
        raise ValueError(f"{donor} state audit has no state_totals")
    bad = {
        str(state): value
        for state, value in totals.items()
        if (
            not str(state)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        )
    }
    if bad:
        raise ValueError(f"{donor} state audit has invalid totals: {bad}")
    by_day = audit.get("state_by_day")
    if not isinstance(by_day, list):
        raise ValueError(f"{donor} state audit has no state_by_day records")
    observed = {str(record.get("state")) for record in by_day}
    if observed != set(totals):
        raise ValueError(
            f"{donor} state audit roster mismatch: totals={sorted(totals)} "
            f"by_day={sorted(observed)}"
        )
    return audit


def load_donor_states(donor: str) -> list[str]:
    """Return the sorted canonical per-cell state roster for ``donor``."""
    return sorted(load_state_audit(donor)["state_totals"])


def validate_pathway_states(
    donor: str,
    senders: list[str] | set[str],
    receivers: list[str] | set[str],
) -> list[str]:
    """Reject pathway endpoint names outside the donor's canonical roster."""
    roster = set(load_donor_states(donor))
    endpoints = {str(value) for value in [*senders, *receivers]}
    unknown = sorted(endpoints - roster)
    if unknown:
        raise ValueError(
            f"{donor} pathway output contains states outside the per-cell "
            f"roster: {unknown}"
        )
    return sorted(roster)
