"""Kedro pipeline registry.

Reintroduced 2026-05-27. Starts EMPTY and grows one pipeline per pass in
Phases 2-5 (ingest -> bulk_mea -> decomposition_mea -> incytr_pair ->
reference/viewer). Registering pipelines that don't exist yet would break
`kedro run`; entries are added only as each `alz/pipelines/<name>/` is wired
over its existing helpers and verified against its parity gate.

See docs/foundation/live_pipeline_contract.md.
"""
from __future__ import annotations

from kedro.pipeline import Pipeline

from alz.pipelines import ingest


def register_pipelines() -> dict[str, Pipeline]:
    ingest_pipeline = ingest.create_pipeline()
    return {
        "ingest": ingest_pipeline,
        "__default__": ingest_pipeline,
    }
