"""Kedro project settings.

Reintroduced 2026-05-27 (reverses the 2026-05-21 strip, commit 0c2a997) so the
pipelines can be orchestrated on cellsignal/bioplat via Argo. See
docs/plans/kedro_argo_reintroduction_2026-05-26.md.

Defaults are intentionally minimal. The config loader already honours the
existing conf/{base,local,full_cohort,human_nbb}/ overlays selected by
KEDRO_ENV (on-cluster: `kedro run --env={{workflow.namespace}}`).
"""
from __future__ import annotations

from kedro.config import OmegaConfigLoader

CONFIG_LOADER_CLASS = OmegaConfigLoader
