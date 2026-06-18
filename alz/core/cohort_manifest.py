"""Packet 1A — Cohort manifest descriptors.

Per-cohort dataclasses that capture:
- which artifact kinds are expected,
- which tracks produce MEA vs OLS outputs,
- which files are absent_by_design,
- structural expectations for sample manifests and manifests.

No data is loaded here.  All paths are relative to PROJECT_ROOT.
Imports are stdlib-only (no pandas / pyarrow / numpy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MEA output kind: song/5xfad produce OLS, mukesh/tcells produce NES/FDR matrices
MEAOutputKind = Literal["ols", "nes_fdr_matrix"]


@dataclass
class TrackSpec:
    """One phospho track for a cohort."""
    track_name: str       # "st" or "py"
    output_suffix: str    # "" for ST, "_pY" for pY
    mea_capable: bool     # True unless gated out (donor2 pY = no_motif)
    mea_skip_reason: str | None = None   # e.g. "no_motif" if not mea_capable
    notes: str = ""


@dataclass
class CohortManifest:
    """Full manifest for one cohort."""
    cohort_name: str
    mea_output_kind: MEAOutputKind
    tracks: list[TrackSpec]
    has_ols_table: bool          # True for Song and 5xFAD
    has_nes_fdr_matrix: bool     # True for Mukesh and T-cell
    has_recurrence: bool         # True for Mukesh and T-cell
    sample_manifest_path: str | None   # relative path or None
    absent_by_design: list[str]        # rel paths expected absent
    protected_file_key: str            # key in baseline_inventory._build_protected_files()
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def st_suffix(self) -> str:
        return ""

    @property
    def py_suffix(self) -> str:
        return "_pY"

    def track_for(self, track_name: str) -> TrackSpec | None:
        for t in self.tracks:
            if t.track_name == track_name:
                return t
        return None

    def mea_capable_tracks(self) -> list[TrackSpec]:
        return [t for t in self.tracks if t.mea_capable]


# ---------------------------------------------------------------------------
# Song cohort
# ---------------------------------------------------------------------------

SONG = CohortManifest(
    cohort_name="song",
    mea_output_kind="ols",
    tracks=[
        TrackSpec("st", output_suffix="", mea_capable=True,
                  notes="Ser/Thr IMAC track; suffix '' (legacy)"),
        TrackSpec("py", output_suffix="_pY", mea_capable=True,
                  notes="Tyr pY track; suffix '_pY'"),
    ],
    has_ols_table=True,
    has_nes_fdr_matrix=False,
    has_recurrence=False,
    sample_manifest_path=None,   # input-side manifest only, not a protected output
    absent_by_design=[],
    protected_file_key="song",
    notes=(
        "Song 5XFAD-Tau mouse model (bulk). Produces OLS site-level tables "
        "and MEA long tables; no NES/FDR matrices. "
        "Decomposition subtree lives under outputs/reports/decomposition/levy_t5/."
    ),
    extra={
        "has_decomposition": True,
        "has_recovery": True,
    },
)


# ---------------------------------------------------------------------------
# Mukesh cohort (human AD)
# ---------------------------------------------------------------------------

MUKESH = CohortManifest(
    cohort_name="mukesh",
    mea_output_kind="nes_fdr_matrix",
    tracks=[
        TrackSpec("st", output_suffix="", mea_capable=True,
                  notes="Stoich track; suffix ''"),
        TrackSpec("py", output_suffix="_pY", mea_capable=True,
                  notes="pY track; suffix '_pY'"),
        # raw-phospho variants exist: suffix '_raw' and '_raw_pY'
    ],
    has_ols_table=False,
    has_nes_fdr_matrix=True,
    has_recurrence=True,
    sample_manifest_path=None,   # no protected sample manifest for Mukesh
    absent_by_design=[],
    protected_file_key="mukesh",
    notes=(
        "Mukesh human AD per-donor cohort. Produces NES/FDR matrices per donor "
        "and recurrence tables. No OLS tables. Also has raw-phospho track variants "
        "('_raw', '_raw_pY') for MEA long and NES/FDR matrices."
    ),
    extra={
        "perdonor_dir": "outputs/reports/kinase_attribution_human/perdonor",
        "has_celltype_specificity": True,
    },
)


# ---------------------------------------------------------------------------
# T-cells cohort
# ---------------------------------------------------------------------------

TCELLS = CohortManifest(
    cohort_name="tcells",
    mea_output_kind="nes_fdr_matrix",
    tracks=[
        TrackSpec("st", output_suffix="", mea_capable=True,
                  notes="Stoich track donor1 only; donor2 ST matrix absent"),
        TrackSpec("py", output_suffix="_pY", mea_capable=True,
                  notes=(
                      "pY track; donor1 MEA ran; donor2 pY matrices present "
                      "but MEA skipped (reason: no_motif)"
                  )),
    ],
    has_ols_table=False,
    has_nes_fdr_matrix=True,
    has_recurrence=True,
    sample_manifest_path=None,   # no protected sample manifest for T-cells
    absent_by_design=[
        "outputs/reports/kinase_attribution_tcells/donor2/mea/mea_timecourse.csv",
    ],
    protected_file_key="tcells",
    notes=(
        "T-cell exhaustion cohort: donor1 (full MEA), donor2 (pY matrices only, "
        "no MEA — ST matrix absent, pY MEA skipped due to no_motif). "
        "mea_manifest.json gates which tracks ran per donor."
    ),
    extra={
        "donor1_mea_dir": "outputs/reports/kinase_attribution_tcells/donor1/mea",
        "donor2_dir": "outputs/reports/kinase_attribution_tcells/donor2",
    },
)


# ---------------------------------------------------------------------------
# 5xFAD cohort
# ---------------------------------------------------------------------------

FIVEXFAD = CohortManifest(
    cohort_name="fivexfad",
    mea_output_kind="ols",
    tracks=[
        TrackSpec("st", output_suffix="", mea_capable=True,
                  notes="ST track; suffix '' (but filenames use '_st_')"),
        TrackSpec("py", output_suffix="_pY", mea_capable=True,
                  notes="pY track; filenames use '_py_'"),
    ],
    has_ols_table=True,
    has_nes_fdr_matrix=False,
    has_recurrence=False,
    sample_manifest_path="outputs/reports/kinase_attribution_5xfad/sample_manifest.csv",
    absent_by_design=[],
    protected_file_key="fivexfad",
    notes=(
        "5xFAD mouse model (cortex + hippocampus × ST + pY). Produces OLS tables "
        "and MEA long tables; no NES/FDR matrices. "
        "File names use '<region>_<track>_' prefix (e.g. cortex_st_). "
        "Celltype MEA lives under celltype_mea/ subdir."
    ),
    extra={
        "regions": ["cortex", "hippocampus"],
        "has_celltype_mea": True,
        "has_sample_manifest": True,
    },
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_COHORTS: dict[str, CohortManifest] = {
    "song": SONG,
    "mukesh": MUKESH,
    "tcells": TCELLS,
    "fivexfad": FIVEXFAD,
}


def get_cohort(name: str) -> CohortManifest:
    if name not in ALL_COHORTS:
        raise ValueError(f"Unknown cohort {name!r}. Valid: {sorted(ALL_COHORTS)}")
    return ALL_COHORTS[name]
