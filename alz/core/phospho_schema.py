"""Packet 1A — Phospho-proteomics schema descriptors.

Lightweight dataclasses describing the column contracts for the four artifact
kinds produced by the bulk-MEA pipeline.  These are *descriptive* schema
objects — they do not enforce at write time; enforcement lives in validation.py.

No data is loaded here.  Imports are stdlib-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Track suffix convention (from alz/shared/config.py PHOSPHO_TRACKS)
# ---------------------------------------------------------------------------

TRACK_SUFFIX_ST: str = ""       # Ser/Thr IMAC track — legacy, no suffix
TRACK_SUFFIX_PY: str = "_pY"   # Tyr track suffix

KNOWN_TRACK_SUFFIXES: tuple[str, ...] = (TRACK_SUFFIX_ST, TRACK_SUFFIX_PY)

# The Mukesh and T-cell pipelines also emit a "_raw" infix (raw-phospho track).
# The convention is: base = suffix (e.g. ""), raw variant = "_raw" + suffix.
KNOWN_TRACK_INFIXES: tuple[str, ...] = ("", "_raw")  # "" = stoich/normalised

# MEA skip reasons (recorded in mea_manifest.json)
MEA_SKIP_REASON_NO_MOTIF: str = "no_motif"
MEA_SKIP_REASON_MATRIX_ABSENT: str = "matrix_absent"
KNOWN_MEA_SKIP_REASONS: tuple[str, ...] = (
    MEA_SKIP_REASON_NO_MOTIF,
    MEA_SKIP_REASON_MATRIX_ABSENT,
)

# MEA FDR threshold (from alz/shared/config.py)
MEA_FDR_THRESH: float = 0.25

# Sign convention: + = up in disease
SIGN_CONVENTION: Literal["+"] = "+"


# ---------------------------------------------------------------------------
# Sample manifest concept
# ---------------------------------------------------------------------------

@dataclass
class SampleManifestSchema:
    """Schema concept for a cohort sample manifest.

    The actual on-disk manifests differ per cohort, so this only records the
    *minimum required* columns and what they mean.
    """
    name: str
    required_columns: list[str]
    notes: str = ""

    def check_columns(self, actual: list[str]) -> list[str]:
        """Return list of missing required columns."""
        return [c for c in self.required_columns if c not in actual]


FIVEXFAD_SAMPLE_MANIFEST = SampleManifestSchema(
    name="5xFAD sample_manifest.csv",
    required_columns=["tissue", "assay", "age_months", "genotype", "biological_sample_id"],
    notes="5xFAD tissue × genotype × age design; analysis_action column gates sample inclusion.",
)

SONG_SNRNA_SAMPLE_MANIFEST = SampleManifestSchema(
    name="Song snRNA manifest (data/datasets/song/transcriptomics/snrna_sample_manifest.csv)",
    required_columns=[],  # input-side; validated separately
    notes="Input manifest, not a protected output. Presence check only.",
)


# ---------------------------------------------------------------------------
# Site matrix concept (wide normalized matrices)
# ---------------------------------------------------------------------------

@dataclass
class SiteMatrixSchema:
    """Schema concept for a wide site-level normalized matrix.

    Rows = phospho sites (site_id or similar key), columns = samples after
    a small fixed metadata prefix.  Sample columns are validated by
    cross-referencing the sample manifest.
    """
    name: str
    required_prefix_columns: list[str]
    notes: str = ""

    def check_prefix_columns(self, actual: list[str]) -> list[str]:
        """Return list of missing required prefix columns."""
        return [c for c in self.required_prefix_columns if c not in actual]

    def sample_columns(self, actual: list[str]) -> list[str]:
        """Return sample columns (everything after the prefix)."""
        prefix_set = set(self.required_prefix_columns)
        return [c for c in actual if c not in prefix_set]


# Song stoichiometry matrix: site_id, gene_symbol, motif, matched_protein, <samples>
SONG_STOICH_MATRIX = SiteMatrixSchema(
    name="song_stoichiometry_matrix",
    required_prefix_columns=["site_id", "gene_symbol", "motif", "matched_protein"],
    notes="Song Ser/Thr stoichiometry-normalized matrix; samples are plex_channel identifiers.",
)

# Song raw-phospho normalized: site_id, gene_symbol, motif, <samples>
SONG_RAW_NORM_MATRIX = SiteMatrixSchema(
    name="song_raw_phospho_normalized",
    required_prefix_columns=["site_id", "gene_symbol", "motif"],
    notes="Song Ser/Thr raw-phospho matrix; samples are plex_channel identifiers.",
)

# Mukesh/T-cell normalized matrix (both stoich and raw share the same prefix)
HUMAN_NORM_MATRIX = SiteMatrixSchema(
    name="human_norm_matrix",
    required_prefix_columns=["site_id", "protein_id", "gene_symbol", "site_position", "motif"],
    notes="Mukesh/T-cell normalized matrix; sample columns are donor-timepoint identifiers.",
)

# 5xFAD stoichiometry matrix (has residue_type and matched_protein extra)
FIVEXFAD_STOICH_MATRIX = SiteMatrixSchema(
    name="fivexfad_stoichiometry_matrix",
    required_prefix_columns=["site_id", "gene_symbol", "motif", "site_position",
                              "residue_type", "matched_protein"],
    notes="5xFAD stoichiometry-normalized matrix.",
)

# 5xFAD raw-phospho normalized matrix (has residue_type but NOT matched_protein)
FIVEXFAD_RAW_NORM_MATRIX = SiteMatrixSchema(
    name="fivexfad_raw_phospho_normalized",
    required_prefix_columns=["site_id", "gene_symbol", "motif", "site_position", "residue_type"],
    notes="5xFAD raw-phospho normalized matrix.",
)

# 5xFAD matched total protein matrix
FIVEXFAD_MATCHED_PROTEIN_MATRIX = SiteMatrixSchema(
    name="fivexfad_matched_total_protein",
    required_prefix_columns=["site_id", "gene_symbol", "matched_protein"],
    notes="5xFAD matched total protein matrix.",
)


# ---------------------------------------------------------------------------
# Contrast manifest concept
# ---------------------------------------------------------------------------

@dataclass
class ContrastManifestSchema:
    """Schema concept for a contrast QC / manifest table."""
    name: str
    required_columns: list[str]
    notes: str = ""

    def check_columns(self, actual: list[str]) -> list[str]:
        return [c for c in self.required_columns if c not in actual]


FIVEXFAD_CONTRAST_QC = ContrastManifestSchema(
    name="fivexfad_contrast_qc",
    required_columns=["tissue", "track", "contrast", "contrast_status"],
    notes="5xFAD contrast-level QC; contrast_status gates MEA.",
)

TCELL_MEA_MANIFEST = ContrastManifestSchema(
    name="tcell_mea_manifest",
    required_columns=["donor", "mea_ran", "mea_skipped"],
    notes=(
        "T-cell MEA skip/run manifest; mea_skipped entries carry 'reason' = "
        "'no_motif' or 'matrix_absent'. donor2 pY MEA skip reason must be "
        "'no_motif' (matrix present, motif absent) — not 'matrix_absent'."
    ),
)


# ---------------------------------------------------------------------------
# MEA output concept
# ---------------------------------------------------------------------------

@dataclass
class MEALongTableSchema:
    """Schema concept for a MEA long table (one row per kinase × contrast)."""
    name: str
    required_columns: list[str]
    numeric_columns: list[str] = field(default_factory=list)
    notes: str = ""

    def check_columns(self, actual: list[str]) -> list[str]:
        return [c for c in self.required_columns if c not in actual]


# Shared MEA long table schema (Song, Mukesh, T-cell, Song-decomp)
MEA_LONG_TABLE = MEALongTableSchema(
    name="mea_long_table",
    required_columns=["kinase", "ES", "NES", "p-value", "FDR",
                      "Subs fraction", "Leading substrates",
                      "contrast", "residue_type", "track"],
    numeric_columns=["ES", "NES", "p-value", "FDR"],
    notes=(
        "Standard GSEApy prerank output, augmented with contrast/residue_type/track. "
        "Sign convention: NES > 0 = up in disease. "
        "'Subs fraction' is a fractional string ('136/715'), not a float column."
    ),
)

# 5xFAD MEA long table has extra tissue/analysis_track prefix columns
FIVEXFAD_MEA_LONG_TABLE = MEALongTableSchema(
    name="fivexfad_mea_long_table",
    required_columns=["tissue", "analysis_track", "kinase", "ES", "NES",
                      "p-value", "FDR", "Subs fraction", "Leading substrates",
                      "contrast", "residue_type", "track"],
    numeric_columns=["ES", "NES", "p-value", "FDR", "Subs fraction"],
    notes="5xFAD variant adds 'tissue' and 'analysis_track' prefix columns.",
)

# 5xFAD celltype MEA parquet adds cell_type and tissue suffix columns
FIVEXFAD_CELLTYPE_MEA = MEALongTableSchema(
    name="fivexfad_celltype_mea_parquet",
    required_columns=["kinase", "ES", "NES", "p-value", "FDR",
                      "Subs fraction", "Leading substrates",
                      "contrast", "residue_type", "track", "cell_type", "tissue"],
    numeric_columns=["ES", "NES", "p-value", "FDR", "Subs fraction"],
    notes="5xFAD celltype MEA parquet; extends base schema with cell_type + tissue.",
)

# Song decomp MEA parquet adds cluster prefix
SONG_DECOMP_MEA_PARQUET = MEALongTableSchema(
    name="song_decomp_mea_per_cluster_parquet",
    required_columns=["cluster", "kinase", "ES", "NES", "p-value", "FDR",
                      "Subs fraction", "Leading substrates",
                      "contrast", "residue_type", "track"],
    numeric_columns=["ES", "NES", "p-value", "FDR", "Subs fraction"],
    notes="Song decomposition MEA per cluster; adds 'cluster' prefix.",
)


@dataclass
class MEANESFDRMatrixSchema:
    """Schema concept for a NES or FDR wide matrix (kinases × donors/timepoints).

    Applies only to Mukesh and T-cell cohorts.
    """
    name: str
    required_prefix_columns: list[str]   # always ["kinase"]
    notes: str = ""

    def check_prefix_columns(self, actual: list[str]) -> list[str]:
        return [c for c in self.required_prefix_columns if c not in actual]

    def donor_columns(self, actual: list[str]) -> list[str]:
        prefix_set = set(self.required_prefix_columns)
        return [c for c in actual if c not in prefix_set]


NES_FDR_MATRIX = MEANESFDRMatrixSchema(
    name="nes_fdr_matrix",
    required_prefix_columns=["kinase"],
    notes=(
        "Wide NES or FDR matrix; kinases × donors (Mukesh) or × timepoints (T-cell). "
        "Song and 5xFAD do NOT produce NES/FDR matrices — they produce OLS tables."
    ),
)


@dataclass
class OLSTableSchema:
    """Schema concept for a site-level OLS/effect-size table.

    Applies to Song (flat) and 5xFAD (flat per region/track).
    Mukesh and T-cell do NOT produce OLS tables.
    """
    name: str
    required_columns: list[str]
    notes: str = ""

    def check_columns(self, actual: list[str]) -> list[str]:
        return [c for c in self.required_columns if c not in actual]


SONG_OLS_TABLE = OLSTableSchema(
    name="song_site_level_ols",
    required_columns=["site_id", "gene_symbol", "matched_protein", "n_obs_stoich"],
    notes=(
        "Song OLS table; contrast columns are wide (stoich_lfc_<contrast>, "
        "raw_lfc_<contrast>, etc.). No NES/FDR columns."
    ),
)

FIVEXFAD_OLS_TABLE = OLSTableSchema(
    name="fivexfad_site_level_ols",
    required_columns=["site_id", "gene_symbol", "matched_protein"],
    notes=(
        "5xFAD OLS table; contrast columns are wide (stoich_lfc_<contrast>, etc.). "
        "n_obs_raw also present (both n_obs_stoich and n_obs_raw). No NES/FDR."
    ),
)

SONG_DECOMP_OLS_PARQUET = OLSTableSchema(
    name="song_decomp_site_level_ols_per_cluster_parquet",
    required_columns=["cluster", "contrast", "site_id", "gene_symbol", "lfc", "pval", "fdr"],
    notes="Song decomposition OLS per cluster (long format, not wide).",
)

FIVEXFAD_CELLTYPE_OLS_PARQUET = OLSTableSchema(
    name="fivexfad_celltype_site_level_ols_parquet",
    required_columns=["tissue", "track", "cell_type", "contrast",
                       "site_id", "gene_symbol", "lfc", "pval", "fdr"],
    notes="5xFAD celltype OLS parquet; long format.",
)


@dataclass
class RecurrenceTableSchema:
    """Schema concept for kinase recurrence tables (Mukesh, T-cell)."""
    name: str
    required_columns: list[str]
    notes: str = ""

    def check_columns(self, actual: list[str]) -> list[str]:
        return [c for c in self.required_columns if c not in actual]


MUKESH_RECURRENCE = RecurrenceTableSchema(
    name="mukesh_recurrence",
    required_columns=["kinase", "n_donors_sig", "n_donors_up", "n_donors_down",
                       "n_donors_tested", "median_nes"],
    notes="Mukesh recurrence table; one row per kinase.",
)

TCELL_RECURRENCE = RecurrenceTableSchema(
    name="tcell_recurrence",
    required_columns=["kinase", "n_timepoints_sig", "n_timepoints_up", "n_timepoints_down",
                       "n_timepoints_tested", "median_nes"],
    notes="T-cell recurrence table; one row per kinase.",
)
