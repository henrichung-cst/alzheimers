# alz/ingest/ — Layer 1: bespoke per-dataset ingest modules.
#
# Each module reads a single collaborator's raw drop (Song proteomics,
# Mukesh human cohort, Lucie 5xFAD) and emits canonical artifacts that
# the shared analysis pipelines consume. Cohort identity is kept inside
# the ingest module; the shared pipelines never see raw vocabulary.
