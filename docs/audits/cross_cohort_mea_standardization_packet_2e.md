# Cross-Cohort MEA Standardization Packet 2E

Date: 2026-06-18

Packet 2E is intentionally skipped as a task addition.

The mechanism attribution producers are additive outputs of the existing MEA
commands:

- `python -m alz.cohorts.mukesh.mea --track both`
- `python -m alz.cohorts.tcells.mea --donor both`
- `python -m alz.cohorts.fivexfad.ingest --mea`

Adding separate Pixi tasks such as `human-mechanism` or `tcells-mechanism`
would duplicate the same producer surface and create ambiguity about whether
MEA should be rerun independently from mechanism attribution. Keep the existing
MEA tasks as the public command interface unless a future packet adds a
read-only mechanism regeneration mode.
