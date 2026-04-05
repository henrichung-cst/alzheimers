# SAP Document Map

Read this folder by analytical role, not by creation order. The package now has a six-document foundation layer and a separated archive layer.

## Front Door

Start here for live work:

1. [`foundation/analysis_charter.md`](./foundation/analysis_charter.md)
2. [`foundation/analysis_rationale.md`](./foundation/analysis_rationale.md)
3. [`foundation/statistical_constraints.md`](./foundation/statistical_constraints.md)
4. [`foundation/repo_retention_policy.md`](./foundation/repo_retention_policy.md)
5. [`foundation/repo_surface_index.md`](./foundation/repo_surface_index.md)
6. [`foundation/live_pipeline_contract.md`](./foundation/live_pipeline_contract.md)

## What Each Foundation Document Does

| File | Role |
|:---|:---|
| [`foundation/analysis_charter.md`](./foundation/analysis_charter.md) | Single source of truth for the live 72-sample analysis program |
| [`foundation/analysis_rationale.md`](./foundation/analysis_rationale.md) | Concise explanation of why the project pivoted and why the current path is defensible |
| [`foundation/statistical_constraints.md`](./foundation/statistical_constraints.md) | Governing identifiability and interpretation limits carried forward from the old SAP |
| [`foundation/repo_retention_policy.md`](./foundation/repo_retention_policy.md) | Main / supporting / archived assets, reproducibility expectations, and banned code paths |
| [`foundation/repo_surface_index.md`](./foundation/repo_surface_index.md) | Explicit file-level `main` / `supporting` / `archived` inventory within the existing repo layout |
| [`foundation/live_pipeline_contract.md`](./foundation/live_pipeline_contract.md) | Stage-by-stage live runner contract: prerequisites, outputs, failure modes, and ordered run sequence |

## Active Package

The live analysis story is now:

- total proteome integration,
- stoichiometry correction,
- mechanism stratification,
- Track A abundance-coupled attribution,
- Track B activity-driven attribution,
- final attribution-table assembly.

That whole story should be readable from `foundation/` without opening archive material.

## Integrations Layer

The `integrations/` folder is a sidecar reference layer for external dataset bundles, InCytr input mapping, and provenance checks. It is not a replacement for the live SAP foundation, but it is the place to look when the question is "what external bundle do we actually have and how does it map into the runtime workflow?"

### Integration references

- [`integrations/integrations-structure.md`](./integrations/integrations-structure.md)
- [`integrations/alzheimers-incytr-input-validation.md`](./integrations/alzheimers-incytr-input-validation.md)

These documents explain:

- how the upstream collaborator-owned archive is organized,
- which locations are upstream provenance versus current operational workspaces,
- which Alzheimer bundles are currently plausible InCytr inputs,
- what has been validated directly versus what remains only cohort-level or post-collapse inference.

### Machine-readable manifest

- [`integrations/5xfad-lucie-manifest.json`](./integrations/5xfad-lucie-manifest.json)

This is a structured inventory of the local Lucie 5xFAD `.sne` files and should be treated as supporting integration evidence rather than a narrative guidance document.

## Archive Layer

### Legacy design records

- [`archive/legacy_design/sap_24group_identifiability_record.md`](../archive/sap_docs/legacy_design/sap_24group_identifiability_record.md)
- [`archive/legacy_design/sap_factor_model_failure_record.md`](../archive/sap_docs/legacy_design/sap_factor_model_failure_record.md)
- [`archive/legacy_design/sap_rescue_record.md`](../archive/sap_docs/legacy_design/sap_rescue_record.md)

These preserve the design constraints and the failed rescue branches that justify the live program.

### Transitional notes

- [`archive/transitional_notes/sap_atlas.md`](../archive/sap_docs/transitional_notes/sap_atlas.md)
- [`archive/transitional_notes/sap_primary_path_summary.md`](../archive/sap_docs/transitional_notes/sap_primary_path_summary.md)
- [`archive/transitional_notes/sap_cleanup.md`](../archive/sap_docs/transitional_notes/sap_cleanup.md)
- [`archive/transitional_notes/sap_rewrite.md`](../archive/sap_docs/transitional_notes/sap_rewrite.md)

These are superseded intermediary summaries from the cleanup phase.

### Atlas working notes

- [`archive/atlas_working_notes/sap_atlas_part2.md`](../archive/sap_docs/atlas_working_notes/sap_atlas_part2.md)
- [`archive/atlas_working_notes/sap_atlas_part3.md`](../archive/sap_docs/atlas_working_notes/sap_atlas_part3.md)
- [`archive/atlas_working_notes/sap_atlas_part4.md`](../archive/sap_docs/atlas_working_notes/sap_atlas_part4.md)
- [`archive/atlas_working_notes/sap_atlas_part5.md`](../archive/sap_docs/atlas_working_notes/sap_atlas_part5.md)
- [`archive/atlas_working_notes/sap_atlas_series_distilled.md`](../archive/sap_docs/atlas_working_notes/sap_atlas_series_distilled.md)

These are provenance records for the atlas-series exploration and should not be treated as live guidance.

## Reading Rule

- If the goal is to understand what the team should do next, stay inside [`foundation/`](./foundation/analysis_charter.md).
- If the goal is to determine where an external Alzheimer/InCytr bundle lives or how upstream inputs map into the local runtime layout, open [`integrations/`](./integrations/integrations-structure.md).
- If the goal is to justify why a path was closed, open [`archive/legacy_design/`](../archive/sap_docs/legacy_design/sap_24group_identifiability_record.md).
- If the goal is historical context, open the transitional notes or atlas working notes.
