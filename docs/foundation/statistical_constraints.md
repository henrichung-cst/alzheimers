# Statistical Constraints

This document preserves the irreducible statistical facts that define the boundary conditions of the project. It is not a methods plan and it is not a recommendation to reopen the retired estimators.

## 1. Design Facts

The original deconvolution problem is anchored to a complete `2 x 3 x 4` factorial design:

- 2 sexes,
- 3 timepoints,
- 4 conditions,
- 24 bulk phosphoproteomics groups total,
- zero biological replication within group.

Each group is paired to a composition vector from matched snRNA-seq, but the composition matrix is the limiting object for cell-type-specific condition recovery.

## 2. Identifiability Boundary

The pooled composition matrix has effective rank near 2. The retained singular values are:

- `1.37`
- `0.14`
- `0.04`
- `0.02`
- `0.02`

Only the first two directions sit meaningfully above the noise floor. Inverting the weak directions amplifies noise by roughly:

- `1 / 1.37 ~= 0.7x` on the best direction,
- `1 / 0.14 ~= 7x`,
- `1 / 0.04 ~= 25x`,
- `1 / 0.02 = 50x` on the weakest directions.

That is the core identifiability boundary: the design does not contain enough independent composition variation to support unique cell-type-specific condition attribution at the per-site level.

## 3. Why Per-Site Recovery Fails

With 6 samples per condition and bulk residual MAD near 100 intensity units, the standard error of a bulk condition mean is about 41. After projection into weak composition directions, that error inflates into the hundreds or thousands, exceeding the effect sizes used in synthetic validation. The resulting per-site cell-type signal-to-noise ratio stays far below 1 in the poorly identified directions.

Practical implication: the data can support bulk condition effects, but not reliable decomposition of those effects across cell types.

## 4. Evidence That The Failure Is Structural

The non-identifiability conclusion is not based on one failed implementation. An 8-phase synthetic validation campaign, four rescue strategies, a joint factor model, two-compartment collapse, and transcript-only rescue all returned near-zero recovery. See [`analysis_rationale.md`](./analysis_rationale.md) for the full pivot logic and [`deconvolution_infeasibility.md`](../../archive/deconvolution/docs/deconvolution_infeasibility.md) for the synthetic validation.

The factor-model failure matters because it shows that parameter reduction and kinase-substrate structure did not solve the real problem. The bottleneck is the composition geometry, not the size of the parameterization.

## 5. Why Diagnostics Do Not Reopen The Old Path

The Hurdle-Tweedie deconvolution stack passed internal diagnostics, but passing diagnostics does not rehabilitate the model:

- it means the old model was built coherently,
- it argues against a trivial coding or fitting bug,
- it does not restore biological attribution capability.

This prevents future work from reopening the old branch under the assumption that better tuning alone would fix it.

## 6. Supported And Unsupported Statistical Claims

Supported:

- bulk phospho condition effects,
- total-proteome-enabled stoichiometry analysis,
- mechanism classification into abundance-driven, both, and activity-driven classes,
- direction concordance for abundance-coupled classes,
- detection-gated attributions for activity-driven kinases,
- the merged final attribution table within the current mechanism-stratified framework.

Not supported by the retired 24-group design:

- unique per-site cell-type condition estimates from direct deconvolution,
- claims that factor-model or compartment-collapse rescues solved the attribution problem,
- presenting old deconvolution outputs as confirmed biological localization.

## 7. Downstream Interpretation Rules

Use these constraints as guardrails:

1. treat old deconvolution outputs as provenance only,
2. treat bulk-level findings as the strongest directly supported signal from the 24-group branch,
3. keep mechanism-stratified attribution separate from retired deconvolution claims,
4. do not describe archived estimators in enough detail that they read like recommended live methods.

## Bottom Line

The project’s governing statistical fact is that cell-type-specific condition attribution is not uniquely determined by the original 24-group composition design. The live program works by changing the target of inference, not by rescuing that retired inverse problem.
