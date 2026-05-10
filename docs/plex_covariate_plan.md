---
status: deferred (diagnostic complete 2026-05-09; activation pending)
scope: males-only OLS only — full_cohort cannot be modified (see §Constraints)
---

# Plex covariate addition — deferred plan

## Context

The post-IRS PCA (`outputs/reports/kinase_attribution/pca_plots/tp_norm_by_plex.png`)
shows residual plex structure on plexes 3 and 5 in the males-only run. IRS
knocks the dominant TMT batch effect down (raw PCA shows six tight clumps,
one per plex; normalized PCA mostly mixes), but doesn't fully eliminate it.

The factorial OLS in `alz/kinase_enrich.py:_build_design_matrix` does **not**
currently include a plex covariate. Residual plex variance therefore lands
in the residual term and inflates contrast SE.

## Design constraint that scopes the fix

Plex × Sex are perfectly confounded in this study:

```
Plex × Sex (sample_mapping.csv):
sex    F   M
plex 1     0  12
plex 2    12   0
plex 3     0  12
plex 4    12   0
plex 5     0  12
plex 6    12   0
```

Genotype and timepoint are perfectly balanced across plexes (3 of each
genotype, 4 of each timepoint per plex). This means:

- **Males-only run** sees plexes 1, 3, 5 only — plex is independent of all
  modeled biology, so adding plex dummies is safe and well-conditioned.
- **Full-cohort run** sees all 6 plexes — adding plex dummies would absorb
  all of `female`, since plex *is* sex. The change must be males-only.

## Diagnostic results (2026-05-09)

Diagnostic ran the OLS both ways on `stoichiometry_matrix.csv` (males-only,
33 samples after outlier exclusion: plex 1 = 10, plex 3 = 12, plex 5 = 11).
Compared site-level residuals and ran MEA on each design.

### Site-level residual variance

| Metric | Baseline (10 params) | +Plex (12 params) |
|---|---:|---:|
| Residual DOF | 23 | 21 |
| Median residual SD | 0.2952 | 0.2635 (**−10.7%**) |
| Per-site SSE reduction by adding plex | — | median **+19%**, positive at 5,790/5,790 sites |
| Plex-3 β magnitude | — | median \|β\| = 0.14, 95th pct \|β\| = 0.52 |
| Plex-5 β magnitude | — | median \|β\| = 0.15, 95th pct \|β\| = 0.52 |

Plex absorbs real per-site variance everywhere. The 2-DOF cost is mild;
t-critical at α=0.05 two-sided is 2.069 (df=23) vs 2.080 (df=21).

### Per-contrast significant kinases (FDR < 0.25, st track)

| Contrast | Baseline | +Plex | Δ | NES corr | Jaccard |
|---|---:|---:|---:|---:|---:|
| App_2mo  | 34   | 27   | **−7**  | 0.93 | 0.56 |
| App_4mo  | 179  | 194  | **+15** | 0.96 | 0.89 |
| App_6mo  | 169  | 169  | 0       | 0.98 | 0.93 |
| Tau_2mo  | 70   | 75   | +5      | 0.97 | 0.93 |
| Tau_4mo  | 183  | 182  | −1      | 0.94 | 0.82 |
| Tau_6mo  | 153  | 153  | 0       | 0.99 | 0.89 |
| ApTt_2mo | 32   | 32   | 0       | 1.00 | 1.00 |
| ApTt_4mo | 192  | 198  | +6      | 0.96 | 0.91 |
| ApTt_6mo | 200  | 208  | +8      | 0.97 | 0.90 |
| **Total**| **1212** | **1238** | **+26** | median 0.97 | median 0.90 |

### Interpretation

- **Net power gain is small but positive** (+26 kinase·contrasts, ~+2%).
  Tighter SE from the residual SD reduction outweighs the 2-DOF cost.
- **Top hits are stable.** Median NES rank correlation is 0.97; ApTt_2mo
  is a perfect 1.00.
- **Borderline hits churn ~10%** (median Jaccard 0.90). The churn is
  symmetric (some lost, some gained) — predictable for kinases sitting
  near the FDR boundary when SE shifts.
- **Worst churn: App_2mo (Jaccard 0.56).** Smallest hit set (34) means
  any flip looks proportionally large. Inspect the lost set before
  committing if any downstream narrative names App_2mo kinases.
- **App_2mo is the only contrast that nominally loses hits.** The most
  likely interpretation: those hits' nominal significance was propped up
  by plex variance lumped into the residual. Removing them is a
  feature, not a bug.

## Concrete change when activating

Three edits, all in the live pipeline. No CONTRAST_COEFS change needed —
plex columns get 0-weight contrast vectors automatically, which is the
correct conditioning behavior (extract biology *while holding plex fixed*).

### 1. `alz/kinase_enrich.py:_build_design_matrix` (line 141)

Append plex dummies (drop one as reference) when `analysis_mode == "males_only"`:

```python
if analysis_mode == "males_only":
    plex = meta["plex"].astype(int).values
    plexes = sorted(set(plex))
    for p in plexes[1:]:
        X[f"plex_{p}"] = (plex == p).astype(float)
```

Place after the existing biology columns and before the time/interaction
columns (or at the end — order doesn't matter; contrasts are looked up by
name in `CONTRAST_COEFS` consumers at `alz/pipelines/enrich/nodes.py:74`).

### 2. Update the docstring at `_build_design_matrix`

```text
males_only mode:  N x (10 + n_plex - 1) — adds plex dummies (one per plex
                  beyond the reference) to absorb residual TMT batch effect
                  not removed by IRS.
full_cohort mode: N x 11 (adds 'female') — plex dummies NOT added (plex ×
                  sex are perfectly confounded; would absorb all sex variance).
```

### 3. Methods note

Add to whatever methods document gets generated for the manuscript:

> Per-site OLS for the males-only primary analysis included TMT plex
> as a fixed-effect categorical covariate (plexes 1, 3, 5; reference =
> plex 1) to absorb residual batch variance not corrected by IRS
> normalization. The full-cohort sensitivity analysis omits the plex
> covariate because plex and sex are perfectly confounded by study
> design (plexes 1/3/5 = males, plexes 2/4/6 = females).

No other code changes required:

- `param_names = list(X.columns)` at `alz/pipelines/enrich/nodes.py:56`
  flows new columns through automatically.
- `xtxinv_per_site` shape `(n_sites, n_params, n_params)` expands
  automatically.
- DOF computation `nobs - len(param_names)` adapts automatically.
- `kinase_mechanism.py` reuses Stage 2 helpers — no change needed.
- `snrna_integration.py` builds its own design matrix; not affected.

## Validation when activated

1. Run `pixi run enrich` (default males-only) end-to-end; confirm
   `site_level_ols.csv` has no schema regressions.
2. Diff `mea_stoichiometry.csv` against the pre-change version — expect
   total significant kinase·contrasts to move from 1212 → ~1238 (±churn).
3. Confirm App_2mo specifically lost hits match the diagnostic prediction
   (12 lost, 5 gained vs the pre-change file). If the lost kinases
   matter for an active narrative, escalate before committing.
4. `KEDRO_ENV=full_cohort pixi run enrich` should emit **no plex columns**
   in the design matrix. Verify by adding a print of `list(X.columns)`
   during smoke testing.
5. Re-run `kinase_attribute.py` and `attribution_recovery.py` — concordance
   columns and the final hypothesis table will shift slightly with the
   new NES values. Spot-check the top kinases haven't moved.
6. Re-run `pixi run dual` to confirm the full_cohort sensitivity track
   still produces the same output it did before (full_cohort is unchanged
   by this plan).

## Constraints / things that close this plan

- **Do not apply to full_cohort.** Plex × Sex confounding makes it
  ill-conditioned. Documented above.
- **Do not extend to additional batch covariates without checking
  identifiability first.** This plan works because the males-only design
  happens to have plex orthogonal to genotype × timepoint. A different
  cohort or a different sample-mapping fix could break that.
- **Do not change CONTRAST_COEFS.** Plex contrasts are not biology of
  interest; the 0-weight contrast vector for plex columns is the correct
  treatment.

## Why this is deferred rather than committed

The diagnostic shows the change is principled and the impact is modest.
Deferring lets:

- the current `kinase_hypothesis_table.csv` and downstream artifacts stay
  stable while ongoing work depends on them;
- a focused validation pass be done at the same time as any other planned
  change to `kinase_enrich.py` (avoids spurious diff churn for reviewers);
- App_2mo lost-kinase inspection happen in the same change-set, so the
  scientific impact is documented alongside the methods change.

## Activation checklist

When picking this back up:

- [ ] Re-read this doc end-to-end
- [ ] Re-run the diagnostic to confirm numbers haven't drifted (re-create
      `scratch/plex_covariate_mea_check.py` if no longer present — the
      logic is fully described in §Diagnostic results above)
- [ ] Apply the three code edits in §Concrete change
- [ ] Run validation steps 1–6
- [ ] Inspect App_2mo lost kinases and document any scientific implications
- [ ] Update CLAUDE.md gotcha section if the change adds any non-obvious
      operational behavior
- [ ] Commit as a single `refactor(enrich):` commit so the methods change
      is one atomic diff
