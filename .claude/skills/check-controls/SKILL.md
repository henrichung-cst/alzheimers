---
name: check-controls
description: >
  Sanity-check a just-produced result for biological plausibility. Forms prior
  expectations in-the-moment from the analysis scope (using model biological
  knowledge, cited where possible), then judges whether the result agrees. No
  predefined control list — expectations are derived per result. The one hard
  rule: expectations are sourced to external biology, never to our own pipeline
  output. Output is conversational; nothing is written to disk. Invoke as:
  /check-controls [optional hint].
allowed-tools:
  - Read
  - Bash
  - Glob
---

# Biological plausibility check

Judge whether a result the analysis just produced is biologically plausible,
against prior expectations formed **now** from the scope of that analysis. This
is a control check: it catches results that are internally consistent but
externally implausible.

## The one hard rule

**Expectations come from external biology — literature, atlases, established
mechanism — never from our own pipeline output.** A result cannot be its own
control. If the only reason to expect X is that our pipeline reported X, that is
not an expectation; say so and skip the axis.

Expectations are model biological knowledge. Cite a PMID / atlas / mechanism
when one is available; when the expectation is a reasoned guess rather than
consensus, **flag it as a hypothesis** — a result disagreeing with a hypothesis
is noted as such, not treated as a pipeline bug.

## Procedure

### 1. Identify the result and its scope (from context)

Read the recent conversation and any artifact just produced or discussed. Pin
down:
- **What quantity** — cell-type home / specificity, disease direction (sign),
  enrichment or magnitude, a specific gene-cohort claim, etc.
- **What scope** — cohort, tissue/region, contrast, cell population. Scope
  determines which biology is the right yardstick (a 5xFAD cortex kinase
  enrichment is judged against different priors than a T-cell exhaustion state).

If the result or scope is ambiguous, state what you inferred and proceed; ask
only if you genuinely cannot tell what is being checked.

### 2. Read the actual values

Fetch the real numbers from the artifact or context — do **not** re-derive or
re-run the analysis. If the value isn't available on disk / in context, name the
file or step that would produce it and continue with what is present.

### 3. Form prior expectations — before judging

State, for each axis of the result, what known biology predicts **for this
scope**: expected cell-type home / family, expected direction, rough magnitude
if meaningful. Write these down *before* comparing, so the judgment isn't
back-fitted to the result. Cite where possible; flag hypotheses.

### 4. Compare and judge each axis

Show the actual value next to the expectation, then apply one verdict:

- `✓ as-expected` — agrees with a settled external expectation
- `⚠ off` — disagrees with a settled (non-hypothesis) expectation
- `~ borderline` — right family but weak, related subtype, or sign present but
  small
- `? vs hypothesis` — expectation was flagged a hypothesis; note agree/disagree
- `– N/A` — no external expectation exists, or the value isn't available

Judge by biological family, not a numeric threshold: astrocyte vs astrocyte
subtype is agreement; astrocyte vs microglia is borderline; astrocyte vs neuron
is off. For a direction/sign, consider whether it is stable across contrasts.

### 5. Report to conversation

Render a compact table — one row per (result, axis) — with the actual value, the
prior expectation (+ source), and the verdict. Follow it with a one-line overall
read. **Write nothing to disk.**

```
<result> | <scope>
  <axis>: actual=<value>  |  expected=<expectation> [source / hypothesis]  →  <verdict>
```

If nothing checkable is present, say what result or artifact you'd need and stop.
