# Multi-Agent Operating Protocol

Date: 2026-06-17

## Purpose

This protocol defines how agents should work on the cohort abstraction refactor.
It is mandatory for every phase.

## Global Rules

1. **Clean start required.**
   An agent must record `git status --short --branch` before editing. If the
   working tree is dirty, the agent must classify the changes and stop unless
   the orchestrator explicitly assigns those changes to the agent.

2. **Small scoped changes only.**
   Each agent receives a work packet with an explicit file scope. Do not edit
   files outside that scope unless the decision is logged and approved.

3. **No destructive commands.**
   Do not delete, overwrite, or regenerate canonical outputs unless the phase
   explicitly permits it. New implementations must write to scratch/versioned
   output roots until parity is reviewed.

4. **No hidden behavior changes.**
   Changes to sample inclusion, contrast definitions, sign conventions, filters,
   thresholds, output schemas, confidence tiers, or viewer payload semantics are
   behavior changes. They require explicit decision-log entries.

5. **No synthetic analytical scores.**
   Do not add arbitrary numeric analysis outputs. Internal implementation gates
   may exist, but user-facing outputs should expose categorical calls and raw
   evidence columns with clear units.

6. **Prefer additive compatibility.**
   When introducing new modules, keep existing commands/imports working through
   wrappers or adapters until the phase explicitly retires them. Superseding
   note for the implemented cohort move: Mukesh/T-cell/5xFAD old Python paths
   are retired without wrappers; use `python -m alz.cohorts...`.

7. **Do not mix provenance.**
   Scratch, diagnostic, frozen, and canonical inputs must remain clearly labeled.
   A production output must stamp its input roots and code path.

## Work Packet Template

Each agent assignment should include:

```text
packet_id:
phase:
owner_agent:
files_allowed:
files_read_only:
goal:
non_goals:
expected_outputs:
commands_to_run:
parity_checks:
decision_log_path:
rollback_plan:
```

## Required Start Checklist

Before edits:

```bash
git status --short --branch
git rev-parse HEAD
```

Record both values in the phase monitoring report.

## Required End Checklist

After edits:

```bash
git status --short
rg "<old path or old import patterns relevant to the phase>" -n ...
python -m py_compile <changed python files>
```

Run phase-specific validators and parity checks.

## Decision Log Template

Each decision entry should use this format:

```markdown
## Decision: <short title>

- Date:
- Phase:
- Agent:
- Files affected:
- Decision:
- Reason:
- Alternatives considered:
- Analysis behavior changed: yes/no
- Output schema changed: yes/no
- Backward compatibility impact:
- Validation/parity evidence:
- Reviewer:
```

## Drift Exception Template

Use only when parity does not hold and the drift is intended.

```markdown
## Drift Exception: <short title>

- Date:
- Phase:
- Agent:
- Output files affected:
- Row/key impact:
- Numeric field impact:
- Categorical field impact:
- Reason drift is intended:
- Downstream interpretation impact:
- Viewer impact:
- Approval:
```

## Conflict Handling

If two agents need the same file:

1. Stop both work packets before editing.
2. Split the file ownership by function or sequence the packets.
3. Record the sequencing decision.

## Rollback Handling

Every packet must be rollbackable by reverting its commit or dropping its
scratch outputs. If rollback would require manual reconstruction, the packet is
too large and must be split.
