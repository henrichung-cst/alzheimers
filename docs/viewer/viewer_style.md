# Unified Viewer — Writing Style Guide

Reference for all panel notes, drawer content, tooltips, and annotations in
`alz/build_unified_viewer.py`. Apply when writing new copy or revising existing text.

## Audience floor

Write as if the reader is a working biologist who knows nothing of our methods.

You may assume the reader knows: what a kinase is, what phosphorylation is,
what receptors / ligands / effectors / target genes are, roughly what RNA-seq
measures, what a knock-in mouse model is, that FDR is a thing.

You may **not** assume the reader knows: what MEA, NES, TPDS, null1/null2,
stoichiometry-correction, IRS normalization, or kinase enrichment are; what
"the kinase library" refers to; what a "candidate route" is unless you have
just told them; that "snRNA-seq" is the same as single-nucleus RNA-seq;
that "sender" and "receiver" are paired roles in this analysis rather than
generic labels.

The litmus test: would a friend who finished a biology PhD ten years ago
and never read this paper be able to picture every noun in your sentence?
If not, you are writing for yourself.

## Core principles

**Ground every claim in observable biology.**
The single biggest failure mode is substituting jargon with evocative
metaphors that only decode if the reader already has the technical model
loaded. Phrases like "the kinases line up with the route's wiring,"
"the chain stands out from the chance baseline," or "the route carries
disease-linked evidence" *feel* like simplification but are really
vagueness — every term in them is a hand-wave that requires a private
mental graph to interpret.

The right move is not a softer-sounding metaphor. The right move is to
state the mechanism in literal biological language: kinases phosphorylate
proteins; we measured phosphorylation; we know which kinases phosphorylate
which proteins from prior work; we therefore ask whether the kinases that
look unusually active in disease happen to be the ones that phosphorylate
the proteins in this chain — more often than they would if we drew kinases
at random. Every noun in that sentence points at something a biologist
can picture. No metaphor required.

If you find yourself reaching for "wiring," "alignment," "lines up,"
"stands out," "carries evidence," "supports," "concentrates near,"
"informs," or "speaks to" — stop. Ask what the literal procedure is and
write that.

**Introduce mechanism once, then reference.**
Build the mechanistic core in one place — typically the figure's "How
it was generated" section. Subsequent sections can then use shorthand
like "kinases flagged as disease-active" because the phrase has a
definite meaning the reader has already absorbed. Do not re-derive the
mechanism in every paragraph; do not assume the reader inferred it
either.

**Define labels, do not just use them.**
"Candidate route" means nothing until you say what made it a candidate
("the cell expresses all three proteins"). "Disease-active kinase"
means nothing until you say how that flag was assigned. Treat every
adjective on a noun as a debt: the reader needs to know what condition
it represents before the noun is usable.

**State the mechanism, not just the conclusion.**
Explain *why* a number or pattern looks the way it does, not just that
it does.
Avoid: "FDR controls stringency." Prefer: "FDR sets how often we expect a
false positive among the routes flagged as significant — lower means
fewer false alarms but also fewer discoveries."

**Anchor every number to its meaning.**
When a specific number appears, follow it immediately with what it
implies.
Avoid: "152 significant kinases." Prefer: "152 significant kinases —
nearly half the kinase library — so a randomly drawn kinase set overlaps
a chain's substrates almost as well as the truly disease-active set."

**Use a comparison to calibrate.**
Whenever a result looks surprising (empty columns, very high or low
counts), name a contrast or timepoint that behaves differently and say
why. This prevents the reader from wondering whether the test ever works.
Avoid: "No routes pass for Tau_4mo." Prefer: "Tau_2mo passes because
only 74 kinases are flagged as active, leaving room for specific chains
to score above random; Tau_4mo and Tau_6mo do not, because at those
timepoints nearly half the kinase library looks active and no chain
can stand above the random baseline."

**End on interpretation, not limitation.**
Close explanatory notes with what the result tells us scientifically,
not with a hedge or an apology. Reserve caveats for genuinely
unresolved ambiguity.
Avoid: "This may be a limitation of the test." Prefer: "Tau biology
starts focused and broadens until specific chains can no longer be
distinguished."

**Define jargon at every use, not just first use.**
This codebase's drawer copy parenthesizes jargon on every appearance,
not only the first one. The reader is not expected to remember a
definition from earlier in the same panel. Examples:
- "ApTt (App-Tau double knock-in)" every time the abbreviation appears.
- "FDR < 0.25 (false-discovery rate; fewer than one in four flagged
  chains is expected to be a chance result)" every time FDR is named.
- "Tau_2mo (Tau genotype, 2 months)" when the contrast label is
  referenced as a label, not a count.

To keep this from drowning the prose, prefer descriptive phrasing
over abbreviations where you have a choice — "the double genotype"
in flowing text, "ApTt" only when referencing a label the reader
sees on the figure.

## Six-section structure for figure descriptions

Drawer copy for any figure tab follows this fixed order. The renderer
in `alz/build_unified_viewer.py:renderHowToDrawer` produces these
section headers automatically when a `TAB_GUIDE` entry uses the
`{ preamble, method, shows, howTo, conclusions, toggles }` schema.

1. **Preamble (no header)** — what the figure is, literally. Rows are
   X. Columns are Y. Cell color encodes Z. One short paragraph.
2. **How it was generated** — the methods minimum, written in observable
   biology. State the mechanism once, here. Avoid named methods (MEA,
   Incytr, stoichiometry) unless the named thing is itself the subject;
   prefer the underlying procedure.
3. **What it shows** — the row/column reading rules, then the headline
   patterns from this dataset as anchored bullets. Each bullet should
   pair a number with what it implies.
4. **How to read it** — interaction model (clicking, propagation), color
   scale caveats, and the reminder that count is a prioritization signal
   not a measure of importance.
5. **Conclusions** — one paragraph that synthesizes the headline patterns
   one level higher than the bullets and points the reader toward the
   first round of follow-up.
6. **Adjustable toggles** — every UI control that affects the figure,
   each with a one-line description of what changes when you move it
   and when to reach for it.

## Voice and grammar

- Active voice. "The test asks whether…" not "Routes are tested for…"
- Present tense for what the viewer shows; past tense for what the
  pipeline did.
- No hedging phrases: "it should be noted," "it is important to mention,"
  "one may observe." State the thing directly.
- No em-dash stacking. One em-dash aside per sentence maximum (a paired
  pair counts as one).
- Spell out numbers below ten; use numerals for 10 and above. Treat
  figure-axis labels (2mo, 4mo, 6mo, App_4mo, Tau_2mo) as names, not
  counts — they keep their canonical form even when the count rule
  would say otherwise.
- Percent sign (%) attached to numeral, no space.

## Sentence structure

- Lead with the subject the reader cares about (the biological entity,
  the contrast, the cell type), not with the method.
  Avoid: "MEA analysis of stoichiometry-corrected phosphosites identifies…"
  Prefer: "Kinase activity is inferred from…"
- Keep panel-note opening sentences to one clause. Save the mechanism for
  the second sentence.
- Bulleted lists only for parallel items of equal weight. Do not use
  bullets to break up prose that reads naturally as sentences.

## Worked example — the metaphor trap

**Before (compressed metaphor — fails the audience-floor test):**
> For each route, the analysis asked whether the kinases that look
> unusually active in disease phosphoproteomics line up with that
> route's specific wiring more strongly than they would by chance.

What does "wiring" mean? What does "line up" mean? What does "by chance"
refer to? Each phrase decodes only if the reader already knows the
kinase–substrate graph and the permutation-test framing. A biologist
who has not read our methods has no way to reconstruct what is being
asked.

**After (mechanism in observable biology):**
> For each chain, the analysis asked whether the kinases flagged as
> disease-active in that genotype-by-timepoint context happen to be
> the same kinases that phosphorylate the receptor, effector, or target
> proteins in the chain — more often than they would if we drew kinases
> at random.

Every noun now points at something the reader can picture: kinases (the
enzymes), the proteins in the chain (named explicitly), the random draw
(the literal control). No private mental model required.

## Worked example — panel note

**Before (generic):**
> Use this panel to ask which kinases are most implicated. Each row is a
> kinase from the MEA analysis. NES describes the direction and strength
> of inferred kinase activity from stoichiometry-corrected phosphosites;
> FDR controls how stringent the significance count is.

**After (style-conformant):**
> Each row is a kinase whose substrate phosphosites shift coherently
> across the ranked phosphoproteome in at least one disease contrast.
> NES (normalized enrichment score) captures the direction and magnitude
> of that shift — positive means the kinase's substrates are more
> phosphorylated in disease, negative means less. FDR (false-discovery
> rate) sets how often we expect a false positive among the kinases
> flagged: at FDR < 0.25, roughly one in four flagged kinases is a
> false positive, which is appropriate for hypothesis generation.
> Select a row to see how the enrichment evolves across timepoints,
> which cell types show supporting transcriptomic evidence, and which
> signaling routes the kinase wires into.
