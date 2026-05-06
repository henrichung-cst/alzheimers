"use strict";

// ---------------------------------------------------------------------------
// Payload
// ---------------------------------------------------------------------------
const PAYLOAD = JSON.parse(document.getElementById("payload-data").textContent);
const META = PAYLOAD.meta;
const CONTRASTS = META.contrasts;
const RECEIVERS = PAYLOAD.celltypes.name;
const TISSUE_CAT = PAYLOAD.celltypes.tissue_category;
const DISEASE_COLORS = META.diseaseColors;
const PATHWAY_EVIDENCE_LABELS = {
  "expression-confirmed": "Expression-confirmed",
  "kinase-imputed": "Kinase-imputed",
  "mixed": "Mixed",
};

// ---------------------------------------------------------------------------
// Store — reducer-style with {selection, filters, view} slices
// ---------------------------------------------------------------------------
const INITIAL_STATE = {
  selection: { kinase:null, backbone:null, celltype:null },
  filters:   { contrast:"ALL", direction:"ALL", receiver:"ALL", sender:null,
               pathwayEvidence:"ALL", fdr:0.25, score:0.0, graphNodeIds:null,
               tpdsSig:"OFF" },
  view:      { activeTab:"signal", overviewMode:"count",
               overviewSort:"tissue", glossaryOpen:false,
               graphLayout:"concentric", graphMinDegree:1,
               graphGenotype:"App", graphTimepoint:"2mo",
               graphTpdsMin:0, graphTopN:null,
               senderMatrixMode:"count",
               senderMatrixAxis:"timepoint", senderMatrixAnchor:"ApTt",
               senderMatrixLastAnchorByAxis:{ genotype:"2mo", timepoint:"ApTt" },
               kinaseAuditTab:"measurement-trace",
               temporalLevel:"kinase", temporalMetric:"count",
               temporalTissue:"ALL",
               additivityLevel:"kinase", additivityTimepoint:"ALL",
               temporalScoreMin:0, additivityScoreMin:0, pathwayScoreMin:0 },
};

const _clone = (typeof structuredClone === "function")
  ? structuredClone
  : (v) => JSON.parse(JSON.stringify(v));

function reducer(state, action) {
  const s = _clone(state);
  if (action.type === "SET_FILTER") s.filters[action.key] = action.value;
  else if (action.type === "SET_SELECTION") s.selection[action.key] = action.value;
  else if (action.type === "SET_VIEW") s.view[action.key] = action.value;
  else return state;
  return s;
}

const Store = (function(){
  let state = _clone(INITIAL_STATE);
  const subs = [];
  return {
    get state() { return state; },
    subscribe(fn) { subs.push(fn); return () => {
      const i = subs.indexOf(fn); if (i >= 0) subs.splice(i, 1);
    }; },
    dispatch(action) {
      const prev = state;
      const next = reducer(state, action);
      if (next === prev) return;
      state = next;
      for (const fn of subs) fn(next, prev);
    },
  };
})();
window.Store = Store;  // expose for console smoke test

// ---------------------------------------------------------------------------
// Canonical metric glossary — single source of truth for tooltips, column
// header labels, and the per-tab "How to read" drawer. Static HTML uses
// `data-metric="<key>"` to reference an entry; applyMetricTooltips() stamps
// the .short text into `title=` at boot. Dynamic render functions read
// METRIC_DEFS[key].short directly.
// ---------------------------------------------------------------------------
const METRIC_DEFS = {
  // Global filters
  contrast: {
    label: "Contrast",
    short: "Disease-by-timepoint comparison (e.g. App_4mo). Pick one to scope panels that need a single contrast; All shows pooled views where supported.",
    howToRead: "Pick a contrast first; the rest of the bar narrows from there." },
  direction: {
    label: "Direction",
    short: "Up- vs down-regulated in disease. Filters by signed TPDS for pathways and signed NES for kinases.",
    howToRead: "Use to isolate gain-of-activity vs loss-of-activity drivers." },
  receiver: {
    label: "Receiver",
    short: "Downstream cell type that hosts the pathway. Restricts backbones to one receiver.",
    howToRead: "Useful when investigating a single cell type's signaling." },
  pathwayEvidence: {
    label: "Support",
    short: "How a backbone's chain was assembled: every protein detected, kinase-imputed, or mixed.",
    howToRead: "Expression-confirmed across multiple contrasts is the strongest evidence; imputed is exploratory." },
  fdr: {
    label: "FDR",
    short: "False-discovery-rate threshold for significant kinase activity (NES vs WT).",
    howToRead: "Lower = stricter. Default 0.25 follows GSEA convention." },
  score: {
    label: "|Score|",
    short: "Minimum absolute pathway score (TPDS or observed) to keep a backbone.",
    howToRead: "Raise to focus on high-magnitude pathways." },

  // Kinase explorer columns
  kinaseName:    { label: "Kinase",        short: "Kinase identifier from the MEA / integration tables." },
  kinaseFamily:  { label: "Family",        short: "Kinase family annotation." },
  kinaseGene:    { label: "Gene",          short: "Gene symbol associated with the kinase." },
  nSig:          { label: "Sig vs WT",     short: "Number of contrasts where this kinase's MEA FDR is below the header threshold." },
  peakNES:       { label: "Peak NES",      short: "Largest |NES| across contrasts. Sign indicates direction." },
  topCelltype:   { label: "Top cell type", short: "Top attributed receiver cell type from the attribution evidence table." },
  highConfAttr:  { label: "Conf",          short: "Whether the kinase has high-confidence cell-type attribution." },
  nBackbones:    { label: "#Backbones",    short: "Number of distinct pathway backbones with significant support from this kinase, across all contrasts." },

  // Pathway browser columns
  receiverCol:     { label: "Receiver",         short: "Receiver cell type for the backbone." },
  receptorCol:     { label: "Receptor",         short: "Receptor gene in the backbone." },
  emCol:           { label: "EM",               short: "Extracellular-matrix or intermediate molecule in the backbone." },
  targetCol:       { label: "Target",           short: "Downstream target gene in the backbone." },
  tpds:            { label: "TPDS",
                     short: "Transcript-level pathway differential score for the selected contrast (max |TPDS| when All is selected).",
                     howToRead: "Magnitude tells you how strongly the chain shifts; sign tells you which way." },
  passingContrasts:{ label: "Passing contrasts",
                     short: "Genotype-by-timepoint contrasts where this backbone passed both permutation nulls.",
                     howToRead: "More contrasts = more reproducible. Use the contrast-set chips above to combine exact sets." },
  nSenders:        { label: "Senders",          short: "Number of significant sender cell types detected for this backbone." },
  maxAbsTpds:      { label: "Max |TPDS|",       short: "Largest absolute TPDS observed across contrasts." },

  // Pathway-detail h4 sections
  passedNulls:    { label: "Passed both nulls by contrast",
                    short: "Conditions where this pathway passed both significance tests (kinase-enrichment null and receiver-specific wiring null).",
                    howToRead: "More chips = more reproducible. Only pathways passing in ≥1 contrast appear in the viewer." },
  pathwaySupportH:{ label: "Pathway support by contrast",
                    short: "Whether each chain step was directly measured or imputed, per contrast.",
                    howToRead: "Expression-confirmed across multiple contrasts is the strongest evidence." },
  tpdsCross:      { label: "TPDS across contrasts",
                    short: "Signed pathway score per contrast.",
                    howToRead: "Red = up in disease, blue = down. Black outline marks contrasts that passed both nulls — those bars are the trustworthy ones." },
  drivingKinasesH:{ label: "Driving kinases",
                    short: "Kinases ranked by how much signal they push into this pathway.",
                    howToRead: "Top rows are the strongest driver candidates. Direction tells you whether the drive is up or down in disease." },

  // Driving-kinase columns
  support:         { label: "Support",
                     short: "Total signal a kinase pushes into this pathway. Bigger = stronger driver.",
                     howToRead: "Use this to rank top driver candidates." },
  drivingDirection:{ label: "Direction",
                     short: "Signed Support: + = more active in disease, − = less, ~0 = mixed evidence.",
                     howToRead: "High Support + strong sign = clean driver. Near-zero relative to Support = weaker candidate." },
  trend:           { label: "Trend",
                     short: "Quick-read direction: ↑ mostly up, ↓ mostly down, — balanced. Counts in parens are (up-evidence / down-evidence).",
                     howToRead: "Counts evidence, not magnitude — use Direction for magnitude." },
};

function _metricShort(key) {
  const m = METRIC_DEFS[key];
  return m ? m.short : "";
}

// Stamp data-metric -> title on every element with a known key. Idempotent;
// safe to call after dynamic re-renders.
function applyMetricTooltips(root) {
  const scope = root || document;
  scope.querySelectorAll("[data-metric]").forEach(el => {
    const key = el.dataset.metric;
    const m = METRIC_DEFS[key];
    if (m && m.short) {
      const raw = el.dataset.col || key;
      el.title = `Display label: ${m.label || el.textContent.trim()}\nRaw column: ${raw}\nDefinition: ${m.short}`;
      el.setAttribute("aria-label", el.title);
    }
  });
}
window.applyMetricTooltips = applyMetricTooltips;

// ---------------------------------------------------------------------------
// Per-tab "How to read" drawer content. Each entry distills purpose,
// primary-view orientation, metric cues (joined with METRIC_DEFS), and
// conclusions. Keep copy declarative — don't repeat tab labels.
// ---------------------------------------------------------------------------

const TAB_GUIDE = {
  signal: {
    preamble: "Rows are receiver cell types in the cortex. Columns are nine disease contexts: three genotypes (App, Tau, and the App-Tau double knock-in) measured at three timepoints (2, 4, and 6 months). Each cell's color encodes the number of receptor → effector → target gene chains, inside that receiver cell type, that the analysis flagged as disease-linked under that genotype and timepoint. Brighter cells mean more flagged chains; blank cells mean none cleared the test.",
    method: [
      "Kinases are enzymes that phosphorylate proteins, so changes in kinase activity show up as changes in how much of a particular site on a particular protein is phosphorylated. The phosphoproteomics in this study measured those phosphorylation levels across thousands of sites in App, Tau, and ApTt (App-Tau double knock-in) mice and compared them with controls. Combining the sites that moved with already-published kinase–substrate relationships, the analysis inferred which kinases must be unusually active or inactive to explain the observed pattern in each genotype-by-timepoint combination.",
      "Separately, single-nucleus RNA-seq identified which receptors, internal effector molecules, and target genes are expressed by each receiver cell type in the cortex. For every cell type, the analysis listed every receptor → effector → target chain in which the cell expresses all three proteins — chains the cell type could plausibly run as a signaling route.",
      "For each chain, the analysis asked whether the kinases flagged as disease-active in that genotype-by-timepoint context happen to be the same kinases that phosphorylate the receptor, effector, or target proteins in the chain — more often than they would if we drew kinases at random. A second check repeated the question with cell-type labels shuffled, asking whether the chain's kinase support is specific to this receiver or could come from any cell type. A chain is counted on the heatmap only if both checks gave positive results at FDR < 0.25 (false-discovery rate; fewer than one in four flagged chains is expected to be a chance result).",
    ],
    shows: {
      lead: "A bright row means one cell type carries flagged chains across many disease contexts — broadly affected. A bright column means one disease context produces flagged chains in many cell types — a widespread effect. An isolated bright cell points to a context-specific lead.",
      bullets: [
        "The double genotype (App-Tau, abbreviated ApTt) drives 32,356 chain-by-context entries — more than App and Tau combined. 25,839 of those chains never appear in App or Tau alone, so the combined pathology produces signaling-chain disturbance that neither single pathology generates by itself.",
        "App ramps up then resolves: 6,854 chains at 2 months, peaking at 9,726 at 4 months, declining to 5,644 at 6 months. The amyloid response concentrates at mid-disease.",
        "Tau is front-loaded: 10,027 chains at 2 months, then zero at 4 months and 6 months. The blank Tau columns are not biological silence. At those later timepoints, 152 then 180 different kinases were flagged as disease-active — nearly half of all kinases in the reference library. When that many kinases look active everywhere, a randomly-drawn kinase set overlaps a chain's substrates almost as well as the truly disease-active set, and no individual chain can pass the over-representation test. Tau_2mo (the Tau genotype at 2 months) passes because only 74 kinases are flagged as active, leaving room for specific chains to score above random. Tau biology starts focused and broadens until specific chains can no longer be distinguished.",
        "L5 IT is the sharpest Tau-early receiver: 2,987 Tau_2mo (Tau genotype, 2 months) chains and 2,763 ApTt_2mo (App-Tau double knock-in, 2 months) chains converge there, against 1,410 App chains summed across all three timepoints.",
        "Lamp5 Lhx6 is the most broadly affected receiver — 13,827 chains across all genotypes — making it a convergence point regardless of which pathology drives the disease.",
      ],
    },
    howTo: "Click any cell to pin that receiver as a filter; the selection carries through to every other tab, so the same cell type becomes the subject of the Pathway, Kinase, and Graph drill-downs. The color scale is log-compressed, so a block twice as bright does not represent twice as many chains. Treat chain count as a prioritization signal rather than a measure of biological importance — a single chain with a strong disease-vs-control phosphorylation shift is often more informative than many weakly-supported ones.",
    conclusions: [
      "Disease-linked signaling-chain disturbance in this cohort concentrates in the double genotype and arrives early. The combined App-Tau pathology produces the broadest disturbance and the most genotype-unique chains; Tau drives an early, focused signal that broadens until individual chains can no longer be resolved by 4 months; App rises and falls across the time course. Combined with the receiver concentration in Lamp5 Lhx6 and L5 IT, the map points the first round of follow-up toward early-disease, multi-genotype effects on specific neuronal subclasses rather than a uniform pan-cell-type response.",
    ],
    toggles: [
      { name: "Contrast filter", desc: "pick one disease context (one genotype at one timepoint) to focus the map on a single column." },
      { name: "Direction mode", desc: "switch from \"any flagged chain\" to a directional score that separates chains where phosphorylation went up in disease from chains where it went down, useful for asking whether a bright cell represents activated or suppressed signaling." },
      { name: "FDR threshold (false-discovery rate, the proportion of flagged chains expected to be chance results)", desc: "raise to require stronger statistical support, lower to capture weaker signal. The default of 0.25 is hypothesis-generation territory; 0.10 is closer to confirmatory." },
    ],
  },
  senders: {
    preamble: "Three 22×22 sender-by-receiver grids are shown side-by-side, each a real disease contrast — no averaging across panels. Rows are sender cell types; columns are receiver cell types. Both axes list the same 22 cortical cell types, since any cell type can play either role. Each cell's color encodes the number of receptor → effector → target gene chains, in that panel's disease context, in which the ligand at the start of the chain came from the sender cell type and the receptor that catches it sits on the receiver cell type. The Compare control above the grid sets which dimension is varied across the three panels: hold the genotype fixed and vary the timepoint, or hold the timepoint fixed and vary the genotype.",
    method: [
      "A signaling chain in this analysis runs from one cell type to another: the sender cell type expresses and releases a ligand, the receiver cell type expresses a receptor that binds it, an internal effector molecule in the receiver passes the signal along, and a target gene at the end of the chain is switched on or off. Single-nucleus RNA-seq identified which cell types express which ligands, receptors, effectors, and target genes; for each sender-receiver pair, the analysis listed every chain in which the sender expresses the ligand and the receiver expresses the receptor, effector, and target.",
      "Each chain was then filtered the same way as in the Signal Map. Phosphoproteomics measured which protein sites moved up or down in the chosen disease genotype-and-timepoint compared with controls. Combining the moved sites with already-published kinase–substrate relationships, the analysis inferred which kinases must be unusually active or inactive to explain the pattern. For each chain it asked whether those disease-active kinases happen to be the same kinases that phosphorylate the receptor, effector, or target proteins in the chain — more often than they would if we drew kinases at random — and a second check repeated the question with cell-type labels shuffled. Chains that passed both checks at FDR < 0.25 (false-discovery rate; fewer than one in four flagged chains is expected to be a chance result) are counted in the cell for that sender-receiver pair.",
    ],
    shows: {
      lead: [
        "Within any one panel, a dense row means one sender cell type is the source of many flagged chains landing across many receiver cell types — a broadly broadcasting cell in that disease context. A dense column means one receiver cell type is on the receiving end of flagged chains from many senders — a cell with broad incoming remodeling. Bright cells on the diagonal point to within-cell-type signaling, where the same cell type plays both roles. The point of showing three panels at once is comparison: read across the row of panels and ask which sender or receiver patterns persist, change, or appear only in one slice.",
        "The default view (ApTt at 2mo, 4mo, 6mo) was chosen because the trajectory of the double genotype across time is the most biologically loaded read in this dataset. Three other reading paths follow from flipping the axis or stepping the anchor:",
      ],
      bullets: [
        "App across time (App at 2mo, 4mo, 6mo). Step the anchor to App. Watch which sender rows thicken into mid-disease at 4 months and collapse back at 6 months — those are the cell types whose ligand output rises and resolves with the amyloid response. Receiver columns that stay dense across all three panels mark cell types whose incoming remodeling is sustained.",
        "Tau across time (Tau at 2mo, 4mo, 6mo). Step the anchor to Tau. Tau_4mo and Tau_6mo render as blank panels for the same reason as in the Signal Map: at those timepoints nearly half the kinase library is flagged as disease-active, so no individual sender-receiver pair stands above the random baseline. The interpretable Tau structure is at 2 months alone.",
        "Cross-section at one timepoint (App, Tau, ApTt at 2mo). Flip the axis to Compare genotypes and set the anchor to 2mo. The early-disease snapshot lets you ask which sender or receiver patterns are shared across all three genotypes versus which are genotype-specific.",
        "Persistent dense rows or columns across all three panels — whatever the axis — identify sender or receiver cell types whose role is structural rather than stage- or genotype-specific.",
      ],
    },
    howTo: "The Compare control sets the comparison axis: Compare timepoints holds the genotype fixed (the anchor) and shows that genotype at 2mo, 4mo, 6mo; Compare genotypes holds the timepoint fixed and shows App, Tau, ApTt at that timepoint. Step the anchor with ← and →; flip the comparison axis with ↑ or ↓. Each axis remembers the last anchor you used on it, so flipping back returns to where you were. The color scale is pinned across all nine contrasts, so brightness is directly comparable across panels and across anchor steps — a cell that looks faint in one panel really is fainter than the same cell in another, not just rescaled. Click any cell to pin its receiver as a global filter; the choice carries through to the Pathway, Kinase, and Graph drill-downs. The color scale in count mode is log-compressed, so a cell twice as bright does not represent twice as many chains. A dense cell can come from many distinct chains or from repeated use of a few highly-shared ligand–receptor combinations — to distinguish, drill into that cell's chains in the Pathway tab.",
    conclusions: [
      "Showing three real disease contrasts side-by-side, never an average, is the design choice that lets sender-receiver structure be read as a trajectory rather than a single snapshot. Senders with dense rows in all three panels are broadcasting cell types whose role does not depend on the dimension being varied; receivers with dense columns across all three are absorption hubs in the same sense. Stage-specific or genotype-specific patterns — a sender row dense only at 4 months, a receiver column that thickens only when amyloid pathology is added — are the most informative for asking what each cell type does at a particular disease moment.",
    ],
    toggles: [
      { name: "Compare", desc: "selects the axis varied across the three panels. Compare timepoints holds the genotype fixed (the anchor) and shows that genotype at 2mo, 4mo, 6mo; Compare genotypes holds the timepoint fixed and shows App, Tau, ApTt at that timepoint." },
      { name: "Anchor", desc: "the dimension held fixed across the three panels. Its options switch with the axis: when comparing timepoints the anchor is one of App, Tau, ApTt; when comparing genotypes it is one of 2mo, 4mo, 6mo. Each axis remembers the last anchor you used on it." },
      { name: "Mode", desc: "switch between count (number of flagged chains per sender-receiver pair, log-scaled so the brightest cells do not crowd out moderate ones) and direction (chains where disease-vs-control phosphorylation went up minus chains where it went down, useful for asking whether a pair is dominated by activation or suppression)." },
      { name: "Arrow keys", desc: "← and → step the anchor within the current axis; ↑ and ↓ flip the axis. The color scale is pinned across all nine contrasts so brightness is comparable as you step." },
    ],
  },
  temporal: {
    preamble: "For each genotype, three points trace how the kinase or pathway signal evolves across 2, 4, and 6 months of age. In kinase mode, the y-axis is the count of kinases whose substrate phosphosites shift coherently in disease versus control at that timepoint. In backbone mode, the y-axis is the count of receptor → effector → target chains that cleared both permutation null tests at that timepoint. Three colored lines, one per genotype (App, Tau, ApTt), so flat versus rising versus peaked trajectories can be compared in one read.",
    method: [
      "Phosphoproteomics measured how much of each protein site is phosphorylated in each disease mouse line at each timepoint, normalized to the parent protein's abundance so changes in total protein do not show up as apparent kinase activity changes. Combining the sites that moved with already-published kinase–substrate relationships, the analysis inferred which kinases must be unusually active or inactive to explain the pattern in each genotype-by-timepoint context — those are the kinases counted in kinase mode at that timepoint.",
      "For backbone mode, single-nucleus RNA-seq listed every receptor → effector → target chain in which the receiver cell type expresses all three proteins. For each chain the analysis asked whether the disease-active kinases happen to be the same kinases that phosphorylate the receptor, effector, or target proteins in the chain — more often than they would if we drew kinases at random — and a second check repeated the question with cell-type labels shuffled. Chains that passed both checks at FDR < 0.25 (false-discovery rate; fewer than one in four flagged chains is expected to be a chance result) are counted in backbone mode at that timepoint.",
    ],
    shows: {
      lead: "Backbone mode exposes two readouts that are designed to disagree in the diffuse phase. The count line — passing chains per timepoint — drops to zero in late Tau because the chain test loses its statistical handle when the disease-active kinase set grows toward half the library. The mean |TPDS| line — pathway dysregulation magnitude across every enumerated chain in the receiver slice, with no significance gate — keeps climbing across the same timepoints, because TPDS measures how much pathway flux is shifted regardless of whether any individual route can be picked out from random. The two readouts read from different data: count and mean observed score iterate the chain-test-passing payload (the same chains the rest of the viewer shows), while mean |TPDS| and percent upregulated read a build-time summary computed across every chain the cell type expresses, including the diffuse-phase chains the viewer otherwise filters out. Read count for 'how many specific routes resolve' and mean |TPDS| for 'how much pathway burden is present'; their divergence is the focal-to-diffuse signature, not a contradiction. Pinning a specific sender or selecting a kinase falls back to the chain-test-passing payload for both metrics — magnitude in those scoped views is necessarily restricted to passing chains.",
      bullets: [
        "Kinase counts and chain counts often move in opposite directions across the time course — most visibly in Tau, where 74 disease-active kinases at 2mo grow to 152 at 4mo and 180 at 6mo while passing chains collapse from 10,027 to zero. This is not a contradiction. The two modes ask different questions: kinase mode counts enzymes whose substrate sites cluster coherently in disease (an enrichment test that gets stronger as more enzymes are weakly perturbed), while backbone mode counts chains where the disease-active kinases are specifically the ones that phosphorylate this chain's proteins more often than a random kinase draw would (an over-representation test that loses power as the active set approaches half the kinase library). Climbing kinases plus collapsing chains describes a shift from focal early signaling — a few specific axes resolvable by the chain test — to diffuse late remodeling that the chain test cannot pin to particular routes. Treat the divergence as the trajectory's headline read, not an artifact.",
        "App rises and falls in backbone mode: 6,854 chains at 2mo, peaking at 9,726 at 4mo, declining to 5,644 at 6mo. The amyloid response concentrates at mid-disease.",
        "Tau diverges between count and magnitude. The count metric is front-loaded: 10,027 passing chains at 2mo, then zero at 4mo and 6mo — the chain test cannot resolve specific routes once randomly drawn kinase sets overlap any chain almost as well as the truly disease-active set. The mean |TPDS| metric, applied across all enumerated chains rather than only passing ones, climbs across the same trajectory: median |TPDS| 0.014 → 0.017 → 0.018 and p95 0.097 → 0.106 → 0.120 from 2mo to 6mo. The blank count line is the diffuse-phase symptom; the climbing TPDS line is what's actually happening underneath.",
        "ApTt (App-Tau double knock-in) front-loads: 15,610 chains at 2mo, declining through 4mo (10,048) and 6mo (6,698). The trajectory resembles Tau's early-loading more than App's gradual build, suggesting the double genotype's early dynamic is largely Tau-shaped.",
        "Restricting to one receiver tests whether the genotype-specific timing is a property of one cell type or a cohort-wide pattern.",
      ],
    },
    howTo: "Switch between kinase mode and backbone mode with the toggle above the plot. In backbone mode the metric selector chooses between four readouts with deliberately different gating: 'count' is significance-gated (passing chains only); 'mean observed score' is significance-gated by definition (observed score is undefined for non-passing chains); 'mean |TPDS|' and '% upregulated' are not significance-gated and operate on every enumerated chain in the receiver/sender slice. The hover on any line shows both the passing-chain count and the chain-with-TPDS count so you can see how the two are weighted. The local |TPDS| ≥ control thins all metrics to chains whose mean total pathway dysregulation score (TPDS — the integrated shift in modeled signaling probability for the chain in disease versus control) clears the chosen value; default zero keeps every chain. The global FDR slider (false-discovery rate; fewer than one in four flagged chains or kinases is expected to be a chance result) tightens upstream kinase selection. Restrict by receiver in the global filter bar to ask whether timing is cell-type-specific. A flat count line can mean stable remodeling or absence of signal — read mean |TPDS| at the same timepoints to distinguish the two.",
    conclusions: [
      "The trajectory's two modes stratify the disease into a focal phase and a diffuse phase. In the focal phase — App across its full course, Tau at 2mo, ApTt at 2mo — the kinase landscape is specific enough that the chain test resolves particular receptor → effector → target routes, and kinase counts and chain counts move together. In the diffuse phase — Tau at 4mo and 6mo, where kinase counts climb past half the library — the chain test loses its statistical handle and chain counts collapse despite continued (broader, weaker) phosphoproteomic remodeling. Pathway remodeling is also not synchronous across genotypes: App builds to a mid-disease peak and partially resolves; ApTt and Tau front-load at 2 months, with ApTt resembling Tau's early dynamic more than App's gradual build. The first follow-up is to read late-phase signal through TPDS magnitude on the Pathway tab — a magnitude-based score that does not depend on active-set specificity — rather than asking the chain count to describe a regime where it is by design uninformative.",
    ],
    toggles: [
      { name: "Mode", desc: "switch between kinase mode (count of kinases passing at each timepoint) and backbone mode (count of receptor → effector → target chains clearing both permutation tests)." },
      { name: "|TPDS| ≥", desc: "local cut on chains in backbone mode. Default zero counts every passing chain; raise to keep only chains whose mean total pathway dysregulation score clears the chosen value." },
      { name: "FDR threshold (false-discovery rate)", desc: "applies to upstream kinase or chain selection. Tighten to focus on robust signals; loosen to capture weaker early signals." },
      { name: "Receiver", desc: "restricts both modes to one receiver cell type. Use to ask whether the trajectory is shared across cells or specific to one." },
    ],
  },
  additivity: {
    preamble: "A scatter that asks whether the App-Tau double genotype (ApTt) behaves like the sum of App and Tau or whether the two pathologies interact. The y-axis is the signal in ApTt; the x-axis is the predicted signal if App and Tau add linearly. Each point is one kinase (in kinase mode) or one receptor → effector → target chain (in backbone mode), shown separately at 2, 4, and 6 months. The diagonal is pure additivity — points above mean the double genotype exceeds the prediction, points below mean it falls short.",
    method: [
      "For each kinase, the analysis took the kinase's enrichment score (NES — normalized enrichment score; positive means the kinase's substrates are more phosphorylated in disease, negative means less) in the App, Tau, and ApTt contrasts at one timepoint. The x-axis plots App's NES + Tau's NES; the y-axis plots ApTt's NES. A point on the diagonal would mean the double genotype's enrichment is what you'd get from stacking App and Tau side-by-side.",
      "In backbone mode, each point is one receptor → effector → target chain in the current filter, plotted with x = the chain's observed score in App + its observed score in Tau, and y = its observed score in ApTt, at one timepoint. The observed score is the chain's permutation-tested pathway score (mean kinase support across the chain's nodes). No FDR filter is applied to the scatter itself — every chain admitted by the active support and receiver filters contributes a point — so the cloud is dense and a sub-additive bias is read from the bulk distribution, not from individual labeled points. Distance above or below the diagonal is the deviation from the additive prediction.",
    ],
    shows: {
      lead: "Points above the diagonal are supra-additive — the double genotype produces more signal than App and Tau together would predict. Points below are sub-additive — the two pathologies partly cancel or share a saturating mechanism. Spread along the diagonal at any one timepoint reveals magnitude; consistency across all three timepoints reveals whether the interaction is a stable feature or a stage-specific one.",
      bullets: [
        "Kinase mode shows only kinases that pass FDR (false-discovery rate; fewer than one in four flagged kinases is expected to be a chance result) in at least one of App, Tau, or ApTt at the chosen timepoint, color-coded by which subset of contrasts they clear: App only, Tau only, ApTt only, or Multi (≥2). Kinases that fail FDR everywhere are not plotted, matching the filtering convention used by the rest of the viewer.",
        "At 2 months in backbone mode, the bulk of the chain cloud sits below the diagonal — the double genotype's per-chain pathway score is mildly sub-additive relative to App + Tau. A separate count-based view of the same effect: 15,610 chains pass FDR in ApTt against 16,881 expected from App's 6,854 + Tau's 10,027, a ratio of 0.92×.",
        "At 4 and 6 months, Tau contributes essentially zero passing chains and a near-zero observed-score distribution, so the predicted (App + Tau) score collapses to App's score and the scatter cannot distinguish additivity from independence. Read these timepoints as confirming the double genotype is not silenced by Tau pathology, not as evidence for or against synergy.",
        "The kinase-level and backbone-level scatters can disagree on the same entity: a kinase can be supra-additive in NES while its supported chains are sub-additive in observed score, or vice versa. Both readings are legitimate, because they measure different layers — kinase-level enrichment versus per-chain pathway score.",
      ],
    },
    howTo: "Switch between kinase mode and backbone mode with the toggle. The local score-min control thins to points where either App, Tau, or ApTt's signal magnitude clears the chosen value, removing low-signal noise from the diagonal cloud. The global FDR slider tightens upstream kinase or chain selection. Step through the three timepoints — 2mo, 4mo, 6mo — before treating any single point as a stable interaction; an interaction that holds at one timepoint and inverts at another is a stage-specific phenomenon, not an additivity failure of the double genotype.",
    conclusions: [
      "The most interpretable additivity reading is at 2 months, when both single genotypes generate their own passing chains and their sum is a meaningful prediction. The 0.92× backbone-level ratio there is a mild sub-additive signal, suggesting partial mechanistic overlap or competition for shared signaling machinery rather than true independence. At later timepoints, Tau's collapse to zero passing chains makes the prediction degenerate; ApTt tracking App is consistent with either additivity or the double genotype defaulting to the App-driven mid-disease arc. The kinase-level scatter at each timepoint nominates individual enzymes whose interaction signature should be cross-checked against their cell-type attribution and supported chains in the Kinase and Pathway tabs.",
    ],
    toggles: [
      { name: "Mode", desc: "kinase mode plots NES (normalized enrichment score) per kinase; backbone mode plots mean kinase support score across passing chains." },
      { name: "Score min", desc: "drops points whose App, Tau, or ApTt signal magnitude falls below the chosen value. Use to thin the cloud near the origin and emphasize interactions among strongly-active entities." },
      { name: "FDR threshold (false-discovery rate)", desc: "tightens upstream kinase or chain selection before the scatter is computed." },
    ],
  },
  kinase: {
    preamble: "A ranked table of the 240 kinases whose substrate phosphosites shift coherently in at least one disease contrast. Each row is one kinase. NES (normalized enrichment score) columns capture the direction and magnitude of that shift in each genotype-by-timepoint context. Cell-type columns place that activity onto cortical subclasses using independent transcriptomic evidence. The backbone count is how many passing receptor → effector → target chains the kinase appears among the inferred drivers of.",
    method: [
      "Phosphoproteomics measured how much of each protein site is phosphorylated in App, Tau, and ApTt (App-Tau double knock-in) mice at each timepoint, normalized to the parent protein's abundance so changes in total protein do not show up as apparent kinase activity changes. For each disease contrast, the analysis ranked every measured site by its disease-versus-control change and asked, for each kinase in the reference library, whether that kinase's known substrate sites cluster toward the top or bottom of the ranking more strongly than they would if we drew sites at random — a positive NES means the substrates concentrate among the upregulated sites, a negative NES means they concentrate among the downregulated sites.",
      "Independently, single-nucleus RNA-seq from a separate human Alzheimer's cohort and a mouse brain reference atlas provided per-cell-type expression and disease-direction concordance for each kinase; those become the cell-type columns. The backbone count comes from the same chain analysis used elsewhere in the viewer: a chain passing at FDR < 0.25 (false-discovery rate; fewer than one in four flagged chains is expected to be a chance result) and naming this kinase among the over-represented substrate-phosphorylators contributes one to its count.",
    ],
    shows: {
      lead: "NES columns answer where in the disease landscape the kinase's substrates are most coherently shifted. Cell-type columns answer where in the cortex the kinase's transcript is concordantly differentially expressed in disease. Backbone count answers how broadly the kinase appears among inferred drivers of passing chains — a structural prevalence signal, not a per-chain magnitude.",
      bullets: [
        "240 kinases pass FDR < 0.25 in at least one of nine disease contrasts. Of those, 124 (52%) follow a peaked trajectory — enrichment rises and falls across the time course — while only three remain sustained across all three timepoints of one genotype.",
        "Peak enrichment concentrates in the double genotype: 125 kinases peak in an ApTt (App-Tau double knock-in) contrast versus 69 in App and four in Tau alone. The strongest individual signals by NES magnitude — AKT1, AKT2, AKT3 — are negative and peak at App_4mo (App genotype, 4 months), meaning their substrate phosphorylation is reduced relative to protein abundance specifically in amyloid disease at mid-disease. This AKT hypoactivity signature is absent from the Tau genotype.",
        "The broadest backbone supporters — CAMK2D (15,028 chains), CDK1 (14,776), CHK1 (13,098) — have moderate NES across many contrasts. They are structural participants in many chains rather than strong disease-specific signals.",
        "High NES with weak cell-type attribution is not evidence against the kinase; it is evidence that the transcriptomic side has less to say about where it acts. The reverse is also true.",
      ],
    },
    howTo: "Sort by any column to surface kinases by enrichment magnitude, cell-type concordance, or backbone breadth. Click a row to pin that kinase across the viewer — its trajectory across timepoints opens in the side panel, the Pathway tab restricts to chains it drives, and any cell-type filter on Signal Map or Sender × Receiver applies the same constraint. The global FDR slider (false-discovery rate) tightens upstream kinase selection: at 0.25, roughly one in four flagged kinases is a false positive, which is hypothesis-generation territory; at 0.10, the count falls but each remaining kinase is closer to a confirmatory call.",
    conclusions: [
      "The peaked-trajectory majority is the headline structural feature of the kinase landscape — most disease-active kinases turn on and off with stage rather than accumulating. Concentration of peaks in the double genotype, combined with the App-specific AKT hypoactivity signature and the broad-but-moderate enrichment of CAMK2 / CDK1 / CHK1, points the first round of follow-up to two questions: what is the cell-type origin of the AKT suppression in App_4mo, and do the structural backbone supporters carry chain-level direction information that the per-kinase NES summary obscures. Both questions chain directly into the Pathway and Sender × Receiver tabs.",
    ],
    toggles: [
      { name: "FDR threshold (false-discovery rate)", desc: "sets the cutoff for which kinases enter the table. 0.25 is hypothesis generation; 0.10 is closer to confirmatory." },
      { name: "Receiver, Support", desc: "when set in the global filter bar, restrict the backbone count column to chains landing on that receiver or carrying that support type, so the rank reflects the kinase's role in the chosen subset rather than its total prevalence." },
    ],
  },
  pathway: {
    preamble: "A scrollable list of the receptor → effector → target chains that passed both permutation null tests in at least one disease contrast. Each row is one chain. TPDS (total pathway dysregulation score) columns measure how strongly the chain's modeled signaling probability shifts in each disease context — positive means more activity in disease, negative less. The Passing contrasts column lists the disease contexts where this chain cleared both null tests. Click a chain to expand its driving kinases — the kinases whose disease-active substrate signature put the chain over the threshold in each contrast.",
    method: [
      "Single-nucleus RNA-seq identified which receptors, intracellular effector molecules (EM — signaling components linking receptor binding to gene expression), and target genes are expressed by each receiver cell type. For every receiver, the analysis listed every receptor → EM → target chain in which the cell expresses all three components.",
      "Phosphoproteomics measured which protein sites moved up or down in each genotype-by-timepoint context, normalized to the parent protein's abundance so changes in total protein do not show up as apparent kinase activity changes. For each chain, the analysis asked whether the kinases inferred as unusually active in disease happen to be the same kinases that phosphorylate the receptor, effector, or target proteins in the chain — more often than they would if we drew kinases at random. A second test repeated the question with cell-type labels shuffled, asking whether the chain's kinase support is specific to its receiver or could come from any cell type. Chains that cleared both tests at FDR < 0.25 (false-discovery rate; fewer than one in four flagged chains is expected to be a chance result) appear here.",
      "TPDS is computed independently of the over-representation test: it integrates the modeled signaling-probability shift along the entire chain — receptor, effector, and target components together — into one signed magnitude per chain per contrast.",
    ],
    shows: {
      lead: "Each row is one chain that passed both null tests somewhere. The Passing contrasts column shows where; the TPDS columns show how strongly the chain's modeled signaling probability shifted there. The trajectory buttons let you ask which chains pass under specific contrast patterns — one genotype's three timepoints (App, Tau, or ApTt trajectory), one timepoint across all three genotypes (2mo, 4mo, or 6mo cross-section), or every contrast.",
      bullets: [
        "Of 55,859 unique chains in the dataset, 230 pass in all three genotypes — the most reproducible leads across disease contexts.",
        "Most chains are genotype-specific: 25,839 pass only in ApTt (App-Tau double knock-in), 16,232 only in App, 8,061 only in Tau. Genotype-specific chains are the hypothesis-rich zone — each clears stringent permutation thresholds in one context but not others.",
        "Recurrence stratifies quality: a chain passing in six of nine contrasts is more likely to reflect stable structural remodeling than one passing in one. Use the trajectory buttons to require multi-contrast passage before drilling in.",
        "TPDS magnitude and recurrence answer different questions. A chain can pass many contrasts at modest TPDS (broad reproducible structure) or pass one contrast at large TPDS (a single-context shift large enough that the test catches it on its own). Both are worth examining for different reasons.",
      ],
    },
    howTo: "Filter contrast patterns with the seven trajectory buttons above the table. The three trajectory buttons (App / Tau / ApTt) keep chains that pass any of that genotype's three timepoints. The three cross-section buttons (2mo / 4mo / 6mo) keep chains that pass any of that timepoint's three genotypes. The All button removes the contrast-pattern restriction. Tighten the local |TPDS| ≥ control to keep only chains whose maximum |TPDS| across passing contrasts clears the chosen value — a magnitude floor on top of the permutation-pass requirement. The global Receiver and Support filters restrict to one receiver cell type or to one chain-support category (expression-confirmed routes have direct transcriptomic evidence for receptor, effector, and target; kinase-imputed routes are inferred from substrate evidence). Click a row to open its driving kinases panel.",
    conclusions: [
      "The chain catalog is dominated by genotype-specific routes — the double genotype alone contributes nearly half the unique chains — with a small core of 230 chains reproducible across all three genotypes. Recurrence is the primary quality stratifier here; the seven trajectory buttons let you go after specific patterns (one genotype's full trajectory, one timepoint's cross-section, all contrasts) without juggling contrast lists by hand. From any row, the driving kinases panel names the enzymes responsible for the chain clearing the over-representation test in each contrast — that is the link back to the Kinase tab, and the path from a chain hypothesis to a testable molecular driver.",
    ],
    toggles: [
      { name: "Trajectory buttons", desc: "App / Tau / ApTt keep chains that pass any of that genotype's three timepoints. 2mo / 4mo / 6mo keep chains that pass any of that timepoint's three genotypes. All removes the restriction." },
      { name: "|TPDS| ≥", desc: "local magnitude floor. Keeps chains whose maximum |TPDS| across passing contrasts reaches the chosen value. Use to add a magnitude requirement on top of the permutation-pass requirement." },
      { name: "Receiver", desc: "global filter; restricts to chains whose receiver cell type matches." },
      { name: "Support", desc: "global filter; switches between expression-confirmed (direct transcriptomic evidence for receptor, effector, and target) and kinase-imputed (inferred from substrate evidence)." },
      { name: "FDR threshold (false-discovery rate)", desc: "applies to the upstream chain selection. Tighten to require stronger statistical support." },
    ],
  },
  graph: {
    preamble: "One genotype-by-timepoint snapshot at a time, built from the routes that passed both permutation null tests in that contrast. Nodes are the receptor, EM, and target genes of those routes; edges connect genes that co-appear in the same passing route. Pick a genotype with the dropdown above the graph and step through the three timepoints with the Timepoint control or arrow keys; each step is an independent snapshot, not a fade between time-collapsed views. Tau_4mo and Tau_6mo render empty because at those timepoints nearly half the kinase library is flagged as disease-active and no individual route stands above the random baseline.",
    method: [
      "Two upstream filters define the universe of routes shown. First, single-nucleus RNA-seq identified which cell types in the cortex express each receptor, intracellular effector, and target protein, and Incytr enumerated every chain in which the receiver cell type expresses all three components. Second, phosphoproteomics measured which protein sites moved up or down in the chosen disease genotype-and-timepoint compared with controls, and the analysis asked, for each chain, whether the kinases flagged as unusually active in disease happen to be the same kinases that phosphorylate the receptor, effector, or target proteins in the chain — more often than they would if we drew kinases at random. A second test repeated the question with cell-type labels shuffled. Only chains that passed both tests at FDR < 0.25 (false-discovery rate; fewer than one in four flagged chains is expected to be a chance result) flow into this view.",
      "The remaining filters are local rendering choices. Min degree drops nodes connected to fewer than that many surviving routes. The |TPDS| ≥ X cut hides edges whose mean total pathway dysregulation score (TPDS — the integrated shift in modeled signaling probability for the chain in disease versus control) falls below the chosen value; default zero shows everything in the passing-both-nulls universe, drag up to thin out weak-signal edges. The optional Max edges cap is a separate safety net: when on, only the top N edges by |TPDS| are drawn.",
    ],
    shows: {
      lead: [
        "Each edge in the graph encodes shared route membership: two genes connect when they appear together in a chain that passed both permutation tests at this snapshot. Edge color comes from the mean TPDS across those shared chains — red for routes pointing up in disease (more modeled signaling activity than control), blue for routes pointing down, grey near zero. Read the network for two things at once: structure (which genes converge into hubs, which sit at the periphery) and direction (whether those hubs sit on red, blue, or mixed-color edges).",
      ],
      bullets: [
        "A receptor (R-prefix) gene with high degree means many distinct routes start at the same incoming signal — a converging-input hub. A target (T-prefix) gene with high degree means many routes converge on the same downstream effector — a converging-output hub. Either pattern points to a small number of molecular focal points carrying the disease signal in this snapshot.",
        "EM (E-prefix) genes with very high degree should be interpreted with one caveat: the Incytr effector database is densely curated for some EM genes, so their connectivity partly reflects how many curated substrate links exist rather than how biologically central they are at this disease moment. Cross-check EM hubs against the Pathway tab.",
        "Stepping through the three timepoints of one genotype shows whether the network is structurally stable (the same hubs appear at each step) or stage-shifting (hubs appear, change, or disappear across 2mo → 4mo → 6mo). Switching genotypes asks whether App, Tau, and ApTt converge on the same molecular focal points or address disease through different routes.",
      ],
    },
    howTo: "Pick a genotype, then step the timepoint with the dropdown or with ← / → on the keyboard. Each step is a clean rebuild of the network for that single contrast. Click any node to focus its closed neighborhood — its direct neighbors stay coloured, everything else fades. Click empty space to clear the focus. The detail panel on the right shows how many passing chains the selected node sits in and the up-versus-down breakdown of those chains. Adjust min-degree to thin sparsely-connected genes, raise |TPDS| to drop weak-signal edges, or set Max edges as a hard rendering cap when a snapshot stalls. Layout choice is a presentation control, not an analytic one: concentric forces R → EM → T into rings, flow snaps the same three roles into columns, force-directed lets the network find its own layout based on edge counts.",
    conclusions: [
      "Each snapshot answers one question — which genes are tied together by passing-both-nulls routes in this genotype at this timepoint, and in what direction. Stepping the timepoint shows trajectory; switching the genotype tests whether the disease arc has a shared molecular substrate. Convergent routes sharing a receptor suggest a common incoming signal; convergent routes sharing a target suggest a common downstream effector; scattered routes with no shared hubs suggest broad remodeling without a single focal point. Empty graphs (notably Tau_4mo and Tau_6mo) are not a failure of the test — they are the predicted consequence of broad kinase activation overwhelming the per-chain over-representation signal, the same dynamic visible in the Signal Map and Sender × Receiver tabs.",
    ],
    toggles: [
      { name: "Genotype", desc: "selects which disease model's network is rendered: App, Tau, or ApTt (App-Tau double knock-in)." },
      { name: "Timepoint", desc: "selects which timepoint of the chosen genotype is rendered: 2mo, 4mo, or 6mo. Stepping forward or backward triggers a clean rebuild for the new snapshot." },
      { name: "Layout", desc: "presentation only; concentric arranges nodes in R → EM → T rings, flow snaps the three roles into columns, force-directed runs an unconstrained spring layout." },
      { name: "Min degree", desc: "drops nodes connected to fewer than this many passing routes. Raise to thin out genes that participate in only a few chains and emphasize convergence hubs." },
      { name: "|TPDS| ≥", desc: "hides edges whose mean |TPDS| falls below this value. Default zero shows everything in the passing-both-nulls universe; drag up to reveal only the strongest-signal edges." },
      { name: "Max edges", desc: "optional rendering cap. When set to a number, keeps only the top N edges by |TPDS|. Leave blank for no cap; flip on if a snapshot stalls or exceeds what the layout can handle clearly." },
      { name: "Arrow keys", desc: "← and → step the timepoint within the current genotype. Switch genotype with the dropdown." },
    ],
  },
  methods: {
    preamble: "This panel contains the long-form methods documentation: pipeline stages, statistical model specifications, metric definitions, and integration design decisions. It is a reference companion to the analytical tabs, not an analytical view itself.",
    purpose: "Long-form methods reference: pipeline stages, statistical models, and metric definitions in full detail.",
    primary: "Start with the Key viewer concepts and Stage 6 Incytr integration sections when a term in another tab needs more context. Stage 7 covers cross-pair aggregation and the backbone permutation tests.",
  },
};

function _escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function _auditManifest() {
  return (PAYLOAD.audit_tables && PAYLOAD.audit_tables.tables) || {};
}

function _measurementTraceManifest() {
  return (PAYLOAD.audit_tables && PAYLOAD.audit_tables.measurement_trace) || {};
}

function _isLikelyNumericColumn(col) {
  const c = String(col).toLowerCase();
  return /(^n_|_n$|nes|^es$|fdr|p-value|pval|lfc|score|fold|value|_sn_mean$|site_id)/.test(c);
}

const MEA_PREP_COL_DEFS = {
  site_id: {label:"Site ID", definition:"Stable phosphosite identifier used to join site matrices and model outputs.", format:"integer"},
  gene_symbol: {label:"Gene", definition:"Gene symbol associated with the phosphosite.", format:"text"},
  motif: {label:"Motif", definition:"Peptide motif centered on the phosphorylated residue.", format:"text"},
  n_obs_stoich: {label:"N obs", definition:"Number of biological samples with usable stoichiometry for this site (site-level availability count).", format:"integer"},
  raw_lfc: {label:"Raw LFC", definition:"Site-level stoichiometry log fold change for the selected contrast (site_level_ols.stoich_lfc_<contrast>).", format:"float"},
  centered_lfc: {label:"Centered LFC", definition:"raw_lfc minus the contrast's median shift. Derived at view time.", format:"float"},
  clipped_lfc: {label:"Clipped LFC", definition:"centered_lfc clipped to the contrast's winsor bounds; the value passed to GSEA prerank. Derived at view time.", format:"float"},
  was_winsorized: {label:"Winsorized?", definition:"True when the centered LFC was clipped to the bounds.", format:"text"},
  rank_in_contrast: {label:"Rank", definition:"Descending rank of clipped_lfc across all ranked sites for the contrast (1 = most up-shifted; recomputed at view time).", format:"integer"},
  in_leading_edge: {label:"Leading edge?", definition:"Annotation from MEA output: true when the site's motif appears in this kinase's Leading substrates for the contrast.", format:"text"},
};

const MEA_CMP_COL_DEFS = {
  metric: {label:"Metric", definition:"MEA output metric being compared between tracks.", format:"text"},
  stoich: {label:"Stoichiometry (primary)", definition:"Value from mea_stoichiometry.csv for the selected kinase × contrast.", format:"text"},
  raw: {label:"Raw phospho (sensitivity)", definition:"Value from mea_raw_phospho.csv for the selected kinase × contrast. Empty rows mean the kinase has no row in the raw track for this contrast.", format:"text"},
  delta: {label:"Δ (stoich − raw)", definition:"Signed difference, stoichiometry minus raw. — for non-numeric metrics.", format:"text"},
};

function _auditColMeta(tableKey, raw) {
  if (tableKey === "mea_input_derived" && MEA_PREP_COL_DEFS[raw]) {
    return {raw, ...MEA_PREP_COL_DEFS[raw]};
  }
  if (tableKey === "mea_track_comparison" && MEA_CMP_COL_DEFS[raw]) {
    return {raw, ...MEA_CMP_COL_DEFS[raw]};
  }
  const t = tableKey === "measurement_trace" ? _measurementTraceManifest() : (_auditManifest()[tableKey] || {});
  const cols = t.columns || [];
  return cols.find(c => c.raw === raw) || {
    raw, label: raw, definition: "Source column " + raw + ".",
    format: _isLikelyNumericColumn(raw) ? "float" : "text",
  };
}

function _auditHeaderHtml(tableKey, raw) {
  const m = _auditColMeta(tableKey, raw);
  const tip = `Display label: ${m.label}\nRaw column: ${m.raw}\nDefinition: ${m.definition}`;
  return `<th title="${_escapeHtml(tip)}" aria-label="${_escapeHtml(tip)}" data-raw="${_escapeHtml(raw)}">${_escapeHtml(m.label)}</th>`;
}

function _formatAuditValue(v, col) {
  if (v == null || v === "") return "";
  if (_isLikelyNumericColumn(col)) {
    const n = Number(v);
    if (Number.isFinite(n)) {
      if (Number.isInteger(n) && Math.abs(n) < 100000) return String(n);
      return Math.abs(n) >= 1000 ? n.toFixed(1) : n.toPrecision(4);
    }
  }
  const s = String(v);
  return s.length > 90 ? s.slice(0, 87) + "..." : s;
}

function _parseCsv(text) {
  const rows = [];
  let row = [], cur = "", inQ = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i], nx = text[i + 1];
    if (inQ) {
      if (ch === '"' && nx === '"') { cur += '"'; i++; }
      else if (ch === '"') inQ = false;
      else cur += ch;
    } else {
      if (ch === '"') inQ = true;
      else if (ch === ",") { row.push(cur); cur = ""; }
      else if (ch === "\n") { row.push(cur); rows.push(row); row = []; cur = ""; }
      else if (ch !== "\r") cur += ch;
    }
  }
  if (cur.length || row.length) { row.push(cur); rows.push(row); }
  if (!rows.length) return [];
  const header = rows.shift();
  return rows.filter(r => r.some(v => v !== "")).map(r => {
    const obj = {};
    header.forEach((h, i) => { obj[h] = r[i] == null ? "" : r[i]; });
    return obj;
  });
}

const AuditDataStore = (() => {
  const cache = new Map();
  const fileMode = location.protocol === "file:";
  async function load(tableKey) {
    if (cache.has(tableKey)) return cache.get(tableKey);
    const meta = _auditManifest()[tableKey];
    if (!meta) throw new Error("Unknown audit table: " + tableKey);
    if (fileMode || !meta.relative_path) {
      const preview = meta.preview || [];
      cache.set(tableKey, preview);
      return preview;
    }
    const resp = await fetch(meta.relative_path);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} loading ${meta.relative_path}`);
    const text = await resp.text();
    let rows;
    if (meta.type === "json") {
      const obj = JSON.parse(text);
      rows = Array.isArray(obj) ? obj : Object.entries(obj).map(([key, value]) => ({key, value: JSON.stringify(value)}));
    } else {
      rows = _parseCsv(text);
    }
    cache.set(tableKey, rows);
    return rows;
  }
  return { load, fileMode };
})();

const MeasurementTraceStore = (() => {
  const cache = new Map();
  // Track-aware lookup: ST kinases pull from manifest.sample_files (legacy);
  // pY kinases pull from manifest.tracks.Y.sample_files (per-track sidecars).
  async function load(sample, residueType) {
    const manifest = _measurementTraceManifest();
    const tracks = manifest.tracks || {};
    const block = (residueType && tracks[residueType]) || tracks.ST || manifest;
    const files = block.sample_files || {};
    const key = (residueType || "ST") + "|" + sample;
    if (!files[sample]) {
      if (AuditDataStore.fileMode) return block.preview || manifest.preview || [];
      throw new Error("No measurement trace source for sample: " + sample);
    }
    if (cache.has(key)) return cache.get(key);
    if (AuditDataStore.fileMode) {
      const preview = block.preview || manifest.preview || [];
      cache.set(key, preview);
      return preview;
    }
    const resp = await fetch(files[sample]);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} loading ${files[sample]}`);
    const rows = _parseCsv(await resp.text());
    cache.set(key, rows);
    return rows;
  }
  return { load };
})();

class AuditTable {
  constructor(hostId, opts) {
    this.host = document.getElementById(hostId);
    this.tableKey = opts.tableKey || "adhoc";
    this.columns = opts.columns || null;
    this.rows = opts.rows || [];
    this.pageSize = opts.pageSize || 20;
    this.page = 0;
    this.query = "";
    this.sortCol = null;
    this.sortAsc = true;
    this.title = opts.title || "";
    this.fullSourceKey = opts.fullSourceKey === false ? null : (opts.fullSourceKey || this.tableKey);
  }
  setRows(rows, columns) {
    this.rows = rows || [];
    if (columns) this.columns = columns;
    this.page = 0;
    this.render();
  }
  visibleColumns() {
    if (this.columns && this.columns.length) return this.columns;
    return Object.keys(this.rows[0] || {});
  }
  filteredRows() {
    const q = this.query.trim().toLowerCase();
    let rows = this.rows;
    if (q) rows = rows.filter(r => Object.values(r).some(v => String(v ?? "").toLowerCase().includes(q)));
    if (this.sortCol) {
      const c = this.sortCol, asc = this.sortAsc;
      rows = rows.slice().sort((a, b) => {
        const an = Number(a[c]), bn = Number(b[c]);
        let cmp = Number.isFinite(an) && Number.isFinite(bn)
          ? an - bn : String(a[c] ?? "").localeCompare(String(b[c] ?? ""));
        return asc ? cmp : -cmp;
      });
    }
    return rows;
  }
  exportRows(rows, cleanHeaders) {
    const cols = this.visibleColumns();
    const headers = cleanHeaders ? cols.map(c => _auditColMeta(this.tableKey, c).label) : cols;
    const esc = v => {
      const s = String(v == null ? "" : v);
      return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    return [headers.map(esc).join(",")].concat(rows.map(r => cols.map(c => esc(r[c])).join(","))).join("\n");
  }
  downloadCsv(rows, label, cleanHeaders) {
    const blob = new Blob([this.exportRows(rows, cleanHeaders)], {type:"text/csv"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = label + ".csv";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  render() {
    if (!this.host) return;
    const cols = this.visibleColumns();
    const rows = this.filteredRows();
    const pages = Math.max(1, Math.ceil(rows.length / this.pageSize));
    if (this.page >= pages) this.page = pages - 1;
    const start = this.page * this.pageSize;
    const pageRows = rows.slice(start, start + this.pageSize);
    const cleanId = `${this.host.id}-clean`;
    const fullButton = this.fullSourceKey ? `<button data-action="export-full">Export full source</button>` : "";
    const body = pageRows.map(r => `<tr>${cols.map(c => {
      const cls = _isLikelyNumericColumn(c) ? ' class="numeric-cell"' : "";
      const raw = r[c] == null ? "" : String(r[c]);
      return `<td${cls} title="${_escapeHtml(raw)}">${_escapeHtml(_formatAuditValue(raw, c))}</td>`;
    }).join("")}</tr>`).join("");
    const fileNotice = AuditDataStore.fileMode
      ? '<div class="notice show">Full audit table loading requires serving outputs/reports/unified_viewer/ over HTTP. Showing embedded previews and selected in-payload data.</div>'
      : "";
    this.host.innerHTML =
      `${fileNotice}<div class="audit-controls">` +
      `<input type="search" placeholder="Search rows" aria-label="Search ${_escapeHtml(this.title || this.tableKey)}">` +
      `<button data-action="export-filtered">Export filtered</button>` +
      fullButton +
      `<label><input type="checkbox" id="${cleanId}"> Clean headers</label>` +
      `<span class="muted">${rows.length.toLocaleString()} rows</span></div>` +
      `<div class="audit-table-wrap"><table class="data-table"><thead><tr>${cols.map(c => _auditHeaderHtml(this.tableKey, c)).join("")}</tr></thead><tbody>${body}</tbody></table></div>` +
      `<div class="audit-pager"><button data-action="prev"${this.page === 0 ? " disabled" : ""}>Prev</button>` +
      `<span>${rows.length ? start + 1 : 0}-${Math.min(start + this.pageSize, rows.length)} of ${rows.length}</span>` +
      `<button data-action="next"${this.page >= pages - 1 ? " disabled" : ""}>Next</button></div>`;
    const search = this.host.querySelector('input[type="search"]');
    search.value = this.query;
    search.addEventListener("input", ev => { this.query = ev.target.value; this.page = 0; this.render(); });
    this.host.querySelectorAll("th").forEach(th => th.addEventListener("click", () => {
      const c = th.dataset.raw;
      if (this.sortCol === c) this.sortAsc = !this.sortAsc;
      else { this.sortCol = c; this.sortAsc = true; }
      this.render();
    }));
    this.host.querySelector('[data-action="prev"]').addEventListener("click", () => { this.page--; this.render(); });
    this.host.querySelector('[data-action="next"]').addEventListener("click", () => { this.page++; this.render(); });
    this.host.querySelector('[data-action="export-filtered"]').addEventListener("click", () => {
      this.downloadCsv(rows, `${this.tableKey}_filtered`, document.getElementById(cleanId).checked);
    });
    const fullBtn = this.host.querySelector('[data-action="export-full"]');
    if (fullBtn) fullBtn.addEventListener("click", async () => {
      const full = await AuditDataStore.load(this.fullSourceKey);
      this.downloadCsv(full, `${this.fullSourceKey}_full`, document.getElementById(cleanId).checked);
    });
  }
}

