"use strict";

// ---------------------------------------------------------------------------
// Payload — fetched lazily from payload.json.gz at boot. PAYLOAD and the
// derived globals start as `null` / `undefined` and are populated by
// `_loadPayload()` (called from boot.js) before any tab is rendered. All
// downstream JS that references PAYLOAD does so inside function bodies, so
// they read the populated globals at call time.
// ---------------------------------------------------------------------------
let PAYLOAD = null;
let META = null;
let CONTRASTS = null;
let RECEIVERS = null;
let DISEASE_COLORS = null;

async function _loadPayload() {
  // 1) Inline payload (default for self-contained HTML): read from the
  //    <script type="application/json" id="payload-data"> tag the build emits.
  // 2) If the inline payload is empty/null/missing, fall back to fetching
  //    the .json.gz / .json sidecars (kept for hosting modes where the build
  //    intentionally writes a small index.html + separate payload file).
  let text = null;
  const inlineEl = document.getElementById("payload-data");
  if (inlineEl && inlineEl.textContent && inlineEl.textContent.trim() !== "null"
      && inlineEl.textContent.trim() !== "") {
    text = inlineEl.textContent;
  }
  let gzErr = null;
  if (text === null) {
    try {
      const resp = await fetch("unified_viewer.payload.json.gz");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();
      const stream = blob.stream().pipeThrough(new DecompressionStream("gzip"));
      text = await new Response(stream).text();
    } catch (e) {
      gzErr = e;
    }
  }
  if (text === null) {
    const resp2 = await fetch("unified_viewer.payload.json");
    if (!resp2.ok) {
      throw new Error(
        `payload fetch failed (gzip: ${gzErr && gzErr.message ? gzErr.message : gzErr}; `
        + `json: HTTP ${resp2.status})`);
    }
    text = await resp2.text();
  }
  PAYLOAD = JSON.parse(text);
  META = PAYLOAD.meta;
  CONTRASTS = META.contrasts;
  RECEIVERS = ViewerPayload.celltypes().name;
  DISEASE_COLORS = META.diseaseColors;
  HAS_HUMAN = !!(PAYLOAD && PAYLOAD.human);
  // Populate human-tab cached refs (declared in kinase_human.js).
  if (typeof _KH_HAS !== "undefined") {
    _KH_HAS = HAS_HUMAN;
    _KH = HAS_HUMAN ? PAYLOAD.human : null;
  }
}

// ---------------------------------------------------------------------------
// Store — reducer-style with {selection, filters, view} slices
// ---------------------------------------------------------------------------
// Populated after _loadPayload() resolves.
let HAS_HUMAN = false;
const INITIAL_STATE = {
  selection: { kinase:null, backbone:null, celltype:null, kinaseHuman:null,
               context:"song_ad" },
  filters:   { contrast:"ALL", fdr:0.25 },
  view:      { mode:"mouse", activeTab:"kinase", glossaryOpen:false,
               kinaseAuditTab:"measurement-trace",
               temporalLevel:"kinase" },
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
