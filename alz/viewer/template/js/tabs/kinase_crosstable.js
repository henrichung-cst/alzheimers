// ---------------------------------------------------------------------------
// Kinase Crosstable — cross-dataset agreement view (master/detail).
//
// MASTER (left): a slim, agreement-grouped list of kinases — one row per kinase
// with the mouse 3×3 NES glyph, the agreement badge, the human per-donor strip,
// and mouse/human specificity tiers at the selected cluster. The per-sample NES
// detail lives in the DETAIL panel, not as wide columns.
//
// DETAIL (right, #kx-detail): a cross-dataset comparison of the selected kinase,
// two columns (Mouse · Song | Human · Mukesh) under a verdict header, with two
// sub-tabs that REUSE the per-dataset Kinase-tab renderers verbatim:
//   Activity      — _renderKinaseNesPlot (mouse) | _khRenderNESAcrossDonors (human)
//   Specificity   — _renderKinaseCelltypeEvidence (mouse) | _khRenderAttribution (human)
//
// Indexes are joined client-side from existing PAYLOAD keys (see _kxBuildIndexes).
// The cross-dataset join key is the kinase abbreviation + residue (species-neutral;
// no ortholog mapping). Selection is LOCAL to this tab (does not drive the other
// tabs' selection). Agreement is recomputed live against the FDR slider.
// ---------------------------------------------------------------------------

let _KX_ROWS = null;
let _KX_HUMAN_BY_NAME = null;
let _KX_HUMAN_PERDONOR = null;    // Map<kid, Map<donor, {nes, fdr}>>
let _KX_HUMAN_SPEC_BY_NAME = null;
let _KX_WMB_BY_KIN_CLUSTER = null; // Map<`kid|cluster`, wmb_specificity>
let _KX_WMB_MAX_BY_KID = null;     // Map<kid, {frac, cluster}> — peak WMB specificity (for "Any")
let _KX_SONG_BY_KID = null;        // Map<kid, {tau, topCluster, topShare}> — Song location specificity (per kinase)
let _KX_SEAAD_MAX_BY_NAME = null;  // Map<name, {score, cluster}> — peak SEA-AD specificity (for "Any")
let _KX_DECOMP_BY_KIN_CTX = null;
let _KX_CLUSTERS = null;
let _KX_AD_DONORS = [];
let _KX_CTRL_DONORS = [];
let _KX_INITIALIZED = false;

// Song location-specificity tau bands (the primary mouse signal): tau is the
// Yanai tissue-specificity index over the 31 spine clusters, 0 = expressed
// evenly → 1 = confined to one cell type. Unlike a peak/even-split ratio it is
// normalized for cluster count, so housekeeping kinases read low at any
// resolution. See docs/plans/specificity_validation_2026-06-05.md §6.
const _KX_SONG_TAU_HIGH = 0.85;   // "highly specific to one cell type"
const _KX_SONG_TAU_SPEC = 0.60;   // "specific"

// Uniform baseline for the WMB cross-check tier. WMB specificity is a share
// normalized over the retained WMB classes the spine maps onto (~11), so the
// honest "even split" is 1/N_retained, emitted canonically as meta.wmb_uniform.
let _KX_WMB_UNIFORM = (PAYLOAD && PAYLOAD.meta && PAYLOAD.meta.wmb_uniform) || (1 / 11);

function _kxState() {
  if (!Store.state.view.crosstable) {
    Store.state.view.crosstable = {
      cluster: "",
      residueTrack: "",
      search: "",
      sortKey: "agree_score",
      sortDir: -1,
      mSpecMin: 0,            // minimum mouse WMB specificity tier (0=any,1,2,5,10 × uniform)
      hSpecMin: 0,            // minimum human SEA-AD specificity tier (0=any,1,2,5,10 ×)
      allSamples: false,      // median + agreement over ALL measured units, not just sig
      agreeCat: "",           // agreement category to show ("" = any), via the toolbar dropdown
      selectedKey: null,      // `name|residue` of the kinase shown in the detail panel
      detailTab: "activity",  // "activity" | "specificity"
    };
  }
  return Store.state.view.crosstable;
}

function _kxBuildIndexes() {
  if (_KX_INITIALIZED) return;
  const K = ViewerPayload.kinases();
  const famMap = (META && META.familyMap) || {};

  _KX_ROWS = [];
  for (let i = 0; i < K.id.length; i++) {
    const nesByC = {}; const fdrByC = {};
    for (const c of CONTRASTS) {
      nesByC[c] = K["NES_" + c] ? K["NES_" + c][i] : null;
      fdrByC[c] = K["FDR_" + c] ? K["FDR_" + c][i] : null;
    }
    _KX_ROWS.push({
      kid: K.id[i],
      name: K.name[i],
      gene: K.gene_symbol[i] || "",
      family: famMap[K.name[i]] || "",
      residue: (K.residue_type && K.residue_type[i]) || "ST",
      peak_NES: K.peak_NES[i],
      trajectory: K.trajectory ? (K.trajectory[i] || "") : "",
      n_sig_contrasts: K.n_sig_contrasts ? (K.n_sig_contrasts[i] || 0) : 0,
      n_celltype_candidates: K.n_celltype_candidates ? (K.n_celltype_candidates[i] || 0) : 0,
      _nes: nesByC, _fdr: fdrByC,
    });
  }

  _KX_DECOMP_BY_KIN_CTX = new Map();
  const D = PAYLOAD.decomposition_index || {kinase_id: []};
  for (let k = 0; k < D.kinase_id.length; k++) {
    const ctName = CONTRASTS[D.contrast_id[k]];
    if (!ctName) continue;
    const key = `${D.kinase_id[k]}|${ctName}|${D.cell_type[k]}`;
    _KX_DECOMP_BY_KIN_CTX.set(key, {nes: D.decomp_nes[k], fdr: D.decomp_fdr[k]});
  }
  const decompClusters = new Set();
  for (let k = 0; k < D.cell_type.length; k++) decompClusters.add(D.cell_type[k]);

  _KX_WMB_BY_KIN_CLUSTER = new Map();
  _KX_WMB_MAX_BY_KID = new Map();
  _KX_SONG_BY_KID = new Map();
  const A = PAYLOAD.attribution_index;
  if (A && A.kinase_id) {
    for (let i = 0; i < A.kinase_id.length; i++) {
      const kid = A.kinase_id[i], ct = A.cell_type[i];
      const v = A.wmb_specificity ? A.wmb_specificity[i] : null;
      if (v != null && isFinite(v)) {
        const key = `${kid}|${ct}`;
        // attribution_index can carry multiple rows per (kinase, cluster) across
        // contrasts/confidence strata; keep the max specificity seen (it's a
        // cluster-level property, not contrast-level — picks the populated row).
        const prev = _KX_WMB_BY_KIN_CLUSTER.get(key);
        if (prev == null || v > prev) _KX_WMB_BY_KIN_CLUSTER.set(key, v);
        // Peak specificity across clusters, for the "Any cell type" pivot.
        const mx = _KX_WMB_MAX_BY_KID.get(kid);
        if (!mx || v > mx.frac) _KX_WMB_MAX_BY_KID.set(kid, {frac: v, cluster: ct});
      }
      // Song location specificity is per-kinase (tau + the cell type it
      // concentrates in), constant across the rows for a kid — stamp once.
      if (!_KX_SONG_BY_KID.has(kid) && A.song_tau) {
        const t = A.song_tau[i];
        if (t != null && isFinite(t)) {
          _KX_SONG_BY_KID.set(kid, {
            tau: t,
            topCluster: (A.song_top_cluster && A.song_top_cluster[i]) || "",
            topShare: (A.song_top_share && isFinite(A.song_top_share[i])) ? A.song_top_share[i] : null,
          });
        }
      }
    }
  }

  _KX_HUMAN_BY_NAME = new Map();
  _KX_HUMAN_PERDONOR = new Map();
  _KX_HUMAN_SPEC_BY_NAME = new Map();
  _KX_SEAAD_MAX_BY_NAME = new Map();
  const specClusters = new Set();
  const H = PAYLOAD.human;
  _KX_AD_DONORS = [];
  _KX_CTRL_DONORS = [];
  if (H) {
    _KX_AD_DONORS = Array.isArray(H.donors) ? H.donors.slice() : [];
    _KX_CTRL_DONORS = Array.isArray(H.ctrl_donors) ? H.ctrl_donors.slice() : [];
  }
  if (H && H.kinases) {
    const HK = H.kinases;
    for (let i = 0; i < HK.id.length; i++) {
      const name = HK.name[i];
      const residue = HK.residue_type[i] || "ST";
      const key = `${name}|${residue}`;
      _KX_HUMAN_BY_NAME.set(key, {
        kid: HK.id[i],
        median_nes: HK.median_nes_sig_only ? HK.median_nes_sig_only[i] : null,
        n_sig: HK.n_donors_sig ? HK.n_donors_sig[i] : null,
        n_up: HK.n_donors_up ? HK.n_donors_up[i] : null,
        n_down: HK.n_donors_down ? HK.n_donors_down[i] : null,
        n_tested: HK.n_donors_tested ? HK.n_donors_tested[i] : null,
        sea_ad_lfc: HK.sea_ad_lfc ? HK.sea_ad_lfc[i] : null,
        sea_ad_n: HK.sea_ad_n_supertypes ? HK.sea_ad_n_supertypes[i] : null,
      });
    }
  }
  if (H && H.perdonor_index) {
    const PD = H.perdonor_index;
    for (let i = 0; i < PD.kinase_id.length; i++) {
      const kid = PD.kinase_id[i];
      let m = _KX_HUMAN_PERDONOR.get(kid);
      if (!m) { m = new Map(); _KX_HUMAN_PERDONOR.set(kid, m); }
      m.set(PD.donor[i], {nes: PD.NES[i], fdr: PD.FDR[i]});
    }
  }
  if (H && H.celltype_specificity) {
    const CS = H.celltype_specificity;
    const seaad = CS.seaad_mtg && CS.seaad_mtg.ranked_by_kinase;
    const hbca = CS.allen_hbca && CS.allen_hbca.ranked_by_kinase;
    const allNames = new Set([
      ...(seaad ? Object.keys(seaad) : []),
      ...(hbca ? Object.keys(hbca) : []),
    ]);
    for (const name of allNames) {
      const entry = {seaad: new Map(), hbca: new Map()};
      if (seaad && seaad[name]) {
        for (const r of seaad[name]) {
          entry.seaad.set(r.celltype, {score: r.score, lfc: r.sea_ad_lfc, expr: r.mean_log2_expression});
          specClusters.add(r.celltype);
          if (r.score != null && isFinite(r.score)) {
            const mx = _KX_SEAAD_MAX_BY_NAME.get(name);
            if (!mx || r.score > mx.score) _KX_SEAAD_MAX_BY_NAME.set(name, {score: r.score, cluster: r.celltype});
          }
        }
      }
      if (hbca && hbca[name]) {
        for (const r of hbca[name]) {
          entry.hbca.set(r.celltype, {score: r.score, expr: r.mean_log2_expression});
          specClusters.add(r.celltype);
        }
      }
      _KX_HUMAN_SPEC_BY_NAME.set(name, entry);
    }
  }

  const clusters = Array.from(decompClusters);
  for (const c of specClusters) if (!decompClusters.has(c)) clusters.push(c);
  clusters.sort();
  _KX_CLUSTERS = clusters;

  for (const r of _KX_ROWS) r._human = _KX_HUMAN_BY_NAME.get(`${r.name}|${r.residue}`) || null;

  // Union: append human-only kinases (measured in Mukesh but not in the mouse
  // MEA) so the agreement view can categorize them. Mouse columns render dashes;
  // these rows are flagged _humanOnly so the detail shows a mouse-side placeholder.
  const seen = new Set(_KX_ROWS.map(r => `${r.name}|${r.residue}`));
  for (const [key, human] of _KX_HUMAN_BY_NAME) {
    if (seen.has(key)) continue;
    const bar = key.lastIndexOf("|");
    const name = key.slice(0, bar);
    const residue = key.slice(bar + 1);
    _KX_ROWS.push({
      kid: human.kid, name, gene: "", family: famMap[name] || "", residue,
      peak_NES: null, trajectory: "", n_sig_contrasts: 0, n_celltype_candidates: 0,
      _nes: {}, _fdr: {}, _human: human, _humanOnly: true,
    });
  }

  // Default pivot is "" = Any cell type (peak specificity across clusters).
  _KX_INITIALIZED = true;
}

// ---------- specificity tier badges ----------

// Log2 tier reuses _khSpecTierRank/_khSpecTierBadge from kinase_human.js (same
// thresholds, same CSS classes); WMB tier stays local because its uniform
// baseline depends on the active levy_t5 spine size.
function _kxLog2TierBadge(score, atCluster) {
  if (score == null || !isFinite(score)) return `<td class="muted">–</td>`;
  const at = atCluster ? ` @ ${atCluster}` : "";
  const rank = _khSpecTierRank(score);
  if (!rank) return `<td class="muted" title="log2 ${score.toFixed(3)}${_escapeHtml(at)}">&lt;1×</td>`;
  return `<td style="text-align:center;padding:2px 4px;" title="log2 ${score.toFixed(3)}${_escapeHtml(at)}">${_khSpecTierBadge(rank)}</td>`;
}
// Primary mouse specificity: Song tau (per kinase). `pinned` is the active
// cluster pivot (or "") — when a cluster is pinned we flag whether it IS the
// kinase's top cell type, but the tier itself is the per-kinase tau (it does
// not vary by cluster the way the WMB share does).
function _kxSongTierBadge(song, pinned) {
  if (!song || song.tau == null || !isFinite(song.tau)) return `<td class="muted">–</td>`;
  const tau = song.tau;
  const top = song.topCluster || "?";
  const pct = song.topShare != null ? `${(song.topShare * 100).toFixed(0)}% in ${top}` : top;
  let cls, label;
  if (tau >= _KX_SONG_TAU_HIGH)      { cls = "badge vhi"; label = "highly specific"; }
  else if (tau >= _KX_SONG_TAU_SPEC) { cls = "badge hi";  label = "specific"; }
  else { return `<td class="muted" title="τ ${tau.toFixed(2)} · ${_escapeHtml(pct)}">broad</td>`; }
  let mark = "";
  if (pinned) mark = (pinned === top)
    ? ` <span title="this pinned cell type IS where it concentrates">★</span>`
    : ` <span class="muted" title="concentrates in ${_escapeHtml(top)}, not the pinned ${_escapeHtml(pinned)}">·</span>`;
  return `<td style="text-align:center;padding:2px 4px;"><span class="${cls}" title="Song location τ ${tau.toFixed(2)} · ${_escapeHtml(pct)}">${label}</span>${mark}</td>`;
}
function _kxWmbTierBadge(frac, atCluster) {
  if (frac == null || !isFinite(frac)) return `<td class="muted">–</td>`;
  const u = _KX_WMB_UNIFORM;
  const at = atCluster ? ` @ ${atCluster}` : "";
  let cls, label;
  if (frac >= 10 * u) { cls = "badge vhi"; label = "≥10×"; }
  else if (frac >= 5 * u) { cls = "badge hi"; label = "≥5×"; }
  else if (frac >= 2 * u) { cls = "badge mid"; label = "≥2×"; }
  else if (frac >= u)     { cls = "badge lo"; label = "≥1×"; }
  else                    { return `<td class="muted" title="${frac.toFixed(4)} vs uniform ${u.toFixed(4)}${_escapeHtml(at)}">&lt;1×</td>`; }
  return `<td style="text-align:center;padding:2px 4px;"><span class="${cls}" title="WMB specificity ${frac.toFixed(4)} vs uniform ${u.toFixed(4)}${_escapeHtml(at)}">${label}</span></td>`;
}

// ---------- cross-dataset agreement model ----------

// Labels + badge styling for the agreement categories (toolbar dropdown + row badge).
const _KX_AGREE_META = {
  "concordant-up":   {cls: "badge vhi", glyph: "↑ agree",   label: "Concordant — up",   tip: "Significant in BOTH datasets, both up in disease."},
  "concordant-down": {cls: "badge hi",  glyph: "↓ agree",   label: "Concordant — down", tip: "Significant in BOTH datasets, both down in disease."},
  "discordant":      {cls: "badge mix", glyph: "✕ discord", label: "Discordant",        tip: "Significant in both datasets but opposite direction."},
  "mouse-only":      {cls: "badge mid", glyph: "M only",    label: "Mouse-only",        tip: "Significant in mouse (Song) only."},
  "human-only":      {cls: "badge imp", glyph: "H only",    label: "Human-only",        tip: "Significant in human (Mukesh) only."},
  "neither":         {cls: "badge lo",  glyph: "–",         label: "Neither",           tip: "Not significant in either dataset at this FDR."},
};

function _kxMedian(arr) {
  if (!arr.length) return null;
  const a = arr.slice().sort((x, y) => x - y);
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

// Stamp _mouseSig / _humanSig / _agreeCategory / _agreeScore on each row, live
// against the FDR gate. BOTH summary NES are the median over the dataset's units
// (mouse contrasts / human AD donors) that are significant at the gate — the same
// statistic on both sides, so the magnitudes and concordance signs are comparable
// (peak vs median would bias the mouse magnitude upward). Significance and sign
// are both derived from the SAME live threshold.
// allSamples = include EVERY measured unit in the median (and treat "measured"
// as the inclusion criterion for the category), instead of only units FDR-
// significant at the gate. Lets the user see the all-sample direction/agreement.
function _kxComputeAgreement(rows, fdrGate, allSamples) {
  for (const r of rows) {
    let mouseSig = false, mNes = null;
    const mSigNes = [];
    for (const c of (CONTRASTS || [])) {
      const nes = r._nes[c];
      if (nes == null || !isFinite(nes)) continue;
      const f = r._fdr[c];
      if (allSamples || (f != null && isFinite(f) && f < fdrGate)) mSigNes.push(nes);
    }
    if (mSigNes.length) { mouseSig = true; mNes = _kxMedian(mSigNes); }

    let humanSig = false, hNes = null;
    const pd = r._human ? _KX_HUMAN_PERDONOR.get(r._human.kid) : null;
    if (pd) {
      const sigNes = [];
      for (const d of _KX_AD_DONORS) {
        const e = pd.get(d);
        if (!e || e.nes == null || !isFinite(e.nes)) continue;
        if (allSamples || (e.fdr != null && isFinite(e.fdr) && e.fdr < fdrGate)) sigNes.push(e.nes);
      }
      if (sigNes.length) { humanSig = true; hNes = _kxMedian(sigNes); }
    }

    r._mouseSig = mouseSig; r._humanSig = humanSig; r._mNes = mNes; r._hNes = hNes;
    let cat, score;
    if (mouseSig && humanSig) {
      if (mNes != null && hNes != null && mNes > 0 && hNes > 0)      { cat = "concordant-up";   score = Math.abs(mNes) * Math.abs(hNes); }
      else if (mNes != null && hNes != null && mNes < 0 && hNes < 0) { cat = "concordant-down"; score = Math.abs(mNes) * Math.abs(hNes); }
      else                                                           { cat = "discordant";      score = -Math.abs(mNes || 0) * Math.abs(hNes || 0); }
    } else if (mouseSig) { cat = "mouse-only"; score = Math.abs(mNes || 0) * 0.1; }
    else if (humanSig)   { cat = "human-only"; score = Math.abs(hNes || 0) * 0.1; }
    else                 { cat = "neither";    score = 0; }
    r._agreeCategory = cat; r._agreeScore = score;
  }
}

// Median-of-significant NES driving the Agree call. "–" when nothing is
// significant on that side at the live gate (so the row's category is *-only or
// neither, not concordant/discordant). Colored by sign, same red/blue as glyphs.
function _kxMedNesCell(val) {
  if (val == null || !isFinite(val)) return `<td class="muted kx-nes-num">–</td>`;
  const col = val >= 0 ? "#c53030" : "#2b6cb0";
  return `<td class="kx-nes-num" style="color:${col};" title="median NES driving the Agree call">${val >= 0 ? "+" : ""}${val.toFixed(2)}</td>`;
}

function _kxAgreeCategoryCell(cat) {
  const m = _KX_AGREE_META[cat] || _KX_AGREE_META["neither"];
  return `<td style="text-align:center;padding:2px 4px;"><span class="${m.cls}" title="${_escapeHtml(m.tip)}">${m.glyph}</span></td>`;
}

// Mouse NES-profile glyph: adapt the crosstable row's contrast-keyed _nes/_fdr
// dicts to the array form (indexed by CONTRASTS) that _renderNesProfile expects,
// reusing the kinase-explorer glyph verbatim. maxAbs=3 matches the score scale.
function _kxMakeNesProfileRow(r) {
  const nesArr = [], fdrArr = [];
  for (let ci = 0; ci < (CONTRASTS || []).length; ci++) {
    const c = CONTRASTS[ci];
    nesArr[ci] = (r._nes && r._nes[c] != null) ? r._nes[c] : null;
    fdrArr[ci] = (r._fdr && r._fdr[c] != null) ? r._fdr[c] : null;
  }
  return {id: r.kid, _nes: nesArr, _fdr: fdrArr};
}
function _kxMouseGlyphCell(r, fdrGate) {
  if (r._humanOnly) return `<td class="kx-mglyph muted" title="Not measured in mouse">–</td>`;
  return `<td class="kx-mglyph">${_renderNesProfile(_kxMakeNesProfileRow(r), fdrGate, 3)}</td>`;
}

// Human per-donor NES strip — same .npc markup + color encoding as
// _khRenderProfile, kept local because that function dereferences the human
// tab's mutable globals (_KH/_KHState), which are null until the human tab
// renders. AD donors first, then a muted CTRL group (for reference — controls
// are scored against the CTRL mean so they bias toward zero by design).
function _kxDonorCells(pd, donorList, fdrGate) {
  const cells = [];
  for (const d of donorList) {
    const e = pd.get(d);
    const nes = e ? e.nes : null;
    const fdrV = e ? e.fdr : null;
    const sig = fdrV != null && isFinite(fdrV) && fdrV < fdrGate;
    let bg = "#fff";
    if (nes != null && isFinite(nes)) {
      const a = Math.min(1, Math.abs(nes) / 3);
      const rgb = nes >= 0 ? [197, 48, 48] : [43, 108, 176];
      bg = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.15 + 0.85 * a).toFixed(3)})`;
    }
    const tip = nes == null ? `${d}: n/a`
      : `${d}: NES ${nes.toFixed(2)}${fdrV != null ? `, FDR ${fdrV.toExponential(1)}` : ""}${sig ? " (sig)" : ""}`;
    cells.push(`<div class="npc${sig ? " sig" : ""}" style="background:${bg};" title="${_escapeHtml(tip)}"></div>`);
  }
  return cells;
}
function _kxHumanGlyphCell(r, fdrGate) {
  const pd = r._human ? _KX_HUMAN_PERDONOR.get(r._human.kid) : null;
  if (!pd || !_KX_AD_DONORS.length) return `<td class="kx-hglyph muted" title="Not measured in human">–</td>`;
  const adCells = _kxDonorCells(pd, _KX_AD_DONORS, fdrGate);
  const adBlock = `<div class="nes-profile-cell" style="grid-template-columns:repeat(${adCells.length || 1},1fr);">${adCells.join("")}</div>`;
  const ctrlCells = _KX_CTRL_DONORS.length ? _kxDonorCells(pd, _KX_CTRL_DONORS, fdrGate) : [];
  const ctrlBlock = ctrlCells.length
    ? `<span class="nes-profile-spacer" aria-hidden="true"></span><div class="nes-profile-cell kh-ctrl-group" style="grid-template-columns:repeat(${ctrlCells.length},1fr);" title="CTRL donors scored against the same CTRL mean — reference only; they bias toward zero by design.">${ctrlCells.join("")}</div>`
    : "";
  return `<td class="kx-hglyph"><div class="nes-profile-wrap">${adBlock}${ctrlBlock}</div></td>`;
}

// ---------- slim master header + row builders ----------

function _kxBuildHeader(s) {
  const TH = (key, label, title, extra) => {
    const sortAttr = key ? ` data-sortkey="${key}"` : "";
    const cls = extra ? ` class="${extra}"` : "";
    return `<th${sortAttr}${cls} title="${_escapeHtml(title)}">${label}</th>`;
  };
  const cells = [];
  cells.push(TH("name", "Kinase", "Kinase identifier from MEA. Click to sort."));
  cells.push(TH("gene", "Gene", "Gene symbol. Click to sort."));
  cells.push(TH("residue", "Res", "Phospho-residue track (ST or Y). Click to sort."));
  cells.push(TH("family", "Family", "Kinase family. Click to sort."));
  cells.push(TH(null, "Mouse", "Mouse (Song) bulk MEA NES across 3 diseases × 3 timepoints (red=up, blue=down; outlined=FDR<header).", "kx-mglyph"));
  cells.push(TH(null, "Human", "Human (Mukesh) per-donor MEA NES: AD donors, then a muted CTRL reference group (red=up, blue=down; outlined=FDR<header).", "kx-hglyph"));
  cells.push(TH("m_med", "M med", "Mouse median NES over the contrasts feeding the Agree call (FDR-significant, or all when 'All samples' is on). – = none. Sort by magnitude.", "kx-nes-num"));
  cells.push(TH("h_med", "H med", "Human median NES over the AD donors feeding the Agree call (FDR-significant, or all when 'All samples' is on). – = none. Sort by magnitude.", "kx-nes-num"));
  cells.push(TH("agree_score", "Agree", "Cross-dataset agreement from M med + H med: both significant at the header FDR AND same direction in disease. Grouped by category; click to sort within groups.", "kx-agree-col"));
  const specScope = s.cluster ? `at ${s.cluster}` : "— peak across all clusters (hover for the cluster)";
  cells.push(TH("m_spec", "M-spec", `Mouse Song location specificity (τ, 0–1): ≥${_KX_SONG_TAU_HIGH} highly specific, ≥${_KX_SONG_TAU_SPEC} specific. One value per kinase; the cell type it concentrates in is in the tooltip. ★ = the pinned cluster IS that cell type.`, "kx-spec"));
  cells.push(TH("wmb", "WMB", `WMB atlas cross-check: independent mouse-brain cell-type specificity tier ${specScope} (× uniform 1/${(1 / _KX_WMB_UNIFORM).toFixed(0)}). Confirms the Song call against an outside atlas.`, "kx-spec"));
  cells.push(TH("h_spec", "H-spec", `Human SEA-AD MTG cell-type specificity tier ${specScope}.`, "kx-spec"));
  return `<thead><tr>${cells.join("")}</tr></thead>`;
}

// Resolve a row's mouse (WMB) + human (SEA-AD) specificity for the active pivot:
// "" = Any cell type → peak across clusters (with the argmax cluster for the
// tooltip); a named cluster → the value at that cluster. Shared by the M-spec /
// H-spec columns AND the M-spec / H-spec minimum-tier filters so they agree.
function _kxResolveSpec(r, cluster) {
  const song = _KX_SONG_BY_KID.get(r.kid) || null;  // per-kinase, pivot-independent
  if (cluster) {
    const w = _KX_WMB_BY_KIN_CLUSTER.get(`${r.kid}|${cluster}`);
    const spec = _KX_HUMAN_SPEC_BY_NAME.get(r.name);
    const seaadCt = spec && spec.seaad.get(cluster);
    return {song, wmb: (w == null ? null : w), wmbAt: null, seaad: seaadCt ? seaadCt.score : null, seaadAt: null};
  }
  const wmx = _KX_WMB_MAX_BY_KID.get(r.kid);
  const smx = _KX_SEAAD_MAX_BY_NAME.get(r.name);
  return {
    song,
    wmb: wmx ? wmx.frac : null, wmbAt: wmx ? wmx.cluster : null,
    seaad: smx ? smx.score : null, seaadAt: smx ? smx.cluster : null,
  };
}

function _kxBuildRow(r, s, fdrGate) {
  const sp = _kxResolveSpec(r, s.cluster);
  const key = `${r.name}|${r.residue}`;

  const tds = [];
  tds.push(`<td>${_escapeHtml(r.name)}</td>`);
  tds.push(`<td>${_escapeHtml(r.gene)}</td>`);
  tds.push(`<td class="muted">${_escapeHtml(r.residue)}</td>`);
  tds.push(`<td class="muted">${_escapeHtml(r.family)}</td>`);
  tds.push(_kxMouseGlyphCell(r, fdrGate));
  tds.push(_kxHumanGlyphCell(r, fdrGate));
  tds.push(_kxMedNesCell(r._mNes));
  tds.push(_kxMedNesCell(r._hNes));
  tds.push(_kxAgreeCategoryCell(r._agreeCategory));
  tds.push(r._humanOnly ? `<td class="muted">–</td>` : _kxSongTierBadge(sp.song, s.cluster));
  tds.push(r._humanOnly ? `<td class="muted">–</td>` : _kxWmbTierBadge(sp.wmb, sp.wmbAt));
  tds.push(_kxLog2TierBadge(sp.seaad, sp.seaadAt));
  const sel = s.selectedKey === key ? " selected" : "";
  return `<tr class="kx-data-row${sel}" data-key="${_escapeHtml(key)}">${tds.join("")}</tr>`;
}

const _KX_STRING_KEYS = new Set(["name", "gene", "residue", "family"]);

function _kxSortRows(rows, s) {
  const cluster = s.cluster;
  if (s.sortKey === "agree_score") {
    // Signed sort: concordant (high +) first, discordant (−) last.
    rows.sort((a, b) => s.sortDir * ((b._agreeScore || 0) - (a._agreeScore || 0)));
    return;
  }
  if (_KX_STRING_KEYS.has(s.sortKey)) {
    const k = s.sortKey;
    rows.sort((a, b) => s.sortDir * (a[k] || "").localeCompare(b[k] || ""));
    return;
  }
  rows.sort((a, b) => {
    let av, bv;
    if (s.sortKey === "m_spec") {
      // Song tau — per kinase, pivot-independent.
      const sa = _KX_SONG_BY_KID.get(a.kid), sb = _KX_SONG_BY_KID.get(b.kid);
      const ta = sa ? sa.tau : null, tb = sb ? sb.tau : null;
      const ax = (ta == null || !isFinite(ta)) ? -Infinity : ta;
      const bx = (tb == null || !isFinite(tb)) ? -Infinity : tb;
      return s.sortDir * (bx - ax);
    } else if (s.sortKey === "wmb") {
      if (cluster) {
        av = _KX_WMB_BY_KIN_CLUSTER.get(`${a.kid}|${cluster}`);
        bv = _KX_WMB_BY_KIN_CLUSTER.get(`${b.kid}|${cluster}`);
      } else {
        const ma = _KX_WMB_MAX_BY_KID.get(a.kid), mb = _KX_WMB_MAX_BY_KID.get(b.kid);
        av = ma ? ma.frac : null; bv = mb ? mb.frac : null;
      }
    } else if (s.sortKey === "h_spec") {
      if (cluster) {
        const ea = _KX_HUMAN_SPEC_BY_NAME.get(a.name);
        const eb = _KX_HUMAN_SPEC_BY_NAME.get(b.name);
        const ca = ea && ea.seaad.get(cluster);
        const cb = eb && eb.seaad.get(cluster);
        av = ca ? ca.score : null; bv = cb ? cb.score : null;
      } else {
        const ma = _KX_SEAAD_MAX_BY_NAME.get(a.name), mb = _KX_SEAAD_MAX_BY_NAME.get(b.name);
        av = ma ? ma.score : null; bv = mb ? mb.score : null;
      }
    } else if (s.sortKey === "m_med") {
      av = a._mNes; bv = b._mNes;
    } else if (s.sortKey === "h_med") {
      av = a._hNes; bv = b._hNes;
    } else {
      av = a.peak_NES; bv = b.peak_NES;
    }
    const ax = (av == null || !isFinite(av)) ? -Infinity : Math.abs(av);
    const bx = (bv == null || !isFinite(bv)) ? -Infinity : Math.abs(bv);
    return s.sortDir * (bx - ax);
  });
}

// ---------- master render ----------

function _kxFilteredRows(s) {
  const searchQ = (s.search || "").toLowerCase().trim();
  const trackQ = (s.residueTrack || "").trim();
  const mMin = +(s.mSpecMin || 0);
  const hMin = +(s.hSpecMin || 0);
  const hLog2Min = hMin > 0 ? Math.log2(hMin) : 0;  // SEA-AD tiers are log2-based
  return _KX_ROWS.filter(r => {
    if (trackQ && r.residue !== trackQ) return false;
    if (searchQ) {
      const hay = (r.name + " " + r.gene).toLowerCase();
      if (!hay.includes(searchQ)) return false;
    }
    if (mMin > 0 || hMin > 0) {
      const sp = _kxResolveSpec(r, s.cluster);
      // mMin is a Song tau threshold (0.60 / 0.85), per kinase.
      if (mMin > 0 && (!sp.song || sp.song.tau == null || sp.song.tau < mMin)) return false;
      if (hMin > 0 && (sp.seaad == null || sp.seaad < hLog2Min)) return false;
    }
    return true;
  });
}

function _kxRenderTable() {
  _kxBuildIndexes();
  const s = _kxState();
  const fdrGate = (Store.state.filters && Store.state.filters.fdr) || 0.25;
  const wrap = document.getElementById("kx-table-wrap");
  const countEl = document.getElementById("kx-count");
  if (!wrap) return;

  const rows = _kxFilteredRows(s);
  _kxComputeAgreement(rows, fdrGate, s.allSamples);

  const head = _kxBuildHeader(s);

  // Agreement-category filter lives here (post-agreement) because the category
  // isn't known until _kxComputeAgreement runs against the live FDR. "" = any.
  const shown = s.agreeCat ? rows.filter(r => r._agreeCategory === s.agreeCat) : rows.slice();
  _kxSortRows(shown, s);
  const bodyParts = shown.map(r => _kxBuildRow(r, s, fdrGate));
  wrap.innerHTML = `<div class="ke-table-wrap" style="overflow:auto;max-height:75vh;"><table class="data-table" id="kx-table">${head}<tbody>${bodyParts.join("")}</tbody></table></div>`;

  if (countEl) {
    const n = shown.length;
    let cu = 0, cd = 0, dis = 0;
    for (const r of rows) {
      if (r._agreeCategory === "concordant-up") cu++;
      else if (r._agreeCategory === "concordant-down") cd++;
      else if (r._agreeCategory === "discordant") dis++;
    }
    countEl.textContent = `${n.toLocaleString()} kinase${n === 1 ? "" : "s"} · concordant ↑${cu} ↓${cd} · discordant ${dis} · cluster=${s.cluster || "any"} · ${s.allSamples ? "all samples" : `fdr<${fdrGate}`}`;
  }

  wrap.querySelectorAll("th[data-sortkey]").forEach(th => {
    th.style.cursor = "pointer";
    const k = th.dataset.sortkey;
    if (s.sortKey === k) th.innerHTML += ` <span class="muted" style="font-size:10px;">${s.sortDir > 0 ? "▲" : "▼"}</span>`;
    th.addEventListener("click", () => {
      if (s.sortKey === k) s.sortDir = -s.sortDir;
      else { s.sortKey = k; s.sortDir = _KX_STRING_KEYS.has(k) ? 1 : -1; }
      _kxRenderTable();
    });
  });
  wrap.querySelectorAll("tr.kx-data-row").forEach(tr => {
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      s.selectedKey = tr.dataset.key;
      // Local selection only — highlight in place, repaint the detail panel.
      wrap.querySelectorAll("tr.kx-data-row.selected").forEach(x => x.classList.remove("selected"));
      tr.classList.add("selected");
      _kxRenderDetail();
    });
  });

  _kxRenderDetail();
}

// ---------- cross-dataset detail panel ----------

function _kxNotMeasured(species) {
  const lbl = species === "mouse" ? "mouse (Song)" : "human (Mukesh)";
  return `<div class="kx-detail-placeholder muted">Not measured in ${lbl}.</div>`;
}

// _khRenderNESAcrossDonors / _khRenderAttribution read _KHState.auditDonor; pick
// this human row's first AD donor with a finite NES so the panels populate.
function _kxPrimeHumanDonor(hrow) {
  if (typeof _KHState === "undefined" || !_KH || !Array.isArray(_KH.donors)) return;
  let pick = null;
  if (hrow && hrow._nes) {
    for (let di = 0; di < _KH.donors.length; di++) {
      const v = hrow._nes[di];
      if (v != null && isFinite(v)) { pick = _KH.donors[di]; break; }
    }
  }
  _KHState.auditDonor = pick || _KH.donors[0] || null;
}

function _kxRenderDetail() {
  const host = document.getElementById("kx-detail");
  if (!host) return;
  const s = _kxState();
  const row = s.selectedKey ? _KX_ROWS.find(r => `${r.name}|${r.residue}` === s.selectedKey) : null;
  if (!row) {
    host.innerHTML = `<div class="kx-detail-idle muted">Select a kinase to compare mouse and human activity.</div>`;
    return;
  }
  const fdrGate = (Store.state.filters && Store.state.filters.fdr) || 0.25;
  _kxComputeAgreement([row], fdrGate, s.allSamples);
  const meta = _KX_AGREE_META[row._agreeCategory] || _KX_AGREE_META["neither"];
  const mouseKid = row._humanOnly ? null : row.kid;
  const humanKid = row._human ? row._human.kid : null;

  // Verdict-header significance counts (live).
  let mSig = 0, mTot = 0;
  for (const c of (CONTRASTS || [])) {
    const f = row._fdr[c];
    if (f != null && isFinite(f)) { mTot++; if (f < fdrGate) mSig++; }
  }
  let hSig = 0, hTot = 0;
  const pd = humanKid != null ? _KX_HUMAN_PERDONOR.get(humanKid) : null;
  if (pd) for (const d of _KX_AD_DONORS) {
    const e = pd.get(d);
    if (e && e.nes != null && isFinite(e.nes)) { hTot++; if (e.fdr != null && e.fdr < fdrGate) hSig++; }
  }
  const mNesTxt = row._mNes != null ? `${row._mNes >= 0 ? "+" : ""}${row._mNes.toFixed(2)}` : "n/a";
  const hNesTxt = row._hNes != null ? `${row._hNes >= 0 ? "+" : ""}${row._hNes.toFixed(2)}` : "n/a";
  const allS = s.allSamples;
  const nesLbl = allS ? "median NES (all samples)" : "median NES";
  const mCntTxt = allS ? `${mTot} contrast${mTot === 1 ? "" : "s"}` : `${mSig}/${mTot || "–"} contrasts sig`;
  const hCntTxt = allS ? `${hTot} donor${hTot === 1 ? "" : "s"}` : `${hSig}/${hTot || "–"} donors sig`;
  const tab = s.detailTab || "activity";

  host.innerHTML = `
    <div class="kx-detail-header">
      <div class="kx-detail-title">
        <b>${_escapeHtml(row.name)}</b>
        <span class="muted">${_escapeHtml(row.gene || "")}${row.gene ? " · " : ""}${_escapeHtml(row.residue)}</span>
        <span class="${meta.cls}" title="${_escapeHtml(meta.tip)}">${meta.glyph} ${_escapeHtml(meta.label)}</span>
      </div>
      <div class="kx-detail-verdict muted">
        Mouse (Song) ${nesLbl} <b>${mNesTxt}</b> (${mCntTxt}) ·
        Human (Mukesh) ${nesLbl} <b>${hNesTxt}</b> (${hCntTxt})
      </div>
    </div>
    <nav class="kx-detail-tabs">
      <button data-kxd-tab="activity" class="${tab === "activity" ? "active" : ""}">NES Activity</button>
      <button data-kxd-tab="specificity" class="${tab === "specificity" ? "active" : ""}">Cell-type Specificity</button>
    </nav>
    <div class="kx-detail-grid">
      <div class="kx-detail-col"><div class="kx-detail-col-head">Mouse · Song</div><div id="kx-detail-m"></div></div>
      <div class="kx-detail-col"><div class="kx-detail-col-head">Human · Mukesh</div><div id="kx-detail-h"></div></div>
    </div>`;

  host.querySelectorAll("button[data-kxd-tab]").forEach(b => b.addEventListener("click", () => {
    s.detailTab = b.dataset.kxdTab;
    _kxRenderDetail();
  }));

  const mEl = document.getElementById("kx-detail-m");
  const hEl = document.getElementById("kx-detail-h");
  if (typeof _ensureKinaseIndexes === "function") _ensureKinaseIndexes();
  const humanReady = (typeof _KH_HAS !== "undefined") && _KH_HAS && _KH;
  const hrow = (humanReady && humanKid != null && typeof _khAllRows === "function")
    ? _khAllRows().find(x => x.id === humanKid) : null;

  if (tab === "activity") {
    if (mouseKid != null && typeof _renderKinaseNesPlot === "function") _renderKinaseNesPlot("kx-detail-m", mouseKid);
    else mEl.innerHTML = _kxNotMeasured("mouse");
    if (hrow && typeof _khRenderNESAcrossDonors === "function") {
      _kxPrimeHumanDonor(hrow);
      _khRenderNESAcrossDonors("kx-detail-h", hrow, _KHState.auditDonor);
    } else hEl.innerHTML = _kxNotMeasured("human");
  } else {
    if (mouseKid != null && typeof _renderKinaseCelltypeEvidence === "function") _renderKinaseCelltypeEvidence("kx-detail-m", mouseKid);
    else mEl.innerHTML = _kxNotMeasured("mouse");
    if (hrow && (_KH.celltype_specificity) && typeof _khRenderAttribution === "function") {
      _kxPrimeHumanDonor(hrow);
      _khRenderAttribution(hEl, hrow);
    } else hEl.innerHTML = hrow ? `<div class="kx-detail-placeholder muted">No human cell-type specificity reference loaded.</div>` : _kxNotMeasured("human");
  }
}

// ---------- controls + wiring ----------

function _kxSyncControls() {
  const s = _kxState();
  const clSel = document.getElementById("kx-cluster");
  if (clSel) {
    clSel.innerHTML = `<option value="">Any cell type</option>` + (_KX_CLUSTERS || []).map(c =>
      `<option value="${_escapeHtml(c)}">${_escapeHtml(c)}</option>`
    ).join("");
    clSel.value = (s.cluster && (_KX_CLUSTERS || []).includes(s.cluster)) ? s.cluster : "";
  }
  const tSel = document.getElementById("kx-track");
  if (tSel) tSel.value = s.residueTrack || "";
  const sInput = document.getElementById("kx-search");
  if (sInput) sInput.value = s.search || "";
  const mSel = document.getElementById("kx-mspec");
  if (mSel) mSel.value = String(s.mSpecMin || 0);
  const hSel = document.getElementById("kx-hspec");
  if (hSel) hSel.value = String(s.hSpecMin || 0);
  const aChk = document.getElementById("kx-allsamples");
  if (aChk) aChk.checked = !!s.allSamples;
  const agSel = document.getElementById("kx-agree");
  if (agSel) agSel.value = s.agreeCat || "";
}

function wireKinaseCrosstable() {
  _kxBuildIndexes();
  const clSel = document.getElementById("kx-cluster");
  if (clSel) clSel.addEventListener("change", () => { _kxState().cluster = clSel.value; _kxRenderTable(); });
  const tSel = document.getElementById("kx-track");
  if (tSel) tSel.addEventListener("change", () => { _kxState().residueTrack = tSel.value; _kxRenderTable(); });
  const sInput = document.getElementById("kx-search");
  if (sInput) sInput.addEventListener("input", () => { _kxState().search = sInput.value; _kxRenderTable(); });
  const mSel = document.getElementById("kx-mspec");
  if (mSel) mSel.addEventListener("change", () => { _kxState().mSpecMin = +mSel.value; _kxRenderTable(); });
  const hSel = document.getElementById("kx-hspec");
  if (hSel) hSel.addEventListener("change", () => { _kxState().hSpecMin = +hSel.value; _kxRenderTable(); });
  const aChk = document.getElementById("kx-allsamples");
  if (aChk) aChk.addEventListener("change", () => { _kxState().allSamples = aChk.checked; _kxRenderTable(); });
  const agSel = document.getElementById("kx-agree");
  if (agSel) agSel.addEventListener("change", () => { _kxState().agreeCat = agSel.value; _kxRenderTable(); });
  const reset = document.getElementById("kx-reset");
  if (reset) reset.addEventListener("click", () => {
    const s = _kxState();
    s.cluster = "";   // Any cell type
    s.residueTrack = ""; s.search = "";
    s.mSpecMin = 0; s.hSpecMin = 0; s.allSamples = false;
    s.sortKey = "agree_score"; s.sortDir = -1;
    s.agreeCat = "";
    s.selectedKey = null; s.detailTab = "activity";
    _kxSyncControls();
    _kxRenderTable();
  });
}

function renderKinaseCrosstable() {
  _kxBuildIndexes();
  _kxSyncControls();
  _kxRenderTable();
}
