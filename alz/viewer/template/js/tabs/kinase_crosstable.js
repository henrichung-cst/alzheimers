// ---------------------------------------------------------------------------
// Kinase Evidence Crosstable — one row per kinase joining every evidence
// source side-by-side. All mouse contrasts and per-donor human NES are
// rendered as fixed column grids (not filters); the user controls breadth via
// a per-column-group visibility panel.
//
// Joins are client-side from existing PAYLOAD keys:
//   PAYLOAD.kinases                              mouse bulk NES_<contrast>
//   PAYLOAD.decomposition_index                  mouse cluster×contrast NES
//   PAYLOAD.attribution_index                    WMB specificity per cluster
//   PAYLOAD.human.kinases                        cohort median, SEA-AD LFC
//   PAYLOAD.human.perdonor_index                 per-donor NES (AD + CTRL)
//   PAYLOAD.human.celltype_specificity.{seaad_mtg,allen_hbca}.ranked_by_kinase
//
// Spec tier thresholds match the existing tabs:
//   WMB (raw fraction): kinase_explorer's _WMB_UNIFORM tiers (10×/5×/2×/1×)
//   SEA-AD MTG, HBCA (log2 score): kinase_human's log2(10/5/2/1) tiers
// ---------------------------------------------------------------------------

let _KX_ROWS = null;
let _KX_HUMAN_BY_NAME = null;
let _KX_HUMAN_PERDONOR = null;    // Map<kid, Map<donor, {nes, fdr}>>
let _KX_HUMAN_SPEC_BY_NAME = null;
let _KX_WMB_BY_KIN_CLUSTER = null; // Map<`kid|cluster`, wmb_specificity>
let _KX_DECOMP_BY_KIN_CTX = null;
let _KX_CLUSTERS = null;
let _KX_AD_DONORS = [];
let _KX_CTRL_DONORS = [];
let _KX_INITIALIZED = false;

// Uniform baseline for WMB specificity (denominator across active clusters).
// kinase_explorer uses 1/34 (WMB legacy); we use 1/N over the active levy_t5
// spine which is what the wmb_specificity column was computed against.
let _KX_WMB_UNIFORM = 1 / 34;

function _kxState() {
  if (!Store.state.view.crosstable) {
    Store.state.view.crosstable = {
      cluster: "",
      residueTrack: "",
      search: "",
      sortKey: "peak_NES",
      sortDir: -1,
      cols: null,    // initialized after indexes build
    };
  }
  return Store.state.view.crosstable;
}

function _kxDefaultCols() {
  // Default: everything visible. Users prune via the disclosure panel.
  return {
    m_bulk:   new Set(CONTRASTS || []),
    m_decomp: new Set(CONTRASTS || []),
    h_ad:     new Set(_KX_AD_DONORS),
    h_ctrl:   new Set(_KX_CTRL_DONORS),
    spec_raw:  new Set(["wmb", "seaad", "hbca"]),
    spec_tier: new Set(["wmb", "seaad", "hbca"]),
    show_median:    true,
    show_n_sig:     true,
    show_seaad_lfc: true,
    show_seaad_celllfc: true,
  };
}

function _kxBuildIndexes() {
  if (_KX_INITIALIZED) return;
  const K = PAYLOAD.kinases;
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
  const A = PAYLOAD.attribution_index;
  if (A && A.kinase_id) {
    for (let i = 0; i < A.kinase_id.length; i++) {
      const v = A.wmb_specificity ? A.wmb_specificity[i] : null;
      if (v == null || !isFinite(v)) continue;
      const key = `${A.kinase_id[i]}|${A.cell_type[i]}`;
      // attribution_index can carry multiple rows per (kinase, cluster) across
      // contrasts/confidence strata; keep the max specificity seen (it's a
      // cluster-level property, not contrast-level — picks the populated row).
      const prev = _KX_WMB_BY_KIN_CLUSTER.get(key);
      if (prev == null || v > prev) _KX_WMB_BY_KIN_CLUSTER.set(key, v);
    }
  }

  _KX_HUMAN_BY_NAME = new Map();
  _KX_HUMAN_PERDONOR = new Map();
  _KX_HUMAN_SPEC_BY_NAME = new Map();
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
  if (_KX_CLUSTERS.length) _KX_WMB_UNIFORM = 1 / _KX_CLUSTERS.length;

  for (const r of _KX_ROWS) r._human = _KX_HUMAN_BY_NAME.get(`${r.name}|${r.residue}`) || null;

  const s = _kxState();
  if (!s.cluster && _KX_CLUSTERS.length) s.cluster = _KX_CLUSTERS[0];
  if (!s.cols) s.cols = _kxDefaultCols();

  _KX_INITIALIZED = true;
}

// ---------- cell renderers ----------

function _kxNesCellHtml(nes, fdr, fdrGate, title) {
  if (nes == null || !isFinite(nes)) return `<td class="muted">–</td>`;
  const sig = fdr != null && isFinite(fdr) && fdr < fdrGate;
  const sign = nes >= 0 ? "+" : "";
  const mag = Math.min(1, Math.abs(nes) / 3);
  const hue = nes >= 0 ? "0,80%" : "220,70%";
  const bg = `hsl(${hue},${90 - mag * 30}%)`;
  const fg = mag > 0.5 ? "#fff" : "#222";
  const outline = sig ? "outline:1.5px solid #111;outline-offset:-2px;" : "";
  const ttl = title || `NES=${nes.toFixed(3)}, FDR=${fdr != null ? fdr.toFixed(3) : "n/a"}`;
  return `<td style="background:${bg};color:${fg};text-align:right;${outline}font-size:11px;padding:2px 4px;" title="${_escapeHtml(ttl)}">${sign}${nes.toFixed(2)}</td>`;
}

function _kxRawSpecCellHtml(score, scale) {
  if (score == null || !isFinite(score)) return `<td class="muted">–</td>`;
  // scale="log2" colours diverge around 0; scale="frac" diverges around uniform.
  let signed;
  if (scale === "log2") signed = score;
  else signed = score - _KX_WMB_UNIFORM;
  const mag = scale === "log2"
    ? Math.min(1, Math.abs(signed) / 3)
    : Math.min(1, Math.abs(signed) / Math.max(1e-9, _KX_WMB_UNIFORM * 5));
  const hue = signed >= 0 ? "120,55%" : "30,55%";
  const bg = `hsl(${hue},${88 - mag * 28}%)`;
  const fg = mag > 0.55 ? "#fff" : "#222";
  const txt = scale === "log2" ? score.toFixed(2) : score.toFixed(3);
  return `<td style="background:${bg};color:${fg};text-align:right;font-size:11px;padding:2px 4px;" title="${scale === "log2" ? "log2 " : ""}${score.toFixed(4)}">${txt}</td>`;
}

function _kxLfcCellHtml(lfc) {
  if (lfc == null || !isFinite(lfc)) return `<td class="muted">–</td>`;
  const mag = Math.min(1, Math.abs(lfc) / 1.5);
  const hue = lfc >= 0 ? "0,70%" : "220,60%";
  const bg = `hsl(${hue},${92 - mag * 32}%)`;
  const fg = mag > 0.55 ? "#fff" : "#222";
  const sign = lfc >= 0 ? "+" : "";
  return `<td style="background:${bg};color:${fg};text-align:right;font-size:11px;padding:2px 4px;" title="LFC ${lfc.toFixed(3)}">${sign}${lfc.toFixed(2)}</td>`;
}

// Specificity tier badges. Log2 tier reuses _khSpecTierRank/_khSpecTierBadge
// from kinase_human.js (same thresholds, same CSS classes); WMB tier stays
// local because its uniform baseline depends on the active levy_t5 spine size.
function _kxLog2TierBadge(score) {
  if (score == null || !isFinite(score)) return `<td class="muted">–</td>`;
  const rank = _khSpecTierRank(score);
  if (!rank) return `<td class="muted" title="log2 ${score.toFixed(3)}">&lt;1×</td>`;
  return `<td style="text-align:center;padding:2px 4px;">${_khSpecTierBadge(rank)}</td>`;
}
function _kxWmbTierBadge(frac) {
  if (frac == null || !isFinite(frac)) return `<td class="muted">–</td>`;
  const u = _KX_WMB_UNIFORM;
  let cls, label;
  if (frac >= 10 * u) { cls = "badge vhi"; label = "≥10×"; }
  else if (frac >= 5 * u) { cls = "badge hi"; label = "≥5×"; }
  else if (frac >= 2 * u) { cls = "badge mid"; label = "≥2×"; }
  else if (frac >= u)     { cls = "badge lo"; label = "≥1×"; }
  else                    { return `<td class="muted" title="${frac.toFixed(4)} vs uniform ${u.toFixed(4)}">&lt;1×</td>`; }
  return `<td style="text-align:center;padding:2px 4px;"><span class="${cls}" title="WMB specificity ${frac.toFixed(4)} vs uniform ${u.toFixed(4)}">${label}</span></td>`;
}

// ---------- header + row builders ----------

function _kxBuildHeader(s, cols) {
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
  // Mouse bulk
  for (const c of (CONTRASTS || [])) {
    if (!cols.m_bulk.has(c)) continue;
    cells.push(TH(`mbulk:${c}`, `M-bulk ${c}`, `Mouse bulk MEA NES — ${c}. Outlined cells = FDR < header threshold.`, "kx-mbulk"));
  }
  // Mouse decomp at the selected cluster
  for (const c of (CONTRASTS || [])) {
    if (!cols.m_decomp.has(c)) continue;
    cells.push(TH(`mdec:${c}`, `M-decomp ${c}`, `Mouse per-cluster (decomposed) MEA NES — ${c} × ${s.cluster}. Outlined cells = FDR < header threshold.`, "kx-mdec"));
  }
  // Human cohort summary
  if (cols.show_median)        cells.push(TH("human_median", "H-median NES", "Human (Mukesh/NBB) cohort median NES across AD donors significant at the header FDR.", "kx-hsum"));
  if (cols.show_n_sig)         cells.push(TH(null, "H n_sig", "Human AD donors significant at the header FDR. Hover for up/down split.", "kx-hsum"));
  if (cols.show_seaad_lfc)     cells.push(TH("seaad_lfc", "SEA-AD LFC", "SEA-AD MTG cohort-level AD-vs-control log2FC (CR03 'full' stratum).", "kx-hsum"));
  // Per-donor AD
  for (const d of _KX_AD_DONORS) {
    if (!cols.h_ad.has(d)) continue;
    cells.push(TH(null, `H ${d}`, `Per-donor MEA NES for AD donor ${d}. Outlined cells = FDR < header threshold.`, "kx-had"));
  }
  // Per-donor CTRL
  for (const d of _KX_CTRL_DONORS) {
    if (!cols.h_ctrl.has(d)) continue;
    cells.push(TH(null, `H ${d}`, `Per-donor MEA NES for control donor ${d}. Outlined cells = FDR < header threshold.`, "kx-hctrl"));
  }
  // Specificity at the selected cluster — raw + tier per source
  if (cols.spec_raw.has("wmb"))    cells.push(TH("wmb_spec", "WMB spec", `Mouse WMB cell-type specificity (raw fraction) at ${s.cluster}.`, "kx-spec"));
  if (cols.spec_tier.has("wmb"))   cells.push(TH(null, "WMB ≥N×", `WMB specificity tier vs uniform (1/${_KX_CLUSTERS.length}).`, "kx-spec"));
  if (cols.spec_raw.has("seaad"))  cells.push(TH("seaad_spec", "SEA-AD spec", `SEA-AD MTG cell-type specificity log2(cluster mean / brain mean) at ${s.cluster}.`, "kx-spec"));
  if (cols.spec_tier.has("seaad")) cells.push(TH(null, "SEA-AD ≥N×", "SEA-AD specificity tier.", "kx-spec"));
  if (cols.show_seaad_celllfc)     cells.push(TH(null, "SEA-AD cell LFC", `SEA-AD per-supertype AD-vs-control LFC, rolled up to ${s.cluster}.`, "kx-spec"));
  if (cols.spec_raw.has("hbca"))   cells.push(TH("hbca_spec", "HBCA spec", `Allen HBCA cell-type specificity log2(cluster mean / brain mean) at ${s.cluster}.`, "kx-spec"));
  if (cols.spec_tier.has("hbca"))  cells.push(TH(null, "HBCA ≥N×", "HBCA specificity tier.", "kx-spec"));
  return `<thead><tr>${cells.join("")}</tr></thead>`;
}

function _kxBuildRow(r, s, cols, fdrGate) {
  const cluster = s.cluster;
  const human = r._human;
  const perDonor = human ? _KX_HUMAN_PERDONOR.get(human.kid) : null;
  const spec = _KX_HUMAN_SPEC_BY_NAME.get(r.name);
  const seaadCt = spec && spec.seaad.get(cluster);
  const hbcaCt = spec && spec.hbca.get(cluster);
  const wmbSpec = _KX_WMB_BY_KIN_CLUSTER.get(`${r.kid}|${cluster}`);

  const tds = [];
  tds.push(`<td>${_escapeHtml(r.name)}</td>`);
  tds.push(`<td>${_escapeHtml(r.gene)}</td>`);
  tds.push(`<td class="muted">${_escapeHtml(r.residue)}</td>`);
  tds.push(`<td class="muted">${_escapeHtml(r.family)}</td>`);
  for (const c of (CONTRASTS || [])) {
    if (!cols.m_bulk.has(c)) continue;
    tds.push(_kxNesCellHtml(r._nes[c], r._fdr[c], fdrGate, `${c}: NES=${(r._nes[c]||0).toFixed ? r._nes[c].toFixed(3) : "–"}, FDR=${r._fdr[c] != null ? r._fdr[c].toFixed(3) : "n/a"}`));
  }
  for (const c of (CONTRASTS || [])) {
    if (!cols.m_decomp.has(c)) continue;
    const d = _KX_DECOMP_BY_KIN_CTX.get(`${r.kid}|${c}|${cluster}`);
    tds.push(_kxNesCellHtml(d ? d.nes : null, d ? d.fdr : null, fdrGate, `${c} × ${cluster}`));
  }
  if (cols.show_median) tds.push(_kxNesCellHtml(human ? human.median_nes : null, null, 1, "Human cohort median NES (sig only)"));
  if (cols.show_n_sig)  tds.push(`<td style="text-align:right;" title="${human ? `up=${human.n_up}, down=${human.n_down}, tested=${human.n_tested}` : ""}">${human && human.n_sig != null ? human.n_sig : "–"}</td>`);
  if (cols.show_seaad_lfc) tds.push(_kxLfcCellHtml(human ? human.sea_ad_lfc : null));
  for (const d of _KX_AD_DONORS) {
    if (!cols.h_ad.has(d)) continue;
    const pd = perDonor ? perDonor.get(d) : null;
    tds.push(_kxNesCellHtml(pd ? pd.nes : null, pd ? pd.fdr : null, fdrGate, `${d} (AD)`));
  }
  for (const d of _KX_CTRL_DONORS) {
    if (!cols.h_ctrl.has(d)) continue;
    const pd = perDonor ? perDonor.get(d) : null;
    tds.push(_kxNesCellHtml(pd ? pd.nes : null, pd ? pd.fdr : null, fdrGate, `${d} (CTRL)`));
  }
  if (cols.spec_raw.has("wmb"))    tds.push(_kxRawSpecCellHtml(wmbSpec != null ? wmbSpec : null, "frac"));
  if (cols.spec_tier.has("wmb"))   tds.push(_kxWmbTierBadge(wmbSpec != null ? wmbSpec : null));
  if (cols.spec_raw.has("seaad"))  tds.push(_kxRawSpecCellHtml(seaadCt ? seaadCt.score : null, "log2"));
  if (cols.spec_tier.has("seaad")) tds.push(_kxLog2TierBadge(seaadCt ? seaadCt.score : null));
  if (cols.show_seaad_celllfc)     tds.push(_kxLfcCellHtml(seaadCt ? seaadCt.lfc : null));
  if (cols.spec_raw.has("hbca"))   tds.push(_kxRawSpecCellHtml(hbcaCt ? hbcaCt.score : null, "log2"));
  if (cols.spec_tier.has("hbca"))  tds.push(_kxLog2TierBadge(hbcaCt ? hbcaCt.score : null));
  return `<tr data-kid="${r.kid}">${tds.join("")}</tr>`;
}

const _KX_STRING_KEYS = new Set(["name", "gene", "residue", "family"]);

function _kxSortRows(rows, s) {
  const cluster = s.cluster;
  if (_KX_STRING_KEYS.has(s.sortKey)) {
    const k = s.sortKey;
    rows.sort((a, b) => s.sortDir * (a[k] || "").localeCompare(b[k] || ""));
    return;
  }
  rows.sort((a, b) => {
    let av, bv;
    if (s.sortKey && s.sortKey.startsWith("mbulk:")) {
      const c = s.sortKey.slice(6);
      av = a._nes[c]; bv = b._nes[c];
    } else if (s.sortKey && s.sortKey.startsWith("mdec:")) {
      const c = s.sortKey.slice(5);
      const ka = _KX_DECOMP_BY_KIN_CTX.get(`${a.kid}|${c}|${cluster}`);
      const kb = _KX_DECOMP_BY_KIN_CTX.get(`${b.kid}|${c}|${cluster}`);
      av = ka ? ka.nes : null; bv = kb ? kb.nes : null;
    } else if (s.sortKey === "human_median") {
      av = (a._human||{}).median_nes; bv = (b._human||{}).median_nes;
    } else if (s.sortKey === "seaad_lfc") {
      av = (a._human||{}).sea_ad_lfc; bv = (b._human||{}).sea_ad_lfc;
    } else if (s.sortKey === "wmb_spec") {
      av = _KX_WMB_BY_KIN_CLUSTER.get(`${a.kid}|${cluster}`);
      bv = _KX_WMB_BY_KIN_CLUSTER.get(`${b.kid}|${cluster}`);
    } else if (s.sortKey === "seaad_spec" || s.sortKey === "hbca_spec") {
      const side = s.sortKey === "seaad_spec" ? "seaad" : "hbca";
      const ea = _KX_HUMAN_SPEC_BY_NAME.get(a.name);
      const eb = _KX_HUMAN_SPEC_BY_NAME.get(b.name);
      const ca = ea && ea[side].get(cluster);
      const cb = eb && eb[side].get(cluster);
      av = ca ? ca.score : null; bv = cb ? cb.score : null;
    } else {
      av = a.peak_NES; bv = b.peak_NES;
    }
    const ax = (av == null || !isFinite(av)) ? -Infinity : Math.abs(av);
    const bx = (bv == null || !isFinite(bv)) ? -Infinity : Math.abs(bv);
    return s.sortDir * (bx - ax);
  });
}

// ---------- column-visibility disclosure ----------

function _kxRenderColsPanel() {
  const panel = document.getElementById("kx-cols-panel");
  if (!panel) return;
  const s = _kxState();
  const c = s.cols;
  const grp = (title, items, getter, setter) => {
    const boxes = items.map(it => {
      const id = `kxc-${title.toLowerCase().replace(/[^a-z0-9]+/g,'-')}-${it.value}`;
      const checked = getter(it.value) ? "checked" : "";
      return `<label style="display:inline-block;margin:0 8px 2px 0;font-size:11px;white-space:nowrap;"><input type="checkbox" id="${id}" data-kxc-grp="${title}" data-kxc-val="${_escapeHtml(it.value)}" ${checked}> ${_escapeHtml(it.label)}</label>`;
    }).join("");
    return `<div style="margin:4px 0;"><b style="font-size:11px;">${title}:</b>
      <button class="ke-filter-reset" data-kxc-bulk="${title}" data-kxc-mode="all" style="margin:0 4px;font-size:10px;padding:1px 6px;">All</button>
      <button class="ke-filter-reset" data-kxc-bulk="${title}" data-kxc-mode="none" style="margin:0 4px;font-size:10px;padding:1px 6px;">None</button>
      ${boxes}</div>`;
  };
  const contrastItems = (CONTRASTS || []).map(c => ({value: c, label: c}));
  const adItems = _KX_AD_DONORS.map(d => ({value: d, label: d}));
  const ctrlItems = _KX_CTRL_DONORS.map(d => ({value: d, label: d}));
  const specSrcItems = [
    {value: "wmb", label: "WMB (mouse)"},
    {value: "seaad", label: "SEA-AD"},
    {value: "hbca", label: "HBCA"},
  ];
  const specVals = () => ["wmb","seaad","hbca"];
  const contrastVals = () => (CONTRASTS || []).slice();
  // Single source of truth: each group's display label, current set, and the
  // universe used by All/None. data-kxc-bulk/grp attributes carry the label
  // verbatim — keep these three columns in lockstep.
  const GROUPS = [
    {label: "Mouse bulk contrasts",    items: contrastItems, set: () => c.m_bulk,    all: contrastVals, has: v => c.m_bulk.has(v)},
    {label: "Mouse decomp contrasts",  items: contrastItems, set: () => c.m_decomp,  all: contrastVals, has: v => c.m_decomp.has(v)},
    {label: "Human AD donors",         items: adItems,       set: () => c.h_ad,      all: () => _KX_AD_DONORS.slice(),   has: v => c.h_ad.has(v)},
    {label: "Human CTRL donors",       items: ctrlItems,     set: () => c.h_ctrl,    all: () => _KX_CTRL_DONORS.slice(), has: v => c.h_ctrl.has(v)},
    {label: "Specificity raw",         items: specSrcItems,  set: () => c.spec_raw,  all: specVals, has: v => c.spec_raw.has(v)},
    {label: "Specificity tier (≥N×)",  items: specSrcItems,  set: () => c.spec_tier, all: specVals, has: v => c.spec_tier.has(v)},
  ];
  const grpToSet = Object.fromEntries(GROUPS.map(g => [g.label, g.set]));
  const grpToAll = Object.fromEntries(GROUPS.map(g => [g.label, g.all]));

  panel.innerHTML = [
    ...GROUPS.map(g => grp(g.label, g.items, g.has)),
    `<div style="margin:4px 0;font-size:11px;">
      <b>Human cohort:</b>
      <label><input type="checkbox" data-kxc-flag="show_median" ${c.show_median?"checked":""}> Median NES</label>
      <label style="margin-left:8px;"><input type="checkbox" data-kxc-flag="show_n_sig" ${c.show_n_sig?"checked":""}> n_sig</label>
      <label style="margin-left:8px;"><input type="checkbox" data-kxc-flag="show_seaad_lfc" ${c.show_seaad_lfc?"checked":""}> SEA-AD LFC</label>
      <label style="margin-left:8px;"><input type="checkbox" data-kxc-flag="show_seaad_celllfc" ${c.show_seaad_celllfc?"checked":""}> SEA-AD cell LFC</label>
     </div>`,
  ].join("");

  panel.querySelectorAll("input[type=checkbox][data-kxc-grp]").forEach(cb => {
    cb.addEventListener("change", () => {
      const set = grpToSet[cb.dataset.kxcGrp]();
      if (cb.checked) set.add(cb.dataset.kxcVal); else set.delete(cb.dataset.kxcVal);
      _kxRenderTable();
    });
  });
  panel.querySelectorAll("input[type=checkbox][data-kxc-flag]").forEach(cb => {
    cb.addEventListener("change", () => {
      c[cb.dataset.kxcFlag] = cb.checked;
      _kxRenderTable();
    });
  });
  panel.querySelectorAll("button[data-kxc-bulk]").forEach(btn => {
    btn.addEventListener("click", () => {
      const set = grpToSet[btn.dataset.kxcBulk]();
      set.clear();
      if (btn.dataset.kxcMode === "all") {
        for (const v of grpToAll[btn.dataset.kxcBulk]()) set.add(v);
      }
      _kxRenderColsPanel();
      _kxRenderTable();
    });
  });
}

// ---------- main render ----------

function _kxRenderTable() {
  _kxBuildIndexes();
  const s = _kxState();
  const cols = s.cols;
  const fdrGate = (Store.state.filters && Store.state.filters.fdr) || 0.25;
  const wrap = document.getElementById("kx-table-wrap");
  const countEl = document.getElementById("kx-count");
  if (!wrap) return;

  const searchQ = (s.search || "").toLowerCase().trim();
  const trackQ = (s.residueTrack || "").trim();
  const rows = _KX_ROWS.filter(r => {
    if (trackQ && r.residue !== trackQ) return false;
    if (searchQ) {
      const hay = (r.name + " " + r.gene).toLowerCase();
      if (!hay.includes(searchQ)) return false;
    }
    return true;
  });
  _kxSortRows(rows, s);

  const head = _kxBuildHeader(s, cols);
  const body = rows.map(r => _kxBuildRow(r, s, cols, fdrGate)).join("");
  wrap.innerHTML = `<div class="ke-table-wrap" style="overflow:auto;max-height:75vh;"><table class="data-table" id="kx-table">${head}<tbody>${body}</tbody></table></div>`;

  if (countEl) {
    const n = rows.length;
    countEl.textContent = `${n.toLocaleString()} kinase${n === 1 ? "" : "s"} · cluster=${s.cluster || "n/a"} · fdr<${fdrGate}`;
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
  wrap.querySelectorAll("tbody tr").forEach(tr => {
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      const kid = parseInt(tr.dataset.kid, 10);
      if (!isNaN(kid)) Store.dispatch({type: "SET_SELECTION", key: "kinase", value: kid});
    });
  });
}

function _kxSyncControls() {
  const s = _kxState();
  const clSel = document.getElementById("kx-cluster");
  if (clSel) {
    clSel.innerHTML = (_KX_CLUSTERS || []).map(c =>
      `<option value="${_escapeHtml(c)}">${_escapeHtml(c)}</option>`
    ).join("");
    if (s.cluster && (_KX_CLUSTERS || []).includes(s.cluster)) clSel.value = s.cluster;
  }
  const tSel = document.getElementById("kx-track");
  if (tSel) tSel.value = s.residueTrack || "";
  const sInput = document.getElementById("kx-search");
  if (sInput) sInput.value = s.search || "";
}

function wireKinaseCrosstable() {
  _kxBuildIndexes();
  const clSel = document.getElementById("kx-cluster");
  if (clSel) clSel.addEventListener("change", () => { _kxState().cluster = clSel.value; _kxRenderTable(); });
  const tSel = document.getElementById("kx-track");
  if (tSel) tSel.addEventListener("change", () => { _kxState().residueTrack = tSel.value; _kxRenderTable(); });
  const sInput = document.getElementById("kx-search");
  if (sInput) sInput.addEventListener("input", () => { _kxState().search = sInput.value; _kxRenderTable(); });
  const reset = document.getElementById("kx-reset");
  if (reset) reset.addEventListener("click", () => {
    const s = _kxState();
    s.cluster = (_KX_CLUSTERS && _KX_CLUSTERS[0]) || "";
    s.residueTrack = ""; s.search = "";
    s.sortKey = "peak_NES"; s.sortDir = -1;
    s.cols = _kxDefaultCols();
    _kxSyncControls();
    _kxRenderColsPanel();
    _kxRenderTable();
  });
  _kxRenderColsPanel();
}

function renderKinaseCrosstable() {
  _kxBuildIndexes();
  _kxSyncControls();
  _kxRenderColsPanel();
  _kxRenderTable();
}
