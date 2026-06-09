// Human (Mukesh / NBB) per-donor kinase explorer.
//
// Mirrors the mouse kinase tab in look/feel but runs against PAYLOAD.human:
//   - rows are (kinase, residue_type) pairs from per-donor MEA
//   - contrast axis is N donor-vs-CTRLmean strings, not 9 disease×timepoint
//   - filters: search, donor multiselect, n_donors_sig minimum, residue track
//   - detail panel scopes per-cohort: per-donor NES bar, recurrence,
//     leading substrates, global-shift diagnostics

// PAYLOAD is async-loaded; declare these as `let` so boot.js can re-assign
// them after _loadPayload() resolves, before any tab is rendered.
let _KH_HAS = false;
let _KH = null;

const _KHState = {
  search: "",
  donors: new Set(),   // empty = all (filters AD axis only)
  nsigMin: 0,
  track: "",
  celltype: "",
  confidence: "",
  specificityTier: 0,
  seaad: "",           // "" | "agree" | "disagree" | "na"
  adDir: "",           // "" | "all_up" | "all_down" | "mixed" | "none"
  ctrlDir: "",         // "" | "all_up" | "all_down" | "mixed" | "none"
  dirMode: "sig",      // "sig" = significant donors only · "tested" = any donor with a finite NES
  showCtrl: true,      // render side-by-side CTRL column group
  sortCol: "conf",
  sortAsc: false,
  auditTab: "score",     // trace | prep | score
  auditDonor: null,      // donor selected for donor-scoped sub-tabs
};

// Direction pattern over (n_up, n_down). "none" = no donors in scope.
function _khDirPattern(nUp, nDown) {
  const u = nUp || 0, d = nDown || 0;
  if (u === 0 && d === 0) return "none";
  if (u > 0 && d === 0) return "all_up";
  if (d > 0 && u === 0) return "all_down";
  return "mixed";
}

// Count (up, down) over a NES vector, optionally gated by FDR.
// mode = "sig": only donors with fdr < fdrThresh contribute.
// mode = "tested": every donor with a finite NES contributes (sign of NES).
function _khCountUpDown(nesVec, fdrVec, mode, fdrThresh) {
  let u = 0, d = 0;
  const n = nesVec.length;
  for (let i = 0; i < n; i++) {
    const v = nesVec[i];
    if (v == null || !isFinite(v)) continue;
    if (mode === "sig") {
      const f = fdrVec ? fdrVec[i] : null;
      if (f == null || !(f < fdrThresh)) continue;
    }
    if (v > 0) u += 1;
    else if (v < 0) d += 1;
  }
  return [u, d];
}

const _KH_AUDIT_TABS = [
  {id: "trace",       label: "Measurement Trace"},
  {id: "prep",        label: "MEA Preparation"},
  {id: "score",       label: "MEA Score"},
  {id: "attribution", label: "Attribution"},
];

// Lazy site-by-motif index for trace lookups: motif -> [site_indices...]
let _khSiteByMotif = null;
function _khEnsureSiteIndex() {
  if (_khSiteByMotif || !_KH || !_KH.sites) return;
  _khSiteByMotif = new Map();
  const M = _KH.sites.motif || [];
  for (let i = 0; i < M.length; i++) {
    const m = _normMotif(M[i]);
    if (!m) continue;
    if (!_khSiteByMotif.has(m)) _khSiteByMotif.set(m, []);
    _khSiteByMotif.get(m).push(i);
  }
}

function _khPerdonorFor(khid, donor) {
  // Scalar per-(kinase, donor) record. The two heavy string fields
  // (leading_substrates, substrate_motifs) are sharded out to
  // edge_slices/human_perdonor/ and fetched on demand via
  // _khSubstrateFor — keeps PAYLOAD parse under ~30 MB.
  const PI = _KH && _KH.perdonor_index;
  if (!PI) return null;
  for (let i = 0; i < PI.kinase_id.length; i++) {
    if (PI.kinase_id[i] === khid && PI.donor[i] === donor) {
      return {
        NES: PI.NES[i], FDR: PI.FDR[i],
        ES: PI.ES ? PI.ES[i] : null,
        p_value: PI.p_value ? PI.p_value[i] : null,
        subs_fraction: PI.subs_fraction ? PI.subs_fraction[i] : "",
        raw_NES: PI.raw_NES ? PI.raw_NES[i] : null,
        raw_FDR: PI.raw_FDR ? PI.raw_FDR[i] : null,
        raw_p_value: PI.raw_p_value ? PI.raw_p_value[i] : null,
      };
    }
  }
  return null;
}

async function _khSubstrateFor(khid, donor) {
  // Returns {leading, motifs} for one (kinase, donor); both default to "".
  // Empty result means either the shard has no row for this donor (no
  // leading-edge hits) or the kinase isn't in present_human_perdonor_kinase_ids.
  if (!window.SliceCache || !SliceCache.loadHumanPerdonorSubstrate) {
    return {leading: "", motifs: "", klPercentiles: ""};
  }
  try {
    const byDonor = await SliceCache.loadHumanPerdonorSubstrate(khid);
    return byDonor.get(String(donor)) || {leading: "", motifs: "", klPercentiles: ""};
  } catch (e) {
    console.warn("loadHumanPerdonorSubstrate failed", khid, e);
    return {leading: "", motifs: "", klPercentiles: ""};
  }
}

function _khCtrlMean(siteIdx, kind) {
  // kind: "stoich" or "raw"
  const ctrl = _KH.ctrl_donors || [];
  const all = _KH.donors_all || [];
  const arr = kind === "raw" ? _KH.raw_phospho_by_site : _KH.stoich_by_site;
  if (!arr || !arr[siteIdx]) return null;
  const row = arr[siteIdx];
  let sum = 0, n = 0;
  for (const d of ctrl) {
    const di = all.indexOf(d);
    if (di < 0) continue;
    const v = row[di];
    if (v != null && isFinite(v)) { sum += v; n += 1; }
  }
  return n > 0 ? sum / n : null;
}

function _khValAt(siteIdx, donor, kind) {
  const all = _KH.donors_all || [];
  const di = all.indexOf(donor);
  if (di < 0) return null;
  const arr = kind === "raw" ? _KH.raw_phospho_by_site : _KH.stoich_by_site;
  if (!arr || !arr[siteIdx]) return null;
  const v = arr[siteIdx][di];
  return (v == null || !isFinite(v)) ? null : v;
}

function _khFmt(v, digits) {
  if (v == null || !isFinite(v)) return "—";
  return Number(v).toFixed(digits == null ? 3 : digits);
}

function _khRows() {
  const K = _KH.kinases;
  const donors = _KH.donors;
  const ctrlDonors = _KH.ctrl_donors || [];
  const n = K.id.length;
  const out = [];
  const famMap = (typeof META !== "undefined" && META && META.familyMap) || {};
  const _pick = (d, prefix, i) => {
    const v = K[prefix + d + "_vs_CTRLmean"];
    return v ? v[i] : null;
  };
  // (kinase_name, residue) -> CTRL recurrence row from PAYLOAD.human.
  const ctrlRecLookup = new Map();
  for (const r of (_KH.recurrence_ctrl || [])) {
    ctrlRecLookup.set(r.kinase + "|" + r.residue_type, r);
  }
  for (let i = 0; i < n; i++) {
    const nesVec = donors.map(d => _pick(d, "NES_", i));
    const fdrVec = donors.map(d => _pick(d, "FDR_", i));
    const nesCtrl = ctrlDonors.map(d => _pick(d, "NES_", i));
    const fdrCtrl = ctrlDonors.map(d => _pick(d, "FDR_", i));
    const rec = ctrlRecLookup.get(K.name[i] + "|" + K.residue_type[i]);
    const nCtrlSig = rec ? rec.n_donors_sig : 0;
    const nCtrlSigUp = rec ? rec.n_donors_up : 0;
    const nCtrlSigDn = rec ? rec.n_donors_down : 0;
    // CTRL spread: 1 SD of finite CTRL NES values.
    let ctrlSd = null;
    const finite = nesCtrl.filter(v => v != null && isFinite(v));
    if (finite.length >= 2) {
      const m = finite.reduce((a, b) => a + b, 0) / finite.length;
      const v = finite.reduce((a, b) => a + (b - m) * (b - m), 0) / (finite.length - 1);
      ctrlSd = Math.sqrt(v);
    }
    out.push({
      id: K.id[i],
      name: K.name[i],
      gene_symbol: K.gene_symbol[i],
      family: famMap[K.name[i]] || "",
      residue_type: K.residue_type[i],
      n_donors_sig: K.n_donors_sig[i],
      n_donors_up: K.n_donors_up[i],
      n_donors_down: K.n_donors_down[i],
      n_donors_tested: K.n_donors_tested[i],
      n_ctrl_sig: nCtrlSig,
      n_ctrl_sig_up: nCtrlSigUp,
      n_ctrl_sig_down: nCtrlSigDn,
      ctrl_sd: ctrlSd,
      median_nes: K.median_nes[i],
      median_nes_sig_only: K.median_nes_sig_only[i],
      sea_ad_lfc: K.sea_ad_lfc ? K.sea_ad_lfc[i] : null,
      sea_ad_n: K.sea_ad_n_supertypes ? K.sea_ad_n_supertypes[i] : 0,
      sea_ad_direction_agreement: K.sea_ad_direction_agreement ? K.sea_ad_direction_agreement[i] : null,
      _nes: nesVec,
      _fdr: fdrVec,
      _nesCtrl: nesCtrl,
      _fdrCtrl: fdrCtrl,
    });
  }
  return out;
}

let _khRowCache = null;
function _khAllRows() {
  if (!_khRowCache) _khRowCache = _khRows();
  return _khRowCache;
}

// Cohort-level SEA-AD agreement: compares median_nes_sig_only sign to sea_ad_lfc sign.
// Returns "agree" | "disagree" | "na".
function _khSeaAdAgreement(r) {
  const nes = r.median_nes_sig_only;
  const lfc = r.sea_ad_lfc;
  if (nes == null || !isFinite(nes) || nes === 0) return "na";
  if (lfc == null || !isFinite(lfc) || lfc === 0) return "na";
  return (Math.sign(nes) === Math.sign(lfc)) ? "agree" : "disagree";
}

function _khFilter(rows) {
  const q = _KHState.search.trim().toLowerCase();
  const minSig = _KHState.nsigMin;
  const track = _KHState.track;
  const celltype = _KHState.celltype;
  const confidence = _KHState.confidence;
  const specificityTier = _KHState.specificityTier;
  const seaad = _KHState.seaad;
  const adDir = _KHState.adDir;
  const ctrlDir = _KHState.ctrlDir;
  const mode = _KHState.dirMode;
  const fdrThresh = Store.state.filters.fdr;
  return rows.filter(r => {
    if (track && r.residue_type !== track) return false;
    if (r.n_donors_sig < minSig) return false;
    if (q && !(String(r.name).toLowerCase().includes(q)
            || String(r.gene_symbol).toLowerCase().includes(q))) return false;
    if ((celltype || confidence || specificityTier > 0) && _khHasCelltypeSpec()) {
      const summary = _khAttributionSummary(r);
      if (celltype && !summary.rows.some(row => row.cell_type === celltype)) return false;
      if (confidence && _khAttrConfRank(summary.conf) < _khAttrConfRank(confidence)) return false;
      if (specificityTier > 0 && summary.maxTierRank < specificityTier) return false;
    }
    if (seaad && _khSeaAdAgreement(r) !== seaad) return false;
    if (adDir) {
      const [u, d] = _khCountUpDown(r._nes, r._fdr, mode, fdrThresh);
      if (_khDirPattern(u, d) !== adDir) return false;
    }
    if (ctrlDir) {
      const [u, d] = _khCountUpDown(r._nesCtrl, r._fdrCtrl, mode, fdrThresh);
      if (_khDirPattern(u, d) !== ctrlDir) return false;
    }
    return true;
  });
}

function _khSort(rows) {
  const col = _KHState.sortCol;
  const asc = _KHState.sortAsc;
  const cmp = (a, b) => {
    let va, vb;
    if (col === "nes_profile") {
      va = Math.max(...a._nes.map(v => v == null ? -Infinity : Math.abs(v)));
      vb = Math.max(...b._nes.map(v => v == null ? -Infinity : Math.abs(v)));
    } else if (col === "median_nes_sig_only") {
      va = a.median_nes_sig_only == null ? -Infinity : Math.abs(a.median_nes_sig_only);
      vb = b.median_nes_sig_only == null ? -Infinity : Math.abs(b.median_nes_sig_only);
    } else if (col === "n_attributed_celltypes") {
      va = _khAttributionSummary(a).count;
      vb = _khAttributionSummary(b).count;
    } else if (col === "max_specificity_tier") {
      va = _khAttributionSummary(a).maxTierRank;
      vb = _khAttributionSummary(b).maxTierRank;
    } else if (col === "conf") {
      va = _khAttrConfRank(_khAttributionSummary(a).conf);
      vb = _khAttrConfRank(_khAttributionSummary(b).conf);
    } else { va = a[col]; vb = b[col]; }
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === "string") return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    return asc ? (va - vb) : (vb - va);
  };
  const out = rows.slice();
  out.sort(cmp);
  // Tie-break by |median_nes_sig_only| desc when primary col is n_donors_sig.
  if (col === "n_donors_sig") {
    out.sort((a, b) => {
      if (a.n_donors_sig !== b.n_donors_sig)
        return asc ? (a.n_donors_sig - b.n_donors_sig) : (b.n_donors_sig - a.n_donors_sig);
      const am = a.median_nes_sig_only == null ? -Infinity : Math.abs(a.median_nes_sig_only);
      const bm = b.median_nes_sig_only == null ? -Infinity : Math.abs(b.median_nes_sig_only);
      return bm - am;
    });
  } else if (col === "conf") {
    out.sort((a, b) => {
      const ar = _khAttrConfRank(_khAttributionSummary(a).conf);
      const br = _khAttrConfRank(_khAttributionSummary(b).conf);
      if (ar !== br) return asc ? (ar - br) : (br - ar);
      const at = _khAttributionSummary(a).maxTierRank;
      const bt = _khAttributionSummary(b).maxTierRank;
      if (at !== bt) return bt - at;
      const ac = _khAttributionSummary(a).count;
      const bc = _khAttributionSummary(b).count;
      if (ac !== bc) return bc - ac;
      const am = a.median_nes_sig_only == null ? -Infinity : Math.abs(a.median_nes_sig_only);
      const bm = b.median_nes_sig_only == null ? -Infinity : Math.abs(b.median_nes_sig_only);
      return bm - am;
    });
  }
  return out;
}

function _khRenderProfile(r, fdrThresh, donors, donorMask, maxAbs) {
  const _renderCells = (donorList, nesArr, fdrArr) => {
    const cells = [];
    for (let di = 0; di < donorList.length; di++) {
      if (donorMask && !donorMask.has(donorList[di])) continue;
      const nes = nesArr[di];
      const fdrV = fdrArr[di];
      const sig = fdrV != null && fdrV < fdrThresh;
      let bg = "#fff";
      if (nes != null && isFinite(nes) && maxAbs > 0) {
        const a = Math.min(1, Math.abs(nes) / maxAbs);
        const rgb = nes >= 0 ? [197,48,48] : [43,108,176];
        bg = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.15 + 0.85 * a).toFixed(3)})`;
      }
      const tip = nes == null
        ? `${donorList[di]}: n/a`
        : `${donorList[di]}: NES ${nes.toFixed(2)}${fdrV != null ? `, FDR ${fdrV.toExponential(1)}` : ""}${sig ? " (sig)" : ""}`;
      cells.push(`<div class="npc${sig ? " sig" : ""}" style="background:${bg};" title="${_escapeHtml(tip)}"></div>`);
    }
    return cells;
  };
  const adCells = _renderCells(donors, r._nes, r._fdr);
  const ctrlDonors = _KH.ctrl_donors || [];
  const showCtrl = _KHState.showCtrl && ctrlDonors.length;
  const adBlock = `<div class="nes-profile-cell" style="grid-template-columns:repeat(${adCells.length || 1},1fr);">${adCells.join("")}</div>`;
  if (!showCtrl) {
    return `<div class="nes-profile-wrap">${adBlock}</div>`;
  }
  // CTRL group is not filtered by the donor multiselect (which targets AD).
  const ctrlCells = _renderCells(ctrlDonors, r._nesCtrl, r._fdrCtrl);
  const ctrlBlock = `<div class="nes-profile-cell kh-ctrl-group" style="grid-template-columns:repeat(${ctrlCells.length || 1},1fr);" title="CTRL donors scored against the same CTRL mean — muted because they bias toward zero by design.">${ctrlCells.join("")}</div>`;
  return `<div class="nes-profile-wrap">${adBlock}<span class="nes-profile-spacer" aria-hidden="true"></span>${ctrlBlock}</div>`;
}

function renderKinaseHuman() {
  if (!_KH_HAS) return;
  const fdrThresh = Store.state.filters.fdr;
  const donors = _KH.donors;
  const donorMask = _KHState.donors.size ? _KHState.donors : null;
  const rowsAll = _khFilter(_khAllRows());
  // Compute maxAbs over the visible donor subset for color saturation.
  let maxAbs = 0;
  for (const r of rowsAll) {
    for (let di = 0; di < donors.length; di++) {
      if (donorMask && !donorMask.has(donors[di])) continue;
      const v = r._nes[di];
      if (v != null && isFinite(v)) maxAbs = Math.max(maxAbs, Math.abs(v));
    }
  }
  const rows = _khSort(rowsAll);

  const tbody = document.querySelector("#kh-table tbody");
  if (!tbody) return;
  const selKid = Store.state.selection.kinaseHuman;
  const html = rows.map(r => {
    const sigCls = r.id === selKid ? " kh-row-selected" : "";
    const mNES = r.median_nes_sig_only;
    const mStr = (mNES == null || !isFinite(mNES)) ? "—" : mNES.toFixed(2);
    const hasSpec = _khHasCelltypeSpec();
    const attrSummary = hasSpec ? _khAttributionSummary(r) : null;
    return `<tr data-khid="${r.id}" class="${sigCls}" tabindex="0">`
      + `<td>${_escapeHtml(r.name)}</td>`
      + `<td>${_escapeHtml(r.gene_symbol || "")}</td>`
      + `<td>${_escapeHtml(r.family || "")}</td>`
      + `<td>${_escapeHtml(r.residue_type || "")}</td>`
      + `<td>${_khRenderProfile(r, fdrThresh, donors, donorMask, maxAbs)}</td>`
      + `<td>${mStr}</td>`
      + `<td>${r.n_donors_sig}</td>`
      + `<td>${r.n_donors_up}</td>`
      + `<td>${r.n_donors_down}</td>`
      + `<td class="kh-ctrl-col" title="${r.n_ctrl_sig_up || 0} up / ${r.n_ctrl_sig_down || 0} down · CTRL spread ±${r.ctrl_sd == null ? "—" : r.ctrl_sd.toFixed(2)} NES">${r.n_ctrl_sig}</td>`
      + (hasSpec ? `<td class="kh-spec-col">${_khRenderCelltypePills(attrSummary)}</td>` : "")
      + (hasSpec ? `<td class="kh-spec-col">${_khSpecTierBadge(attrSummary.maxTierRank)}</td>` : "")
      + (hasSpec ? `<td class="kh-spec-col">${_khConfBadge(attrSummary.conf, attrSummary)}</td>` : "")
      + `</tr>`;
  }).join("");
  tbody.innerHTML = html;
  // Toggle CTRL column visibility on the table.
  const tbl = document.getElementById("kh-table");
  if (tbl) tbl.classList.toggle("show-ctrl-off", !_KHState.showCtrl);
  const count = document.getElementById("kh-count");
  if (count) count.textContent = `${rows.length} kinases · ${donors.length} donors · ` +
                                 (donorMask ? `${donorMask.size} selected` : "all donors");
  // Re-render detail panel if a selection is active.
  if (selKid != null) _khRenderDetail(selKid);
}

function updateKinaseHumanSelection(khid) {
  document.querySelectorAll("#kh-table tbody tr").forEach(tr => {
    tr.classList.toggle("kh-row-selected", String(tr.dataset.khid) === String(khid));
  });
  if (khid != null) _khRenderDetail(khid);
  else {
    const det = document.getElementById("kh-detail");
    if (det) det.innerHTML = `<div class="muted">Select a kinase to see per-donor details.</div>`;
  }
}

function _khRenderDetail(khid) {
  const det = document.getElementById("kh-detail");
  if (!det) return;
  const rows = _khAllRows();
  const r = rows.find(x => x.id === khid);
  if (!r) { det.innerHTML = `<div class="muted">Kinase not found.</div>`; return; }

  const ctrlDonorsAll = _KH.ctrl_donors || [];
  const donorPool = _KH.donors.concat(ctrlDonorsAll);
  // Initialize auditDonor on first selection: prefer first case donor with NES data.
  if (_KHState.auditDonor == null || !donorPool.includes(_KHState.auditDonor)) {
    const donors = _KH.donors;
    let pick = donors[0];
    for (let di = 0; di < donors.length; di++) {
      if (r._nes[di] != null) { pick = donors[di]; break; }
    }
    _KHState.auditDonor = pick;
  }

  const mNES = r.median_nes_sig_only;
  const recSummary = `${r.n_donors_sig}/${r.n_donors_tested} donors sig` +
    ` · up ${r.n_donors_up} · down ${r.n_donors_down}` +
    ` · median NES (sig) ${(mNES == null || !isFinite(mNES)) ? "—" : mNES.toFixed(2)}`;
  const seaLfc = r.sea_ad_lfc;
  const seaAgree = _khSeaAdAgreement(r);
  const seaLine = (seaLfc == null || !isFinite(seaLfc))
    ? `SEA-AD: <span class="muted">no coverage for ${_escapeHtml(r.gene_symbol || r.name)}</span>`
    : `SEA-AD median LFC ${seaLfc.toFixed(2)} across ${r.sea_ad_n || 0} supertypes`
      + ` · ${seaAgree === "agree" ? `<span style="color:#1b5e20;">agrees</span>` : seaAgree === "disagree" ? `<span style="color:#b71c1c;">disagrees</span>` : `<span class="muted">no signed NES</span>`}`
      + ` with median NES (sig)`;

  const tabsHtml = _KH_AUDIT_TABS.map(t =>
    `<button type="button" class="kh-audit-tab${t.id === _KHState.auditTab ? " active" : ""}"`
    + ` data-kh-audit-tab="${t.id}">${_escapeHtml(t.label)}</button>`
  ).join("");
  // Donor selector applies to all sub-tabs (score uses it for the per-donor scorecard + running enrichment).
  const needsDonor = true;
  const _opt = (d, group) =>
    `<option value="${_escapeHtml(d)}"${d === _KHState.auditDonor ? " selected" : ""}>${_escapeHtml(d)}${group ? " (CTRL)" : ""}</option>`;
  const adOpts = _KH.donors.map(d => _opt(d, false)).join("");
  const ctrlOpts = ctrlDonorsAll.length
    ? `<optgroup label="Control">${ctrlDonorsAll.map(d => _opt(d, true)).join("")}</optgroup>`
    : "";
  const donorSelHtml = needsDonor
    ? `<div class="kh-audit-toolbar"><label class="ke-filter-label">Donor `
      + `<select id="kh-audit-donor">`
      + (ctrlDonorsAll.length ? `<optgroup label="AD">${adOpts}</optgroup>` : adOpts)
      + ctrlOpts
      + `</select></label></div>`
    : "";

  det.innerHTML = `
    <h3 style="margin:0 0 6px 0;">${_escapeHtml(r.name)} <span class="muted" style="font-weight:400;">(${_escapeHtml(r.gene_symbol || "")} · ${_escapeHtml(r.residue_type || "")})</span></h3>
    <div class="muted" style="margin-bottom:2px;">${_escapeHtml(recSummary)}</div>
    <div class="muted" style="margin-bottom:8px;">${seaLine}</div>
    <nav class="kh-audit-tabs" role="tablist">${tabsHtml}</nav>
    ${donorSelHtml}
    <div id="kh-audit-body" class="kh-audit-body"></div>
  `;
  // Wire sub-tab buttons.
  det.querySelectorAll(".kh-audit-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      _KHState.auditTab = btn.dataset.khAuditTab;
      _khRenderDetail(khid);
    });
  });
  const ds = document.getElementById("kh-audit-donor");
  if (ds) ds.addEventListener("change", e => {
    _KHState.auditDonor = e.target.value;
    _khRenderAuditBody(r);
  });
  _khRenderAuditBody(r);
}

function _khRenderAuditBody(r) {
  const body = document.getElementById("kh-audit-body");
  if (!body) return;
  const tab = _KHState.auditTab;
  if (tab === "trace")    return _khRenderTrace(body, r);
  if (tab === "prep")     return _khRenderPrep(body, r);
  if (tab === "score")    return _khRenderScore(body, r);
  if (tab === "attribution") return _khRenderAttribution(body, r);
}

async function _khRenderTrace(body, r) {
  if (!_KH.sites) { body.innerHTML = `<div class="muted">Site matrices not available in payload.</div>`; return; }
  _khEnsureSiteIndex();
  const donor = _KHState.auditDonor;
  const pd = _khPerdonorFor(r.id, donor);
  if (!pd) {
    body.innerHTML = `<p class="kinase-stage-note">No per-donor MEA record for ${_escapeHtml(r.name)} in donor ${_escapeHtml(donor)}.</p>`;
    return;
  }
  body.innerHTML = `<p class="muted" style="padding:0.5em">Loading leading-edge substrates…</p>`;
  const sub = await _khSubstrateFor(r.id, donor);
  if (!sub.leading) {
    body.innerHTML = `<p class="kinase-stage-note">No leading-edge substrates recorded for ${_escapeHtml(r.name)} in donor ${_escapeHtml(donor)} (may not have reached MEA significance).</p>`;
    return;
  }
  const motifs = sub.leading.split(";").map(s => _normMotif(s)).filter(Boolean);
  // (motif, kl_percentile) parallel-pair index from the full substrate set
  // (sub.motifs + sub.klPercentiles). 0-100 substrate-vs-kinase agreement
  // strength from kinase_library — higher = stronger motif match.
  const klByMotif = new Map();
  const subsArr = (sub.motifs || "").split(";").map(_normMotif);
  const klArr = (sub.klPercentiles || "").split(";");
  for (let i = 0; i < subsArr.length; i++) {
    if (!subsArr[i]) continue;
    const v = Number(klArr[i]);
    if (Number.isFinite(v)) klByMotif.set(subsArr[i], v);
  }
  const S = _KH.sites;
  const seen = new Set();
  const rows = [];
  for (const m of motifs) {
    const idxs = _khSiteByMotif.get(m) || [];
    for (const i of idxs) {
      if (S.residue_type[i] !== r.residue_type) continue;
      if (seen.has(i)) continue;
      seen.add(i);
      const stoichD = _khValAt(i, donor, "stoich");
      const rawD    = _khValAt(i, donor, "raw");
      const stoichCM = _khCtrlMean(i, "stoich");
      const rawCM    = _khCtrlMean(i, "raw");
      const delta = (stoichD != null && stoichCM != null) ? (stoichD - stoichCM) : null;
      rows.push({
        site_id: S.site_id[i], motif: S.motif[i], gene: S.gene_symbol[i], pos: S.site_position[i],
        kl_percentile: klByMotif.get(_normMotif(S.motif[i])) ?? null,
        raw_d: rawD, raw_cm: rawCM, stoich_d: stoichD, stoich_cm: stoichCM, delta,
      });
    }
  }
  rows.sort((a, b) => {
    const da = a.delta == null ? -Infinity : Math.abs(a.delta);
    const db = b.delta == null ? -Infinity : Math.abs(b.delta);
    return db - da;
  });
  const motif = (PAYLOAD.kinase_motifs || {})[r.name] || null;
  const logoBlock = SequenceLogo.buildBlock(r.name, motif, "kh-trace-logo");
  const note = `<p class="kinase-stage-note">Per-site measurement trace for ${_escapeHtml(r.name)} (${_escapeHtml(r.residue_type)}) in donor <strong>${_escapeHtml(donor)}</strong>. Each row is a leading-edge substrate site for this (kinase, donor). CTRL mean is averaged across ${_KH.ctrl_donors.length} control donors; Δ = donor stoichiometry − CTRL mean. <code>kl_%</code> is the kinase-library substrate percentile: 0-100, where higher means this motif scores higher than that many phosphosites in the library's reference for this kinase.</p>`;
  const head = `<thead><tr>`
    + `<th>site_id</th><th>gene</th><th>motif</th><th title="Kinase library substrate percentile (0-100, higher = stronger match).">kl_%</th><th>raw_phospho (donor)</th>`
    + `<th>raw_phospho (CTRL μ)</th><th>stoich (donor)</th><th>stoich (CTRL μ)</th><th>Δ stoich</th>`
    + `</tr></thead>`;
  const tbody = `<tbody>` + rows.map(rw =>
    `<tr><td>${_escapeHtml(rw.site_id)}</td>`
    + `<td>${_escapeHtml(rw.gene)}</td>`
    + `<td>${_escapeHtml(rw.motif)}</td>`
    + `<td>${_khFmt(rw.kl_percentile, 1)}</td>`
    + `<td>${_khFmt(rw.raw_d, 3)}</td>`
    + `<td>${_khFmt(rw.raw_cm, 3)}</td>`
    + `<td>${_khFmt(rw.stoich_d, 3)}</td>`
    + `<td>${_khFmt(rw.stoich_cm, 3)}</td>`
    + `<td>${_khFmt(rw.delta, 3)}</td></tr>`
  ).join("") + `</tbody>`;
  body.innerHTML = logoBlock + note + `<div class="kh-audit-tablewrap"><table class="data-table">${head}${tbody}</table></div>`;
  if (motif) SequenceLogo.render(document.getElementById("kh-trace-logo"), motif);
}

function _khRenderPrep(body, r) {
  const donor = _KHState.auditDonor;
  const residue = r.residue_type;
  const contrast = donor + "_vs_CTRLmean";
  const shift = (_KH.global_shift || []).find(g => g.donor === donor && g.residue_type === residue) || null;
  const shiftLine = shift
    ? `Median stoichiometry LFC across ${donor}'s ranked ${residue} sites: <strong>${_khFmt(shift.median_shift, 4)}</strong>. Subtracted from every site before GSEA to center the prerank at zero.`
    : `<span class="muted">No global-shift record for ${donor} / ${residue}.</span>`;

  // Winsorized sites for (donor, residue) — rebuild as row dicts matching the mouse column schema.
  const W = _KH.winsor || null;
  const wRows = [];
  if (W) {
    for (let i = 0; i < W.donor.length; i++) {
      if (W.donor[i] !== donor || W.residue_type[i] !== residue) continue;
      wRows.push({
        contrast: contrast,
        original_lfc: W.original_lfc[i],
        clipped_lfc: W.clipped_lfc[i],
        lower_bound: W.lower_bound[i],
        upper_bound: W.upper_bound[i],
        site_id: W.site_id[i],
        gene_symbol: W.gene_symbol[i],
      });
    }
  }
  const bounds = wRows.length
    ? `bounds [${_khFmt(wRows[0].lower_bound, 3)}, ${_khFmt(wRows[0].upper_bound, 3)}] · ${wRows.length.toLocaleString()} sites clipped across ${donor}/${residue}`
    : `<span class="muted">No winsorization receipts for ${donor} / ${residue}.</span>`;

  body.innerHTML =
    `<section class="audit-panel"><h4>Step 1 · Global shift <span class="muted">(mea_global_shift.csv)</span></h4>`
    + `<p class="kinase-stage-note">${shiftLine}</p>`
    + `<div id="kh-mea-shift"></div></section>`
    + `<section class="audit-panel"><h4>Step 2 · Winsorization <span class="muted">(winsorized_sites.csv)</span></h4>`
    + `<p class="kinase-stage-note">Centered LFCs clipped to the 1st/99th percentile so individual sites cannot dominate the prerank. ${bounds}.</p>`
    + `<div id="kh-mea-winsor"></div></section>`;

  const shiftRows = shift ? [{
    contrast: contrast,
    median_shift: shift.median_shift,
    mean_before: shift.mean_before,
    pct_pos_before: shift.pct_pos_before,
    pct_pos_after: shift.pct_pos_after,
  }] : [];
  new AuditTable("kh-mea-shift", {
    tableKey: "mea_global_shift", rows: shiftRows,
    columns: ["contrast", "median_shift", "mean_before", "pct_pos_before", "pct_pos_after"],
    fullSourceKey: false,
  }).render();
  new AuditTable("kh-mea-winsor", {
    tableKey: "winsorized_sites", rows: wRows,
    columns: ["contrast", "site_id", "gene_symbol", "original_lfc", "clipped_lfc",
              "lower_bound", "upper_bound"],
    fullSourceKey: false,
  }).render();
}

function _khBuildPrerank(donor, residue) {
  // Build the prerank list of (motif, clipped_lfc, site_id, gene_symbol) for (donor, residue):
  //   raw_lfc = stoich_by_site[siteIdx][donorIdx]
  //   centered = raw_lfc − median_shift
  //   clipped = clamp(centered, [lower_bound, upper_bound])
  // Returns null if any input is missing.
  if (!_KH || !_KH.sites || !_KH.stoich_by_site) return null;
  const all = _KH.donors_all || [];
  const di = all.indexOf(donor);
  if (di < 0) return null;
  const shift = (_KH.global_shift || []).find(g => g.donor === donor && g.residue_type === residue);
  if (!shift || shift.median_shift == null) return null;
  const W = _KH.winsor || null;
  let lo = null, hi = null;
  if (W) {
    for (let i = 0; i < W.donor.length; i++) {
      if (W.donor[i] === donor && W.residue_type[i] === residue) {
        lo = Number(W.lower_bound[i]); hi = Number(W.upper_bound[i]);
        break;
      }
    }
  }
  const med = Number(shift.median_shift);
  const S = _KH.sites;
  const N = S.site_id.length;
  const out = [];
  for (let i = 0; i < N; i++) {
    if (S.residue_type[i] !== residue) continue;
    const row = _KH.stoich_by_site[i];
    if (!row) continue;
    const raw = row[di];
    if (raw == null || !isFinite(raw)) continue;
    let v = raw - med;
    if (lo != null && hi != null) v = Math.min(hi, Math.max(lo, v));
    out.push({
      site_id: S.site_id[i], gene_symbol: S.gene_symbol[i],
      motif: _normMotif(S.motif[i]), clipped: v,
    });
  }
  out.sort((a, b) => b.clipped - a.clipped);
  return out;
}

function _khRunningES(ranked, substrateMotifs) {
  // Mirrors mouse _computeRunningES (kinase_audit.js): weighted GSEA walk.
  const N = ranked.length;
  if (!N || !substrateMotifs || !substrateMotifs.size) return null;
  const isHit = new Array(N);
  let Nh = 0;
  let hitSum = 0;
  for (let i = 0; i < N; i++) {
    const h = substrateMotifs.has(ranked[i].motif);
    isHit[i] = h;
    if (h) { Nh += 1; hitSum += Math.abs(ranked[i].clipped); }
  }
  if (!Nh || hitSum === 0) return null;
  const missRate = 1 / (N - Nh || 1);
  const running = new Array(N);
  const hitIndices = [];
  let cur = 0, peak = 0, peakIdx = 0;
  for (let i = 0; i < N; i++) {
    if (isHit[i]) {
      cur += Math.abs(ranked[i].clipped) / hitSum;
      hitIndices.push(i);
    } else {
      cur -= missRate;
    }
    running[i] = cur;
    if (Math.abs(cur) > Math.abs(peak)) { peak = cur; peakIdx = i; }
  }
  const leadingEdge = peak >= 0
    ? hitIndices.filter(i => i <= peakIdx)
    : hitIndices.filter(i => i >= peakIdx);
  return {N, Nh, running, hitIndices, peakES: peak, peakIdx, leadingEdge};
}

async function _khRenderRunningEnrichment(hostId, r, donor) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const pd = _khPerdonorFor(r.id, donor);
  if (!pd) {
    host.innerHTML = `<div class="muted" style="padding:1em">No per-donor MEA record for ${_escapeHtml(r.name)} in ${_escapeHtml(donor)} — running enrichment unavailable.</div>`;
    return;
  }
  host.innerHTML = `<div class="muted" style="padding:1em">Loading substrate-set motifs…</div>`;
  const sub = await _khSubstrateFor(r.id, donor);
  const motifsStr = sub.motifs || "";
  if (!motifsStr) {
    host.innerHTML = `<div class="muted" style="padding:1em">No substrate-set motifs recorded for ${_escapeHtml(r.name)} in ${_escapeHtml(donor)} (mea_substrate_sets.csv). Re-run <code>python alz/ingest_mukesh_perdonor.py</code>.</div>`;
    return;
  }
  const ranked = _khBuildPrerank(donor, r.residue_type);
  if (!ranked || !ranked.length) {
    host.innerHTML = `<div class="muted" style="padding:1em">Prerank could not be built (missing site matrix, global shift, or winsor bounds for ${_escapeHtml(donor)} / ${_escapeHtml(r.residue_type)}).</div>`;
    return;
  }
  const subs = new Set(motifsStr.split(";").map(_normMotif).filter(Boolean));
  const walk = _khRunningES(ranked, subs);
  if (!walk) {
    host.innerHTML = `<div class="muted" style="padding:1em">Substrate-set motifs did not match any ranked site for ${_escapeHtml(r.name)} / ${_escapeHtml(donor)}.</div>`;
    return;
  }
  const ranks = new Array(walk.N);
  for (let i = 0; i < walk.N; i++) ranks[i] = i + 1;
  const hitX = walk.hitIndices.map(i => i + 1);
  const hitY = walk.hitIndices.map(i => walk.running[i]);
  const hitText = walk.hitIndices.map(i => {
    const e = ranked[i];
    return `rank ${i + 1}<br>${_escapeHtml(e.gene_symbol || "")} · ${_escapeHtml(e.motif)}<br>clipped LFC ${e.clipped.toFixed(3)}<br>running ES ${walk.running[i].toFixed(3)}`;
  });
  const peakX = walk.peakIdx + 1;
  const peakY = walk.peakES;
  const leShape = walk.peakES >= 0
    ? {x0: 1, x1: peakX}
    : {x0: peakX, x1: walk.N};
  Plotly.react(hostId, [
    {type:"scatter", mode:"lines", x: ranks, y: walk.running,
     line:{color:"#1f77b4", width:1.5}, name:"running ES", hoverinfo:"skip"},
    {type:"scatter", mode:"markers", x: hitX, y: hitY,
     marker:{color:"#1f77b4", size:5, opacity:0.9}, name:"hit",
     text: hitText, hovertemplate:"%{text}<extra></extra>"},
    {type:"scatter", mode:"markers", x:[peakX], y:[peakY],
     marker:{color:"#000", size:9, symbol:"diamond"}, name:"peak ES",
     hovertemplate:`peak ES ${peakY.toFixed(3)} at rank ${peakX}<extra></extra>`},
  ], {
    margin:{l:50, r:10, t:30, b:40}, height:300,
    showlegend:false,
    annotations:[{
      x: peakX, y: peakY, xref:"x", yref:"y",
      text: `peak ES ${peakY.toFixed(3)} at rank ${peakX}<br>leading edge: ${walk.leadingEdge.length} of ${walk.Nh} hits`,
      showarrow:true, arrowhead:2, ax: 30, ay: peakY >= 0 ? -40 : 40,
      font:{size:11},
    }],
    shapes:[{
      type:"rect", xref:"x", yref:"paper",
      x0: leShape.x0, x1: leShape.x1, y0: 0, y1: 1,
      fillcolor:"#1f77b4", opacity:0.08, line:{width:0},
    }, {
      type:"line", xref:"x", yref:"y",
      x0: 1, x1: walk.N, y0: 0, y1: 0,
      line:{color:"#999", width:1, dash:"dot"},
    }],
    xaxis:{title:"prerank rank (1 = most up-shifted)", range:[1, walk.N]},
    yaxis:{title:"running ES", zeroline:false},
  }, {displaylogo:false, responsive:true});
}

function _khRenderNESAcrossDonors(hostId, r, selectedDonor) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const donors = _KH.donors;
  const ctrlDonors = _KH.ctrl_donors || [];
  const showCtrl = _KHState.showCtrl && ctrlDonors.length;
  const fdrThresh = Store.state.filters.fdr;
  const _hexToRgba = (hex, alpha) => {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || "");
    if (!m) return hex;
    return `rgba(${parseInt(m[1],16)},${parseInt(m[2],16)},${parseInt(m[3],16)},${alpha})`;
  };
  const adColors = donors.map((_, di) => {
    const nes = r._nes[di];
    const fdr = r._fdr[di];
    const base = (nes != null && nes < 0) ? "#1f5fa6" : "#c8261c";
    const sig = fdr != null && fdr < fdrThresh;
    return sig ? base : _hexToRgba(base, 0.28);
  });
  const selIdx = donors.indexOf(selectedDonor);
  const adOutlines = donors.map((_, i) => i === selIdx ? "#000" : "rgba(0,0,0,0)");
  const adLineW = donors.map((_, i) => i === selIdx ? 2.5 : 0);
  const traces = [{
    type:"bar", x: donors, y: r._nes.map(v => v == null ? 0 : v),
    marker:{color: adColors, line:{color: adOutlines, width: adLineW}},
    name:"AD",
    hovertemplate:"%{x}<br>NES %{y:.2f}<extra></extra>",
  }];
  const layoutShapes = [];
  const layoutAnnotations = [];
  if (showCtrl) {
    // CTRL bars use the same red/blue palette as AD so sig-vs-insig is
    // legible. Hatched fill pattern marks them as the CTRL group instead
    // of relying on a desaturated palette (which buried the signal).
    const ctrlColors = ctrlDonors.map((_, di) => {
      const nes = r._nesCtrl[di];
      const fdr = r._fdrCtrl[di];
      const sig = fdr != null && fdr < fdrThresh;
      const base = (nes != null && nes < 0) ? "#1f5fa6" : "#c8261c";
      return sig ? base : _hexToRgba(base, 0.28);
    });
    const ctrlSelIdx = ctrlDonors.indexOf(selectedDonor);
    const ctrlOutlines = ctrlDonors.map((_, i) => i === ctrlSelIdx ? "#000" : "rgba(0,0,0,0)");
    const ctrlLineW = ctrlDonors.map((_, i) => i === ctrlSelIdx ? 2.5 : 0);
    traces.push({
      type:"bar", x: ctrlDonors, y: r._nesCtrl.map(v => v == null ? 0 : v),
      marker:{
        color: ctrlColors,
        line:{color: ctrlOutlines, width: ctrlLineW},
        pattern:{shape:"/", size:5, solidity:0.45, fgcolor:"#ffffff"},
      },
      name:"CTRL",
      hovertemplate:"%{x} (CTRL)<br>NES %{y:.2f}<extra></extra>",
    });
    // Dashed divider between AD and CTRL groups (placed between the last
    // AD x category and the first CTRL x category).
    if (donors.length && ctrlDonors.length) {
      layoutShapes.push({
        type:"line", xref:"x", yref:"paper",
        x0: donors.length - 0.5, x1: donors.length - 0.5, y0: 0, y1: 1,
        line:{color:"#b0bec5", width:1, dash:"dash"},
      });
    }
    // CTRL spread annotation (1 SD over the CTRL NES values).
    if (r.ctrl_sd != null && isFinite(r.ctrl_sd)) {
      layoutAnnotations.push({
        xref:"paper", yref:"paper", x: 1.0, y: 1.02, xanchor:"right", yanchor:"bottom",
        text: `CTRL spread = ±${r.ctrl_sd.toFixed(2)} NES`,
        showarrow:false, font:{size:11, color:"#546e7a"},
      });
    }
  }
  Plotly.react(hostId, traces, {
    margin:{l:40, r:10, t:24, b:60}, height:220,
    yaxis:{zeroline:true, zerolinecolor:"#bbb", title:"NES"},
    xaxis:{tickangle:-35, type:"category"},
    showlegend:false,
    barmode:"group",
    shapes: layoutShapes,
    annotations: layoutAnnotations,
  }, {displaylogo:false, responsive:true}).then(() => {
    if (host.on && !host.__khTrajWired) {
      host.__khTrajWired = true;
      host.on("plotly_click", (ev) => {
        const pts = ev && ev.points ? ev.points : null;
        if (!pts || !pts[0]) return;
        const target = pts[0].x;
        const ctrlList = _KH.ctrl_donors || [];
        if (_KH.donors.includes(target) || ctrlList.includes(target)) {
          _KHState.auditDonor = target;
          _khRenderDetail(r.id);
        }
      });
    }
  });
}

function _khRenderScore(body, r) {
  const donor = _KHState.auditDonor;
  const pd = _khPerdonorFor(r.id, donor);
  const fdrThresh = Store.state.filters.fdr;
  const nesVal = pd ? pd.NES : null;
  const fdrVal = pd ? pd.FDR : null;
  const esVal = pd ? pd.ES : null;
  const pVal = pd ? pd.p_value : null;
  const subsFrac = pd ? pd.subs_fraction : "";
  const tier = (() => {
    if (fdrVal == null || !isFinite(fdrVal)) return {label: "no FDR", cls: "muted"};
    if (fdrVal < fdrThresh) return {label: `FDR ${fdrVal.toFixed(3)} · passes ${fdrThresh}`, cls: "chip-pass"};
    if (fdrVal < fdrThresh * 2) return {label: `FDR ${fdrVal.toFixed(3)} · borderline`, cls: "chip-borderline"};
    return {label: `FDR ${fdrVal.toFixed(3)} · fails ${fdrThresh}`, cls: "chip-fail"};
  })();
  const nesColor = (nesVal == null || !isFinite(nesVal)) ? "#666"
    : (nesVal > 0 ? "#c8261c" : "#1f5fa6");
  const nesText = (nesVal == null || !isFinite(nesVal)) ? "—" : nesVal.toFixed(2);
  const rawNes = pd ? pd.raw_NES : null;
  const rawFdr = pd ? pd.raw_FDR : null;
  const rawP   = pd ? pd.raw_p_value : null;
  const rawAvailable = rawNes != null && isFinite(rawNes);

  const num = (v) => (v == null || v === "" || !isFinite(Number(v))) ? null : Number(v);
  const fmt = (v, d=3) => v == null ? "—" : Number(v).toFixed(d);
  const fmtSigned = (v, d=3) => v == null ? "—" : (v > 0 ? "+" : "") + Number(v).toFixed(d);
  const delta = (a, b) => (a == null || b == null) ? null : a - b;
  // ES and Subs-fraction are not stored for the raw track; only NES / p / FDR are compared.
  const cmp = [
    {metric: "NES",          stoich: num(nesVal),  raw: num(rawNes)},
    {metric: "p-value",      stoich: num(pVal),    raw: num(rawP)},
    {metric: "FDR",          stoich: num(fdrVal),  raw: num(rawFdr)},
  ];
  const cmpHtml = cmp.map(row => {
    const s = row.stoich, r = row.raw;
    const d = delta(s, r);
    const dgts = row.metric === "p-value" ? 4 : (row.metric === "FDR" ? 3 : 2);
    return `<tr><td>${_escapeHtml(row.metric)}</td>`
      + `<td>${fmt(s, dgts)}</td>`
      + `<td>${fmt(r, dgts)}</td>`
      + `<td>${fmtSigned(d, dgts)}</td></tr>`;
  }).join("");
  const cmpTable = rawAvailable
    ? `<div class="kh-audit-tablewrap"><table class="data-table">`
      + `<thead><tr><th>metric</th><th>stoich</th><th>raw</th><th>Δ (stoich − raw)</th></tr></thead>`
      + `<tbody>${cmpHtml}</tbody></table></div>`
    : `<div class="muted" style="padding:1em">Per-donor raw-phospho MEA not yet exported for this kinase / donor. Re-run <code>python alz/ingest_mukesh_perdonor.py</code> to regenerate the <code>mea_perdonor_raw{,_pY}.csv</code> files.</div>`;

  body.innerHTML =
    `<p class="kinase-stage-note">Per-donor MEA score for ${_escapeHtml(r.name)} (${_escapeHtml(r.residue_type)}) on ${_escapeHtml(donor)} vs CTRL mean.</p>`
    + `<section class="audit-panel"><h4>Score for ${_escapeHtml(donor)}</h4>`
    + `<div class="mea-scorecard">`
    +   `<div class="mea-score-nes" style="color:${nesColor}">`
    +     `<div class="mea-score-label">NES</div>`
    +     `<div class="mea-score-value">${nesText}</div>`
    +     `<div class="mea-score-chip ${tier.cls}">${_escapeHtml(tier.label)}</div>`
    +   `</div>`
    +   `<dl class="mea-score-stats">`
    +     `<dt>ES</dt><dd>${_khFmt(esVal, 3)}</dd>`
    +     `<dt>p-value</dt><dd>${_khFmt(pVal, 4)}</dd>`
    +     `<dt>Substrates tested</dt><dd>${_escapeHtml(subsFrac || "—")}<span class="muted"> (kinase substrates &cap; donor prerank)</span></dd>`
    +     `<dt>Raw phospho NES</dt><dd>${rawAvailable ? fmt(rawNes, 2) : "—"}${rawAvailable ? `<span class="muted"> · FDR ${fmt(rawFdr, 3)}</span>` : `<span class="muted"> (re-run ingest_mukesh_perdonor.py for raw track)</span>`}</dd>`
    +   `</dl>`
    + `</div></section>`
    + `<section class="audit-panel"><h4>Running enrichment for ${_escapeHtml(donor)}</h4>`
    + `<p class="kinase-stage-note">GSEA walk recomputed at view time from the cached prerank. The curve steps up at every substrate hit (weighted by |clipped LFC|) and down at every miss. Peak ES and the leading-edge prefix are marked. Hit set is this kinase's full substrate motif set (<code>mea_substrate_sets.csv</code>) restricted to this donor's prerank — same set GSEA used to score the kinase. Tie-breaking among sites with duplicated clipped values may differ from gseapy's internal order.</p>`
    + `<div id="kh-mea-running" style="height:300px"></div></section>`
    + `<section class="audit-panel"><h4>NES across all donors</h4>`
    + `<p class="kinase-stage-note">Per-donor NES bars. Full saturation when FDR &lt; ${fdrThresh}, faded when not significant. The selected donor is outlined in black. Click a bar to switch the selected donor.</p>`
    + `<div id="kh-mea-trajectory" style="height:220px"></div></section>`
    + `<section class="audit-panel"><h4>Stoichiometry vs raw phospho for ${_escapeHtml(donor)}</h4>`
    + `<p class="kinase-stage-note">Per-metric comparison of the same kinase × donor scored against two preprocessing tracks. Stoichiometry is primary; raw phospho is the sensitivity check. Δ = stoichiometry − raw. Sign-flips or significance divergence flag abundance-driven vs activity-driven signals.</p>`
    + cmpTable
    + `</section>`;

  _khRenderRunningEnrichment("kh-mea-running", r, donor);
  _khRenderNESAcrossDonors("kh-mea-trajectory", r, donor);
}

// ---------------------------------------------------------------------------
// Attribution helpers — human mirror of the mouse Attribution sub-tab.
//
// Row spine on the human side is the per-kinase ranked list of human reference
// cell types (Levy T5 clusters rolled up from SEA-AD MTG supertypes and Allen
// HBCA superclusters) from celltype_specificity.ranked_by_kinase. Columns mirror the mouse Attribution
// tab in spirit; columns that have no human equivalent (Song LFC, Decomp NES /
// FDR, vs Bulk) are omitted rather than re-invented. The contrast-varying
// signal is the per-donor kinase NES from perdonor_index at the currently
// selected donor — broadcast across all rows of the kinase.
// ---------------------------------------------------------------------------

// Whether the celltype_specificity block is present in the current payload.
// PAYLOAD is loaded asynchronously, so this must be evaluated at render time.
function _khHasCelltypeSpec() {
  return !!(_KH_HAS && _KH && _KH.celltype_specificity);
}

const _KH_REF_LABEL = {
  "seaad_mtg":  "SEA-AD MTG",
  "allen_hbca": "Allen HBCA",
};

function _khAttributionKeys(rOrName) {
  if (rOrName && typeof rOrName === "object") {
    const keys = [rOrName.gene_symbol, rOrName.name]
      .filter(v => v != null && String(v).trim() !== "")
      .map(v => String(v).trim());
    return Array.from(new Set(keys));
  }
  return [String(rOrName || "").trim()].filter(Boolean);
}

function _khRankedForReference(rOrName, reference) {
  if (!_khHasCelltypeSpec()) return [];
  const spec = _KH.celltype_specificity;
  if (!spec || !spec[reference] || !spec[reference].ranked_by_kinase) return [];
  const rankedByKinase = spec[reference].ranked_by_kinase;
  for (const key of _khAttributionKeys(rOrName)) {
    if (rankedByKinase[key]) return rankedByKinase[key];
  }
  return [];
}

// Retrieve top-N cell types for a kinase from a given reference. Reads the
// full ranked list emitted by build_celltype_specificity_payload and slices
// to top-N for compact callers (e.g. the kinase row preview columns).
// Returns [] when data is unavailable.
function _khTopCelltypes(rOrName, reference, n) {
  return _khRankedForReference(rOrName, reference).slice(0, n || 8);
}

// Render the top-1 cell type as a compact table cell string.
function _khTopCelltypeCell(kinaseName, reference) {
  const tops = _khTopCelltypes(kinaseName, reference, 1);
  if (!tops.length) return `<span class="muted">—</span>`;
  const t = tops[0];
  const scoreStr = t.score != null ? t.score.toFixed(2) : "—";
  const source = t.source_celltypes && t.source_celltypes.length
    ? `; source labels: ${t.source_celltypes.join(", ")}`
    : "";
  return `<span title="${_escapeHtml(t.celltype)} (log₂-ratio ${scoreStr}${source})">${_escapeHtml(t.celltype)}</span>`;
}

function _khTopCelltypeName(kinaseName, reference) {
  const tops = _khTopCelltypes(kinaseName, reference, 1);
  return tops.length ? String(tops[0].celltype || "") : "";
}

const KH_ATTR_COLS = [
  {key:"cell_type",            label:"Cell type",   type:"str", group:"id",
   title:"Levy T5 cluster. SEA-AD supertypes and HBCA superclusters are rolled up to this shared nomenclature before ranking."},
  {key:"confidence_tier",      label:"Conf",        type:"conf", group:"attr",
   title:"Human mirror of the mouse confidence tier: high / moderate / low / none. High requires both human location evidence ≥ log2(2) and |SEA-AD LFC| ≥ 0.1."},
  {key:"specificity",          label:"Location",    type:"num", group:"attr",
   title:"Human expression location evidence: log2(cell-type mean / reference-wide mean). > 0 means enriched in this cell type."},
  {key:"specificity_tier_rank",label:"Tier",        type:"num", group:"attr",
   title:"Location bucket by multiple of uniform: ≥10× / ≥5× / ≥2× / ≥1×."},
  {key:"mean_log2_expression", label:"log2 expr",   type:"num", group:"attr",
   title:"Mean log2 expression in this cell type. Low absolute expression can make location enrichment less interpretable."},
  {key:"sea_ad_lfc",           label:"SEA-AD LFC",  type:"num", group:"attr",
   title:"SEA-AD AD-vs-control LFC for this kinase in this SEA-AD supertype. HBCA rows have no SEA-AD analog."},
  {key:"donor_nes",            label:"Donor NES",   type:"num", group:"activity",
   title:"Per-donor kinase MEA NES for the selected donor. Same value is broadcast to each row for this kinase."},
  {key:"donor_fdr",            label:"Donor FDR",   type:"num", group:"activity",
   title:"Per-donor kinase MEA FDR for the selected donor. Bold values pass < 0.25."},
];

function _khAttrConfRank(conf) {
  if (conf === "high") return 3;
  if (conf === "moderate") return 2;
  if (conf === "low") return 1;
  return 0;
}

function _khSpecTierRank(score) {
  if (score == null || !isFinite(score)) return null;
  if (score >= Math.log2(10)) return 4;
  if (score >= Math.log2(5)) return 3;
  if (score >= Math.log2(2)) return 2;
  if (score >= 0) return 1;
  return 0;
}

function _khSpecTierLabel(rank) {
  if (rank >= 4) return "≥10×";
  if (rank >= 3) return "≥5×";
  if (rank >= 2) return "≥2×";
  if (rank >= 1) return "≥1×";
  return "";
}

function _khSpecTierBadge(rank) {
  if (!rank) return `<span class="muted">—</span>`;
  const cls = rank >= 4 ? "wmb-tier-10x"
            : rank >= 3 ? "wmb-tier-5x"
            : rank >= 2 ? "wmb-tier-2x"
            : "wmb-tier-1x";
  return `<span class="wmb-tier ${cls}" title="${_khSpecTierLabel(rank)} uniform human-reference specificity">${_khSpecTierLabel(rank)}</span>`;
}

function _khMergeAttributionRows(rawRows) {
  const byCell = new Map();
  for (const raw of rawRows || []) {
    const cell = raw.cell_type || raw.celltype;
    if (!cell) continue;
    const ref = raw.reference || "";
    const spec = raw.specificity != null ? Number(raw.specificity)
               : raw.score != null ? Number(raw.score)
               : null;
    const expr = raw.mean_log2_expression == null ? null : Number(raw.mean_log2_expression);
    const seaLfc = raw.sea_ad_lfc == null ? null : Number(raw.sea_ad_lfc);
    const rank = raw.rank_in_ref == null ? raw.rank : raw.rank_in_ref;
    let row = byCell.get(cell);
    if (!row) {
      row = {
        cell_type: cell,
        references: [],
        source_by_reference: {},
        source_celltypes: [],
        specificity: null,
        score: null,
        mean_log2_expression: null,
        sea_ad_lfc: null,
        rank_in_ref: rank == null ? null : Number(rank),
      };
      byCell.set(cell, row);
    }
    if (ref && !row.references.includes(ref)) row.references.push(ref);
    const src = Array.isArray(raw.source_celltypes) ? raw.source_celltypes : [];
    if (ref && src.length) {
      const prev = row.source_by_reference[ref] || [];
      row.source_by_reference[ref] = Array.from(new Set(prev.concat(src))).sort();
    }
    row.source_celltypes = Array.from(new Set(row.source_celltypes.concat(src))).sort();
    if (rank != null && isFinite(Number(rank))) {
      row.rank_in_ref = row.rank_in_ref == null ? Number(rank) : Math.min(row.rank_in_ref, Number(rank));
    }
    const prevSpec = row.specificity == null || !isFinite(row.specificity) ? -Infinity : row.specificity;
    if (spec != null && isFinite(spec) && spec > prevSpec) {
      row.specificity = spec;
      row.score = spec;
      row.mean_log2_expression = expr;
    } else if (row.mean_log2_expression == null && expr != null && isFinite(expr)) {
      row.mean_log2_expression = expr;
    }
    if (seaLfc != null && isFinite(seaLfc)) {
      const prevLfc = row.sea_ad_lfc;
      if (prevLfc == null || Math.abs(seaLfc) > Math.abs(prevLfc)) row.sea_ad_lfc = seaLfc;
    }
  }
  const rows = Array.from(byCell.values());
  for (const row of rows) {
    row.references.sort((a, b) => String(a).localeCompare(String(b)));
    row.conf = _khAttrConf(row.specificity, row.sea_ad_lfc);
    row.confidence_tier = row.conf;
    row.conf_rank = _khAttrConfRank(row.conf);
    row.tier_rank = _khSpecTierRank(row.specificity);
    row.specificity_tier_rank = row.tier_rank;
  }
  return rows;
}

function _khSourceEvidenceTitle(row) {
  const refs = (row.references || []).map(ref => {
    const label = _KH_REF_LABEL[ref] || ref;
    const src = row.source_by_reference && row.source_by_reference[ref];
    return src && src.length ? `${label}: ${src.join(", ")}` : label;
  });
  return refs.length ? refs.join(" | ") : (row.source_celltypes || []).join(", ");
}

function _khRowsForAttributionSummary(rOrName) {
  if (!_khHasCelltypeSpec()) return [];
  const spec = _KH.celltype_specificity;
  const refs = (spec.references || []).filter(ref => spec[ref] && spec[ref].ranked_by_kinase);
  const out = [];
  for (const ref of refs) {
    const ranked = _khRankedForReference(rOrName, ref);
    for (const ent of ranked) {
      const score = ent.score == null ? null : Number(ent.score);
      const seaLfc = ent.sea_ad_lfc == null ? null : Number(ent.sea_ad_lfc);
      const conf = _khAttrConf(score, seaLfc);
      out.push({
        reference: ref,
        cell_type: ent.celltype,
        source_celltypes: ent.source_celltypes || [],
        score: score,
        sea_ad_lfc: seaLfc,
        conf: conf,
        conf_rank: _khAttrConfRank(conf),
        tier_rank: _khSpecTierRank(score),
      });
    }
  }
  return _khMergeAttributionRows(out);
}

function _khAttributionSummary(r) {
  const rows = _khRowsForAttributionSummary(r);
  if (!rows.length) return {count:0, maxTierRank:0, conf:"none", rows:[]};
  const deduped = rows.slice();
  deduped.sort((a, b) =>
    (b.conf_rank - a.conf_rank) ||
    ((b.tier_rank || 0) - (a.tier_rank || 0)) ||
    ((b.score || -Infinity) - (a.score || -Infinity)) ||
    String(a.cell_type).localeCompare(String(b.cell_type))
  );
  const attributed = deduped.filter(row => row.conf === "high" || row.conf === "moderate");
  const maxTierRank = Math.max(0, ...deduped.map(row => row.tier_rank || 0));
  const bestConfRank = Math.max(0, ...deduped.map(row => row.conf_rank || 0));
  const conf = bestConfRank >= 3 ? "high"
            : bestConfRank >= 2 ? "moderate"
            : bestConfRank >= 1 ? "low"
            : "none";
  return {count: attributed.length, maxTierRank, conf, rows: attributed};
}

function _khRenderCelltypePills(summary) {
  if (!summary || !summary.rows.length) {
    return `<span class="muted" title="No high or moderate human attribution rows.">—</span>`;
  }
  const shown = summary.rows.slice(0, 3).map(row => {
    const cls = row.conf === "high" ? "hi" : "mid";
    const evidence = _khSourceEvidenceTitle(row);
    const tip = `${row.cell_type} · ${row.conf}` +
      (row.score != null && isFinite(row.score) ? ` · specificity ${row.score.toFixed(2)}` : "") +
      (evidence ? ` · evidence: ${evidence}` : "");
    return `<span class="ctx-chip ${cls}" title="${_escapeHtml(tip)}">${_escapeHtml(row.cell_type)}</span>`;
  }).join("");
  const overflow = summary.rows.length > 3
    ? `<span class="ctx-overflow" title="${summary.rows.length - 3} additional high/moderate human attribution rows">+${summary.rows.length - 3}</span>`
    : "";
  return shown + overflow;
}

function _khConfBadge(conf, summary) {
  const cls = conf === "high" ? "hi"
            : conf === "moderate" ? "mid"
            : "lo";
  const label = conf === "high" ? "HIGH"
              : conf === "moderate" ? "MOD"
              : conf === "low" ? "low"
              : "none";
  const count = summary ? summary.count : 0;
  const tip = conf === "none"
    ? "No human attribution rows above low confidence."
    : `${label} human attribution; ${count} high/moderate cell type${count === 1 ? "" : "s"}.`;
  return `<span class="badge ${cls}" title="${_escapeHtml(tip)}">${label}</span>`;
}

function _khAttrCmp(a, b, key, type, asc) {
  let va, vb;
  if (type === "num") {
    va = a[key]; vb = b[key];
    va = (va == null || !isFinite(va)) ? null : Number(va);
    vb = (vb == null || !isFinite(vb)) ? null : Number(vb);
  } else if (type === "conf") {
    va = _khAttrConfRank(a[key]);
    vb = _khAttrConfRank(b[key]);
  } else {
    va = (a[key] || "").toString();
    vb = (b[key] || "").toString();
  }
  if (va == null && vb == null) return 0;
  if (va == null) return 1;
  if (vb == null) return -1;
  if (typeof va === "string") return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  return asc ? (va - vb) : (vb - va);
}

// Derive a confidence tier from the human evidence available for one row.
// Mirrors the mouse confidence rubric (high / moderate / low / none) but
// without Song (no human equivalent). Gates:
//   high     — specificity ≥ 2× uniform (log2 ≥ 1.0) AND |SEA-AD LFC| ≥ 0.1
//   moderate — specificity ≥ 2× uniform OR  |SEA-AD LFC| ≥ 0.1 (when known)
//   low      — specificity ≥ 0 (cell-type ≥ brain mean) but neither gate cleared
//   none     — specificity < 0 (cell-type below brain mean)
// HBCA rows have no SEA-AD LFC — they cap at moderate / low / none.
function _khAttrConf(score, seaLfc) {
  if (score == null || !isFinite(score)) return "none";
  const specOk = score >= 1.0;          // log2(2) = 2× uniform
  const lfcOk  = seaLfc != null && isFinite(seaLfc) && Math.abs(seaLfc) >= 0.1;
  if (specOk && lfcOk) return "high";
  if (specOk || lfcOk) return "moderate";
  if (score >= 0)      return "low";
  return "none";
}

// Render the human Attribution sub-tab.
function _khRenderAttribution(body, r) {
  if (!_khHasCelltypeSpec()) {
    body.innerHTML = `<div class="muted" style="padding:1em;">
      Attribution data is not available in this payload build.
      Run <code>python alz/atlas_reference.py --sea-ad-expression</code> and
      <code>python alz/atlas_reference.py --hbca-download</code>, then
      <code>python alz/human_reference_expression.py --ref both --force</code>
      and rebuild the viewer.
    </div>`;
    return;
  }

  const spec = _KH.celltype_specificity;
  const refs = (spec.references || []).filter(ref => spec[ref] && spec[ref].ranked_by_kinase);
  if (!refs.length) {
    body.innerHTML = `<div class="muted" style="padding:1em;">No reference cell-type rankings for ${_escapeHtml(r.name)}.</div>`;
    return;
  }

  // Collect ranked Levy T5 evidence from every available human reference, then
  // collapse to one displayed row per Levy T5 cell type.
  const rawRows = [];
  for (const ref of refs) {
    const ranked = _khRankedForReference(r, ref);
    for (const ent of ranked) {
      rawRows.push({
        reference: ref,
        cell_type: ent.celltype,
        source_celltypes: ent.source_celltypes || [],
        rank_in_ref: ent.rank,
        specificity: ent.score,
        mean_log2_expression: ent.mean_log2_expression == null ? null : ent.mean_log2_expression,
        sea_ad_lfc: ent.sea_ad_lfc == null ? null : ent.sea_ad_lfc,
      });
    }
  }
  const rows = _khMergeAttributionRows(rawRows);

  // Per-donor kinase NES — broadcast across every row of this kinase. Changes
  // when the donor selector fires.
  const donor = _KHState.auditDonor;
  const pd = (donor != null) ? _khPerdonorFor(r.id, donor) : null;
  const donorNES = pd ? pd.NES : null;
  const donorFDR = pd ? pd.FDR : null;

  for (const row of rows) {
    row.confidence_tier  = _khAttrConf(row.specificity, row.sea_ad_lfc);
    row.specificity_tier_rank = _khSpecTierRank(row.specificity);
    row.donor_nes = donorNES;
    row.donor_fdr = donorFDR;
  }

  const sortKey = body.dataset.khAttrSortKey || "confidence_tier";
  const sortAsc = body.dataset.khAttrSortAsc === "1";
  const sortCol = KH_ATTR_COLS.find(c => c.key === sortKey)
    || KH_ATTR_COLS.find(c => c.key === "confidence_tier")
    || KH_ATTR_COLS[KH_ATTR_COLS.length - 1];
  rows.sort((a, b) => _khAttrCmp(a, b, sortCol.key, sortCol.type, sortAsc));
  if (sortCol.key === "confidence_tier") {
    rows.sort((a, b) => {
      const primary = _khAttrCmp(a, b, sortCol.key, sortCol.type, sortAsc);
      if (primary !== 0) return primary;
      return ((b.specificity_tier_rank || 0) - (a.specificity_tier_rank || 0)) ||
             ((b.specificity || -Infinity) - (a.specificity || -Infinity));
    });
  }

  // Donor selector: surface so the user can change which donor's NES is shown.
  // Defaults to the first case donor when none is picked.
  const donors = _KH.donors || [];
  const donorOpts = donors.map(d =>
    `<option value="${_escapeHtml(d)}"${d === donor ? " selected" : ""}>${_escapeHtml(d)}</option>`
  ).join("");
  const donorSelector =
    `<div class="attr-bulk-anchor">Donor for NES column: ` +
      `<select id="kh-attr-donor-select" class="attr-bulk-pill">` +
        `<option value=""${donor == null ? " selected" : ""}>—</option>${donorOpts}` +
      `</select> ` +
      `<span class="attr-bulk-pill">` +
        (donor != null && donorNES != null && isFinite(donorNES)
          ? (donorNES > 0
              ? `<span class="attr-bulk-up">↑ NES = +${donorNES.toFixed(2)}</span>`
              : `<span class="attr-bulk-down">↓ NES = ${donorNES.toFixed(2)}</span>`)
          : `<span class="attr-bulk-ns">NES n/a</span>`) +
        " · " +
        (donor != null && donorFDR != null && isFinite(donorFDR)
          ? `FDR = ${donorFDR.toFixed(3)}${donorFDR < 0.25 ? "" : " (n.s.)"}`
          : "FDR n/a") +
      `</span> ` +
      `<span class="muted">— per-donor kinase NES from the human MEA; broadcast to every row.</span>` +
    `</div>`;

  const num = (v, d=3) => (v == null || !isFinite(v)) ? "" : Number(v).toFixed(d);
  const tbody = rows.map((row, i) => {
    const tierChip =
      `<span class="${_attrConfidenceClass(row.confidence_tier)}">${_escapeHtml(row.confidence_tier.replace('_', ' '))}</span>`;
    // Location tier: uses the same ≥10× / ≥5× / ≥2× / ≥1× of uniform logic
    // as the mouse WMB tier, applied to the human log2-ratio score directly.
    const specTier = row.specificity == null || !isFinite(row.specificity)
      ? "" : _wmbSpecToHumanTier(row.specificity);
    const seaCell = (row.sea_ad_lfc == null || !isFinite(row.sea_ad_lfc))
      ? `<td class="attr-num attr-empty">—</td>`
      : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(row.sea_ad_lfc)}">${num(row.sea_ad_lfc, 3)}</td>`;
    const nesCell = (donorNES == null || !isFinite(donorNES))
      ? `<td class="attr-num attr-empty">—</td>`
      : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(donorNES)}">${num(donorNES, 2)}</td>`;
    const fdrCell = (donorFDR == null || !isFinite(donorFDR))
      ? `<td class="attr-num attr-empty">—</td>`
      : `<td class="attr-num"${donorFDR < 0.25 ? ' style="font-weight:600"' : ''}>${num(donorFDR, 3)}</td>`;
    const evidenceTitle = _khSourceEvidenceTitle(row);
    return `<tr class="attr-verdict-row${i === 0 ? ' attr-verdict-selected' : ''}">` +
      `<td class="attr-celltype" title="${_escapeHtml(evidenceTitle)}">${_escapeHtml(row.cell_type)}</td>` +
      `<td>${tierChip}</td>` +
      `<td class="attr-num">${num(row.specificity, 3)}</td>` +
      `<td class="attr-num">${specTier}</td>` +
      `<td class="attr-num">${num(row.mean_log2_expression, 2)}</td>` +
      seaCell +
      nesCell +
      fdrCell +
      `</tr>`;
  }).join("");

  const headCells = KH_ATTR_COLS.map(c => {
    const arrow = (c.key === sortCol.key) ? (sortAsc ? " ▲" : " ▼") : "";
    const title = c.title ? ` title="${_escapeHtml(c.title)}"` : "";
    return `<th class="attr-verdict-th" data-sort-key="${c.key}"${title}>${c.label}${arrow}</th>`;
  }).join("");
  const head = `<tr>${headCells}</tr>`;
  const superHead =
    `<tr class="attr-verdict-supergroup">` +
      `<th class="attr-supergroup-spacer" colspan="1"></th>` +
      `<th class="attr-supergroup-attr" colspan="6" title="Cell-type attribution evidence (human references — transcript-level location evidence + SEA-AD AD effect).">Attribution</th>` +
      `<th class="attr-supergroup-decomp" colspan="2" title="Per-donor kinase activity from the human MEA. Broadcast per kinase; changes with the donor selector.">Donor kinase activity</th>` +
    `</tr>`;

  body.innerHTML =
    `<p class="kinase-stage-note">` +
    `Cell-type attribution of <strong>${_escapeHtml(r.gene_symbol || r.name)}</strong> ` +
    `across human reference atlases, consolidated to Levy T5 clusters before display. ` +
    `Location evidence is a transcript-level prior; ` +
    `SEA-AD LFC is the AD-vs-control effect rolled up to the same cluster; donor NES ` +
    `is the kinase's per-donor MEA activity broadcast to every row. ` +
    `HBCA rows lack SEA-AD LFC.` +
    `</p>` +
    donorSelector +
    `<table class="attr-verdict-table"><thead>${superHead}${head}</thead><tbody>${tbody}</tbody></table>`;

  // Scope to this panel's own body — the crosstable detail reuses this renderer,
  // so a document-wide getElementById would grab the human tab's select instead.
  const sel = body.querySelector("#kh-attr-donor-select");
  if (sel) sel.addEventListener("change", () => {
    _KHState.auditDonor = sel.value || null;
    _khRenderAttribution(body, r);
  });
  body.querySelectorAll("th.attr-verdict-th").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.sortKey;
    if (body.dataset.khAttrSortKey === k) {
      body.dataset.khAttrSortAsc = body.dataset.khAttrSortAsc === "1" ? "0" : "1";
    } else {
      body.dataset.khAttrSortKey = k;
      const col = KH_ATTR_COLS.find(c => c.key === k);
      body.dataset.khAttrSortAsc = (col && col.type === "str") ? "1" : "0";
    }
    _khRenderAttribution(body, r);
  }));
}

// Specificity → tier badge string, mirroring _wmbTierBadge styling. The
// human references have different cell-type counts than WMB-34, so the
// "multiple of uniform" thresholds for high/moderate/low are interpreted
// directly off the log2-ratio score: ≥1.0 → ≥2× uniform → "high" tier.
function _wmbSpecToHumanTier(score) {
  const rank = _khSpecTierRank(score);
  if (rank) return _khSpecTierBadge(rank);
  return "";
}

function _khBuildDonorChips() {
  const host = document.getElementById("kh-ms-donors");
  if (!host) return;
  const donors = _KH.donors;
  host.innerHTML = `<label class="ke-filter-label" style="margin-right:4px;">Donors</label>`
    + donors.map(d =>
        `<button type="button" class="chip kh-donor-chip" data-donor="${_escapeHtml(d)}">${_escapeHtml(d)}</button>`
      ).join("");
  host.querySelectorAll(".kh-donor-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      const d = btn.dataset.donor;
      if (_KHState.donors.has(d)) _KHState.donors.delete(d);
      else _KHState.donors.add(d);
      btn.classList.toggle("active", _KHState.donors.has(d));
      renderKinaseHuman();
    });
  });
}

function _khPopulateCelltypeFilter() {
  const sel = document.getElementById("kh-filter-celltype");
  if (!sel || !_khHasCelltypeSpec()) return;
  const seen = new Set();
  const spec = _KH.celltype_specificity;
  for (const ref of (spec.references || [])) {
    const rankedByKinase = spec[ref] && spec[ref].ranked_by_kinase;
    if (!rankedByKinase) continue;
    for (const rows of Object.values(rankedByKinase)) {
      for (const ent of rows || []) {
        const score = ent.score == null ? null : Number(ent.score);
        const seaLfc = ent.sea_ad_lfc == null ? null : Number(ent.sea_ad_lfc);
        const conf = _khAttrConf(score, seaLfc);
        if (conf === "high" || conf === "moderate") seen.add(ent.celltype);
      }
    }
  }
  const opts = Array.from(seen).sort((a, b) => String(a).localeCompare(String(b)))
    .map(ct => `<option value="${_escapeHtml(ct)}">${_escapeHtml(ct)}</option>`)
    .join("");
  sel.innerHTML = `<option value="">Any</option>${opts}`;
  sel.value = _KHState.celltype || "";
}

function wireKinaseHuman() {
  if (!_KH_HAS) return;
  _khBuildDonorChips();
  _khPopulateCelltypeFilter();
  const search = document.getElementById("kh-search");
  if (search) search.addEventListener("input", e => {
    _KHState.search = e.target.value; renderKinaseHuman();
  });
  const nsig = document.getElementById("kh-filter-nsig");
  if (nsig) nsig.addEventListener("change", e => {
    _KHState.nsigMin = parseInt(e.target.value, 10) || 0; renderKinaseHuman();
  });
  const track = document.getElementById("kh-filter-track");
  if (track) track.addEventListener("change", e => {
    _KHState.track = e.target.value; renderKinaseHuman();
  });
  const celltype = document.getElementById("kh-filter-celltype");
  if (celltype) celltype.addEventListener("change", e => {
    _KHState.celltype = e.target.value || ""; renderKinaseHuman();
  });
  const confidence = document.getElementById("kh-filter-confidence");
  if (confidence) confidence.addEventListener("change", e => {
    _KHState.confidence = e.target.value || ""; renderKinaseHuman();
  });
  const specificity = document.getElementById("kh-filter-specificity");
  if (specificity) specificity.addEventListener("change", e => {
    _KHState.specificityTier = Math.max(0, parseInt(e.target.value, 10) || 0);
    renderKinaseHuman();
  });
  const seaad = document.getElementById("kh-filter-seaad");
  if (seaad) seaad.addEventListener("change", e => {
    _KHState.seaad = e.target.value; renderKinaseHuman();
  });
  const adDir = document.getElementById("kh-filter-ad-dir");
  if (adDir) adDir.addEventListener("change", e => {
    _KHState.adDir = e.target.value; renderKinaseHuman();
  });
  const dirMode = document.getElementById("kh-filter-dir-mode");
  if (dirMode) dirMode.addEventListener("change", e => {
    _KHState.dirMode = e.target.value; renderKinaseHuman();
  });
  const ctrlDir = document.getElementById("kh-filter-ctrl-dir");
  if (ctrlDir) {
    const ctrlCount = (_KH.ctrl_donors || []).length;
    if (!ctrlCount) {
      const wrap = ctrlDir.closest("label");
      if (wrap) wrap.style.display = "none";
    } else {
      ctrlDir.addEventListener("change", e => {
        _KHState.ctrlDir = e.target.value; renderKinaseHuman();
      });
    }
  }
  const showCtrl = document.getElementById("kh-show-ctrl");
  // Hide the toggle entirely when there are no CTRL donors in the payload
  // (legacy build before CTRL columns were added).
  if (showCtrl) {
    const ctrlCount = (_KH.ctrl_donors || []).length;
    if (!ctrlCount) {
      const lab = document.getElementById("kh-show-ctrl-label");
      if (lab) lab.style.display = "none";
      _KHState.showCtrl = false;
    } else {
      showCtrl.checked = _KHState.showCtrl;
      showCtrl.addEventListener("change", e => {
        _KHState.showCtrl = !!e.target.checked;
        renderKinaseHuman();
      });
    }
  }
  const reset = document.getElementById("kh-filter-reset");
  if (reset) reset.addEventListener("click", () => {
    _KHState.search = ""; _KHState.donors.clear();
    _KHState.nsigMin = 0; _KHState.track = ""; _KHState.celltype = "";
    _KHState.confidence = ""; _KHState.specificityTier = 0; _KHState.seaad = "";
    _KHState.adDir = ""; _KHState.ctrlDir = ""; _KHState.dirMode = "sig";
    if (search) search.value = "";
    if (nsig) nsig.value = 0;
    if (track) track.value = "";
    if (celltype) celltype.value = "";
    if (confidence) confidence.value = "";
    if (specificity) specificity.value = "0";
    if (seaad) seaad.value = "";
    if (adDir) adDir.value = "";
    if (ctrlDir) ctrlDir.value = "";
    if (dirMode) dirMode.value = "sig";
    document.querySelectorAll(".kh-donor-chip").forEach(b => b.classList.remove("active"));
    renderKinaseHuman();
  });
  // Column-header sort.
  document.querySelectorAll("#kh-table thead th[data-col]").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (_KHState.sortCol === col) _KHState.sortAsc = !_KHState.sortAsc;
      else { _KHState.sortCol = col; _KHState.sortAsc = false; }
      renderKinaseHuman();
    });
  });
  // Row-click selection.
  const tbody = document.querySelector("#kh-table tbody");
  if (tbody) tbody.addEventListener("click", ev => {
    const tr = ev.target.closest("tr[data-khid]");
    if (!tr) return;
    const khid = parseInt(tr.dataset.khid, 10);
    Store.dispatch({type:"SET_SELECTION", key:"kinaseHuman",
      value: Store.state.selection.kinaseHuman === khid ? null : khid});
  });
  // Splitter wiring is centralized in 02_ui_chrome.js (_wireSplitter).
}
