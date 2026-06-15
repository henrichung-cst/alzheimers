// 5xFAD mouse kinase explorer.
//
// Keeps 5xFAD as a first-class mode while using the same master/detail kinase
// workbench grammar as the Song and Mukesh kinase tabs.

const _F5_AGES = [3, 6, 9, 12];
const _F5_AUDIT_TABS = [
  {id: "measurement-trace", label: "Measurement Trace"},
  {id: "mea-input", label: "MEA Preparation"},
  {id: "mea-score", label: "MEA Score"},
  {id: "attribution", label: "Attribution"},
];

let _F5Groups = null;
let _F5QcByKey = null;
let _F5RawByKey = null;
let _F5AttrByKinase = null;
let _F5Wired = false;
const _F5DetailCache = new Map();

const _F5State = {
  search: "",
  tissue: "",
  age: "",
  celltype: "",
  confidence: "",
  songMin: 0,
  wmbMin: 0,
  nsigMin: 0,
  pattern: "",
  sortCol: "peakAbsNes",
  sortAsc: false,
  auditTab: "mea-score",
  auditAge: null,
};

function _f5Block() {
  return (PAYLOAD && PAYLOAD.supporting_5xfad) || null;
}

function _f5Esc(s) {
  return (typeof _escapeHtml === "function") ? _escapeHtml(s == null ? "" : s) : String(s == null ? "" : s);
}

function _f5Num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function _f5Fmt(v, digits) {
  const n = _f5Num(v);
  return n == null ? "—" : n.toFixed(digits == null ? 3 : digits);
}

function _f5TrackLabel(v) {
  if (v === "stoichiometry") return "stoichiometry";
  if (v === "raw_phospho") return "raw phospho";
  return String(v || "");
}

function _f5SurfaceLabel(group) {
  return `${group.tissue} · ${group.residue_type || group.assay}`;
}

function _f5RowKey(row) {
  return [
    row.kinase || "",
    row.tissue || "",
    row.assay || "",
    "stoichiometry",
  ].join("|");
}

function _f5QcKey(tissue, track, age) {
  return [tissue || "", track || "", String(age || "")].join("|");
}

function _f5SelectedKey() {
  return Store.state.selection.kinaseFiveXFAD || null;
}

function _f5SetSelectedKey(key) {
  Store.dispatch({type: "SET_SELECTION", key: "kinaseFiveXFAD", value: key || null});
}

function _f5DetailShard(group) {
  const shards = (_f5Block() && _f5Block().detail_shards) || {};
  return shards[group.key] || "";
}

function _f5Family(name) {
  const famMap = (typeof META !== "undefined" && META && META.familyMap) || {};
  return famMap[name] || "";
}

function _f5ConfRank(conf) {
  return (_CONF_RANK && _CONF_RANK[conf]) || 0;
}

function _f5ConfPass(conf, threshold) {
  if (!threshold) return true;
  return _f5ConfRank(conf) >= _f5ConfRank(threshold);
}

function _f5AttrRows(kinase) {
  _f5EnsureIndexes();
  return (_F5AttrByKinase && _F5AttrByKinase.get(String(kinase))) || [];
}

function _f5CmpAttr(a, b) {
  const cr = _f5ConfRank(b.confidence_tier) - _f5ConfRank(a.confidence_tier);
  if (cr) return cr;
  const sr = (_f5Num(b.song_specificity) || -1) - (_f5Num(a.song_specificity) || -1);
  if (sr) return sr;
  return (_f5Num(b.wmb_specificity) || -1) - (_f5Num(a.wmb_specificity) || -1);
}

const F5_ATTR_COLS = [
  {key: "cell_type", label: "Cell type", type: "str", group: "id",
   title: "Levy T5 cluster from the canonical mouse kinase attribution evidence table."},
  {key: "confidence_tier", label: "Conf", type: "conf", group: "attr",
   title: "Canonical mouse confidence tier. Hover the chip for the evidence basis."},
  {key: "song_specificity", label: "Song", type: "num", group: "attr",
   title: "Song mouse location evidence, shown as fold over the even-split baseline (1/31)."},
  {key: "wmb_tier", label: "WMB tier", type: "num", group: "attr",
   title: "WMB cross-check tier as a multiple of uniform expression across WMB classes."},
  {key: "sea_ad_lfc", label: "SEA-AD LFC", type: "num", group: "attr",
   title: "Human SEA-AD AD-vs-control log2 fold change mapped to this cell type where available."},
  {key: "fivexfad_nes", label: "5xFAD NES", type: "num", group: "activity",
   title: "Bulk 5xFAD kinase MEA NES for the selected tissue and age, broadcast to each attribution row."},
  {key: "fivexfad_fdr", label: "5xFAD FDR", type: "num", group: "activity",
   title: "Bulk 5xFAD kinase MEA FDR for the selected tissue and age, broadcast to each attribution row."},
];

function _f5AttrCmp(a, b, key, type, asc) {
  let va, vb;
  if (type === "num") {
    va = a[key]; vb = b[key];
    va = (va == null || !isFinite(va)) ? null : Number(va);
    vb = (vb == null || !isFinite(vb)) ? null : Number(vb);
  } else if (type === "conf") {
    va = _f5ConfRank(a[key]);
    vb = _f5ConfRank(b[key]);
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

function _f5SortAttrRows(rows, body) {
  const sortKey = body.dataset.f5AttrSortKey || "confidence_tier";
  const sortAsc = body.dataset.f5AttrSortAsc === "1";
  const sortCol = F5_ATTR_COLS.find(c => c.key === sortKey)
    || F5_ATTR_COLS.find(c => c.key === "confidence_tier")
    || F5_ATTR_COLS[0];
  rows.sort((a, b) => _f5AttrCmp(a, b, sortCol.key, sortCol.type, sortAsc));
  if (sortCol.key === "confidence_tier") {
    rows.sort((a, b) => {
      const primary = _f5AttrCmp(a, b, sortCol.key, sortCol.type, sortAsc);
      if (primary !== 0) return primary;
      return (_f5SongTier(b) - _f5SongTier(a)) ||
             (_f5WmbTier(b) - _f5WmbTier(a)) ||
             ((_f5Num(b.song_specificity) || -Infinity) - (_f5Num(a.song_specificity) || -Infinity));
    });
  }
  return {sortCol, sortAsc};
}

function _f5ScopedAttrRows(group) {
  const rows = _f5AttrRows(group.kinase).filter(r => {
    if (_F5State.celltype && r.cell_type !== _F5State.celltype) return false;
    if (!_f5ConfPass(r.confidence_tier || "none", _F5State.confidence)) return false;
    if (_F5State.songMin && _f5SongTier(r) < _F5State.songMin) return false;
    if (_F5State.wmbMin && _f5WmbTier(r) < _F5State.wmbMin) return false;
    return true;
  });
  return rows.sort(_f5CmpAttr);
}

function _f5SongTier(row) {
  const share = _f5Num(row && (row.song_specificity != null ? row.song_specificity : row.song_top_share));
  return typeof _msTier === "function" ? _msTier(share) : 0;
}

function _f5WmbTier(row) {
  const spec = _f5Num(row && row.wmb_specificity);
  return typeof _wmbTier === "function" ? _wmbTier(spec) : 0;
}

function _f5BestAttr(group) {
  const rows = _f5ScopedAttrRows(group);
  return rows.length ? rows[0] : null;
}

function _f5CellTypeCount(group) {
  const rows = _f5ScopedAttrRows(group).filter(r =>
    ["very_high", "high", "moderate"].includes(r.confidence_tier || ""));
  return new Set(rows.map(r => r.cell_type)).size;
}

function _f5CellTypesCell(group) {
  const rows = _f5ScopedAttrRows(group).filter(r =>
    ["very_high", "high", "moderate"].includes(r.confidence_tier || ""));
  const byCell = new Map();
  for (const r of rows) {
    const prev = byCell.get(r.cell_type);
    if (!prev || _f5CmpAttr(r, prev) < 0) byCell.set(r.cell_type, r);
  }
  const displayRows = Array.from(byCell.values()).sort(_f5CmpAttr);
  if (!displayRows.length) return '<span class="muted">—</span>';
  const tip = displayRows.map(r => {
    const song = _f5Num(r.song_specificity);
    const songTxt = song == null ? "Song n/a" : `Song ${(song / _MS_UNIFORM).toFixed(1)}x`;
    return `${r.cell_type} (${songTxt}, ${String(r.confidence_tier || "").replace("_", " ")})`;
  }).join("\n");
  const pills = displayRows.slice(0, 3).map(r => {
    const cls = r.confidence_tier === "very_high" ? "vhi"
      : r.confidence_tier === "high" ? "hi"
      : "mid";
    return `<span class="badge ${cls}">${_f5Esc(r.cell_type)}</span>`;
  }).join(" ");
  const extra = displayRows.length > 3 ? ` <span class="muted">+${displayRows.length - 3}</span>` : "";
  return `<span title="${_f5Esc(tip)}"><strong>${displayRows.length}</strong> ${pills}${extra}</span>`;
}

function _f5SongBadge(group) {
  const row = _f5BestAttr(group);
  if (!row || typeof _msTierBadge !== "function") return '<span class="muted">—</span>';
  const share = _f5Num(row.song_specificity != null ? row.song_specificity : row.song_top_share);
  return _msTierBadge(_f5SongTier(row), share, row.song_top_cluster || "", _f5Num(row.song_tau));
}

function _f5WmbBadge(group) {
  let best = 0;
  for (const row of _f5ScopedAttrRows(group)) best = Math.max(best, _f5WmbTier(row));
  return typeof _wmbTierBadge === "function" ? _wmbTierBadge(best) : '<span class="muted">—</span>';
}

function _f5ConfBadge(group) {
  const row = _f5BestAttr(group);
  const conf = row ? (row.confidence_tier || "none") : "none";
  if (!row || conf === "none") return '<span class="badge lo" title="No high or moderate attribution in scope.">low</span>';
  const cls = conf === "very_high" ? "vhi" : (conf === "high" ? "hi" : (conf === "moderate" ? "mid" : "lo"));
  return `<span class="badge ${cls}" title="${_f5Esc(row.confidence_basis || conf)}">${_f5Esc(conf.replace("_", " "))}</span>`;
}

function _f5TrendMatches(group, pattern) {
  const pat = (window.TrendFilter && TrendFilter.normalize) ? TrendFilter.normalize(pattern || "") : (pattern || "");
  if (!pat) return true;
  const vals = _F5_AGES.map(age => {
    const row = group.rows.get(age);
    const v = row ? _f5Num(row.NES) : null;
    return v == null ? null : v;
  });
  if (window.TrendFilter && TrendFilter.vectorMatches) return TrendFilter.vectorMatches(vals, pat);
  return true;
}

function _f5EnsureIndexes() {
  if (_F5Groups && _F5QcByKey && _F5RawByKey && _F5AttrByKinase) return;
  const block = _f5Block();
  _F5Groups = [];
  _F5QcByKey = new Map();
  _F5RawByKey = new Map();
  _F5AttrByKinase = new Map();
  if (!block) return;

  for (const q of (block.contrast_qc || [])) {
    _F5QcByKey.set(_f5QcKey(q.tissue, q.track, q.age_months), q);
  }

  for (const r of (block.attribution_rows || [])) {
    const k = String(r.kinase || "");
    if (!k) continue;
    if (!_F5AttrByKinase.has(k)) _F5AttrByKinase.set(k, []);
    _F5AttrByKinase.get(k).push(r);
  }
  for (const rows of _F5AttrByKinase.values()) rows.sort(_f5CmpAttr);

  const byKey = new Map();
  for (const row of (block.rows || [])) {
    if (row.analysis_track === "raw_phospho") {
      const rawKey = _f5RowKey(row);
      if (!_F5RawByKey.has(rawKey)) _F5RawByKey.set(rawKey, new Map());
      _F5RawByKey.get(rawKey).set(Number(row.age_months), row);
      continue;
    }
    if (row.analysis_track && row.analysis_track !== "stoichiometry") continue;
    const key = _f5RowKey(row);
    let rec = byKey.get(key);
    if (!rec) {
      rec = {
        key,
        kinase: row.kinase || "",
        gene_symbol: row.gene_symbol || row.kinase || "",
        family: _f5Family(row.kinase || ""),
        tissue: row.tissue || "",
        track: row.track || "",
        assay: row.assay || "",
        residue_type: row.residue_type || "",
        analysis_track: row.analysis_track || "",
        rows: new Map(),
      };
      byKey.set(key, rec);
    }
    rec.rows.set(Number(row.age_months), row);
  }
  _F5Groups = Array.from(byKey.values());
}

function _f5AgeScope() {
  return _F5State.age ? [Number(_F5State.age)] : _F5_AGES.slice();
}

function _f5Metric(group, ages) {
  let peakAbsNes = null;
  let sigCount = 0;
  const fdr = Store.state.filters.fdr;
  for (const age of ages) {
    const row = group.rows.get(Number(age));
    if (!row) continue;
    const nes = _f5Num(row.NES);
    if (nes != null) {
      const abs = Math.abs(nes);
      if (peakAbsNes == null || abs > peakAbsNes) peakAbsNes = abs;
    }
    const q = _f5Num(row.FDR);
    if (q != null && q < fdr) sigCount += 1;
  }
  return {peakAbsNes, sigCount};
}

function _f5GroupPasses(group, ages, metric) {
  const q = _F5State.search.trim().toLowerCase();
  if (q && !(String(group.kinase).toLowerCase().includes(q)
          || String(group.gene_symbol).toLowerCase().includes(q))) return false;
  if (_F5State.tissue && group.tissue !== _F5State.tissue) return false;
  if (metric.sigCount < _F5State.nsigMin) return false;
  if (_F5State.pattern && !_f5TrendMatches(group, _F5State.pattern)) return false;
  if (_F5State.celltype || _F5State.confidence || _F5State.songMin || _F5State.wmbMin) {
    if (_f5ScopedAttrRows(group).length === 0) return false;
  }
  if (_F5State.age) {
    const age = Number(_F5State.age);
    if (!group.rows.has(age)) return false;
  }
  return true;
}

function _f5FilteredRows() {
  _f5EnsureIndexes();
  const ages = _f5AgeScope();
  const out = [];
  for (const group of _F5Groups) {
    const metric = _f5Metric(group, ages);
    if (_f5GroupPasses(group, ages, metric)) {
      out.push({...group, ...metric});
    }
  }
  const col = _F5State.sortCol;
  const asc = _F5State.sortAsc;
  out.sort((a, b) => {
    let va = col === "profile" ? a.peakAbsNes : a[col];
    let vb = col === "profile" ? b.peakAbsNes : b[col];
    if (col === "n_attributed_celltypes") {
      va = _f5CellTypeCount(a);
      vb = _f5CellTypeCount(b);
    } else if (col === "song_spec") {
      va = _f5BestAttr(a) ? _f5Num(_f5BestAttr(a).song_specificity) : null;
      vb = _f5BestAttr(b) ? _f5Num(_f5BestAttr(b).song_specificity) : null;
    } else if (col === "wmb_max_tier") {
      va = Math.max(0, ..._f5ScopedAttrRows(a).map(_f5WmbTier));
      vb = Math.max(0, ..._f5ScopedAttrRows(b).map(_f5WmbTier));
    } else if (col === "conf") {
      va = _f5BestAttr(a) ? _f5ConfRank(_f5BestAttr(a).confidence_tier) : 0;
      vb = _f5BestAttr(b) ? _f5ConfRank(_f5BestAttr(b).confidence_tier) : 0;
    }
    if (va == null && vb == null) return String(a.kinase).localeCompare(String(b.kinase));
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === "string") return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    return asc ? (va - vb) : (vb - va);
  });
  return out;
}

function _f5MaxAbsVisible(rows) {
  let maxAbs = 0;
  for (const group of rows) {
    for (const age of _F5_AGES) {
      const row = group.rows.get(age);
      const nes = row ? _f5Num(row.NES) : null;
      if (nes != null) maxAbs = Math.max(maxAbs, Math.abs(nes));
    }
  }
  return maxAbs > 0 ? maxAbs : 1;
}

function _f5ProfileCell(group, age, maxAbs) {
  const row = group.rows.get(Number(age));
  if (!row) {
    const title = `${age}mo: no MEA row`;
    return `<div class="npc" title="${_f5Esc(title)}"></div>`;
  }
  const nes = _f5Num(row.NES);
  const q = _f5Num(row.FDR);
  const sig = q != null && q < Store.state.filters.fdr;
  let bg = "#fff";
  if (nes != null && maxAbs > 0) {
    const a = Math.min(1, Math.abs(nes) / maxAbs);
    const rgb = nes >= 0 ? [197, 48, 48] : [43, 108, 176];
    bg = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.15 + 0.85 * a).toFixed(3)})`;
  }
  const title = `${age}mo: NES ${_f5Fmt(nes, 2)}, FDR ${q == null ? "—" : q.toExponential(1)}${sig ? " (sig)" : ""}`;
  return `<div class="npc${sig ? " sig" : ""}" style="background:${bg};" title="${_f5Esc(title)}"></div>`;
}

function _f5Profile(group, maxAbs) {
  const cells = _F5_AGES.map(age => _f5ProfileCell(group, age, maxAbs)).join("");
  const labels = _F5_AGES.map(age => `<span>${age}</span>`).join("");
  return `<div class="nes-profile-wrap" title="Age cells: 3, 6, 9, 12 months">` +
    `<div class="nes-profile-age-stack">` +
    `<div class="nes-profile-age-labels">${labels}</div>` +
    `<div class="nes-profile-cell" style="grid-template-columns:repeat(${_F5_AGES.length},1fr);">${cells}</div>` +
    `</div>` +
    `</div>`;
}

function _f5SetOptions(id, values, labels, allowAny) {
  const el = document.getElementById(id);
  if (!el) return;
  const cur = el.value || "";
  const opts = allowAny === false ? [] : ['<option value="">Any</option>'];
  values.forEach(v => {
    const lab = labels && labels[v] ? labels[v] : v;
    opts.push(`<option value="${_f5Esc(v)}">${_f5Esc(lab)}</option>`);
  });
  el.innerHTML = opts.join("");
  el.value = values.includes(cur) ? cur : (allowAny === false ? (values[0] || "") : "");
}

function _f5PopulateControls() {
  const block = _f5Block();
  if (!block) return;
  const filters = block.filters || {};
  _f5SetOptions("f5-filter-tissue", filters.tissue || []);
  const ageVals = (filters.age_months || _F5_AGES).map(v => String(v));
  _f5SetOptions("f5-filter-age", ageVals, {3: "3mo", 6: "6mo", 9: "9mo", 12: "12mo"});
  const celltypes = Array.from(new Set((block.attribution_rows || [])
    .map(r => r.cell_type).filter(Boolean))).sort();
  _f5SetOptions("f5-filter-celltype", celltypes);
}

function _f5SyncControls() {
  const byId = {
    "f5-search": "search",
    "f5-filter-tissue": "tissue",
    "f5-filter-age": "age",
    "f5-filter-celltype": "celltype",
    "f5-filter-confidence": "confidence",
    "f5-filter-song": "songMin",
    "f5-filter-wmb": "wmbMin",
    "f5-filter-nsig": "nsigMin",
    "f5-filter-pattern": "pattern",
  };
  Object.entries(byId).forEach(([id, key]) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = ["nsigMin", "songMin", "wmbMin"].includes(key)
      ? String(_F5State[key] || 0)
      : (_F5State[key] || "");
  });
}

function wireFiveXFADKinase() {
  if (_F5Wired) return;
  _F5Wired = true;
  const bind = (id, key, numeric) => {
    const el = document.getElementById(id);
    if (!el) return;
    const apply = () => {
      _F5State[key] = numeric ? Math.max(0, parseInt(el.value || "0", 10)) : el.value;
      renderFiveXFADKinase();
    };
    el.addEventListener("input", apply);
    el.addEventListener("change", apply);
  };
  bind("f5-search", "search");
  bind("f5-filter-tissue", "tissue");
  bind("f5-filter-age", "age");
  bind("f5-filter-celltype", "celltype");
  bind("f5-filter-confidence", "confidence");
  bind("f5-filter-song", "songMin", true);
  bind("f5-filter-wmb", "wmbMin", true);
  bind("f5-filter-nsig", "nsigMin", true);
  bind("f5-filter-pattern", "pattern");
  const reset = document.getElementById("f5-filter-reset");
  if (reset) reset.addEventListener("click", () => {
    Object.assign(_F5State, {
      search: "", tissue: "", age: "", celltype: "", confidence: "",
      songMin: 0, wmbMin: 0, nsigMin: 0, pattern: "",
      sortCol: "peakAbsNes", sortAsc: false,
      auditTab: "mea-score", auditAge: null,
    });
    _f5SetSelectedKey(null);
    renderFiveXFADKinase();
  });
  const table = document.getElementById("f5-table");
  if (table) {
    table.querySelectorAll("thead th[data-col]").forEach(th => {
      th.addEventListener("click", () => {
        const col = th.dataset.col;
        if (_F5State.sortCol === col) _F5State.sortAsc = !_F5State.sortAsc;
        else { _F5State.sortCol = col; _F5State.sortAsc = false; }
        renderFiveXFADKinase();
      });
    });
    const tbody = table.querySelector("tbody");
    tbody.addEventListener("click", ev => {
      const tr = ev.target.closest("tr[data-f5-key]");
      if (!tr) return;
      _F5State.auditAge = null;
      _f5SetSelectedKey(_f5SelectedKey() === tr.dataset.f5Key ? null : tr.dataset.f5Key);
    });
    tbody.addEventListener("keydown", ev => {
      if (ev.key !== "Enter" && ev.key !== " ") return;
      const tr = ev.target.closest("tr[data-f5-key]");
      if (!tr) return;
      ev.preventDefault();
      _F5State.auditAge = null;
      _f5SetSelectedKey(_f5SelectedKey() === tr.dataset.f5Key ? null : tr.dataset.f5Key);
    });
  }
}

function renderFiveXFADKinase() {
  const panel = document.getElementById("tab-fivexfadkinase");
  if (!panel) return;
  if (typeof renderUnmetPrerequisite === "function" && renderUnmetPrerequisite(panel, "fivexfadkinase")) return;
  const block = _f5Block();
  if (!block || !Array.isArray(block.rows) || block.rows.length === 0) {
    const detail = document.getElementById("f5-detail");
    const count = document.getElementById("f5-count");
    if (count) count.textContent = "5xFAD payload unavailable";
    if (detail) detail.innerHTML = '<div class="muted">5xFAD payload data are not available in this viewer build.</div>';
    return;
  }
  _f5PopulateControls();
  _f5SyncControls();
  const rows = _f5FilteredRows();
  const selectedKey = _f5SelectedKey();
  if (selectedKey && !rows.some(r => r.key === selectedKey)) {
    _F5State.auditAge = null;
    _f5SetSelectedKey(null);
    return;
  }
  const count = document.getElementById("f5-count");
  if (count) count.textContent = `${rows.length.toLocaleString()} kinases`;
  const tbody = document.querySelector("#f5-table tbody");
  if (!tbody) return;
  const maxAbs = _f5MaxAbsVisible(rows);
  const ages = _f5AgeScope();
  const denom = ages.length;
  const html = rows.map(r => {
    const sel = r.key === selectedKey ? " selected" : "";
    const sub = r.sigCount === 0 ? " sub-thresh" : "";
    const residueBadge = r.residue_type === "Y"
      ? ' <span class="track-badge track-y" title="Tyrosine kinase (pY track)">pY</span>'
      : "";
    return `<tr data-f5-key="${_f5Esc(r.key)}" tabindex="0" class="ke-row${sel}${sub}" aria-label="5xFAD kinase ${_f5Esc(r.kinase)}; ${r.sigCount} significant ages">
      <td>${_f5Esc(r.kinase)}${residueBadge}</td>
      <td>${_f5Esc(r.gene_symbol || "")}</td>
      <td>${_f5Esc(r.family || "")}</td>
      <td>${_f5Esc(r.residue_type || "")}</td>
      <td>${_f5Profile(r, maxAbs)}</td>
      <td class="attr-num">${r.peakAbsNes == null ? '<span class="muted">—</span>' : r.peakAbsNes.toFixed(2)}</td>
      <td class="attr-num">${r.sigCount}<span class="muted" style="font-size:10px;"> / ${denom}</span></td>
      <td>${_f5CellTypesCell(r)}</td>
      <td style="text-align:center;">${_f5SongBadge(r)}</td>
      <td style="text-align:center;">${_f5WmbBadge(r)}</td>
      <td>${_f5ConfBadge(r)}</td>
    </tr>`;
  }).join("");
  tbody.innerHTML = html || '<tr><td colspan="11" class="muted">No 5xFAD rows match the active filters.</td></tr>';
  _f5SyncSortIndicators();
  renderFiveXFADKinaseDetail();
}

function _f5SyncSortIndicators() {
  document.querySelectorAll("#f5-table thead th").forEach(th => {
    const c = th.dataset.col;
    th.textContent = th.textContent.replace(/[ ▲▼]+$/, "");
    if (c === _F5State.sortCol) th.textContent += _F5State.sortAsc ? " ▲" : " ▼";
  });
}

function _f5SelectedGroup() {
  _f5EnsureIndexes();
  const key = _f5SelectedKey();
  return _F5Groups.find(g => g.key === key) || null;
}

function _f5SelectionLabel(key) {
  _f5EnsureIndexes();
  const group = _F5Groups.find(g => g.key === key);
  return group ? `${group.kinase} · ${_f5SurfaceLabel(group)}` : String(key || "");
}

function updateFiveXFADKinaseSelection(key) {
  _f5UpdateRowSelection(key);
  if (key == null) _F5State.auditAge = null;
  renderFiveXFADKinaseDetail();
}

function _f5UpdateRowSelection(key) {
  const tbody = document.querySelector("#f5-table tbody");
  if (!tbody) return;
  const prev = tbody.querySelector("tr.ke-row.selected");
  if (prev) prev.classList.remove("selected");
  if (key == null) return;
  const safeKey = String(key).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  const row = tbody.querySelector(`tr.ke-row[data-f5-key="${safeKey}"]`);
  if (row) row.classList.add("selected");
}

function _f5GroupsForKinase(name) {
  _f5EnsureIndexes();
  return _F5Groups.filter(g => g.kinase === name);
}

function _f5SurfaceValues(group, field) {
  return Array.from(new Set(_f5GroupsForKinase(group.kinase)
    .filter(g => g.assay === group.assay)
    .map(g => g[field]).filter(Boolean))).sort();
}

function _f5SurfaceOptions(group, field, labels) {
  return _f5SurfaceValues(group, field).map(v => {
    const lab = labels && labels[v] ? labels[v] : v;
    return `<option value="${_f5Esc(v)}"${v === group[field] ? " selected" : ""}>${_f5Esc(lab)}</option>`;
  }).join("");
}

function _f5GroupForSurface(kinase, tissue, assay) {
  return _f5GroupsForKinase(kinase).find(g =>
    g.tissue === tissue && g.assay === assay && g.analysis_track === "stoichiometry"
  ) || null;
}

function _f5SelectedAge(group) {
  if (_F5State.auditAge && group.rows.has(Number(_F5State.auditAge))) return Number(_F5State.auditAge);
  if (_F5State.age && group.rows.has(Number(_F5State.age))) return Number(_F5State.age);
  let bestAge = null;
  let bestAbs = -1;
  for (const age of _F5_AGES) {
    const row = group.rows.get(age);
    const nes = row ? _f5Num(row.NES) : null;
    if (nes != null && Math.abs(nes) > bestAbs) {
      bestAbs = Math.abs(nes);
      bestAge = age;
    }
  }
  if (bestAge == null) bestAge = _F5_AGES.find(age => group.rows.has(age)) || _F5_AGES[0];
  _F5State.auditAge = bestAge;
  return bestAge;
}

function renderFiveXFADKinaseDetail() {
  const host = document.getElementById("f5-detail");
  if (!host) return;
  const group = _f5SelectedGroup();
  if (!group) {
    host.innerHTML = '<div class="muted">Select a kinase to see details.</div>';
    return;
  }
  const age = _f5SelectedAge(group);
  const tissueOptions = _f5SurfaceOptions(group, "tissue");
  const ageOptions = _F5_AGES.map(a => {
    const disabled = group.rows.has(a) ? "" : " disabled";
    return `<option value="${a}"${a === age ? " selected" : ""}${disabled}>${a}mo</option>`;
  }).join("");
  const tabs = _F5_AUDIT_TABS;
  if (!tabs.some(t => t.id === _F5State.auditTab)) _F5State.auditTab = "mea-score";
  const tabButtons = tabs.map(t =>
    `<button type="button" data-f5-audit-tab="${t.id}" class="${t.id === _F5State.auditTab ? "active" : ""}">${_f5Esc(t.label)}</button>`
  ).join("");

  host.innerHTML = `
    <div class="kinase-workbench-header">
      <div class="kinase-workbench-title">
        <h3>${_f5Esc(group.kinase)}</h3>
        <div class="muted">${_f5Esc(group.gene_symbol || "")}${group.residue_type ? " · " + _f5Esc(group.residue_type) : ""} · ${_f5Esc(group.tissue || "")}</div>
      </div>
      <div class="kinase-workbench-controls">
        <label>Tissue <select id="f5-audit-tissue">${tissueOptions}</select></label>
        <label>Age <select id="f5-audit-age">${ageOptions}</select></label>
      </div>
    </div>
    <div class="kinase-audit-tabs" role="tablist" aria-label="5xFAD kinase audit walkthrough">${tabButtons}</div>
    <div class="kinase-audit-tab-body" id="f5-audit-body"></div>
  `;

  const updateSurface = () => {
    const tissue = document.getElementById("f5-audit-tissue")?.value || group.tissue;
    const next = _f5GroupForSurface(group.kinase, tissue, group.assay);
    if (next) {
      _F5State.auditAge = null;
      _f5SetSelectedKey(next.key);
    }
  };
  ["f5-audit-tissue"].forEach(id => {
    const sel = document.getElementById(id);
    if (sel) sel.addEventListener("change", updateSurface);
  });
  const ageSel = document.getElementById("f5-audit-age");
  if (ageSel) ageSel.addEventListener("change", ev => {
    _F5State.auditAge = Number(ev.target.value);
    _f5RenderAuditBody(group);
  });
  host.querySelectorAll("[data-f5-audit-tab]").forEach(btn => {
    btn.addEventListener("click", () => {
      _F5State.auditTab = btn.dataset.f5AuditTab;
      renderFiveXFADKinaseDetail();
    });
  });
  _f5RenderAuditBody(group);
}

function _f5RenderAuditBody(group) {
  const body = document.getElementById("f5-audit-body");
  if (!body) return;
  if (_F5State.auditTab === "measurement-trace") return _f5RenderTrace(body, group);
  if (_F5State.auditTab === "mea-input") return _f5RenderPrep(body, group);
  if (_F5State.auditTab === "attribution") return _f5RenderAttribution(body, group);
  return _f5RenderScore(body, group);
}

function _f5SelectedRow(group) {
  const age = _f5SelectedAge(group);
  return group.rows.get(age) || null;
}

function _f5ScoreTier(row) {
  const f = _f5Num(row && row.FDR);
  const gate = Store.state.filters.fdr;
  if (f == null) return {label: "no FDR", cls: "muted"};
  if (f < gate) return {label: `FDR ${f.toFixed(3)} · passes ${gate}`, cls: "chip-pass"};
  if (f < gate * 2) return {label: `FDR ${f.toFixed(3)} · borderline`, cls: "chip-borderline"};
  return {label: `FDR ${f.toFixed(3)} · fails ${gate}`, cls: "chip-fail"};
}

function _f5RowForTrack(group, age, analysisTrack) {
  if (analysisTrack === "stoichiometry") return group.rows.get(Number(age)) || null;
  const rawRows = _F5RawByKey ? _F5RawByKey.get(group.key) : null;
  return rawRows ? (rawRows.get(Number(age)) || null) : null;
}

function _f5FmtSigned(v, digits) {
  const n = _f5Num(v);
  if (n == null) return "—";
  return `${n > 0 ? "+" : ""}${n.toFixed(digits == null ? 3 : digits)}`;
}

function _f5SubstrateText(row) {
  return row && row.substrate_hits != null ? `${row.substrate_hits}/${row.substrate_universe}` : "—";
}

function _f5ComparisonHtml(group, age) {
  const stoich = _f5RowForTrack(group, age, "stoichiometry");
  const raw = _f5RowForTrack(group, age, "raw_phospho");
  const rows = [
    {metric: "NES", stoich: stoich && stoich.NES, raw: raw && raw.NES, digits: 2},
    {metric: "ES", stoich: stoich && stoich.ES, raw: raw && raw.ES, digits: 3},
    {metric: "p-value", stoich: stoich && stoich.p_value, raw: raw && raw.p_value, digits: 4},
    {metric: "FDR", stoich: stoich && stoich.FDR, raw: raw && raw.FDR, digits: 3},
  ].map(r => {
    const s = _f5Num(r.stoich);
    const rw = _f5Num(r.raw);
    const delta = (s == null || rw == null) ? null : s - rw;
    return `<tr><td>${_f5Esc(r.metric)}</td><td>${_f5Fmt(s, r.digits)}</td><td>${_f5Fmt(rw, r.digits)}</td><td>${_f5FmtSigned(delta, r.digits)}</td></tr>`;
  }).join("");
  const subs = `<tr><td>Substrates tested</td><td>${_f5Esc(_f5SubstrateText(stoich))}</td><td>${_f5Esc(_f5SubstrateText(raw))}</td><td>—</td></tr>`;
  return `<div class="kh-audit-tablewrap"><table class="data-table">`
    + `<thead><tr><th>metric</th><th>stoichiometry</th><th>raw phospho</th><th>Δ (stoich − raw)</th></tr></thead>`
    + `<tbody>${rows}${subs}</tbody></table></div>`;
}

function _f5NesColor(nes, fdr) {
  const n = _f5Num(nes);
  if (n == null) return "#eceff1";
  const base = n >= 0 ? [197, 48, 48] : [43, 108, 176];
  const sig = _f5Num(fdr) != null && _f5Num(fdr) < Store.state.filters.fdr;
  const alpha = sig ? 0.9 : 0.32;
  return `rgba(${base[0]},${base[1]},${base[2]},${alpha})`;
}

function _f5RenderTrajectory(hostId, group, age) {
  const host = document.getElementById(hostId);
  if (!host || typeof Plotly === "undefined") return;
  const rawRowsByAge = _F5RawByKey ? _F5RawByKey.get(group.key) : null;
  const labels = _F5_AGES.map(a => `${a}mo`);
  const stoichRows = _F5_AGES.map(a => group.rows.get(a) || null);
  const rawRows = _F5_AGES.map(a => rawRowsByAge ? (rawRowsByAge.get(a) || null) : null);
  const colors = stoichRows.map(r => _f5NesColor(r && r.NES, r && r.FDR));
  const outlines = _F5_AGES.map(a => Number(a) === Number(age) ? "#000" : "rgba(0,0,0,0)");
  const lineWidths = _F5_AGES.map(a => Number(a) === Number(age) ? 2.5 : 0);
  Plotly.react(hostId, [
    {
      type: "bar",
      x: labels,
      y: stoichRows.map(r => r ? _f5Num(r.NES) : null),
      marker: {color: colors, line: {color: outlines, width: lineWidths}},
      name: "stoichiometry NES",
      hovertemplate: "%{x}<br>stoich NES %{y:.2f}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "markers",
      x: labels,
      y: rawRows.map(r => r ? _f5Num(r.NES) : null),
      marker: {color: "#000", size: 9, symbol: "diamond-open", line: {width: 1.5, color: "#000"}},
      name: "raw phospho NES",
      hovertemplate: "%{x}<br>raw NES %{y:.2f}<extra></extra>",
    },
  ], {
    margin: {l: 40, r: 10, t: 10, b: 45},
    height: 220,
    yaxis: {zeroline: true, zerolinecolor: "#bbb", title: "NES"},
    showlegend: false,
  }, {displaylogo: false, responsive: true}).then(() => {
    if (host.removeAllListeners) host.removeAllListeners("plotly_click");
    if (host.on) {
      host.on("plotly_click", ev => {
        const pts = ev && ev.points ? ev.points : null;
        if (!pts || !pts[0]) return;
        const next = parseInt(String(pts[0].x).replace("mo", ""), 10);
        if (_F5_AGES.includes(next) && group.rows.has(next)) {
          _F5State.auditAge = next;
          renderFiveXFADKinaseDetail();
        }
      });
    }
  });
}

function _f5RenderRunningEnrichment(hostId, group, age) {
  const host = document.getElementById(hostId);
  if (!host) return;
  _f5LoadDetail(group).then(detail => {
    if (!_f5StillSelected(group)) return;
    if (!detail) {
      host.innerHTML = _f5DetailMissing(group);
      return;
    }
    const contrast = _f5ContrastForAge(age);
    const rec = (detail.running_enrichment || []).find(r => r.contrast === contrast);
    if (!rec || !rec.line || !rec.line.length || typeof Plotly === "undefined") {
      host.innerHTML = `<div class="muted" style="padding:1em">Running enrichment requires the full prerank list and substrate-set motifs for this kinase surface.</div>`;
      return;
    }
    const line = rec.line || [];
    const hits = rec.hits || [];
    const peakRank = rec.peak_rank;
    const peakEs = _f5Num(rec.peak_es);
    Plotly.react(hostId, [
      {
        type: "scatter",
        mode: "lines",
        x: line.map(r => r.rank),
        y: line.map(r => r.running_es),
        line: {color: "#1f77b4", width: 1.5},
        name: "running ES",
        hoverinfo: "skip",
      },
      {
        type: "scatter",
        mode: "markers",
        x: hits.map(r => r.rank),
        y: hits.map(r => r.running_es),
        marker: {color: "#1f77b4", size: 5, opacity: 0.9},
        name: "substrate hit",
        text: hits.map(r => `rank ${r.rank}<br>${_f5Esc(_f5SiteLabel(r))} · ${_f5Esc(r.motif || "")}<br>clipped LFC ${_f5Fmt(r.clipped_lfc, 3)}<br>running ES ${_f5Fmt(r.running_es, 3)}`),
        hovertemplate: "%{text}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "markers",
        x: [peakRank],
        y: [peakEs],
        marker: {color: "#000", size: 9, symbol: "diamond"},
        name: "peak ES",
        hovertemplate: `peak ES ${_f5Fmt(peakEs, 3)} at rank ${peakRank}<extra></extra>`,
      },
    ], {
      margin: {l: 50, r: 10, t: 30, b: 40},
      height: 300,
      showlegend: false,
      annotations: [{
        x: peakRank,
        y: peakEs,
        xref: "x",
        yref: "y",
        text: `peak ES ${_f5Fmt(peakEs, 3)} at rank ${peakRank}<br>leading edge: ${rec.leading_edge_count || 0} of ${rec.n_hits || 0} hits`,
        showarrow: true,
        arrowhead: 2,
        ax: 30,
        ay: peakEs >= 0 ? -40 : 40,
        font: {size: 11},
      }],
      shapes: [{
        type: "line",
        xref: "x",
        yref: "y",
        x0: 1,
        x1: rec.n_ranked || peakRank,
        y0: 0,
        y1: 0,
        line: {color: "#999", width: 1, dash: "dot"},
      }],
      xaxis: {title: "prerank rank (1 = most up-shifted)", range: [1, rec.n_ranked || peakRank]},
      yaxis: {title: "running ES", zeroline: false},
    }, {displaylogo: false, responsive: true});
  });
}

function _f5RenderScore(body, group) {
  const row = _f5SelectedRow(group);
  const age = _f5SelectedAge(group);
  const tier = _f5ScoreTier(row);
  const nes = _f5Num(row && row.NES);
  const nesColor = nes == null ? "#666" : (nes > 0 ? "#c53030" : "#2b6cb0");
  body.innerHTML = `
    <p class="kinase-stage-note">The score for ${_f5Esc(group.kinase)} on ${age}mo TG vs WT: how the kinase's substrate set concentrates in the contrast prerank. Stoichiometry is primary; raw phospho is shown alongside for cross-track sanity.</p>
    <section class="audit-panel">
      <h4>Score for ${age}mo TG vs WT</h4>
      <div class="mea-scorecard">
        <div class="mea-score-nes" style="color:${nesColor}">
          <div class="mea-score-label">NES</div>
          <div class="mea-score-value">${nes == null ? "—" : nes.toFixed(2)}</div>
          <div class="mea-score-chip ${tier.cls}">${_f5Esc(tier.label)}</div>
        </div>
        <dl class="mea-score-stats">
          <dt>ES</dt><dd>${_f5Fmt(row && row.ES, 3)}</dd>
          <dt>p-value</dt><dd>${_f5Fmt(row && row.p_value, 4)}</dd>
          <dt>Substrates tested</dt><dd>${_f5Esc(_f5SubstrateText(row))}</dd>
          <dt>Samples</dt><dd>WT ${row && row.n_wt != null ? row.n_wt : "—"} · TG ${row && row.n_tg != null ? row.n_tg : "—"}</dd>
        </dl>
      </div>
    </section>
    <section class="audit-panel" style="margin-top:10px;">
      <h4>Running enrichment for ${age}mo TG vs WT</h4>
      <p class="kinase-stage-note">GSEA walk recomputed from the packaged prerank receipt. The curve steps up at substrate hits and down at misses. Peak ES and the leading-edge count are marked.</p>
      <div id="f5-mea-running" style="height:300px"></div>
    </section>
    <section class="audit-panel" style="margin-top:10px;">
      <h4>NES across ages</h4>
      <p class="kinase-stage-note">Stoichiometry NES bars: full saturation when FDR is below the header threshold, faded otherwise. Raw phospho NES is shown as paired open diamonds. Click a bar to switch age.</p>
      <div id="f5-mea-trajectory" style="height:220px"></div>
    </section>
    <section class="audit-panel" style="margin-top:10px;">
      <h4>Stoichiometry vs raw phospho for ${age}mo TG vs WT</h4>
      <p class="kinase-stage-note">Per-metric comparison of the same kinase, tissue, assay, and age scored against the two preprocessing tracks. Δ = stoichiometry − raw phospho.</p>
      ${_f5ComparisonHtml(group, age)}
    </section>`;
  _f5RenderRunningEnrichment("f5-mea-running", group, age);
  _f5RenderTrajectory("f5-mea-trajectory", group, age);
}

function _f5SmallTable(rows, cols) {
  if (!rows || !rows.length) return '<div class="muted">No rows available.</div>';
  const head = cols.map(c => `<th>${_f5Esc(c.label)}</th>`).join("");
  const body = rows.map(r => `<tr>${cols.map(c => {
    const v = r[c.key];
    const txt = c.fmt ? c.fmt(v, r) : (v == null ? "" : v);
    return `<td>${c.html ? txt : _f5Esc(txt)}</td>`;
  }).join("")}</tr>`).join("");
  return `<div class="kh-audit-tablewrap"><table class="data-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function _f5SiteLabel(row) {
  if (!row) return "";
  const explicit = String(row.site_label || "").trim();
  if (explicit) return explicit;
  const gene = String(row.gene_symbol || "").trim();
  const residue = String(row.residue_type || "").trim().toUpperCase();
  const pos = String(row.site_position || "").trim();
  if (gene && residue && pos) return `${gene}_${residue}${pos}`;
  const sid = String(row.site_id || "").trim();
  const match = sid.match(/_([STY])(\d+)(?:_[^_\s]+)*$/i);
  if (gene && match) return `${gene}_${match[1].toUpperCase()}${match[2]}`;
  const symbol = sid.match(/gene_symbol:([A-Za-z0-9_.-]+)/);
  if (symbol && match) return `${symbol[1]}_${match[1].toUpperCase()}${match[2]}`;
  return sid;
}

function _f5SiteCell(_value, row) {
  const label = _f5SiteLabel(row);
  const sid = String(row && row.site_id != null ? row.site_id : "");
  return sid && sid !== label
    ? `<span title="${_f5Esc(sid)}">${_f5Esc(label)}</span>`
    : _f5Esc(label);
}

function _f5FmtShort(v) {
  const n = _f5Num(v);
  if (n == null) return "—";
  if (Math.abs(n) > 0 && Math.abs(n) < 0.001) return n.toExponential(2);
  return n.toFixed(3);
}

function _f5LoadDetail(group) {
  const path = _f5DetailShard(group);
  if (!path) return Promise.resolve(null);
  if (_F5DetailCache.has(path)) return _F5DetailCache.get(path);
  const p = fetch(path).then(resp => {
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
  }).catch(err => {
    console.warn("5xFAD detail shard fetch failed", path, err);
    return null;
  });
  _F5DetailCache.set(path, p);
  return p;
}

function _f5StillSelected(group) {
  return group && group.key === _f5SelectedKey();
}

function _f5ContrastForAge(age) {
  return `TG_vs_WT_${age}mo`;
}

function _f5DetailMissing(group) {
  const shard = _f5DetailShard(group);
  if (!shard) return '<div class="muted">No 5xFAD detail shard is listed for this kinase surface.</div>';
  if (window.location && window.location.protocol === "file:") {
    return '<div class="muted">Detail sidecars are blocked under file://. Serve the unified viewer directory over HTTP to inspect this tab.</div>';
  }
  return '<div class="muted">5xFAD detail sidecar could not be loaded.</div>';
}

function _f5RenderAsyncPanel(body, group, renderer) {
  body.innerHTML = '<div class="muted" style="padding:1em">Loading 5xFAD detail shard…</div>';
  _f5LoadDetail(group).then(detail => {
    if (!_f5StillSelected(group)) return;
    if (!detail) {
      body.innerHTML = _f5DetailMissing(group);
      return;
    }
    renderer(detail);
  });
}

function _f5RenderPrep(body, group) {
  const age = _f5SelectedAge(group);
  _f5RenderAsyncPanel(body, group, detail => {
    const contrast = _f5ContrastForAge(age);
    const shiftRows = (detail.global_shift || []).filter(r => r.contrast === contrast);
    const winRows = (detail.winsorized_sites || []).filter(r => r.contrast === contrast).slice(0, 25);
    const prepRows = (detail.prepared_mea_input || []).filter(r => r.contrast === contrast);
    const siteRows = (detail.site_stats || []).filter(r => r.contrast === contrast);
    body.innerHTML = `
      <section class="audit-panel">
        <h4>Step 1 · Global shift</h4>
        <p class="kinase-stage-note">Median LFC across the contrast's ranked sites, subtracted before GSEA so the prerank is centered at zero. Contrast-level, not kinase-specific.</p>
        ${_f5SmallTable(shiftRows, [
          {key: "contrast", label: "Contrast"},
          {key: "median_shift", label: "Median shift", fmt: _f5FmtShort},
          {key: "mean_before", label: "Mean before", fmt: _f5FmtShort},
          {key: "pct_pos_before", label: "% positive before", fmt: _f5FmtShort},
          {key: "pct_pos_after", label: "% positive after", fmt: _f5FmtShort},
        ])}
      </section>
      <section class="audit-panel" style="margin-top:10px;">
        <h4>Step 2 · Winsorization</h4>
        <p class="kinase-stage-note">Centered LFCs clipped at the contrast bounds so individual sites cannot dominate the prerank.</p>
        ${_f5SmallTable(winRows, [
          {key: "site_id", label: "Site", fmt: _f5SiteCell, html: true},
          {key: "gene_symbol", label: "Gene"},
          {key: "original_lfc", label: "Original LFC", fmt: _f5FmtShort},
          {key: "clipped_lfc", label: "Clipped LFC", fmt: _f5FmtShort},
          {key: "lower_bound", label: "Lower", fmt: _f5FmtShort},
          {key: "upper_bound", label: "Upper", fmt: _f5FmtShort},
        ])}
      </section>
      <section class="audit-panel audit-wide" style="margin-top:10px;">
        <h4>Step 3 · Prepared MEA input for this kinase</h4>
        <p class="kinase-stage-note">One row per site whose motif is in this kinase's substrate set and present in the contrast prerank. Rank, centered LFC, clipped LFC, and leading-edge status mirror the inputs used to score the kinase.</p>
        ${_f5SmallTable(prepRows, [
          {key: "rank_in_contrast", label: "Rank"},
          {key: "site_id", label: "Site", fmt: _f5SiteCell, html: true},
          {key: "gene_symbol", label: "Gene"},
          {key: "motif", label: "Motif"},
          {key: "kl_percentile", label: "kl_%", fmt: v => _f5Fmt(v, 1)},
          {key: "lfc", label: "LFC", fmt: _f5FmtShort},
          {key: "centered_lfc", label: "Centered LFC", fmt: _f5FmtShort},
          {key: "clipped_lfc", label: "Clipped LFC", fmt: _f5FmtShort},
          {key: "was_winsorized", label: "Winsorized"},
          {key: "in_leading_edge", label: "Leading edge"},
        ])}
      </section>
      <section class="audit-panel audit-wide" style="margin-top:10px;">
        <h4>Substrate-site OLS details <span class="muted">(${age}mo TG vs WT)</span></h4>
        <p class="kinase-stage-note">Substrate-site contrast rows for this kinase, restricted to the same prepared input universe shown above.</p>
        ${_f5SmallTable(siteRows, [
          {key: "rank_in_contrast", label: "Rank"},
          {key: "site_id", label: "Site", fmt: _f5SiteCell, html: true},
          {key: "gene_symbol", label: "Gene"},
          {key: "motif", label: "Motif"},
          {key: "kl_percentile", label: "kl_%", fmt: v => _f5Fmt(v, 1)},
          {key: "lfc", label: "LFC", fmt: _f5FmtShort},
          {key: "p_value", label: "p-value", fmt: _f5FmtShort},
          {key: "fdr", label: "FDR", fmt: _f5FmtShort},
          {key: "n_wt", label: "WT"},
          {key: "n_tg", label: "TG"},
          {key: "clipped_lfc", label: "Clipped LFC", fmt: _f5FmtShort},
        ])}
      </section>`;
  });
}

function _f5RenderOls(body, group) {
  const age = _f5SelectedAge(group);
  _f5RenderAsyncPanel(body, group, detail => {
    const contrast = _f5ContrastForAge(age);
    const rows = (detail.site_stats || []).filter(r => r.contrast === contrast);
    body.innerHTML = `
      <section class="audit-panel">
        <h4>OLS Details <span class="muted">(${age}mo TG vs WT)</span></h4>
        <p class="kinase-stage-note">Substrate-site rows for this kinase and contrast. LFC, p-value, FDR, and group counts come from the 5xFAD site-level OLS output used upstream of MEA.</p>
        ${_f5SmallTable(rows, [
          {key: "rank_in_contrast", label: "Rank"},
          {key: "site_id", label: "Site", fmt: _f5SiteCell, html: true},
          {key: "gene_symbol", label: "Gene"},
          {key: "motif", label: "Motif"},
          {key: "lfc", label: "LFC", fmt: _f5FmtShort},
          {key: "p_value", label: "p-value", fmt: _f5FmtShort},
          {key: "fdr", label: "FDR", fmt: _f5FmtShort},
          {key: "n_wt", label: "WT"},
          {key: "n_tg", label: "TG"},
          {key: "clipped_lfc", label: "Clipped LFC", fmt: _f5FmtShort},
        ])}
      </section>`;
  });
}

function _f5RenderTrace(body, group) {
  const age = _f5SelectedAge(group);
  _f5RenderAsyncPanel(body, group, detail => {
    const siteRows = (detail.site_stats || []).filter(r => r.contrast === _f5ContrastForAge(age));
    const siteSet = new Set(siteRows.slice(0, 8).map(r => r.site_id));
    const rows = (detail.measurement_trace || [])
      .filter(r => Number(r.age_months) === age && (!siteSet.size || siteSet.has(r.site_id)))
      .slice(0, 160);
    const motif = (PAYLOAD.kinase_motifs || {})[group.kinase] || null;
    const logoBlock = (typeof SequenceLogo !== "undefined" && SequenceLogo.buildBlock)
      ? SequenceLogo.buildBlock(group.kinase, motif, "f5-trace-logo")
      : "";
    body.innerHTML = logoBlock + `
      <section class="audit-panel">
        <h4>Measurement Trace <span class="muted">(${age}mo TG vs WT)</span></h4>
        <p class="kinase-stage-note">Sample-level receipt for leading substrate sites in the selected contrast. <code>kl_%</code> is the kinase-library substrate percentile; higher values indicate stronger kinase-library motif support.</p>
        ${_f5SmallTable(rows, [
          {key: "site_id", label: "Site", fmt: _f5SiteCell, html: true},
          {key: "gene_symbol", label: "Gene"},
          {key: "motif", label: "Motif"},
          {key: "kl_percentile", label: "kl_%", fmt: v => _f5Fmt(v, 1)},
          {key: "sample", label: "Sample"},
          {key: "genotype", label: "Genotype"},
          {key: "raw_phospho", label: "Raw phospho", fmt: _f5FmtShort},
          {key: "matched_total_protein", label: "Protein", fmt: _f5FmtShort},
          {key: "stoichiometry", label: "Stoichiometry", fmt: _f5FmtShort},
        ])}
      </section>`;
    if (motif && typeof SequenceLogo !== "undefined" && SequenceLogo.render) {
      SequenceLogo.render(document.getElementById("f5-trace-logo"), motif);
    }
  });
}

function _f5RenderAttribution(body, group) {
  const age = _f5SelectedAge(group);
  const row = _f5SelectedRow(group);
  const nes = _f5Num(row && row.NES);
  const fdr = _f5Num(row && row.FDR);
  const rows = _f5AttrRows(group.kinase).map(r => Object.assign({}, r, {
    wmb_tier: _f5WmbTier(r),
    fivexfad_nes: nes,
    fivexfad_fdr: fdr,
  }));
  const {sortCol, sortAsc} = _f5SortAttrRows(rows, body);
  const showAllId = "f5-attr-show-all";
  const showAll = body.dataset.f5AttrShowAll === "1";
  const visibleRows = showAll
    ? rows
    : rows.filter(r => ["very_high", "high", "moderate"].includes(r.confidence_tier || ""));
  const hiddenCount = rows.length - visibleRows.length;
  const anchor = nes == null
    ? '<span class="attr-bulk-ns">NES n/a</span>'
    : (nes > 0
        ? `<span class="attr-bulk-up">↑ NES = +${nes.toFixed(2)}</span>`
        : `<span class="attr-bulk-down">↓ NES = ${nes.toFixed(2)}</span>`);
  const fdrText = fdr == null ? "FDR n/a" : `FDR = ${fdr.toFixed(3)}${fdr < Store.state.filters.fdr ? "" : " (n.s.)"}`;
  const headCells = F5_ATTR_COLS.map(c => {
    const arrow = c.key === sortCol.key ? (sortAsc ? " ▲" : " ▼") : "";
    const title = c.title ? ` title="${_f5Esc(c.title)}"` : "";
    return `<th class="attr-verdict-th" data-sort-key="${_f5Esc(c.key)}"${title}>${_f5Esc(c.label)}${arrow}</th>`;
  }).join("");
  const groupCounts = F5_ATTR_COLS.reduce((acc, c) => {
    acc[c.group] = (acc[c.group] || 0) + 1;
    return acc;
  }, {});
  const superHead =
    `<tr class="attr-verdict-supergroup">` +
      `<th class="attr-supergroup-spacer" colspan="${groupCounts.id || 0}"></th>` +
      `<th class="attr-supergroup-attr" colspan="${groupCounts.attr || 0}" title="Cell-type attribution evidence from Song, WMB, and SEA-AD reference layers.">Attribution</th>` +
      `<th class="attr-supergroup-decomp" colspan="${groupCounts.activity || 0}" title="Bulk 5xFAD kinase activity for the selected tissue and age.">5xFAD kinase activity</th>` +
    `</tr>`;
  const num = (v, d=3) => {
    const n = _f5Num(v);
    return n == null ? "" : n.toFixed(d);
  };
  const tbody = visibleRows.map((r, i) => {
    const conf = r.confidence_tier || "none";
    const confChip = `<span class="${_attrConfidenceClass(conf)}" title="${_f5Esc(r.confidence_basis || conf)}">${_f5Esc(conf.replace("_", " "))}</span>`;
    const songCell = typeof _msTierBadge === "function"
      ? _msTierBadge(_f5SongTier(r), _f5Num(r.song_specificity), r.song_top_cluster || "", _f5Num(r.song_tau))
      : '<span class="muted">—</span>';
    const wmbCell = typeof _wmbTierBadge === "function"
      ? _wmbTierBadge(_f5WmbTier(r))
      : '<span class="muted">—</span>';
    const seaCell = _f5Num(r.sea_ad_lfc) == null
      ? `<td class="attr-num attr-empty">—</td>`
      : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(r.sea_ad_lfc)}">${num(r.sea_ad_lfc, 3)}</td>`;
    const nesCell = nes == null
      ? `<td class="attr-num attr-empty">—</td>`
      : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(nes)}">${num(nes, 2)}</td>`;
    const fdrCell = fdr == null
      ? `<td class="attr-num attr-empty">—</td>`
      : `<td class="attr-num"${fdr < Store.state.filters.fdr ? ' style="font-weight:600"' : ""}>${num(fdr, 3)}</td>`;
    return `<tr data-cell-type="${_f5Esc(r.cell_type || "")}" class="attr-verdict-row${i === 0 ? " attr-verdict-selected" : ""}">` +
      `<td class="attr-celltype">${_f5Esc(r.cell_type || "")}</td>` +
      `<td>${confChip}</td>` +
      `<td class="attr-num">${songCell}</td>` +
      `<td class="attr-num">${wmbCell}</td>` +
      seaCell +
      nesCell +
      fdrCell +
      `</tr>`;
  }).join("");
  body.innerHTML = `
    <p class="kinase-stage-note">Cell-type attribution of <strong>${_f5Esc(group.gene_symbol || group.kinase)}</strong> uses the same canonical mouse reference evidence shown in the Song tab. The 5xFAD bulk kinase activity is shown as the activity anchor for the selected tissue and age.</p>
    <div class="attr-bulk-anchor">Bulk MEA anchor for TG_vs_WT_${age}mo:
      <span class="attr-bulk-pill" title="Positive NES = kinase substrates concentrated among sites higher in 5xFAD TG than WT.">${anchor} · ${fdrText}</span>
      <span class="muted">— bulk 5xFAD activity is broadcast to each attribution row because no 5xFAD cell-type decomposition artifact is currently packaged.</span>
    </div>
    <table class="attr-verdict-table"><thead>${superHead}<tr>${headCells}</tr></thead><tbody>${tbody}</tbody></table>
    ${hiddenCount > 0
      ? `<div class="attr-verdict-toggle"><label><input type="checkbox" id="${showAllId}"${showAll ? " checked" : ""}> Show all Levy-t5 clusters <span class="muted">(${hiddenCount} hidden — low/none confidence)</span></label></div>`
      : ""}
    <details class="attr-explainer"><summary>How to read attribution confidence</summary>
      <div class="attr-explainer-body">
        <p>Confidence is a categorical evidence label, not a continuous score.</p>
        <table class="attr-explainer-table" style="margin-bottom:8px;">
          <thead><tr><th>Source</th><th>What it tells you</th></tr></thead><tbody>
            <tr><td><strong>Song</strong></td><td>Mouse within-cohort location evidence for the kinase transcript across Levy T5 clusters.</td></tr>
            <tr><td><strong>WMB</strong></td><td>Healthy mouse reference expression used as an independent location cross-check.</td></tr>
            <tr><td><strong>SEA-AD</strong></td><td>Human Alzheimer's disease direction where a mapped cell-type effect is available.</td></tr>
            <tr><td><strong>5xFAD NES / FDR</strong></td><td>Bulk 5xFAD kinase activity for the selected tissue and age; it is not a cell-type decomposition.</td></tr>
          </tbody></table>
      </div>
    </details>
    <div id="f5-attr-drawer"></div>`;
  body.querySelectorAll("tr.attr-verdict-row").forEach(tr => tr.addEventListener("click", () => {
    body.querySelectorAll("tr.attr-verdict-row").forEach(r => r.classList.remove("attr-verdict-selected"));
    tr.classList.add("attr-verdict-selected");
    const selected = rows.find(r => String(r.cell_type || "") === tr.dataset.cellType) || null;
    _f5RenderAttributionDrawer("f5-attr-drawer", group, selected);
  }));
  body.querySelectorAll("th.attr-verdict-th").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.sortKey;
    if (body.dataset.f5AttrSortKey === k) {
      body.dataset.f5AttrSortAsc = body.dataset.f5AttrSortAsc === "1" ? "0" : "1";
    } else {
      body.dataset.f5AttrSortKey = k;
      const col = F5_ATTR_COLS.find(c => c.key === k);
      body.dataset.f5AttrSortAsc = (col && col.type === "str") ? "1" : "0";
    }
    _f5RenderAttribution(body, group);
  }));
  const toggle = document.getElementById(showAllId);
  if (toggle) toggle.addEventListener("change", () => {
    body.dataset.f5AttrShowAll = toggle.checked ? "1" : "0";
    _f5RenderAttribution(body, group);
  });
  if (visibleRows[0]) _f5RenderAttributionDrawer("f5-attr-drawer", group, visibleRows[0]);
}

function _f5RenderAttributionDrawer(hostId, group, row) {
  const host = document.getElementById(hostId);
  if (!host) return;
  if (!row) {
    host.innerHTML = '<div class="muted" style="padding:1em;">No attribution row selected.</div>';
    return;
  }
  const gene = group.gene_symbol || row.gene_symbol || group.kinase || "";
  const song = _f5Num(row.song_specificity);
  const songTxt = song == null ? "—" : `${(song / _MS_UNIFORM).toFixed(1)}x even-split`;
  const wmb = _f5Num(row.wmb_specificity);
  const wmbTxt = wmb == null ? "—" : wmb.toFixed(3);
  const sea = _f5Num(row.sea_ad_lfc);
  host.innerHTML =
    `<div class="attr-drawer-header"><strong>${_f5Esc(row.cell_type || "")}</strong>` +
      ` &middot; <span class="muted">${_f5Esc(gene)} / ${_f5Esc(group.tissue || "")}</span>` +
      ` &middot; ${typeof _allenABALink === "function" ? _allenABALink(gene) : ""}</div>` +
    `<div class="attr-drawer-grid">` +
      `<section class="attr-section"><h5>Mouse location evidence</h5>` +
        `<p class="muted attr-caption">Canonical Song/WMB attribution evidence reused by the mouse kinase viewers.</p>` +
        `<div class="kh-audit-tablewrap"><table class="data-table"><tbody>` +
          `<tr><th>Confidence</th><td><span class="${_attrConfidenceClass(row.confidence_tier || "none")}" title="${_f5Esc(row.confidence_basis || "")}">${_f5Esc(String(row.confidence_tier || "none").replace("_", " "))}</span></td></tr>` +
          `<tr><th>Song</th><td>${typeof _msTierBadge === "function" ? _msTierBadge(_f5SongTier(row), song, row.song_top_cluster || "", _f5Num(row.song_tau)) : _f5Esc(songTxt)} <span class="muted">${_f5Esc(songTxt)}</span></td></tr>` +
          `<tr><th>WMB</th><td>${typeof _wmbTierBadge === "function" ? _wmbTierBadge(_f5WmbTier(row)) : ""} <span class="muted">${_f5Esc(wmbTxt)}</span></td></tr>` +
        `</tbody></table></div></section>` +
      `<section class="attr-section"><h5>Cross-dataset disease evidence</h5>` +
        `<p class="muted attr-caption">SEA-AD evidence is a reference disease-direction layer. 5xFAD activity remains bulk tissue-level activity.</p>` +
        `<div class="kh-audit-tablewrap"><table class="data-table"><tbody>` +
          `<tr><th>SEA-AD LFC</th><td>${sea == null ? '<span class="muted">—</span>' : `<span class="attr-num-lfc" style="background:${_attrLfcColor(sea)}">${sea.toFixed(3)}</span>`}</td></tr>` +
          `<tr><th>5xFAD tissue</th><td>${_f5Esc(group.tissue || "")}</td></tr>` +
          `<tr><th>5xFAD age</th><td>${_f5Esc(String(_f5SelectedAge(group)))}mo TG vs WT</td></tr>` +
        `</tbody></table></div></section>` +
    `</div>`;
}
