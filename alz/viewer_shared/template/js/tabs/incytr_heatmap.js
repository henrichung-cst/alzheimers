// ---------------------------------------------------------------------------
// Incytr Heatmap tab — sender×receiver candidate-path counts for a chosen
// contrast. Filter UI: context group × timepoint/baseline selects, optional
// pvalue gate, |PDS| effect-size floor, Reset button + live count line.
//
// State lives in IncytrFilter (shared with the Pathway table tab) so picks
// flow across tabs. Click on a heatmap cell → seeds the table tab's
// senderIn / receiverIn / disease / timepoint filters and switches tabs.
// ---------------------------------------------------------------------------

function _ihBlock() {
  // Grain-aware: when a backbone grain is active, _ipGrainBlock() overlays that
  // grain's heatmap_counts / heatmap_counts_signed. The build aligns grid indices
  // to Full (same sender × receiver × contrast × pvalue × |PDS| axes), so the
  // decoder (_ihCountAt) and every axis/threshold helper work unchanged — the
  // cell value just becomes a distinct-backbone count instead of a full-path count.
  return (typeof window._ipGrainBlock === "function")
    ? window._ipGrainBlock()
    : ViewerPayload.incytr();
}

// "paths" at Full grain, "backbones" at any backbone grain — the counted entity.
function _ihEntityNoun() {
  const block = _ihBlock();
  return (block && block._grain && block._grain !== "Full") ? "backbones" : "paths";
}

function _ihAxisParts() {
  const block = _ihBlock();
  const axis = ViewerPayload.contrastAxis();
  const groups = (block && block.diseases && block.diseases.length)
    ? block.diseases
    : (axis.groups || []);
  const timepoints = (block && block.timepoints && block.timepoints.length)
    ? block.timepoints
    : (axis.timepoints || []);
  if ((groups && groups.length) || (timepoints && timepoints.length) || !block) {
    return { groups: groups || [], timepoints: timepoints || [] };
  }
  const g = [], t = [];
  const seenG = new Set(), seenT = new Set();
  for (const c of block.contrasts || []) {
    const i = String(c).indexOf("_");
    const a = i < 0 ? String(c) : String(c).slice(0, i);
    const b = i < 0 ? "" : String(c).slice(i + 1);
    if (!seenG.has(a)) { seenG.add(a); g.push(a); }
    if (b && !seenT.has(b)) { seenT.add(b); t.push(b); }
  }
  return { groups: g, timepoints: t };
}

function _ihDiseases() { return _ihAxisParts().groups; }
function _ihTimepoints() { return _ihAxisParts().timepoints; }

function _ihContrastFromState() {
  const f = IncytrFilter.get();
  return `${f.hmDisease}_${f.hmTimepoint}`;
}

function _ihScaleMode() {
  const f = IncytrFilter.get();
  return f.hmScale === "log1p" ? "log1p" : "linear";
}

function _ihScaleValue(n) {
  return _ihScaleMode() === "log1p" ? Math.log1p(n || 0) : n;
}

function _ihColorbarTitle() {
  const noun = _ihEntityNoun();
  return _ihScaleMode() === "log1p" ? `log1p(n ${noun})` : `n ${noun}`;
}

function _ihPdsSignMode() {
  const f = IncytrFilter.get();
  // Accept legacy "positive"/"negative" values from saved state; normalise to up/down.
  if (f.hmPdsSign === "positive" || f.hmPdsSign === "up") return "up";
  if (f.hmPdsSign === "negative" || f.hmPdsSign === "down") return "down";
  return "both";
}

function _ihSignIndex() {
  const mode = _ihPdsSignMode();
  if (mode === "up") return 2;
  if (mode === "down") return 0;
  return null;
}

function _ihSignText() {
  const mode = _ihPdsSignMode();
  if (mode === "up") return "up PDS";
  if (mode === "down") return "down PDS";
  return "both PDS signs";
}

function _ihTimelinePanels(f) {
  const block = _ihBlock();
  if (!block) return [];
  const groups = _ihDiseases();
  const timepoints = _ihTimepoints();
  const panels = [];
  if (timepoints.length > 1) {
    const disease = groups.indexOf(f.hmDisease) >= 0 ? f.hmDisease : groups[0];
    for (const tp of timepoints) {
      panels.push({ label: tp, disease, timepoint: tp, contrast: `${disease}_${tp}` });
    }
  } else if (groups.length > 1) {
    const tp = timepoints.indexOf(f.hmTimepoint) >= 0 ? f.hmTimepoint : (timepoints[0] || "");
    for (const disease of groups) {
      panels.push({ label: disease, disease, timepoint: tp, contrast: `${disease}_${tp}` });
    }
  }
  return panels.filter(p => block.contrasts.indexOf(p.contrast) >= 0);
}

function _ihAxisLimitN(f) {
  const raw = String((f && f.hmAxisLimit) || "all");
  if (raw === "all") return null;
  const n = parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : null;
}

function _ihTopIndices(scores, limit) {
  const idx = scores.map((score, i) => ({ i, score }));
  idx.sort((a, b) => (b.score - a.score) || (a.i - b.i));
  return idx.slice(0, Math.min(limit, idx.length)).map(x => x.i).sort((a, b) => a - b);
}

// Heatmap axis-grouping (heatmap-only). The payload ships a cluster → tissue
// category map; "tissue" mode collapses axes to ~7 groups, "celltype" keeps the
// native spine. Counts are additive across a merged axis (subtype-A→X and
// subtype-B→X are distinct paths), so a grouped cell = Σ over member index pairs.
function _ihGroupMap() {
  const block = _ihBlock();
  return (block && block.celltype_groups) || null;
}

function _ihGroupMode() {
  if (!_ihGroupMap()) return "celltype";           // no map → ungrouped, selector hidden
  const f = IncytrFilter.get();
  return f.hmGroupBy === "celltype" ? "celltype" : "tissue";
}

// Bucket the (sparse-filtered) axis indices into ordered groups. At celltype
// mode each index is its own singleton group, so the rest of the render path is
// uniform: every axis entry carries a member-index list (length 1 when ungrouped).
function _ihAxisMembers(allNames, keepIdx, grouped, groupOf, order) {
  if (!grouped) {
    return { labels: keepIdx.map(i => allNames[i]), members: keepIdx.map(i => [i]) };
  }
  const buckets = new Map();
  for (const i of keepIdx) {
    const g = groupOf[allNames[i]] || allNames[i];
    if (!buckets.has(g)) buckets.set(g, []);
    buckets.get(g).push(i);
  }
  const labels = (order || []).filter(g => buckets.has(g));
  for (const g of buckets.keys()) if (labels.indexOf(g) < 0) labels.push(g);
  return { labels, members: labels.map(g => buckets.get(g)) };
}

// A grouped axis entry is "empty" only when every member cell type is empty-DEG.
function _ihGroupEmpty(members, names, emptySet) {
  return members.length > 0 && members.every(i => emptySet.has(names[i]));
}

function _ihVisibleAxes(block, f, snap, snapAp, panels) {
  const allSenders = block.senders || [];
  const allReceivers = block.receivers || [];
  const dropLow = IncytrCelltypeQc.enabled(block);
  const low = dropLow ? IncytrCelltypeQc.lowSignalSet(block) : new Set();
  const allSenderIdx = allSenders
    .map((name, i) => ({ name, i }))
    .filter(x => !low.has(x.name))
    .map(x => x.i);
  const allReceiverIdx = allReceivers
    .map((name, i) => ({ name, i }))
    .filter(x => !low.has(x.name))
    .map(x => x.i);

  // Grouped (tissue) mode: collapse to canonical categories, all groups shown
  // (the top-N axis limiter is meaningless over ~7 groups and is hidden in the UI).
  if (_ihGroupMode() === "tissue") {
    const gm = _ihGroupMap();
    const groupOf = (gm && gm.tissue) || {};
    const order = (gm && gm.tissue_order) || [];
    const sAxis = _ihAxisMembers(allSenders, allSenderIdx, true, groupOf, order);
    const rAxis = _ihAxisMembers(allReceivers, allReceiverIdx, true, groupOf, order);
    return {
      senders: sAxis.labels, receivers: rAxis.labels,
      senderMembers: sAxis.members, receiverMembers: rAxis.members,
      limited: false, grouped: true,
    };
  }

  const limit = _ihAxisLimitN(f);
  if (!limit) {
    return {
      senders: allSenderIdx.map(i => allSenders[i]),
      receivers: allReceiverIdx.map(i => allReceivers[i]),
      senderMembers: allSenderIdx.map(i => [i]),
      receiverMembers: allReceiverIdx.map(i => [i]),
      limited: false, grouped: false,
    };
  }

  const senderScores = new Array(allSenders.length).fill(0);
  const receiverScores = new Array(allReceivers.length).fill(0);
  const contrasts = (panels && panels.length)
    ? panels.map(p => p.contrast)
    : [`${f.hmDisease}_${f.hmTimepoint}`];
  for (const contrast of contrasts) {
    const cIdx = block.contrasts.indexOf(contrast);
    if (cIdx < 0) continue;
    for (const s of allSenderIdx) {
      for (const r of allReceiverIdx) {
        const n = _ihCountAt(s, r, cIdx, snap.index, snapAp.index);
        senderScores[s] += n;
        receiverScores[r] += n;
      }
    }
  }
  const senderIdx = _ihTopIndices(senderScores, limit);
  const receiverIdx = _ihTopIndices(receiverScores, limit);
  return {
    senders: senderIdx.map(i => allSenders[i]),
    receivers: receiverIdx.map(i => allReceivers[i]),
    senderMembers: senderIdx.map(i => [i]),
    receiverMembers: receiverIdx.map(i => [i]),
    limited: true, grouped: false,
  };
}

function _ihSnapPvalue(p) {
  // Snap user input down to the nearest precomputed pvalue threshold. Null
  // (or non-finite) opens the gate by selecting the largest threshold in the
  // grid.
  const block = _ihBlock();
  const thr = block && block.heatmap_counts && block.heatmap_counts.thresholds;
  if (!thr || !thr.length) return { value: null, index: 0, open: true };
  if (p == null || !isFinite(p)) {
    return { value: null, index: thr.length - 1, open: true };
  }
  let idx = -1;
  for (let i = 0; i < thr.length; i++) if (thr[i] <= p) idx = i;
  if (idx < 0) idx = 0;
  return { value: thr[idx], index: idx, open: false };
}

function _ihSnapAbsPds(ap) {
  // Snap up to the nearest |PDS| threshold (≥ semantics — pick the largest
  // grid step that doesn't exceed the requested value, so the on-screen
  // gate is always at least as inclusive as what the user typed). Returns
  // { value:null, index:0 } on payloads that pre-date the |PDS| axis.
  const block = _ihBlock();
  const thr = block && block.heatmap_counts && block.heatmap_counts.abs_pds_thresholds;
  if (!thr || !thr.length) return { value: null, index: 0 };
  const v = (ap == null || !isFinite(ap)) ? 0 : ap;
  let idx = 0;
  for (let i = 0; i < thr.length; i++) if (thr[i] <= v) idx = i;
  return { value: thr[idx], index: idx };
}

function _ihCountAt(sIdx, rIdx, cIdx, tIdx, apIdx) {
  const block = _ihBlock();
  if (!block) return 0;
  const signIdx = _ihSignIndex();
  const signed = block.heatmap_counts_signed;
  if (signIdx != null && signed && signed.shape && signed.shape.length === 6 && signed.counts) {
    const [, nR, nC, nSign, nT, nAP] = signed.shape;
    if (signIdx >= nSign) return 0;
    return signed.counts[
      sIdx * nR * nC * nSign * nT * nAP
      + rIdx * nC * nSign * nT * nAP
      + cIdx * nSign * nT * nAP
      + signIdx * nT * nAP
      + tIdx * nAP
      + (apIdx || 0)
    ];
  }
  const hm = block.heatmap_counts;
  if (!hm || !hm.counts) return 0;
  // Back-compat: a 3D grid (no thresholds) → return the unfiltered count.
  if (!hm.shape || hm.shape.length === 3) {
    const nR = block.receivers.length, nC = block.contrasts.length;
    return hm.counts[sIdx * nR * nC + rIdx * nC + cIdx];
  }
  // Back-compat: a 4D grid (pvalue only, no |PDS| axis) → ignore apIdx.
  if (hm.shape.length === 4) {
    const [, nR, nC, nT] = hm.shape;
    return hm.counts[sIdx * nR * nC * nT + rIdx * nC * nT + cIdx * nT + tIdx];
  }
  // 5D grid (pvalue × |PDS|).
  const [, nR, nC, nT, nAP] = hm.shape;
  return hm.counts[
    sIdx * nR * nC * nT * nAP
    + rIdx * nC * nT * nAP
    + cIdx * nT * nAP
    + tIdx * nAP
    + (apIdx || 0)
  ];
}

function _ihSyncControls() {
  const block = _ihBlock();
  if (!block) return;
  const diseases = _ihDiseases();
  const timepoints = _ihTimepoints();
  let f = IncytrFilter.get();
  let snapDisease = f.hmDisease;
  let snapTimepoint = f.hmTimepoint;
  if (diseases.length && diseases.indexOf(snapDisease) < 0) snapDisease = diseases[0];
  if (timepoints.length && timepoints.indexOf(snapTimepoint) < 0) snapTimepoint = timepoints[0];
  if (snapDisease !== f.hmDisease || snapTimepoint !== f.hmTimepoint) {
    IncytrFilter.set({hmDisease: snapDisease, hmTimepoint: snapTimepoint});
    f = IncytrFilter.get();
  }

  const dSel = document.getElementById("ih-disease");
  if (dSel) {
    dSel.innerHTML = diseases.map(d =>
      `<option value="${_escapeHtml(d)}">${_escapeHtml(d)}</option>`
    ).join("");
    dSel.value = f.hmDisease;
  }
  const tSel = document.getElementById("ih-timepoint");
  if (tSel) {
    tSel.innerHTML = timepoints.map(t =>
      `<option value="${_escapeHtml(t)}">${_escapeHtml(t)}</option>`
    ).join("");
    tSel.value = f.hmTimepoint;
  }
  const pInput = document.getElementById("ih-pvalue");
  if (pInput) {
    pInput.value = (f.hmPvalue == null) ? "" : f.hmPvalue;
    // Backbones carry no pvalue (the count is pvalue-invariant — every band is
    // identical), so hide the gate to avoid implying it filters. |PDS| stays.
    if (pInput.parentElement) {
      const isBb = !!(block._grain && block._grain !== "Full");
      pInput.parentElement.style.display = isBb ? "none" : "";
    }
  }
  const apInput = document.getElementById("ih-abs-pds");
  if (apInput) apInput.value = (f.hmAbsPds == null) ? "" : f.hmAbsPds;
  const signSel = document.getElementById("ih-pds-sign");
  if (signSel) signSel.value = _ihPdsSignMode();
  // Axis grouping: selector hidden when the payload ships no group map; the
  // top-N axis limiter is hidden under grouping (meaningless over ~7 groups).
  const grouped = _ihGroupMode() === "tissue";
  const groupSel = document.getElementById("ih-group");
  if (groupSel) {
    groupSel.value = _ihGroupMode();
    if (groupSel.parentElement)
      groupSel.parentElement.style.display = _ihGroupMap() ? "" : "none";
  }
  const axisSel = document.getElementById("ih-axis-limit");
  if (axisSel) {
    axisSel.value = _ihAxisLimitN(f) ? String(_ihAxisLimitN(f)) : "all";
    if (axisSel.parentElement) axisSel.parentElement.style.display = grouped ? "none" : "";
  }
  const scaleSel = document.getElementById("ih-scale");
  if (scaleSel) {
    scaleSel.value = _ihScaleMode();
    // log1p is a color transform — heatmap-only. The chord re-hides this in
    // _icSyncControls; restore it here so it reappears when the heatmap is shown.
    if (scaleSel.parentElement) scaleSel.parentElement.style.display = "";
  }
  const lowSel = document.getElementById("if-low-signal");
  if (lowSel) {
    const hasLow = IncytrCelltypeQc.hasLowSignal(block);
    lowSel.value = (f.excludeLowSignalCelltypes && hasLow) ? "exclude" : "include";
    if (lowSel.parentElement) lowSel.parentElement.style.display = hasLow ? "" : "none";
  }

  // Contrasts are rendered as side-by-side small multiples (the same panels
  // _ihTimelinePanels enumerates), so there is no per-contrast picker. The
  // disease selector is shown only when it actually narrows the panel set —
  // i.e. both dimensions vary, so it chooses which disease's timepoints to lay
  // out. The timepoint selector is never needed (timepoints are the panels).
  if (dSel && dSel.parentElement)
    dSel.parentElement.style.display =
      (diseases.length > 1 && timepoints.length > 1) ? "" : "none";
  if (tSel && tSel.parentElement) tSel.parentElement.style.display = "none";
}

function wireIncytrHeatmap() {
  const dSel = document.getElementById("ih-disease");
  if (dSel) dSel.addEventListener("change", () => {
    IncytrFilter.set({ hmDisease: dSel.value });
    _ihRenderActive();
  });
  const tSel = document.getElementById("ih-timepoint");
  if (tSel) tSel.addEventListener("change", () => {
    IncytrFilter.set({ hmTimepoint: tSel.value });
    _ihRenderActive();
  });
  const pInput = document.getElementById("ih-pvalue");
  if (pInput) pInput.addEventListener("change", () => {
    if (pInput.value === "") {
      IncytrFilter.set({ hmPvalue: null });
      _ihRenderActive();
      return;
    }
    const raw = parseFloat(pInput.value);
    if (!isFinite(raw) || raw <= 0 || raw > 1) {
      const f = IncytrFilter.get();
      pInput.value = (f.hmPvalue == null) ? "" : f.hmPvalue;
      return;
    }
    IncytrFilter.set({ hmPvalue: raw });
    _ihRenderActive();
  });
  const apInput = document.getElementById("ih-abs-pds");
  if (apInput) apInput.addEventListener("change", () => {
    const raw = apInput.value === "" ? null : parseFloat(apInput.value);
    if (raw == null || !isFinite(raw) || raw < 0) {
      const f = IncytrFilter.get();
      apInput.value = f.hmAbsPds;
      return;
    }
    IncytrFilter.set({ hmAbsPds: raw });
    _ihRenderActive();
  });
  const signSel = document.getElementById("ih-pds-sign");
  if (signSel) signSel.addEventListener("change", () => {
    const v = (signSel.value === "up" || signSel.value === "down") ? signSel.value : "both";
    IncytrFilter.set({ hmPdsSign: v });
    _ihRenderActive();
  });
  const axisSel = document.getElementById("ih-axis-limit");
  if (axisSel) axisSel.addEventListener("change", () => {
    IncytrFilter.set({ hmAxisLimit: axisSel.value || "all" });
    _ihRenderActive();
  });
  const scaleSel = document.getElementById("ih-scale");
  if (scaleSel) scaleSel.addEventListener("change", () => {
    IncytrFilter.set({ hmScale: scaleSel.value === "log1p" ? "log1p" : "linear" });
    _ihRenderActive();
  });
  const groupSel = document.getElementById("ih-group");
  if (groupSel) groupSel.addEventListener("change", () => {
    IncytrFilter.set({ hmGroupBy: groupSel.value === "celltype" ? "celltype" : "tissue" });
    _ihRefreshActive();
  });
  // if-low-signal is wired once in wireIncytrPanel() to avoid double-wiring.
  const resetBtn = document.getElementById("ih-reset");
  if (resetBtn) resetBtn.addEventListener("click", () => {
    const ds = _ihDiseases(), ts = _ihTimepoints();
    IncytrFilter.set({
      hmDisease: ds[0] || null, hmTimepoint: ts[0] || null,
      hmPvalue: null, hmAbsPds: 0.01,
      hmAxisLimit: "all", hmScale: "linear", hmPdsSign: "both",
      hmGroupBy: "tissue",
      excludeLowSignalCelltypes: false,
    });
    _ihRefreshActive();
  });
}

const _IH_COLORSCALE = [
  [0.00, "#ffffff"],
  [0.05, "#f0f4ff"],
  [0.20, "#c3cdf0"],
  [0.45, "#8b9add"],
  [0.70, "#5563b8"],
  [1.00, "#1f2960"],
];

function _ihLayoutParts(senders, receivers, topMargin) {
  const longest = Math.max(...senders.map(s => s.length), ...receivers.map(r => r.length));
  const leftMargin = Math.min(360, Math.max(140, longest * 6 + 30));
  const bottomMargin = Math.min(220, Math.max(100, longest * 4 + 40));
  const nRows = Math.max(1, receivers.length);
  const rowPx = nRows > 25 ? 28 : 32;
  const height = Math.max(520, rowPx * nRows + topMargin + bottomMargin + 48);
  return { leftMargin, bottomMargin, height };
}

function _ihGateText(snap, snapAp) {
  const pTxt = snap.open ? "no pvalue gate" : `pvalue < ${snap.value}`;
  const apTxt = (snapAp.value != null && snapAp.value > 0)
    ? ` · |PDS| ≥ ${snapAp.value}` : "";
  return { pTxt, apTxt };
}

function _ihTotalAtThreshold(block, snap, snapAp) {
  const signIdx = _ihSignIndex();
  if (!IncytrCelltypeQc.enabled(block) && signIdx != null) {
    const totals = block.heatmap_counts_signed && block.heatmap_counts_signed.total_by_sign_threshold;
    if (totals && totals[signIdx] && totals[signIdx][snap.index]) {
      return totals[signIdx][snap.index][snapAp.index] || 0;
    }
  }
  if (!IncytrCelltypeQc.enabled(block)) {
    const totalsByThr = (block.heatmap_counts && block.heatmap_counts.total_by_threshold) || null;
    if (totalsByThr && totalsByThr.length) {
      const row = totalsByThr[snap.index];
      return Array.isArray(row) ? (row[snapAp.index] || 0) : (row || 0);
    }
    return 0;
  }
  const low = IncytrCelltypeQc.lowSignalSet(block);
  let total = 0;
  for (let s = 0; s < (block.senders || []).length; s++) {
    if (low.has(block.senders[s])) continue;
    for (let r = 0; r < (block.receivers || []).length; r++) {
      if (low.has(block.receivers[r])) continue;
      for (let c = 0; c < (block.contrasts || []).length; c++) {
        total += _ihCountAt(s, r, c, snap.index, snapAp.index);
      }
    }
  }
  return total;
}

function _ihSeedPathwayFilters(sender, receiver, disease, timepoint, snap, snapAp) {
  IncytrFilter.set({
    ipMode:     "pair",   // a heatmap-cell click loads that pair's shard — switch to Cell Type.
    pair:       { sender, receiver },
    senderIn:   [sender],
    receiverIn: [receiver],
    disease:    disease ? [disease] : [],
    timepoint:  timepoint ? [timepoint] : [],
    sliderP:    snap.open ? null : snap.value,
    sliderPds:  (snapAp.value != null && snapAp.value > 0) ? snapAp.value : null,
  });
  // Switch the unified Incytr pane to the table in-place (no tab jump).
  _setIncytrPane("table");
}

// Grouped (tissue) cell → the table can't pin a single (sender,receiver) shard,
// so seed the sender/receiver multiselects with every member cluster and load
// the ranked Top view filtered to them, rather than pair mode.
function _ihSeedGroupedFilters(senderNames, receiverNames, disease, timepoint, snap, snapAp) {
  IncytrFilter.set({
    ipMode:     "top",
    pair:       null,
    senderIn:   senderNames,
    receiverIn: receiverNames,
    disease:    disease ? [disease] : [],
    timepoint:  timepoint ? [timepoint] : [],
    sliderP:    snap.open ? null : snap.value,
    sliderPds:  (snapAp.value != null && snapAp.value > 0) ? snapAp.value : null,
  });
  _setIncytrPane("table");
}

// Dispatch a heatmap-cell click. Grouped axes seed the table's multiselects with
// the group's member clusters (Top mode); ungrouped axes pin the single pair.
function _ihHandleCellClick(axes, block, empty, sLabel, rLabel, disease, timepoint, snap, snapAp) {
  const si = axes.senders.indexOf(sLabel);
  const ri = axes.receivers.indexOf(rLabel);
  if (si < 0 || ri < 0) return;
  if (_ihGroupEmpty(axes.senderMembers[si], block.senders, empty)
      || _ihGroupEmpty(axes.receiverMembers[ri], block.receivers, empty)) return;
  if (axes.grouped) {
    _ihSeedGroupedFilters(
      axes.senderMembers[si].map(i => block.senders[i]),
      axes.receiverMembers[ri].map(i => block.receivers[i]),
      disease, timepoint, snap, snapAp);
  } else {
    _ihSeedPathwayFilters(sLabel, rLabel, disease, timepoint, snap, snapAp);
  }
}

function _ihQcRows(block) {
  const qc = block && block.celltype_pathway_qc;
  const rows = qc && qc.rows;
  if (!Array.isArray(rows)) return [];
  return rows.filter(r => r && r.median_n != null && isFinite(Number(r.median_n)));
}

function _ihRenderQcPlot(block) {
  const wrap = document.getElementById("ih-qc-wrap");
  const el = document.getElementById("ih-qc-plot");
  if (!wrap || !el || !window.Plotly) return;
  const rows = _ihQcRows(block);
  if (!rows.length) {
    wrap.style.display = "none";
    try { Plotly.purge(el); } catch(e) {}
    return;
  }
  wrap.style.display = "";

  const lowRows = rows.filter(r => !!r.low_signal_median_le_3);
  const mainRows = rows.filter(r => !r.low_signal_median_le_3);
  const yMetric = (block.celltype_pathway_qc && block.celltype_pathway_qc.y_metric)
    || "receiver_paths_abs_pds_gt1";
  const gate = (block.celltype_pathway_qc && block.celltype_pathway_qc.pds_gate)
    || "abs(PDS) > 1";
  const threshold = (block.celltype_qc && block.celltype_qc.low_signal_median_n_threshold) || 3;
  const topLabelRows = rows
    .slice()
    .sort((a, b) => (Number(b[yMetric]) || 0) - (Number(a[yMetric]) || 0))
    .slice(0, 5);
  const labelSet = new Set([
    ...lowRows.map(r => r.cell_type),
    ...topLabelRows.map(r => r.cell_type),
  ]);

  function traceFor(part, name, color, symbol) {
    return {
      type: "scatter",
      mode: "markers+text",
      name,
      x: part.map(r => Number(r.median_n)),
      y: part.map(r => (Number(r[yMetric]) || 0) + 1),
      text: part.map(r => labelSet.has(r.cell_type) ? r.cell_type : ""),
      textposition: "top right",
      customdata: part.map(r => [
        r.cell_type,
        Number(r[yMetric]) || 0,
        Number(r.sender_paths_abs_pds_gt1) || 0,
        Number(r.endpoint_paths_abs_pds_gt1) || 0,
        r.n_timepoints == null ? "" : r.n_timepoints,
      ]),
      hovertemplate:
        "%{customdata[0]}<br>"
        + "median n=%{x}<br>"
        + "receiver paths=" + "%{customdata[1]:,}<br>"
        + "sender paths=" + "%{customdata[2]:,}<br>"
        + "endpoint paths=" + "%{customdata[3]:,}<br>"
        + "timepoints=%{customdata[4]}<extra></extra>",
      marker: {
        color,
        symbol,
        size: 9,
        opacity: 0.86,
        line: { color: "#263238", width: 0.6 },
      },
      textfont: { size: 10, color: "#333" },
    };
  }

  const traces = [
    traceFor(mainRows, "median n > 3", "#2f5f9f", "circle"),
  ];
  if (lowRows.length) traces.push(
    traceFor(lowRows, "median n <= 3", "#b42318", "diamond")
  );

  const xVals = rows.map(r => Number(r.median_n)).filter(Number.isFinite);
  const maxY = Math.max(...rows.map(r => (Number(r[yMetric]) || 0) + 1), 1);
  const shapes = Number.isFinite(threshold) ? [{
    type: "line",
    x0: threshold,
    x1: threshold,
    y0: 1,
    y1: maxY,
    yref: "y",
    line: { color: "#b42318", width: 1, dash: "dash" },
  }] : [];

  const layout = {
    title: {
      text: `Cell count QC vs receiver-pathway burden (${gate})`,
      font: { size: 13 },
      x: 0,
    },
    margin: { l: 72, r: 24, t: 34, b: 54 },
    height: 360,
    xaxis: {
      title: "Median n_cells",
      range: [Math.min(0, ...xVals) - 2, Math.max(...xVals) + 20],
      zeroline: false,
      fixedrange: false,
    },
    yaxis: {
      title: "Receiver pathways + 1",
      type: "log",
      fixedrange: false,
    },
    legend: { orientation: "h", x: 0, y: 1.12 },
    shapes,
    plot_bgcolor: "#fafafa",
    paper_bgcolor: "#ffffff",
  };
  Plotly.react(el, traces, layout, { displaylogo: false, responsive: true });
}

// One contrast's count matrix over the shared axes (Z + hover text), the
// empty-cell overlay coordinates, and the panel's max/total. Z is the
// color-scaled value; maxN/visibleN are raw counts.
function _ihPanelData(block, axes, empty, cIdx, snap, snapAp) {
  const senders = axes.senders, receivers = axes.receivers;
  const nS = senders.length, nR = receivers.length;
  const noun = _ihEntityNoun();
  const gate = _ihGateText(snap, snapAp);
  const Z = [], text = [];
  const hx = [], hy = [], htxt = [];
  let maxN = 0, visibleN = 0;
  for (let r = 0; r < nR; r++) {
    const row = [], tRow = [];
    const memR = axes.receiverMembers[r];
    for (let s = 0; s < nS; s++) {
      const memS = axes.senderMembers[s];
      const isEmpty = _ihGroupEmpty(memS, block.senders, empty)
        || _ihGroupEmpty(memR, block.receivers, empty);
      if (isEmpty || cIdx < 0 || !memS.length || !memR.length) {
        row.push(null);
        tRow.push("no candidate paths");
        hx.push(senders[s]); hy.push(receivers[r]); htxt.push("no candidate paths");
      } else {
        let n = 0;
        for (const a of memS) for (const b of memR)
          n += _ihCountAt(a, b, cIdx, snap.index, snapAp.index);
        row.push(_ihScaleValue(n));
        if (n > maxN) maxN = n;
        visibleN += n;
        tRow.push(`${n.toLocaleString()} ${noun} · ${_ihSignText()} · ${gate.pTxt}${gate.apTxt}`);
      }
    }
    Z.push(row); text.push(tRow);
  }
  return { Z, text, hx, hy, htxt, maxN, visibleN };
}

// Render one small-multiple heatmap into `plotDiv`. zmax is shared across panels
// so cell colors are comparable; the leftmost panel keeps the receiver labels,
// the rightmost carries the shared colorbar.
function _ihRenderHeatmapPanel(plotDiv, axes, data, opts) {
  const senders = axes.senders, receivers = axes.receivers;
  const nS = senders.length, nR = receivers.length;
  const traces = [{
    type: "heatmap",
    x: senders, y: receivers, z: data.Z,
    text: data.text, hoverinfo: "x+y+text",
    colorscale: _IH_COLORSCALE,
    zmin: 0, zmax: opts.zmax,
    showscale: !!opts.showColorbar,
    colorbar: opts.showColorbar ? { title: _ihColorbarTitle(), thickness: 12, len: 0.7 } : undefined,
    xgap: 1, ygap: 1,
  }];
  if (data.hx.length) {
    traces.push({
      type: "scatter", mode: "markers",
      x: data.hx, y: data.hy, text: data.htxt, hoverinfo: "x+y+text",
      marker: { symbol: "x-thin-open", color: "#9aa0a6", size: 8, line: { width: 1.2 } },
      showlegend: false,
    });
  }
  // automargin: false + explicit category tickvals/range — see the long note in
  // git history; Plotly 2.35 otherwise post-thins category labels every-other.
  const layout = {
    margin: { l: opts.leftMargin, r: opts.showColorbar ? 60 : 12, t: 16, b: opts.bottomMargin },
    height: opts.height,
    xaxis: { title: "Sender", type: "category", tickangle: -45, automargin: false,
             categoryorder: "array", categoryarray: senders,
             tickmode: "array", tickvals: senders, ticktext: senders,
             range: [-0.5, nS - 0.5], fixedrange: true, tickfont: { size: 11 } },
    yaxis: { title: opts.showY ? "Receiver" : "", type: "category", automargin: false,
             categoryorder: "array", categoryarray: receivers,
             tickmode: "array", tickvals: receivers, ticktext: receivers,
             showticklabels: !!opts.showY,
             range: [nR - 0.5, -0.5], fixedrange: true, tickfont: { size: 11 } },
    plot_bgcolor: "#fafafa",
  };
  Plotly.react(plotDiv, traces, layout, { displaylogo: false, responsive: true });
  plotDiv.removeAllListeners && plotDiv.removeAllListeners("plotly_click");
  plotDiv.on && plotDiv.on("plotly_click", ev => {
    if (!ev.points || !ev.points.length) return;
    const p = ev.points[0];
    opts.onClick(p.x, p.y);
  });
}

function _ihRenderPlot() {
  const block = _ihBlock();
  const el = document.getElementById("ih-plot");
  const countEl = document.getElementById("ih-count");
  if (!block || !el) return;
  const f = IncytrFilter.get();
  const snap = _ihSnapPvalue(f.hmPvalue);
  const snapAp = _ihSnapAbsPds(f.hmAbsPds);

  // Contrasts are rendered as side-by-side small multiples (the same panels the
  // chord uses). Falls back to the single active contrast for a 1-contrast cohort.
  let panels = _ihTimelinePanels(f);
  if (!panels.length) {
    panels = [{
      label: _ihContrastFromState(), contrast: _ihContrastFromState(),
      disease: f.hmDisease, timepoint: f.hmTimepoint,
    }];
  }

  const axes = _ihVisibleAxes(block, f, snap, snapAp, panels);
  const senders = axes.senders, receivers = axes.receivers;
  const empty = new Set(block.empty_deg_celltypes || []);
  const noun = _ihEntityNoun();
  const totalAtThr = _ihTotalAtThreshold(block, snap, snapAp);

  // Build every panel's matrix first so the color scale (zmax) is shared and the
  // per-timepoint cell colors are directly comparable across panels.
  const datas = panels.map(p =>
    _ihPanelData(block, axes, empty, block.contrasts.indexOf(p.contrast), snap, snapAp));
  const globalMax = Math.max(1, ...datas.map(d => d.maxN));
  const zmax = Math.max(1, _ihScaleValue(globalMax));

  const lp = _ihLayoutParts(senders, receivers, 16);
  const firstLeft = lp.leftMargin;   // wide enough for the receiver labels
  const restLeft = 40;               // later panels drop the y labels
  const cellPx = 34;

  // One horizontal row of panels; scroll if wider than the pane.
  el.innerHTML = "";
  el.style.display = "flex";
  el.style.flexWrap = "nowrap";
  el.style.alignItems = "flex-start";
  el.style.overflowX = "auto";

  panels.forEach((panel, pi) => {
    const showY = pi === 0;
    const showColorbar = pi === panels.length - 1;
    const leftMargin = showY ? firstLeft : restLeft;
    const w = leftMargin + senders.length * cellPx + (showColorbar ? 60 : 12);

    const wrap = document.createElement("div");
    wrap.style.cssText = "display:inline-flex;flex-direction:column;flex:0 0 auto;";
    const lbl = document.createElement("div");
    lbl.style.cssText = "font-size:12px;font-weight:600;color:#333;text-align:center;margin:2px 0 1px;";
    lbl.textContent = panel.contrast;
    const sub = document.createElement("div");
    sub.className = "muted";
    sub.style.cssText = "font-size:10px;text-align:center;margin-bottom:2px;";
    sub.textContent = `${datas[pi].visibleN.toLocaleString()} ${noun}`;
    const plotDiv = document.createElement("div");
    plotDiv.style.cssText = `width:${w}px;height:${lp.height}px;`;
    wrap.appendChild(lbl); wrap.appendChild(sub); wrap.appendChild(plotDiv);
    el.appendChild(wrap);

    _ihRenderHeatmapPanel(plotDiv, axes, datas[pi], {
      zmax, showY, showColorbar, height: lp.height,
      leftMargin, bottomMargin: lp.bottomMargin,
      onClick: (x, y) =>
        _ihHandleCellClick(axes, block, empty, x, y, panel.disease, panel.timepoint, snap, snapAp),
    });
  });

  if (countEl) {
    const tps = _ihTimepoints();
    const header = tps.length > 1 ? f.hmDisease : (f.hmTimepoint || "all");
    const gate = _ihGateText(snap, snapAp);
    const axesTxt = axes.grouped ? ` · grouped to ${senders.length}×${receivers.length} tissue axes`
      : (axes.limited ? ` · showing top ${senders.length}×${receivers.length} axes` : "");
    const scaleTxt = _ihScaleMode() === "log1p" ? " · log1p color" : "";
    const lowTxt = IncytrCelltypeQc.enabled(block)
      ? ` · ${IncytrCelltypeQc.controlText(block)}` : "";
    const perPanel = panels.map((p, i) => `${p.label}: ${datas[i].visibleN.toLocaleString()}`);
    const lead = panels.length > 1
      ? `${header} · ${panels.length} contrasts side by side (${perPanel.join(" · ")})`
      : `${panels[0].contrast} · ${datas[0].visibleN.toLocaleString()} ${noun}`;
    countEl.textContent = `${lead} at ${gate.pTxt}${gate.apTxt.replace(" · ", " & ")}`
      + ` · ${totalAtThr.toLocaleString()} across all contrasts at this threshold`
      + `${axesTxt}${scaleTxt} · ${_ihSignText()}${lowTxt}.`;
  }
  _ihRenderQcPlot(block);
}

function renderIncytrHeatmap() {
  if (!_ihBlock()) {
    const el = document.getElementById("ih-plot");
    _ihRenderQcPlot(null);
    if (el) el.innerHTML =
      '<div class="muted" style="padding:24px;">No <code>incytr_pathways</code> '
      + 'block in the payload. Rebuild with <code>pixi run viewer</code> after '
      + 'producing <code>outputs/reports/incytr_pair_mode/receiver_cache/</code>.</div>';
    return;
  }
  _ihSyncControls();
  _ihRenderPlot();
}

// ---------------------------------------------------------------------------
// Unified Incytr pane management (J-1)
// The merged "incytr" tab shows either the heatmap or the table in its main
// pane. _incytrPane tracks which view is active. _setIncytrPane switches,
// syncs the DOM, and triggers the appropriate re-render. wireIncytrPanel wires
// the view-switch buttons and the shared if-low-signal control.
// ---------------------------------------------------------------------------

let _incytrPane = "table"; // "table" | "heatmap" | "chord"

// The heatmap and chord panes share the same sidebar controls (incytr-hm-controls)
// and the same count-tensor data layer, differing only in render. These dispatch
// a sync/render to whichever of the two is active so the shared control handlers
// don't need to know which view they're driving.
function _ihSyncActive() {
  if (_incytrPane === "chord") _icSyncControls();
  else _ihSyncControls();
}
function _ihRenderActive() {
  if (_incytrPane === "chord") _icRenderChord();
  else _ihRenderPlot();
}
// Heatmap and chord are the two "vis" panes (table is the third); they share the
// controls + data layer, so most handlers refresh whichever is active.
function _ihVisPaneActive() { return _incytrPane !== "table"; }
function _ihRefreshActive() { _ihSyncActive(); _ihRenderActive(); }

function _syncIncytrPane() {
  const isHeatmap = _incytrPane === "heatmap";
  const isChord = _incytrPane === "chord";
  const isTable = !isHeatmap && !isChord;
  const hp = document.getElementById("incytr-pane-heatmap");
  const cp = document.getElementById("incytr-pane-chord");
  const tp = document.getElementById("incytr-pane-table");
  const hc = document.getElementById("incytr-hm-controls");
  const ic = document.getElementById("incytr-ip-controls");
  const tb = document.getElementById("incytr-view-table");
  const hb = document.getElementById("incytr-view-heatmap");
  const cb = document.getElementById("incytr-view-chord");
  if (hp) hp.hidden = !isHeatmap;
  if (cp) cp.hidden = !isChord;
  if (tp) tp.hidden = !isTable;
  // The heatmap controls drive both the heatmap and the chord.
  if (hc) hc.hidden = isTable;
  if (ic) ic.hidden = !isTable;
  if (tb) tb.classList.toggle("active", isTable);
  if (hb) hb.classList.toggle("active", isHeatmap);
  if (cb) cb.classList.toggle("active", isChord);
}

function _setIncytrPane(pane) {
  _incytrPane = (pane === "heatmap" || pane === "chord") ? pane : "table";
  _syncIncytrPane();
  if (_ihVisPaneActive()) {
    _ihRefreshActive();
  } else {
    // renderIncytrPathways is defined in incytr_pathways.js — loaded in same scope.
    if (typeof renderIncytrPathways === "function") renderIncytrPathways();
  }
}

function wireIncytrPanel() {
  const tableBtn = document.getElementById("incytr-view-table");
  if (tableBtn) tableBtn.addEventListener("click", () => _setIncytrPane("table"));
  const heatmapBtn = document.getElementById("incytr-view-heatmap");
  if (heatmapBtn) heatmapBtn.addEventListener("click", () => _setIncytrPane("heatmap"));
  const chordBtn = document.getElementById("incytr-view-chord");
  if (chordBtn) chordBtn.addEventListener("click", () => _setIncytrPane("chord"));

  // Unified sparse-cell control: wired once here so all views share it without
  // double-event firing. Handler behaviour differs per active pane.
  const lowSel = document.getElementById("if-low-signal");
  if (lowSel) lowSel.addEventListener("change", () => {
    const excluded = lowSel.value === "exclude";
    if (_ihVisPaneActive()) {
      IncytrFilter.set({ excludeLowSignalCelltypes: excluded });
      _ihRefreshActive();
    } else {
      IncytrFilter.set({ excludeLowSignalCelltypes: excluded, pair: null });
      if (typeof _ipBlock === "function") {
        const block = _ipBlock();
        if (block && typeof _ipSyncControls === "function") _ipSyncControls(block);
      }
      if (typeof _ipInvalidateScope === "function") _ipInvalidateScope();
      if (typeof _ipResetPage === "function") _ipResetPage();
      if (typeof _ipEnsureShards === "function") _ipEnsureShards();
    }
  });
}

function _renderIncytrTab() {
  // Re-evaluate the 5xFAD cortex/hippocampus tissue control on every Incytr
  // render so it tracks the active context: it must hide under Song (no tissue
  // split) and only show for a 5xFAD incytr context. Defined in the unified
  // viewer's chrome; absent (and irrelevant) in the t-cell viewer.
  if (typeof _syncFivexfadTissueToggle === "function") _syncFivexfadTissueToggle();
  _syncIncytrPane();
  renderIncytrHeatmap();
  if (typeof renderIncytrPathways === "function") renderIncytrPathways();
}
