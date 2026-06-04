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
  return ViewerPayload.incytr();
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

function _ihViewMode() {
  return "timeline";
}

function _ihScaleMode() {
  const f = IncytrFilter.get();
  return f.hmScale === "log1p" ? "log1p" : "linear";
}

function _ihScaleValue(n) {
  return _ihScaleMode() === "log1p" ? Math.log1p(n || 0) : n;
}

function _ihColorbarTitle() {
  return _ihScaleMode() === "log1p" ? "log1p(n paths)" : "n paths";
}

function _ihPdsSignMode() {
  const f = IncytrFilter.get();
  return f.hmPdsSign === "positive" || f.hmPdsSign === "negative" ? f.hmPdsSign : "both";
}

function _ihSignIndex() {
  const mode = _ihPdsSignMode();
  if (mode === "positive") return 2;
  if (mode === "negative") return 0;
  return null;
}

function _ihSignText() {
  const mode = _ihPdsSignMode();
  if (mode === "positive") return "positive PDS";
  if (mode === "negative") return "negative PDS";
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

function _ihTimelineIndex(f, panels) {
  const raw = parseInt((f && f.hmTimelineIndex) || 0, 10);
  if (!panels || !panels.length) return 0;
  if (!Number.isFinite(raw)) return 0;
  return Math.max(0, Math.min(raw, panels.length - 1));
}

function _ihTimelinePatch(panel, idx) {
  return {
    hmTimelineIndex: idx,
    hmDisease: panel.disease,
    hmTimepoint: panel.timepoint,
  };
}

function _ihSetTimelineIndex(idx) {
  const panels = _ihTimelinePanels(IncytrFilter.get());
  if (panels.length <= 1) return;
  const clamped = Math.max(0, Math.min(idx, panels.length - 1));
  IncytrFilter.set(_ihTimelinePatch(panels[clamped], clamped));
  _ihSyncControls();
  _ihRenderPlot();
}

function _ihSyncTimelineControl(panels, mode) {
  const wrap = document.getElementById("ih-timeline-control");
  if (!wrap) return;
  const slider = document.getElementById("ih-timeline-slider");
  const label = document.getElementById("ih-timeline-label");
  const ticks = document.getElementById("ih-timeline-ticks");
  const prev = document.getElementById("ih-timeline-prev");
  const next = document.getElementById("ih-timeline-next");
  const show = mode === "timeline" && panels.length > 1;
  wrap.style.display = show ? "flex" : "none";
  if (!show || !slider) return;

  const f = IncytrFilter.get();
  const idx = _ihTimelineIndex(f, panels);
  slider.min = "0";
  slider.max = String(panels.length - 1);
  slider.step = "1";
  slider.value = String(idx);
  if (label) label.textContent = panels[idx].contrast;
  if (prev) prev.disabled = idx <= 0;
  if (next) next.disabled = idx >= panels.length - 1;
  if (ticks) {
    ticks.innerHTML = panels.map((p, i) =>
      `<span class="${i === idx ? "active" : ""}" style="flex:1;text-align:${i === 0 ? "left" : (i === panels.length - 1 ? "right" : "center")};">`
      + `${_escapeHtml(p.label)}</span>`
    ).join("");
  }
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

function _ihVisibleAxes(block, f, snap, snapAp, panels) {
  const allSenders = block.senders || [];
  const allReceivers = block.receivers || [];
  const limit = _ihAxisLimitN(f);
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
  if (!limit) {
    return {
      senderIdx: allSenderIdx,
      receiverIdx: allReceiverIdx,
      senders: allSenderIdx.map(i => allSenders[i]),
      receivers: allReceiverIdx.map(i => allReceivers[i]),
      limited: false,
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
    senderIdx,
    receiverIdx,
    senders: senderIdx.map(i => allSenders[i]),
    receivers: receiverIdx.map(i => allReceivers[i]),
    limited: true,
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
    if (tSel.parentElement) tSel.parentElement.style.display = timepoints.length > 1 ? "" : "none";
  }
  const pInput = document.getElementById("ih-pvalue");
  if (pInput) pInput.value = (f.hmPvalue == null) ? "" : f.hmPvalue;
  const apInput = document.getElementById("ih-abs-pds");
  if (apInput) apInput.value = (f.hmAbsPds == null) ? "" : f.hmAbsPds;
  const signSel = document.getElementById("ih-pds-sign");
  if (signSel) signSel.value = _ihPdsSignMode();
  const axisSel = document.getElementById("ih-axis-limit");
  if (axisSel) axisSel.value = _ihAxisLimitN(f) ? String(_ihAxisLimitN(f)) : "all";
  const scaleSel = document.getElementById("ih-scale");
  if (scaleSel) scaleSel.value = _ihScaleMode();
  const lowSel = document.getElementById("ih-low-signal");
  if (lowSel) {
    const hasLow = IncytrCelltypeQc.hasLowSignal(block);
    lowSel.value = (f.excludeLowSignalCelltypes && hasLow) ? "exclude" : "include";
    if (lowSel.parentElement) lowSel.parentElement.style.display = hasLow ? "" : "none";
  }

  const panels = _ihTimelinePanels(f);
  let mode = _ihViewMode();
  if (mode === "timeline" && panels.length <= 1) mode = "single";
  const idx = _ihTimelineIndex(f, panels);
  if (f.hmTimelineIndex !== idx) {
    IncytrFilter.set({ hmTimelineIndex: idx });
    f = IncytrFilter.get();
  }
  if (dSel && dSel.parentElement) {
    dSel.parentElement.style.display =
      (mode === "timeline" && timepoints.length <= 1 && diseases.length > 1) ? "none" : "";
  }
  if (tSel && tSel.parentElement) {
    if (mode === "timeline") {
      tSel.parentElement.style.display = "none";
    } else {
      tSel.parentElement.style.display = timepoints.length > 1 ? "" : "none";
    }
  }
  _ihSyncTimelineControl(panels, mode);
}

function wireIncytrHeatmap() {
  const dSel = document.getElementById("ih-disease");
  if (dSel) dSel.addEventListener("change", () => {
    IncytrFilter.set({ hmDisease: dSel.value });
    _ihRenderPlot();
  });
  const tSel = document.getElementById("ih-timepoint");
  if (tSel) tSel.addEventListener("change", () => {
    IncytrFilter.set({ hmTimepoint: tSel.value });
    _ihRenderPlot();
  });
  const pInput = document.getElementById("ih-pvalue");
  if (pInput) pInput.addEventListener("change", () => {
    if (pInput.value === "") {
      IncytrFilter.set({ hmPvalue: null });
      _ihRenderPlot();
      return;
    }
    const raw = parseFloat(pInput.value);
    if (!isFinite(raw) || raw <= 0 || raw > 1) {
      const f = IncytrFilter.get();
      pInput.value = (f.hmPvalue == null) ? "" : f.hmPvalue;
      return;
    }
    IncytrFilter.set({ hmPvalue: raw });
    _ihRenderPlot();
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
    _ihRenderPlot();
  });
  const signSel = document.getElementById("ih-pds-sign");
  if (signSel) signSel.addEventListener("change", () => {
    const v = signSel.value === "positive" || signSel.value === "negative" ? signSel.value : "both";
    IncytrFilter.set({ hmPdsSign: v });
    _ihRenderPlot();
  });
  const axisSel = document.getElementById("ih-axis-limit");
  if (axisSel) axisSel.addEventListener("change", () => {
    IncytrFilter.set({ hmAxisLimit: axisSel.value || "all" });
    _ihRenderPlot();
  });
  const scaleSel = document.getElementById("ih-scale");
  if (scaleSel) scaleSel.addEventListener("change", () => {
    IncytrFilter.set({ hmScale: scaleSel.value === "log1p" ? "log1p" : "linear" });
    _ihRenderPlot();
  });
  const lowSel = document.getElementById("ih-low-signal");
  if (lowSel) lowSel.addEventListener("change", () => {
    IncytrFilter.set({ excludeLowSignalCelltypes: lowSel.value === "exclude" });
    _ihSyncControls();
    _ihRenderPlot();
  });
  const timelineSlider = document.getElementById("ih-timeline-slider");
  if (timelineSlider) timelineSlider.addEventListener("input", () => {
    const raw = parseInt(timelineSlider.value || "0", 10);
    _ihSetTimelineIndex(Number.isFinite(raw) ? raw : 0);
  });
  const timelinePrev = document.getElementById("ih-timeline-prev");
  if (timelinePrev) timelinePrev.addEventListener("click", () => {
    _ihSetTimelineIndex(_ihTimelineIndex(IncytrFilter.get(), _ihTimelinePanels(IncytrFilter.get())) - 1);
  });
  const timelineNext = document.getElementById("ih-timeline-next");
  if (timelineNext) timelineNext.addEventListener("click", () => {
    _ihSetTimelineIndex(_ihTimelineIndex(IncytrFilter.get(), _ihTimelinePanels(IncytrFilter.get())) + 1);
  });
  const resetBtn = document.getElementById("ih-reset");
  if (resetBtn) resetBtn.addEventListener("click", () => {
    const ds = _ihDiseases(), ts = _ihTimepoints();
    IncytrFilter.set({
      hmDisease: ds[0] || null, hmTimepoint: ts[0] || null,
      hmView: "timeline",
      hmTimelineIndex: 0,
      hmPvalue: null, hmAbsPds: 0.01,
      hmAxisLimit: "all", hmScale: "linear", hmPdsSign: "both",
      excludeLowSignalCelltypes: false,
    });
    _ihSyncControls();
    _ihRenderPlot();
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
    pair:       { sender, receiver },
    senderIn:   [sender],
    receiverIn: [receiver],
    disease:    disease ? [disease] : [],
    timepoint:  timepoint ? [timepoint] : [],
    sliderP:    snap.open ? null : snap.value,
    sliderPds:  (snapAp.value != null && snapAp.value > 0) ? snapAp.value : null,
  });
  Store.dispatch({type:"SET_VIEW", key:"activeTab", value:"incytrpathways"});
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

function _ihRenderTimelinePlot(block, el, countEl, f, snap, snapAp, totalAtThr) {
  const panels = _ihTimelinePanels(f);
  if (panels.length <= 1) return false;
  const panelIdx = _ihTimelineIndex(f, panels);
  const panel = panels[panelIdx];
  const axes = _ihVisibleAxes(block, f, snap, snapAp, panels);
  const senders = axes.senders, receivers = axes.receivers;
  const nS = senders.length, nR = receivers.length;
  const empty = new Set(block.empty_deg_celltypes || []);
  el.style.overflowX = "";
  const cIdx = block.contrasts.indexOf(panel.contrast);

  const Z = [];
  const text = [];
  let maxN = 0;
  let visibleN = 0;
  for (let r = 0; r < nR; r++) {
    const row = [];
    const tRow = [];
    for (let s = 0; s < nS; s++) {
      const isEmpty = empty.has(senders[s]) || empty.has(receivers[r]);
      const senderIdx = axes.senderIdx[s];
      const receiverIdx = axes.receiverIdx[r];
      if (isEmpty || cIdx < 0 || senderIdx < 0 || receiverIdx < 0) {
        row.push(null);
        tRow.push("no candidate paths");
      } else {
        const n = _ihCountAt(senderIdx, receiverIdx, cIdx, snap.index, snapAp.index);
        row.push(_ihScaleValue(n));
        if (n > maxN) maxN = n;
        visibleN += n;
        const gate = _ihGateText(snap, snapAp);
        tRow.push(`${panel.contrast}: ${n.toLocaleString()} paths · ${_ihSignText()} · ${gate.pTxt}${gate.apTxt}`);
      }
    }
    Z.push(row);
    text.push(tRow);
  }

  const traces = [{
    type: "heatmap",
    x: senders, y: receivers, z: Z,
    text, hoverinfo: "x+y+text",
    colorscale: _IH_COLORSCALE,
    zmin: 0, zmax: Math.max(1, _ihScaleValue(maxN)),
    colorbar: { title: _ihColorbarTitle(), thickness: 12, len: 0.7 },
    xgap: 1, ygap: 1,
  }];
  const hx = [], hy = [], htxt = [];
  for (let r = 0; r < nR; r++) for (let s = 0; s < nS; s++) {
    if (empty.has(senders[s]) || empty.has(receivers[r])) {
      hx.push(senders[s]); hy.push(receivers[r]);
      htxt.push("no candidate paths");
    }
  }
  if (hx.length) {
    traces.push({
      type: "scatter", mode: "markers",
      x: hx, y: hy, text: htxt, hoverinfo: "x+y+text",
      marker: { symbol: "x-thin-open", color: "#9aa0a6", size: 8, line: { width: 1.2 } },
      showlegend: false,
    });
  }

  const layoutParts = _ihLayoutParts(senders, receivers, 30);

  const layout = {
    margin: { l: layoutParts.leftMargin, r: 30, t: 30, b: layoutParts.bottomMargin },
    height: layoutParts.height,
    xaxis: { title: "Sender", type: "category", tickangle: -45,
             automargin: false, categoryorder: "array", categoryarray: senders,
             tickmode: "array", tickvals: senders, ticktext: senders,
             range: [-0.5, nS - 0.5], fixedrange: true,
             tickfont: { size: 11 } },
    yaxis: { title: "Receiver", type: "category", automargin: false,
             categoryorder: "array", categoryarray: receivers,
             tickmode: "array", tickvals: receivers, ticktext: receivers,
             range: [nR - 0.5, -0.5], fixedrange: true,
             tickfont: { size: 11 } },
    plot_bgcolor: "#fafafa",
  };
  Plotly.react(el, traces, layout, { displaylogo: false, responsive: true });

  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on && el.on("plotly_click", ev => {
    if (!ev.points || !ev.points.length) return;
    const p = ev.points[0];
    const sender = p.x, receiver = p.y;
    if (empty.has(sender) || empty.has(receiver)) return;
    _ihSeedPathwayFilters(sender, receiver, panel.disease, panel.timepoint, snap, snapAp);
  });

  if (countEl) {
    const gate = _ihGateText(snap, snapAp);
    const axesTxt = axes.limited ? ` · showing top ${senders.length}×${receivers.length} axes` : "";
    const scaleTxt = _ihScaleMode() === "log1p" ? " · log1p color" : "";
    const signTxt = ` · ${_ihSignText()}`;
    const lowTxt = IncytrCelltypeQc.enabled(block)
      ? ` · ${IncytrCelltypeQc.controlText(block)}` : "";
    countEl.textContent =
      `Timeline · ${panel.contrast} (${panelIdx + 1}/${panels.length}) · `
      + `${visibleN.toLocaleString()} paths at ${gate.pTxt}${gate.apTxt}`
      + ` · ${totalAtThr.toLocaleString()} across all contrasts at this threshold.`
      + `${axesTxt}${scaleTxt}${signTxt}${lowTxt}`;
  }
  _ihRenderQcPlot(block);
  return true;
}

function _ihRenderPlot() {
  const block = _ihBlock();
  const el = document.getElementById("ih-plot");
  const countEl = document.getElementById("ih-count");
  if (!block || !el) return;
  const f = IncytrFilter.get();
  const contrast = _ihContrastFromState();
  const cIdx = block.contrasts.indexOf(contrast);
  const snap = _ihSnapPvalue(f.hmPvalue);
  const snapAp = _ihSnapAbsPds(f.hmAbsPds);
  const axes = _ihVisibleAxes(block, f, snap, snapAp, null);
  const senders = axes.senders, receivers = axes.receivers;
  const nS = senders.length, nR = receivers.length;
  const empty = new Set(block.empty_deg_celltypes || []);
  const totalAtThr = _ihTotalAtThreshold(block, snap, snapAp);

  if (_ihViewMode() === "timeline"
      && _ihRenderTimelinePlot(block, el, countEl, f, snap, snapAp, totalAtThr)) {
    return;
  }
  el.style.overflowX = "";

  const Z = [];
  const text = [];
  let maxN = 0;
  let visibleN = 0;
  for (let r = 0; r < nR; r++) {
    const row = [];
    const tRow = [];
    for (let s = 0; s < nS; s++) {
      const isEmpty = empty.has(senders[s]) || empty.has(receivers[r]);
      const senderIdx = axes.senderIdx[s];
      const receiverIdx = axes.receiverIdx[r];
      if (isEmpty || cIdx < 0) {
        row.push(null);
        tRow.push("no candidate paths (empty-DEG cell type)");
      } else {
        const n = _ihCountAt(senderIdx, receiverIdx, cIdx, snap.index, snapAp.index);
        row.push(_ihScaleValue(n));
        if (n > maxN) maxN = n;
        visibleN += n;
        const gate = _ihGateText(snap, snapAp);
        tRow.push(`${n.toLocaleString()} paths · ${_ihSignText()} · ${gate.pTxt}${gate.apTxt}`);
      }
    }
    Z.push(row);
    text.push(tRow);
  }

  const traces = [{
    type: "heatmap",
    x: senders, y: receivers, z: Z,
    text, hoverinfo: "x+y+text",
    colorscale: _IH_COLORSCALE,
    zmin: 0, zmax: Math.max(1, _ihScaleValue(maxN)),
    colorbar: { title: _ihColorbarTitle(), thickness: 12, len: 0.7 },
    xgap: 1, ygap: 1,
  }];
  const hx = [], hy = [], htxt = [];
  for (let r = 0; r < nR; r++) for (let s = 0; s < nS; s++) {
    if (empty.has(senders[s]) || empty.has(receivers[r])) {
      hx.push(senders[s]); hy.push(receivers[r]);
      htxt.push("no candidate paths");
    }
  }
  if (hx.length) {
    traces.push({
      type: "scatter", mode: "markers",
      x: hx, y: hy, text: htxt, hoverinfo: "x+y+text",
      marker: { symbol: "x-thin-open", color: "#9aa0a6", size: 10, line: { width: 1.5 } },
      showlegend: false,
    });
  }
  // Plotly auto-thins category labels to every-other when the per-row pixel
  // budget drops below ~font-line-height. With 19 cell types and the prior
  // 22 px/row, ~12 px font + descender/ascender + cell-padding overflowed and
  // Plotly silently dropped every other label. Fixes:
  //   - Explicit type: "category" so tick0/dtick are unambiguous category indices
  //   - 32 px/row floor for the heatmap canvas (was 22)
  //   - Larger left/bottom margins to seat the longest cell-type names
  //   - tickfont size 11 so labels fit without breaking the row pitch
  const layoutParts = _ihLayoutParts(senders, receivers, 20);
  const layout = {
    margin: { l: layoutParts.leftMargin, r: 30, t: 20, b: layoutParts.bottomMargin },
    height: layoutParts.height,
    // automargin: false because Plotly 2.35's automargin pass interacts
    // badly with category axes — even with tickmode:"array" + explicit
    // tickvals, it post-thins labels to nticks=10 stride=2 during the
    // remeasure. nticks: 100 also explicitly overrides the auto-nticks
    // fallback. tickvals as the category strings themselves (not numeric
    // indices) is the contract Plotly category axes honor in 2.35.
    // Lock the range to the category bounds. Plotly's autoscale otherwise
    // pads each end and the wider range triggers the auto-tick-thinning
    // (every-other label drops) regardless of tickmode/nticks. fixedrange
    // disables user zoom-out, which would re-introduce the same effect.
    xaxis: { title: "Sender", type: "category", tickangle: -45, automargin: false,
             categoryorder: "array", categoryarray: senders,
             tickmode: "array", tickvals: senders, ticktext: senders,
             range: [-0.5, nS - 0.5], fixedrange: true,
             tickfont: { size: 11 } },
    yaxis: { title: "Receiver", type: "category", automargin: false,
             categoryorder: "array", categoryarray: receivers,
             tickmode: "array", tickvals: receivers, ticktext: receivers,
             range: [nR - 0.5, -0.5], fixedrange: true,
             tickfont: { size: 11 } },
    plot_bgcolor: "#fafafa",
  };
  Plotly.react(el, traces, layout, { displaylogo: false, responsive: true });

  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on && el.on("plotly_click", ev => {
    if (!ev.points || !ev.points.length) return;
    const p = ev.points[0];
    const sender = p.x, receiver = p.y;
    if (empty.has(sender) || empty.has(receiver)) return;
    // Seed the table tab via shared filter state, then switch tabs.
    // Propagate |PDS| into the table's sliderPds so the pathway-table view
    // starts at the same effect-size gate the user was looking at.
    _ihSeedPathwayFilters(sender, receiver, f.hmDisease, f.hmTimepoint, snap, snapAp);
  });

  if (countEl) {
    const gate = _ihGateText(snap, snapAp);
    const axesTxt = axes.limited ? ` · showing top ${senders.length}×${receivers.length} axes` : "";
    const scaleTxt = _ihScaleMode() === "log1p" ? " · log1p color" : "";
    const signTxt = ` · ${_ihSignText()}`;
    const lowTxt = IncytrCelltypeQc.enabled(block)
      ? ` · ${IncytrCelltypeQc.controlText(block)}` : "";
    countEl.textContent =
      `${contrast} · ${visibleN.toLocaleString()} paths at ${gate.pTxt}${gate.apTxt.replace(" · ", " & ")}`
      + ` · ${totalAtThr.toLocaleString()} across all contrasts at this threshold.`
      + `${axesTxt}${scaleTxt}${signTxt}${lowTxt}`;
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
