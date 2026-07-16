// ---------------------------------------------------------------------------
// Incytr sender → receiver chord pane.
//
// A circular (Circos-style) directed chord diagram over the same count-tensor
// data layer the heatmap uses: nodes are the visible sender/receiver axes
// (cell types, or — by default — the ~7 tissue groups), and a directed ribbon
// from node A to node B is sized by the number of candidate paths A sends to B
// under a contrast and the active gates. It reuses _ihBlock / _ihVisibleAxes /
// _ihCountAt / grouping / snaps and the heatmap's drill-through; only the render
// differs. d3 v7 (window.d3, UMD) supplies the chord + ribbon layout.
//
// Where the heatmap scrubs ONE contrast at a time with a timeline slider, the
// chord renders the same timeline panels SIDE BY SIDE as small multiples — one
// chord per timepoint (or per disease) — sharing a node ring and color palette
// so the timepoints are directly comparable. Hover-isolate is linked across all
// panels by node.
// ---------------------------------------------------------------------------

function _icSyncControls() {
  // Populate the shared heatmap controls verbatim, then drop the one with no
  // chord analogue: log1p is a heatmap color transform. The disease selector
  // and side-by-side panel layout are shared with the heatmap.
  _ihSyncControls();
  const scaleSel = document.getElementById("ih-scale");
  if (scaleSel && scaleSel.parentElement) scaleSel.parentElement.style.display = "none";
}

function _icPalette(nodes) {
  const N = nodes.length;
  // Per-node fallback when a node has no cohort-supplied color: ≤10 nodes
  // (the grouped/tissue default) → categorical Tableau (maximally distinct);
  // more (full cell-type spine) → spread Turbo so every arc gets a unique hue.
  const fallback = (i) => (N <= 10 && window.d3 && d3.schemeTableau10)
    ? d3.schemeTableau10[i % 10]
    : d3.interpolateTurbo((i + 0.5) / N);
  // When the active cohort defines a categorical state palette (T-cell: CD4
  // oranges / CD8 blues, from the labeling-evidence report), honor it so the
  // chord reads the same orange/blue scheme as the report. typeof-guarded like
  // COHORT_LABELS — non-T-cell cohorts leave it undefined and fall through.
  const map = (typeof TCELL_STATE_COLOR !== "undefined") ? TCELL_STATE_COLOR : null;
  return nodes.map((n, i) => (map && map[n]) || fallback(i));
}

// Path-count matrix for a single contrast over the shared node ring (rows =
// sender node, cols = receiver node, value = Σ over the member index pairs of
// grouped axes). Returns the matrix and its total flow.
function _icBuildMatrix(block, empty, memberS, memberR, nodes, cIdx, snap, snapAp) {
  const N = nodes.length;
  const matrix = Array.from({ length: N }, () => new Array(N).fill(0));
  let total = 0;
  if (cIdx >= 0) {
    for (let i = 0; i < N; i++) {
      const memS = memberS.get(nodes[i]);
      if (!memS || _ihGroupEmpty(memS, block.senders, empty)) continue;
      for (let j = 0; j < N; j++) {
        const memR = memberR.get(nodes[j]);
        if (!memR || _ihGroupEmpty(memR, block.receivers, empty)) continue;
        let n = 0;
        for (const a of memS) for (const b of memR)
          n += _ihCountAt(a, b, cIdx, snap.index, snapAp.index);
        if (n > 0) { matrix[i][j] = n; total += n; }
      }
    }
  }
  return { matrix, total };
}

function _icRenderChord() {
  const el = document.getElementById("ic-plot");
  const countEl = document.getElementById("ic-count");
  if (!el) return;
  const block = _ihBlock();
  if (!block) {
    el.innerHTML = '<div class="muted" style="padding:24px;">No <code>incytr_pathways</code> '
      + 'block in the payload.</div>';
    if (countEl) countEl.textContent = "";
    return;
  }
  if (!window.d3 || !d3.chordDirected || !d3.ribbonArrow) {
    el.innerHTML = '<div class="muted" style="padding:24px;">Chord layout unavailable '
      + '(d3 failed to load — the diagram needs network access to the d3 CDN).</div>';
    return;
  }

  const f = IncytrFilter.get();
  const snap = _ihSnapPvalue(f.hmPvalue);
  const snapAp = _ihSnapAbsPds(f.hmAbsPds);

  // The panels the heatmap timeline would scrub — laid out side by side instead.
  // Falls back to the single active contrast for a 1-contrast cohort.
  let panels = _ihTimelinePanels(f);
  if (!panels.length) {
    panels = [{
      label: _ihContrastFromState(), contrast: _ihContrastFromState(),
      disease: f.hmDisease, timepoint: f.hmTimepoint,
    }];
  }

  const axes = _ihVisibleAxes(block, f, snap, snapAp, panels);
  const empty = new Set(block.empty_deg_celltypes || []);

  // Shared node ring + member maps, computed once and reused for every panel so
  // node identity, order, and color are constant across the small multiples.
  const memberS = new Map(), memberR = new Map();
  axes.senders.forEach((lab, i) => memberS.set(lab, axes.senderMembers[i]));
  axes.receivers.forEach((lab, i) => memberR.set(lab, axes.receiverMembers[i]));
  const nodes = [];
  const seen = new Set();
  for (const lab of axes.senders.concat(axes.receivers))
    if (!seen.has(lab)) { seen.add(lab); nodes.push(lab); }
  const N = nodes.length;
  const palette = _icPalette(nodes);
  const noun = _ihEntityNoun();

  d3.select(el).selectAll("*").remove();
  el.style.flexWrap = "wrap";
  el.style.alignItems = "flex-start";

  // Per-panel size: up to 3 at full size per row, then wrap.
  const perRow = Math.min(panels.length, 3);
  const avail = el.clientWidth || 900;
  const size = Math.max(260, Math.min(440, Math.floor(avail / perRow) - 24));

  let grand = 0;
  const perPanel = [];
  for (const panel of panels) {
    const cIdx = block.contrasts.indexOf(panel.contrast);
    const { matrix, total } = _icBuildMatrix(block, empty, memberS, memberR, nodes, cIdx, snap, snapAp);
    grand += total;
    perPanel.push(`${panel.label}: ${total.toLocaleString()}`);

    const wrap = document.createElement("div");
    wrap.style.cssText = "display:inline-flex;flex-direction:column;align-items:center;margin:6px 8px;";
    const lbl = document.createElement("div");
    lbl.style.cssText = "font-size:11px;font-weight:600;color:#333;margin-bottom:1px;";
    lbl.textContent = panel.contrast;
    const sub = document.createElement("div");
    sub.className = "muted";
    sub.style.cssText = "font-size:10px;margin-bottom:2px;";
    sub.textContent = total ? `${total.toLocaleString()} ${noun}` : `no ${noun}`;
    wrap.appendChild(lbl);
    wrap.appendChild(sub);
    el.appendChild(wrap);
    if (!total) {
      const ph = document.createElement("div");
      ph.className = "muted";
      ph.style.cssText = `width:${size}px;height:${size}px;display:flex;align-items:center;`
        + "justify-content:center;font-size:11px;";
      ph.textContent = "—";
      wrap.appendChild(ph);
      continue;
    }
    _icRenderPanel(el, wrap, nodes, palette, matrix, size, noun,
      (si, ti) => _icChordClick(block, axes, empty, memberS, memberR, nodes, si, ti, snap, snapAp, panel));
  }

  if (countEl) {
    const tps = _ihTimepoints();
    const header = tps.length > 1 ? f.hmDisease : (f.hmTimepoint || "all");
    const gate = _ihGateText(snap, snapAp);
    const grpTxt = axes.grouped ? ` · ${N} tissue nodes`
      : (axes.limited ? ` · top ${N} cell types` : ` · ${N} cell types`);
    if (!grand) {
      countEl.textContent = `${header} · no ${noun} at ${gate.pTxt}${gate.apTxt.replace(" · ", " & ")}.`;
    } else {
      const lead = panels.length > 1
        ? `${header} · ${panels.length} contrasts side by side (${perPanel.join(" · ")})`
        : `${panels[0].contrast} · ${grand.toLocaleString()} ${noun}`;
      countEl.textContent = `${lead} at ${gate.pTxt}${gate.apTxt.replace(" · ", " & ")}`
        + ` · ${_ihSignText()}${grpTxt}.`
        + ` Ribbon width ∝ path count; the arrow points to the receiver.`
        + ` Hover a node to isolate its ribbons across all panels; click a ribbon to open it in the table.`;
    }
  }
}

// One small-multiple chord, drawn into `wrap`. Node order/color come from the
// shared `nodes`/`palette`, so the arc sequence and hues match every panel.
// Hover-isolate is keyed to the shared container `el` so it spans all panels.
function _icRenderPanel(el, wrap, nodes, palette, matrix, size, noun, onRibbonClick) {
  const N = nodes.length;
  const fontPx = size < 360 ? 9 : 11;
  const margin = Math.max(52, size * 0.2);
  const outerR = size / 2 - margin;
  const innerR = outerR - 12;
  const color = i => palette[i];

  const chordGen = d3.chordDirected()
    .padAngle(Math.min(0.04, 0.6 / N))
    .sortGroups(null)                    // keep node arcs in shared index order
    .sortSubgroups(d3.descending)
    .sortChords(d3.descending);
  const chords = chordGen(matrix);
  const arc = d3.arc().innerRadius(innerR).outerRadius(outerR);
  const ribbon = d3.ribbonArrow().radius(innerR - 0.5).headRadius(Math.max(6, innerR / 14));

  const svg = d3.select(wrap).append("svg")
    .attr("width", size).attr("height", size)
    .attr("viewBox", [-size / 2, -size / 2, size, size])
    .attr("style", `max-width:100%;height:auto;font:${fontPx}px 'IBM Plex Sans',system-ui,sans-serif;`);

  // Outer arcs — one per node, sized by total flow through the node.
  const group = svg.append("g").selectAll("g").data(chords.groups).join("g");
  group.append("path")
    .attr("d", arc)
    .attr("fill", d => color(d.index))
    .attr("stroke", "#fff").attr("stroke-width", 1)
    .style("cursor", "pointer")
    .on("mouseover", (ev, d) => _icFade(el, d.index))
    .on("mouseout", () => _icUnfade(el))
    .append("title").text(d => {
      let out = 0, inc = 0;
      for (let j = 0; j < N; j++) { out += matrix[d.index][j]; inc += matrix[j][d.index]; }
      return `${nodes[d.index]}\n${out.toLocaleString()} ${noun} sent · `
        + `${inc.toLocaleString()} received`;
    });

  // Node labels around the ring (flipped on the left half to stay upright).
  group.append("text")
    .each(d => { d.angle = (d.startAngle + d.endAngle) / 2; })
    .attr("dy", "0.35em")
    .attr("transform", d =>
      `rotate(${(d.angle * 180 / Math.PI) - 90}) translate(${outerR + 6})`
      + (d.angle > Math.PI ? " rotate(180)" : ""))
    .attr("text-anchor", d => d.angle > Math.PI ? "end" : "start")
    .attr("fill", "#333")
    .text(d => nodes[d.index]);

  // Directed ribbons (sender → receiver), colored by the sender.
  svg.append("g").attr("fill-opacity", 0.7).selectAll("path").data(chords).join("path")
    .attr("class", "ic-ribbon")
    .attr("d", ribbon)
    .attr("fill", d => color(d.source.index))
    .attr("stroke", "#fff").attr("stroke-width", 0.4)
    .style("cursor", "pointer")
    .attr("data-s", d => d.source.index)
    .attr("data-t", d => d.target.index)
    .on("click", (ev, d) => onRibbonClick(d.source.index, d.target.index))
    .append("title")
    .text(d => `${nodes[d.source.index]} → ${nodes[d.target.index]}: `
      + `${matrix[d.source.index][d.target.index].toLocaleString()} ${noun}`);
}

// Hover-isolate: dim every ribbon (across ALL panels) not incident to the
// hovered node. Node indices are shared, so this links the small multiples.
function _icFade(el, idx) {
  d3.select(el).selectAll(".ic-ribbon").attr("fill-opacity", function () {
    const s = +this.getAttribute("data-s"), t = +this.getAttribute("data-t");
    return (s === idx || t === idx) ? 0.88 : 0.06;
  });
}
function _icUnfade(el) {
  d3.select(el).selectAll(".ic-ribbon").attr("fill-opacity", 0.7);
}

// Ribbon click → open that sender→receiver relationship in the table, mirroring
// the heatmap-cell dispatch: grouped nodes seed the member multiselects (Top
// mode); a full-grain node pins the single (sender,receiver) pair. The panel
// supplies the disease/timepoint so the click lands on the clicked contrast.
function _icChordClick(block, axes, empty, memberS, memberR, nodes, si, ti, snap, snapAp, panel) {
  const memS = memberS.get(nodes[si]), memR = memberR.get(nodes[ti]);
  if (!memS || !memR) return;
  if (_ihGroupEmpty(memS, block.senders, empty) || _ihGroupEmpty(memR, block.receivers, empty)) return;
  if (axes.grouped) {
    _ihSeedGroupedFilters(
      memS.map(i => block.senders[i]),
      memR.map(i => block.receivers[i]),
      panel.disease, panel.timepoint, snap, snapAp);
  } else {
    _ihSeedPathwayFilters(nodes[si], nodes[ti], panel.disease, panel.timepoint, snap, snapAp);
  }
}
