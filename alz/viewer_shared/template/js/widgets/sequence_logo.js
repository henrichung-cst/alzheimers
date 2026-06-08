// ---------------------------------------------------------------------------
// SequenceLogo — browser port of kinase_library.Kinase.seq_logo().
//
// Kinase Library's default motif logo is built from the normalized kinase
// matrix using logo_type="ratio_to_median":
//
//   height = log2(position_value / per-position median)
//
// Values above the median stack above zero; values below the median stack
// below zero. Position 0 is drawn separately from the kinase phosphoacceptor
// preference, scaled to the tallest positive flanking stack, matching the
// package's make_seq_logo() behavior.
// ---------------------------------------------------------------------------

const SequenceLogo = (() => {
  const AA_COLOR = {
    D: "#DC143C", E: "#DC143C",
    s: "#DC143C", t: "#DC143C", y: "#DC143C",
    pS: "#DC143C", pT: "#DC143C", "pS/pT": "#DC143C", pY: "#DC143C",
    R: "#0000FF", K: "#0000FF",
    C: "#DAA520", F: "#DAA520", Y: "#DAA520", W: "#DAA520",
    V: "#DAA520", I: "#DAA520", L: "#DAA520", M: "#DAA520",
    Q: "#8A2BE2", N: "#8A2BE2", H: "#8A2BE2", S: "#8A2BE2", T: "#8A2BE2",
    A: "#008000", G: "#008000",
    P: "#000000",
  };

  const DROP_AA = new Set(["s"]);
  const REPLACE_AA = {t: "pS/pT", y: "pY"};
  const MIN_LETTER_PX = 1.2;

  function _log2(x) {
    return Math.log(x) / Math.log(2);
  }

  function _median(values) {
    const s = values.filter(v => Number.isFinite(v)).sort((a, b) => a - b);
    const n = s.length;
    if (!n) return null;
    return n % 2 ? s[(n - 1) >> 1] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
  }

  function _aaLabel(aa) {
    return REPLACE_AA[aa] || aa;
  }

  function _matrixRows(motif, colIdx) {
    const rows = [];
    const aa = motif.amino_acids || [];
    const matrix = motif.matrix || [];
    for (let i = 0; i < aa.length; i++) {
      const rawAa = String(aa[i]);
      if (DROP_AA.has(rawAa)) continue;
      const v = Number(matrix[i] && matrix[i][colIdx]);
      if (Number.isFinite(v) && v > 0) rows.push({aa: _aaLabel(rawAa), value: v});
    }
    return rows;
  }

  function _stackForCol(motif, colIdx) {
    const rows = _matrixRows(motif, colIdx);
    const med = _median(rows.map(r => r.value));
    if (!(med > 0)) return {pos: [], neg: [], posSum: 0, negSum: 0, median: med || 0};

    const pos = [];
    const neg = [];
    for (const r of rows) {
      const h = _log2(r.value / med);
      if (!Number.isFinite(h) || h === 0) continue;
      if (h > 0) pos.push({aa: r.aa, h, value: r.value});
      else neg.push({aa: r.aa, h, value: r.value});
    }
    pos.sort((a, b) => a.h - b.h);
    neg.sort((a, b) => Math.abs(a.h) - Math.abs(b.h));
    return {
      pos,
      neg,
      posSum: pos.reduce((acc, r) => acc + r.h, 0),
      negSum: neg.reduce((acc, r) => acc + r.h, 0),
      median: med,
    };
  }

  function _centerEntries(kinType, stFav, centerHeight) {
    if (!(centerHeight > 0)) return [];
    if (kinType === "tyrosine") return [{aa: "Y", h: centerHeight}];
    if (stFav) {
      const s = Number(stFav.S) || 0;
      const t = Number(stFav.T) || 0;
      if (s + t > 0) {
        const out = [];
        if (s > 0) out.push({aa: "S", h: centerHeight * s / (s + t)});
        if (t > 0) out.push({aa: "T", h: centerHeight * t / (s + t)});
        return out.sort((a, b) => a.h - b.h);
      }
    }
    return [{aa: "S", h: centerHeight / 2}, {aa: "T", h: centerHeight / 2}];
  }

  function _svgText(entry, x, baselineY, heightPx, baseFontSize) {
    const label = String(entry.aa);
    const color = AA_COLOR[label] || "#222";
    const scaleY = Math.max(0.01, heightPx / baseFontSize);
    const scaleX = label.length > 1 ? Math.max(0.42, Math.min(1, 1.35 / label.length)) : 1;
    return `<text x="0" y="0" text-anchor="middle" `
      + `font-family="Arial Rounded MT Bold, Arial, ui-sans-serif, sans-serif" `
      + `font-weight="700" font-size="${baseFontSize}" fill="${color}" `
      + `transform="matrix(${scaleX} 0 0 ${scaleY} ${x} ${baselineY})">`
      + `${_escapeHtml(label)}</text>`;
  }

  function _niceStep(range) {
    if (!(range > 0)) return 1;
    if (range <= 2) return 0.5;
    if (range <= 6) return 1;
    if (range <= 12) return 2;
    return 4;
  }

  function _axisTicks(yMin, yMax) {
    const ticks = new Set([0]);
    const step = _niceStep(yMax - yMin);
    for (let v = Math.ceil(yMin / step) * step; v <= yMax + 1e-9; v += step) {
      ticks.add(Number(v.toFixed(6)));
    }
    return Array.from(ticks).sort((a, b) => a - b);
  }

  function render(host, motif, opts) {
    if (!host) return;
    if (!motif || !motif.matrix) {
      host.innerHTML = `<div class="muted" style="padding:0.5em">No PSSM available for this kinase.</div>`;
      return;
    }
    opts = opts || {};
    const colW = opts.colWidth || 31;
    const pxPerUnit = opts.pxPerUnit || 10;
    const baseFontSize = opts.baseFontSize || 28;
    const padT = 10;
    const padB = 20;
    const padL = 42;
    const padR = 8;

    const cols = [];
    let negEnd = -1;
    for (let i = 0; i < motif.positions.length; i++) {
      if (motif.positions[i] < 0) negEnd = i;
    }
    for (let i = 0; i <= negEnd; i++) cols.push({pos: motif.positions[i], src: "matrix", idx: i});
    cols.push({pos: 0, src: "center"});
    for (let i = negEnd + 1; i < motif.positions.length; i++) cols.push({pos: motif.positions[i], src: "matrix", idx: i});

    const flank = cols.map(c => c.src === "matrix" ? _stackForCol(motif, c.idx) : null);
    const maxPositive = Math.max(0, ...flank.filter(Boolean).map(s => s.posSum));
    const minNegative = Math.min(0, ...flank.filter(Boolean).map(s => s.negSum));
    const centerEntries = _centerEntries(motif.kin_type, motif.st_fav, maxPositive || 1);

    const yMaxRaw = Math.max(maxPositive, centerEntries.reduce((acc, e) => acc + e.h, 0), 1);
    const yMinRaw = Math.min(minNegative, 0);
    const yMax = Math.ceil(yMaxRaw * 10) / 10;
    const yMin = Math.floor(yMinRaw * 10) / 10;
    const plotH = Math.max(72, (yMax - yMin) * pxPerUnit);
    const width = padL + padR + cols.length * colW;
    const height = padT + padB + plotH;
    const yToPx = y => padT + (yMax - y) * pxPerUnit;
    const zeroY = yToPx(0);

    const ticks = _axisTicks(yMin, yMax);
    const axis = ticks.map(t => {
      const y = yToPx(t);
      const strong = Math.abs(t) < 1e-9;
      const label = Math.abs(t % 1) < 1e-9 ? String(t.toFixed(0)) : String(t);
      return `<line x1="${padL - 4}" y1="${y}" x2="${width - padR}" y2="${y}" `
        + `stroke="${strong ? "#263238" : "#eceff1"}" stroke-width="${strong ? 1.25 : 1}"/>`
        + `<text x="${padL - 8}" y="${y + 3}" text-anchor="end" `
        + `font-size="9" fill="#78909c" `
        + `font-family="ui-monospace, Menlo, Consolas, monospace">${label}</text>`;
    }).join("")
      + `<text x="${padL - 31}" y="${padT + plotH / 2}" text-anchor="middle" `
      + `font-size="9" fill="#78909c" `
      + `font-family="ui-monospace, Menlo, Consolas, monospace" `
      + `transform="rotate(-90 ${padL - 31} ${padT + plotH / 2})">log2 ratio</text>`;

    const groups = cols.map((c, ci) => {
      const x = padL + ci * colW;
      const cx = x + colW / 2;
      const isCenter = c.pos === 0;
      let html = `<g>`;
      if (isCenter) {
        html += `<title>position 0: phosphoacceptor preference</title>`
          + `<rect x="${x}" y="${padT}" width="${colW}" height="${plotH}" fill="#fff5f5" opacity="0.75"/>`;
        let used = 0;
        for (const e of centerEntries) {
          const hPx = e.h * pxPerUnit;
          if (hPx >= MIN_LETTER_PX) {
            html += _svgText(e, cx, zeroY - used, hPx, baseFontSize);
          }
          used += hPx;
        }
        html += `<line x1="${x - 1}" y1="${padT}" x2="${x - 1}" y2="${padT + plotH}" `
          + `stroke="#c43a3a" stroke-width="1"/>`
          + `<line x1="${x + colW + 1}" y1="${padT}" x2="${x + colW + 1}" y2="${padT + plotH}" `
          + `stroke="#c43a3a" stroke-width="1"/>`;
      } else {
        const s = flank[ci] || {pos: [], neg: [], posSum: 0, negSum: 0, median: 0};
        html += `<title>position ${_escapeHtml(String(c.pos))}: median ${s.median.toFixed(4)}, `
          + `positive stack ${s.posSum.toFixed(2)}, negative stack ${s.negSum.toFixed(2)}</title>`;
        let posUsed = 0;
        for (const e of s.pos) {
          const hPx = e.h * pxPerUnit;
          if (hPx >= MIN_LETTER_PX) html += _svgText(e, cx, zeroY - posUsed, hPx, baseFontSize);
          posUsed += hPx;
        }
        let negUsed = 0;
        for (const e of s.neg) {
          const hPx = Math.abs(e.h) * pxPerUnit;
          if (hPx >= MIN_LETTER_PX) html += _svgText(e, cx, zeroY + negUsed + hPx, hPx, baseFontSize);
          negUsed += hPx;
        }
      }
      const label = c.pos > 0 ? `+${c.pos}` : String(c.pos);
      html += `<text x="${cx}" y="${padT + plotH + 13}" text-anchor="middle" `
        + `font-size="11" fill="#555" `
        + `font-family="ui-monospace, Menlo, Consolas, monospace">${label}</text>`;
      return html + `</g>`;
    }).join("");

    host.innerHTML = `<svg class="sequence-logo" width="${width}" height="${height}" `
      + `viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`
      + `<rect width="${width}" height="${height}" fill="#fff"/>`
      + axis + groups + `</svg>`;
  }

  function buildBlock(name, motif, containerId) {
    const nm = _escapeHtml(name);
    if (!motif) {
      return `<section class="audit-panel"><div class="muted">No kinase library PSSM available for ${nm}.</div></section>`;
    }
    const center = motif.kin_type === "tyrosine"
      ? "fixed Y"
      : "rendered from S/T favorability";
    return `<section class="audit-panel"><h4>Substrate motif (kinase library)</h4>`
      + `<div id="${_escapeHtml(containerId)}" class="kinase-motif-logo"></div>`
      + `<p class="kinase-stage-note muted">Position-specific amino-acid preferences from the kinase library PSSM for ${nm}. `
      + `Logo heights use the Kinase Library default: log2(value / per-position median). `
      + `Letters above zero are favored over the position median; letters below zero are disfavored. `
      + `Center (0) is the phosphoacceptor, ${center}, scaled to the tallest positive flanking stack.</p></section>`;
  }

  return {render, buildBlock};
})();
