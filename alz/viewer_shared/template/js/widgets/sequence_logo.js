// ---------------------------------------------------------------------------
// SequenceLogo — pLogo-style renderer for kinase library PSSMs.
//
// Each flanking column is normalized over the 20 canonical amino acids and
// rendered as a sequence-logo stack. Total stack height is the column's
// information content; individual letter heights are probability ×
// information content. Position 0 is the phosphoacceptor: fixed Y for
// tyrosine kinases, or S/T favorability for serine/threonine kinases.
// ---------------------------------------------------------------------------

const SequenceLogo = (() => {
  const AA_COLOR = {
    A: "#222", V: "#222", L: "#222", I: "#222", M: "#b8a000",
    F: "#222", W: "#222", P: "#222", G: "#1f8a3a",
    S: "#1f8a3a", T: "#1f8a3a", C: "#b8a000",
    Y: "#1f8a3a", N: "#1f8a3a", Q: "#1f8a3a", H: "#1f8a3a",
    D: "#c43a3a", E: "#c43a3a",
    K: "#2a59c6", R: "#2a59c6",
    s: "#1f8a3a", t: "#1f8a3a", y: "#1f8a3a",
  };

  const CANONICAL_AA = new Set([
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
  ]);
  const MAX_BITS = Math.log2(20);
  const MAX_VISIBLE_AA = 8;
  const MIN_LETTER_PX = 3;

  function _log2(x) {
    return Math.log(x) / Math.log(2);
  }

  function _canonicalWeights(matrix, aa, colIdx) {
    const weights = [];
    for (let i = 0; i < aa.length; i++) {
      const a = aa[i];
      if (!CANONICAL_AA.has(a)) continue;
      const v = Number(matrix[i] && matrix[i][colIdx]);
      if (Number.isFinite(v) && v > 0) weights.push({aa: a, value: v});
    }
    return weights;
  }

  function _stackForCol(matrix, aa, colIdx, pxPerBit) {
    const weights = _canonicalWeights(matrix, aa, colIdx);
    const total = weights.reduce((acc, w) => acc + w.value, 0);
    if (!(total > 0)) return {entries: [], bits: 0};

    let entropy = 0;
    const probs = weights.map(w => {
      const p = w.value / total;
      entropy -= p > 0 ? p * _log2(p) : 0;
      return {...w, p};
    });
    const bits = Math.max(0, MAX_BITS - entropy);
    if (!(bits > 0)) return {entries: [], bits: 0};

    const entries = probs
      .map(w => ({aa: w.aa, p: w.p, bits: w.p * bits}))
      .filter(w => w.bits * pxPerBit >= MIN_LETTER_PX)
      .sort((a, b) => b.bits - a.bits)
      .slice(0, MAX_VISIBLE_AA)
      .sort((a, b) => a.bits - b.bits);
    return {entries, bits};
  }

  function _centerStack(kinType, stFav) {
    if (kinType === "tyrosine") return {entries: [{aa: "Y", bits: MAX_BITS}], bits: MAX_BITS};
    if (stFav) {
      const s = Number(stFav.S) || 0;
      const t = Number(stFav.T) || 0;
      if (s + t > 0) {
        const entries = [];
        if (s > 0) entries.push({aa: "S", bits: MAX_BITS * s / (s + t)});
        if (t > 0) entries.push({aa: "T", bits: MAX_BITS * t / (s + t)});
        entries.sort((a, b) => a.bits - b.bits);
        return {entries, bits: MAX_BITS};
      }
    }
    return {
      entries: [{aa: "S", bits: MAX_BITS / 2}, {aa: "T", bits: MAX_BITS / 2}],
      bits: MAX_BITS,
    };
  }

  function _svgText(entry, x, baselineY, heightPx, baseFontSize) {
    const col = AA_COLOR[entry.aa] || "#222";
    const scaleY = Math.max(0.01, heightPx / baseFontSize);
    return `<text x="0" y="0" text-anchor="middle" `
      + `font-family="ui-monospace, Menlo, Consolas, monospace" `
      + `font-weight="700" font-size="${baseFontSize}" fill="${col}" `
      + `transform="matrix(1 0 0 ${scaleY} ${x} ${baselineY})">`
      + `${_escapeHtml(entry.aa)}</text>`;
  }

  function render(host, motif, opts) {
    if (!host) return;
    if (!motif || !motif.matrix) {
      host.innerHTML = `<div class="muted" style="padding:0.5em">No PSSM available for this kinase.</div>`;
      return;
    }
    opts = opts || {};
    const colW   = opts.colWidth   || 31;
    const pxPerBit = opts.pxPerBit || 20;
    const stackH = MAX_BITS * pxPerBit;
    const baseFontSize = opts.baseFontSize || 28;
    const padT   = 8;
    const padB   = 20;                      // axis labels
    const padL   = 30;                      // bit axis
    const padR   = 8;

    // Compose column list with center (position 0).
    const cols = [];
    let negEnd = -1;
    for (let i = 0; i < motif.positions.length; i++) {
      if (motif.positions[i] < 0) negEnd = i;
    }
    for (let i = 0; i <= negEnd; i++) {
      cols.push({pos: motif.positions[i], src: "matrix", idx: i});
    }
    cols.push({pos: 0, src: "center"});
    for (let i = negEnd + 1; i < motif.positions.length; i++) {
      cols.push({pos: motif.positions[i], src: "matrix", idx: i});
    }

    const colData = cols.map(c => c.src === "matrix"
      ? _stackForCol(motif.matrix, motif.amino_acids, c.idx, pxPerBit)
      : _centerStack(motif.kin_type, motif.st_fav));

    const width  = padL + padR + cols.length * colW;
    const height = padT + padB + stackH;
    const baseline = padT + stackH;

    const axis = [0, 2, 4].map(t => {
      const y = baseline - t * pxPerBit;
      return `<line x1="${padL - 4}" y1="${y}" x2="${width - padR}" y2="${y}" `
        + `stroke="${t === 0 ? "#90a4ae" : "#eceff1"}" stroke-width="1"/>`
        + `<text x="${padL - 8}" y="${y + 3}" text-anchor="end" `
        + `font-size="9" fill="#78909c" `
        + `font-family="ui-monospace, Menlo, Consolas, monospace">${t}</text>`;
    }).join("")
      + `<text x="${padL - 22}" y="${padT + stackH / 2}" text-anchor="middle" `
      + `font-size="9" fill="#78909c" `
      + `font-family="ui-monospace, Menlo, Consolas, monospace" `
      + `transform="rotate(-90 ${padL - 22} ${padT + stackH / 2})">bits</text>`;

    const colGroups = colData.map((entries, ci) => {
      const x = padL + ci * colW;
      const isCenter = cols[ci].pos === 0;
      let html = `<g><title>position ${_escapeHtml(String(cols[ci].pos))}: `
        + `${entries.bits.toFixed(2)} bits</title>`;
      if (isCenter) {
        html += `<rect x="${x}" y="${padT}" width="${colW}" height="${stackH}" `
          + `fill="#fff5f5" opacity="0.75"/>`;
      }
      let used = 0;
      for (const e of entries.entries) {
        const h = e.bits * pxPerBit;
        const y = baseline - used;
        html += _svgText(e, x + colW / 2, y, h, baseFontSize);
        used += h;
      }
      const lbl = String(cols[ci].pos);
      html += `<text x="${x + colW / 2}" y="${baseline + 13}" `
            + `text-anchor="middle" font-size="11" fill="#555" `
            + `font-family="ui-monospace, Menlo, Consolas, monospace">${lbl}</text>`;
      if (isCenter) {
        html += `<line x1="${x - 1}" y1="${padT}" x2="${x - 1}" y2="${padT + stackH}" `
              + `stroke="#c43a3a" stroke-width="1"/>`
              + `<line x1="${x + colW + 1}" y1="${padT}" x2="${x + colW + 1}" y2="${padT + stackH}" `
              + `stroke="#c43a3a" stroke-width="1"/>`;
      }
      return html + `</g>`;
    }).join("");

    host.innerHTML = `<svg class="sequence-logo" width="${width}" height="${height}" `
                   + `viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`
                   + `<rect width="${width}" height="${height}" fill="#fff"/>`
                   + axis + colGroups + `</svg>`;
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
      + `Center (0) is the phosphoacceptor — ${center}. `
      + `Letter height is scaled by probability × information content; low-information positions appear small or empty.</p></section>`;
  }

  return {render, buildBlock};
})();
