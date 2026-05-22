// ---------------------------------------------------------------------------
// SequenceLogo — consensus-motif renderer for kinase library PSSMs.
//
// Single row of letters, one column per position, sized uniformly. At each
// position we list the amino acids whose probability is at least RATIO_CUT×
// the median probability across the 20 canonical residues at that position;
// positions with no qualifying residue render as "x". Position 0 (the
// phosphoacceptor) is rendered from `st_fav` for ST kinases (sized by
// preference) and as fixed `y` for tyrosine.
//
// RATIO_CUT is a heuristic — 2.5× empirically separates clear preferences
// (R/K at -3 for basophilic kinases, P at +1 for proline-directed kinases)
// from background noise in the 23-row PSSM. Tunable via opts.ratioCut.
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

  const RATIO_CUT_DEFAULT = 1.75;  // "clear preference" threshold
  const MAX_LETTERS_PER_POS = 2;   // never more than two letters per column

  function _median(arr) {
    const s = arr.slice().sort((a, b) => a - b);
    const n = s.length;
    if (!n) return 0;
    return n % 2 ? s[(n - 1) >> 1] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
  }

  function _consensusForCol(matrix, aa, colIdx, ratioCut) {
    const col = matrix.map(row => Number(row[colIdx]) || 0);
    const canonical = [];
    for (let i = 0; i < aa.length; i++) {
      const a = aa[i];
      if (a.length === 1 && a >= "A" && a <= "Z") canonical.push(col[i]);
    }
    const med = _median(canonical);
    if (!(med > 0)) return [];
    const candidates = [];
    for (let i = 0; i < aa.length; i++) {
      const r = col[i] / med;
      if (r >= ratioCut) candidates.push({aa: aa[i], ratio: r});
    }
    candidates.sort((a, b) => b.ratio - a.ratio);
    return candidates.slice(0, MAX_LETTERS_PER_POS);
  }

  function _centerLetters(kinType, stFav) {
    if (kinType === "tyrosine") return [{aa: "y"}];
    if (stFav) {
      const s = Number(stFav.S) || 0;
      const t = Number(stFav.T) || 0;
      if (s + t > 0) {
        const out = [];
        // include both if neither dominates; otherwise just the winner.
        if (s >= 0.7 * t && t >= 0.7 * s) {
          out.push({aa: "S"}); out.push({aa: "T"});
        } else if (s > t) out.push({aa: "S"});
        else out.push({aa: "T"});
        return out;
      }
    }
    return [{aa: "S/T"}];
  }

  function render(host, motif, opts) {
    if (!host) return;
    if (!motif || !motif.matrix) {
      host.innerHTML = `<div class="muted" style="padding:0.5em">No PSSM available for this kinase.</div>`;
      return;
    }
    opts = opts || {};
    const ratioCut = opts.ratioCut || RATIO_CUT_DEFAULT;
    const colW   = opts.colWidth   || 32;
    const rowH   = opts.rowHeight  || 28;   // single letter row
    const stackGap = 2;                     // px between stacked letters
    const padT   = 6;
    const padB   = 18;                      // axis labels
    const padL   = 8;
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
      ? _consensusForCol(motif.matrix, motif.amino_acids, c.idx, ratioCut)
      : _centerLetters(motif.kin_type, motif.st_fav));

    // Cell height = MAX_LETTERS_PER_POS rows so columns align visually.
    const cellH = rowH * MAX_LETTERS_PER_POS + stackGap * (MAX_LETTERS_PER_POS - 1);
    const width  = padL + padR + cols.length * colW;
    const height = padT + padB + cellH;

    const colGroups = colData.map((entries, ci) => {
      const x = padL + ci * colW;
      const isCenter = cols[ci].pos === 0;
      let html = "";
      if (entries.length === 0) {
        const y = padT + cellH / 2 + 7;
        html += `<text x="${x + colW / 2}" y="${y}" text-anchor="middle" `
              + `font-family="ui-monospace, Menlo, Consolas, monospace" `
              + `font-size="20" fill="#bbb">x</text>`;
      } else {
        for (let i = 0; i < entries.length; i++) {
          const e = entries[i];
          const y = padT + (i + 1) * rowH + i * stackGap - 6;
          const col = AA_COLOR[e.aa] || "#222";
          html += `<text x="${x + colW / 2}" y="${y}" text-anchor="middle" `
                + `font-family="ui-monospace, Menlo, Consolas, monospace" `
                + `font-weight="700" font-size="22" fill="${col}">`
                + `${_escapeHtml(e.aa)}</text>`;
        }
      }
      const lbl = String(cols[ci].pos);
      html += `<text x="${x + colW / 2}" y="${padT + cellH + 12}" `
            + `text-anchor="middle" font-size="11" fill="#555" `
            + `font-family="ui-monospace, Menlo, Consolas, monospace">${lbl}</text>`;
      if (isCenter) {
        html += `<line x1="${x - 1}" y1="${padT}" x2="${x - 1}" y2="${padT + cellH}" `
              + `stroke="#c43a3a" stroke-width="1"/>`
              + `<line x1="${x + colW + 1}" y1="${padT}" x2="${x + colW + 1}" y2="${padT + cellH}" `
              + `stroke="#c43a3a" stroke-width="1"/>`;
      }
      return html;
    }).join("");

    host.innerHTML = `<svg class="sequence-logo" width="${width}" height="${height}" `
                   + `viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`
                   + colGroups + `</svg>`;
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
      + `Shown: residues with probability ≥ ${RATIO_CUT_DEFAULT}× the per-position median over the 20 canonical AAs `
      + `(max ${MAX_LETTERS_PER_POS} per column; "x" = none qualify).</p></section>`;
  }

  return {render, buildBlock};
})();
