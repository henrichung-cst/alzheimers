// Incytr global filter-index — the complete pathway universe as packed columns.
//
// Replaces the former top-5000 `top_instances` payload pre-cap. The build
// (alz/build_unified_viewer.py) emits one gzipped binary covering EVERY
// pathway, rows pre-sorted by ABS(PDS) DESC so row position == global rank.
// This module fetches it once, maps each column as a zero-copy TypedArray view,
// and runs filter -> rank -> slice(N) over the whole universe. The per-page
// cap (500/1000/5000) is therefore a true render budget, never a data gate: a
// filter can no longer starve because its survivors sat below an upstream cut.
//
// Contract with incytr_pathways.js (top mode only): filterRank(f) returns the
// already-capped, already-sorted index list + the true universe match count;
// materialize(i) hydrates one display row. Pair mode is untouched.
window.IncytrGlobalIndex = (function() {
  let _data = null;          // { nrows, cols, gi, lc, rank, scratch }
  let _loadPromise = null;

  function _block() {
    return (window.ViewerPayload && ViewerPayload.incytr)
      ? ViewerPayload.incytr() : null;
  }

  function manifest() {
    const b = _block();
    return (b && b.global_index) ? b.global_index : null;
  }

  function available() { return !!manifest(); }

  // Sync check: is the binary already fetched + mapped? (filterRank/materialize
  // require this; callers gate on it and fall back to ensureLoaded() otherwise.)
  function loaded() { return !!_data; }

  // --- float16 (bit-pattern u16) -> float32. Scores only; PDS/pvalue are f4. --
  function _f16(h) {
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  // localeCompare-consistent rank-by-id for a string vocab (so integer-space
  // sorting reproduces the legacy String.localeCompare ordering).
  function _rankByLocale(vocab) {
    const ids = vocab.map((_, i) => i);
    ids.sort((a, b) => vocab[a].localeCompare(vocab[b]));
    const rank = new Int32Array(vocab.length);
    for (let p = 0; p < ids.length; p++) rank[ids[p]] = p;
    return rank;
  }

  async function ensureLoaded() {
    if (_data) return _data;
    if (_loadPromise) return _loadPromise;
    const gi = manifest();
    if (!gi) return null;
    _loadPromise = (async () => {
      const resp = await fetch(gi.url);
      if (!resp.ok) throw new Error(`incytr index fetch ${gi.url} -> ${resp.status}`);
      // Mirror 01_state.js payload gunzip: blob -> DecompressionStream -> buffer.
      const blob = await resp.blob();
      const stream = blob.stream().pipeThrough(new DecompressionStream("gzip"));
      const buf = await new Response(stream).arrayBuffer();
      const N = gi.nrows;
      const cols = {};
      let off = 0;
      for (const c of gi.columns) {
        if (c.type === "f4") cols[c.name] = new Float32Array(buf, off, N);
        else if (c.type === "u2") cols[c.name] = new Uint16Array(buf, off, N);
        else if (c.type === "u1") cols[c.name] = new Uint8Array(buf, off, N);
        else throw new Error(`incytr index: unknown column type ${c.type}`);
        off += c.bytes;
      }
      // Precompute: lowercased vocabs (search), locale ranks (string sort),
      // parsed disease/timepoint per contrast, and a reusable match scratch.
      const lc = {
        gene: gi.gene_vocab.map(s => s.toLowerCase()),
        sender: gi.sender_vocab.map(s => s.toLowerCase()),
        receiver: gi.receiver_vocab.map(s => s.toLowerCase()),
        contrast: gi.contrast_vocab.map(s => s.toLowerCase()),
      };
      const contrastDis = [], contrastTp = [];
      for (const c of gi.contrast_vocab) {
        const u = c.indexOf("_");
        contrastDis.push(u < 0 ? c : c.substring(0, u));
        contrastTp.push(u < 0 ? "" : c.substring(u + 1));
      }
      const rank = {
        gene: _rankByLocale(gi.gene_vocab),
        sender: _rankByLocale(gi.sender_vocab),
        receiver: _rankByLocale(gi.receiver_vocab),
        contrast: _rankByLocale(gi.contrast_vocab),
        geneSpan: gi.gene_vocab.length,  // base for the Path composite key
      };
      _data = {
        nrows: N, cols, gi, lc, rank, contrastDis, contrastTp,
        scratch: new Int32Array(N),
      };
      return _data;
    })();
    return _loadPromise;
  }

  // Drop the cached binary index so the next ensureLoaded() refetches against
  // the active context's manifest. Called on context switch (the index is a
  // per-context artifact — Song AD and each 5xFAD tissue have their own).
  function reset() {
    _data = null;
    _loadPromise = null;
  }

  // Build a Uint8 membership mask of `vocab` ids whose lowercased name includes
  // `tok`. Used per search token across each searchable vocab.
  function _member(lcVocab, tok) {
    const m = new Uint8Array(lcVocab.length);
    for (let i = 0; i < lcVocab.length; i++) if (lcVocab[i].indexOf(tok) >= 0) m[i] = 1;
    return m;
  }

  // Resolve filter state -> integer predicates, then single-scan the universe.
  // Returns { indices, total }: indices = capped+sorted row ids (<= topLimit),
  // total = full count of rows passing the filters (the true universe count).
  function filterRank(f, opts = {}) {
    const d = _data;
    if (!d) return { indices: [], total: 0 };
    const { cols, gi, lc, rank } = d;
    const N = d.nrows;

    // --- effect-size + pvalue gates -------------------------------------
    const sPds = (f.sliderPds != null) ? Number(f.sliderPds) : null;
    const sP = (f.sliderP != null) ? Number(f.sliderP) : null;
    const pdsSign = (f.pdsSign === "up" || f.pdsSign === "down") ? f.pdsSign : "both";
    const scoreGates = [];
    const scoreMinAbs = (f.scoreMinAbs && typeof f.scoreMinAbs === "object")
      ? f.scoreMinAbs : {};
    for (const key of gi.score_columns || []) {
      const raw = scoreMinAbs[key];
      if (raw == null || raw === "") continue;
      const minAbs = Number(raw);
      if (!isFinite(minAbs) || minAbs < 0) continue;
      if (cols[key]) scoreGates.push({ col: cols[key], minAbs });
    }

    // --- disease / timepoint -> allowed contrastId ----------------------
    const diseaseSet = new Set(f.disease || []);
    const timeSet = new Set(f.timepoint || []);
    const nC = gi.contrast_vocab.length;
    const contrastOk = new Uint8Array(nC);
    for (let c = 0; c < nC; c++) {
      contrastOk[c] = ((diseaseSet.size === 0 || diseaseSet.has(d.contrastDis[c]))
        && (timeSet.size === 0 || timeSet.has(d.contrastTp[c]))) ? 1 : 0;
    }
    const senderSet = new Set(f.senderIn || []);
    const receiverSet = new Set(f.receiverIn || []);
    const senderOk = senderSet.size ? new Uint8Array(gi.sender_vocab.length) : null;
    const receiverOk = receiverSet.size ? new Uint8Array(gi.receiver_vocab.length) : null;
    if (senderOk) gi.sender_vocab.forEach((name, i) => { if (senderSet.has(name)) senderOk[i] = 1; });
    if (receiverOk) gi.receiver_vocab.forEach((name, i) => { if (receiverSet.has(name)) receiverOk[i] = 1; });

    // --- sparse-cell QC: exclude low-signal sender/receiver endpoints ----
    let lowSender = null, lowReceiver = null;
    const block = _block();
    if (window.IncytrCelltypeQc && IncytrCelltypeQc.enabled(block)) {
      const lowSet = IncytrCelltypeQc.lowSignalSet(block);
      lowSender = new Uint8Array(gi.sender_vocab.length);
      lowReceiver = new Uint8Array(gi.receiver_vocab.length);
      gi.sender_vocab.forEach((n, i) => { if (lowSet.has(n)) lowSender[i] = 1; });
      gi.receiver_vocab.forEach((n, i) => { if (lowSet.has(n)) lowReceiver[i] = 1; });
    }

    // --- trajectory trend -> bit position in trajBits -------------------
    let trendBit = -1;
    const trend = (window.TrendFilter && f.trend)
      ? TrendFilter.payloadLabel(f.trend) : "";
    if (trend) {
      const bi = gi.traj_label_vocab.indexOf(trend);
      if (bi >= 0) trendBit = bi; else return { indices: [], total: 0 };
    }

    // --- search tokens -> per-token membership masks (AND across tokens) -
    const tokens = (f.searchText || "").toLowerCase().split(/\s+/).filter(Boolean);
    const tokMasks = tokens.map(t => ({
      gene: _member(lc.gene, t),
      sender: _member(lc.sender, t),
      receiver: _member(lc.receiver, t),
      contrast: _member(lc.contrast, t),
    }));

    // --- single scan over the universe ----------------------------------
    const PDS = cols.PDS, PV = cols.pvalue;
    const sid = cols.senderId, rid = cols.receiverId, cid = cols.contrastId;
    const lig = cols.ligandId, rec = cols.receptorId, em = cols.emId, tgt = cols.targetId;
    const trj = cols.trajBits;
    const matched = d.scratch;
    let m = 0;
    for (let i = 0; i < N; i++) {
      if (sPds != null && !(Math.abs(PDS[i]) >= sPds)) continue;
      if (pdsSign === "up" && !(PDS[i] > 0)) continue;
      if (pdsSign === "down" && !(PDS[i] < 0)) continue;
      if (sP != null && !(PV[i] < sP)) continue;          // NaN pvalue excluded
      if (!contrastOk[cid[i]]) continue;
      if (senderOk && !senderOk[sid[i]]) continue;
      if (receiverOk && !receiverOk[rid[i]]) continue;
      if (lowSender && (lowSender[sid[i]] || lowReceiver[rid[i]])) continue;
      if (trendBit >= 0 && !((trj[i] >> trendBit) & 1)) continue;
      if (scoreGates.length) {
        let ok = true;
        for (let g = 0; g < scoreGates.length; g++) {
          const val = _f16(scoreGates[g].col[i]);
          if (!(Math.abs(val) >= scoreGates[g].minAbs)) { ok = false; break; }
        }
        if (!ok) continue;
      }
      if (tokMasks.length) {
        let ok = true;
        for (let k = 0; k < tokMasks.length; k++) {
          const tm = tokMasks[k];
          if (!(tm.gene[lig[i]] || tm.gene[rec[i]] || tm.gene[em[i]] || tm.gene[tgt[i]]
                || tm.sender[sid[i]] || tm.receiver[rid[i]] || tm.contrast[cid[i]])) {
            ok = false; break;
          }
        }
        if (!ok) continue;
      }
      matched[m++] = i;
    }
    const total = m;

    // --- order + cap -----------------------------------------------------
    const topLimit = opts.limit === "all"
      ? total
      : ([500, 1000, 5000].includes(Number(f.topLimit))
          ? Number(f.topLimit) : 500);
    const key = f.sortKey || "rank";
    const dir = f.sortDir || 1;

    // Default (rank): matched is already in ABS(PDS)-desc order == rank asc.
    if (key === "rank") {
      const k = Math.min(total, topLimit);
      if (dir > 0) {
        const out = new Array(k);
        for (let t = 0; t < k; t++) out[t] = matched[t];
        return { indices: out, total };
      }
      // rank desc: the last k matched rows, reversed.
      const out = new Array(k);
      for (let t = 0; t < k; t++) out[t] = matched[total - 1 - t];
      return { indices: out, total };
    }

    // Custom sort: numeric key per matched row, NaN always last (legacy rule).
    const keyV = new Float64Array(total);
    const keyOf = _keyFn(d, key);
    for (let t = 0; t < total; t++) keyV[t] = keyOf(matched[t]);
    const indices = _selectTopK(matched, keyV, total, dir, topLimit);
    return { indices, total };
  }

  // Per-row numeric sort key for a column (vocab string cols -> locale rank).
  function _keyFn(d, key) {
    const { cols, rank } = d;
    if (key === "PDS") return i => cols.PDS[i];
    if (key === "pvalue") return i => cols.pvalue[i];
    if (cols[key] && d.gi.score_columns.indexOf(key) >= 0) {
      const col = cols[key]; return i => _f16(col[i]);
    }
    if (key === "_sender") { const c = cols.senderId, r = rank.sender; return i => r[c[i]]; }
    if (key === "_receiver") { const c = cols.receiverId, r = rank.receiver; return i => r[c[i]]; }
    if (key === "contrast") { const c = cols.contrastId, r = rank.contrast; return i => r[c[i]]; }
    if (key === "Ligand") { const c = cols.ligandId, r = rank.gene; return i => r[c[i]]; }
    if (key === "Receptor") { const c = cols.receptorId, r = rank.gene; return i => r[c[i]]; }
    if (key === "EM") { const c = cols.emId, r = rank.gene; return i => r[c[i]]; }
    if (key === "Target") { const c = cols.targetId, r = rank.gene; return i => r[c[i]]; }
    if (key === "Path") {
      // Composite ligand>receptor>EM>target locale-rank, base = gene count.
      const g = rank.gene, B = rank.geneSpan;
      const L = cols.ligandId, R = cols.receptorId, E = cols.emId, T = cols.targetId;
      return i => ((g[L[i]] * B + g[R[i]]) * B + g[E[i]]) * B + g[T[i]];
    }
    return () => 0;  // unknown key: stable no-op order
  }

  // Bounded top-K selection by keyV with direction; NaN ranks last regardless
  // of dir (reproduces the legacy null-last comparator). Returns matched-id
  // array, length min(total, k), best-first.
  function _selectTopK(matched, keyV, total, dir, k) {
    // cmp(a,b) over positions into keyV: <0 if a ranks before b.
    const cmp = (a, b) => {
      const va = keyV[a], vb = keyV[b];
      const na = va !== va, nb = vb !== vb;  // NaN test
      if (na && nb) return 0;
      if (na) return 1;
      if (nb) return -1;
      if (va === vb) return 0;
      return dir > 0 ? (va < vb ? -1 : 1) : (va > vb ? -1 : 1);
    };
    if (total <= k) {
      const pos = new Array(total);
      for (let t = 0; t < total; t++) pos[t] = t;
      pos.sort(cmp);
      return pos.map(t => matched[t]);
    }
    // Max-heap (root = worst kept) of positions, size k. Evict when a better
    // row arrives. O(total log k).
    const heap = new Int32Array(k);
    let n = 0;
    const swap = (x, y) => { const tmp = heap[x]; heap[x] = heap[y]; heap[y] = tmp; };
    const up = c => {
      while (c > 0) { const p = (c - 1) >> 1; if (cmp(heap[p], heap[c]) >= 0) break; swap(p, c); c = p; }
    };
    const down = p => {
      for (;;) {
        let l = 2 * p + 1, r = l + 1, w = p;
        if (l < n && cmp(heap[l], heap[w]) > 0) w = l;
        if (r < n && cmp(heap[r], heap[w]) > 0) w = r;
        if (w === p) break; swap(p, w); p = w;
      }
    };
    for (let t = 0; t < total; t++) {
      if (n < k) { heap[n] = t; up(n); n++; }
      else if (cmp(t, heap[0]) < 0) { heap[0] = t; down(0); }
    }
    const pos = Array.from(heap.subarray(0, n));
    pos.sort(cmp);
    return pos.map(t => matched[t]);
  }

  function _labelState(code) {
    const states = manifest().label_states; // ["", "DEG", "prG"]
    return (code > 0 && code < states.length) ? states[code] : "";
  }

  function _decodeTraj(bits) {
    const vocab = manifest().traj_label_vocab;
    const out = [];
    for (let i = 0; i < vocab.length; i++) if ((bits >> i) & 1) out.push(vocab[i]);
    return out.join(";");
  }

  // Hydrate one display row from the columns (decode ids/f16/bitfields). Called
  // only for the <=100 rows on the current page.
  function materialize(i) {
    const d = _data;
    if (!d) return null;
    const { cols, gi } = d;
    const Ligand = gi.gene_vocab[cols.ligandId[i]];
    const Receptor = gi.gene_vocab[cols.receptorId[i]];
    const EM = gi.gene_vocab[cols.emId[i]];
    const Target = gi.gene_vocab[cols.targetId[i]];
    const lb = cols.labelBits[i];
    const pv = cols.pvalue[i];
    const row = {
      rank: i + 1,                          // global rank == row position + 1
      _sender: gi.sender_vocab[cols.senderId[i]],
      _receiver: gi.receiver_vocab[cols.receiverId[i]],
      sender: gi.sender_vocab[cols.senderId[i]],
      receiver: gi.receiver_vocab[cols.receiverId[i]],
      Ligand, Receptor, EM, Target,
      Path: `${Ligand}|${Receptor}|${EM}|${Target}`,
      Ligand_label: _labelState(lb & 3),
      Receptor_label: _labelState((lb >> 2) & 3),
      EM_label: _labelState((lb >> 4) & 3),
      Target_label: _labelState((lb >> 6) & 3),
      contrast: gi.contrast_vocab[cols.contrastId[i]],
      pvalue: (pv === pv) ? pv : null,
      PDS: cols.PDS[i],
      traj_labels: _decodeTraj(cols.trajBits[i]),
    };
    for (const sc of gi.score_columns) row[sc] = _f16(cols[sc][i]);
    return row;
  }

  return { available, loaded, manifest, ensureLoaded, filterRank, materialize, reset };
})();
