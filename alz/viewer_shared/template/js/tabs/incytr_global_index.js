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
  const _pathRowsCache = new Map();
  const _PATH_ROWS_CACHE_MAX = 128;

  // J-3: Grain-aware block lookup. incytr_pathways.js registers _ipGrainBlock
  // as window._ipGrainBlock when active; falls back to ViewerPayload.incytr().
  function _block() {
    if (typeof window._ipGrainBlock === "function") return window._ipGrainBlock();
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
      // Transport-tolerant gunzip (mirrors 01_state.js:_decodeGzipBuffer): sniff
      // the gzip magic (0x1f 0x8b) and only decompress when the bytes are
      // actually gzip. Some hosting layers (bioplat voila-gateway) auto-
      // decompress and hand back plain bytes; a stale cache entry stored while
      // the object carried Content-Encoding: gzip lands here too.
      const raw = await resp.arrayBuffer();
      const magic = new Uint8Array(raw);
      let buf;
      if (magic.length >= 2 && magic[0] === 0x1f && magic[1] === 0x8b) {
        const stream = new Response(raw).body.pipeThrough(new DecompressionStream("gzip"));
        buf = await new Response(stream).arrayBuffer();
      } else {
        buf = raw;
      }
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
      // Only the gene vocab needs a lowercased copy (exact match). Descriptive
      // sender/receiver/contrast search reads the original-case vocab directly so
      // _searchSegments can split on camelCase boundaries.
      const lc = {
        gene: gi.gene_vocab.map(s => s.toLowerCase()),
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
    _pathRowsCache.clear();
  }

  // Split a label into lowercased search segments on camelCase humps, digit
  // boundaries, and non-alphanumeric delimiters: "CD8CytotoxicEffector" ->
  // ["cd8", "cytotoxic", "effector"]. Callers must pass the ORIGINAL-case vocab so the
  // camelCase boundaries survive.
  function _searchSegments(value) {
    return String(value == null ? "" : value)
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter(Boolean);
  }

  // Build a Uint8 membership mask for one lowercased search token. Gene symbols
  // use exact matching (PDCD1 must never expand to PDCD10 or PDCD11). The
  // descriptive sender/receiver/contrast vocabularies use segment-prefix search
  // so a gene token ("tox") is not swallowed mid-word by a state name
  // ("CD8CytotoxicEffector") while a descriptive token ("exhaust") still matches
  // "CD8Exhausted".
  function _member(vocab, tok, allowSubstring = false) {
    const m = new Uint8Array(vocab.length);
    for (let i = 0; i < vocab.length; i++) {
      if (allowSubstring
            ? _searchSegments(vocab[i]).some(seg => seg.startsWith(tok))
            : String(vocab[i]).toLowerCase() === tok) m[i] = 1;
    }
    return m;
  }

  // J-3: Timepoint-combination filter applied to a pre-filtered index list.
  // Keeps only rows whose entity (defined by the grain's node_id_columns) appears
  // in the selected timepoints within at least one disease. Evaluated per-disease
  // — a backbone at 2mo in App and 6mo in Tau does NOT satisfy "all {2mo,6mo}".
  function _filterTimepointCombine(indices, f, d) {
    const required = f.timepointCombine || [];
    if (!required.length) return indices;
    const mode = f.timepointCombineMode || "all";
    const { gi, cols, contrastDis, contrastTp } = d;
    // Build entity key from the surviving node id columns (grain-specific).
    const nodeIdCols = (gi.node_id_columns || []).map(name => cols[name]).filter(Boolean);
    const B = (gi.gene_vocab && gi.gene_vocab.length) || 1;
    const entityKey = nodeIdCols.length
      ? (i) => { let k = 0; for (const c of nodeIdCols) k = k * B + c[i]; return k; }
      : (i) => 0;  // no node ids: treat all as one entity
    // Group by (entityKey → Map<disease, Set<timepoint>>) over the input indices.
    const entityDisTp = new Map();
    for (const i of indices) {
      const ek = entityKey(i);
      const cid = cols.contrastId[i];
      const dis = contrastDis[cid];
      const tp  = contrastTp[cid];
      let dm = entityDisTp.get(ek);
      if (!dm) { dm = new Map(); entityDisTp.set(ek, dm); }
      let tset = dm.get(dis);
      if (!tset) { tset = new Set(); dm.set(dis, tset); }
      tset.add(tp);
    }
    const req = new Set(required);
    const qualifies = (tpSet) => mode === "all"
      ? [...req].every(tp => tpSet.has(tp))
      : [...req].some(tp => tpSet.has(tp));
    // Identify entity keys that pass for at least one disease.
    const passing = new Set();
    for (const [ek, dm] of entityDisTp) {
      for (const tpSet of dm.values()) {
        if (qualifies(tpSet)) { passing.add(ek); break; }
      }
    }
    return indices.filter(i => passing.has(entityKey(i)));
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
    // pvalue may be absent for backbone grain indexes.
    const sP = (f.sliderP != null && cols.pvalue) ? Number(f.sliderP) : null;
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
    // J-3: trajBits may be absent in backbone grain indexes.
    let trendBit = -1;
    const trend = (window.TrendFilter && f.trend)
      ? TrendFilter.payloadLabel(f.trend) : "";
    if (trend) {
      const trajVocab = gi.traj_label_vocab;
      if (!trajVocab || !trajVocab.length) return { indices: [], total: 0 };
      const bi = trajVocab.indexOf(trend);
      if (bi >= 0) trendBit = bi; else return { indices: [], total: 0 };
    }

    // --- search tokens -> per-token membership masks (AND across tokens) -
    const tokens = (f.searchText || "").toLowerCase().split(/\s+/).filter(Boolean);
    const tokMasks = tokens.map(t => ({
      gene: _member(lc.gene, t),
      sender: _member(gi.sender_vocab, t, true),
      receiver: _member(gi.receiver_vocab, t, true),
      contrast: _member(gi.contrast_vocab, t, true),
    }));

    // --- single scan over the universe ----------------------------------
    const PDS = cols.PDS, PV = cols.pvalue;     // PV may be null for backbone grains
    const sid = cols.senderId, rid = cols.receiverId, cid = cols.contrastId;
    // J-3: node id columns may be absent for backbone grains (null-guarded below).
    const lig = cols.ligandId, rec = cols.receptorId, em = cols.emId, tgt = cols.targetId;
    const trj = cols.trajBits;                  // null for backbone grains; only accessed when trendBit >= 0
    const matched = d.scratch;
    let m = 0;
    for (let i = 0; i < N; i++) {
      if (sPds != null && !(Math.abs(PDS[i]) >= sPds)) continue;
      if (pdsSign === "up" && !(PDS[i] > 0)) continue;
      if (pdsSign === "down" && !(PDS[i] < 0)) continue;
      if (sP != null && !(PV[i] < sP)) continue;          // NaN pvalue excluded; sP=null when PV absent
      if (!contrastOk[cid[i]]) continue;
      if (senderOk && !senderOk[sid[i]]) continue;
      if (receiverOk && !receiverOk[rid[i]]) continue;
      if (lowSender && (lowSender[sid[i]] || lowReceiver[rid[i]])) continue;
      if (trendBit >= 0 && !((trj[i] >> trendBit) & 1)) continue;  // trj non-null here (guarded above)
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
          // J-3: absent node id columns resolve to index 0 (first gene in vocab).
          // Correct: backbone grains don't emit the dropped node so its gene is irrelevant.
          const lv = lig ? tm.gene[lig[i]] : false;
          const tv = tgt ? tm.gene[tgt[i]] : false;
          if (!(lv || tm.gene[rec[i]] || tm.gene[em[i]] || tv
                || tm.sender[sid[i]] || tm.receiver[rid[i]] || tm.contrast[cid[i]])) {
            ok = false; break;
          }
        }
        if (!ok) continue;
      }
      matched[m++] = i;
    }
    let total = m;

    // J-3: Timepoint-combination filter: applied post-scan, before cap so the
    // cap applies to qualifying entities rather than the pre-filter universe.
    let filteredMatched;
    if ((f.timepointCombine || []).length) {
      filteredMatched = _filterTimepointCombine(
        Array.from(matched.subarray(0, total)), f, d);
      total = filteredMatched.length;
    } else {
      filteredMatched = null;
    }
    // Resolve the final matched source (typed or plain array).
    const matchedArr = filteredMatched || matched;

    // --- order + cap -----------------------------------------------------
    const topLimit = opts.limit === "all"
      ? total
      : ([500, 1000, 5000].includes(Number(f.topLimit))
          ? Number(f.topLimit) : 500);
    const key = f.sortKey || "rank";
    const dir = f.sortDir || 1;

    // Default (rank): matchedArr is already in ABS(PDS)-desc order == rank asc.
    if (key === "rank") {
      const k = Math.min(total, topLimit);
      if (dir > 0) {
        const out = new Array(k);
        for (let t = 0; t < k; t++) out[t] = matchedArr[t];
        return { indices: out, total };
      }
      // rank desc: the last k rows, reversed.
      const out = new Array(k);
      for (let t = 0; t < k; t++) out[t] = matchedArr[total - 1 - t];
      return { indices: out, total };
    }

    // Custom sort: numeric key per matched row, NaN always last (legacy rule).
    const keyV = new Float64Array(total);
    const keyOf = _keyFn(d, key);
    for (let t = 0; t < total; t++) keyV[t] = keyOf(matchedArr[t]);
    // Wrap matchedArr as Int32Array-compatible for _selectTopK.
    const matchedI32 = filteredMatched
      ? new Int32Array(filteredMatched)
      : matched;
    const indices = _selectTopK(matchedI32, keyV, total, dir, topLimit);
    return { indices, total };
  }

  // Per-row numeric sort key for a column (vocab string cols -> locale rank).
  function _keyFn(d, key) {
    const { cols, rank } = d;
    if (key === "PDS") return i => cols.PDS[i];
    // J-3: pvalue absent for backbone grains → treat as NaN (falls to end).
    if (key === "pvalue") return cols.pvalue ? (i => cols.pvalue[i]) : (() => NaN);
    if (cols[key] && d.gi.score_columns.indexOf(key) >= 0) {
      const col = cols[key]; return i => _f16(col[i]);
    }
    if (key === "_sender") { const c = cols.senderId, r = rank.sender; return i => r[c[i]]; }
    if (key === "_receiver") { const c = cols.receiverId, r = rank.receiver; return i => r[c[i]]; }
    if (key === "contrast") { const c = cols.contrastId, r = rank.contrast; return i => r[c[i]]; }
    // J-3: missing node id columns (backbone grains) sort all rows equal (rank 0).
    if (key === "Ligand") { const c = cols.ligandId, r = rank.gene; return c ? (i => r[c[i]]) : (() => 0); }
    if (key === "Receptor") { const c = cols.receptorId, r = rank.gene; return i => r[c[i]]; }
    if (key === "EM") { const c = cols.emId, r = rank.gene; return i => r[c[i]]; }
    if (key === "Target") { const c = cols.targetId, r = rank.gene; return c ? (i => r[c[i]]) : (() => 0); }
    if (key === "Path") {
      // Composite locale-rank; absent node columns contribute 0.
      const g = rank.gene, B = rank.geneSpan;
      const L = cols.ligandId, R = cols.receptorId, E = cols.emId, T = cols.targetId;
      const lf = L ? (i => g[L[i]]) : () => 0;
      const tf = T ? (i => g[T[i]]) : () => 0;
      return i => ((lf(i) * B + g[R[i]]) * B + g[E[i]]) * B + tf(i);
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

  // J-3: label_states / traj_label_vocab absent for backbone grains.
  function _labelState(code) {
    const states = manifest() && manifest().label_states; // ["", "DEG", "prG"]
    if (!states) return "";
    return (code > 0 && code < states.length) ? states[code] : "";
  }

  function _decodeTraj(bits) {
    const vocab = manifest() && manifest().traj_label_vocab;
    if (!vocab) return "";
    const out = [];
    for (let i = 0; i < vocab.length; i++) if ((bits >> i) & 1) out.push(vocab[i]);
    return out.join(";");
  }

  // Hydrate one display row from the columns (decode ids/f16/bitfields). Called
  // only for the <=100 rows on the current page.
  // J-3: backbone grains omit ligandId/targetId/labelBits/pvalue/trajBits.
  // Dropped-node genes are null (renderer shows "—"); label/traj/pvalue are null.
  function materialize(i) {
    const d = _data;
    if (!d) return null;
    const { cols, gi } = d;
    const Ligand   = cols.ligandId  ? gi.gene_vocab[cols.ligandId[i]]  : null;
    const Receptor = gi.gene_vocab[cols.receptorId[i]];
    const EM       = gi.gene_vocab[cols.emId[i]];
    const Target   = cols.targetId  ? gi.gene_vocab[cols.targetId[i]]  : null;
    const lb       = cols.labelBits ? cols.labelBits[i] : 0;
    const pv       = cols.pvalue    ? cols.pvalue[i]    : null;
    const trj      = cols.trajBits  ? cols.trajBits[i]  : 0;
    const pathParts = [Ligand || "—", Receptor, EM, Target || "—"];
    const row = {
      rank: i + 1,                          // global rank == row position + 1
      _sender: gi.sender_vocab[cols.senderId[i]],
      _receiver: gi.receiver_vocab[cols.receiverId[i]],
      sender: gi.sender_vocab[cols.senderId[i]],
      receiver: gi.receiver_vocab[cols.receiverId[i]],
      Ligand, Receptor, EM, Target,
      Path: pathParts.join("|"),
      Ligand_label:   _labelState(lb & 3),
      Receptor_label: _labelState((lb >> 2) & 3),
      EM_label:       _labelState((lb >> 4) & 3),
      Target_label:   _labelState((lb >> 6) & 3),
      contrast: gi.contrast_vocab[cols.contrastId[i]],
      pvalue: (pv !== null && pv === pv) ? pv : null,
      PDS: cols.PDS[i],
      traj_labels: _decodeTraj(trj),
    };
    for (const sc of gi.score_columns) row[sc] = _f16(cols[sc][i]);
    // Backbone grains carry n_paths (distinct Full pathways collapsed into the
    // backbone row); Full grain has no such column.
    if (cols.n_paths) row.n_paths = cols.n_paths[i];
    return row;
  }

  function _cachePathRows(key, rows) {
    if (_pathRowsCache.has(key)) _pathRowsCache.delete(key);
    _pathRowsCache.set(key, rows);
    while (_pathRowsCache.size > _PATH_ROWS_CACHE_MAX) {
      _pathRowsCache.delete(_pathRowsCache.keys().next().value);
    }
    return rows;
  }

  // Fast path for the Incytr drawer score plot in Top mode. It avoids fetching
  // a full sender/receiver parquet shard when the global typed-array index is
  // already mapped for the table.
  // J-3: path is a "|"-joined string of the SURVIVING node genes (null nodes
  // represented as "—") — the same Path format emitted by materialize().
  function pathRows(ident) {
    const d = _data;
    if (!d || !ident) return [];
    const { cols, gi } = d;
    const sender = String(ident.sender || "");
    const receiver = String(ident.receiver || "");
    const parts = String(ident.path || "").split("|");
    if (!sender || !receiver || parts.length !== 4) return [];
    const key = `${sender}||${receiver}||${parts.join("|")}`;
    if (_pathRowsCache.has(key)) {
      const rows = _pathRowsCache.get(key);
      _pathRowsCache.delete(key);
      _pathRowsCache.set(key, rows);
      return rows;
    }
    const sidWant = gi.sender_vocab.indexOf(sender);
    const ridWant = gi.receiver_vocab.indexOf(receiver);
    if (sidWant < 0 || ridWant < 0) return _cachePathRows(key, []);

    // Resolve each node: "—" means absent from this grain (always matches).
    const recWant = gi.gene_vocab.indexOf(parts[1]);
    const emWant  = gi.gene_vocab.indexOf(parts[2]);
    if (recWant < 0 || emWant < 0) return _cachePathRows(key, []);

    const ligWant = (parts[0] === "—" || !cols.ligandId) ? -2 : gi.gene_vocab.indexOf(parts[0]);
    const tgtWant = (parts[3] === "—" || !cols.targetId) ? -2 : gi.gene_vocab.indexOf(parts[3]);
    if (ligWant === -1 || tgtWant === -1) return _cachePathRows(key, []);

    const out = [];
    const sid = cols.senderId, rid = cols.receiverId;
    const lig = cols.ligandId, rec = cols.receptorId, em = cols.emId, tgt = cols.targetId;
    for (let i = 0; i < d.nrows; i++) {
      if (sid[i] !== sidWant || rid[i] !== ridWant) continue;
      if (rec[i] !== recWant || em[i] !== emWant) continue;
      if (ligWant >= 0 && lig && lig[i] !== ligWant) continue;
      if (tgtWant >= 0 && tgt && tgt[i] !== tgtWant) continue;
      out.push(materialize(i));
    }
    return _cachePathRows(key, out);
  }

  // Enumerate the distinct (sender, receiver) pairs whose pathway matches `ident`
  // at the active grain — the read-only "Related cell-type pairs" lookup. Same
  // node resolution as pathRows (absent nodes match-any via the "—" sentinel or a
  // missing column), but the sender/receiver filter is dropped and contrast is
  // ignored, so a backbone row matches by its surviving nodes (spine semantics)
  // and a Full row matches by the exact 4-tuple. Each pair carries a count of the
  // full pathways it contributes: n_paths on backbone grains (one matching row per
  // pair, already aggregated), or 1 per matching row at Full grain. The parent
  // pair (ident.sender/receiver) is included — no sender/receiver filter.
  // Returns [[sender, receiver, nPathways], …].
  function pairsForPath(ident) {
    const d = _data;
    if (!d || !ident) return [];
    const { cols, gi } = d;
    const parts = String(ident.path || "").split("|");
    if (parts.length !== 4) return [];
    const recWant = gi.gene_vocab.indexOf(parts[1]);
    const emWant  = gi.gene_vocab.indexOf(parts[2]);
    if (recWant < 0 || emWant < 0) return [];
    const ligWant = (parts[0] === "—" || !cols.ligandId) ? -2 : gi.gene_vocab.indexOf(parts[0]);
    const tgtWant = (parts[3] === "—" || !cols.targetId) ? -2 : gi.gene_vocab.indexOf(parts[3]);
    if (ligWant === -1 || tgtWant === -1) return [];

    const sid = cols.senderId, rid = cols.receiverId;
    const lig = cols.ligandId, rec = cols.receptorId, em = cols.emId, tgt = cols.targetId;
    const np = cols.n_paths;   // backbone grains only; Full grain counts 1 per row
    const nRec = gi.receiver_vocab.length;
    const byPair = new Map();
    for (let i = 0; i < d.nrows; i++) {
      if (rec[i] !== recWant || em[i] !== emWant) continue;
      if (ligWant >= 0 && lig && lig[i] !== ligWant) continue;
      if (tgtWant >= 0 && tgt && tgt[i] !== tgtWant) continue;
      const k = sid[i] * nRec + rid[i];
      const add = np ? (np[i] || 0) : 1;
      const cur = byPair.get(k);
      if (cur) cur[2] += add;
      else byPair.set(k, [gi.sender_vocab[sid[i]], gi.receiver_vocab[rid[i]], add]);
    }
    return [...byPair.values()];
  }

  return { available, loaded, manifest, ensureLoaded, filterRank, materialize,
           pathRows, pairsForPath, reset };
})();
