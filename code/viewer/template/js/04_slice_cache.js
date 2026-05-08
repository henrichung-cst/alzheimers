// ---------------------------------------------------------------------------
// SliceCache — lazy loader for per-entity edge parquets.
// Backbone-bucket slices back the backbone-selection highlight in the kinase
// explorer; decomp-OLS shards back the Attribution drawer's per-cell evidence.
// LRU-capped to avoid unbounded memory.
// ---------------------------------------------------------------------------
const SliceCache = (function(){
  const ESR = PAYLOAD.edge_slice_ref || {};
  const BUCKET_SIZE = ESR.bucket_size || 256;
  const MAX = 16;                          // LRU cap (per side)
  const bCache = new Map();                // bucket_id -> rows[]

  function _lruTouch(cache, key, value){
    if (cache.has(key)) cache.delete(key);
    cache.set(key, value);
    while (cache.size > MAX) cache.delete(cache.keys().next().value);
  }

  async function _fetchParquet(url){
    let resp;
    try {
      resp = await fetch(url);
    } catch (e) {
      if (window.location.protocol === "file:") {
        throw new Error(
          "Browser blocked local sidecar fetches under file://. " +
          "Serve outputs/reports/unified_viewer over HTTP and open that URL."
        );
      }
      throw e;
    }
    if (!resp.ok) throw new Error(`fetch ${url} → ${resp.status}`);
    const buf = await resp.arrayBuffer();
    if (typeof hyparquet === "undefined") {
      throw new Error("parquet reader not loaded (hyparquet missing)");
    }
    return await hyparquet.parquetReadObjects({
      file: buf, compressors: hyparquet.compressors,
    });
  }

  async function loadBackboneBucket(backbone_id){
    const bkt = Math.floor(backbone_id / BUCKET_SIZE);
    if (bCache.has(bkt)) {
      const v = bCache.get(bkt); _lruTouch(bCache, bkt, v); return v;
    }
    const pad = String(bkt).padStart(3, "0");
    const url = `${ESR.backbone_url}${pad}.parquet`;
    const rows = await _fetchParquet(url);
    _lruTouch(bCache, bkt, rows);
    return rows;
  }

  async function backboneEdges(backbone_id){
    const rows = await loadBackboneBucket(backbone_id);
    return rows.filter(r => r.backbone_id === backbone_id);
  }

  // Decomp-OLS shards: per-kinase substrate-site OLS for every (contrast, wmb_class).
  const dCache = new Map();              // kinase_id -> rows[]
  const dPresent = new Set((ESR.present_decomp_ols_kinase_ids || []).map(Number));
  async function loadDecompOls(kinase_id){
    if (!dPresent.has(Number(kinase_id))) return [];
    if (dCache.has(kinase_id)) {
      const v = dCache.get(kinase_id); _lruTouch(dCache, kinase_id, v); return v;
    }
    if (!ESR.decomp_ols_url) return [];
    const pad = String(kinase_id).padStart(3, "0");
    const url = `${ESR.decomp_ols_url}${pad}.parquet`;
    const rows = await _fetchParquet(url);
    _lruTouch(dCache, kinase_id, rows);
    return rows;
  }

  return { loadBackboneBucket, backboneEdges, loadDecompOls,
           get backboneCacheSize(){ return bCache.size; },
           get decompOlsCacheSize(){ return dCache.size; } };
})();
window.SliceCache = SliceCache;

// ---------------------------------------------------------------------------
// Header wiring
// ---------------------------------------------------------------------------
