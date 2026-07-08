#!/usr/bin/env bash
# Clean 5xFAD Incytr regeneration after the AcK/KGG residue-filter fix.
# See docs/plans/ack_kgg_residue_filter_rebuild.md.
#
# Run in a fresh tmux session from the repo root:
#   tmux new -s ptmfix
#   bash alz/incytr_pair/regeneration/rebuild_ack_kgg_fix.sh
#   # detach: Ctrl-b d   reattach: tmux attach -t ptmfix
#
# Sequence: preflight → backup resume-cache dirs → export-bulk → decompose →
# Incytr pair grid (multi-hour) → viewer → verify (hard gate) → deploy → cleanup.
# All heavy steps are memory-capped (systemd-run cgroup v2). Backups are kept
# until deploy succeeds so the run is rollback-safe.
#
# Env toggles:
#   DEPLOY=0        stop after verify (no S3 push); leaves backups in place
#   KEEP_BACKUP=1   keep .bak-<TS> dirs even after a successful deploy
#   MEM_PAIR=24G MEM_LIGHT=16G MEM_VIEWER=32G   memory caps
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PIXI="$(command -v pixi 2>/dev/null || echo "$HOME/.pixi/bin/pixi")"
[[ -x "$PIXI" ]] || { echo "ERROR: pixi not found at $PIXI" >&2; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
LOG="outputs/reports/ptm_fix_rebuild_${TS}.log"   # outside any backed-up dir
mkdir -p outputs/reports
exec > >(tee -a "$LOG") 2>&1

DEPLOY="${DEPLOY:-1}"
KEEP_BACKUP="${KEEP_BACKUP:-0}"
MEM_PAIR="${MEM_PAIR:-24G}"
MEM_LIGHT="${MEM_LIGHT:-16G}"
MEM_VIEWER="${MEM_VIEWER:-32G}"
MANIFEST="outputs/reports/.ptm_fix_backup_${TS}.manifest"

cap() {  # cap <mem> <unit-suffix> <cmd...>  — run under a memory-hard-capped scope
  local mem="$1" unit="$2"; shift 2
  systemd-run --user --scope -p MemoryMax="$mem" -p MemorySwapMax=0 \
    --unit "ptmfix-${unit}-${TS}" env CONDA_OVERRIDE_CUDA="" "$@"
}

echo "=== $(date -Is) PTM-fix rebuild start ==="
echo "  REPO=$REPO_ROOT  DEPLOY=$DEPLOY  KEEP_BACKUP=$KEEP_BACKUP"
echo "  MEM_PAIR=$MEM_PAIR MEM_LIGHT=$MEM_LIGHT MEM_VIEWER=$MEM_VIEWER"
echo "  Log=$LOG"

# --- 0. Preflight: the fix must be in place, else we'd regenerate stale numbers.
echo "=== $(date -Is) [0] preflight ==="
if ! grep -q 'site_aa == "K"' alz/cohorts/fivexfad/ingest.py; then
  echo "ERROR: residue filter not found in ingest.py — apply the fix first." >&2
  exit 1
fi
echo "  residue filter present."

# --- 1. Backup (rename) the resume-cache dirs. Moving them aside forces a full
#        rebuild AND keeps a rollback copy until deploy succeeds.
echo "=== $(date -Is) [1] backup resume-cache dirs -> .bak-${TS} ==="
: > "$MANIFEST"
BACKUP_TARGETS=(
  "outputs/reports/incytr_pair_mode_5xfad"
  "outputs/reports/unified_viewer/audit_sources/omics_trace_fivexfad_cortex"
  "outputs/reports/unified_viewer/audit_sources/omics_trace_fivexfad_hippocampus"
  "outputs/reports/unified_viewer/edge_slices/incytr_pathways_fivexfad_cortex"
  "outputs/reports/unified_viewer/edge_slices/incytr_pathways_fivexfad_hippocampus"
)
for d in "${BACKUP_TARGETS[@]}"; do
  if [[ -d "$d" ]]; then
    mv "$d" "${d}.bak-${TS}"
    echo "${d}.bak-${TS}" >> "$MANIFEST"
    echo "  moved $d -> ${d}.bak-${TS}"
  else
    echo "  (absent, skip) $d"
  fi
done
echo "  backup manifest: $MANIFEST"

# --- 2. Regenerate inputs (the fix lands in export-bulk).
echo "=== $(date -Is) [2] export-bulk + decompose ==="
cap "$MEM_LIGHT" export-bulk "$PIXI" run 5xfad-export-bulk
cap "$MEM_LIGHT" decompose   "$PIXI" run 5xfad-incytr-decompose

# --- 3. Incytr pair grid (multi-hour). Fresh: resume cache was moved aside.
echo "=== $(date -Is) [3] Incytr pair grid (channels=pr,ps,py,Ack,KGG) ==="
cap "$MEM_PAIR" pair bash "$REPO_ROOT/alz/incytr_pair/run_pair_mode_5xfad.sh"

# --- 4. Rebuild viewer (omics-trace shards + edge-slices + payload; self-verifies).
echo "=== $(date -Is) [4] viewer rebuild ==="
cap "$MEM_VIEWER" viewer "$PIXI" run viewer

# --- 5. Verify (hard gate) — the reported number must have dropped, and no
#        non-lysine PTM rows may survive in any AcK/KGG shard. Written to a temp
#        file (not piped through systemd-run, whose stdin forwarding is flaky).
echo "=== $(date -Is) [5] verify ==="
VERIFY_PY="$(mktemp --tmpdir ptmfix_verify_XXXX.py)"
trap 'rm -f "$VERIFY_PY"' EXIT
cat > "$VERIFY_PY" <<'PY'
import glob, re, sys
import pyarrow.parquet as pq, pyarrow.compute as pc
from collections import defaultdict

base = "outputs/reports/unified_viewer/audit_sources"
fails = []

# (a) no non-K sites in any acetyl/ubiq shard, both tissues
for t in ("cortex", "hippocampus"):
    for f in glob.glob(f"{base}/omics_trace_fivexfad_{t}/*.parquet"):
        tab = pq.read_table(f, columns=["layer", "site_id"])
        m = pc.or_(pc.equal(tab["layer"], "acetyl"), pc.equal(tab["layer"], "ubiq"))
        for sid in tab.filter(m).column("site_id").to_pylist():
            if not sid:
                continue
            mm = re.search(r"_([A-Z])\d+_M\d", sid)   # …_<AA><pos>_M<mult>_…
            if mm and mm.group(1) != "K":
                fails.append(f"non-K PTM site in {t}: {sid}")
                break

# (b) Calm2 ubiq TG_6mo gene-mean must be ~1599 (< 1800), 4 K sites
f = f"{base}/omics_trace_fivexfad_hippocampus/Excitatory_principal_neurons_in_the_hippocampal_dentate_gyrus.parquet"
tab = pq.read_table(f, columns=["layer", "gene_symbol", "site_id", "animal_id", "genotype", "timepoint", "value"])
m = pc.and_(pc.equal(tab["gene_symbol"], "Calm2"), pc.equal(tab["layer"], "ubiq"))
d = tab.filter(m).to_pydict()
sites = set(d["site_id"])
byan = defaultdict(list)
for i in range(len(d["value"])):
    if d["genotype"][i] == "TG" and d["timepoint"][i] == "6mo":
        byan[d["animal_id"][i]].append(d["value"][i])
per = {a: sum(v) / len(v) for a, v in byan.items()}
gm = sum(per.values()) / len(per) if per else float("nan")
print(f"  Calm2 ubiq hippo TG_6mo: {len(sites)} sites, gene-mean={gm:.2f} (was 2006.71)")
if len(sites) != 4:
    fails.append(f"Calm2 ubiq site count = {len(sites)}, expected 4 (K only)")
if not (1400 < gm < 1800):
    fails.append(f"Calm2 ubiq TG_6mo gene-mean {gm:.2f} not in (1400,1800)")

if fails:
    print("VERIFY FAILED:")
    for x in fails:
        print("   -", x)
    sys.exit(1)
print("  VERIFY OK")
PY
cap "$MEM_LIGHT" verify "$PIXI" run python "$VERIFY_PY"
echo "  verify passed."

# --- 6. Deploy (outward-facing: aws s3 sync --delete). Gated on verify above.
if [[ "$DEPLOY" == "1" ]]; then
  echo "=== $(date -Is) [6] deploy (aws s3 sync --delete) — Ctrl-C within 5s to abort ==="
  sleep 5
  "$PIXI" run deploy-viewer
  echo "  deployed."
else
  echo "=== [6] deploy SKIPPED (DEPLOY=0). Backups retained. ==="
  echo "  Eyeball, then deploy:  $PIXI run deploy-viewer"
  echo "  Backups:               $MANIFEST"
  exit 0
fi

# --- 7. Cleanup — only reached after a successful deploy.
if [[ "$KEEP_BACKUP" == "1" ]]; then
  echo "=== [7] cleanup SKIPPED (KEEP_BACKUP=1). Backups: $MANIFEST ==="
else
  echo "=== $(date -Is) [7] remove backups ==="
  while IFS= read -r d; do
    [[ -n "$d" && -d "$d" ]] && { rm -rf "$d"; echo "  removed $d"; }
  done < "$MANIFEST"
  rm -f "$MANIFEST"
fi

echo "=== $(date -Is) PTM-fix rebuild COMPLETE ==="
echo "  Hard-refresh the deployed viewer (Ctrl+Shift+R)."
