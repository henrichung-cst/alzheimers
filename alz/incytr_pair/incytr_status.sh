#!/usr/bin/env bash
# Live progress monitor for the 5xFAD pair-mode run (run_5xfad.sh).
#
#   bash alz/incytr_pair/incytr_status.sh            one snapshot, then exit
#   bash alz/incytr_pair/incytr_status.sh --watch    refresh every 15s (Ctrl-C to stop)
#   bash alz/incytr_pair/incytr_status.sh --watch 30  refresh every 30s
#
# Read-only: derives everything from artifacts already on disk — per-contrast
# .status files + parquet mtimes (accurate durations), the run log tail (live
# pair progress), and the systemd scope cgroup (memory). It adds no load to the
# run and never reads parquet data.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null)" 2>/dev/null || true

BASE="outputs/reports/incytr_pair_mode_5xfad"
TISSUES=(cortex hippocampus)
AGES=(3 6 9 12)
TOTAL=8

g() { command grep "$@"; }   # bypass the rg shadow (grep is aliased to rg here)

dur() { # seconds -> "Hh MMm" / "MMm"
  local s=${1:-0}; ((s < 0)) && s=0
  local h=$((s / 3600)) m=$(((s % 3600) / 60))
  ((h > 0)) && printf '%dh%02dm' "$h" "$m" || printf '%dm' "$m"
}
gib() { awk -v b="${1:-0}" 'BEGIN{ if(b<=0){print "—"} else printf "%.1fG", b/1073741824 }'; }
epoch() { date -d "$1" +%s 2>/dev/null || echo 0; }

snapshot() {
  local now; now=$(date +%s)
  local log; log=$(ls -t "$BASE"/run_*.log 2>/dev/null | head -1)
  local scope astate
  scope=$(systemctl --user list-units --type=scope --all --no-legend 2>/dev/null \
            | g -oE 'incytr-5xfad-[0-9_]+\.scope' | head -1)
  astate=$([ -n "$scope" ] && systemctl --user is-active "$scope" 2>/dev/null || echo gone)

  # Per-contrast START times from the SCORE log markers. The .status file is
  # overwritten with the DONE timestamp on completion, so it cannot give a start;
  # the log marker "=== <ts> [tissue] TG_Xmo vs WT_Xmo ===" is the reliable source.
  declare -A START=()
  if [ -n "$log" ]; then
    while read -r ts tt aa; do
      [ -n "$aa" ] && START["${tt}_${aa}"]=$(epoch "$ts")
    done < <(g -oE '=== [0-9T:+-]+ \[(cortex|hippocampus)\] TG_[0-9]+mo vs WT_[0-9]+mo' "$log" \
               | command sed -E 's/=== ([0-9T:+-]+) \[([a-z]+)\] TG_([0-9]+)mo.*/\1 \2 \3/')
  fi

  # Walk the 8 contrasts in run order; classify and time each.
  local done=0 cur_label="" cur_started=0 first_started=0 sum_done=0
  local -a rows=()
  for t in "${TISSUES[@]}"; do
    local cells; cells="$(printf '%-12s' "$t")"
    for a in "${AGES[@]}"; do
      local pq="$BASE/$t/wide/TG_${a}mo_WT_${a}mo_incytr_output.parquet"
      local started=${START["${t}_${a}"]:-0}
      ((started > 0 && (first_started == 0 || started < first_started))) && first_started=$started
      if [ -s "$pq" ]; then
        local m; m=$(stat -c %Y "$pq" 2>/dev/null || echo 0)
        local d=0; ((started > 0 && m > started)) && d=$((m - started))
        ((done++)); sum_done=$((sum_done + d))
        cells+=$(printf '  %2smo \033[32m✓\033[0m %-7s' "$a" "$(dur "$d")")
      elif ((started > 0)); then
        cur_label="$t TG_${a}mo vs WT_${a}mo"; cur_started=$started
        cells+=$(printf '  %2smo \033[33m▶\033[0m %-7s' "$a" "$(dur $((now - started)))")
      else
        cells+=$(printf '  %2smo \033[2m·\033[0m %-7s' "$a" "")
      fi
    done
    rows+=("$cells")
  done

  # Phase + current-contrast pair progress (from the log tail).
  local phase="SCORE"
  [ -n "$log" ] && g -q 'significance filter' "$log" && phase="FILTER (gate, in place)"
  local pairline pn pt ppct=""
  pairline=$([ -n "$log" ] && g -oE 'pair [0-9]+/[0-9]+' "$log" | tail -1)
  if [ -n "$pairline" ]; then
    pn=${pairline#pair }; pt=${pn#*/}; pn=${pn%/*}
    ((pt > 0)) && ppct=$((100 * pn / pt))
  fi

  # Memory of the scope cgroup.
  local memc=0 memmax=""
  if [ "$astate" = active ]; then
    memc=$(systemctl --user show "$scope" -p MemoryCurrent --value 2>/dev/null)
    [[ "$memc" =~ ^[0-9]+$ ]] || memc=0
    memmax=$(systemctl --user show "$scope" -p MemoryMax --value 2>/dev/null)
    [[ "$memmax" =~ ^[0-9]+$ ]] && memmax=$(gib "$memmax") || memmax="20G"
  fi

  # Rough ETA from average completed-contrast duration.
  local elapsed=0 eta=-1
  ((first_started > 0)) && elapsed=$((now - first_started))
  if ((done > 0)) && [ "$astate" = active ]; then
    local avg=$((sum_done / done)) cur_run=0
    ((cur_started > 0)) && cur_run=$((now - cur_started))
    eta=$((avg * (TOTAL - done) - cur_run)); ((eta < 0)) && eta=0
  fi

  # ── render ──────────────────────────────────────────────────────────────
  local state_disp
  case "$astate" in
    active) state_disp=$'\033[32mRUNNING\033[0m' ;;
    gone)   state_disp=$([ "$done" -eq "$TOTAL" ] && echo $'\033[32mDONE\033[0m' || echo $'\033[31mSTOPPED\033[0m') ;;
    *)      state_disp=$'\033[31m'"$astate"$'\033[0m' ;;
  esac
  printf '\033[1m incytr 5xFAD  pair-mode\033[0m      %b      elapsed %s\n' \
    "$state_disp" "$(dur "$elapsed")"
  printf ' ─────────────────────────────────────────────────────────────────\n'
  printf ' Phase: %s        scored %d/%d contrasts\n\n' "$phase" "$done" "$TOTAL"
  for r in "${rows[@]}"; do printf '   %b\n' "$r"; done
  printf '\n'
  if [ -n "$cur_label" ]; then
    printf ' Current: %s' "$cur_label"
    [ -n "$ppct" ] && printf '   pair %s/%s (%s%%)' "$pn" "$pt" "$ppct"
    printf '\n'
  fi
  [ "$astate" = active ] && printf ' Memory:  %s / %s cap\n' "$(gib "$memc")" "$memmax"
  if ((eta >= 0)); then
    printf ' ETA:     ~%s remaining → finish ~%s  (rough; avg/contrast)\n' \
      "$(dur "$eta")" "$(date -d "@$((now + eta))" '+%a %H:%M')"
  fi
  if [ "$astate" = gone ]; then
    local nwide
    nwide=$(find "$BASE" -path '*/wide/*_incytr_output.parquet' ! -path '*smoke*' 2>/dev/null | wc -l)
    printf ' Final:   wide=%s/%s parquets\n' "$nwide" "$TOTAL"
  fi
  [ -n "$log" ] && printf '\033[2m last: %s\033[0m\n' "$(tail -1 "$log" | cut -c1-78)"
}

if [ "${1:-}" = "--watch" ]; then
  secs="${2:-15}"
  trap 'printf "\033[?25h"; exit 0' INT TERM   # restore cursor on Ctrl-C
  printf '\033[?25l'                            # hide cursor
  while :; do printf '\033[H\033[2J'; snapshot; sleep "$secs"; done
else
  snapshot
fi
