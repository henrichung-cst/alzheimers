#!/usr/bin/env bash
# Read-only live monitor for T-cell pair-mode Incytr.
#
#   bash alz/incytr_pair/tcells_status.sh
#   bash alz/incytr_pair/tcells_status.sh --watch
#   bash alz/incytr_pair/tcells_status.sh --watch 20
#   bash alz/incytr_pair/tcells_status.sh --root outputs/reports/incytr_pair_mode_tcells
#
# The default root is the fresh per-cell rerun. The monitor never reads parquet
# contents, restarts work, or changes any analysis artifact.
set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

ROOT="${INCYTR_TCELLS_OUTPUT_ROOT:-outputs/reports/incytr_pair_mode_tcells_percell}"
WATCH_SECONDS=""

usage() {
  cat <<'EOF'
Usage: bash alz/incytr_pair/tcells_status.sh [--watch [seconds]] [--root path]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch)
      WATCH_SECONDS="${2:-15}"
      [[ "${2:-}" =~ ^[0-9]+$ ]] && shift
      ;;
    --root)
      ROOT="${2:?--root requires a path}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -n "$WATCH_SECONDS" && ! "$WATCH_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Watch interval must be a positive integer.\n' >&2
  exit 2
fi

if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
  GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; CYAN=$'\033[36m'
else
  BOLD=""; DIM=""; RESET=""; GREEN=""; YELLOW=""; RED=""; CYAN=""
fi

DONOR2_DAYS=(5 7 9 11)
DONOR1_DAYS=(13 17 20)
TOTAL=7

duration() {
  local seconds=${1:-0}
  ((seconds < 0)) && seconds=0
  local hours=$((seconds / 3600))
  local minutes=$(((seconds % 3600) / 60))
  if ((hours > 0)); then printf '%dh%02dm' "$hours" "$minutes"; else printf '%dm' "$minutes"; fi
}

epoch() { date -d "$1" +%s 2>/dev/null || echo 0; }

driver_active() {
  pgrep -f '[i]ncytr_commandline.R' >/dev/null 2>&1
}

contrast_start_epoch() {
  local donor=$1 later=$2 log=$3
  [[ -f "$log" ]] || { echo 0; return; }
  local timestamp
  timestamp=$(command grep -oE "=== [0-9T:+-]+ \[$donor\] d${later} vs d2" "$log" \
    | tail -1 | command sed -E 's/=== ([0-9T:+-]+).*/\1/')
  [[ -n "$timestamp" ]] && epoch "$timestamp" || echo 0
}

cell() {
  local donor=$1 later=$2 log=$3 now=$4
  local wide="$ROOT/$donor/wide"
  local parquet="$wide/d${later}_d2_incytr_output.parquet"
  local status="$wide/.status_d${later}_d2.txt"
  local started=0
  started=$(contrast_start_epoch "$donor" "$later" "$log")

  if [[ -s "$parquet" ]]; then
    local completed elapsed=0 done_timestamp=""
    if [[ -f "$status" ]]; then
      done_timestamp=$(command sed -nE 's/^done ([^ ]+)$/\1/p' "$status" | tail -1)
    fi
    if [[ -n "$done_timestamp" ]]; then
      completed=$(epoch "$done_timestamp")
    else
      completed=$(stat -c %Y "$parquet" 2>/dev/null || echo 0)
    fi
    ((started > 0 && completed > started)) && elapsed=$((completed - started))
    printf '%b' "${GREEN}✓${RESET} $(duration "$elapsed")"
  elif [[ -f "$status" ]] && command grep -q '^FAIL' "$status"; then
    printf '%b' "${RED}✕ failed${RESET}"
  elif [[ -f "$status" ]] && command grep -q '^started' "$status"; then
    if driver_active; then
      printf '%b' "${YELLOW}▶ $(duration $((now - started)))${RESET}"
    else
      printf '%b' "${RED}⊘ interrupted${RESET}"
    fi
  else
    printf '%b' "${DIM}· pending${RESET}"
  fi
}

snapshot() {
  local now log done=0 failed=0 interrupted=0
  now=$(date +%s)
  log="$ROOT/pair_run.log"

  local active=false
  driver_active && active=true
  for donor_and_days in "donor2:${DONOR2_DAYS[*]}" "donor1:${DONOR1_DAYS[*]}"; do
    local donor=${donor_and_days%%:*}
    local days=${donor_and_days#*:}
    for later in $days; do
      local wide="$ROOT/$donor/wide"
      local parquet="$wide/d${later}_d2_incytr_output.parquet"
      local status="$wide/.status_d${later}_d2.txt"
      if [[ -s "$parquet" ]]; then
        ((done++))
      elif [[ -f "$status" ]] && command grep -q '^FAIL' "$status"; then
        ((failed++))
      elif [[ -f "$status" ]] && command grep -q '^started' "$status" && [[ "$active" != true ]]; then
        ((interrupted++))
      fi
    done
  done

  local state
  if [[ "$active" == true ]]; then
    state="${GREEN}RUNNING${RESET}"
  elif ((done == TOTAL)); then
    state="${GREEN}DONE${RESET}"
  elif ((failed > 0 || interrupted > 0)); then
    state="${RED}STOPPED${RESET}"
  else
    state="${DIM}NOT STARTED${RESET}"
  fi

  printf '%b\n' "${BOLD} T-cell Incytr · per-cell pair mode${RESET}    $state"
  printf ' ─────────────────────────────────────────────────────────────\n'
  printf ' Root: %s\n' "$ROOT"
  printf ' Progress: %d/%d contrasts complete' "$done" "$TOTAL"
  ((failed > 0)) && printf '  %b' "${RED}${failed} failed${RESET}"
  ((interrupted > 0)) && printf '  %b' "${RED}${interrupted} interrupted${RESET}"
  printf '\n\n'

  local donor day
  for donor in donor2 donor1; do
    printf ' %-7s' "$donor"
    local -a days=("${DONOR2_DAYS[@]}")
    [[ "$donor" == donor1 ]] && days=("${DONOR1_DAYS[@]}")
    for day in "${days[@]}"; do
      printf '  d%-2s %-15b' "$day" "$(cell "$donor" "$day" "$log" "$now")"
    done
    printf '\n'
  done

  if [[ -f "$log" ]]; then
    local pair current
    pair=$(command grep -oE 'pair [0-9]+/[0-9]+' "$log" | tail -1)
    current=$(command grep -oE '=== [0-9T:+-]+ \[donor[12]\] d[0-9]+ vs d2' "$log" | tail -1)
    printf '\n'
    [[ -n "$current" ]] && printf ' Current: %s' "${current#*] }"
    [[ -n "$pair" ]] && printf '  %s' "$pair"
    printf '\n'
    printf '%b\n' "${DIM} Last: $(tail -1 "$log" | cut -c1-100)${RESET}"
  fi
}

if [[ -n "$WATCH_SECONDS" ]]; then
  trap 'printf "\033[?25h"; exit 0' INT TERM
  [[ -t 1 ]] && printf '\033[?25l'
  while :; do
    [[ -t 1 ]] && printf '\033[H\033[2J'
    snapshot
    sleep "$WATCH_SECONDS"
  done
else
  snapshot
fi
