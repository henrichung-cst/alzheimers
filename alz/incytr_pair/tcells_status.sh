#!/usr/bin/env bash
# Read-only live monitor for the T-cell overnight kinase-MEA + Incytr run.
#
#   bash alz/incytr_pair/tcells_status.sh
#   bash alz/incytr_pair/tcells_status.sh --watch
#   bash alz/incytr_pair/tcells_status.sh --watch 20
#   bash alz/incytr_pair/tcells_status.sh --root outputs/reports/incytr_pair_mode_tcells
#
# The default root is the current per-cell-label rerun. Direct `pixi run
# tcells-incytr` executions are also supported. The monitor never reads parquet
# contents, restarts work, or changes any analysis artifact.
set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

ROOT="${INCYTR_TCELLS_OUTPUT_ROOT:-outputs/reports/incytr_pair_mode_tcells_percell_posneg}"
DEFAULT_ROOT="outputs/reports/incytr_pair_mode_tcells_percell_posneg"
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

ROOT_ABS=$(realpath -m "$ROOT")
DEFAULT_ROOT_ABS=$(realpath -m "$DEFAULT_ROOT")

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

process_for_root() {
  local pattern=$1 allow_default_without_override=$2 pid configured_root configured_abs
  while read -r pid; do
    [[ -r "/proc/$pid/environ" ]] || continue
    configured_root=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
      | command sed -n 's/^OUTPUT_DIR_OVERRIDE=//p' | tail -1)
    if [[ -n "$configured_root" ]]; then
      configured_abs=$(realpath -m "$configured_root")
      if [[ "$configured_abs" == "$ROOT_ABS" \
        || "$configured_abs" == "$ROOT_ABS/donor1/wide" \
        || "$configured_abs" == "$ROOT_ABS/donor2/wide" ]]; then
        return 0
      fi
    elif [[ "$allow_default_without_override" == true && "$ROOT_ABS" == "$DEFAULT_ROOT_ABS" ]]; then
      return 0
    fi
  done < <(pgrep -f "$pattern" 2>/dev/null)
  return 1
}

driver_active() {
  process_for_root '[i]ncytr_commandline.R|[r]un_pair_mode_tcells\.sh' true
}

overnight_active() {
  process_for_root '[r]un_backbone_overnight_tcells\.sh' true
}

latest_overnight_log() {
  local -a logs=("$ROOT"/overnight_*.log)
  [[ -e "${logs[0]}" ]] || return 0
  command ls -1t "${logs[@]}" 2>/dev/null | head -1
}

log_event_epoch() {
  local log=$1 event=$2 timestamp
  [[ -f "$log" ]] || { echo 0; return; }
  timestamp=$(command grep -F "$event" "$log" | tail -1 \
    | command sed -E 's/^=== ([^ ]+) .*/\1/')
  [[ -n "$timestamp" ]] && epoch "$timestamp" || echo 0
}

phase_status() {
  local started=$1 completed=$2 active=$3 now=$4
  if ((completed > 0)); then
    local elapsed=0
    ((started > 0 && completed > started)) && elapsed=$((completed - started))
    printf '%b' "${GREEN}✓ $(duration "$elapsed")${RESET}"
  elif [[ "$active" == true ]]; then
    if ((started > 0)); then
      printf '%b' "${YELLOW}▶ $(duration $((now - started)))${RESET}"
    else
      printf '%b' "${YELLOW}▶ running${RESET}"
    fi
  elif ((started > 0)); then
    printf '%b' "${RED}⊘ interrupted${RESET}"
  else
    printf '%b' "${DIM}· pending${RESET}"
  fi
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
  local now log overnight_log done=0 failed=0 interrupted=0
  now=$(date +%s)
  log="$ROOT/pair_run.log"
  overnight_log=$(latest_overnight_log)

  local incytr_active=false wrapper_active=false direct_incytr=false
  driver_active && incytr_active=true
  overnight_active && wrapper_active=true
  [[ "$incytr_active" == true && "$wrapper_active" != true ]] && direct_incytr=true
  if [[ "$wrapper_active" != true && -f "$log" && -n "$overnight_log" ]]; then
    local pair_log_mtime overnight_log_mtime
    pair_log_mtime=$(stat -c %Y "$log" 2>/dev/null || echo 0)
    overnight_log_mtime=$(stat -c %Y "$overnight_log" 2>/dev/null || echo 0)
    ((pair_log_mtime > overnight_log_mtime)) && direct_incytr=true
  fi
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
      elif [[ -f "$status" ]] && command grep -q '^started' "$status" && [[ "$incytr_active" != true ]]; then
        ((interrupted++))
      fi
    done
  done

  local overnight_started=false overnight_done=false incytr_stage_started=false
  local phase_started=0 phase_completed=0 preflight_started=0 preflight_completed=0
  local mea_started=0 mea_completed=0 incytr_started=0 incytr_completed=0
  if [[ -n "$overnight_log" && "$direct_incytr" != true ]]; then
    command grep -q 'overnight t-cell run start' "$overnight_log" && overnight_started=true
    command grep -q 'T-CELL OVERNIGHT RUN COMPLETE' "$overnight_log" && overnight_done=true
    command grep -q '\[2/2\] pair-mode' "$overnight_log" && incytr_stage_started=true
    phase_started=$(log_event_epoch "$overnight_log" '[1/2] donor1 projected-state kinase MEA')
    phase_completed=$(log_event_epoch "$overnight_log" '[1/2] done')
    preflight_started=$(log_event_epoch "$overnight_log" '[1/2] kinase preflight start')
    preflight_completed=$(log_event_epoch "$overnight_log" '[1/2] kinase preflight done')
    mea_started=$(log_event_epoch "$overnight_log" '[1/2] kinase MEA start')
    mea_completed=$(log_event_epoch "$overnight_log" '[1/2] kinase MEA done')
    incytr_started=$(log_event_epoch "$overnight_log" '[2/2] pair-mode')
    incytr_completed=$(log_event_epoch "$overnight_log" '[2/2] done')

    # Logs created before explicit substage markers still get a useful summary.
    if ((phase_completed > 0 && preflight_completed == 0)); then
      preflight_completed=$phase_completed
      mea_started=$phase_started
      mea_completed=$phase_completed
    elif ((phase_started > 0 && preflight_started == 0)); then
      mea_started=$phase_started
    fi
  fi

  local any_active=false
  [[ "$wrapper_active" == true || "$incytr_active" == true ]] \
    && any_active=true
  local state
  if [[ "$any_active" == true ]]; then
    state="${GREEN}RUNNING${RESET}"
  elif [[ "$overnight_done" == true ]] || ((done == TOTAL)); then
    state="${GREEN}DONE${RESET}"
  elif [[ "$overnight_started" == true ]]; then
    state="${RED}STOPPED${RESET}"
  elif ((failed > 0 || interrupted > 0)); then
    state="${RED}STOPPED${RESET}"
  else
    state="${DIM}NOT STARTED${RESET}"
  fi

  printf '%b\n' "${BOLD} T-cell rerun · kinase MEA → Incytr${RESET}    $state"
  printf ' ─────────────────────────────────────────────────────────────\n'
  printf ' Root: %s\n' "$ROOT"
  [[ -n "$overnight_log" && "$direct_incytr" != true ]] && printf ' Run log: %s\n' "$overnight_log"

  local preflight_active=false mea_phase_active=false incytr_phase_active=false
  [[ "$wrapper_active" == true && "$preflight_completed" -eq 0 ]] && preflight_active=true
  [[ "$wrapper_active" == true && "$preflight_completed" -gt 0 && "$mea_completed" -eq 0 ]] \
    && mea_phase_active=true
  [[ "$incytr_active" == true ]] && incytr_phase_active=true

  printf '\n Pipeline\n'
  if [[ "$direct_incytr" == true ]]; then
    printf ' %-13s %b\n' 'Preflight' "${DIM}· not scheduled${RESET}"
    printf ' %-13s %b\n' 'Kinase MEA' "${DIM}· not scheduled${RESET}"
  else
    printf ' %-13s %b\n' 'Preflight' "$(phase_status "$preflight_started" "$preflight_completed" "$preflight_active" "$now")"
    printf ' %-13s %b\n' 'Kinase MEA' "$(phase_status "$mea_started" "$mea_completed" "$mea_phase_active" "$now")"
  fi

  local incytr_stage_status
  if ((done == TOTAL)); then
    if ((incytr_completed > 0)); then
      incytr_stage_status=$(phase_status "$incytr_started" "$incytr_completed" false "$now")
    else
      incytr_stage_status="${GREEN}✓ complete${RESET}"
    fi
  elif [[ "$incytr_phase_active" == true ]]; then
    incytr_stage_status=$(phase_status "$incytr_started" 0 true "$now")
  elif [[ "$incytr_stage_started" == true ]] || ((done > 0 || failed > 0 || interrupted > 0)); then
    incytr_stage_status="${RED}⊘ interrupted${RESET}"
  else
    incytr_stage_status="${DIM}· pending${RESET}"
  fi
  printf ' %-13s %b\n' 'Incytr' "$incytr_stage_status"

  printf '\n Incytr progress: %d/%d contrasts complete' "$done" "$TOTAL"
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
  fi

  local activity_log="$overnight_log"
  [[ -z "$activity_log" || "$direct_incytr" == true ]] && activity_log="$log"
  if [[ -f "$activity_log" ]]; then
    printf '%b\n' "${DIM} Last: $(tail -1 "$activity_log" | cut -c1-100)${RESET}"
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
