#!/usr/bin/env bash
# Benchmark Bazel flag configurations for the real xprof pip package build.
#
# Entry point under test (shipped):
#   bazel run [FLAGS] plugin:build_pip_package
# Script: plugin/build_pip_package.sh  (OUTPUT_DIR default /tmp/profile-pip)
#
# Fairness: each attempt runs `bazel clean` (not --expunge) so object files
# are rebuilt while external repos stay warm. All attempts use the same clean
# level; see CLEAN_MODE. Disk cache is NOT enabled by default so later rows
# do not inherit unfair local hits (remote public_cache is an explicit axis).
#
# MEASURED BEST (2026-07-10 local macOS arm64, bazel clean between attempts):
#   DEVELOPER_DIR=/Library/Developer/CommandLineTools \
#   bazel run --xcode_version_config=//tools/xcode:host_xcodes \
#     --action_env=DEVELOPER_DIR=/Library/Developer/CommandLineTools \
#     --host_action_env=DEVELOPER_DIR=/Library/Developer/CommandLineTools \
#     --config=macos plugin:build_pip_package
#   Matrix: 1107s; confirm re-run: 1028s. Public cache variants were slower.
#
# Best settings discovered by a full matrix are written to BEST_FILE; full
# attempts append to LOG_FILE (TSV). Re-apply the winner with:
#   bazel run <best flags> plugin:build_pip_package
#
# Usage:
#   ./tools/benchmark_pip_package_build.sh              # full matrix + confirm
#   ./tools/benchmark_pip_package_build.sh --list        # print configs only
#   ./tools/benchmark_pip_package_build.sh --dry-run     # log path dry-run, no bazel
#   ./tools/benchmark_pip_package_build.sh --only NAME   # single config
#   ./tools/benchmark_pip_package_build.sh --skip-confirm
#   LOG_DIR=/path ./tools/benchmark_pip_package_build.sh
#
# Environment:
#   LOG_DIR           Directory for logs (default: tools/benchmark_results)
#   XPROF_OUTPUT_DIR  Pip package output dir (default: /tmp/profile-pip)
#   CLEAN_MODE        clean | clean_expunge | none  (default: clean)
#   BAZEL             Bazel binary (default: bazel)
#   CONFIRM_BEST      1 to re-run winner (default: 1)
#   EXTRA_BAZEL_FLAGS Extra flags appended to every attempt

set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-$ROOT/tools/benchmark_results}"
BAZEL="${BAZEL:-bazel}"
CLEAN_MODE="${CLEAN_MODE:-clean}"
XPROF_OUTPUT_DIR="${XPROF_OUTPUT_DIR:-/tmp/profile-pip}"
CONFIRM_BEST="${CONFIRM_BEST:-1}"
EXTRA_BAZEL_FLAGS="${EXTRA_BAZEL_FLAGS:-}"
TARGET="plugin:build_pip_package"

# Host defaults (macOS arm64 local). Always pass --config=macos on Darwin so
# dynamic_lookup / clang_local from .bazelrc apply.
HOST_OS="$(uname -s)"
NCPU="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
# Leave 1 core free for the system when jobs is high.
JOBS_MAX="$NCPU"
JOBS_HIGH=$(( NCPU > 2 ? NCPU - 1 : NCPU ))
JOBS_HALF=$(( NCPU / 2 ))
[[ "$JOBS_HALF" -lt 1 ]] && JOBS_HALF=1

mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/bazel_pip_benchmark_${STAMP}.log}"
BEST_FILE="${BEST_FILE:-$LOG_DIR/best_bazel_pip_settings.txt}"
TSV_FILE="${TSV_FILE:-$LOG_DIR/attempts_${STAMP}.tsv}"

# ---------------------------------------------------------------------------
# Candidate configurations (name|flags)
# Baseline is README-style public_cache, plus --config=macos on Darwin.
# Variants change one primary axis where possible.
# ---------------------------------------------------------------------------
declare -a CONFIG_NAMES=()
declare -a CONFIG_FLAGS=()

add_config() {
  CONFIG_NAMES+=("$1")
  CONFIG_FLAGS+=("$2")
}

# Required platform config for this host.
MAC_CFG=""
# On Darwin without a full Xcode.app, auto-detected xcode_config is empty and
# apple_support falls back to macosx10.11 (missing on modern CLT). Pin a local
# xcode_version_config and DEVELOPER_DIR so pip package builds succeed.
BASE_FLAGS=""
if [[ "$HOST_OS" == "Darwin" ]]; then
  MAC_CFG="--config=macos"
  export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
  BASE_FLAGS="--xcode_version_config=//tools/xcode:host_xcodes --action_env=DEVELOPER_DIR=${DEVELOPER_DIR} --host_action_env=DEVELOPER_DIR=${DEVELOPER_DIR}"
fi

# 1) Baseline: README default + macos (project-documented public_cache).
add_config "baseline_readme_public_cache" \
  "${MAC_CFG} --config=public_cache"

# 2) No remote cache: measure whether public_cache helps or hurts cold rebuild.
add_config "macos_no_remote_cache" \
  "${MAC_CFG}"

# 3) Max local parallelism with public cache.
add_config "public_cache_jobs_max" \
  "${MAC_CFG} --config=public_cache --jobs=${JOBS_MAX}"

# 4) Local spawn + workers (GameSave-inspired, local-only strategies).
add_config "public_cache_local_strategy" \
  "${MAC_CFG} --config=public_cache --jobs=${JOBS_HIGH} --spawn_strategy=local --strategy=CppCompile=worker,local --worker_max_instances=${JOBS_HIGH}"

# 5) Explicit fastbuild + strip (prefer compile speed over binary quality).
add_config "public_cache_fastbuild" \
  "${MAC_CFG} --config=public_cache --jobs=${JOBS_HIGH} --compilation_mode=fastbuild --strip=always"

# 6) Half jobs (control for thrashing / oversubscription).
add_config "public_cache_jobs_half" \
  "${MAC_CFG} --config=public_cache --jobs=${JOBS_HALF}"

# Optional 7th when not Darwin (keep matrix ≥5 either way).
if [[ "$HOST_OS" != "Darwin" ]]; then
  add_config "public_cache_only" "--config=public_cache --jobs=${JOBS_HIGH}"
fi

usage() {
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
}

log() {
  printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

log_raw() {
  # Append without tee (already have content).
  printf '%s\n' "$*" >>"$LOG_FILE"
}

do_clean() {
  case "$CLEAN_MODE" in
    clean)
      log "CLEAN: ${BAZEL} clean"
      "$BAZEL" clean
      ;;
    clean_expunge)
      log "CLEAN: ${BAZEL} clean --expunge"
      "$BAZEL" clean --expunge
      ;;
    none)
      log "CLEAN: none (skipped)"
      ;;
    *)
      echo "Unknown CLEAN_MODE=$CLEAN_MODE" >&2
      exit 2
      ;;
  esac
}

package_stats() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    local count size
    count="$(find "$dir" -type f 2>/dev/null | wc -l | tr -d ' ')"
    size="$(du -sh "$dir" 2>/dev/null | awk '{print $1}')"
    printf 'files=%s size=%s' "$count" "$size"
  else
    printf 'files=0 size=missing'
  fi
}

run_attempt() {
  local name="$1"
  local flags="$2"
  local kind="${3:-matrix}"  # matrix | confirm

  local extra=()
  # shellcheck disable=SC2206
  [[ -n "$EXTRA_BAZEL_FLAGS" ]] && extra=($EXTRA_BAZEL_FLAGS)

  # Merge host base flags (xcode_config etc.) with per-attempt flags.
  local merged
  merged="$(echo "${BASE_FLAGS:-} ${flags}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/  */ /g')"
  # shellcheck disable=SC2206
  local flag_arr=($merged)
  flags="$merged"

  log "============================================================"
  log "ATTEMPT kind=${kind} name=${name}"
  log "FLAGS: ${flags}${EXTRA_BAZEL_FLAGS:+ }${EXTRA_BAZEL_FLAGS}"
  log "TARGET: ${TARGET}"
  log "CLEAN_MODE: ${CLEAN_MODE}"
  log "XPROF_OUTPUT_DIR: ${XPROF_OUTPUT_DIR}"
  log "HOST: $(uname -srm) ncpu=${NCPU}"
  log "START_UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

  do_clean

  # Fresh package dir so "non-empty after success" is meaningful.
  rm -rf "${XPROF_OUTPUT_DIR}"
  mkdir -p "${XPROF_OUTPUT_DIR}"

  export XPROF_OUTPUT_DIR

  local start_epoch end_epoch duration exit_code
  start_epoch="$(date +%s)"
  set +e
  # Safe empty-array expansion under `set -u`
  cmd=("$BAZEL" run)
  if ((${#flag_arr[@]})); then cmd+=("${flag_arr[@]}"); fi
  if ((${#extra[@]})); then cmd+=("${extra[@]}"); fi
  cmd+=("${TARGET}")
  "${cmd[@]}" \
    > >(tee -a "${LOG_DIR}/build_${name}_${STAMP}.stdout.log") \
    2> >(tee -a "${LOG_DIR}/build_${name}_${STAMP}.stderr.log" >&2)
  exit_code=$?
  set -e
  end_epoch="$(date +%s)"
  duration=$(( end_epoch - start_epoch ))

  local pkg
  pkg="$(package_stats "$XPROF_OUTPUT_DIR")"

  log "END_UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  log "DURATION_SEC: ${duration}"
  log "EXIT_CODE: ${exit_code}"
  log "PACKAGE: ${pkg} path=${XPROF_OUTPUT_DIR}"
  log "RESULT: name=${name} exit=${exit_code} duration_sec=${duration} ${pkg}"

  # TSV: kind name duration exit flags package_stats
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$kind" "$name" "$duration" "$exit_code" "$flags" "$pkg" >>"$TSV_FILE"

  return "$exit_code"
}

write_best() {
  local best_name="$1"
  local best_flags="$2"
  local best_duration="$3"
  local confirm_duration="${4:-n/a}"
  local confirm_exit="${5:-n/a}"

  cat >"$BEST_FILE" <<EOF
# Best Bazel settings for xprof plugin:build_pip_package (local matrix)
# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)
# Host: $(uname -srm) ncpu=${NCPU}
# Clean level: ${CLEAN_MODE}
# Log: ${LOG_FILE}
# TSV: ${TSV_FILE}

BEST_NAME=${best_name}
BEST_DURATION_SEC=${best_duration}
CONFIRM_DURATION_SEC=${confirm_duration}
CONFIRM_EXIT=${confirm_exit}

# Re-run the winning configuration:
bazel run ${best_flags} ${TARGET}

# Or with explicit output dir:
# XPROF_OUTPUT_DIR=/tmp/profile-pip bazel run ${best_flags} ${TARGET}

FLAGS=${best_flags}
EOF
  log "Wrote best settings to ${BEST_FILE}"
}

ONLY=""
DRY_RUN=0
LIST_ONLY=0
SKIP_CONFIRM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --list) LIST_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --skip-confirm) SKIP_CONFIRM=1; shift ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$LIST_ONLY" -eq 1 ]]; then
  for i in "${!CONFIG_NAMES[@]}"; do
    printf '%s | %s\n' "${CONFIG_NAMES[$i]}" "${CONFIG_FLAGS[$i]}"
  done
  exit 0
fi

# Header
{
  echo "# xprof pip package Bazel build benchmark"
  echo "# stamp=${STAMP}"
  echo "# root=${ROOT}"
  echo "# target=${TARGET}"
  echo "# clean_mode=${CLEAN_MODE}"
  echo "# output_dir=${XPROF_OUTPUT_DIR}"
  echo "# ncpu=${NCPU} host=$(uname -srm)"
  echo "# GameSave-VCS .bazelrc was NOT copied blindly; candidates are xprof-relevant"
  echo "# axes: public_cache on/off, jobs, spawn strategy/workers, fastbuild"
  echo "# base_flags=${BASE_FLAGS:-}"
  echo "# developer_dir=${DEVELOPER_DIR:-}"
} | tee "$LOG_FILE"

printf 'kind\tname\tduration_sec\texit_code\tflags\tpackage\n' >"$TSV_FILE"

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "DRY_RUN: would clean (${CLEAN_MODE}) and run ${#CONFIG_NAMES[@]} configs"
  for i in "${!CONFIG_NAMES[@]}"; do
    log "DRY_RUN config: ${CONFIG_NAMES[$i]} => ${CONFIG_FLAGS[$i]}"
    printf 'dry\t%s\t0\t0\t%s\tfiles=0 size=n/a\n' \
      "${CONFIG_NAMES[$i]}" "${CONFIG_FLAGS[$i]}" >>"$TSV_FILE"
  done
  log "DRY_RUN log file: ${LOG_FILE}"
  log "DRY_RUN tsv file: ${TSV_FILE}"
  # Write a placeholder best so paths exist for tooling.
  write_best "dry_run" "${CONFIG_FLAGS[0]}" "0" "0" "0"
  exit 0
fi

declare -a OK_NAMES=()
declare -a OK_FLAGS=()
declare -a OK_DURATIONS=()

for i in "${!CONFIG_NAMES[@]}"; do
  name="${CONFIG_NAMES[$i]}"
  flags="${CONFIG_FLAGS[$i]}"
  if [[ -n "$ONLY" && "$name" != "$ONLY" ]]; then
    continue
  fi
  # Trailing whitespace trim for empty MAC_CFG cases
  flags="$(echo "$flags" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  if run_attempt "$name" "$flags" "matrix"; then
    # Re-read last TSV row (duration + full merged flags used for the run)
    last="$(tail -1 "$TSV_FILE")"
    dur="$(printf '%s\n' "$last" | cut -f3)"
    full_flags="$(printf '%s\n' "$last" | cut -f5)"
    OK_NAMES+=("$name")
    OK_FLAGS+=("$full_flags")
    OK_DURATIONS+=("$dur")
    log "SUCCESS recorded for ${name} duration=${dur}s"
  else
    log "FAILURE recorded for ${name} (still logged; cannot win best)"
  fi
done

if [[ ${#OK_NAMES[@]} -eq 0 ]]; then
  log "ERROR: no successful (exit 0) attempts; no best config"
  exit 1
fi

# Pick minimum duration among successes
best_i=0
best_d="${OK_DURATIONS[0]}"
for i in "${!OK_DURATIONS[@]}"; do
  if [[ "${OK_DURATIONS[$i]}" -lt "$best_d" ]]; then
    best_d="${OK_DURATIONS[$i]}"
    best_i="$i"
  fi
done

best_name="${OK_NAMES[$best_i]}"
best_flags="${OK_FLAGS[$best_i]}"
log "BEST (matrix): name=${best_name} duration_sec=${best_d} flags=${best_flags}"

confirm_d="n/a"
confirm_e="n/a"
if [[ "$SKIP_CONFIRM" -eq 1 || "$CONFIRM_BEST" != "1" ]]; then
  log "CONFIRM: skipped"
  write_best "$best_name" "$best_flags" "$best_d" "$confirm_d" "$confirm_e"
else
  if run_attempt "${best_name}" "$best_flags" "confirm"; then
    confirm_e=0
    confirm_d="$(tail -1 "$TSV_FILE" | cut -f3)"
    log "CONFIRM OK duration_sec=${confirm_d}"
  else
    confirm_e="$(tail -1 "$TSV_FILE" | cut -f4)"
    confirm_d="$(tail -1 "$TSV_FILE" | cut -f3)"
    log "CONFIRM FAILED exit=${confirm_e} duration_sec=${confirm_d}"
  fi
  write_best "$best_name" "$best_flags" "$best_d" "$confirm_d" "$confirm_e"
  if [[ "$confirm_e" != "0" ]]; then
    exit 1
  fi
fi

log "DONE. Ranking (successes only):"
for i in "${!OK_NAMES[@]}"; do
  log "  ${OK_DURATIONS[$i]}s  ${OK_NAMES[$i]}  ${OK_FLAGS[$i]}"
done
log "Full log: ${LOG_FILE}"
log "Best file: ${BEST_FILE}"
