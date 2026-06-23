#!/usr/bin/env bash
# =============================================================================
# xprof_fast_build.sh — Fastest practical local XProf Bazel build from scratch
#
# Uses repo .bazelrc configs:
#   --config=macos   (platform / toolchain / Apple Silicon tuning)
#   --config=fast    (opt, local disk+repo cache, local spawn, high parallelism)
#
# Usage:
#   ./scripts/xprof_fast_build.sh                  # fast build (default target)
#   ./scripts/xprof_fast_build.sh --benchmark      # timed BEFORE (baseline) vs AFTER (fast)
#   ./scripts/xprof_fast_build.sh --target //foo   # custom target
#   ./scripts/xprof_fast_build.sh --warm           # second fast build (cache hit timing)
#   ./scripts/xprof_fast_build.sh --clean-cache    # wipe .bazel/disk_cache before build
#   ./scripts/xprof_fast_build.sh --no-macos       # skip --config=macos (Linux/CI)
#   ./scripts/xprof_fast_build.sh --full           # build //plugin:build_pip_package
#   BAZEL_JOBS=16 ./scripts/xprof_fast_build.sh
#
# Default target is intentionally smaller than the full pip package so cold
# builds finish in a useful amount of time; override with --target or --full.
# Compatible with macOS /bin/bash 3.2 (no mapfile/associative arrays required).
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[FAIL]${NC}  $*" >&2; exit 1; }
step()  { echo -e "\n${BOLD}${CYAN}━━━  $*  ━━━${NC}"; }

SCRIPT_START=$(date +%s)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Defaults ──────────────────────────────────────────────────────────────────
MODE="build"                 # build | benchmark
TARGET="//xprof/utils:function_registry"
WARM=0
CLEAN_CACHE=0
USE_MACOS=1
USE_PUBLIC_CACHE=0
EXTRA_BAZEL_ARGS=""
REPORT_FILE="${XPROF_BUILD_REPORT:-${REPO_ROOT}/.bazel/build_report.txt}"
LAST_RC=0
LAST_ELAPSED=0

# ── Args ──────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)      MODE="benchmark"; shift ;;
    --warm)           WARM=1; shift ;;
    --clean-cache)    CLEAN_CACHE=1; shift ;;
    --no-macos)       USE_MACOS=0; shift ;;
    --public-cache)   USE_PUBLIC_CACHE=1; shift ;;
    --full)
      TARGET="//plugin:build_pip_package"
      shift
      ;;
    --target)
      TARGET="${2:?--target requires a value}"
      shift 2
      ;;
    --report)
      REPORT_FILE="${2:?--report requires a path}"
      shift 2
      ;;
    --help|-h)
      sed -n '3,26p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS} $1"
      shift
      ;;
  esac
done

# ── Bazel binary ──────────────────────────────────────────────────────────────
if command -v bazelisk >/dev/null 2>&1; then
  BAZEL_BIN="${BAZEL_BIN:-bazelisk}"
elif command -v bazel >/dev/null 2>&1; then
  BAZEL_BIN="${BAZEL_BIN:-bazel}"
else
  die "Neither bazelisk nor bazel found on PATH"
fi

# Long-lived Bazel server + G1GC (DCodeX-style). Passed as startup flags because
# `startup:config` is not portable across Bazel versions.
export BAZEL_STARTUP_OPTS="${BAZEL_STARTUP_OPTS:---max_idle_secs=10800 --host_jvm_args=-XX:+UseG1GC --host_jvm_args=-Xms1g --host_jvm_args=-Xmx8g}"

# ── Host hardware → jobs / local_resources ────────────────────────────────────
detect_cpus() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    sysctl -n hw.ncpu
  else
    echo 4
  fi
}

detect_mem_mb() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo $(( $(sysctl -n hw.memsize) / 1024 / 1024 ))
  elif [[ -r /proc/meminfo ]]; then
    awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo
  else
    echo 8192
  fi
}

NCPUS="$(detect_cpus)"
MEM_MB="$(detect_mem_mb)"
DEFAULT_JOBS=$(( (NCPUS * 6) / 5 ))
if [ "${DEFAULT_JOBS}" -lt 4 ]; then DEFAULT_JOBS=4; fi
if [ "${DEFAULT_JOBS}" -gt 24 ]; then DEFAULT_JOBS=24; fi
BAZEL_JOBS="${BAZEL_JOBS:-$DEFAULT_JOBS}"

if [ "${BAZEL_JOBS}" -gt 2 ]; then
  LOCAL_CPU="${LOCAL_CPU:-$(( BAZEL_JOBS - 1 ))}"
else
  LOCAL_CPU="${LOCAL_CPU:-${BAZEL_JOBS}}"
fi

if [ "${MEM_MB}" -gt 12288 ]; then
  LOCAL_MEM_MB="${LOCAL_MEM_MB:-$(( MEM_MB - 8192 ))}"
else
  LOCAL_MEM_MB="${LOCAL_MEM_MB:-$(( MEM_MB * 3 / 4 ))}"
fi

mkdir -p "${REPO_ROOT}/.bazel/disk_cache" "${REPO_ROOT}/.bazel/repo_cache"
REPORT_DIR="$(dirname "${REPORT_FILE}")"
mkdir -p "${REPORT_DIR}"

if [ "${CLEAN_CACHE}" -eq 1 ]; then
  warn "Wiping local disk_cache + repo_cache under .bazel/"
  rm -rf "${REPO_ROOT}/.bazel/disk_cache" "${REPO_ROOT}/.bazel/repo_cache"
  mkdir -p "${REPO_ROOT}/.bazel/disk_cache" "${REPO_ROOT}/.bazel/repo_cache"
fi

# Space-separated flag strings (bash 3.2 friendly)
baseline_flags_str() {
  local flags=""
  if [ "${USE_MACOS}" -eq 1 ] && [ "$(uname -s)" = "Darwin" ]; then
    flags="--config=macos"
  fi
  echo "${flags}"
}

fast_flags_str() {
  local flags="--config=fast --jobs=${BAZEL_JOBS} --local_resources=cpu=${LOCAL_CPU} --local_resources=memory=${LOCAL_MEM_MB}"
  if [ "${USE_MACOS}" -eq 1 ]; then
    if [ "$(uname -s)" = "Darwin" ]; then
      flags="${flags} --config=macos --config=macos_fast_link"
    else
      warn "Not on Darwin; skipping --config=macos"
    fi
  fi
  if [ "${USE_PUBLIC_CACHE}" -eq 1 ]; then
    flags="${flags} --config=public_cache"
  fi
  echo "${flags}"
}

run_build() {
  local label="$1"
  local flags_str="$2"
  local start end elapsed logfile rc
  logfile="$(mktemp -t "xprof_bazel_${label}.XXXXXX")"
  # mktemp on macOS doesn't need .log suffix in template sometimes; append
  logfile="${logfile}.log"

  step "Build [${label}] target=${TARGET}"
  info "bazel: ${BAZEL_BIN}"
  info "flags: ${flags_str}${EXTRA_BAZEL_ARGS}"
  info "host:  cpus=${NCPUS} mem_mb=${MEM_MB} jobs=${BAZEL_JOBS} local_cpu=${LOCAL_CPU} local_mem_mb=${LOCAL_MEM_MB}"

  start=$(date +%s)
  set +e
  # shellcheck disable=SC2086
  # BAZEL_STARTUP_OPTS is honored by bazelisk/bazel for server JVM tuning.
  BAZEL_STARTUP_OPTS="${BAZEL_STARTUP_OPTS}" \
    "${BAZEL_BIN}" build ${flags_str} ${EXTRA_BAZEL_ARGS} "${TARGET}" 2>&1 | tee "${logfile}"
  rc=${PIPESTATUS[0]:-$?}
  set -e
  end=$(date +%s)
  elapsed=$(( end - start ))

  if [ "${rc}" -eq 0 ]; then
    ok "[${label}] succeeded in ${elapsed}s"
  else
    warn "[${label}] failed (exit ${rc}) after ${elapsed}s — log: ${logfile}"
  fi

  LAST_RC=$rc
  LAST_ELAPSED=$elapsed
  LAST_LOG=$logfile
  return "${rc}"
}

print_banner() {
  step "XProf fast build"
  info "repo:   ${REPO_ROOT}"
  info "target: ${TARGET}"
  info "mode:   ${MODE}"
  info "os:     $(uname -s) $(uname -m)"
}

write_report_header() {
  {
    echo "XProf fast build report"
    echo "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo: ${REPO_ROOT}"
    echo "git:  $(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "host: $(uname -s) $(uname -m) cpus=${NCPUS} mem_mb=${MEM_MB}"
    echo "target: ${TARGET}"
    echo "bazel: ${BAZEL_BIN} ($(${BAZEL_BIN} --version 2>/dev/null | head -1 || echo unknown))"
    echo ""
  } > "${REPORT_FILE}"
}

append_report_row() {
  local phase="$1" seconds="$2" rc="$3"
  echo "${phase}: ${seconds}s (exit=${rc})" >> "${REPORT_FILE}"
}

# ── Main ──────────────────────────────────────────────────────────────────────
print_banner
write_report_header

if [ "${MODE}" = "benchmark" ]; then
  step "BEFORE / AFTER benchmark"
  info "BEFORE = stock flags (macos only if on Darwin)"
  info "AFTER  = --config=fast + tuned jobs/resources (+ macos on Darwin)"
  echo ""

  "${BAZEL_BIN}" shutdown >/dev/null 2>&1 || true

  BASE_FLAGS="$(baseline_flags_str)"
  if run_build "BEFORE_baseline" "${BASE_FLAGS}"; then
    BEFORE_S=$LAST_ELAPSED
    BEFORE_RC=0
  else
    BEFORE_S=$LAST_ELAPSED
    BEFORE_RC=$LAST_RC
  fi
  append_report_row "BEFORE_baseline" "${BEFORE_S}" "${BEFORE_RC}"

  "${BAZEL_BIN}" shutdown >/dev/null 2>&1 || true

  FAST_FLAGS="$(fast_flags_str)"
  if run_build "AFTER_fast" "${FAST_FLAGS}"; then
    AFTER_S=$LAST_ELAPSED
    AFTER_RC=0
  else
    AFTER_S=$LAST_ELAPSED
    AFTER_RC=$LAST_RC
  fi
  append_report_row "AFTER_fast" "${AFTER_S}" "${AFTER_RC}"

  if [ "${WARM}" -eq 1 ]; then
    if run_build "AFTER_fast_warm" "${FAST_FLAGS}"; then
      WARM_S=$LAST_ELAPSED
      WARM_RC=0
    else
      WARM_S=$LAST_ELAPSED
      WARM_RC=$LAST_RC
    fi
    append_report_row "AFTER_fast_warm" "${WARM_S}" "${WARM_RC}"
  fi

  step "Results"
  echo ""
  echo "┌────────────────────┬──────────┬────────┐"
  echo "│ Phase              │ Seconds  │ Status │"
  echo "├────────────────────┼──────────┼────────┤"
  if [ "${BEFORE_RC}" -eq 0 ]; then BS="OK"; else BS="FAIL"; fi
  if [ "${AFTER_RC}" -eq 0 ]; then AS="OK"; else AS="FAIL"; fi
  printf "│ %-18s │ %8s │ %6s │\n" "BEFORE (baseline)" "${BEFORE_S}" "${BS}"
  printf "│ %-18s │ %8s │ %6s │\n" "AFTER  (fast)"     "${AFTER_S}"  "${AS}"
  if [ "${WARM}" -eq 1 ]; then
    if [ "${WARM_RC}" -eq 0 ]; then WS="OK"; else WS="FAIL"; fi
    printf "│ %-18s │ %8s │ %6s │\n" "AFTER  (fast+warm)" "${WARM_S}" "${WS}"
  fi
  echo "└────────────────────┴──────────┴────────┘"

  if [ "${BEFORE_RC}" -eq 0 ] && [ "${AFTER_RC}" -eq 0 ] && [ "${BEFORE_S}" -gt 0 ]; then
    DELTA=$(( AFTER_S - BEFORE_S ))
    PCT=$(( (DELTA * 100) / BEFORE_S ))
    if [ "${DELTA}" -lt 0 ]; then
      ok "AFTER is $(( -DELTA ))s faster (~$(( -PCT ))% reduction vs BEFORE)"
    elif [ "${DELTA}" -gt 0 ]; then
      warn "AFTER is ${DELTA}s slower (~${PCT}% increase vs BEFORE) — first fast run often fills disk_cache; rerun with --warm"
    else
      info "AFTER matched BEFORE wall time"
    fi
    echo "delta_seconds: ${DELTA}" >> "${REPORT_FILE}"
    echo "delta_percent: ${PCT}" >> "${REPORT_FILE}"
  fi

  echo ""
  info "Full report: ${REPORT_FILE}"
  TOTAL_ELAPSED=$(( $(date +%s) - SCRIPT_START ))
  info "Script total wall time: ${TOTAL_ELAPSED}s"
  echo "script_total_seconds: ${TOTAL_ELAPSED}" >> "${REPORT_FILE}"

  if [ "${BEFORE_RC}" -ne 0 ] || [ "${AFTER_RC}" -ne 0 ]; then
    exit 1
  fi
  exit 0
fi

# Single fast build (default)
FAST_FLAGS="$(fast_flags_str)"
run_build "fast" "${FAST_FLAGS}" || true
append_report_row "fast" "${LAST_ELAPSED}" "${LAST_RC}"
BUILD_RC=$LAST_RC

if [ "${WARM}" -eq 1 ]; then
  run_build "fast_warm" "${FAST_FLAGS}" || true
  append_report_row "fast_warm" "${LAST_ELAPSED}" "${LAST_RC}"
  if [ "${LAST_RC}" -ne 0 ]; then BUILD_RC=$LAST_RC; fi
fi

TOTAL_ELAPSED=$(( $(date +%s) - SCRIPT_START ))
echo "" >> "${REPORT_FILE}"
echo "script_total_seconds: ${TOTAL_ELAPSED}" >> "${REPORT_FILE}"
ok "Done in ${TOTAL_ELAPSED}s — report: ${REPORT_FILE}"
exit "${BUILD_RC}"
