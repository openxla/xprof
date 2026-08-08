# Bazel pip package build: local speed benchmark

**Date:** 2026-07-10  
**Host:** macOS Darwin arm64, 10 logical CPUs, ~64 GB RAM  
**Bazel:** 7.4.1 (via Bazelisk)  
**Target:** `plugin:build_pip_package` (real shipped entrypoint via `plugin/build_pip_package.sh`)  
**Clean level:** `bazel clean` between every attempt (object graph rebuilt; external repos retained; not `--expunge`)  
**Output:** `XPROF_OUTPUT_DIR=/tmp/profile-pip` (551M, 122 files after successful runs)

## Why this work

Cold rebuilds of the xprof pip package (C++/pywrap, frontend, WASM, convert) dominate local iteration. Flags from other projects (e.g. GameSave-VCS `.bazelrc`) were **not** copied blindly; configs were chosen for xprof’s `.bazelrc` surface and timed from a cleaned state on this machine.

## Host prerequisite (macOS CLT-only)

This machine has **Command Line Tools only** (no full `Xcode.app`). Auto-detected `local_config_xcode` is empty, so `apple_support` falls back to **macosx10.11**, which is missing on modern CLT SDKs. Every successful run therefore pins:

| Flag / env | Purpose |
|------------|---------|
| `--xcode_version_config=//tools/xcode:host_xcodes` | Local `xcode_config` with macOS SDK 15.5 |
| `DEVELOPER_DIR=/Library/Developer/CommandLineTools` | Point xcrun/clang at CLT |
| `--action_env=DEVELOPER_DIR=…` / `--host_action_env=…` | Propagate into sandboxed actions |
| `--config=macos` | Required linker/`clang_local` settings from repo `.bazelrc` |

Definition: [`tools/xcode/BUILD`](../tools/xcode/BUILD).

## Recommended (fastest measured) invocation

```bash
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
bazel run \
  --xcode_version_config=//tools/xcode:host_xcodes \
  --action_env=DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  --host_action_env=DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  --config=macos \
  plugin:build_pip_package
```

Optional output directory:

```bash
XPROF_OUTPUT_DIR=/tmp/profile-pip bazel run … plugin:build_pip_package
```

**Do not assume** README’s `--config=public_cache` is fastest for local cold rebuilds on every network; see results below.

## Results ranking (matrix, exit 0, ascending wall-clock)

All attempts used the CLT base flags above, then the variant flags listed. Times are wall-clock seconds for clean + full `bazel run plugin:build_pip_package`.

| Rank | Name | Duration (s) | Approx. | Variant flags (after base) | Exit | Package |
|------|------|-------------:|---------|----------------------------|------|---------|
| 1 | `macos_no_remote_cache` | **1107** | 18.5 min | `--config=macos` only | 0 | 122 files / 551M |
| 2 | `baseline_readme_public_cache` | 1211 | 20.2 min | `--config=macos --config=public_cache` | 0 | 122 files / 551M |
| 3 | `public_cache_local_strategy` | 1294 | 21.6 min | public_cache + `--jobs=9 --spawn_strategy=local --strategy=CppCompile=worker,local --worker_max_instances=9` | 0 | same |
| 4 | `public_cache_jobs_max` | 1326 | 22.1 min | public_cache + `--jobs=10` | 0 | same |
| 5 | `public_cache_fastbuild` | 1380 | 23.0 min | public_cache + `--jobs=9 --compilation_mode=fastbuild --strip=always` | 0 | same |
| 6 | `public_cache_jobs_half` | 2092 | 34.9 min | public_cache + `--jobs=5` | 0 | same |

### Confirmation re-run (best config, clean again)

| Kind | Name | Duration (s) | Exit |
|------|------|-------------:|------|
| confirm | `macos_no_remote_cache` | **1028** | 0 |

Confirm succeeded with the same flags; package non-empty after the win path.

## Findings

1. **Fastest cold rebuild:** local `--config=macos` **without** `--config=public_cache`.
2. **Public remote cache** (`https://storage.googleapis.com/xprof-bazel-cache/`) was reachable but **slower** than pure local for this target set on this network (lookup/miss overhead likely dominates).
3. **Higher `--jobs` with public_cache** did not beat no-remote; **half jobs** was worst (under-parallelism).
4. **Local spawn/workers** and explicit **fastbuild/strip** with public_cache did not beat no-remote either.
5. GameSave-style global defaults (disk cache, ultra jobs, Docker-only strategies) were not adopted; only xprof-valid, measured axes were kept.

## How to re-run the matrix

```bash
# Full matrix + confirm (writes tools/benchmark_results/)
./tools/benchmark_pip_package_build.sh

# List configs only
./tools/benchmark_pip_package_build.sh --list

# Dry-run logging paths (no bazel build)
LOG_DIR=/tmp/bench-dry ./tools/benchmark_pip_package_build.sh --dry-run
```

Unit tests for the driver (no full rebuild):

```bash
python3 tools/test_benchmark_pip_package_build.py -v
```

## Artifacts in-repo

| Path | Role |
|------|------|
| [`tools/benchmark_pip_package_build.sh`](../tools/benchmark_pip_package_build.sh) | Clean → run → log TSV/duration/exit/package stats; picks best; confirm re-run |
| [`tools/test_benchmark_pip_package_build.py`](../tools/test_benchmark_pip_package_build.py) | Structural tests for real entrypoint + dry-run logging |
| [`tools/xcode/BUILD`](../tools/xcode/BUILD) | CLT-compatible `xcode_config` |
| [`tools/benchmark_results/best_bazel_pip_settings.txt`](../tools/benchmark_results/best_bazel_pip_settings.txt) | Machine-generated best-flag note from the 2026-07-10 run |

Large per-attempt stdout/stderr logs are gitignored under `tools/benchmark_results/` (see root `.gitignore`).

## Relation to README

`README.md` documents:

```text
bazel run --config=public_cache plugin:build_pip_package
```

That remains a reasonable default for environments that benefit from the public cache (or CI). For **local macOS arm64 cold rebuilds** as measured here, prefer the no-remote command in [Recommended invocation](#recommended-fastest-measured-invocation), including the CLT xcode pins when full Xcode is not installed.
