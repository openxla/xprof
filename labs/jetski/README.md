# XProf Labs Jetski Hermetic Python Runner (`//third_party/xprof/labs/jetski`)

This package provides the hermetic Python execution runner
(`xprof_py_runner.py`) invoked as a subprocess data dependency by the XProf
collector backend.

## Overview

*   **`xprof_py_runner.py`**: Executes analytical Python scripts in an isolated
    namespace (`np`, `pd`, `plt`, `session_id`, `use_spanner`,
    `read_only_mode`).
*   **Memory Quotas**: Applies `RLIMIT_AS` virtual memory capping (4GB) to
    prevent memory exhaustion.
*   **Headless Matplotlib**: Uses the `Agg` backend and formats open figures as
    single-line Base64 SVG strings.
*   **GKE Compatibility**: Sanitates stdout and traceback lines to eliminate
    newlines (`\n` -> `\\n`).

## Verification

To run the unit test suite:

```bash
bazel test //third_party/xprof/labs/jetski:xprof_py_runner_test
```
