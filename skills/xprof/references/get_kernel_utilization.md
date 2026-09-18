# `get_kernel_utilization` & `compute_utilization` Reference

Fetches hardware unit utilization metrics (MXU / XLU) for device kernels in an
XProf session. Both `get_kernel_utilization` and its alias `compute_utilization`
invoke the same underlying analysis.

## Prerequisites

-   You must have the log directory path (`<logdir>`), direct run directory, or
    session ID for the specific XProf run you are attempting to analyze.

## Instructions

Run the CLI command to retrieve per-kernel hardware compute unit utilization
records:

```bash
xprof get_kernel_utilization <logdir> [--kernel_name=<NAME>]
# Or equivalently:
xprof compute_utilization <logdir> [--kernel_name=<NAME>]
```

### Arguments

-   `<logdir>` or `--session_id`: The XProf log directory or session ID.
-   `--kernel_name` (optional): Filter results to a specific kernel by name or
    prefix.
-   `--duration_us` (optional): Override execution duration in microseconds for
    zero-duration hardware counter events.
-   `--force_duration` (optional): Force using `--duration_us` even if timeline
    duration is non-zero (default: `False`).
-   `--device` (optional): Specific device/core index to filter (e.g. `0`).
-   `--host` (optional): Specific host name to filter.
-   `--output_format` (optional): Output format, either `'json'` or `'text'`
    (default: `'json'`).
-   `--bypass_cache` (optional): Recompute metrics without reading from cache
    (default: `False`).

## Related Kernel & Step Timing Tools

-   `xprof get_kernel_stats <logdir>`: Returns raw kernel execution latencies,
    min/max/avg durations, and occurrence counts across device kernels.
-   `xprof get_avg_step_time <logdir>`: Computes the average step duration (in
    milliseconds) directly from the session's step trace or kernel statistics.

## Hardware Counter & Fallback Behavior

On TPU v7x and v6e, when hardware counter samples carry zero duration on the
counter timeline, `get_kernel_utilization` automatically recovers kernel
durations from the `PALLAS` / `XLA Ops` timeline planes or hardware cycle
counters (`UNPRIVILEGED_CYCLE_COUNT`). If hardware counter utilization planes
are completely absent from the trace, it automatically falls back to
`get_kernel_stats` to provide accurate kernel durations (`total_duration_us` and
`avg_duration_us`).
