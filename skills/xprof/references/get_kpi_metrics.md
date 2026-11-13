# `get_kpi_metrics` Reference

This reference explains how to get a consolidated set of Key Performance
Indicators (KPIs) for a specific XProf session. Use this to get a quick,
token-efficient overview of step time, duty cycle, roofline utilization,
MXU/device utilization, peak memory, and accelerator info.

## Prerequisites

-   You must have the log directory path (`<logdir>`), direct run directory, or
    session ID for the specific XProf run you are attempting to analyze.

## Instructions

1.  Use the `xprof` command-line tool, providing the log directory or session
    ID:

    ```bash
    xprof get_kpi_metrics <logdir>
    ```

    Or with explicit named flags:

    ```bash
    xprof get_kpi_metrics --logdir=<path_to_logdir> [--session_id=<run_name>]
    ```
2.  Review the JSON returned by the tool, which contains:

    -   `step_time_ms`
    -   `duty_cycle_percent`
    -   `mxu_utilization_percent`: from hardware performance counters. It is
        `null` when the counters were not captured. Then
        `mxu_utilization_status` (`status: "unavailable"`, `reason`,
        `how_to_get`) is also present.
    -   `compute_efficiency_percent`: program FLOP rate relative to the peak
        device FLOP rate, from the XLA cost model. Present when roofline data
        exists.
    -   `roofline_utilization`
    -   `peak_hbm_gib`
    -   `accelerator_info` (device type and core count)

3.  If `mxu_utilization_percent` is `null`, say that MXU utilization is unknown
    (not measured), not 0%. Use `compute_efficiency_percent` as a cost-model
    estimate instead, or run
    [`get_kernel_utilization`](get_kernel_utilization.md) for per-kernel values.
    See
    [Missing Hardware Counters](get_overview.md#missing-hardware-counters).

## Multi-Host Runs

On a multi-host session or a run directory with more than one trace,
`step_time_ms` takes the per-step maximum across all non-SparseCore cores on
all hosts (`ComputeStepTimeSummaryInMs`), not a per-host value or cross-host
mean. Both `capture` (for local multi-trace directories) and the KPI payload
include `"step_time_scope": "per_step_max_across_hosts"`, a warning naming
`step_time_ms`, and `per_host_step_time_ms` (mapping each host or trace stem to
its average step time in milliseconds from the owned `ALL_HOSTS.op_stats_v2.pb`
on local runs or `pod_viewer` step records as a fallback / on remote sessions):

-   For per-host step time, inspect `per_host_step_time_ms` or run once per
    host with `--host=<name>` (names from `capture.files_available`).
-   `xprof get_step_trace <run_dir>` averages per-core step durations across
    all hosts when `pod_viewer` step data is present, but its generic/GPU
    `input_pipeline_analyzer` fallback reads core 0 of the first host only.

A single-host selection (one file or `--host=<name>`) has no
`step_time_scope` field.

## Example Usage

If the user says: "What are the key metrics for /path/to/logdir?", you should:

1.  Run the CLI command:

    ```bash
    xprof get_kpi_metrics /path/to/logdir
    ```
2.  Summarize the KPIs to the user.
