# `get_overview` Reference

This reference documentation explains how to get a comprehensive overview of an
XProf session. Use this as a starting point for diagnosing performance issues,
evaluating performance summary statistics, or retrieving environmental context
about a run.

## Prerequisites

-   You must have the log directory path (`<logdir>`), direct run directory, or
    session ID for the specific XProf run you are attempting to analyze.

## Instructions

1.  Use the `xprof` command-line tool, providing the log directory or session
    ID:

    ```bash
    xprof get_overview <logdir>
    ```

    Or with explicit named flags:

    ```bash
    xprof get_overview --logdir=<path_to_logdir> [--session_id=<run_name>] [--include_command]
    ```
2.  Review the JSON returned by the tool, which contains:

    -   `performance_summary`: Key performance metrics including:
        -   `steptime_ms_average`: Average step execution time.
        -   `device_duty_cycle_percent`: Percentage of time accelerator is
            actively computing.
        -   `mxu_utilization_percent`: Matrix unit utilization from hardware
            performance counters. It is `null` when the counters were not
            captured. See
            [Missing Hardware Counters](#missing-hardware-counters).
        -   `compute_efficiency_percent`: Program FLOP rate relative to the peak
            device FLOP rate, from the XLA cost model. Present when roofline
            data exists.
        -   `flop_rate_utilization_relative_to_roofline`: Flop rate relative to
            theoretical roofline.
        -   `device_idle_time_percent` & `host_idle_time_percent`: Idle time
            breakdown.
        -   `host_tf_op_percent` vs `device_tf_op_percent`: Host vs device
            execution share.
    -   `run_environment`: Hardware type (`device_type`), core count
        (`device_core_count`), host count, and training mode.

## Missing Hardware Counters

When the capture has no hardware performance counters (for example, a TPU
profile captured without `tpu_perf_counters=True`), `mxu_utilization_percent`
is `null` and `performance_summary.mxu_utilization_status` has `status:
"unavailable"`, a `reason` and a `how_to_get` hint. In that case:

-   Say that MXU utilization is unknown (not measured). Do not report it as 0%.
    A `null` value does not mean that the MXU was idle.
-   Use `compute_efficiency_percent` as the compute estimate, and say that it
    comes from the XLA cost model, not from a hardware counter.
-   For per-kernel MXU values, run
    [`get_kernel_utilization`](get_kernel_utilization.md). It also needs
    counter data, so it can return no records on the same capture.
-   To measure MXU, capture the profile again with counters enabled.

## Multi-Host Runs

On a multi-host session or a run directory with more than one trace, the
step-time fields (`steptime_ms_average`, `steptime_ms_standard_deviation`) take
the per-step maximum across all non-SparseCore cores on all hosts
(`ComputeStepTimeSummaryInMs`), while TPU per-step breakdown averages such as
`tc_idle_ms_average` use the per-step minimum across all cores on all hosts
(`ComputeTpuPerStepDataAcrossCores`). When one host is a CPU straggler and
another blocks inside a collective on device, the merged summary can pair the
longest device step duration with the shortest device idle duration and mask
the straggler.

On multi-host runs, `capture` (for local multi-trace directories) and
`performance_summary` include:

```json
"step_time_scope": "per_step_max_across_hosts"
```

along with a warning and `performance_summary.per_host_step_time_ms` (mapping
each host or trace stem to its own `steptime_ms_average`,
`steptime_ms_standard_deviation`, `steptime_ms_minimum`, and
`steptime_ms_maximum` from the owned `ALL_HOSTS.op_stats_v2.pb` on local runs or
`pod_viewer` step records as a fallback / on remote sessions):

-   For per-host step time and breakdown metrics, inspect
    `performance_summary.per_host_step_time_ms` or run once per host with
    `--host=<name>` (names from `capture.files_available`).
-   `xprof get_step_trace <run_dir>` averages per-core step durations across
    all hosts when `pod_viewer` step data is present, but its generic/GPU
    `input_pipeline_analyzer` fallback reads core 0 of the first host only.

A single-host selection (one file or `--host=<name>`) has no
`step_time_scope` field.

## Example Usage

If the user says: "What was the compute percent for session /path/to/logdir?",
you should:

1.  Run the CLI command:

    ```bash
    xprof get_overview /path/to/logdir
    ```
2.  Answer the user using that value.

## Related Summary Tool (`get_profile_summary`)

To fetch a concise JSON summary containing top-level session metadata, step time
breakdowns, and summary text (`summary_text`) without the full overview tables:

```bash
xprof get_profile_summary <logdir>
```
