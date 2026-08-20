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
    -   `mxu_utilization_percent`
    -   `roofline_utilization`
    -   `peak_hbm_gib`
    -   `accelerator_info` (device type and core count)

## Example Usage

If the user says: "What are the key metrics for /path/to/logdir?", you should:

1.  Run the CLI command:

    ```bash
    xprof get_kpi_metrics /path/to/logdir
    ```
2.  Summarize the KPIs to the user.
