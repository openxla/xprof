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
        -   `mxu_utilization_percent`: Matrix unit utilization.
        -   `flop_rate_utilization_relative_to_roofline`: Flop rate relative to
            theoretical roofline.
        -   `device_idle_time_percent` & `host_idle_time_percent`: Idle time
            breakdown.
        -   `host_tf_op_percent` vs `device_tf_op_percent`: Host vs device
            execution share.
    -   `run_environment`: Hardware type (`device_type`), core count
        (`device_core_count`), host count, and training mode.

## Example Usage

If the user says: "What was the compute percent for session /path/to/logdir?",
you should:

1.  Run the CLI command:

    ```bash
    xprof get_overview /path/to/logdir
    ```
2.  Answer the user using that value.
