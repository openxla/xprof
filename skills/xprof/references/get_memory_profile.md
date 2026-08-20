# `get_memory_profile` Reference

This reference provides a detailed memory profile analysis of an XProf session.
You should use this as your starting point for diagnosing memory issues, finding
peak device/HBM memory usage, heap allocations, stack reservations, free memory
bytes, and overall capacity.

## Prerequisites

-   You must have the log directory path (`<logdir>`), direct run directory, or
    session ID for the specific XProf run you are attempting to analyze.

## Instructions

1.  Use the `xprof` command-line tool, providing the log directory or session
    ID:

    ```bash
    xprof get_memory_profile <logdir>
    ```

    Or with explicit named flags:

    ```bash
    xprof get_memory_profile --logdir=<path_to_logdir> [--session_id=<run_name>]
    ```
2.  Review the JSON returned by the tool, which contains:

    -   `memory_capacity_gib`: Total memory capacity in GiB.
    -   `peak_memory_usage_gib`: Peak memory usage in GiB.
    -   `peak_usage_details`: Details including `heap_allocation_gib`,
        `stack_reservation_gib`, `free_memory_gib`, `fragmentation_percent`, and
        `utilization_percent`.

> [!NOTE] **Unmeasured Values**: A value of `-1.0` in any field indicates that
> the metric was unmeasured or unavailable in the collected trace data.

## Example Usage

If the user says: "What was the peak memory usage for /path/to/logdir?", you
should:

1.  Run the CLI command:

    ```bash
    xprof get_memory_profile /path/to/logdir
    ```
2.  Extract the `peak_memory_usage_gib` from the JSON response.
3.  Answer the user using that value.
