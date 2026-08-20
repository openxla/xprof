---
name: xprof
description: >-
  Central entry point for ALL XProf operations and analyses. Use this skill first for any task involving XProf traces, performance, memory, HLO ops, collecting/triggering XProf profiles, or numerical correctness verification.
---

> ⚠️ **CRITICAL RULES** * **Path & Session Resolution**: Pass the log directory
> path, direct profile run folder, or individual `.xplane.pb` file directly as a
> positional argument (e.g. `xprof <subcommand> /path/to/logdir` or `xprof
> <subcommand> /path/to/logdir/plugins/profile/<run_name>`), or use explicit
> `--logdir=<path>` and `--session_id=<run_name>` flags. * **Timeline
> Coverage**: ALWAYS specify `--max_events=200000` (or `--max_events=-1`) when
> calling `list_xplane_events` to ensure complete trace capture across
> iterations. * **Asynchronous Execution**: **NEVER** block on long-running
> operations. Delegate to a subagent.

# xprof

This skill consolidates various tools related to XProf operations and analysis.

## CLI

The primary tool for interacting with XProf data is the XProf CLI bundled with
the `xprof` package.

```bash
xprof <subcommand> [flags]
```

## Discovery of Workflows

**CRITICAL for Agents**: Many advanced workflows (like diffing sessions,
numerical verification, or mapping architecture blocks) are documented in this
skill's markdown files but are NOT visible by running `xprof -h`.

-   **DO NOT rely solely on `xprof -h`** to discover capabilities.
-   **Always read the
    [Supported Capabilities](#supported-capabilities--references)** section
    below and the linked reference files to find complex analysis workflows.

## Referring to Sessions & Inputs

`xprof` tools accept trace inputs through multiple flexible formats:

1.  **Base Log Directory (Auto-Discovery)**: Pass the root directory where
    profiles are stored. `xprof` automatically discovers the latest profile run
    under `plugins/profile/`:

    ```bash
    xprof get_overview /path/to/logdir
    ```
2.  **Direct Run Directory**: Pass the direct session run folder containing the
    trace data:

    ```bash
    xprof get_overview /path/to/logdir/plugins/profile/2026_08_17_22_58_50
    ```
3.  **Direct Trace File**: Pass an individual `.xplane.pb` file directly:

    ```bash
    xprof get_overview /path/to/logdir/plugins/profile/2026_08_17_22_58_50/worker0.xplane.pb
    ```
4.  **Explicit Named Flags**: Pass explicit `--logdir` and `--session_id` flags:

    ```bash
    xprof get_overview --logdir=/path/to/logdir --session_id=2026_08_17_22_58_50
    ```

## Best Practices

-   **Asynchronous Execution**: Some operations (such as processing large traces
    or running complex queries) can take a long time. For these, consider
    spawning a subagent. However, for quick lookups like `get_hosts` or
    `get_overview`, execute the command directly.
-   **Trace Buffer Overflows**: If traces show dropped events or truncated
    steps, shorten the profiling duration (e.g. 2–5 steps instead of 20) or
    filter targeted event types using `--plane_regex` in timeline tools.
-   When running commands that get sent to the background as tasks, **DO NOT**
    attempt to guess the log file path or use `grep` manually to poll for
    completion.

## Workflows

### Bottleneck Analysis

When asked to find performance bottlenecks for a session:

1.  **Execute** `get_overview` to identify the high-level breakdown (Compute vs
    Host vs Communication).
2.  **Verify** if the workload is compute-bound, memory-bound, or host-bound
    using `get_device_information` hardware roofline constants.
3.  **Execute** `get_top_hlo_ops` or `get_hlo_op_profile` to find expensive
    operations if HLO profiles are available.
4.  **Execute** `list_xplane_events --max_events=200000` for detailed timeline
    attribution and step-time evaluation.
5.  **Inspect** HLO code using `list_hlo_modules` and `get_hlo_module_content`
    for suspect modules.
6.  **Report** findings directly to the user with concrete data points derived
    from the analysis.

<h2 id="supported-capabilities--references">Supported Capabilities & References</h2>

-   **[Get Graph Viewer Data](references/get_graph_viewer.md)**: Get graph
    viewer data (HLO text) and source line mappings from XProf.
-   **[Get Session Overview](references/get_overview.md)**: Get a comprehensive
    overview (Performance Summary, Run Environment) of an XProf session.
-   **[Get Memory Profile](references/get_memory_profile.md)**: Get a detailed
    memory profile analysis (Peak/device memory details) of an XProf session.
-   **[Get Peak Allocations](references/get_peak_allocations.md)**: Get HLO
    modules and buffers ordered by memory usage.
-   **[Get Top HLO Operations](references/get_top_hlo_ops.md)**: Identify top
    HLO operations by time, FLOPs, or bytes accessed.
-   **[Get KPI Metrics](references/get_kpi_metrics.md)**: Fetch consolidated KPI
    metrics (step time, duty cycle, MXU utilization, roofline) for a session.
-   **[Get HLO Neighborhood](references/get_hlo_neighborhood.md)**: Fetch the
    BFS neighborhood of an HLO instruction to identify fusion blockers.
-   **[Get Utilization Viewer](references/get_utilization_viewer.md)**: Fetch
    utilization metrics filtered by host, device, or node.
-   **[Analyze XLA Module Performance](references/analysis.md)**: Analyze XLA
    module performance, inspect HLO operations, and query timeline events.
-   **[Import Trace File](references/upload_trace.md)**: Import raw trace files
    into an xprof logdir for analysis.
-   **[Get Smart Suggestions](references/smart_suggestions.md)**: Dynamic
    bottleneck rules and static HLO optimization patterns (data types, einsum
    folding, layout alignment).
-   **[Diff Sessions](references/diff_session.md)**: Compare performance, kernel
    execution times, top operations, and HLO graphs between baseline and
    candidate sessions.
-   **[Numerical Verification](references/numerical_correctness.md)**: Verify
    numerical equivalence between baseline and candidate kernels using
    multi-regime distributions, discrete bounded indices, monotonic segment IDs,
    boolean masks, and ULP distance metrics.
