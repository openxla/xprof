# XProf Analysis Reference

This reference provides a comprehensive guide for analyzing XLA module
performance, HLO operations, and timeline events from XProf trace sessions.

## 1. Extract Module & Kernel Step Times

You can extract step times (such as `jit_update`, `jit_prefill`, and
`jit_generate`) directly using the `xprof` CLI on any logdir or trace file:

```bash
# Get overall execution breakdown and step time
xprof get_overview /path/to/logdir

# Get detailed kernel execution statistics and step durations
xprof get_kernel_stats /path/to/logdir --include_summary=True

# Filter statistics for a specific JIT module
xprof get_kernel_stats /path/to/logdir --kernel_name="jit_update"
```

### Common XLA Modules

Module         | Description
:------------- | :-----------------------------------
`jit_update`   | Training step time
`jit_prefill`  | Prefill / prompt processing time
`jit_generate` | Autoregressive token generation time

--------------------------------------------------------------------------------

## 2. HLO Module & Op Analysis

Use the HLO inspection tools in `xprof` to discover compiled modules, inspect
top operations by execution time, and navigate graph neighborhoods.

```bash
# List all HLO modules compiled in the trace session
xprof list_hlo_modules /path/to/logdir

# Get the top N most expensive HLO operations by self-time
xprof get_hlo_op_profile /path/to/logdir --top_n=15

# Get the full HLO code for a specific module
xprof get_hlo_module_content /path/to/logdir --module_name=<module_name>

# Get HLO code with Python source mapping metadata
xprof get_hlo_module_content /path/to/logdir --module_name=<module_name> --print_metadata=True

# Inspect the neighborhood of a specific HLO instruction (e.g., fusion.123)
xprof get_hlo_neighborhood /path/to/logdir --instruction_name=<instr_name> --radius=2

# Retrieve the full HLO module content as raw text
xprof get_hlo_text /path/to/logdir --module_name=<module_name>

# Retrieve the focused HLO neighborhood of a specific operation
xprof get_hlo_text /path/to/logdir --module_name=<module_name> --op_name=<op_name>

# Save HLO text directly to a local file
xprof get_hlo_text /path/to/logdir --module_name=<module_name> --path=/tmp/module.hlo
```

--------------------------------------------------------------------------------

## 3. Timeline Event Analysis (XPlane)

Query and aggregate low-level events across hardware and host timelines:

```bash
# Search for specific events in the XPlane timeline (e.g., all Fusion events on device planes)
xprof list_xplane_events /path/to/logdir \
  --plane_regex="Device.*" --event_regex="Fusion.*" --max_events=200000

# Aggregate statistics for matching timeline events
xprof aggregate_xplane_events /path/to/logdir \
  --plane_regex="Device.*" --event_regex="Fusion.*"
```

> [!TIP] **Timeline Query Efficiency**: If you already know the target event
> name (e.g., `'Fusion'`), do not run `list_xplane_events` first to locate
> instances. Run `aggregate_xplane_events` directly to compute event statistics
> in a single step and prevent context bloat.

--------------------------------------------------------------------------------

## 4. Analysis Workflows & Best Practices

### Bottleneck Analysis Workflow

1.  **Overview Check:** Run `get_overview` to check the high-level breakdown
    (Compute vs. Host vs. Communication).
2.  **Verify Roofline & KPIs:** Run `get_kpi_metrics` to inspect compute
    utilization, memory bandwidth, and step times.
3.  **Find Expensive Ops:** If compute-bound, run `get_hlo_op_profile` (see
    [get_hlo_op_profile](get_hlo_op_profile.md)) with `--view=category` for
    macro category breakdown, then drill down into specific categories with
    `--category='<name>'` (or `get_top_hlo_ops` for top leaf lists).
4.  **Inspect HLO Neighborhoods:** Run `get_hlo_neighborhood` around expensive
    fusions to diagnose layout transformations, copies, or fusion blockers.
5.  **Root-Cause Debugging:** Run `get_graph_viewer` with
    `--module_name=<module_name>` to trace compiled instructions back to exact
    Python source line numbers.

### Core Rules for Agents

*   **Rule 1: Discover Modules Before Querying**
    *   **ALWAYS** run `list_hlo_modules` first to discover the exact module
        name (e.g., `jit_convert_element_type(5275733382363401132)`). Module
        names often contain unique hash suffixes.
*   **Rule 2: Avoid Full HLO Dump Loops on Truncated Files**
    *   Large HLO modules are truncated to prevent context window bloat and
        latency.
    *   **DO NOT** attempt full HLO text dumps of large modules unless
        explicitly needed. Use `get_hlo_neighborhood` with `--instruction_name`
        to inspect target operations directly.
*   **Rule 3: Prefer Clean Text for Static Analysis**
    *   Use `get_hlo_module_content` when reviewing compiled HLO graphs or
        tensor layout definitions.
    *   Use `get_graph_viewer` when you need source-to-HLO line mappings
        (`FileLocations` and `StackFrames`).
*   **Rule 4: Prefer Graph Viewer for Root-Cause Debugging**
    *   Use `get_graph_viewer` with `--module_name=<name>` to trace an
        instruction back to the Python source file and function that generated
        it.
*   **Rule 5: Minimize CLI Invocations to Prevent Timeouts**
    *   Do not query tools in circular loops or repeatedly list modules if you
        already have the names. Jump directly to `get_hlo_op_profile` or target
        operations.

--------------------------------------------------------------------------------

## 5. Concepts: XSpace, XPlane, XLine, and Events

A single XProf session captures structured hierarchy across hosts and devices:

*   **XSpace**: A single `XSpace` proto represents all profiling data for the
    trace session.
*   **XPlane**: An `XSpace` contains multiple `XPlane` protos. Each `XPlane`
    represents data from a specific profiling source (e.g., host CPU, TPU
    device, or GPU).
*   **XLine**: An `XPlane` contains parallel timelines called `XLines`.
*   **Events (`XEvents`)**: Each `XLine` contains events representing timed
    activities with a name, start time, and duration.

> [!TIP] If a trace is large, filter to a single device or host using
> `--plane_regex` in `list_xplane_events` or `aggregate_xplane_events`.
