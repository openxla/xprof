---
name: xprof
description: >-
  Central entry point for ALL XProf operations and analyses. Use this skill first for any task involving XProf traces, performance, memory, HLO ops, or collecting/triggering XProf profiles. Don't use for standalone numerical parity verification (use xparity).
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

All `xprof` subcommands emit **JSON** to stdout. There is no output-format flag;
pipe the output to `jq` or redirect it to a file if you need another
representation.

> [!NOTE] **10 MB Spill-to-File Guard (`SAVED_TO_FILE`)**: To prevent
> terminal and agent buffer overflows, any CLI output exceeding **10 MB**
> (`10,485,760` bytes) is automatically written to a temporary file and
> replaced on stdout by a compact JSON envelope:
>
> ```json
> {
>   "status": "SAVED_TO_FILE",
>   "size_bytes": 15482910,
>   "size_mib": 14.77,
>   "file_path": "/tmp/xprof_spill_list_xplane_events_abc123.json",
>   "message": "Payload (14.77 MB) exceeded 10 MB. Saved to file."
> }
> ```
>
> When `status == "SAVED_TO_FILE"`, inspect the payload directly from
> `file_path` using `jq` (e.g., `jq '.events[:10]' <file_path>`) or
> targeted file reads rather than loading the entire file into memory.

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

### Bottleneck Analysis (7-Phase Protocol)

When asked to find or analyze performance bottlenecks for an XProf session,
follow the structured **7-Phase Performance Analysis Protocol** detailed in
[references/analysis.md](references/analysis.md) (Turn-1 Parallel Triage,
Macro-to-Micro Op Drilldown, Headroom & EIC Quantification, Actionable Code
Proposal, Empirical Validation, Numerical Parity, and Artifact Closure).

### Profile Collection & Ingestion

When collecting a performance trace or preparing an external trace for
analysis:

1.  **Collect Profile**: For JAX, PyTorch, or TensorFlow workloads, capture
    traces programmatically or via remote profiler servers (see
    [Profile Collection Guide](references/collect_profile.md)).
2.  **Import Standalone Traces**: For pre-existing trace files (`.xplane.pb` or
    `.xspace.pb`), run `xprof upload_trace` to stage them into your
    `<logdir>` (see [Import Trace File](references/upload_trace.md)).

### Low Level Optimizer (LLO) & Custom Call Profiling

> ⚠️ **EXPERIMENTAL FEATURE**: Low Level Optimizer (LLO) analysis and custom
> call profiling are **experimental**. To access these features and all CLI
> subcommands (`get_kernel_stats`,
> `get_llo_analysis`, `get_llo_debug_string`), users **MUST install
> `xprof-nightly`** (`pip install xprof-nightly` or `uv pip install
> xprof-nightly 'jax[tpu]'`), as the main `xprof` PyPI release (2.23.1) lacks
> these subcommands.

When capturing or analyzing fine-grained LLO traces for custom kernels (e.g.
Pallas or Mosaic):

1.  **Toolchain Prerequisites**: Workload VMs must run **Python 3.11+** (Python
    3.12 recommended via `uv`) and **JAX >= 0.11.0**. (Default Cloud TPU VM
    images running Python 3.10 cap JAX at 0.6.2 and pull `libtpu` 0.0.17 which
    lacks LLO flag support and causes `ERROR: Unknown command line flag`).
2.  **Environment Initialization**: `LIBTPU_INIT_ARGS` must be exported
    **strictly before `import jax`**:

    ```bash
    export LIBTPU_INIT_ARGS="--xla_xprof_register_llo_debug_info=true"
    python your_jax_workload.py
    ```
3.  **Hardware Compatibility**:

    *   LLO analysis, disassembly, and custom call tracing work on **any
        supported TPU** (v6e, v5e, v4, etc.) — it is **NOT** gated to v7x.
    *   Periodic hardware runtime counters
        (`tpu_enable_periodic_counter_sampling`) require Ironwood TPU7x+.
4.  **Flag Symbol Verification (Optional Diagnostic)**: Inspect installed
    `*libtpu*.so` to verify supported flags before launching:

    ```python
    import glob, libtpu, os
    so = glob.glob(os.path.dirname(libtpu.__file__) + "/*libtpu*.so")[0]
    blob = open(so, "rb").read()
    for f in (b"xla_xprof_register_llo_debug_info", b"xla_xprof_enable_custom_call_tracing"):
      print(f.decode(), "PRESENT" if f in blob else "ABSENT")
    ```
5.  **Canonical Flags**:

    *   `--xla_xprof_register_llo_debug_info=true`: Registers compile-time LLO
        debug info, disassembly, and source map in XProf traces. **This is the
        default you want** — on its own it provides the complete LLO source map
        for `get_llo_analysis` and `get_llo_debug_string` without adding
        runtime trace overhead, keeping the HLO op stream intact.
    *   `--xla_xprof_enable_custom_call_tracing=true`: Canonical flag
        (reconciles legacy `--xla_enable_custom_call_region_trace=true`). **Add
        it only when you need runtime intra-kernel timeline lanes** (`Pallas
        Primitives`, `LLO Ops`, and per-unit instruction lanes in Trace Viewer).
        It activates bundle-level instrumentation
        (`xla_tpu_bundle_instrumentation_options` with default
        `trace_best_effort_frequency=10` and `trace_guaranteed_frequency=10`),
        inserting a `vtrace` every 10 VLIW bundles inside custom calls. On
        long-running custom calls, the extra event volume can overflow the
        hardware trace buffer and drop the outer HLO `Begin`/`End` events, so
        HLO-level tools (`get_hlo_stats`, `get_roofline_model`,
        `get_top_hlo_ops`) report `IDLE` or `NO_DATA`. If that happens,
        increase `trace_best_effort_frequency` and `trace_guaranteed_frequency`
        (e.g., to `50` or `100`) or collect a second profile without the flag.
        See [custom call profiling](../../docs/custom_call_profiling.md).
6.  **Analysis Execution & Metrics Interpretation**:

    *   `xprof get_llo_analysis <logdir_or_session_id>`: Extracts instruction
        counts, cycle estimates, and execution schedules.
    *   `xprof get_llo_debug_string <logdir_or_session_id>`: Disassembles Low
        Level Optimizer (LLO) instructions (inner loops appear as `// Loop body
        not available in lite proto`).
    *   `xprof get_kernel_stats <logdir_or_session_id>`: Provides kernel
        latency. *(Note: `get_roofline_model` and `get_overview` will report 0.0
        GFLOP/s and 0% MXU for `tpu_custom_call` because XLA has no cost model
        for custom calls; use `get_kernel_stats` for latency and
        `get_llo_analysis` for instruction counts).*
    *   **Trace Validation**: Test LLO presence by calling `get_llo_analysis`
        and reading `success`, not by inspecting trace line names.

<h2 id="supported-capabilities--references">Supported Capabilities & References</h2>

-   **[Get Graph Viewer Data](references/get_graph_viewer.md)**: Get graph
    viewer data (HLO text) and source line mappings from XProf.
-   **[Get Session Overview & Summary](references/get_overview.md)**: Get a
    comprehensive overview (`get_overview`) or concise summary metadata
    (`get_profile_summary`) of an XProf session.
-   **[Get Memory Profile](references/get_memory_profile.md)**: Get a detailed
    memory profile analysis (Peak/device memory details) of an XProf session.
-   **[Get Step Trace & Average Step Time](references/get_step_trace.md)**:
    Retrieve step-by-step execution breakdowns (`get_step_trace`) or scalar
    average step duration (`get_avg_step_time`) to analyze step time variance
    and communication overhead.
-   **[Get Peak Allocations](references/get_peak_allocations.md)**: Get HLO
    modules and buffers ordered by memory usage.
-   **[Get Top HLO Operations](references/get_top_hlo_ops.md)**: Identify top
    HLO operations by time, FLOPs, or bytes accessed.
-   **[Get HLO Operation Statistics](references/get_hlo_stats.md)**: Fetch
    per-operation statistics (`get_hlo_stats`) sorted by self time, total time,
    occurrences, FLOPs, or memory bandwidth.
-   **[Get Kernel Utilization & Stats](references/get_kernel_utilization.md)**:
    Fetch hardware unit utilization (`get_kernel_utilization` /
    `compute_utilization`) and raw kernel execution latencies
    (`get_kernel_stats`).
-   **[Get KPI Metrics](references/get_kpi_metrics.md)**: Fetch consolidated KPI
    metrics (step time, duty cycle, MXU utilization, roofline) for a session.
-   **[Get Roofline Model](references/get_roofline_model.md)**: Identify
    compute vs memory bandwidth bottlenecks at program and per-operation level.
-   **[Get HLO Neighborhood](references/get_hlo_neighborhood.md)**: Fetch the
    BFS neighborhood of an HLO instruction to identify fusion blockers.
-   **[Get Utilization Viewer](references/get_utilization_viewer.md)**: Fetch
    utilization metrics filtered by host, device, or node.
-   **[Query Events Database](references/events_db.md)**: Materialize and query
    raw XProf event traces via SQL for fine-grained event, timeline, and
    argument analysis.
-   **[Analyze XLA Module Performance](references/analysis.md)**: Analyze XLA
    module performance, inspect HLO operations, query timeline events, and
    inspect raw XSpace protobufs (`get_xspace_proto`).
-   **[Architecture Mapping](references/architecture_mapping.md)**: Map model
    architecture blocks to the HLO ops and timeline events that implement them.
-   **[Import Trace File](references/upload_trace.md)**: Import raw trace files
    into an xprof logdir for analysis.
-   **[Collect XProf Profile](references/collect_profile.md)**: Collect
    performance profiles across JAX, PyTorch, and TensorFlow workloads via
    programmatic tracing or remote capture.
-   **[Diff Sessions](references/diff_session.md)**: Compare performance, kernel
    execution times, top operations, and HLO graphs between baseline and
    candidate sessions.
-   **[Custom Call & LLO Profiling][custom-call-doc]**: Trace custom kernel
    execution, instruction metrics, and register LLO debug info.

[custom-call-doc]: https://openxla.org/xprof/custom_call_profiling
