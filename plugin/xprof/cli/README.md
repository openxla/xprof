<!-- disableFinding(LINE_OVER_80) -->
# XProf CLI

`xprof_cli` is a command-line interface tool for interacting with Google's XProf profiling data. It provides various commands to extract, analyze, and query different aspects of TPU and GPU profiles across both 1P and open-source environments.

## Usage

```bash
# Standalone PyPI / Open-Source CLI
xprof <command> <path_to_trace_or_logdir> [flags...]

# Directly analyze traces stored on Google Cloud Storage (gs://)
xprof get_overview gs://<bucket>/<path_to_trace_or_run_dir>
xprof get_overview --logdir=gs://<bucket>/<logdir> [--session_id=<run_name>]

# In Google3 monorepo
bazel run //third_party/xprof/plugin/xprof/cli:xprof_cli -- <command> <path_to_trace> [flags...]
```

> **GCS (`gs://`) Support:** Authenticate with Application Default Credentials
> (`gcloud auth application-default login`) before reading `gs://` URIs.
>
> *   **Supported on `gs://`:** Converter-backed commands (`get_overview`,
>     `get_roofline_model`, `get_memory_profile`, `get_top_hlo_ops`,
>     `get_kpi_metrics`, `get_hosts`, etc.) and XPlane-inspection commands
>     (`list_xplane_events`, `aggregate_xplane_events`, `get_xspace_proto`,
>     `get_kernel_stats`) accept `gs://` file, run-directory, and `--logdir`
>     URIs natively without writing cache files back to the bucket.
> *   **Local paths only:** HLO-proto extraction commands (`get_hlo_text`,
>     `get_hlo_module_content`, `get_hlo_neighborhood`, `detect_*`), Events DB
>     commands, and `upload_trace` require local paths.
> *   **Local mirror cache:** Downloaded GCS traces are cached locally under
>     `$TMPDIR/xprof_cli_cache_<user>/gcs` (default `/tmp/xprof_cli_cache_<user>/gcs`)
>     keyed by blob generation. Remove that directory to reclaim disk space.

## Available Commands

The CLI provides 28 core tools for comprehensive accelerator profile analysis:

### Overview & Telemetry

*   **`get_overview`**: High-level performance overview (Compute vs Host vs Communication, step times, duty cycle, and fallback roofline metrics).
*   **`get_roofline_model`**: Detailed roofline efficiency, arithmetic intensity (FLOP/Byte), hardware limits, and per-op bottleneck bounds.
*   **`get_profile_summary`**: Executive-level text summary of top bottlenecks and memory utilization.
*   **`get_kpi_metrics`**: Key Performance Indicators (duty cycle, step time, FLOP utilization).
*   **`get_device_information`**: Hardware topology and theoretical roofline compute/bandwidth constants.
*   **`get_hosts`**: List hostnames and workers present in the profile session.

### Memory & Allocation Analysis

*   **`get_memory_profile`**: Detailed memory breakdown (HBM capacity, peak usage, fragmentation).
*   **`get_peak_allocations`**: Identifies peak memory allocations and buffer consumption ordered by HBM usage.

### HLO & Compiler Analysis

*   **`get_top_hlo_ops`**: Fetches the most expensive HLO operations sorted by execution time, FLOPs, or memory traffic.
*   **`get_hlo_op_profile`**: Formatted HLO operation profile breakdown table.
*   **`list_hlo_modules`**: Lists all HLO modules available in the profile session.
*   **`get_hlo_text`**: Retrieves textual representation of compiled HLO instructions.
*   **`get_hlo_module_content`**: Fetches full HLO instruction graph for a selected module.
*   **`get_hlo_neighborhood`**: Traverses BFS neighborhood (operands and users) around a target HLO instruction.
*   **`get_graph_viewer`**: Fetches DOT/pbtxt/text AST graph representation of HLO computation graphs.

### Static HLO Detectors

> These detectors parse the raw HLO proto using the HLO schema bundled with xprof, so they need no extra packages (such as `tensorflow`). They rank candidates by the device-side op profile, so use a TPU or GPU trace; a CPU-only trace reports no findings.

*   **`detect_unfused_reshapes`**: Flags reshape ops that materialize large tensors to HBM instead of being fused, adding avoidable memory traffic.
*   **`detect_unfused_updates`**: Detects unfused sequences of small elementwise/update ops that could be fused to reduce HBM round-trips.
*   **`detect_layout_mismatch_copies`**: Finds copy ops inserted solely to reconcile layout mismatches, causing avoidable HBM overhead.
*   **`detect_unnecessary_convert_reduce`**: Identifies unnecessary f32 promotions around reduction ops.
*   **`detect_unnecessary_convert_dynamic_scale`**: Identifies unnecessary f32 upcasts during dynamic-scale calculation and quantization.

### Timeline & XPlane Inspection

*   **`list_xplane_events`**: Streams timeline execution events from accelerator execution planes.
*   **`aggregate_xplane_events`**: Aggregates event occurrences and total active durations.
*   **`get_xspace_proto`**: Extracts raw XSpace protocol buffer bytes or dumps to file.
*   **`get_kernel_stats`**: Performance statistics and step times across workloads.
*   **`get_utilization_viewer`**: Accelerator utilization metrics across execution streams.

### Low-Level Operator (LLO) & Diagnostics

*   **`get_llo_analysis`**: Low-level instruction metrics and execution schedules.
*   **`get_llo_debug_string`**: Raw low-level operator debug disassembly.

### Ingestion

*   **`upload_trace`**: Ingests and registers `.xplane.pb` traces into profile run directories.

## Kernel Statistics & Disjoint Interval Union

When evaluating kernel statistics with `include_summary=True`, `xprof_cli` calculates total device compute duration using **Disjoint Interval Union**.

On modern accelerators, multiple execution planes (e.g., MXU compute kernels, DMA/ICI transfers) operate concurrently. Summing individual kernel durations double-counts overlapping execution, while taking the bounding box `max(t_end) - min(t_start)` includes inter-step idle gaps.

Disjoint Interval Union uses a sweep-line algorithm to merge all active `[start, end]` intervals across hardware XPlanes into non-overlapping disjoint intervals, reporting the exact active hardware compute time without double-counting or idle gap padding.

## LLO (Low Level Operator) Analysis

For detailed instructions on extracting, loading, and querying LLO events, see the [LLO Analysis Documentation](internal/google/LLO_ANALYSIS_README.md).

