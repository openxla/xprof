# `create_events_db` & `query_events_db` Reference

The XProf Events Database provides a queryable record of individual operations
and timeline events recorded during an XProf session, exposed as a SQL view
named `Events` using DuckDB SQL.

This reference covers:

1.  [Pre-warming / Materializing the Events DB (`create_events_db`)](#1-pre-warming-materializing-the-events-db-create_events_db)
2.  [Querying the Events DB via SQL (`query_events_db`)](#2-querying-the-events-db-via-sql-query_events_db)
3.  [`Events` Table Schema Reference](#3-events-table-schema-reference)
4.  [SQL Query Examples](#4-sql-query-examples)
5.  [Analytical Advice](#5-analytical-advice)

## Prerequisites

-   You must have a `<logdir>`, a specific run directory, or a direct path to a
    single `.xplane.pb` / `.xspace.pb` trace file.
-   **Single-trace constraint**: Multi-file Events DB generation is not yet
    supported. If a run directory contains multiple `.xplane.pb` files (for
    example, multi-worker traces), pass the path to a specific `.xplane.pb` file
    directly.

--------------------------------------------------------------------------------

## 1. Pre-warming / Materializing the Events DB (`create_events_db`)

Materializing the Events DB from large `.xplane.pb` traces can take noticeable
time. **Do not block your main execution thread waiting for generation.**

Delegate `create_events_db` to a background subagent (or background task) to
prepare the cached Events DB ahead of time:

```bash
xprof create_events_db <logdir_or_trace_path> [--bypass_cache]
```

### Arguments

-   `<logdir_or_trace_path>`: Log directory, run directory, or direct path to a
    `.xplane.pb` / `.xspace.pb` file.
-   `--bypass_cache` (optional): Force regeneration of the cached Events DB even
    if it already exists (default: `False`).

### Return Payload

Returns a JSON object with:

-   `"path"`: Absolute path to the cached database file.
-   `"skipped"`: `true` if a cached database already existed and generation was
    skipped; `false` if it was freshly generated.

--------------------------------------------------------------------------------

## 2. Querying the Events DB via SQL (`query_events_db`)

Run a single read-only `SELECT` query (DuckDB SQL dialect) against the `Events`
view using `--query`:

```bash
xprof query_events_db <logdir_or_trace_path> \
  --query="SELECT COUNT(*) AS total_events FROM Events" \
  [--bypass_cache]
```

Or, when passing `--logdir` as a flag (or when a default logdir is already
configured):

```bash
xprof query_events_db \
  --logdir=<logdir_or_trace_path> \
  --query="SELECT COUNT(*) AS total_events FROM Events"
```

### Execution & Caching Behavior

1.  **Pre-registered `Events` View**: `query_events_db` automatically resolves
    (or materializes, if not yet cached) the Events DB for the session and
    exposes it as a SQL view named `Events` (`FROM Events`).
2.  **Sandboxed Read-Only SQL**: Only a single DuckDB `SELECT` statement is
    allowed (CTEs via `WITH ... SELECT ...` and `FROM Events SELECT ...` syntax
    are supported). DDL/DML statements (`COPY`, `SET`, `ATTACH`, etc.) and
    external filesystem reads are blocked.
3.  **Cached JSON Output**: Query results are saved as a JSON array to a cached
    `.json` file, and the CLI prints a JSON envelope containing the file path:

    ```json
    {
      "path": "/tmp/xprof_cli_cache_user/events_db/<digest>/<query_hash>.json",
      "skipped": false
    }
    ```

4.  **Inspecting Results**: Read the JSON file at `"path"` directly (for
    example, using `jq` or file-viewing tools):

    ```bash
    RESULT_PATH=$(xprof query_events_db /path/to/trace.xplane.pb \
      --query="SELECT COUNT(*) AS total_events FROM Events" | jq -r '.path')
    cat "$RESULT_PATH"
    ```

> **Note**: Passing `--bypass_cache` to `query_events_db` re-executes the SQL
> query and overwrites the cached `.json` result without rebuilding the
> underlying Events DB. To rebuild the Events DB itself, run
> `xprof create_events_db <logdir_or_trace_path> --bypass_cache`.

--------------------------------------------------------------------------------

## 3. `Events` Table Schema Reference

Each row in `Events` represents a recorded trace event. All 31 columns listed
below are always present in the `Events` table schema; if a field is not
populated for a given event, its value in that row is `NULL`.

### Identity & Placement Columns

-   `device` (`VARCHAR`): Device or host identifier (for example, `"cpu:0"`,
    `"gpu:0"`, `"TPU:0"`, `"TPU:0 SparseCore 0"`). Leading `"/device:"` prefixes
    on TPU planes are stripped automatically.
-   `category` (`VARCHAR`): Event category or timeline line name (`"host"` for
    CPU host events, `"device"` for GPU stream events, or the XLine name such as
    `"XLA Ops"`, `"XLA Modules"`, and `"Steps"` on TPU/XLA lines).
-   `step` (`VARCHAR`): Training or execution step name (for example,
    `"step:0"` on TPU or the group metadata step name on CPU/GPU).
-   `stream_id` (`UINTEGER`): Stream / XLine identifier on device planes (`NULL`
    on CPU host events).
-   `thread_id` (`UINTEGER`): CPU host thread ID (`NULL` on device events).
-   `thread_name` (`VARCHAR`): CPU host thread display name.
-   `correlation_id` (`UINTEGER`): Launch correlation ID linking a host runtime
    launch event to its corresponding device execution event.

### Timing Columns (Nanoseconds)

-   `start_ns` (`UBIGINT`): Event start timestamp in nanoseconds.
-   `end_ns` (`UBIGINT`): Event end timestamp in nanoseconds (
    `end_ns - start_ns` is the total inclusive wall-clock duration).
-   `self_time_ns` (`UBIGINT`): Event self-time in nanoseconds (duration
    excluding nested child events on hierarchical lines such as TPU TensorCore).

### Operation, HLO & Source Attribution Columns

-   `hlo_op` (`VARCHAR`): XLA HLO operation name (`DisplayName` on TPU
    `"XLA Ops"` or innermost HLO op on GPU).
-   `hlo_module` (`VARCHAR`): HLO module name (formatted as
    `<module_name>(<program_id>)`).
-   `hlo_fingerprint` (`UBIGINT`): 64-bit fingerprint of the HLO instruction's
    canonical text representation.
-   `tf_op_name` (`VARCHAR`): Framework-level operation name (for example, JAX
    or TensorFlow op scope).
-   `tf_op_type` (`VARCHAR`): Framework-level operation type.
-   `kernel_name` (`VARCHAR`): Raw event name (`TraceMe` name on CPU, CUDA
    kernel name on GPU, or raw `XEvent` name / `"step:..."` / `"HLO Module:..."`
    marker on TPU).
-   `kernel_details` (`VARCHAR`): GPU kernel launch configuration (grid/block
    dimensions, registers, shared memory).
-   `source_line` (`VARCHAR`): Python/framework user source file and line number
    (`file.py:line`).

### Cost, Tensor & Trace Argument Columns

-   `flops` (`UBIGINT`): Estimated FLOPs from the XLA compiler's cost analysis.
-   `memory_accessed` (`UBIGINT`): Estimated bytes accessed from the XLA
    compiler's cost analysis.
-   `input_tensors` (`VARCHAR[]`): Shapes of the operation's input tensors.
-   `output_tensors` (`VARCHAR[]`): Shapes of the operation's output tensors.
-   `trace_args` (`VARCHAR`): Comma-separated `"key=value"` pairs of additional
    `TraceMe` / event stats not mapped to dedicated columns.
-   `flow` (`UBIGINT`): Async flow ID linking producer and consumer host/DCN
    events (for example, `SendStart` -> `SendFinished`).

### DCN / Multi-Slice Collective Columns

-   `dcn_collective_name` (`VARCHAR`): Collective communication identifier.
-   `dcn_src_slice_id` (`BIGINT`) / `dcn_dst_slice_id` (`BIGINT`): Source and
    destination slice IDs.
-   `dcn_src_logical_device_id` (`BIGINT`) /
    `dcn_dst_logical_device_id` (`BIGINT`): Source and destination logical TPU
    device IDs.
-   `dcn_duration_us` (`UBIGINT`): DCN transfer duration in microseconds.
-   `dcn_payload_size_bytes` (`UBIGINT`): DCN payload size in bytes.

--------------------------------------------------------------------------------

## 4. SQL Query Examples

### 1. Top 10 HLO Operations by Total Device Time

```bash
xprof query_events_db <logdir> --query="
  SELECT
    hlo_op,
    category,
    COUNT(*) AS occurrences,
    SUM(self_time_ns) / 1000.0 AS total_time_us
  FROM Events
  WHERE hlo_op IS NOT NULL
    AND category = 'XLA Ops'
  GROUP BY hlo_op, category
  ORDER BY total_time_us DESC
  LIMIT 10
"
```

### 2. Per-Step Duration and Event Count Breakdown

```bash
xprof query_events_db <logdir> --query="
  SELECT
    device,
    step,
    (MAX(end_ns) - MIN(start_ns)) / 1e6 AS step_span_ms,
    COUNT(*) AS num_events
  FROM Events
  WHERE step IS NOT NULL
  GROUP BY device, step
  ORDER BY device, step
"
```

### 3. Latency Distribution (Min, P50, P99) Across Repetitions of an HLO Op

Use DuckDB's `QUANTILE_CONT` function for percentile calculations:

```bash
xprof query_events_db <logdir> --query="
  SELECT
    hlo_op,
    COUNT(*) AS n,
    MIN(self_time_ns) / 1000.0 AS min_us,
    QUANTILE_CONT(self_time_ns / 1000.0, 0.50) AS p50_us,
    QUANTILE_CONT(self_time_ns / 1000.0, 0.99) AS p99_us
  FROM Events
  WHERE category = 'XLA Ops'
    AND hlo_op IS NOT NULL
  GROUP BY hlo_op
  HAVING COUNT(*) > 1
  ORDER BY p99_us DESC
  LIMIT 10
"
```

### 4. Source Code Provenance for High-Cost HLO Operations

```bash
xprof query_events_db <logdir> --query="
  SELECT
    hlo_op,
    source_line,
    SUM(self_time_ns) / 1000.0 AS total_time_us,
    SUM(flops) AS total_flops,
    SUM(memory_accessed) AS total_bytes
  FROM Events
  WHERE category = 'XLA Ops'
    AND source_line IS NOT NULL
  GROUP BY hlo_op, source_line
  ORDER BY total_time_us DESC
  LIMIT 10
"
```

--------------------------------------------------------------------------------

## 5. Analytical Advice

1.  **Filter `category = 'XLA Ops'` for Low-Level Device Ops**: TPU device
    planes contain multiple timeline lines (`"Steps"`, `"XLA Modules"`,
    `"XLA Ops"`, etc.). Filtering `WHERE category = 'XLA Ops'` avoids
    double-counting parent module spans (`"XLA Modules"`) or step markers
    (`"Steps"`).
2.  **Group or Filter by `device` Across Multi-Plane Traces**: A single
    `.xplane.pb` file often contains multiple `XPlane`s (for example, `"cpu:0"`
    plus `"TPU:0"`..`"TPU:3"` or `"gpu:0"`..`"gpu:7"`). Because step counters
    (`step`) are numbered per plane and identical steps across cores do not
    start or end at the exact same timestamp, always include `device` in
    `GROUP BY device, step` (or filter to a single core such as
    `WHERE device = 'TPU:0'`) when computing step spans or per-core counts.
3.  **Use `IS NOT NULL` for Unset Fields**: Unset fields on an event are stored
    as SQL `NULL` rather than empty strings (`''`). Always filter with
    `WHERE <column> IS NOT NULL` (for example, `WHERE hlo_op IS NOT NULL` or
    `WHERE step IS NOT NULL`).
4.  **Multi-Worker Directories**: If `create_events_db` or `query_events_db`
    raises `NotImplementedError: Multiple (N) trace files found`, pass one of
    the listed `.xplane.pb` file paths directly as the first argument.
