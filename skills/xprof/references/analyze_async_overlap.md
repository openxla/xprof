# `analyze_async_overlap` Reference

This reference explains how to measure how much of each asynchronous XLA
operation (collectives, async copies, `send`/`recv`) is hidden behind compute
on the same device, and how much time multiple engines on the same TPU chip
(TensorCore and SparseCore) spend busy concurrently.

## Instructions

1.  Run `analyze_async_overlap` on a session ID, run directory, or `.xplane.pb`
    trace file:

    ```bash
    xprof analyze_async_overlap <trace_or_session_id> \
        [--host=<host>] \
        [--hidden_threshold=0.9] \
        [--exposed_threshold=0.1] \
        [--top_k=20]
    ```

2.  Inspect `status` first:
    *   `"OK"`: Device timeline events were found and analyzed.
    *   `"NO_DEVICE_TIMELINE_EVENTS"`: The capture has no `XLA Ops`,
        `Async XLA Ops`, or `Sparse Core Ops` events; check `reason`.

3.  Read the returned sections:
    *   `summary`: Overall async op count (`count`), total window duration
        (`span_ps`), uncovered duration (`exposed_ps`), overall covered fraction
        (`overlap_ratio`), counts per class (`hidden`, `partial`, `exposed`),
        and `unpaired_ops` when start/done fallback pairing is used.
    *   `by_op`: Breakdown keyed by op family (`all-gather`, `all-reduce`,
        `all-to-all`, `reduce-scatter`, `copy`, `send`, `recv`, ...), ordered
        by `exposed_ps` descending.
    *   `by_plane`: Per-device breakdown plus `window_source`
        (`"Async XLA Ops"` when that line is present, or `"start_done_pairs"`
        when windows were paired from `<op>-start` / `<op>-done` events on the
        op line).
    *   `top_exposed`: Top `top_k` individual async op windows ordered by
        `exposed_ps` descending (`plane`, `name`, `op`, `start_ps`, `span_ps`,
        `exposed_ps`, `overlap_ratio`, `class`).
    *   `concurrency`: For each TPU chip with at least two busy engines (for
        example `/device:TPU:0` `XLA Ops` and `/device:TPU:0 SparseCore 0`
        `Sparse Core Ops`), lists per-engine `busy_ps` and pairwise
        `concurrent_ps` and `concurrent_ratio` (fraction of the less-busy
        engine's active time spent concurrently with the other engine).

## Classification Thresholds

| Class | Condition | Default |
| :--- | :--- | :--- |
| `hidden` | `overlap_ratio >= hidden_threshold` | `>= 0.9` |
| `partial` | `exposed_threshold < overlap_ratio < hidden_threshold` | `(0.1, 0.9)` |
| `exposed` | `overlap_ratio <= exposed_threshold` | `<= 0.1` |

Thresholds must satisfy `0 <= exposed_threshold < hidden_threshold <= 1`.
Coverage is measured against non-async ops on the **same plane** only and
excludes the start and done bookend ops themselves.

## Example Usage

If the user asks "Are my all-gather collectives hidden behind compute, or are
they exposed on the critical path?", you should:

1.  Run:

    ```bash
    xprof analyze_async_overlap <trace_or_session_id> --top_k=10
    ```

2.  Report `summary.overlap_ratio`, the `by_op["all-gather"]` bucket (`count`,
    `span_ps`, `exposed_ps`, `hidden` / `partial` / `exposed` counts), and the
    most exposed windows from `top_exposed`.
