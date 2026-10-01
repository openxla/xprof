# `get_hlo_stats` Reference

Fetches detailed performance statistics for individual HLO operations from the
`hlo_stats` database, including self time, total time, occurrence count, FLOPs,
memory bandwidth, roofline bottleneck classification (`bound_by`), and source
code provenance (`source_file`, `source_line`).

## Prerequisites

-   You must have the log directory path (`<logdir>`), direct run directory, or
    session ID for the specific XProf run you are attempting to analyze.

## Instructions

Run the CLI command with optional sorting, filtering, and row limits:

```bash
xprof get_hlo_stats <logdir> \
   [--limit=<LIMIT>] \
   [--sort_by=<METRIC>] \
   [--category_filter=<CATEGORY>] \
   [--include_nested=True]
```

### Arguments

-   `<logdir>` or `--session_id`: The XProf log directory or session ID.
-   `--limit` (optional): Maximum number of HLO operation records to return
    (default: `20`).
-   `--sort_by` (optional): Metric used to order operations in descending order.
    Supported values: `'self_time'`, `'total_time'`, `'occurrences'`, `'flops'`,
    and `'bandwidth'` (default: `'self_time'`).
-   `--category_filter` (optional): Substring filter on `hlo_category` (e.g.
    `'custom-call'`, `'convolution fusion'`, `'loop fusion'`).
-   `--include_nested` (optional): Also return nested operations (default:
    `False`). See [Nested operations](#nested-operations).
-   `--bypass_cache` (optional): Recompute metrics without reading from cache
    (default: `False`).

## Output

A JSON list of operations. Each row includes `rank`, `self_time_percent`,
`core_type` (`TensorCore` or `SparseCore`) and `parent_op_name`.

### Nested operations

On TPUs with SparseCore offload (e.g. v7x collectives), a TensorCore offload op
starts work on a SparseCore (e.g. `reduce-scatter.542` under
`reduce-scatter.543.cloned.1.call-start`). The SparseCore op is listed as a
nested op: it has a non-empty `parent_op_name`, `core_type: SparseCore`,
`rank: 0` and `self_time_percent: 0`. Its time is SparseCore time, not part of
the TensorCore device time that top-level rows are ranked against, so it is not
ranked and must never be added to top-level rows. Whether the parent's
`async-done`/`call-done` also covers that time depends on whether the
TensorCore waits for the SparseCore; it can be far shorter than the nested op.

Nested ops are hidden by default, so the default output does not show
SparseCore busy time. stderr gets one `xprof-note: hid N nested operation(s)`
line when any were hidden. Pass `--include_nested=True` to see them. With
`--include_nested=True`, nested rows count against `--limit`, so raise `--limit`
if you still need the full top-N of top-level ops.

## Example Usage

1.  **Top 10 Operations by Self Time**:

    ```bash
    xprof get_hlo_stats /path/to/logdir --limit=10 --sort_by=self_time
    ```

2.  **Filter Custom Call Operations by Memory Bandwidth**:

    ```bash
    xprof get_hlo_stats /path/to/logdir --category_filter="custom-call" --sort_by=bandwidth
    ```

3.  **SparseCore Collectives Inside Offload Ops**:

    ```bash
    xprof get_hlo_stats /path/to/logdir --include_nested=True --category_filter="reduce-scatter"
    ```
