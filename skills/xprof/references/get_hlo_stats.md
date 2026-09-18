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
   [--category_filter=<CATEGORY>]
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
-   `--bypass_cache` (optional): Recompute metrics without reading from cache
    (default: `False`).

## Example Usage

1.  **Top 10 Operations by Self Time**:

    ```bash
    xprof get_hlo_stats /path/to/logdir --limit=10 --sort_by=self_time
    ```

2.  **Filter Custom Call Operations by Memory Bandwidth**:

    ```bash
    xprof get_hlo_stats /path/to/logdir --category_filter="custom-call" --sort_by=bandwidth
    ```
