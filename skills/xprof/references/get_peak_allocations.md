# `get_peak_allocations` Reference

This reference provides a breakdown of memory allocations by HLO module and
buffer size for an XProf session. You should use this to identify which HLO
modules are using the most memory and which specific buffers are contributing to
that usage.

## Prerequisites

-   You must have the log directory path (`<logdir>`), direct run directory, or
    session ID for the specific XProf run you are attempting to analyze.

## Instructions

1.  Use the `xprof` command-line tool, providing the log directory or session
    ID:

    ```bash
    xprof get_peak_allocations <logdir>
    ```

2.  You can customize the output using optional flags:

    -   `--limit=<n>`: Limit the output to top `<n>` modules (default: 10). Use
        0 for no limit.
    -   `--min_size_mib=<size>`: Group buffers smaller than `<size>` MiB into an
        "Others" category (default: 1.0).
    -   `--output_format=<json|markdown>`: Choose output format (default: json).
    -   `--include_summary`: Include a high-level summary at the top (default:
        True).
    -   `--aggregate_instructions=<True|False>`: Whether to aggregate similar
        instructions (default: True).

3.  Review the output, which contains a high-level summary at the top (if
    `--include_summary` is True), followed by a list of top modules (up to the
    specified `--limit`) ordered by total memory usage, and for each module, a
    list of top buffers ordered by size.

## Output Format and Aggregation

When `aggregate_instructions` is enabled (default) and using `markdown` output
format, the tool applies aggregation logic to keep the output concise:

-   **Instruction Aggregation**: Buffers with similar names (e.g., `name.1`,
    `name.2`) and identical sizes are aggregated into a single entry named
    `name.* (N occurrences of size S MiB)`.
-   **Size Threshold**: Buffers smaller than the specified `--min_size_mib`
    (default 1.0 MiB) are aggregated into an "Others" category at the bottom of
    the table for each module.

## Example Usage

If the user says: "Show me the memory allocations by buffer size for
/path/to/logdir in markdown, showing top 5 modules, ignoring buffers smaller
than 2.0 MiB, and without aggregating instructions", you should:

1.  Run the CLI command:

    ```bash
    xprof get_peak_allocations /path/to/logdir \
      --output_format=markdown \
      --include_summary \
      --limit=5 \
      --min_size_mib=2.0 \
      --aggregate_instructions=False
    ```
2.  Present the returned markdown to the user.
