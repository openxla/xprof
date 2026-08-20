# `get_top_hlo_ops` Reference

This reference explains how to get a list of top HLO operations sorted by three
different criteria: Self Time, FLOPs, and Bytes Accessed. Use this to identify
which operations are consuming the most resources.

## Prerequisites

-   You must have the log directory path (`<logdir>`), direct run directory, or
    session ID for the specific XProf run you are attempting to analyze.

## Instructions

1.  Use the `xprof` command-line tool, providing the log directory or session
    ID:

    ```bash
    xprof get_top_hlo_ops <logdir>
    ```

    Optional flags:

    -   `--limit=<n>`: Maximum number of operations to return per category
        (default: 10).
    -   `--category_filter=<category>`: Filter operations by category (e.g.,
        `fusion`, `convolution`).

2.  Review the JSON returned by the tool, which contains three lists:

    -   `top_by_time`: Top operations by self time.
    -   `top_by_flops`: Top operations by FLOPs.
    -   `top_by_bytes_accessed`: Top operations by bytes accessed.

    Each item in these lists provides performance statistics and Python source
    line mapping:

    ```json
    {
      "name": "by_program/jit_train_step/convolution fusion/fusion.822",
      "category": "convolution fusion",
      "total_self_time_ms": 84.15,
      "occurrences": 432,
      "flops": 5.9e13,
      "bytes_accessed": 7.9e10,
      "source_file": "layers/linears.py",
      "source_line": 99,
      "stack_frame": "models/gemma.py:163:14\nlayers/decoders.py:956:17"
    }
    ```

> [!NOTE] **Compilation Proto Requirement**: If operation profile data is
> missing from a trace session, ensure compilation was captured with
> `XLA_FLAGS='--xla_dump_to=<logdir> --xla_dump_hlo_as_proto'`.

## Example Usage

If the user says: "What are the top HLO ops for /path/to/logdir?", you should:

1.  Run the CLI command:

    ```bash
    xprof get_top_hlo_ops /path/to/logdir
    ```
2.  Summarize the top operations from each of the three lists (Time, FLOPs,
    Bytes Accessed) to the user.
