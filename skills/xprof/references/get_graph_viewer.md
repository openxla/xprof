# `get_graph_viewer` Reference

This reference documentation explains how to get graph viewer data (HLO text or
JSON) from XProf. You can retrieve the data either by using an **XSymbol ID**
(with session ID set to 'xsymbol') or by specifying a specific **XProf Session
ID** and module/node name. Use this to retrieve the exact HLO instruction graph
for detailed analysis.

## Prerequisites

-   You must have **either**:
    -   The `symbol_id` and `symbol_type` for the specific XLA program you are
        attempting to analyze.
    -   **OR** a specific `session_id` and the `module_name` or `node_name` you
        want to analyze.

## Instructions

1.  Use the `xprof` command-line tool, providing the log directory or session:

```bash
# View HLO text for the primary module in a logdir
xprof get_graph_viewer <logdir> --output_type=short_txt

# View subgraph around a specific node
xprof get_graph_viewer <logdir> --module_name=<module_name> --node_name=<node_name> --output_type=graph
```

> [!NOTE] **Compilation Proto Requirement**: To capture HLO graph structures in
> trace sessions, ensure workloads are run with
> `XLA_FLAGS='--xla_dump_to=<logdir> --xla_dump_hlo_as_proto'`.

### Parameters

-   `<logdir>` or `--session_id`: The XProf log directory or session ID.
    Optional if `--symbol_id` is provided, but **cannot be set together with
    `--symbol_id`**.
-   `--symbol_id`: The unique ID of the symbol to fetch. Optional, but required
    if `--session_id` is not provided, and **cannot be set together with
    `--session_id`**.
-   `--symbol_type`: The type of the symbol (optional, e.g.,
    `XDB_COMPILER_METADATA`, `XLA_HLO_MODULE_METADATA`).
-   `--graph_type`: The type of graph to view. Defaults to `'xla'`.
-   `--module_name`: The name of the module (optional).
-   `--output_type`: The format of the output or query type. Defaults to
    `'short_txt'`. Supported values include `'short_txt'`, `'module_list'`, and
    `'graph'`.
-   `--show_metadata`: Whether to include metadata in the output. Defaults to
    `True`.
-   `--node_name`: The name of the node (optional, used with
    `output_type='graph'`).
-   `--graph_width`: Graph width (optional, used with `output_type='graph'`,
    defaults to 1).
-   `--merge_fusion`: Whether to merge fusion nodes (optional, used with
    `output_type='graph'`, defaults to False).
-   `--tag`: Optional tag (e.g., `'graph_viewer'`).
-   `--tool`: Optional tool name (e.g., `'graph_viewer'`).
-   `--op_profile_limit`: Optional limit for op profile (e.g., 1).
-   `--use_xplane`: Optional flag to use xplane (e.g., 1).

## Example Usage

1.  **View Module HLO Text**:

    ```bash
    xprof get_graph_viewer /path/to/logdir --module_name=jit_train_step --output_type=short_txt
    ```

2.  **Lookup by Symbol ID**:

    ```bash
    xprof get_graph_viewer /path/to/logdir --symbol_id=5872881440787210439 --symbol_type=XDB_COMPILER_METADATA
    ```
