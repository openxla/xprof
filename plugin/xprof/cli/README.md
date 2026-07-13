<!-- disableFinding(LINE_OVER_80) -->
# XProf CLI

`xprof_cli` is a command-line interface tool for interacting with Google's XProf profiling data. It provides various commands to extract, analyze, and query different aspects of TPU and GPU profiles.

## Usage

```bash
bazel run //third_party/xprof/plugin/xprof/cli:xprof_cli -- <command> [args...]
```

## Available Commands and Parameters

The CLI exposes numerous tools for analysis. Most tools require at least a `--session_id`.

*   **`llo_load`**: Load LLO traces from a session into a SQLite DB.
    *   `--session_id=<str>`: The XProf session ID.
    *   `--db_path=<str>`: Path to the SQLite database file.
*   **`llo_query`**: Query a generated LLO SQLite DB.
    *   `--db_path=<str>`: Path to the SQLite database file.
    *   `--query=<str>`: SQL query string or pre-baked query name.
*   **`get_overview`**: Get high-level overview of a session.
    *   `--session_id=<str>`
    *   `--include_command=<bool>` (Optional)
*   **`get_memory_profile`**: Analyze memory allocations.
    *   `--session_id=<str>`
*   **`get_top_hlo_ops`**: Fetch the most expensive HLO operations.
    *   `--session_id=<str>`
    *   `--limit=<int>` (Optional, default 10)
*   **`get_kpi_metrics`**: Get Key Performance Indicators.
    *   `--session_id=<str>`
*   **`detect_unfused_reshapes`**: Find reshapes that aren't fused.
    *   `--session_id=<str>`
*   **`diff_sessions`**: Compare two profiling sessions.
    *   `--session_id_1=<str>`
    *   `--session_id_2=<str>`
*   **`find_session`**: Locate an XProf session.
    *   `--session_id=<str>`
*   **`get_events_db_session_root`**: Locate the events DB session root.
    *   `--session_id=<str>`
*   **`get_graph_viewer`**: Fetch graph viewer data.
    *   `--session_id=<str>`
    *   `--graph_type=<str>` (Optional)
    *   `--output_type=<str>` (Optional)
    *   `--show_metadata=<bool>` (Optional)
    *   `--graph_width=<int>` (Optional)
    *   `--merge_fusion=<bool>` (Optional)
*   **`get_hlo_neighborhood`**: Get the neighborhood of an HLO instruction.
    *   `--session_id=<str>`
    *   `--module_name=<str>`
    *   `--instruction_name=<str>`
*   **`get_hlo_text`**: Get textual representation of an HLO module.
    *   `--session_id=<str>`
    *   `--module_name=<str>`
*   **`get_peak_allocations`**: Identify peak memory allocations.
    *   `--session_id=<str>`
*   **`get_smart_suggestions`**: Receive automated performance suggestions.
    *   `--session_id=<str>`
*   **`get_utilization_viewer`**: Fetch utilization data.
    *   `--session_id=<str>`
*   **`detect_layout_mismatch_copies`**: Find copies caused by layout mismatches.
    *   `--session_id=<str>`

## LLO (Low Level Operator) Analysis

For detailed instructions on extracting, loading, and querying LLO events, see the [LLO Analysis Documentation](internal/google/LLO_ANALYSIS_README.md).
