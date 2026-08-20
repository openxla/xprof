# `upload_trace` Reference

This reference explains how to import raw trace files into an xprof logdir
directory structure for analysis using XProf tools.

## Overview

`xprof` tools can analyze profile data in two ways:

1.  **Direct Path**: Pass an individual `.xplane.pb` or `.trace.json.gz` file
    directly to any `xprof` CLI command.
2.  **Logdir Import (`upload_trace`)**: Import raw trace files into a structured
    xprof logdir directory (`<logdir>/plugins/profile/<run_name>/`), making them
    automatically discoverable by the XProf Web Server and CLI tools.

--------------------------------------------------------------------------------

## Instructions

1.  Use the `upload_trace` subcommand, specifying the source `--file_path`:

```bash
# Import a trace into a logdir with an automatic or custom run name
xprof upload_trace --file_path=/path/to/trace.xplane.pb --logdir=/path/to/logdir --run_name=my_run
```

### Parameters

*   `--file_path` (required): Path to the source trace file (`.xplane.pb` or
    `.xspace.pb`).
*   `--logdir` (optional): Path to the root xprof log directory where profiles
    are organized.
*   `--run_name` (optional): The session/run directory name under
    `plugins/profile/` (defaults to `imported_trace`).

--------------------------------------------------------------------------------

## JSON Response

The tool outputs a structured JSON summary of the imported trace:

```json
{
  "status": "success",
  "message": "Successfully imported trace to run 'my_run'",
  "run_name": "my_run",
  "run_path": "/path/to/logdir/plugins/profile/my_run",
  "imported_file": "/path/to/logdir/plugins/profile/my_run/trace.xplane.pb"
}
```

--------------------------------------------------------------------------------

## Example Usage

If the user says: "Import this trace file `/tmp/worker0.xplane.pb` into my
experiment logdir `/tmp/logs` under run name `step_100`", you should:

1.  Run the CLI command:

    ```bash
    xprof upload_trace --file_path=/tmp/worker0.xplane.pb --logdir=/tmp/logs --run_name=step_100
    ```
2.  Verify the JSON response and confirm that the trace is accessible at
    `/tmp/logs` for downstream analysis (e.g., `xprof get_overview /tmp/logs`).
