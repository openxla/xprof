# `get_utilization_viewer` Reference

Fetches utilization viewer data for a given XProf session and returns key
metrics in JSON format.

## Prerequisites

-   You must have the log directory path (`<logdir>`), direct run directory, or
    session ID for the specific XProf run you are attempting to analyze.

## Instructions

Run the CLI command, including optional flags to filter by host, device, or
node:

```bash
xprof get_utilization_viewer <logdir> \
   [--host=<HOST>] \
   [--device=<DEVICE>] \
   [--node=<NODE>]
```

### Arguments

-   `<logdir>` or `--session_id`: The XProf log directory or session ID.
-   `--host` (optional): The host ID to filter by (default is 0).
-   `--device` (optional): The device ID to filter by (default is 0).
-   `--node` (optional): The node ID to filter by (default is 0).

**Note:** Do not include optional flags if you do not have specific values for
them.
