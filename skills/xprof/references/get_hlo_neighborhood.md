# `get_hlo_neighborhood` Reference

This reference explains how to get the BFS neighborhood of an HLO instruction.
Use this for compiler performance debugging (identifying potential fusion
blockers) by inspecting BFS neighborhoods for bitcasts, copies, or layout
mismatches.

## Prerequisites

-   You must have the log directory path (`<logdir>`), direct run directory, or
    session ID for the specific XProf run you are attempting to analyze.
-   You need the name of the HLO operation to center the neighborhood around.

## Instructions

1.  Use the `xprof` command-line tool, providing the log directory and
    `instruction_name`:

    ```bash
    xprof get_hlo_neighborhood <logdir> --instruction_name=<instruction_name>
    ```
2.  You can optionally specify a radius (default is 2), a specific HLO module
    name, and whether to include source metadata:

    ```bash
    xprof get_hlo_neighborhood <logdir> \
      --instruction_name=<instruction_name> \
      --radius=<radius> \
      [--module_name=<module_name>] \
      [--print_metadata=<True|False>]
    ```
3.  Review the output, which provides the HLO instructions in the BFS
    neighborhood.

## Example Usage

If the user says: "Show me the neighborhood of op 'fusion.123' for
/path/to/logdir", you should:

1.  Run the CLI command:

    ```bash
    xprof get_hlo_neighborhood /path/to/logdir --instruction_name=fusion.123
    ```
2.  Present the neighborhood content to the user.
