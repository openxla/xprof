---
name: xprof
description: >-
  Central entry point for ALL XProf operations and analyses. Use this skill first for any task involving XProf traces, performance, memory, HLO ops, collecting/triggering XProf profiles, or numerical correctness verification.
---

> ⚠️ **CRITICAL RULES** * **ALWAYS** use `find_session` first to get the correct
> `session_id` if not provided (as other tools in this skill require a valid
> session ID). * **NEVER** block on long-running operations (e.g., generating
> Events DB). Delegate to a subagent.

# xprof

This skill consolidates various tools related to XProf operations and analysis.

## CLI

The primary tool for interacting with XProf data is the XProf CLI. This is
bundled along with the `xprof` PyPI package.

```bash
xprof <subcommand> [flags]
```

## Discovery of Workflows

**CRITICAL for Agents**: Many advanced workflows (like diffing sessions or
mapping architecture blocks) are documented in this skill's markdown files but
are NOT visible by running `xprof -h`.

-   **DO NOT rely solely on `xprof -h`** to discover capabilities.
-   **Always read the
    [Supported Capabilities](#supported-capabilities--references)** section
    below and the linked reference files to find complex analysis workflows.

## Referring to Sessions

Most subcommands require a `logdir` and a `session_id`. You can find the
`session_id` by looking in the XProf UI and seeing the run name, or by looking
at run directories within the logdir.

## Best Practices

-   **Asynchronous Execution**: Some operations, like running complex queries,
    can take a long time. For these, consider spawning a subagent. However, for
    quick lookups like `get_hosts` or `get_overview`, execute the command
    directly to avoid unnecessary overhead.
-   When running commands that get sent to the background as tasks, **DO NOT**
    attempt to guess the log file path or use `grep` manually to poll for
    completion.

## Workflows

### Bottleneck Analysis

When asked to find performance bottlenecks for a session: 1. **Execute**
`get_overview` to identify the high-level breakdown (Compute vs Host vs
Communication). 2. **Verify** if the workload is compute-bound or bound by other
factors. 3. **Execute** `get_hlo_op_profile` to find expensive operations if
compute-bound. 4. **Execute** `aggregate_xplane_events` if timeline analysis is

1.  **Inspect** HLO code using `list_hlo_modules` and `get_hlo_module_content`
    for suspect modules.
2.  **Report** findings directly to the user with concrete data points derived
    from the analysis.

<h2 id="supported-capabilities--references">Supported Capabilities & References</h2>

-   **[Numerical Verification](references/numerical_correctness.md)**: Verify
    numerical equivalence between baseline and candidate kernels using
    multi-regime distributions, discrete bounded indices, monotonic segment IDs,
    boolean masks, and ULP distance metrics.
