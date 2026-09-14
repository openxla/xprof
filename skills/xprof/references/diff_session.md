# XProf Session Diff Guide

This reference provides a procedure for performing a performance diff analysis
between two XProf trace sessions (e.g., baseline run vs. optimized candidate
run).

--------------------------------------------------------------------------------

## Prerequisites

*   **Baseline Trace Path / Logdir**: Path to the baseline profile directory or
    `.xplane.pb` file.
*   **Candidate Trace Path / Logdir**: Path to the candidate/optimized profile
    directory or `.xplane.pb` file.

--------------------------------------------------------------------------------

## Procedure

### CRITICAL: Parallel Diagnostics Invocation

To minimize turns and execution latency, **always launch independent baseline
and candidate diagnostic tool calls in parallel**:

-   Run `get_overview` for both baseline and candidate in parallel.
-   Run `get_kpi_metrics` for both sessions in parallel.
-   Run `get_kernel_stats` for both sessions in parallel.

### 1. Verify Workload Configuration Parity

> [!CRITICAL] Before comparing performance metrics (like latency or step time),
> **ALWAYS** verify that the workload configurations are identical (batch size,
> sequence lengths, `max_decodes`, dtypes, hardware slice topology). Comparing
> sessions with different parameters leads to false optimization conclusions.

*   Inspect `frontend_attributes` (such as `problem_config`, `kernel_config`,
    `batch_size`, `q_len`, `max_decodes`) in top custom calls via
    `get_graph_viewer` or `get_hlo_text`.
*   If configurations differ, state this clearly in your report as the primary
    reason for latency variations.

--------------------------------------------------------------------------------

### 2. High-Level Profiling Diff (`get_overview` & `get_kpi_metrics`)

Compare high-level metrics between both sessions:

```bash
# 1. Compare execution overview (Compute vs. Host vs. Communication)
xprof get_overview /path/to/baseline_logdir
xprof get_overview /path/to/candidate_logdir

# 2. Compare KPI metrics (step time, duty cycle, MXU utilization)
xprof get_kpi_metrics /path/to/baseline_logdir
xprof get_kpi_metrics /path/to/candidate_logdir
```

--------------------------------------------------------------------------------

### 3. Step Time & Kernel Duration Diff (`get_kernel_stats`)

Extract precise kernel execution durations and host overhead without requiring
raw database queries:

```bash
# 1. Compare active hardware step times and kernel distributions
xprof get_kernel_stats /path/to/baseline_logdir --include_summary=True
xprof get_kernel_stats /path/to/candidate_logdir --include_summary=True

# 2. Inspect specific JIT module execution times
xprof get_kernel_stats /path/to/baseline_logdir --kernel_name="jit_update"
xprof get_kernel_stats /path/to/candidate_logdir --kernel_name="jit_update"
```

> [!IMPORTANT] **CRITICAL: Host Overhead Masking Rule** When host-side dispatch
> latency (e.g. 2–3 ms) is orders of magnitude larger than device execution time
> (e.g. 50–100 µs), the host overhead **MASKS** the device-level algorithmic
> speedup in end-to-end wall-clock latency. In your final report, you **MUST
> explicitly state** whether host overhead masks device execution gains.

--------------------------------------------------------------------------------

### 4. Top Operations Comparison (`get_top_hlo_ops`)

Identify which HLO operations shifted in duration, FLOPs, or memory bandwidth:

```bash
xprof get_top_hlo_ops /path/to/baseline_logdir --limit=15
xprof get_top_hlo_ops /path/to/candidate_logdir --limit=15
```

--------------------------------------------------------------------------------

### 5. Targeted HLO Graph & Text Diffing

To inspect structural graph differences for suspect modules:

```bash
# 1. Export HLO text from both sessions for comparison
xprof get_hlo_text /path/to/baseline_logdir --module_name=<module_name> --path=/tmp/baseline.hlo
xprof get_hlo_text /path/to/candidate_logdir --module_name=<module_name> --path=/tmp/candidate.hlo

# 2. Diff the exported HLO text
diff -u /tmp/baseline.hlo /tmp/candidate.hlo

# 3. Inspect localized neighborhoods around modified operations
xprof get_hlo_neighborhood /path/to/candidate_logdir --instruction_name=<op_name> --radius=2
```

--------------------------------------------------------------------------------

### 6. Numerical Verification (`verify_numerical_parity`)

If the optimization changed kernel implementations (e.g. custom Pallas or Triton
kernels, einsum fusions):

```bash
xprof verify_numerical_parity \
  --kernel_ref="module.baseline_fn" \
  --kernel_candidate="module.candidate_fn" \
  --shapes="[(16, 1024), (1024, 1024)]" \
  --dtype_str="bfloat16" \
  --tier="fast_agent"
```

--------------------------------------------------------------------------------

## Gotchas & Analytical Tips

*   **Early Stopping on Identical Graphs**: If the exported HLO module text diff
    shows 0 structural differences, stop further execution analysis and report
    that the compiler optimization did not alter the generated graph.
*   **Avoid Massive HLO Dumps**: Use `get_hlo_neighborhood` or
    `get_graph_viewer` rather than dumping entire multi-megabyte modules to
    prevent context window saturation.
*   **Host vs. Device Attribution**: Always separate host launch latency from
    compiled device kernel runtime.
