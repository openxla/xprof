# 7-Phase Performance Analysis Protocol

This reference provides the authoritative 7-phase procedure for analyzing XProf
profiling sessions (Phases 1-6 for bottleneck diagnosis and verification, and
Phase 7 for artifact closure). Follow this protocol whenever investigating
compute, memory, host, or step-time bottlenecks.

## Tool Execution

All examples use the open-source `xprof` CLI. Install it via pip (use
`xprof-nightly` to access experimental subcommands such as
`verify_numerical_parity`):

```bash
pip install xprof-nightly
```

All tools accept a `<logdir>` (or a direct run folder / `.xplane.pb` file) as a
positional argument:

```bash
xprof <subcommand> <logdir> [flags]
```

--------------------------------------------------------------------------------

## Phase 1: Turn-1 Parallel Triage Dispatch

To eliminate turn latency and avoid circular searches, agents **MUST issue
parallel tool calls in Turn 1**:

```bash
# Executed concurrently in Turn 1:
xprof get_overview <logdir>
xprof get_roofline_model <logdir> --top_n=10
xprof check_host_boundness <logdir>
```

### Triage Decision Matrix

Evaluate the consolidated outputs against the following four gates:

| Diagnostic State        | Trigger Conditions      | Action                  |
| :---------------------- | :---------------------- | :---------------------- |
| **Host / Infeed Bound** | `check_host_boundness`  | Proceed to Phase 3 (EIC |
:                         : returns `HOST_BOUND`    : Calculation) & Phase 4  :
:                         : (Idle Time Ratio >      : (Input Pipeline         :
:                         : 10.0%, MXU Idleness >   : Proposal)               :
:                         : 70.0%, HBM BW < 30.0%,  :                         :
:                         : ICI < 30.0%)            :                         :
| **Memory-Bound**        | `bound_by == "HBM"`,    | Proceed to Phase 2      |
:                         : Operational Intensity < : (Macro-to-Micro HLO     :
:                         : Ridge Point (e.g. <     : Drilldown)              :
:                         : 279.1 FLOP/Byte on TPU  :                         :
:                         : v5e)                    :                         :
| **Compute-Bound**       | `bound_by ==            | Proceed to Phase 2      |
:                         : "Compute"`, Operational : (Macro-to-Micro HLO     :
:                         : Intensity > Ridge       : Drilldown)              :
:                         : Point, high MXU compute :                         :
:                         : efficiency              :                         :
| **Communication-Bound** | High collective time    | Proceed to Phase 2      |
:                         : (`all-reduce`,          : (Communication Category :
:                         : `all-gather`,           : Drilldown)              :
:                         : `all-reduce-scatter     :                         :
:                         : fusion`) in op profile  :                         :
:                         : or timeline stalls      :                         :

--------------------------------------------------------------------------------

## Phase 2: Macro-to-Micro Op Drilldown

If compute, memory, or communication bound, execute progressive macro-to-micro
drilldown:

### Step 2.1: Macro Category Summary

```bash
xprof get_hlo_op_profile <logdir> --view=category
```

Identify the dominant category with highest fraction of total execution time
(`convolution fusion`, `loop fusion`, `custom-call`, `data formatting`).

### Step 2.2: Category Drilldown & Source Attribution

```bash
xprof get_hlo_op_profile <logdir> --category="<dominant_category>"
```

Isolate leaf operations, their self-time, FLOPs, bytes accessed, and source code
mapping (`source_file` and `source_line`).

> ⚠️ **STRICT ANTI-PATTERN**: Do NOT dump raw `.hlo` text or protobuf files. Use
> `get_hlo_neighborhood` with `--op_name` if subgraph inspection is required.

--------------------------------------------------------------------------------

## Phase 3: Headroom & Resource Waste Quantification

Quantify optimization potential and hardware waste:

<!-- disableFinding(LINE_OVER_80) -->

### 1. Roofline Headroom Percentage

$$\text{Headroom \%} = 100.0\% - \text{Roofline Efficiency \%}$$

### 2. Theoretical Step Latency Reduction ($\Delta \text{ms}$)

$$\Delta \text{Step Latency (ms)} = \text{Op Self Time (ms)} \times \left(\frac{\text{Headroom \%}}{100.0\%}\right)$$

### 3. Equivalent Idle Chips (EIC) (for Host / Infeed Stalls)

$$\text{EIC} = \text{Total TPU Cores} \times \left(\frac{\text{Idle Time Ratio}}{1.0 + \text{Idle Time Ratio}}\right)$$

<!-- endDisableFinding(LINE_OVER_80) -->

--------------------------------------------------------------------------------

## Phase 4: Actionable Code Optimization Proposal

Agents MUST provide concrete, line-level code or configuration modifications:

-   **Memory-Bound Optimization**:
    -   Fuse layout conversions and reshapes using `jnp.einsum`.
    -   Eliminate unnecessary fp32 upcasting in reduction operations (enforce
        `dtype=jnp.bfloat16` or `astype(x.dtype)`).
    -   Eliminate intermediate memory materializations.
-   **Compute-Bound Optimization**:
    -   Tune kernel block sizes / tiling dimensions via autotuning sweeps over
        candidate block-size and tiling configurations.
    -   Reorder operations or apply Split-K matrix decomposition.
-   **Host / Infeed-Bound Optimization**:
    -   Apply `tf.data.AUTOTUNE` and `.prefetch()`.
    -   Enable host software offload or parallel data loading workers.

--------------------------------------------------------------------------------

## Phase 5: Empirical Validation Execution

Agents MUST formulate a copy-pasteable, deterministic execution command to
benchmark performance. Use standard reproduction commands, for example:

```bash
# Run a standalone benchmark script with a fixed reproduction config.
python -m benchmark_script --config=repro_config.py

# Or execute a performance regression test.
pytest tests/test_perf.py

# Or run a Bazel benchmark target in optimized mode.
bazel run -c opt //path/to:benchmark_target -- --config=repro_config.py
```

-   Assert post-refactoring step latency, verify speedup, and check for
    throughput regressions.

--------------------------------------------------------------------------------

## Phase 6: Numerical Correctness & Parity Verification

Never recommend or merge performance changes without enforcing numerical
correctness contracts:

```bash
xprof verify_numerical_parity \
  --reference_kernel="module.ref_func" \
  --candidate_kernel="module.cand_func" \
  --shape="(32, 2048)" \
  --dtype="bfloat16" \
  --max_allowed_ulp=2
```

### Mandatory Parity Guardrails:

1.  **Multi-Regime Testing**: Evaluate normal distribution, extreme boundary
    values ($0$, $\pm \infty$), and heavy-tailed / activation overflow regimes.
2.  **Tolerance Contracts**:
    -   Elementwise / direct fusions: $\text{max\_allowed\_ulp} \le 2$.
    -   Split-K non-associative reductions: $\text{max\_allowed\_ulp} \le 4$.
    -   Discrete indices & masks: $\text{discrete\_delta} = 0$,
        $\text{mismatch\_count} = 0$.
3.  **Safety Ceiling**: Reject tolerance abuse (hard ceiling of 8 ULP for
    bfloat16).

--------------------------------------------------------------------------------

## Phase 7: Artifact Closure

Conclude analysis with concrete operational deliverables:

-   Produce a ready-to-submit Git commit message / pull request summary
    describing the optimization and measured speedup.
-   Share execution logs and findings as a Markdown report (e.g. a Gist or
    attached artifact).
-   Provide the local or cluster rerun command required to reproduce the
    verification results.
