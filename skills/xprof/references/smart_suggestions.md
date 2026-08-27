# XProf Smart Suggestions & Optimization Reference

This reference documents the heuristics and rules used to identify performance
bottlenecks and suggest optimizations across dynamic runtime metrics and static
HLO graph patterns.

--------------------------------------------------------------------------------

## 1. Dynamic Metric-Based Bottleneck Rules

The open-source smart-suggestion engine exposes a single dynamic, metric-based
bottleneck rule. Use `xprof get_overview` and `xprof get_kpi_metrics` to extract
the runtime utilization metrics this rule relies on.

### Barrier / Synchronization Bound (`BarrierCoresRule`)

*   **Condition**: Barrier / synchronization event time >= `10%` of step time.
*   **Description**: Indicates worker imbalance, stragglers, or excessive
    cross-chip synchronization barriers.

--------------------------------------------------------------------------------

## 2. Static HLO Optimization Patterns

These optimization patterns target common anti-patterns in compiled HLO graphs
and JAX/XLA programs. Detect them by inspecting the HLO graph with the core
tools `xprof get_top_hlo_ops` (to find expensive ops) and `xprof
get_hlo_neighborhood <logdir> --instruction_name=<op>` (to inspect the localized
fusion neighborhood of a suspect op).

> [!WARNING] Do **not** dump the full HLO module to search for these patterns —
> it will exceed token limits and time out. Inspect only localized neighborhoods
> around ops flagged by `get_top_hlo_ops`.

### Minimize Unnecessary Data Type Conversions

*   **Pattern**: Unnecessary promotion of `bf16` or `f16` to `f32` in reduction
    operations (`jnp.mean`, `jnp.sum`, `jnp.var`, `jnp.std`), followed by a
    downcast back to `bf16`/`f16`. This increases intermediate memory bandwidth
    by $2\times$ and adds explicit conversion ops.
*   **HLO signature**:
    1.  Upcast: `%convert = f32[...] convert(%input_bf16_or_f16)`
    2.  Reduce: `%reduce = f32[...] reduce(%convert, ...)`
    3.  Downcast: `%downcast = bf16/f16[...] convert(%reduce)` (often inside a
        downstream fusion)
*   **Detection**: Check `xprof get_top_hlo_ops` for expensive `reduce` and
    `convert` instructions. Inspect the fusion neighborhood with `xprof
    get_hlo_neighborhood <logdir> --instruction_name=<op>`.
*   **Remedy**: Explicitly set the reduction `dtype` to match input precision:
    *   **Inefficient**: `mean_x = jnp.mean(x, axis=0)` (may default to `f32`
        accumulation)
    *   **Optimized**: `mean_x = jnp.mean(x, axis=0, dtype=x.dtype)` (forces
        reduction in native precision)

### Fold Reshapes and Transposes into Compute Operations

*   **Pattern**: Standalone `reshape`, `transpose`, or `copy` operations
    preceding a matmul (`dot`, `dot-general`, or `custom-call`). When un-fused
    (not inside a `fused_computation`), this forces **materialization of
    explicit intermediate tensors in device memory**, wasting memory bandwidth.
*   **HLO signature**: `%reshape.123 = f32[1000, 2048] reshape(%input)` feeding
    directly into `%dot.456 = f32[1000, 4096] dot-general(%reshape.123,
    %weights)`.
*   **Detection**: Inspect the neighborhood of matmul fusions with `xprof
    get_hlo_neighborhood <logdir> --instruction_name=<op>`.
*   **Remedy**: Express the multi-dimensional layout transform directly inside
    `jnp.einsum`:

    *   **Inefficient**:

    ```python
    k_reshaped = jnp.reshape(K, new_shape)
    attention_logits = jnp.einsum('qnh, kh -> nqk', Q, k_reshaped)
    ```

    *   **Optimized**:

    ```python
    # Fold logical reshape/transpose directly into the einsum indices
    attention_logits = jnp.einsum('qnh, psdh -> nqpsd', Q, K)
    ```

### Optimize Data Layout for Accelerators (Sandwiched Layout Mismatch Copies)

*   **Pattern**: `copy` operations causing physical layout reordering (layout
    mismatch) sandwiched between compute-intensive stages (convolution, dot,
    reduce, scatter, gather, or compute fusions), or `copy` operations with
    non-optimal minor-most dimensions (not multiples of the hardware lane size)
    causing padding / alignment overhead.
*   **Remedy**: Align tensor dimensions to hardware lane sizes (e.g., multiples
    of 128 elements for F32/BF16 on TPU, multiples of 32 on GPU). Move
    transposes or dimension reorderings earlier in the model. Consider a
    "head-along-lane" layout for KV-cache updates.

### Consolidate Small Updates into Fused Functions

*   **Pattern**: Multiple consecutive small operations (e.g. updating key,
    value, and scaling factors independently in Multi-Head Attention) executed
    as isolated kernels. This disrupts XLA layout propagation and forces
    redundant layout transformations.
*   **Remedy**: Combine logically related tensor transformations into a single
    JAX/Python function or fused block, enabling the compiler to propagate
    optimal tensor layouts across the entire attention computation.

> [!TIP] **Fused Instruction Resolution**: If `get_hlo_neighborhood` reports
> that an instruction name from an error trace or log does not exist, it was
> likely folded into a fusion computation. Inspect top fusion ops (`fusion.*`)
> via `get_top_hlo_ops` instead of searching for the raw individual op.

### Compiler Tuning Flags

For complex workloads, consider evaluating standard XLA compiler flags:

*   Enable aggressive memory scavenging when approaching OOM limits:
    `--xla_tpu_vmem_scavenging_mode=AGGRESSIVE`
*   Enable asynchronous collective fusion:
    `--xla_tpu_enable_async_collective_fusion=true`

--------------------------------------------------------------------------------

## References

*   [XProf documentation](https://openxla.org/xprof)
*   [Overview Page](https://openxla.org/xprof/overview_page) — surfaces
    bottleneck insights and smart suggestions
*   [HLO Op Profile](https://openxla.org/xprof/hlo_op_profile) and
    [HLO Op Stats](https://openxla.org/xprof/hlo_op_stats) — inspect expensive
    HLO operations for the static patterns above
*   [Roofline Model](https://openxla.org/xprof/roofline_model) — classify
    compute- vs memory-bound bottlenecks
*   [openxla/xprof source repository](https://github.com/openxla/xprof)
