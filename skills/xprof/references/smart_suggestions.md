# XProf Smart Suggestions & Optimization Reference

This reference documents the heuristics and rules used to identify performance
bottlenecks and suggest optimizations across dynamic runtime metrics and static
HLO graph patterns.

--------------------------------------------------------------------------------

## 1. Dynamic Metric-Based Bottleneck Rules

Use `xprof get_overview` and `xprof get_kpi_metrics` to extract runtime
utilization metrics, then apply the following standard rules:

### Compute Bound (`ComputeBoundRule`)

*   **Condition**: MXU / TensorCore utilization > `70%` and Memory (HBM)
    bandwidth utilization < `50%`.
*   **Description**: Primary bottleneck is raw mathematical processing power of
    the hardware.
*   **Remedy**: Maximize arithmetic intensity, increase batch sizes, or reduce
    precision (e.g., FP32 $\to$ BF16/FP8).

### Memory Bound (`MemoryBoundRule`)

*   **Condition**: Memory (HBM) bandwidth utilization > `70%` and Compute
    utilization < `50%`.
*   **Description**: Processors spend significant time waiting for memory access
    to complete.
*   **Remedy**: Fuse memory-bandwidth-bound operations, align tensor dimensions
    with hardware memory burst sizes (multiples of 128), or enable
    rematerialization.

### Input / Host Bound (`InputBoundRule` & `HostProcessingBoundRule`)

*   **Condition**: Input pipeline wait time > `10%` of step time, or non-enqueue
    time >= `50%` of input time.
*   **Description**: Accelerator is starved while waiting for data preprocessing
    or host dispatch.
*   **Remedy**: Optimize host dataset pipelines using prefetching
    (`tf.data.AUTOTUNE`), parallelize `map` transforms, and batch data before
    transfers.

### Barrier / Synchronization Bound (`BarrierCoresRule`)

*   **Condition**: Barrier / synchronization event time >= `10%` of step time.
*   **Description**: Indicates worker imbalance, stragglers, or excessive
    cross-chip synchronization barriers.

### TensorCore / Device Idle Time (`TensorCoreIdleBoundRule`)

*   **Condition**: Latency-bound operations cause Device Idle Time > `10%` of
    step time.
*   **Description**: Compute cores spend significant time idle between kernel
    launches.
*   **Remedy**: Reduce host dispatch overhead, fuse consecutive small kernels,
    and use CUDA/TPU stream pipelining.

### Collective Communication Bound (`CollectiveBoundRule`)

*   **Condition**: Collective operations (e.g., `AllReduce`, `AllGather`,
    `ReduceScatter`) >= `30%` of step time.
*   **Description**: Inter-device interconnect bandwidth or network latency is
    the primary bottleneck.
*   **Remedy**: Overlap communication with computation (async collectives), tune
    mesh topology dimensions, or use pipeline parallelism.

### Data Shuffle & Non-Sequential Access Bound (`DataShuffleBoundRule`)

*   **Condition**: Non-sequential operations (sort, gather, scatter, slice) >=
    `30%` of step time.
*   **Description**: Non-contiguous memory accesses bottleneck hardware
    pipelines.

### Data Transfer Bound (`DataTransferBoundRule`)

*   **Condition**: Host-to-device transfer time >= `30%` of step time.
*   **Description**: PCIe bus transfers delay kernel execution.
*   **Remedy**: Use pinned memory and asynchronous host-to-device transfers.

### Debug Printing Overhead (`DebugPrintRule`)

*   **Condition**: `debug_print` event time >= `5%` on any host or device.
*   **Description**: Unintentional debug print calls (`jax.debug.print`,
    `tf.print`, or host logger calls inside loops) cause synchronous
    busy-waiting and stall distributed worker synchronization.
*   **Remedy**: Disable debug printing, callbacks, and verbose logging inside
    compiled step functions.

--------------------------------------------------------------------------------

## 2. Static HLO Optimization Patterns

These optimization patterns target common anti-patterns in compiled HLO graphs
and JAX/XLA programs.

### Minimize Unnecessary Data Type Conversions

*   **Pattern**: Unnecessary promotion of `bf16` or `f16` to `f32` in reduction
    operations (`jnp.mean`, `jnp.sum`, `jnp.var`, `jnp.std`), followed by a
    downcast back to `bf16`/`f16`. This increases intermediate memory bandwidth
    by $2\times$ and adds explicit conversion ops.
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
    preceding a matmul (`dot`, `dot-general`, or `custom-call`). When un-fused,
    this forces **materialization of explicit intermediate tensors in device
    memory**, wasting memory bandwidth.
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

### Optimize Data Layout for Accelerators

*   **Pattern**: Non-optimal minor dimensions causing layout mismatch copies or
    sub-lane hardware padding.
*   **Remedy**: Align tensor dimensions to hardware lane sizes (e.g., multiples
    of 128 elements for F32/BF16 on TPU, multiples of 32 on GPU).

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
