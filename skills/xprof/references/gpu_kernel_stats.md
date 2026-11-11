# GPU Kernel Stats Reference

How to read `get_kernel_stats` on GPU profiles (device planes named
`/device:GPU:N`), and which of its GPU fields to trust.

## Pick the device

```bash
xprof get_kernel_stats <logdir> --include_summary=True            # first GPU
xprof get_kernel_stats <logdir> --device_to_use=GPU:3 --include_summary=True
xprof get_kernel_stats <logdir> --device_to_use=all --include_summary=True
```

-   By default one device on one host is aggregated: `/device:GPU:0` on GPU
    profiles (the first TPU chip on TPU profiles). The summary reports it in
    `device` and `device_planes`.
-   Multi-host profiles: when the source is a run directory with one trace per
    host, the default and any single `--device_to_use` read only the first
    host's trace. The summary reports `host` and `num_hosts`, and a stderr
    note names the host. Pass `--host=<name>` for another host. A single
    XSpace that holds several hosts keeps only the first copy of each device
    plane, and `host` is `null`. Remote TPU sessions read only the first host
    partition of the session.
-   `--device_to_use=all` sums every device on every host, so times scale with
    the device count (about 8x on one 8-GPU host). Use it only when you want
    totals over all devices.
-   An unknown device fails and lists the available ones.
-   If the summary has no `device` field, the `xprof` build predates per-device
    selection and sums every device plane; divide by the GPU count or upgrade.

## What counts as a kernel

Only events with launch parameters (`kernel_details`) are kernels. Memcpy and
memset events share CUDA streams with kernels. They are not ranked and not
counted in `total_device_duration_us`. The summary reports them under
`excluded_gpu_non_kernel_events`:

-   `count` and `total_duration_us`: the sum over every stream. Streams overlap,
    so this sum can be larger than the capture.
-   `wall_duration_us`: their union in time over the whole capture.
-   `wall_duration_in_kernel_span_us` and `kernel_span_us`: their union between
    the first kernel start and the last kernel end, and the length of that
    span.
-   `by_kind`: the same counts for `memcpy` and `memset`.

Compare `wall_duration_in_kernel_span_us` with `total_device_duration_us` and
state the ratio in your answer. Never compare `total_duration_us` with the
kernel time: it counts overlapping streams more than once, and transfers can
run long before or after the kernels (for example weight loads). On GPU 0 of an
8x H100 inference trace, the kernels took 17.5 ms in a 21.3 ms kernel span. The
818 memcpy/memset events summed to 754.7 ms, over 4 overlapping copy streams in
a 7.2 s capture, but only 46.6 us of them fell inside the kernel span. So the
transfers did not slow the kernels down.

Other GPU events are not counted:

-   Processed sessions add derived lines (`XLA Ops`, `Steps`, `Framework Ops`,
    `Source code` and others) to GPU planes. Their events repeat the stream
    events, so they are skipped and listed in `excluded_gpu_derived_lines`.
-   Stream events with neither launch parameters nor memcpy/memset details are
    skipped.
-   With `--kernel_name` or `--trace_matchers`, `total_device_duration_us`
    covers only the matched kernels. `excluded_gpu_non_kernel_events` is then
    omitted, and `excluded_gpu_non_kernel_events_note` says why. Run without
    the filters to compare transfers with kernels.

## Per-kernel GPU fields

| Field | Meaning |
| --- | --- |
| `registers_per_thread` | Registers per thread of the launch. |
| `static_shmem_bytes`, `dynamic_shmem_bytes`, `shmem_bytes` | Shared memory per block (static, dynamic, total). |
| `block_dim`, `grid_dim` | Launch dimensions `[x, y, z]`. |
| `occupancy_pct` | Theoretical occupancy from the profiler; `null` when it was not computed (for example on Blackwell traces). Not achieved occupancy. |
| `num_launch_configs`, `launch_config_share_pct` | A kernel launched with several configurations reports the one with the most time and its share of the kernel's time. |
| `is_kernel_using_tensor_core`, `is_op_tensor_core_eligible` | Kernel-name and op-name heuristics (see caveats). |
| `hlo_op_name` | The HLO op with the most launch time, when the trace records it. `num_hlo_ops` is set when several HLO ops launched the kernel. |

`--output_format=markdown` adds matching columns only when GPU records are
present; TPU output is unchanged.

## Caveats

-   The TensorCore flags are name heuristics. cuBLASLt `nvjet_*`, Triton
    `gemm_fusion_*` and cuDNN SDPA kernels report `false` even though they use
    TensorCores. Never conclude that TensorCores are unused from these flags
    alone.
-   Low theoretical occupancy is normal for tuned GEMM and attention kernels.
    Treat it as a lead only when the kernel also has a large share of time.
-   `ncclDevKernel_*` kernels are collectives; their time includes waiting for
    other GPUs.

## Smart suggestions

`get_smart_suggestions` includes `GpuLowOccupancyRule`. It fires when kernels
with `occupancy_pct` below 25% take at least 10% of the GPU kernel time whose
occupancy is known, and lists the three longest such kernels. Matrix
multiplication and attention kernels are not counted: kernels with a
TensorCore flag, or with a name that contains `gemm`, `gemv`, `matmul`, `dot`,
`nvjet`, `cublas`, `cutlass`, `xmma`, `wgmma`, `flash`, `fmha`, `mha`, `attn`,
`attention` or `sdpa`. The suggestion gives their low-occupancy share of time
separately. Kernels without occupancy data are ignored, so the rule stays
silent on traces where the profiler did not compute occupancy. TPU profiles are
unaffected. Before you recommend fewer registers or smaller tiles for a listed
kernel, check that it is not a GEMM or attention kernel with a name the
patterns miss.
