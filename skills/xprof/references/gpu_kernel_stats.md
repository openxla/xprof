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

None of this time is in `total_device_duration_us` or in the per-kernel rows, so
never count it as kernel time. To check whether transfers slow the kernels
down, compare `wall_duration_in_kernel_span_us` with `total_device_duration_us`,
not with `kernel_span_us`, and report the ratio. Never compare
`total_duration_us` with the kernel time: it counts overlapping streams more
than once, and transfers can run long before or after the kernels (for example
weight loads). For example, on GPU 2 of a 4x A100 training trace, the kernels
took 42.0 ms in a 48.5 ms kernel span. The 1290 memcpy/memset events summed to
1.9 s over 3 overlapping copy streams in a 12 s capture, but only 310 us of them
fell inside the kernel span: about 0.7% of the 42.0 ms of kernel time. So the
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
| `occupancy_pct` | Theoretical occupancy from the profiler; `null` when it was not computed (for example on Blackwell traces). Not achieved occupancy. Weigh a low value against the kernel's share of `total_device_duration_us` (see caveats). |
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
-   Low theoretical occupancy is not a problem by itself. Tuned GEMM and
    attention kernels often run at low occupancy by design, because they keep
    many registers and much shared memory per thread to reuse data. To judge a
    low value, compute the kernel's share of `total_device_duration_us` and
    state it: that share is the most time an occupancy fix can save. Then check
    achieved throughput before you recommend fewer registers or smaller tiles.
    If the kernel's own duration is not known, get it first.
-   `ncclDevKernel_*` kernels are collectives; their time includes waiting for
    other GPUs.
