
# Roofline Model Analysis (`get_roofline_model`)

This reference explains how to fetch and interpret Roofline Model analysis from
XProf, to diagnose compute vs. memory-bandwidth bottlenecks at both the program
level and the per-operation level.

> [!IMPORTANT]
>
> `get_roofline_model` is **not in the released wheel** (`xprof==2.23.1` and
> earlier). It requires a nightly:
> `pip install 'xprof-nightly>=2.24.1a20260827'`.
> If `xprof get_roofline_model` reports an unknown command, you are on a stable
> wheel — check with `xprof -h` and count the subcommands.

--------------------------------------------------------------------------------

## 1. Prerequisites & Invocation

You need the log directory path (`<logdir>`), a direct run directory, or a
session ID for the run you are analyzing. See
[Referring to Sessions & Inputs](../SKILL.md#referring-to-sessions--inputs) for
all accepted forms.

```bash
xprof get_roofline_model <logdir_or_run_dir_or_pb> [--top_n=15]
```

Or with explicit named flags:

```bash
xprof get_roofline_model --logdir=<path> \
  --session_id='"<run_name>"' [--top_n=15]
```

### Flags

| flag | default | what it does |
|---|---|---|
| `--top_n` | `15` | Number of operations returned in `top_operations`. |
| `--bypass_cache` | `False` | Re-fetches instead of serving cached result. |
| `--group_by` | `"program"` | **Inert.** Accepted and immediately discarded. |

> [!WARNING]
>
> **`--group_by` does nothing.** It appears in `xprof get_roofline_model -h` and
> advertises `'program'` or `'step'`, but the implementation discards it before
> use (`del group_by`). Passing `--group_by=step` returns whole-session program
> metrics, not per-step ones, and reports no error. Do not build a per-step
> analysis on it — use `list_xplane_events --max_events=200000` instead.

> [!WARNING]
>
> **Results are cached for 24 hours.** If you profile, change something, and
> re-profile into the **same** session, a second call returns the first call's
> answer. Pass `--bypass_cache=True` whenever you are comparing before/after on
> a session path you have re-written.

> [!NOTE]
>
> **Quote session IDs.** Ids like `2026_08_24_04_28_33` are valid Python integer
> literals and the CLI's argument parser coerces them. Always double-quote
> inside single quotes: `'"2026_08_24_04_28_33"'`.

--------------------------------------------------------------------------------

## 2. Output

Real output, captured from the MaxText/Gemma v6e-4 training trace bundled in
this repo:

```bash
xprof get_roofline_model demo/plugins/profile/v6e-4-training --top_n=2
```

```json
{
  "program": {
    "bound_by": "HBM",
    "operational_intensity_flop_per_byte": 288.0455,
    "bottleneck_operational_intensity_flop_per_byte": 387.3531,
    "roofline_efficiency_percent": "49.33%",
    "compute_efficiency_percent": "33.06%",
    "max_mem_bw_utilization_percent": "49.33%",
    "optimal_flop_rate_gflops": 634481.73,
    "dma_stall_percent": "0.00%",
    "measured_flop_rate_gflops": 312992.93,
    "model_flop_rate_gflops": 312992.93,
    "measured_memory_bw_gibs": 1011.98,
    "hbm_bw_gibs": 752.54,
    "hbm_read_bw_utilization_percent": "49.33%",
    "hbm_write_bw_utilization_percent": "49.33%",
    "cmem_read_bw_utilization_percent": "N/A",
    "cmem_write_bw_utilization_percent": "N/A",
    "vmem_read_bw_utilization_percent": "0.64%",
    "vmem_write_bw_utilization_percent": "0.68%",
    "total_time_ms": 1866.928
  },
  "device_info": {
    "cmem_read_ridge_point": 0.0,
    "cmem_write_ridge_point": 0.0,
    "device_type": "TPU v6 Lite",
    "has_cmem": 0.0,
    "has_merged_vmem": 1.0,
    "hbm_ridge_point": 577.963,
    "megacore": 0.0,
    "peak_cmem_read_bw": 0.0,
    "peak_cmem_write_bw": 0.0,
    "peak_flop_rate": 946700.0,
    "peak_hbm_bw": 1525.5,
    "peak_vmem_read_bw": 21696.1,
    "peak_vmem_write_bw": 15020.4,
    "time_scale_multiplier": 1.0,
    "vmem_read_ridge_point": 40.6379,
    "vmem_write_ridge_point": 58.699
  },
  "top_operations": [
    {
      "rank": 1,
      "name": "fusion.668",
      "category": "convolution fusion",
      "total_self_time_ms": 101.499,
      "total_self_time_percent": "5.44%",
      "operational_intensity_flop_per_byte": 445.1204,
      "bottleneck_operational_intensity_flop_per_byte": 445.1351,
      "roofline_efficiency_percent": "53.77%",
      "compute_efficiency_percent": "41.41%",
      "max_mem_bw_utilization_percent": "53.77%",
      "optimal_flop_rate_gflops": 729128.3,
      "dma_stall_percent": "0.00%",
      "bound_by": "HBM",
      "hlo_module_id": "15845321592809624413",
      "source_info": "/tmp/maxtext/src/maxtext/trainers/pre_train/train.py:666:4 -> .../absl/app.py:261:13 -> "
    },
    {
      "rank": 2,
      "name": "fusion.686",
      "category": "convolution fusion",
      "total_self_time_ms": 65.094,
      "total_self_time_percent": "3.49%",
      "operational_intensity_flop_per_byte": 433.0658,
      "bottleneck_operational_intensity_flop_per_byte": 457.125,
      "roofline_efficiency_percent": "40.79%",
      "compute_efficiency_percent": "32.26%",
      "max_mem_bw_utilization_percent": "40.79%",
      "optimal_flop_rate_gflops": 748767.62,
      "dma_stall_percent": "0.00%",
      "bound_by": "HBM",
      "hlo_module_id": "15845321592809624413",
      "source_info": "/tmp/maxtext/src/maxtext/trainers/pre_train/train.py:666:4 -> .../absl/app.py:261:13 -> "
    }
  ],
  "total_operations_analyzed": 893
}
```

`total_operations_analyzed` (893) is the full deduplicated op count, **not** the
length of `top_operations` (2). Never report `top_operations` as if it were the
whole program.

--------------------------------------------------------------------------------

## 3. Units — read this before doing any arithmetic

The single most common error with this tool is a units mismatch that no field
name warns you about:

-   `peak_hbm_bw`, `peak_vmem_*_bw`, `peak_cmem_*_bw` are in **GiB/s**
    (binary, 2^30 bytes) — despite carrying no unit suffix.
-   `operational_intensity_flop_per_byte` is per **byte** (decimal).
-   `*_gflops` rates are decimal (10^9 FLOP/s).
-   `hbm_bw_gibs` and `measured_memory_bw_gibs` do carry the `_gibs` suffix.

So any calculation combining a peak bandwidth with an operational intensity must
convert GiB/s to GB/s first:

```
peak_hbm_bw_GB = 1525.5 * 2**30 / 1e9 = 1637.99 GB/s
```

Recomputing the ridge point from the published constants:

```
946700.0 / 1637.99  = 577.963   correct, matches device_info.hbm_ridge_point
946700.0 / 1525.5   = 620.58    WRONG - skipped the conversion, off by 7.4%
```

If your recomputed ridge point disagrees with `hbm_ridge_point`, do not average
the two or pick one — you almost certainly skipped this conversion.

### Identities that hold in real output

Use these to confirm you are reading the fields correctly. All four are exact on
the capture above:

| identity | check |
|---|---|
| `roofline_efficiency` = `measured_flop_rate` / `optimal_flop_rate` | 312992.93 / 634481.73 = 49.33% |
| `compute_efficiency` = `measured_flop_rate` / `peak_flop_rate` | 312992.93 / 946700.0 = 33.06% |
| `max_mem_bw_utilization` = `hbm_bw_gibs` / `peak_hbm_bw` | 752.54 / 1525.5 = 49.33% |
| `optimal_flop_rate` = `bottleneck_OI` x `peak_hbm_bw_GB` | 387.3531 x 1637.99 = 634,482 |

Note the last one uses the **converted** GB/s figure, same as the ridge point.

--------------------------------------------------------------------------------

## 4. Deciding the bound

`bound_by` is the tool's own verdict — report it. But when you are asked to
*justify* the bound, the test is a comparison against the ridge point, not a
glance at the efficiency percentages:

```
operational_intensity < hbm_ridge_point  ->  memory bound
operational_intensity > hbm_ridge_point  ->  compute bound
```

On the capture above, OI 288.05 sits **below** the ridge of 577.96, so the
workload is HBM bound. This is worth stating explicitly because two of the
headline numbers invite the opposite conclusion: roofline efficiency is 49.33%
and compute efficiency 33.06%, and MXU utilization on this trace reads 26.7%.
High utilization is not evidence of compute-boundness.

> [!NOTE]
>
> `operational_intensity` and `bottleneck_operational_intensity` are different
> numbers (288.0455 vs 387.3531 here). The first is the achieved FLOP/byte; the
> second is the intensity at the limiting memory band, and it is the one that
> reconciles `optimal_flop_rate`. Compare `operational_intensity` against the
> ridge point.

--------------------------------------------------------------------------------

## 5. `bound_by` values

| value | meaning |
|---|---|
| `HBM` | Limited by HBM bandwidth. |
| `Compute` | Limited by the compute roofline. |
| `CMEM` / `VMEM` | Limited by an on-chip memory band. Compare against `cmem_*_ridge_point` / `vmem_*_ridge_point`, not `hbm_ridge_point`. |
| `Unknown` | FLOPs/bytes could not be inferred. See below. |
| `CustomCall (opaque)` | An opaque custom call was detected among the returned ops. |

### `Unknown` and Pallas kernels — do not report the zeros

For custom calls and Pallas kernels, the profiler cannot see inside the op.
`bound_by` comes back `Unknown` (or `CustomCall (opaque)`), FLOP counts and
operational intensity are `0`, and the top op may show up as `IDLE`. **Those
zeros are the profiler failing to see the kernel, not a finding about the
kernel.** Reporting "0 FLOPs, memory bound" for a Pallas workload is wrong.

Switch to the timeline instead:

```bash
xprof aggregate_xplane_events <logdir>              # where time actually went
xprof get_llo_analysis <logdir>                     # if built with embedded features
```

When an opaque custom call is present, the response gains a top-level
`guidance` field saying exactly this. Note the detection is computed over the
**returned** `top_operations` only — a custom call ranked below `--top_n` will
not trigger it. If you suspect Pallas, raise `--top_n` before concluding the
field's absence means anything.

--------------------------------------------------------------------------------

## 6. Reading `top_operations`

-   Rows are deduplicated by `(rank, name)` and re-sorted by
    `total_self_time_ms` descending. `rank` comes from the underlying table —
    do not assume it equals position + 1.
-   `source_info` is a stack chain joined by ` -> `, not a single line, and can
    end with a trailing ` -> `. Take the first frame for attribution.
-   `total_self_time_percent` is a share of total device time; the returned rows
    will not sum to 100% unless `--top_n` covers all
    `total_operations_analyzed` ops.
-   Per-op fields obey the same identities as the program block, against that
    op's own `optimal_flop_rate`.

--------------------------------------------------------------------------------

## 7. Worked example

> "Is my training workload compute-bound or memory-bandwidth bound, and what are
> the top bottleneck ops?"

```bash
xprof get_roofline_model demo/plugins/profile/v6e-4-training --top_n=10
```

Then report:

1.  **The bound, with its justification.** "HBM bound. Operational intensity is
    288.05 FLOP/byte against a ridge point of 577.96, so the workload sits below
    the ridge — bandwidth is the limit, not the MXU."
2.  **The headroom.** Roofline efficiency 49.33% of the achievable rate at this
    intensity; compute efficiency 33.06% of peak. The gap between those two is
    the bandwidth ceiling, not idle compute you can reclaim.
3.  **The device constants**, so the reader can check the arithmetic:
    `TPU v6 Lite`, peak 946,700 GFLOP/s, peak HBM 1525.5 GiB/s (= 1637.99 GB/s),
    ridge point 577.963.
4.  **The top ops by self time**, each with its own `bound_by` and the first
    frame of `source_info` — and say how many of the 893 analyzed ops you are
    showing.

--------------------------------------------------------------------------------

## 8. Related tools

| question | tool |
|---|---|
| Starved, or inefficient? | `get_kpi_metrics` (duty cycle) |
| Device hardware constants alone | `get_device_information` (same `device_info` block) |
| Which op dominates? | `get_top_hlo_ops`, `get_hlo_op_profile` |
| Where did time actually go? | `aggregate_xplane_events`, `list_xplane_events --max_events=200000` |
| Inside a Pallas kernel | `get_llo_analysis`, `get_llo_debug_string` |
| Memory headroom | `get_memory_profile`, `get_peak_allocations` |

**Rule of thumb:** for XLA-level work start here; for Pallas start at the
timeline (`aggregate_xplane_events`) and never read op-profile zeros as a
finding.
