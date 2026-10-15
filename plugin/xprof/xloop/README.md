# XLoop Engine

The XLoop Engine is how you answer a question that is surprisingly easy to get
wrong on a TPU: *is this kernel actually faster, and does it still compute the
same thing?*

It is the execution half of an optimisation loop — for a person tuning a
Pallas kernel by hand, or for an agent generating candidates and keeping the
ones that win. You give it a benchmark item containing a reference
implementation and one or two replacements, plus the code that generates their
inputs. It leases a chip per worker, compiles and warms up each
implementation, feeds all of them the same inputs, checks their outputs
against the reference, reads the device execution times back out of the XProf
trace, and leaves the traces behind so you can ask *why*.

```shell
xloop run kernels.jsonl /tmp/run --num_warmups=10 --num_executions=20
```

```
item              status                       correct  custom vs ref  verdict  opt vs ref
----------------  ---------------------------  -------  -------------  -------  ----------
flash_attention   CODE_STATUS_EXECUTE_SUCCESS  pass     1.84x          faster   2.10x
layer_norm        CODE_STATUS_EXECUTE_SUCCESS  FAIL     3.02x          faster   -
```

## Why an engine instead of a script

Three things go wrong with the thirty-line benchmark script everyone writes
first, and each one produces numbers that look plausible and point in the
wrong direction:

1.  **Host timing measures the dispatch path, not the kernel.** Wrapping a
    jitted call in `time.perf_counter()` — even with `block_until_ready()` —
    times Python, JAX dispatch, the driver, and a host-device round trip. On a
    v6e that costs about **160 µs**, against **2.5 µs** of actual compute for
    a 1024³ matmul: the kernel is under 2% of the number, and a genuine 100×
    speedup reads as 1.7×. XLoop reads each sample's duration off the device
    plane of the trace instead.
2.  **A broken kernel is often a fast one.** A tile size that drops the tail
    of a reduction or an accumulator in the wrong dtype can easily run 3×
    faster. XLoop checks every sample's output against the reference on the
    exact same generated inputs before reporting a speedup.
3.  **A TPU VM usually has several chips and more than one person on it.** Two
    processes that both call `jax.devices()` collide on chip 0, while seven
    chips sit idle. XLoop brokers chips across the host: a worker leases what
    it needs, sees only those chips, queues when none are free, and releases
    its lease automatically even if `SIGKILL`ed.

## What a run produces

`xloop run` writes a directory per `(item, input_config)` containing the
serialized `XLoopItem` with `execution_results` and `execution_report` filled
in, plus one XProf trace directory per implementation:

```
/tmp/run/
  items.json
  flash_attention/
    seq2048/
      reference_profile/   *.xplane.pb
      custom_profile/      *.xplane.pb
      results.json
```

The traces are ordinary XProf log directories, and `trace_dirs` in the output
maps each `<config>/<role>` to its path. Point `xprof` at them to move from
*whether* a candidate is faster to *why*:

Question                                         | Tool
:----------------------------------------------- | :---
Which kernel dominates device time?              | `get_kernel_stats`
Which HLO ops cost the most?                     | `get_hlo_stats`, `get_top_hlo_ops`
Is it compute-bound or bandwidth-bound?          | `get_roofline_model`
Is the output numerically sound, not just close? | `verify_numerical_parity`

Pallas kernels appear in the device plane as `tpu_custom_call` with their HLO
op names intact.

> [!NOTE]
> `get_llo_analysis` and `get_llo_debug_string` will find nothing in traces
> collected on a Cloud TPU VM: they read an `llo_proto` XStat off the
> `/host:metadata` plane, and the GCE `libtpu` does not emit one.

## Writing an item

An item file is JSON, JSONL, textproto, or binary proto holding one or more
`XLoopItem` messages (defined in `protobuf/xloop_item.proto`). Each item
carries the source text of up to three implementations — `reference_code`,
`custom_code`, `optimized_code` — and one or more named input configurations:

```json
{
  "name": "softmax",
  "kernel_type": "KERNEL_TYPE_JAX",
  "reference_code": "import jax\n\ndef get_model_identifier():\n  return jax.nn.softmax\n",
  "custom_code": "...",
  "input_generation_configs": {
    "r4096_c512": {
      "input_generation_code": "import jax\nimport jax.numpy as jnp\n\ndef get_input_generator():\n  def generate(key):\n    return (jax.random.normal(key, (4096, 512), jnp.bfloat16),), {}\n  return generate\n",
      "atol": 1e-2,
      "rtol": 1e-2
    }
  }
}
```

Each entry in `input_generation_configs` is run and reported separately, so a
single item can sweep shapes or dtypes. Within a config, every implementation
is fed the exact same generated inputs for each sample.

### The module contract

**Implementation code** must define `get_model_identifier()`, returning the
callable to benchmark — a function, a `functools.partial`, a class (which is
instantiated), or a Flax module (whose parameters are initialized and bound).
Two optional attributes on the callable adjust how it is wrapped:

Attribute         | Effect
:---------------- | :-----
`static_argnames` | Forwarded to `jax.jit`, for shape or flag arguments.
`skip_jit`        | Do not wrap in `jax.jit` (for code that already is, or must not be).

If the callable accepts a `config_name` keyword, it receives the active config
name and that argument is marked static automatically, so one kernel can pick
block sizes per shape regime.

**Input-generation code** must define `get_input_generator()`, returning
`generate(key) -> (args, kwargs)`. If `generate` accepts `sample_index`, it
receives the iteration number. The module may also define `get_init_inputs()`
(constructor args when `get_model_identifier()` returns a class) and
`get_atol()` / `get_rtol()` (fallbacks when the config does not set them).

**Shared helpers** go in `auxiliary_modules`, keyed by dotted module name;
they are written into the worker's directory and imported normally.

## CLI reference

```shell
xloop run ITEM_PATH OUTPUT_DIR [flags]
xloop summarize ITEM_PATH [--detailed_summary]
xloop chips
```

Flag                   | Default | Effect
:--------------------- | :------ | :-----
`--num_warmups`        | 3       | Untimed iterations before tracing starts.
`--num_executions`     | 10      | Timed samples per implementation.
`--roles`              | all     | Subset of `reference,custom,optimized`.
`--seed`               | 0       | PRNG seed for input generation.
`--tpu_topology`       | 1 chip  | Accelerator shape per worker, e.g. `2x2`.
`--max_local_workers`  | chips   | Cap on concurrent workers.
`--first_config_only`  | false   | Run only the first input config per item.
`--detailed_summary`   | false   | Include per-sample times in the table.
`--in_process`         | false   | Run serially in this process (for a debugger).

## How timing works

Each sample's time is read from the `XLA Modules` line of the `/device:TPU:*`
plane in the `.xplane.pb` trace. The harness runs a small marker op between
samples so consecutive executions of the computation under test are cleanly
separated on the device timeline and can be matched by position.

If a trace cannot be parsed, the engine falls back to wall clock rather than
failing, and records which it used per role in `timing_source` (`"trace"` or
`"wall_clock"`). **Check `timing_source` in any automated loop**: a
`"wall_clock"` result includes the ~160 µs dispatch floor and is not a valid
kernel measurement.

Device time still includes real on-device fixed costs (such as an input copy
or a dtype conversion at the boundary), so `K` back-to-back operations cost
`a + b·K`, not `K·b`.

## How chip sharing works

Each `(item, config)` task runs in its own subprocess holding a lease from the
broker:

```mermaid
flowchart LR
  A["item file"] --> B["orchestrator<br/>(xloop_lib)"]
  B --> C["chip broker"]
  C -->|"lease + env"| D["worker subprocess"]
  D --> E["harness<br/>(eval_jax_lib)"]
  E --> F["trace + results"]
```

A subprocess per task is required because `libtpu` reads `TPU_VISIBLE_DEVICES`
and `LIBTPU_INIT_ARGS` once at initialization; it also gives each item a clean
compilation cache and contains a `libtpu` segfault to the worker that caused
it.

Leases live in `/tmp/xloop_chips` (host-global, mode `1777`, override with
`XLOOP_CHIP_LEASE_DIR`) as `flock` locks held on open file descriptors:

*   Because the directory is host-wide rather than per-user, two people on the
    same VM see the same lease state and do not get handed the same chip.
*   Because the lock is tied to the open descriptor, a worker killed with
    `SIGKILL` releases its chip through the kernel the instant it dies.
*   When a worker's environment is constructed, all three `libtpu`
    device-selection variables (`TPU_VISIBLE_DEVICE_PATHS`,
    `TPU_VISIBLE_CHIPS`, `TPU_VISIBLE_DEVICES`) are cleared before setting the
    one for the lease, so an exported variable in the caller's shell cannot
    silently outrank it.
*   A run that needs more chips than are currently free waits in FIFO order
    and logs its queue position.

`xloop chips` inspects PCI devices in `/sys/bus/pci/devices` and probes the
lock files without importing JAX — importing JAX would itself claim every
visible chip.

The broker is cooperative and exclusive: a lease owns a whole chip for the
duration of the worker, and a non-XLoop process that opens `/dev/vfio/*`
directly is not stopped from doing so (the kernel's single-open rule on a vfio
group is the only hard boundary).
