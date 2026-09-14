# XProf Architecture Mapping Reference

Map high-level model architecture blocks (e.g., "attention", "MoE router",
"token embedding") to the HLO operations and timeline events they produce in an
XProf trace. This lets you attribute wall-clock time to specific parts of a
model and locate the exact kernels that implement a given computation.

All commands operate directly on a logdir or trace file: `xprof <command>
/path/to/logdir --flag=value`. No external database or config resolution service
is required.

## Prerequisites

-   An XProf **logdir or trace file** (`.xplane.pb` / `.xspace.pb`) that
    contains device (TPU/GPU) events. Prefer a single-step or steady-state
    capture: a pipelined prefill splits work across multiple XLA modules and
    makes block mapping unreliable, so a generate/decode or training-step trace
    is easier to map.
-   The name of the architecture block to map (e.g., "grouped query attention").
-   (Recommended) A reference to the model source (the framework code that emits
    the `jax.named_scope` / op-name hierarchy), so you can connect scope names
    to the block.

## Procedure

Copy this checklist and track progress:

-   [ ] Step 0: Confirm the trace has device events
-   [ ] Step 1: Discover the compiled HLO modules
-   [ ] Step 2: Map the block source to op-name scopes
-   [ ] Step 3: Locate the scope in the timeline
-   [ ] Step 4: Connect ops back to source with the graph viewer
-   [ ] Step 5: Filter to the exact block and summarize

### Step 0: Confirm the trace has device events

```bash
xprof get_device_information /path/to/logdir
xprof get_hosts /path/to/logdir
```

If `hasDeviceTrace` is `false`, the capture only contains host events and cannot
be mapped to device kernels; ask for a trace captured with the device profiler
enabled.

### Step 1: Discover the compiled HLO modules

```bash
xprof list_hlo_modules /path/to/logdir
```

Module names carry unique hash suffixes (e.g., `jit_generate(1234567890)`).
Record the exact module name for the step you want to analyze; you will pass it
to the HLO tools below.

### Step 2: Map the block source to op-name scopes

Frameworks annotate ops with a hierarchical scope name derived from
`jax.named_scope(...)` (JAX/Flax) or `tf_op_name` (TF). A block like grouped
query attention typically appears as a scope such as
`.../jit(generate)/.../grouped_query_attention/...`.

Identify the leaf ops the block emits (matmul/einsum, softmax, or a custom-call
for a Pallas kernel) and the scope substring that uniquely identifies it. Start
with a broad keyword (the block name) and narrow down.

### Step 3: Locate the scope in the timeline (XPlane)

Query device-plane events whose names match the block's scope. If you already
know the scope keyword, aggregate directly to avoid context bloat:

```bash
# Aggregate device events for the block's scope in one step
xprof aggregate_xplane_events /path/to/logdir \
  --plane_regex="Device.*" --event_regex=".*grouped_query_attention.*"

# Or list individual events (start/duration) when you need per-instance detail
xprof list_xplane_events /path/to/logdir \
  --plane_regex="Device.*" --event_regex=".*grouped_query_attention.*" \
  --max_events=200000
```

The aggregation gives total/self time and event counts for the block; the event
list gives per-instance start/duration for building a per-op breakdown.

### Step 4: Connect ops back to source with the graph viewer

To confirm which Python source produced an HLO op (and to disambiguate scopes
that share a keyword), use the graph viewer and HLO content tools:

```bash
# Map compiled instructions back to Python source line numbers
xprof get_graph_viewer /path/to/logdir --module_name=<module_name>

# Inspect the compiled HLO for a module, with source metadata
xprof get_hlo_module_content /path/to/logdir --module_name=<module_name> --print_metadata=True

# Inspect the neighborhood of a specific instruction (e.g. a fusion or custom-call)
xprof get_hlo_neighborhood /path/to/logdir --instruction_name=<instr_name> --radius=2
```

### Step 5: Filter to the exact block and summarize

Keep only the ops whose HLO type and shape match the target computation (discard
supporting ops such as scale computation or masking). Summarize the result as a
table:

Column       | Source
------------ | ----------------------------------------------------------
Scope suffix | last segment of the event/op name
HLO op       | the `%`-prefixed instruction name (e.g. `%fusion.63`)
Output shape | from the op's kernel/output shape (e.g. `f32[2,6,4,4096]`)
Duration     | event duration (`end - start`)
Role         | `target` or `auxiliary`

Report the total block time as the sum of `target` durations, and note the
dominant kernel(s).

## Concepts: op-name hierarchy

Device events and HLO ops carry a hierarchical name that mirrors the framework
call stack, e.g.
`jit(generate)/transformer/layer_5/grouped_query_attention/pallas_call`. The
final segment is the leaf op; the parent segments identify the block. Use the
parent segments to group ops into blocks and the leaf segment to identify the
kernel.

See [Analyze XLA Module Performance](analysis.md) for the full set of HLO and
timeline query commands.

## Pitfalls

-   **Dead-code branches (most common false negative).** Config flags can select
    entirely different implementations. If a scope is completely absent from the
    trace, the corresponding Python branch may never have executed (e.g. a fused
    kernel path was taken instead of the unfused one). Confirm which branch ran
    from the model config before concluding a block is "missing"; do not
    reverse-engineer config values from trace presence/absence.
-   **Fused kernels span multiple blocks.** A single `custom-call` (e.g. a fused
    Pallas kernel) may implement several architecture blocks (write + gating +
    read) as one event. The diagram shows 3 blocks; the trace has 1 event.
-   **Parent scope != target method.** A scope like `prepare_embedding` may
    contain output-embedding setup + input encode + masking. Always drill into
    sub-scopes.
-   **Scopes nest deeply.** A block like `grouped_query_attention` may be nested
    inside `.../shard_map/jit(grouped_query_attention)/pallas_call`. Start with
    a broad keyword and narrow the `--event_regex`.
-   **Duplicate kernels for the same op.** A single attention block can produce
    multiple kernels, e.g. one for the generation cache (small, fast) and one
    for the prefill cache (large KV, dominant). Check event counts and
    durations.
-   **XLA rewrites ops.** `one_hot + einsum` may compile to a single
    `gather`/fusion. Match against the actual HLO (Step 4), not the source op
    names.
-   **XLA interleaves freely.** Ops from unrelated scopes appear in the same
    time window as your target. Gaps between events are filled with interleaved
    work, not idle time.
-   **Global vs local layer scope mismatch.** If a `layer_N/local_layer` filter
    returns nothing, the layer may be global (`layer_N/global_layer`). Check the
    config.
-   **`hasDeviceTrace: false`** from `get_hosts` means device events are not in
    the XPlane; the trace was captured without the device profiler and cannot be
    mapped to kernels.
