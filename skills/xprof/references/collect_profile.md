
# Profile Collection Guide (`collect_profile`)

This guide explains how to capture XProf performance traces from accelerated ML
workloads across JAX, PyTorch (including Torch-XLA on Google Cloud TPU), and
TensorFlow, and prepare them for analysis with `xprof`.

--------------------------------------------------------------------------------

## 1. Overview & Storage Layout

XProf tools analyze trace data stored in standard TensorBoard profile directory
structures:

```
<logdir>/plugins/profile/<run_name>/
    ├── <worker0>.xplane.pb
    ├── <worker1>.xplane.pb
    └── ...
```

Traces can be collected using two primary paradigms:

1.  **Programmatic Tracing**: Embedded within the training script using
    framework-native profiler APIs.
2.  **On-Demand Remote Capture**: Running a lightweight profiler server on
    workers and capturing traces remotely via the XProf Web Server or client
    APIs without restarting workloads.

--------------------------------------------------------------------------------

## 2. JAX Workload Profiling

### Programmatic Tracing

Use `jax.profiler` directly in your Python code:

```python
import jax

# Option A: Context manager (recommended for specific loops or steps)
with jax.profiler.trace("/tmp/xprof_logs"):
  # Run 2-5 warm iterations
  for _ in range(5):
    step()

# Option B: Explicit start and stop
jax.profiler.start_trace("/tmp/xprof_logs")
step()
jax.profiler.stop_trace()
```

### Remote Profiling Server (On-Demand Capture)

Running a profiler server allows you to capture performance traces on demand
from long-running training jobs without modifying loop code or restarting
workloads.

#### Step 1: Start the Profiler Server in your JAX workload

Start a background gRPC profiler server before entering your training loop:

```python
import jax

# Start profiler gRPC server on port 50051
server = jax.profiler.start_server(port=50051)

for step in range(num_steps):
  train_step()
```

> [!NOTE]
> The server runs as a background daemon and does not write trace files until a
> client requests a capture. In multi-host environments (e.g., Cloud TPU Pods),
> run `start_server` on all hosts, or target worker 0 (`localhost:50051`).

#### Step 2: Trigger the Capture from an External Client or Web Server

While the training loop is executing, trigger a capture using either:

**Option A: Via XProf Web Server (HTTP / UI)**:
Start `xprof server` and trigger capture using the web UI or HTTP `curl`
request. See Section 5 (XProf Server & Remote Web Capture) below for server
setup, web UI parameters, and endpoint examples.

**Option B: Via Python Client**:
```python
import tensorflow as tf

tf.profiler.experimental.client.trace(
    service_addr="localhost:50051",
    logdir="/tmp/xprof_logs",
    duration_ms=5000,
)
```

The captured trace file (`.xplane.pb`) will be saved automatically to:
`/tmp/xprof_logs/plugins/profile/<timestamp>/<hostname>.xplane.pb`

### Low-Level Optimizer (LLO) & Custom Call Tracing (Cloud TPU)

To capture fine-grained VLIW instruction metrics and Low Level Optimizer (LLO)
disassembly for custom Pallas or Mosaic kernels, export `LIBTPU_INIT_ARGS`
**before importing JAX**:

```bash
export LIBTPU_INIT_ARGS="\
--xla_xprof_enable_custom_call_tracing=true \
--xla_xprof_register_llo_debug_info=true"
python your_jax_workload.py
```

--------------------------------------------------------------------------------

## 3. PyTorch & Torch-XLA Profiling

> [!NOTE]
> OpenXLA XProf ingests XSpace binary traces (`.xplane.pb` / `.xspace.pb`). For
> PyTorch workloads, profiling is supported via **Torch-XLA**
> (`torch_xla.debug.profiler`), which emits `.xplane.pb` traces. Standard
> non-XLA PyTorch profiler generates Chrome trace JSON, which is intended
> for the PyTorch TensorBoard Profiler plugin and cannot be ingested by XProf.

### Programmatic Tracing

Capture execution traces programmatically around your training loop:

```python
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp

# Directory where trace files (.xplane.pb) will be stored
log_dir = "/tmp/xprof_logs"

# Start tracing
xp.start_trace(log_dir)

for step, batch in enumerate(dataloader):
  with xp.Trace("train_step"):
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    xm.optimizer_step(optimizer)

# Stop tracing and flush trace to disk
xp.stop_trace()
```

### Remote Profiler Server (Cloud TPU / GPU)

Start a background profiler server to allow on-demand capture via `xprof server`
or remote client:

```python
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp

# Start profiler server on master worker (default port 50051)
server = xp.start_server(port=50051)

for step, batch in enumerate(dataloader):
  optimizer.zero_grad()
  loss = model(batch)
  loss.backward()
  xm.optimizer_step(optimizer)
```

--------------------------------------------------------------------------------

## 4. TensorFlow Profiling

### Programmatic Tracing

```python
import tensorflow as tf

# Start trace
tf.profiler.experimental.start("/tmp/xprof_logs")

# Run 2-5 representative steps
for step in range(5):
  train_step()

# Stop trace and flush to disk
tf.profiler.experimental.stop()
```

### Remote Tracing via Client

Trigger a capture against a remote TensorFlow/JAX/Torch-XLA worker from Python:

```python
import tensorflow as tf

tf.profiler.experimental.client.trace(
    service_addr="localhost:50051",
    logdir="/tmp/xprof_logs",
    duration_ms=10000,
)
```

--------------------------------------------------------------------------------

## 5. XProf Server & Remote Web Capture

The `xprof` web server can capture profiles from any running profiler service
(JAX, PyTorch Torch-XLA, or TensorFlow):

### Step 1: Start XProf Server

```bash
xprof server --logdir=/tmp/xprof_logs --port=8791
```

### Step 2: Trigger Capture

-   **Via Web UI**: Open `http://localhost:8791`, click **Capture Profile**,
    enter `localhost:50051` in the Profile Service Address field, set Duration
    (e.g., `5000` ms), and click **Capture**.
-   **Via HTTP Endpoint**:

```bash
curl -X GET "http://localhost:8791/capture_profile?\
service_addr=localhost:50051&duration=5000"
```

The captured trace will be saved automatically to:
`/tmp/xprof_logs/plugins/profile/<run_name>/`.

--------------------------------------------------------------------------------

## 6. Analyzing Standalone Trace Files & Logdir Ingestion

If you already have a standalone `.xplane.pb` or `.xspace.pb` file, you can
either analyze it directly or import it into a structured logdir.

### Direct File Analysis

You can pass the trace file path directly to any `xprof` CLI tool without
manual directory organization:

```bash
# Analyze a single trace file directly
xprof get_overview /path/to/trace.xplane.pb
xprof get_kpi_metrics /path/to/trace.xplane.pb
xprof get_roofline_model /path/to/trace.xplane.pb
```

### Ingestion into Logdir via `upload_trace`

To organize traces for multi-run comparisons or viewing in the XProf Web
Server UI, use `upload_trace` to import the trace into the standard
`<logdir>/plugins/profile/<run_name>/` hierarchy:

```bash
# Import a standalone trace file into a structured logdir
xprof upload_trace \
  --file_path=/path/to/trace.xplane.pb \
  --logdir=/tmp/xprof_logs \
  --run_name=run_01

# Start the Web Server to visualize all imported runs
xprof server --logdir=/tmp/xprof_logs --port=8791
```

Alternatively, you can manually copy files:

```bash
mkdir -p /tmp/xprof_logs/plugins/profile/run_01
cp /path/to/trace.xplane.pb /tmp/xprof_logs/plugins/profile/run_01/
xprof server --logdir=/tmp/xprof_logs --port=8791
```

--------------------------------------------------------------------------------

## 7. Best Practices & Mitigating Trace Drops

-   **Capture Steady State**: Avoid capturing during the first step (initial
    compilation and weight allocation can distort step metrics). Wait for step
    10 or higher.
-   **Trace Duration**: Profile 2–5 steps (typically 5–10 seconds). Capturing
    too many steps can exceed the on-device trace buffer size and trigger
    dropped events.
-   **Trace Buffer Dropped Events**: If the UI shows "Trace Buffer Dropped" or
    events appear truncated:
    1.  Reduce the capture duration (e.g. `duration=5000` in the HTTP capture
        endpoint, or `duration_ms=5000` in `client.trace`).
    2.  Profile fewer chips per task if multi-chip tracing.
    3.  Lower host or python tracer levels if memory-constrained.

--------------------------------------------------------------------------------

## 8. Downstream Analysis Workflow

Once a profile is captured and available in `<logdir>`, analyze it using the
core deterministic tools:

```bash
# High-level breakdown (compute vs host vs communication)
xprof get_overview /tmp/xprof_logs

# Key performance indicators (step time, duty cycle, roofline)
xprof get_kpi_metrics /tmp/xprof_logs

# Program & op-level roofline analysis (compute vs memory bounds)
xprof get_roofline_model /tmp/xprof_logs

# Top operations by self-time
xprof get_top_hlo_ops /tmp/xprof_logs --limit=10

# Fine-grained timeline events
xprof list_xplane_events /tmp/xprof_logs --max_events=200000
```

See [SKILL.md](../SKILL.md) for the complete list of supported tools and
workflows.
