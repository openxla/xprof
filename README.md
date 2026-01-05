# XProf (+ Tensorboard Profiler Plugin)

XProf offers a number of tools to analyse and visualize the
performance of your model across multiple devices. Some of the tools include:

*   **Overview**: A high-level overview of the performance of your model. This
    is an aggregated overview for your host and all devices. It includes:
    *   Performance summary and breakdown of step times.
    *   A graph of individual step times.
    *   High level details of the run environment.
*   **Trace Viewer**: Displays a timeline of the execution of your model that shows:
    *   The duration of each op.
    *   Which part of the system (host or device) executed an op.
    *   The communication between devices.
*   **Memory Profile Viewer**: Monitors the memory usage of your model.
*   **Graph Viewer**: A visualization of the graph structure of HLOs of your model.

To learn more about the various XProf tools, check out the [XProf documentation](https://openxla.org/xprof)

## Demo
First time user? Come and check out this [Colab Demo](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html).

## Quick Start

### Prerequisites

* xprof >= 2.20.0
* (optional) TensorBoard >= 2.20.0

Note: XProf requires access to the Internet to load the [Google Chart library](https://developers.google.com/chart/interactive/docs/basic_load_libs#basic-library-loading).
Some charts and tables may be missing if you run XProf entirely offline on
your local machine, behind a corporate firewall, or in a datacenter.

If you use Google Cloud to run your workloads, we recommend the
[xprofiler tool](https://github.com/AI-Hypercomputer/cloud-diagnostics-xprof).
It provides a streamlined profile collection and viewing experience using VMs
running XProf.

### Installation

To get the most recent release version of XProf, install it via pip:

```
$ pip install xprof
```

## Running XProf

XProf can be launched as a standalone server or used as a plugin within
TensorBoard. For large-scale use, it can be deployed in a distributed mode with
separate aggregator and worker instances ([more details on it later in the
doc](#distributed-profiling)).

### Command-Line Arguments

When launching XProf from the command line, you can use the following arguments:

*   **`logdir`** (optional): The directory containing XProf profile data (files
    ending in `.xplane.pb`). This can be provided as a positional argument or
    with `-l` or `--logdir`. If provided, XProf will load and display profiles
    from this directory. If omitted, XProf will start without loading any
    profiles, and you can dynamically load profiles using `session_path` or
    `run_path` URL parameters, as described in the [Log Directory
    Structure](#log-directory-structure) section.
*   **`-p <port>`**, **`--port <port>`**: The port for the XProf web server.
    Defaults to `8791`.
*   **`-gp <grpc_port>`**, **`--grpc_port <grpc_port>`**: The port for the gRPC
    server used for distributed processing. Defaults to `50051`. This must be
    different from `--port`.
*   **`-wsa <addresses>`**, **`--worker_service_address <addresses>`**: A
    comma-separated list of worker addresses (e.g., `host1:50051,host2:50051`)
    for distributed processing. Defaults to to `0.0.0.0:<grpc_port>`.
*   **`-hcpb`**, **`--hide_capture_profile_button`**: If set, hides the 'Capture
    Profile' button in the UI.

### Standalone

If you have profile data in a directory (e.g., `profiler/demo`), you can view it
by running:

```
$ xprof profiler/demo --port=6006
```

Or with the optional flag:

```
$ xprof --logdir=profiler/demo --port=6006
```

### With TensorBoard

If you have TensorBoard installed, you can run:

```
$ tensorboard --logdir=profiler/demo
```

If you are behind a corporate firewall, you may need to include the `--bind_all`
tensorboard flag.

Go to `localhost:6006/#profile` of your browser, you should now see the demo
overview page show up.
Congratulations! You're now ready to capture a profile.

### Log Directory Structure

When using XProf, profile data must be placed in a specific directory structure.
XProf expects `.xplane.pb` files to be in the following path:

```
<log_dir>/plugins/profile/<session_name>/
```

*   `<log_dir>`: This is the root directory that you supply to `tensorboard
    --logdir`.
*   `plugins/profile/`: This is a required subdirectory.
*   `<session_name>/`: Each subdirectory inside `plugins/profile/` represents a
    single profiling session. The name of this directory will appear in the
    TensorBoard UI dropdown to select the session.

**Example:**

If your log directory is structured like this:

```
/path/to/your/log_dir/
└── plugins/
    └── profile/
        ├── my_experiment_run_1/
        │   └── host0.xplane.pb
        └── benchmark_20251107/
            └── host1.xplane.pb
```

You would launch TensorBoard with:

```bash
tensorboard --logdir /path/to/your/log_dir/
```

The runs `my_experiment_run_1` and `benchmark_20251107` will be available in the
"Sessions" tab of the UI.

You can also dynamically load sessions from a GCS bucket or local filesystem by
passing URL parameters when loading XProf in your browser. This method works
whether or not you provided a `logdir` at startup and is useful for viewing
profiles from various locations without restarting XProf.

For example, if you start XProf with no log directory:

```bash
xprof
```

You can load sessions using the following URL parameters.

Assume you have profile data stored on GCS or locally, structured like this:

```
gs://your-bucket/profile_runs/
├── my_experiment_run_1/
│   ├── host0.xplane.pb
│   └── host1.xplane.pb
└── benchmark_20251107/
    └── host0.xplane.pb
```

There are two URL parameters you can use:

*   **`session_path`**: Use this to load a *single* session directly. The path
    should point to a directory containing `.xplane.pb` files for one session.

    *   GCS Example:
        `http://localhost:8791/?session_path=gs://your-bucket/profile_runs/my_experiment_run_1`
    *   Local Path Example:
                `http://localhost:8791/?session_path=/path/to/profile_runs/my_experiment_run_1`
    *   Result: XProf will load the `my_experiment_run_1`
                        session, and you will see its data in the UI.

*   **`run_path`**: Use this to point to a directory that contains *multiple*
    session directories.

    *   GCS Example:
        `http://localhost:8791/?run_path=gs://your-bucket/profile_runs/`
    *   Local Path Example:
        `http://localhost:8791/?run_path=/path/to/profile_runs/`
    *   Result: XProf will list all session directories found under `run_path`
        (i.e., `my_experiment_run_1` and `benchmark_20251107`) in the "Sessions"
    dropdown in the UI, allowing you to switch between them.

**Loading Precedence**

If multiple sources are provided, XProf uses the following order of precedence
to determine which profiles to load:

1.  **`session_path`** URL parameter
2.  **`run_path`** URL parameter
3.  **`logdir`** command-line argument

### Distributed Profiling

XProf supports distributed profile processing by using an aggregator that
distributes work to multiple XProf workers. This is useful for processing large
profiles or handling multiple users.

**Note**: Currently, distributed processing only benefits the following tools:
`overview_page`, `framework_op_stats`, `input_pipeline`, and `pod_viewer`.

**Note**: The ports used in these examples (`6006` for the aggregator HTTP
server, `9999` for the worker HTTP server, and `50051` for the worker gRPC
server) are suggestions and can be customized.

**Worker Node**

Each worker node should run XProf with a gRPC port exposed so it can receive
processing requests. You should also hide the capture button as workers are not
meant to be interacted with directly.

```
$ xprof --grpc_port=50051 --port=9999 --hide_capture_profile_button
```

**Aggregator Node**

The aggregator node runs XProf with the `--worker_service_address` flag pointing
to all available workers. Users will interact with aggregator node's UI.

```
$ xprof --worker_service_address=<worker1_ip>:50051,<worker2_ip>:50051 --port=6006 --logdir=profiler/demo
```

Replace `<worker1_ip>, <worker2_ip>` with the addresses of your worker machines.
Requests sent to the aggregator on port 6006 will be distributed among the
workers for processing.

For deploying a distributed XProf setup in a Kubernetes environment, see
[Kubernetes Deployment Guide](docs/kubernetes_deployment.md).

## Nightlies

Every night, a nightly version of the package is released under the name of
`xprof-nightly`. This package contains the latest changes made by the XProf
developers.

To install the nightly version of profiler:

```
$ pip uninstall xprof tensorboard-plugin-profile
$ pip install xprof-nightly
```

## Next Steps

* [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html#xprof-tensorboard-profiling)
* [PyTorch/XLA Profiling Guide](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
* [TensorFlow Profiling Guide](https://tensorflow.org/guide/profiler)
* [Cloud TPU Profiling Guide](https://cloud.google.com/tpu/docs/cloud-tpu-tools)
* [Colab Tutorial](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)
* [Tensorflow Colab](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)

