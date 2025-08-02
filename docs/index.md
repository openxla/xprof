## XProf (+ Tensorboard Profiler Plugin)

XProf is a command-line application and
[Tensorboard](https://www.tensorflow.org/tensorboard) plugin that includes a
suite of tools for [JAX](https://jax.readthedocs.io/),
[TensorFlow](https://www.tensorflow.org/), and
[PyTorch/XLA](https://github.com/pytorch/xla). These tools help you understand,
debug and optimize programs to run on CPUs, GPUs and TPUs.

![Xprof overview page](images/xprof_overview.png)

XProf offers a number of tools to analyse and visualize the
performance of your model across multiple devices. Some of the tools include:

*   **Overview**: A high-level overview of the performance of your model. This
    is an aggregated overview for your host and all devices. It includes:
    *   Performance summary and breakdown of step times.
    *   A graph of individual step times.
    *   A table of the top 10 most expensive operations.
*   **Trace Viewer**: Displays a timeline of the execution of your model that
    shows:
    *   The duration of each op.
    *   Which part of the system (host or device) executed an op.
    *   The communication between devices.
*   **Memory Profile Viewer**: Monitors the memory usage of your model.
*   **Graph Viewer**: A visualization of the graph structure of HLOs of your
    model.

Each tool is described in detail in its own page.

### Get Started

XProf is a TensorBoard plugin used in notebook environments such as Colab. To
get started, install the plugin to your notebook environment with

```shell
!pip install -U tensorboard_plugin_profile
```

and clone the XProf repository:

```shell
!git clone http://github.com/openxla/xprof
```

Load the TensorBoard notebook extension:

```shell
%load_ext tensorboard
```

Finally, launch TensorBoard specifying the `logdir` where your profiling data is
stored.

```shell
%tensorboard --logdir=xprof/demo
```

> **Note:** XProf requires access to the Internet to load the
> [Google Chart library](https://developers.google.com/chart/interactive/docs/basic_load_libs#basic-library-loading).
> Some charts and tables may be missing if you run TensorBoard entirely offline on
> your local machine, behind a corporate firewall, or in a datacenter.

### Demos

First time user?

- Explore the XProf interface with this [notebook demo](xprof_demo.ipynb).
- Check out this
  [Colab Demo](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html)
  to see how XProf can be used with JAX.
