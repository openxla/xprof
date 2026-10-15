# XProf Terminology

This page describes terms used in the context of XProf and ML performance
profiling.

## General Terms

* **Profile**:
    * The data collected about your program's execution performance. This
    includes memory used by various operations, duration of operations, size of
    data transmissions, and much more.
* **Session**:
    * A specific instance of data capture. It has a unique name, and each
    subdirectory inside `plugins/profile/` represents a single profiling
    session.
* **Run**:
    * A single training job or workflow, synonymous with an "experiment".
* **Step**:
    * One iteration of a model training loop. Step time is the time it takes
    for one iteration, and is the primary unit of measurement used by XProf.

## Hardware Terms

* **Host**:
    * The CPU of the system where your program is executed. It controls program
    flow and data transfer. Host "memory" refers to the system memory (RAM).
* **Device**:
    * The accelerator (GPU or TPU) where computations are executed. Device
    "memory" refers to the high bandwidth memory (HBM) connected to the
    accelerator.
* **HBM (High Bandwidth Memory)**:
    * The memory directly attached to the accelerator device. HBM has much
    higher bandwidth than system RAM but is more limited in capacity.
* **VMEM (Vector Memory)**:
    * On TPUs, the on-chip scratchpad memory used by the vector and matrix
    units. Much smaller but faster than HBM.
* **TensorCore**:
    * Specialized hardware units on NVIDIA GPUs (Volta and later) designed to
    accelerate matrix multiply-accumulate operations. TensorCore utilization
    measures how effectively your model uses these units.
* **MXU (Matrix Multiply Unit)**:
    * The equivalent of TensorCores on Google TPUs. MXUs perform large
    matrix multiplications efficiently.
* **SXM / PCIe**:
    * GPU interconnect form factors. SXM provides higher power and bandwidth
    compared to PCIe.

## Performance Metrics

* **FLOPS Utilization**:
    * The ratio of achieved floating-point operations per second to the
    theoretical peak FLOPS of the device. Higher utilization indicates better
    hardware usage.
* **Duty Cycle**:
    * The percentage of time the device is actively performing computation
    versus being idle. A low duty cycle indicates the device is waiting
    (e.g., for data from the host).
* **Memory Bandwidth Utilization**:
    * The ratio of achieved memory bandwidth to the peak memory bandwidth of
    the device. Memory-bound operations benefit from higher utilization.
* **Goodput Efficiency**:
    * How your model performs relative to ideal performance on the hardware.
    Accounts for overhead such as compilation, data loading, and
    communication.

## XLA / Compilation Terms

* **HLO (High Level Operations)**:
    * The intermediate representation used by the XLA compiler. HLO graphs
    show how your model's operations are compiled and optimized.
* **Op (Operation)**:
    * A single computation unit in the model graph. Examples include matrix
    multiplies, convolutions, and element-wise operations.
* **Fusion**:
    * A compiler optimization that combines multiple operations into a single
    kernel launch, reducing memory traffic and kernel launch overhead.
* **Kernel**:
    * A function that runs on the device (GPU or TPU). Each kernel invocation
    appears as a span in the Trace Viewer.
* **Custom Call**:
    * An operation in XLA that invokes user-defined or library-provided code
    (e.g., cuDNN, cuBLAS). Custom calls appear in profiling data with their
    own performance metrics.

## Communication Terms

* **DCN (Data Center Network)**:
    * The network connecting multiple hosts in a distributed training setup.
    DCN profiling measures inter-host communication overhead.
* **ICI (Inter-Chip Interconnect)**:
    * The high-speed link connecting TPU chips within a single TPU pod slice.
    ICI is faster than DCN.
* **All-Reduce**:
    * A collective communication operation that aggregates data across all
    devices (e.g., summing gradients during distributed training).
* **All-Gather**:
    * A collective communication operation that gathers data from all devices,
    so each device ends up with the full dataset.
* **Collective Permute**:
    * A communication pattern where each device sends data to a specific
    target device, used in pipeline parallelism.

## Profiling Tools

* **Trace Viewer**:
    * A timeline visualization showing when operations executed on the host
    and device. Useful for identifying idle time and communication bottlenecks.
* **Op Profile**:
    * Shows the time spent in each operation category. Helps identify the
    most expensive operations in your model.
* **Memory Viewer**:
    * Visualizes HBM memory allocation over time. Helps identify peak memory
    usage and potential memory leaks.
* **Roofline Model**:
    * A visual performance model that shows whether operations are
    compute-bound or memory-bound relative to the device's capabilities.
* **Graph Viewer**:
    * Displays the HLO computation graph, showing how operations are
    connected and fused by the compiler.
* **Kernel Profiling**:
    * Detailed profiling of individual GPU/TPU kernels, including hardware
    performance counters.

For details about terms related to XLA, refer to
[XLA Terminology](https://openxla.org/xla/terminology).
