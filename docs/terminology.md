# XProf terminology

This page describes some terms used in the context of XProf.

* Profile:
    * Refers to the data collected/captured about your program's execution performance. This includes memory used by various operations, duration of operations, size of data transmissions, and much more.
* Session:
    * Refers to a specific instance of data capture. It has a unique name, and each subdirectory inside `plugins/profile/` represents a single profiling session.
* Run:
    * Refers to a single training job or workflow, and can be considered synonymous with an "experiment".
* Host:
    * The CPU of the system where your program is executed, it controls program flow and data transfer. Host "memory" refers to the system memory (RAM).
* Device:
    * The accelerator, GPU or TPU of the system where your program is executed, it executes the actual computations. Device "memory" refers to the high bandwidth memory (HBM) connected to the accelerator.
* Step:
    * Refers to one iteration of a model training loop. Step time refers to time it takes for the iteration, and is the unit of measurement used by XProf to report performance.

For details about some terms related to XLA, refer to [XLA Terminology](https://openxla.org/xla/terminology).
