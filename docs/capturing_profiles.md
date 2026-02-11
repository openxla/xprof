# Capturing profiles

In order to use Xprof, you need to first enable profile capture within your
model workload code. There are two ways to capture profiles, detailed below.

## Programmatic capture

With programmatic capture, you need to annotate your model code in order to
specify where in your code you want to capture profiles. Typically users collect
profiles for a few steps during their training loop, or profile a specific block
within their model. There are different ways to capture traces in the different
frameworks JAX, Pytorch XLA and Tensorflow - either api-based start/stop trace
or context manager based.

## On-demand capture (a.k.a manual capture)

On-demand profile capture is used when you want to capture profiles ad-hoc
during your run, for time periods when you did not enable programmatic profile
capture. This is typically used when you see some problem with your model
metrics during the run, and want to capture profiles at that instant for some
period in order to diagnose the problem.

To enable on-demand profile capture, you still need to start the `xprof` server
within your code. In JAX, for example, enabling
[`jax.profiler.start_server`](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.start_server.html)
will start an `xprof` server on your ML workload which is listening for the
on-demand capture trigger to start capturing profiles.

## Multiple sessions per run

When you capture profiles, you can capture profiles for a single run as multiple
sessions. Let us say you capture profiles in a training run from step 1-3 and
later capture profiles from step 8-10. So these are profiles for the same run,
but the first capture from step 1-3 will be session1 and second capture from
step 8-10 will be session2. The different sessions will be denoted with
different date stamps under each run. You can capture profiles in different
sessions either programmatically or on-demand or a mix of both.

## Continuous profiling snapshots

Continuous profiling snapshots are used to capture a profile ending at any
specific time instant, unlike on-demand profiling where you capture profiles
for a duration later in time. This is useful for long-running jobs where you
want to capture a profile at the instant when the problem is diagnosed.

To enable continuous profiling snapshots, you need to start the `xprof` server
within your code. In JAX, for example, enabling
[`jax.profiler.start_server`](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.start_server.html)
will start an `xprof` server on your ML workload which is listening for the
snapshot profiling trigger to start capturing profiles.

## XProf and Tensorboard on Google Cloud

On Google Cloud, we recommend using
[`cloud-diagnostics-xprof`](https://github.com/AI-Hypercomputer/cloud-diagnostics-xprof)
library to make it easier to host Tensorboard and XProf. Some of the main
benefits of using this library on GCP:

- Easy setup and packaging of XProf and tensorboard dependencies;
- Store your profiles in GCS which can be useful for long term retention and
  post-run analysis (local profiles captured will be deleted after researcher
  finishes run);
- Fast loading of large profiles and multiple profiles by provisioning
  Tensorboard on GCE VM or GKE pod, with option to change machine type based on
  user needs for loading speed and cost;
- Create a link for easy sharing of profiles and collaboration with team members
  and Google engineers;
- Easier on-demand profiling of workloads on GKE and GCE to choose any host
  running your workload to capture profiles.

## Framework-specific instructions

Check out how to enable programmatic profiling and on-demand profiling in
different frameworks:

- [JAX](jax_profiling.md)
- [PyTorch/XLA](pytorch_xla_profiling.md)
- [Tensorflow](tensorflow_profiling.md)

## Troubleshooting

### GPU profiling

Programs running on GPU should produce traces for the GPU streams near the top
of the trace viewer. If you're only seeing the host traces, check your program
logs and/or output for the following error messages.

**If you get an error like: `Could not load dynamic library 'libcupti.so.10.1'`**<br />
Full error:
```
W external/org_tensorflow/tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcupti.so.10.1'; dlerror: libcupti.so.10.1: cannot open shared object file: No such file or directory
2020-06-12 13:19:59.822799: E external/org_tensorflow/tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1422] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI could not be loaded or symbol could not be found.
```

Add the path to `libcupti.so` to the environment variable `LD_LIBRARY_PATH`.
(Try `locate libcupti.so` to find the path.) For example:
```shell
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
```

If you still get the `Could not load dynamic library` message after doing this,
check if the GPU trace shows up in the trace viewer anyway. This message
sometimes occurs even when everything is working, since it looks for the
`libcupti` library in multiple places.

**If you get an error like: `failed with error CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`**<br />
Full error:
```shell
E external/org_tensorflow/tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1445] function cupti_interface_->EnableCallback( 0 , subscriber_, CUPTI_CB_DOMAIN_DRIVER_API, cbid)failed with error CUPTI_ERROR_INSUFFICIENT_PRIVILEGES
2020-06-12 14:31:54.097791: E external/org_tensorflow/tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1487] function cupti_interface_->ActivityDisable(activity)failed with error CUPTI_ERROR_NOT_INITIALIZED
```

Run the following commands (note this requires a reboot):
```shell
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee -a /etc/modprobe.d/nvidia-kernel-common.conf
sudo update-initramfs -u
sudo reboot now
```

See [NVIDIA's documentation on this
error](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti)
for more information.

### Profiling on a remote machine

If the program you'd like to profile is running on a remote machine, one
option is to run all the instructions above on the remote machine (in
particular, start the XProf server on the remote machine), then use SSH
local port forwarding to access the XProf web UI from your local
machine. Use the following SSH command to forward the default XProf port
8791 from the local to the remote machine:

```shell
ssh -L 8791:localhost:8791 <remote server address>
```

or if you're using Google Cloud:
```bash
$ gcloud compute ssh <machine-name> -- -L 8791:localhost:8791
```

### Multiple TensorBoard installs

**If starting TensorBoard fails with an error like: `ValueError: Duplicate
plugins for name projector`**

It's often because there are two versions of TensorBoard and/or TensorFlow
installed (e.g. the `tensorflow`, `tf-nightly`, `tensorboard`, and `tb-nightly`
pip packages all include TensorBoard). Uninstalling a single pip package can
result in the `tensorboard` executable being removed which is then hard to
replace, so it may be necessary to uninstall everything and reinstall a single
version:

```shell
pip uninstall tensorflow tf-nightly tensorboard tb-nightly xprof xprof-nightly tensorboard-plugin-profile tbp-nightly
pip install tensorboard xprof
```

### Resolve privilege issues

When you run profiling with CUDA® Toolkit in a Docker environment or on Linux,
you may encounter issues related to insufficient CUPTI privileges
(`CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`). Go to the
[NVIDIA Developer Docs](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters)
to learn more about how you can resolve these issues on Linux.

To resolve CUPTI privilege issues in a Docker environment, run

```shell
docker run option '--privileged=true'
```
