# Optimize TensorFlow performance using XProf

This guide demonstrates how to use the tools available with XProf to track the
performance of your TensorFlow models on the host (CPU), the device (GPU), or on
a combination of both the host and device(s).

Profiling helps understand the hardware resource consumption (time and memory)
of the various TensorFlow operations (ops) in your model and resolve performance
bottlenecks and, ultimately, make the model execute faster.

This guide will walk you through how to use the various tools available and the
different modes of how the Profiler collects performance data.

If you want to profile your model performance on Cloud TPUs, refer to the
[Cloud TPU guide](https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_profile).

## Collect performance data

XProf collects host activities and GPU traces of your TensorFlow model. You can
configure XProf to collect performance data through either the programmatic mode
or the sampling mode.

### Profiling APIs

You can use the following APIs to perform profiling.

- Programmatic mode using the TensorBoard Keras Callback (`tf.keras.callbacks.TensorBoard`)

  ```python
  # Profile from batches 10 to 15
  tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                               profile_batch='10, 15')

  # Train the model and use the TensorBoard Keras callback to collect
  # performance profiling data
  model.fit(train_data,
            steps_per_epoch=20,
            epochs=5,
            callbacks=[tb_callback])
  ```

- Programmatic mode using the `tf.profiler` Function API

  ```python
  tf.profiler.experimental.start('logdir')
  # Train the model here
  tf.profiler.experimental.stop()
  ```

- Programmatic mode using the context manager

  ```python
  with tf.profiler.experimental.Profile('logdir'):
      # Train the model here
      pass
  ```

Note: Running the Profiler for too long can cause it to run out of memory. It is
recommended to profile no more than 10 steps at a time. Avoid profiling the
first few batches to avoid inaccuracies due to initialization overhead.

- Sampling mode: Perform on-demand profiling by using
  `tf.profiler.experimental.server.start` to start a gRPC server with your
  TensorFlow model run. After starting the gRPC server and running your model,
  you can capture a profile through the **Capture Profile** button in XProf.
  Use the script in the Install profiler section above to launch a TensorBoard
  instance if it is not already running.

  As an example,

  ```python
  # Start a profiler server before your model runs.
  tf.profiler.experimental.server.start(6009)
  # (Model code goes here).
  #  Send a request to the profiler server to collect a trace of your model.
  tf.profiler.experimental.client.trace('grpc://localhost:6009',
                                        'gs://your_tb_logdir', 2000)
  ```

  An example for profiling multiple workers:

  ```python
  # E.g., your worker IP addresses are 10.0.0.2, 10.0.0.3, 10.0.0.4, and you
  # would like to profile for a duration of 2 seconds.
  tf.profiler.experimental.client.trace(
      'grpc://10.0.0.2:8466,grpc://10.0.0.3:8466,grpc://10.0.0.4:8466',
      'gs://your_tb_logdir',
      2000)
  ```

![](./images/capture_profile.png)

Use the **Capture Profile** dialog to specify:

- A comma-delimited list of profile service URLs or TPU names.
- A profiling duration.
- The level of device, host, and Python function call tracing.
- How many times you want the Profiler to retry capturing profiles if
  unsuccessful at first.

### Profiling custom training loops

To profile custom training loops in your TensorFlow code, instrument the
training loop with the `tf.profiler.experimental.Trace` API to mark the step
boundaries for XProf.

The `name` argument is used as a prefix for the step names, the `step_num`
keyword argument is appended in the step names, and the `_r` keyword argument
makes this trace event get processed as a step event by XProf.

As an example,

```python
for step in range(NUM_STEPS):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_data = next(dataset)
        train_step(train_data)
```

This will enable XProf's step-based performance analysis and cause the step
events to show up in the trace viewer.

Make sure that you include the dataset iterator within the
`tf.profiler.experimental.Trace` context for accurate analysis of the input
pipeline.

The code snippet below is an anti-pattern:

Warning: This will result in inaccurate analysis of the input pipeline.

```python
for step, train_data in enumerate(dataset):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_step(train_data)
```

### Profiling use cases

The profiler covers a number of use cases along four different axes. Some of the
combinations are currently supported and others will be added in the future.
Some of the use cases are:

- _Local vs. remote profiling_: These are two common ways of setting up your
  profiling environment. In local profiling, the profiling API is called on the
  same machine your model is executing, for example, a local workstation with
  GPUs. In remote profiling, the profiling API is called on a different machine
  from where your model is executing, for example, on a Cloud TPU.
- _Profiling multiple workers_: You can profile multiple machines when using the
  distributed training capabilities of TensorFlow.
- _Hardware platform_: Profile CPUs, GPUs, and TPUs.

The table below provides a quick overview of the TensorFlow-supported use cases
mentioned above:

| Profiling API                | Local     | Remote    | Multiple  | Hardware  |
:                              :           :           : workers   : Platforms :
| :--------------------------- | :-------- | :-------- | :-------- | :-------- |
| **TensorBoard Keras          | Supported | Not       | Not       | CPU, GPU  |
: Callback**                   :           : Supported : Supported :           :
| **`tf.profiler.experimental` | Supported | Not       | Not       | CPU, GPU  |
: start/stop [API][API_0]**    :           : Supported : Supported :           :
| **`tf.profiler.experimental` | Supported | Supported | Supported | CPU, GPU, |
: client.trace [API][API_1]**  :           :           :           : TPU       :
| **Context manager API**      | Supported | Not       | Not       | CPU, GPU  |
:                              :           : supported : Supported :           :

[API_0]: https://www.tensorflow.org/api_docs/python/tf/profiler/experimental#functions_2
[API_1]: https://www.tensorflow.org/api_docs/python/tf/profiler/experimental/client/trace

## Additional resources

- The
  [TensorFlow Profiler: Profile model performance](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)
  tutorial with Keras and TensorBoard where you can apply the advice in this
  guide.
- The
  [Performance profiling in TensorFlow 2](https://www.youtube.com/watch?v=pXHAQIhhMhI)
  talk from the TensorFlow Dev Summit 2020.
- The [TensorFlow Profiler demo](https://www.youtube.com/watch?v=e4_4D7uNvf8)
  from the TensorFlow Dev Summit 2020.

## Known limitations

### Profiling multiple GPUs on TensorFlow 2.2 and TensorFlow 2.3

TensorFlow 2.2 and 2.3 support multiple GPU profiling for single host systems
only; multiple GPU profiling for multi-host systems is not supported. To profile
multi-worker GPU configurations, each worker has to be profiled independently.
From TensorFlow 2.4 multiple workers can be profiled using the
`tf.profiler.experimental.client.trace` API.

CUDA® Toolkit 10.2 or later is required to profile multiple GPUs. As TensorFlow
2.2 and 2.3 support CUDA® Toolkit versions only up to 10.1, you need to create
symbolic links to `libcudart.so.10.1` and `libcupti.so.10.1`:

```shell
sudo ln -s /usr/local/cuda/lib64/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.1
sudo ln -s /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.2 /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.1
```
