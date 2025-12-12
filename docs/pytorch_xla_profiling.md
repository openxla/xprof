# Profile PyTorch XLA workloads

Performance optimization is a crucial part of building efficient machine
learning models. You can use the XProf profiling tool to measure the performance
of your machine learning workloads. XProf lets you capture detailed traces of
your model's execution on XLA devices. These traces can help you to identify
performance bottlenecks, understand device utilization, and optimize your code.

This guide describes the process of programmatically capturing a trace from your
PyTorch XLA script and visualizing using XProf.

## Capture a trace programmatically

You can capture a trace by adding a few lines of code to your existing training
script. The primary tool for capturing a trace is the `torch_xla.debug.profiler`
module, which is typically imported with the alias `xp`.

### 1. Start the profiler server

Before you can capture a trace, you need to start the profiler server. This
server runs in the background of your script and collects the trace data. You
can start it by calling `xp.start_server(<port>)` near the beginning of your
main execution block.

### 2. Define the trace duration

Wrap the code you want to profile within `xp.start_trace()` and
`xp.stop_trace()` calls. The `start_trace` function takes a path to a directory
where the trace files are saved.

It's common practice to wrap the main training loop to capture the most relevant
operations.

```python
# The directory where the trace files are stored.
log_dir = '/root/logs/'

# Start tracing
xp.start_trace(log_dir)

# ... your training loop or other code to be profiled ...
train_mnist()

# Stop tracing
xp.stop_trace()
```

### 3. Add custom trace labels

By default, the traces captured are low-level Pytorch XLA functions and can be
hard to navigate. You can add custom labels to specific sections of your code
using the `xp.Trace()` context manager. These labels will appear as named blocks
in the profiler's timeline view, making it much easier to identify specific
operations like data preparation, the forward pass, or the optimizer step.

The following example shows how you can add context to different parts of a
training step.

```python
def forward(self, x):
    # This entire block will be labeled 'forward' in the trace
    with xp.Trace('forward'):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# You can also nest context managers for more granular detail
for batch_idx, (data, target) in enumerate(train_loader):
    with torch_xla.step():
        with xp.Trace('train_step_data_prep_and_forward'):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)

        with xp.Trace('train_step_loss_and_backward'):
            loss = loss_fn(output, target)
            loss.backward()

        with xp.Trace('train_step_optimizer_step_host'):
            optimizer.step()
```

## Complete example

The following example shows how to capture a trace from a PyTorch XLA script,
based on the `mnist_xla.py` file.

```python
import torch
import torch.optim as optim
from torchvision import datasets, transforms

# PyTorch/XLA specific imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp

def train_mnist():
    # ... (model definition and data loading code) ...
    print("Starting training...")
    # ... (training loop as defined in the previous section) ...
    print("Training finished!")

if __name__ == '__main__':
    # 1. Start the profiler server
    server = xp.start_server(9012)

    # 2. Start capturing the trace and define the output directory
    xp.start_trace('/root/logs/')

    # Run the training function that contains custom trace labels
    train_mnist()

    # 3. Stop the trace
    xp.stop_trace()
```

## Visualize the trace

When your script has finished, the trace files are saved in the directory you
specified (for example, `/root/logs/`). You can visualize this trace using
XProf.

You can launch the profiler UI directly using the standalone XProf command by
pointing it to your log directory:

```shell
$ xprof --port=8791 /root/logs/
Attempting to start XProf server:
  Log Directory: /root/logs/
  Port: 8791
  Worker Service Address: 0.0.0.0:50051
  Hide Capture Button: False
XProf at http://localhost:8791/ (Press CTRL+C to quit)
```

Navigate to the provided URL (e.g., http://localhost:8791/) in your browser
to view the profile.

You will be able to see the custom labels you created and analyze the execution
time of different parts of your model.

If you use Google Cloud to run your workloads, we recommend the
[cloud-diagnostics-xprof tool](https://github.com/AI-Hypercomputer/cloud-diagnostics-xprof).
It provides a streamlined profile collection and viewing experience using VMs
running XProf.
