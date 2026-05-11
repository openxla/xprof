# Profiling JAX on GPUs with XProf and ML Diagnostics

Optimizing large-scale JAX models on GPUs requires deep visibility into
performance bottlenecks. This guide provides a comprehensive, end-to-end (E2E)
workflow for running and profiling JAX workloads on GPUs (such as NVIDIA L4)
using Google Cloud ML Diagnostics and XProf. By leveraging these tools, you can
identify inefficient operations, optimize compute resource usage, and accelerate
your training runs.

> Note: For more context on managed Machine Learning Diagnostics and support for
> TPUs, refer also to the official
> [Google Cloud ML Diagnostics](https://docs.cloud.google.com/tpu/docs/ml-diagnostics/overview)
> documentation.

By following this guide, you will learn how to:

1.  Instrument a simple JAX training loop for profiling.
2.  Containerize the workload with appropriate CUDA support.
3.  Deploy the workload on Google Kubernetes Engine (GKE) using JobSet.
4.  Capture and visualize performance profiles dynamically.

--------------------------------------------------------------------------------

## Prerequisites

Before you begin, ensure you have:

*   A Google Cloud project with billing enabled.
*   A GKE cluster with GPU support (e.g., NVIDIA L4).
*   A Google Cloud Storage (GCS) bucket to store profiles.
*   `gcloud` and `kubectl` CLIs installed and configured.
*   Workload Identity configured for your GKE cluster to access GCS.

--------------------------------------------------------------------------------

## Step 1: Instrumenting the JAX Workload

First, create a JAX training script (e.g., `train.py`). We use the
`google-cloud-mldiagnostics` SDK to interact with the managed profiling
infrastructure.

**Important:** Ensure you replace placeholders like `<your-project-id>` and
`<your-gcs-bucket>` with your actual Google Cloud project details.

> [!WARNING] The script below includes an infinite loop to keep the GPU busy for
> on-demand profiling demonstrations. Remember to manually stop the job or
> delete the GKE resources after you are done to avoid unnecessary billing
> costs.

```python
import logging
import os
import time
from google_cloud_mldiagnostics import machinelearning_run
from google_cloud_mldiagnostics import xprof
import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Starting JAX training job...")

    # Coordinates multihost collective operations and healthchecks
    jax.distributed.initialize()

    logging.info(
        f"JAX initialized: process_index={jax.process_index()}, "
        f"process_count={jax.process_count()}"
    )

    # Syncs metadata with the mldiag hook & launches reverse proxy daemons
    machinelearning_run(
        name=f"jax-gpu-run-{int(time.time())}",
        configs={"learning_rate": 1e-5, "batch_size": 8192},
        project=os.environ.get("PROJECT_ID", "<your-project-id>"),
        region=os.environ.get("REGION", "us-central1"),
        gcs_path=os.environ.get("GCS_BUCKET", "gs://<your-gcs-bucket>"),
        on_demand_xprof=True,
    )

    key = jax.random.PRNGKey(0)
    size = 4096
    matrix = jax.random.normal(key, (size, size), dtype=jnp.float32)

    def train_step(x):
        return jnp.dot(x, x)

    train_step = jax.jit(train_step)

    # Triggers XLA compilation ahead of tracing steps so compilation overhead isn't profiled
    matrix = train_step(matrix)
    matrix.block_until_ready() # Wait for compilation to complete.

    prof = xprof()
    prof.start(session_id="warmup_phase")

    for _ in range(5):
        matrix = train_step(matrix)
        matrix.block_until_ready()

    prof.stop()
    logging.info("Programmatic profile capture complete.")

    logging.info("Entering training loop. Ready for on-demand profiling...")
    try:
        while True:
            # Continuously pump steps keeping GPUs occupied for on-demand capture triggers
            matrix = train_step(matrix)
            matrix.block_until_ready() # Ensure GPU work completes before next step.
            time.sleep(0.5)
    except KeyboardInterrupt:
        logging.info("Training loop interrupted.")

if __name__ == "__main__":
    main()
```

--------------------------------------------------------------------------------

## Step 2: Containerization (Dockerfile)

Create a `Dockerfile` to package your JAX script with the required CUDA
dependencies and the ML Diagnostics SDK.

```dockerfile
# Use an official NVIDIA CUDA base image compatible with JAX
FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04

# Install Python, venv, and other OS dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up a virtual environment and update PATH to use it implicitly
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# At this point, pip and python implicitly map to the virtual env!
# No need for --break-system-packages.

# Upgrade pip inside the venv
RUN pip install --upgrade pip

# Install JAX with CUDA support
RUN pip install --upgrade "jax[cuda13]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install ML Diagnostics SDK and XProf tools
RUN pip install --no-cache-dir \
    google-cloud-mldiagnostics \
    xprof-nightly

WORKDIR /app
COPY train.py .

CMD ["python3", "train.py"]
```

Build and push the image to Artifact Registry:

```bash
docker build -t us-central1-docker.pkg.dev/<project-id>/<repo>/jax-gpu-workload:latest .
docker push us-central1-docker.pkg.dev/<project-id>/<repo>/jax-gpu-workload:latest
```

--------------------------------------------------------------------------------

## Step 3: Deployment (Kubernetes Manifest)

Deploy the workload using a GKE **JobSet** or standard Job. To enable the ML
Diagnostics platform to inject metadata and route profile requests, apply the
label `managed-mldiagnostics-gke: "true"`. For more details on configuring GKE
for ML Diagnostics, refer to the official
[GKE Setup Guide](https://docs.cloud.google.com/tpu/docs/ml-diagnostics/gke).

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: jax-gpu-job
  namespace: ai-workloads
  labels:
    managed-mldiagnostics-gke: "true"
spec:
  replicatedJobs:
  - name: gpu-nodes
    replicas: 1
    template:
      spec:
        parallelism: 1
        completions: 1
        backoffLimit: 0
        template:
          metadata:
            labels:
              managed-mldiagnostics-gke: "true"
          spec:
            # Must match the GKE Service Account with Workload Identity permissions
            serviceAccountName: <your-service-account>
            hostNetwork: true
            dnsPolicy: ClusterFirstWithHostNet
            nodeSelector:
              cloud.google.com/gke-accelerator: nvidia-l4 # Or other GPU
            containers:
            - name: workload
              image: us-central1-docker.pkg.dev/<project-id>/<repo>/jax-gpu-workload:latest
              imagePullPolicy: Always
              # Expose ports required for profile daemons
              ports:
              - containerPort: 8471 # JAX distributed coordinator port
              - containerPort: 8080 # ML Diagnostics agent/proxy port
              - containerPort: 9999 # XProf server port for on-demand profiling
              resources:
                limits:
                  nvidia.com/gpu: 1
```

Apply the manifest:

```bash
kubectl apply -f deploy.yaml
```

--------------------------------------------------------------------------------

## Step 4: Capture & Visualization

### Programmatic Capture

If you included `prof.start()` / `prof.stop()` in your script, those profiles
are automatically uploaded to your GCS bucket under the path:
`gs://<your-gcs-bucket>/<run-name>/plugins/profile/<session-id>/`

### On-Demand Capture

Because `on_demand_xprof=True` is set in `machinelearning_run`, you can capture
profiles dynamically while the job is running.

For detailed instructions on how to use the TensorBoard UI to trigger on-demand
profiles, select specific pods, and view the captured traces, please refer to
the official public documentation:
[Google Cloud ML Diagnostics - On-demand profile capture](https://docs.cloud.google.com/tpu/docs/ml-diagnostics/sdk#on-demand-profile).

You can also capture profiles using the gcloud CLI as described in the
[ML Diagnostics CLI Guide](https://docs.cloud.google.com/tpu/docs/ml-diagnostics/cli).

This public documentation applies to both TPU and GPU workloads managed by ML
Diagnostics.
