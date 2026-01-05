# Building an XProf Docker Image

This document describes how to build a Docker image for XProf using the official
release from PyPI.

## Dockerfile

Create a file named `Dockerfile` with the following content:

**Note**: The following Dockerfile provides a basic configuration for running
XProf. You may need to modify it to fit your specific security requirements,
base image preferences, or other environmental needs.

```dockerfile
FROM python:3.11-slim

ARG XPROF_VERSION=2.21.0

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir xprof==${XPROF_VERSION}

EXPOSE 8791 50051

ENTRYPOINT ["xprof"]

CMD ["--logdir=/app/logs", "--port=8791"]
```

## How to Build

1.  Save the content above as `Dockerfile` in an empty directory.
2.  Build the image using the following command:

    ```bash
    docker build --platform=linux/amd64 -t xprof:2.21.0 .
    ```

You can change the version by modifying the `XPROF_VERSION` argument in the
Dockerfile or by passing `--build-arg XPROF_VERSION=<version>` to the `docker
build` command.

## How to Run

### Run with Local Logs

Map your local log directory to `/app/logs` in the container.

```bash
docker run -p 8791:8791 \
  -v /tmp/xprof_logs:/app/logs \
  xprof:2.21.0
```

### Run with GCS Logs

Mount your local gcloud credentials so xprof can authenticate with Google Cloud
Storage.

```bash
docker run -p 8791:8791 \
  -v ~/.config/gcloud:/root/.config/gcloud \
  xprof:2.21.0 \
  --logdir=gs://your-bucket-name/xprof_logs --port=8791
```
