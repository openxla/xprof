# Building an XProf Docker Image

This document describes how to build a Docker image for XProf using the official
release from PyPI.

## Dockerfile

Create a file named `Dockerfile` with the following content:

**Note**: The following Dockerfile provides a basic configuration for running
XProf. You may need to modify it to fit your specific security requirements,
base image preferences, or other environmental needs.

```dockerfile
FROM python:3.12-slim

ARG XPROF_VERSION=2.21.3

ENV PYTHONUNBUFFERED=1 \
    XPROF_HOST=0.0.0.0

WORKDIR /app

RUN pip install --no-cache-dir xprof==${XPROF_VERSION}

EXPOSE 8791 50051

ENTRYPOINT ["xprof"]

CMD ["--logdir=/app/logs", "--port=8791"]
```

**Note**: `xprof` versions after `2.24.2` bind the HTTP server to `127.0.0.1`
by default. Setting `ENV XPROF_HOST=0.0.0.0` binds the server across all
container interfaces on newer releases while remaining compatible with
`xprof <= 2.24.2` (on versions after `2.24.2`, you can also pass
`--host=0.0.0.0` explicitly).

**Security Warning**: `ENV XPROF_HOST=0.0.0.0` binds across all container
interfaces while keeping `Host`-header validation active for `localhost`,
loopback, and non-unspecified IP literals (set `XPROF_ALLOWED_HOSTS` or
`--allowed_hosts` when serving behind a custom DNS hostname or reverse proxy).
For local browser access, publish ports on host loopback
(`-p 127.0.0.1:8791:8791`). Publishing `-p 8791:8791` without `127.0.0.1:` binds
`0.0.0.0` on the Docker host and exposes unauthenticated profile reads, mounted
Google Cloud Storage (GCS) Application Default Credentials (ADC) reads, and
`/capture_profile` to the network unless protected by a firewall or
authenticating reverse proxy.

## How to Build

1.  Save the content above as `Dockerfile` in an empty directory.
2.  Build the image using the following command:

    ```bash
    docker build --platform=linux/amd64 -t xprof:2.21.3 .
    ```

You can change the version by modifying the `XPROF_VERSION` argument in the
Dockerfile or by passing `--build-arg XPROF_VERSION=<version>` to the `docker
build` command.

## How to Run

### Run with Local Logs

Map your local log directory to `/app/logs` in the container.

```bash
docker run -p 127.0.0.1:8791:8791 \
  -v /tmp/xprof_logs:/app/logs \
  xprof:2.21.3
```

### Run with GCS Logs

Mount your local gcloud credentials so xprof can authenticate with Google Cloud
Storage.

```bash
docker run -p 127.0.0.1:8791:8791 \
  -v ~/.config/gcloud:/root/.config/gcloud \
  xprof:2.21.3 \
  --logdir=gs://your-bucket-name/xprof_logs --port=8791
```
