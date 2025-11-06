"""Conftest for xprof integration tests."""

import os
import shutil
import socket
import subprocess
import sys
import time
from absl import logging
import pytest


def get_free_port():
  """Finds a random free port on the local machine."""
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return s.getsockname()[1]


def wait_for_port(port, timeout=60):
  """Waits until a port is actively accepting connections."""
  start_time = time.time()
  while time.time() - start_time < timeout:
    try:
      with socket.create_connection(("localhost", port), timeout=1):
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
      time.sleep(0.5)
  return False


@pytest.fixture(scope="session")
def sample_log_dir():
  """Locates the sample logs directory relative to this test file."""
  base_dir = os.path.dirname(__file__)
  data_dir = os.path.join(base_dir, "data", "test_xplanes")

  if not os.path.isdir(data_dir):
    pytest.fail(f"Regression test data missing at: {data_dir}")
  return data_dir


@pytest.fixture(scope="session")
def server_url(log_dir):
  """Starts the installed 'xprof' CLI command as a subprocess."""
  port = get_free_port()

  executable = shutil.which("xprof")
  if not executable:
    pytest.fail(
        "Could not find 'xprof' executable in PATH. Is the package installed?"
    )

  cmd = [
      executable,
      "--logdir",
      log_dir,
      "--port",
      str(port),
  ]

  logging.info("Launching server: %s", " ".join(cmd))

  proc = subprocess.Popen(
      cmd, stdout=sys.stdout, stderr=sys.stderr, start_new_session=True
  )

  try:
    if wait_for_port(port, timeout=60):
      yield f"http://localhost:{port}"
    else:
      pytest.fail(f"Server failed to start on port {port}")
  finally:
    logging.info("Tearing down server...")
    proc.terminate()
    try:
      proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
      logging.warning("Server did not exit gracefully, forcing kill.")
      proc.kill()
