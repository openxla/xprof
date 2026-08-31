"""Playwright fixtures and XProf server lifecycle management."""

# pylint: disable=redefined-outer-name,g-doc-args
# pylint: disable=g-doc-return-or-yield,g-short-docstring-punctuation

import dataclasses
import http
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time
from typing import Generator, List
import urllib.error
import urllib.request
from playwright.sync_api import Page
import pytest

# Server configuration
HOST = os.environ.get("XPROF_HOST", "127.0.0.1")
STARTUP_TIMEOUT_SECONDS = 15.0
MAX_STARTUP_ATTEMPTS = 3
RELATIVE_PROFILE_DATA_DIR = pathlib.Path("demo/plugins/profile")

# Known upstream bugs in the unpatched server baseline (tracked in b/552235521).
# Silenced during baseline test runs so assertions focus on invariant checks.
KNOWN_UPSTREAM_BUGS = [
    "favicon.ico",
    "gstatic.com",
    "Cannot set properties of null",  # Standalone iframe reload bug
    "EmptyError",  # RxJS stream termination without defaultIfEmpty
    "google.visualization",  # Google charts async initialization race
    "DataTable",
    "arrayToDataTable",
    "Invalid column index",  # DataView column filter bug
    "Expected number",  # SVG negative percentage tick
    "Expected length",
]


def _find_repo_root() -> pathlib.Path:
  """Locates repository root by searching upward for standard markers."""
  current = pathlib.Path(__file__).resolve()
  for parent in [current] + list(current.parents):
    if (parent / "pytest.ini").exists() or (parent / ".git").exists():
      return parent

  test_srcdir = os.environ.get("TEST_SRCDIR")
  if test_srcdir:
    for sub in ("google3/third_party/xprof", "third_party/xprof", "xprof"):
      cand = pathlib.Path(test_srcdir) / sub
      if cand.is_dir():
        return cand

  return pathlib.Path.cwd()


@pytest.fixture(scope="session")
def logdir() -> str:
  """Locates the profile data directory dynamically from the repo root."""
  if custom_logdir := os.environ.get("XPROF_LOGDIR"):
    return custom_logdir

  repo_root = _find_repo_root()
  profile_dir = repo_root / RELATIVE_PROFILE_DATA_DIR
  if not profile_dir.is_dir():
    raise FileNotFoundError(
        f"Profile test datasets not found at: {profile_dir}. "
        "Set XPROF_LOGDIR environment variable to override."
    )
  return str(profile_dir)


def _find_free_port(host: str = HOST) -> int:
  """Asks the OS for an available ephemeral port."""
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((host, 0))
    return s.getsockname()[1]


def _is_server_ready(url: str) -> bool:
  """Checks if the server responds with a healthy HTTP status."""
  try:
    with urllib.request.urlopen(url, timeout=0.5) as resp:
      return resp.status < http.HTTPStatus.INTERNAL_SERVER_ERROR
  except urllib.error.HTTPError as err:
    return err.code < http.HTTPStatus.INTERNAL_SERVER_ERROR
  except (urllib.error.URLError, OSError):
    return False


@pytest.fixture(scope="session")
def server_url(logdir: str) -> Generator[str, None, None]:
  """Launches the XProf web server and provides its URL to all tests."""
  if existing_url := os.environ.get("XPROF_SERVER_URL"):
    yield existing_url
    return

  binary = shutil.which("xprof") or os.path.join(
      os.path.dirname(sys.executable), "xprof"
  )
  if not binary or not os.path.exists(binary):
    raise RuntimeError(
        "`xprof` executable was not found on PATH or in current virtualenv."
    )

  server = None
  url = None

  for _ in range(MAX_STARTUP_ATTEMPTS):
    port = _find_free_port(HOST)
    url = f"http://{HOST}:{port}"
    server = subprocess.Popen(
        [binary, f"--logdir={logdir}", f"--port={port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline and server.poll() is None:
      if _is_server_ready(url):
        break
      time.sleep(0.1)
    else:
      server.kill()
      continue
    break
  else:
    _, stderr = (
        server.communicate()
        if server
        else ("", "Server process failed to create")
    )
    raise RuntimeError(
        f"Server failed to start after {MAX_STARTUP_ATTEMPTS} attempts:\n"
        f"{stderr}"
    )

  try:
    yield url
  finally:
    if server:
      server.terminate()
      try:
        server.wait(timeout=5)
      except subprocess.TimeoutExpired:
        server.kill()


@dataclasses.dataclass
class BrowserErrors:
  """Captures unexpected browser console errors and page crashes."""

  page_errors: List[str] = dataclasses.field(default_factory=list)
  console_errors: List[str] = dataclasses.field(default_factory=list)
  ignored_patterns: List[str] = dataclasses.field(default_factory=list)

  def ignore(self, *patterns: str):
    """Allows specific tests to explicitly ignore expected error strings."""
    self.ignored_patterns.extend(patterns)

  def assert_clean(self, context: str = ""):
    """Verifies that no unhandled errors occurred during the test."""
    all_ignored = KNOWN_UPSTREAM_BUGS + self.ignored_patterns
    active = [
        err
        for err in self.page_errors + self.console_errors
        if not any(pattern in err for pattern in all_ignored)
    ]
    ctx = f" during {context}" if context else ""
    assert not active, f"Unexpected browser errors{ctx}:\n" + "\n".join(active)


@pytest.fixture(autouse=True)
def browser_errors(page: Page) -> Generator[BrowserErrors, None, None]:
  """Automatically hooks into browser logs for every test."""
  tracker = BrowserErrors()

  def on_console(msg):
    if msg.type in ("error", "assert"):
      tracker.console_errors.append(f"[{msg.type.upper()}] {msg.text}")

  page.on("console", on_console)
  page.on(
      "pageerror",
      lambda exc: tracker.page_errors.append(f"[UNHANDLED] {exc}"),
  )
  yield tracker
  tracker.assert_clean()
