# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Playwright fixtures and XProf server lifecycle management."""

# pylint: disable=redefined-outer-name,g-doc-args
# pylint: disable=g-doc-return-or-yield,g-short-docstring-punctuation

from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterator
import contextlib
import dataclasses
import http
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

from tests.ui import journey_capture
from tests.ui import sxs_diff_engine
from tests.ui import ui_helpers

# Server configuration
HOST = os.environ.get("XPROF_HOST", "127.0.0.1")
STARTUP_TIMEOUT_SECONDS = 15.0
MAX_STARTUP_ATTEMPTS = 3
RELATIVE_PROFILE_DATA_DIR = pathlib.Path("demo/plugins/profile")

# Known upstream bugs in the unpatched server baseline, scoped as
# `(message_substring, resource_or_page_url_substring)` pairs so unrelated app
# errors on other routes are never silenced.
KNOWN_UPSTREAM_BUGS: tuple[tuple[str, str], ...] = (
    ("Content Security Policy", "gstatic."),
    ("Failed to load resource", "gstatic."),
    ("Logging before InitGoogle()", "trace_viewer_v2.js"),
    ("absl::InitializeLog()", "trace_viewer_v2.js"),
    ("imgui_webgpu_backend.cc", "trace_viewer_v2.js"),
)


def _iter_repo_root_candidates() -> Iterator[pathlib.Path]:
  """Yields resolved candidate repository roots in priority order."""
  if env_root := os.environ.get("XPROF_REPO_ROOT"):
    candidate = pathlib.Path(env_root).resolve()
    if candidate.is_dir():
      yield candidate

  current = pathlib.Path(__file__).resolve()
  for parent in [current, *current.parents]:
    if (parent / "pytest.ini").exists() or (parent / ".git").exists():
      yield parent
      break

  if test_srcdir := os.environ.get("TEST_SRCDIR"):
    test_ws = os.environ.get("TEST_WORKSPACE", "")
    subs = [f"{test_ws}/third_party/xprof", test_ws] if test_ws else []
    for sub in (*subs, "third_party/xprof", "xprof"):
      candidate = (pathlib.Path(test_srcdir) / sub).resolve()
      if candidate.is_dir():
        yield candidate

  yield pathlib.Path.cwd().resolve()


def _find_repo_root() -> pathlib.Path:
  """Locates the repository root from XPROF_REPO_ROOT, markers, or runfiles."""
  return next(_iter_repo_root_candidates())


def _source_logdir() -> str:
  """Resolves the absolute path to the demo profile dataset directory."""
  if custom_logdir := os.environ.get("XPROF_LOGDIR"):
    resolved_custom = pathlib.Path(custom_logdir).resolve()
    profile_subdir = resolved_custom / "plugins" / "profile"
    if profile_subdir.is_dir():
      return str(profile_subdir)
    return str(resolved_custom)

  for root in _iter_repo_root_candidates():
    candidate = root / RELATIVE_PROFILE_DATA_DIR
    if candidate.is_dir():
      return str(candidate)

  current = pathlib.Path(__file__).resolve().parent
  for parent in [current, *current.parents]:
    candidate = parent / RELATIVE_PROFILE_DATA_DIR
    if candidate.is_dir():
      return str(candidate)

  raise FileNotFoundError(
      f"Could not locate demo profile directory '{RELATIVE_PROFILE_DATA_DIR}'."
  )


def _symlink_traces(src_dir: str | pathlib.Path, dst_dir: pathlib.Path) -> None:
  """Symlinks `run/*.xplane.pb` trace files from `src_dir` into `dst_dir`."""
  for trace in pathlib.Path(src_dir).glob("*/*.xplane.pb"):
    (dst_dir / trace.parent.name).mkdir(exist_ok=True)
    (dst_dir / trace.parent.name / trace.name).symlink_to(trace.resolve())


@pytest.fixture(scope="session")
def logdir() -> Iterator[str]:
  """Links the demo traces into a per-session directory the server writes to.

  The server caches converted tool data next to each trace, so serving the
  checked-in directory would leave those files in the source tree.

  Yields:
    Path to the profile log directory for the session.

  Raises:
    RuntimeError: If `XPROF_STAGE_SYNTHETIC_RUNS=1` and `gpu-training` is
      absent.
  """
  source_dir = _source_logdir()
  if (
      os.environ.get("XPROF_LOGDIR")
      and os.environ.get("XPROF_STAGE_SYNTHETIC_RUNS") != "1"
      and os.environ.get("XPROF_SERVER_URL")
  ):
    # An external server (Kokoro) already serves this directory, so tool URLs
    # must name its paths rather than those of a copy.
    yield source_dir
    return
  with tempfile.TemporaryDirectory(prefix="xprof_logdir_") as staged_root:
    staged = pathlib.Path(staged_root, "demo", "plugins", "profile")
    staged.mkdir(parents=True)
    _symlink_traces(source_dir, staged)
    if (
        not os.environ.get("XPROF_LOGDIR")
        or os.environ.get("XPROF_STAGE_SYNTHETIC_RUNS") == "1"
    ):
      ui_helpers.stage_synthesized_runs(staged, _find_repo_root())
      if (
          os.environ.get("XPROF_STAGE_SYNTHETIC_RUNS") == "1"
          and not (staged / "gpu-training").is_dir()
      ):
        raise RuntimeError(
            "XPROF_STAGE_SYNTHETIC_RUNS=1 requires gpu-training fixture"
            " (convert/test_xplanes/gpu_training_2.xplane.pb)."
        )
    yield str(staged)


@pytest.fixture(scope="session")
def synthesized_logdir(logdir: str) -> Iterator[str]:
  """Provides a writable directory containing staged synthesized trace runs."""
  required_runs = tuple(sorted(ui_helpers.SYNTHESIZED_RUNS - {"gpu-training"}))
  if all(os.path.isdir(os.path.join(logdir, r)) for r in required_runs):
    yield logdir
    return
  with tempfile.TemporaryDirectory(prefix="xprof_synth_") as synth_root:
    staged = pathlib.Path(synth_root, "demo", "plugins", "profile")
    staged.mkdir(parents=True)
    _symlink_traces(logdir, staged)
    ui_helpers.stage_synthesized_runs(staged, _find_repo_root())
    yield str(staged)


@pytest.fixture(scope="session")
def resolve_run_dir(
    logdir: str,
    synthesized_logdir: str,
    resolve_run: Callable[[str], str],
) -> Callable[[str], tuple[str, str]]:
  """Resolves `(run_name, session_dir)` across staged logdirs."""

  def _resolve(run: str) -> tuple[str, str]:
    if run in ui_helpers.SYNTHESIZED_RUNS:
      synth_candidate = os.path.join(synthesized_logdir, run)
      if os.path.isdir(synth_candidate):
        return run, synth_candidate
    resolved = resolve_run(run)
    return resolved, os.path.join(logdir, resolved)

  return _resolve


def _allocate_port_pair(host: str) -> tuple[int, int]:
  """Allocates two distinct ephemeral TCP ports simultaneously on `host`."""
  with (
      socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1,
      socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2,
  ):
    s1.bind((host, 0))
    s2.bind((host, 0))
    return s1.getsockname()[1], s2.getsockname()[1]


def _is_server_ready(
    url: str,
    expected_runs: set[str] | None = None,
    profile_logdir: str | None = None,
) -> bool:
  """Polls the HTTP server until the profile plugin reports ready."""
  return ui_helpers.is_server_ready(url, expected_runs, profile_logdir)


@pytest.fixture(scope="session")
def browser_type_launch_args(
    browser_type_launch_args: dict[str, Any],
) -> dict[str, Any]:
  """Enables software WebGPU flags on headless Linux Chromium."""
  if not sys.platform.startswith("linux"):
    return browser_type_launch_args
  return {
      **browser_type_launch_args,
      "args": [
          *browser_type_launch_args.get("args", ()),
          "--enable-unsafe-webgpu",
          "--enable-unsafe-swiftshader",
          "--use-webgpu-adapter=swiftshader",
          "--use-angle=swiftshader",
          "--use-vulkan=swiftshader",
          "--enable-features=Vulkan",
          "--ignore-gpu-blocklist",
          "--disable-vulkan-surface",
          "--disable-gpu-watchdog",
      ],
  }


@pytest.fixture
def context(context: Any) -> Any:
  """Blocks external Google Fonts egress across all Playwright tests."""
  journey_capture.block_external_fonts(context)
  return context


@pytest.fixture(scope="session")
def server_url(logdir: str) -> Iterator[str]:
  """Launches the XProf web server and provides its URL to all tests."""
  if existing_url := os.environ.get("XPROF_SERVER_URL"):
    normalized = existing_url.strip().rstrip("/")
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
      if _is_server_ready(normalized):
        yield normalized
        return
      time.sleep(0.1)
    raise RuntimeError(
        f"External XPROF_SERVER_URL {normalized} did not respond ready."
    )

  binary = shutil.which("xprof") or os.path.join(
      os.path.dirname(sys.executable), "xprof"
  )
  if not binary or not os.path.exists(binary):
    raise RuntimeError(
        "`xprof` executable was not found on PATH or in current virtualenv."
    )

  server = None
  url = None
  server_logdir = ui_helpers.resolve_server_logdir(logdir)
  expected_runs = {
      p.name for p in pathlib.Path(logdir).iterdir() if p.is_dir()
  }

  with (
      tempfile.TemporaryDirectory(prefix="xprof_srv_tmp_") as server_tmpdir,
      tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file,
  ):
    server_env = {**os.environ, "TMPDIR": server_tmpdir}
    for attempt in range(MAX_STARTUP_ATTEMPTS):
      port, grpc_port = _allocate_port_pair(HOST)
      url = f"http://{HOST}:{port}"
      server = subprocess.Popen(
          [
              binary,
              "server",
              f"--logdir={server_logdir}",
              f"--port={port}",
              f"--grpc_port={grpc_port}",
          ],
          env=server_env,
          stdout=subprocess.DEVNULL,
          stderr=stderr_file,
          text=True,
      )

      deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
      ready = False
      while time.monotonic() < deadline and server.poll() is None:
        if _is_server_ready(url, expected_runs, logdir):
          ready = True
          break
        time.sleep(0.1)
      if not ready or server.poll() is not None:
        server.kill()
        try:
          server.wait(timeout=5)
        except subprocess.TimeoutExpired:
          pass
        if attempt < MAX_STARTUP_ATTEMPTS - 1:
          stderr_file.seek(0)
          stderr_file.truncate(0)
        continue
      break
    else:
      stderr_file.seek(0)
      stderr = stderr_file.read()
      raise RuntimeError(
          f"Server failed to start after {MAX_STARTUP_ATTEMPTS} attempts:\n"
          f"{stderr}"
      )

    try:
      yield url
    finally:
      if server:
        rc = server.poll()
        if rc is not None:
          stderr_file.seek(0)
          tail = "\n".join(stderr_file.read().splitlines()[-40:])
          print(
              f"\n[xprof server exited unexpectedly with code {rc}]\n{tail}",
              file=sys.stderr,
          )
        else:
          server.terminate()
          try:
            server.wait(timeout=5)
          except subprocess.TimeoutExpired:
            server.kill()
            try:
              server.wait(timeout=5)
            except subprocess.TimeoutExpired:
              pass


@pytest.fixture(scope="session")
def baseline_server_url(server_url: str) -> str:
  """Returns the baseline server URL, defaulting to the candidate server."""
  raw = os.environ.get("XPROF_BASELINE_SERVER_URL", "").strip().rstrip("/")
  return raw or server_url


@pytest.fixture(scope="session")
def resolve_run(logdir: str) -> Callable[[str], str]:
  """Returns a resolver mapping a declared run name onto one in the logdir."""
  return sxs_diff_engine.make_run_resolver(logdir)


@pytest.fixture
def open_tool(
    page: Page,
    server_url: str,
    resolve_run_dir: Callable[[str], tuple[str, str]],
) -> Callable[..., str]:
  """Returns a function that deep-links the page to a tool of a demo run.

  The function maps the declared run onto one in the logdir (or staged
  synthesized logdir), opens the tool with any extra query parameters, and
  returns the run it opened.
  """

  def _open(
      run: str, tag: str, *, verify_url: bool = True, **params: str
  ) -> str:
    resolved_run, session_dir = resolve_run_dir(run)
    page.goto(
        ui_helpers.build_tool_url(
            server_url, session_dir, resolved_run, tag, **params
        ),
        wait_until="domcontentloaded",
    )
    if verify_url:
      expect(page).to_have_url(
          journey_capture.settled_tool_url_pattern(tag), timeout=20000
      )
    return resolved_run

  return _open


@dataclasses.dataclass
class BrowserErrors:
  """Captures unexpected browser console errors, page crashes, and HTTP 5xx."""

  page_errors: list[str] = dataclasses.field(default_factory=list)
  console_errors: list[str] = dataclasses.field(default_factory=list)
  http_errors: list[str] = dataclasses.field(default_factory=list)
  _event_urls: dict[tuple[str, int], tuple[str, str]] = dataclasses.field(
      default_factory=dict, repr=False
  )
  _forgiven: set[tuple[str, int]] = dataclasses.field(
      default_factory=set, repr=False
  )

  def attach(self, page: Page) -> None:
    """Attaches console, pageerror, and HTTP 5xx listeners to `page`."""

    def on_console(msg: Any) -> None:
      if msg.type in ("error", "assert"):
        idx = len(self.console_errors)
        self._event_urls[("console", idx)] = (
            msg.location.get("url", ""),
            page.url,
        )
        self.console_errors.append(f"[{msg.type.upper()}] {msg.text}")

    def on_pageerror(exc: Any) -> None:
      idx = len(self.page_errors)
      self._event_urls[("page", idx)] = ("", page.url)
      self.page_errors.append(f"[UNHANDLED] {exc}")

    def on_response(response: Any) -> None:
      if (
          response.status >= http.HTTPStatus.INTERNAL_SERVER_ERROR
          and "/data/plugin/profile/" in response.url
      ):
        idx = len(self.http_errors)
        self._event_urls[("http", idx)] = (response.url, page.url)
        self.http_errors.append(
            f"[HTTP {response.status}] {response.request.method} {response.url}"
        )

    page.on("console", on_console)
    page.on("pageerror", on_pageerror)
    page.on("response", on_response)

  @contextlib.contextmanager
  def ignoring(self, *patterns: str, url_pattern: str = "") -> Iterator[None]:
    """Forgives matching errors emitted inside this `with` block."""
    starts = (
        ("page", len(self.page_errors), self.page_errors),
        ("console", len(self.console_errors), self.console_errors),
        ("http", len(self.http_errors), self.http_errors),
    )
    try:
      yield
    finally:
      for kind, start, bucket in starts:
        for idx in range(start, len(bucket)):
          res_url, pg_url = self._event_urls.get((kind, idx), ("", ""))
          if any(p in bucket[idx] for p in patterns) and (
              not url_pattern or url_pattern in res_url or url_pattern in pg_url
          ):
            self._forgiven.add((kind, idx))

  def unhandled_errors(self) -> list[str]:
    """Returns active errors not matched by known or scoped ignore rules."""
    active: list[str] = []
    for kind, bucket in (
        ("page", self.page_errors),
        ("console", self.console_errors),
        ("http", self.http_errors),
    ):
      for idx, err in enumerate(bucket):
        if (kind, idx) in self._forgiven:
          continue
        res_url, pg_url = self._event_urls.get((kind, idx), ("", ""))
        if any(
            pat in err and (url_sub in res_url or url_sub in pg_url)
            for pat, url_sub in KNOWN_UPSTREAM_BUGS
        ):
          continue
        active.append(err)
    return active

  def assert_clean(self, context: str = "") -> None:
    """Verifies that no unhandled errors occurred during the test."""
    active = self.unhandled_errors()
    ctx = f" during {context}" if context else ""
    assert not active, f"Unexpected browser errors{ctx}:\n" + "\n".join(active)


def dump_failure_artifacts_for_page(
    page: Page | None,
    artifact_stem: str,
    tracker: BrowserErrors | None = None,
) -> pathlib.Path | None:
  """Writes failure screenshot, DOM HTML, and console logs for `page`.

  Returns:
    Output directory containing the written artifacts, or None when artifact
    dumping is disabled or `page` is None.
  """
  out_root = ui_helpers.resolve_artifacts_dir()
  if not out_root or page is None:
    return None
  out_dir = pathlib.Path(out_root)
  out_dir.mkdir(parents=True, exist_ok=True)
  safe_name = "".join(
      ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
      for ch in artifact_stem
  )[:180]
  try:
    page.screenshot(path=str(out_dir / f"{safe_name}.png"), full_page=True)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    sys.stderr.write(f"Failed to capture screenshot for {safe_name}: {exc}\n")
  try:
    (out_dir / f"{safe_name}.html").write_text(
        page.content(), encoding="utf-8"
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    sys.stderr.write(f"Failed to capture DOM HTML for {safe_name}: {exc}\n")
  if tracker is not None:
    lines = tracker.page_errors + tracker.console_errors + tracker.http_errors
    try:
      (out_dir / f"{safe_name}.console.log").write_text(
          "\n".join(lines), encoding="utf-8"
      )
    except OSError as exc:
      sys.stderr.write(
          f"Failed to capture console log for {safe_name}: {exc}\n"
      )
  return out_dir


def _dump_failure_artifacts(
    item: pytest.Item, tracker: BrowserErrors | None = None
) -> None:
  """Writes failure screenshot, DOM HTML, and console logs to artifact dir."""
  if getattr(item, "_xprof_artifacts_dumped", False):
    return
  funcargs = getattr(item, "funcargs", {})
  page: Page | None = funcargs.get("page")
  if dump_failure_artifacts_for_page(page, item.nodeid, tracker) is not None:
    setattr(item, "_xprof_artifacts_dumped", True)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[object]
) -> Generator[None, Any, None]:
  """Captures diagnostic browser artifacts whenever a Playwright test fails."""
  del call
  outcome = yield
  report = outcome.get_result()
  if report.failed and report.when in ("call", "teardown"):
    funcargs = getattr(item, "funcargs", {})
    _dump_failure_artifacts(item, funcargs.get("browser_errors"))


@pytest.fixture(autouse=True)
def browser_errors(request: pytest.FixtureRequest) -> Iterator[BrowserErrors]:
  """Automatically hooks into browser logs for every Playwright page test."""
  tracker = BrowserErrors()
  if "page" in request.fixturenames:
    tracker.attach(request.getfixturevalue("page"))
  yield tracker
  try:
    tracker.assert_clean()
  except AssertionError:
    _dump_failure_artifacts(request.node, tracker)
    raise
