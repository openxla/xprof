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
"""Tests for server HTTP protocol, packaging integrity, and cleanliness."""

from collections.abc import Callable
import email.message
import gzip
import importlib.util
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import traceback
import types
from typing import Any
import unittest
from unittest import mock
import urllib.error
import urllib.parse
import urllib.request
from wsgiref import simple_server

# pylint: disable=g-import-not-at-top,g-importing-member
try:
  from absl.testing import absltest
except ImportError:
  absltest = None  # pyrefly: ignore[assignment]

try:
  from tensorboard_plugin_profile import (
      profile_plugin,
  )
  from tensorboard_plugin_profile import (
      profile_plugin_loader,
  )
  from tensorboard_plugin_profile import server
  from tensorboard_plugin_profile import (
      standalone,
  )
  base_plugin = standalone.base_plugin
  plugin_event_multiplexer = standalone.plugin_event_multiplexer
  from google3.third_party.xprof.tests.ui import invariants
  from google3.third_party.xprof.tests.ui import ui_helpers
except ImportError:
  try:
    from xprof import profile_plugin  # pyrefly: ignore[missing-import]
    from xprof import profile_plugin_loader  # pyrefly: ignore[missing-import]
    from xprof import server  # pyrefly: ignore[missing-import]
    from xprof.standalone import (  # pyrefly: ignore[missing-import]
        base_plugin,
    )
    from xprof.standalone import (  # pyrefly: ignore[missing-import]
        plugin_event_multiplexer,
    )
    from tests.ui import invariants  # pyrefly: ignore[missing-import]
    from tests.ui import ui_helpers  # pyrefly: ignore[missing-import]
  except ImportError:
    profile_plugin = None  # pyrefly: ignore[assignment]
    profile_plugin_loader = None  # pyrefly: ignore[assignment]
    server = None  # pyrefly: ignore[assignment]
    base_plugin = None  # pyrefly: ignore[assignment]
    plugin_event_multiplexer = None  # pyrefly: ignore[assignment]
    import invariants  # pyrefly: ignore[missing-import]
    import ui_helpers  # pyrefly: ignore[missing-import]
# pylint: enable=g-import-not-at-top,g-importing-member


def _find_repo_root() -> pathlib.Path | None:
  """Locates the repository root across local workspace and runfiles."""
  test_srcdir = os.environ.get("TEST_SRCDIR")
  test_workspace = os.environ.get("TEST_WORKSPACE", "google3")
  if test_srcdir:
    candidate = pathlib.Path(test_srcdir) / test_workspace / "third_party/xprof"
    if (candidate / "frontend" / "app").is_dir() and (
        candidate / "demo"
    ).is_dir():
      return candidate

  search_candidates = [pathlib.Path.cwd().resolve()]
  if "__file__" in globals():
    search_candidates.insert(0, pathlib.Path(__file__).resolve())
  for p in search_candidates:
    for parent in [p, *p.parents]:
      if (parent / "demo").is_dir() and (parent / "frontend" / "app").is_dir():
        return parent

  return None


class _PluginFlags(dict[str, object]):
  """Flags container supporting both dict and attribute access.

  TensorBoard plugin loaders access flags via item lookup (`flags[...]`),
  while internal xprof helpers expect attribute access (`flags.foo`).
  """

  def __init__(self) -> None:
    super().__init__()
    self["master_tpu_unsecure_channel"] = ""
    self.master_tpu_unsecure_channel = ""


class ServerAndPackagingContractTest(unittest.TestCase):
  """Tests for server HTTP protocol, packaging integrity, and cleanliness."""

  server_url: str | None = None
  _httpd: simple_server.WSGIServer | None = None
  _server_thread: threading.Thread | None = None
  _managed_static_dir: bool = False
  # Why the server could not be started, surfaced in the failure message so a
  # startup problem is not reported as a bare missing URL.
  _setup_error: str | None = None

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    existing_url = os.environ.get("XPROF_SERVER_URL")
    if existing_url:
      cls.server_url = existing_url
      return

    if server is None or profile_plugin is None or base_plugin is None:
      cls._setup_error = (
          "xprof server modules did not import under either the google3 or the"
          " open-source module path."
      )
      return

    try:
      if "XPROF_STATIC_DIR" not in os.environ:
        static_candidate = (
            pathlib.Path(profile_plugin.__file__).resolve().parent / "static"
        )
        if (static_candidate / "runtime.js").is_file():
          os.environ["XPROF_STATIC_DIR"] = str(static_candidate)
          cls._managed_static_dir = True
        else:
          repo_root = _find_repo_root()
          if repo_root is not None:
            static_dir = (
                repo_root / "plugin" / "tensorboard_plugin_profile" / "static"
            )
            if (static_dir / "runtime.js").is_file():
              os.environ["XPROF_STATIC_DIR"] = str(static_dir)
              cls._managed_static_dir = True

      flags = _PluginFlags()
      dp = (
          plugin_event_multiplexer.DataProvider("")
          if plugin_event_multiplexer is not None
          else None
      )
      context = base_plugin.TBContext(
          logdir="",
          data_provider=dp,  # pyrefly: ignore[bad-argument-type]
          flags=flags,
      )
      plugin = None
      if profile_plugin_loader is not None:
        loader = profile_plugin_loader.ProfilePluginLoader()
        plugin = loader.load(context)
      if plugin is None:
        plugin = profile_plugin.ProfilePlugin(context)

      if plugin is not None:
        wsgi_app = server.make_wsgi_app(plugin)
        cls._httpd = simple_server.make_server("127.0.0.1", 0, wsgi_app)
        host, port = cls._httpd.server_address[:2]
        cls.server_url = f"http://{host}:{port}"
        cls._server_thread = threading.Thread(
            target=cls._httpd.serve_forever, daemon=True
        )
        cls._server_thread.start()
    except Exception:  # pylint: disable=broad-except
      if cls._httpd is not None:
        cls._httpd.server_close()
        cls._httpd = None
      cls._setup_error = traceback.format_exc()
      cls.server_url = None

  @classmethod
  def tearDownClass(cls) -> None:
    if cls._managed_static_dir:
      os.environ.pop("XPROF_STATIC_DIR", None)
      cls._managed_static_dir = False
    if cls._httpd is not None:
      if cls._server_thread is not None and cls._server_thread.is_alive():
        cls._httpd.shutdown()
    if cls._server_thread is not None:
      cls._server_thread.join(timeout=2.0)
      cls._server_thread = None
    if cls._httpd is not None:
      cls._httpd.server_close()
      cls._httpd = None
    super().tearDownClass()

  def test_runtime_js_serves_javascript_mime(self) -> None:
    """Verifies that runtime.js is served with javascript MIME type."""
    self.assertIsNotNone(
        self.server_url,
        f"XProf server URL must be available for testing: {self._setup_error}",
    )
    url = f"{self.server_url.rstrip('/')}/data/plugin/profile/runtime.js"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
      status = resp.status
      content_type = resp.headers.get("Content-Type", "")
    self.assertEqual(status, 200)
    self.assertIn("javascript", content_type.lower())

  def test_index_html_declares_default_base_href(self) -> None:
    """Verifies packaged index.html declares the default <base href>."""
    self.assertIsNotNone(
        self.server_url,
        f"XProf server URL must be available for testing: {self._setup_error}",
    )
    url = f"{self.server_url.rstrip('/')}/data/plugin/profile/index.html"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=5.0) as response:
      self.assertEqual(response.status, 200)
      raw = response.read()
      if (
          raw.startswith(b"\x1f\x8b")
          or response.headers.get("Content-Encoding") == "gzip"
      ):
        raw = gzip.decompress(raw)
      html = raw.decode("utf-8")

    self.assertIn('<base href="/data/plugin/profile/">', html)

  def test_nonexistent_route_returns_404(self) -> None:
    """Verifies that unmapped static routes return HTTP 404 rather than 200."""
    self.assertIsNotNone(
        self.server_url,
        f"XProf server URL must be available for testing: {self._setup_error}",
    )
    url = (
        f"{self.server_url.rstrip('/')}/data/plugin/profile/"
        "nonexistent_route_abc.json"
    )
    req = urllib.request.Request(url, method="GET")
    status = 0
    try:
      with urllib.request.urlopen(req, timeout=5.0) as resp:
        status = resp.status
    except urllib.error.HTTPError as err:
      status = err.code
      err.close()
    self.assertEqual(status, 404)

  def test_flat_logdir_readiness_and_megascale_conversion(self) -> None:
    """Tests flat-logdir readiness polling and self-contained MegaScale data."""
    self.assertIsNotNone(
        self.server_url,
        f"XProf server URL must be available for testing: {self._setup_error}",
    )
    with tempfile.TemporaryDirectory() as tmp:
      flat_logdir = pathlib.Path(tmp) / "custom_runs"
      flat_logdir.mkdir()
      ui_helpers.stage_synthesized_runs(flat_logdir)

      self.assertTrue(
          ui_helpers.is_server_ready(
              self.server_url, {"megascale-1host"}, str(flat_logdir)
          )
      )
      self.assertFalse(
          ui_helpers.is_server_ready(
              self.server_url, {"nonexistent-run"}, str(flat_logdir)
          )
      )

      mega_dir = flat_logdir / "megascale-2host"
      self.assertEqual(list(mega_dir.glob("*.dcn_collective_stats.pb")), [])
      self.assertFalse((mega_dir / "cache_version.txt").exists())

      def fetch_json(endpoint: str, **params: str) -> Any:
        query = urllib.parse.urlencode(params)
        base = self.server_url.rstrip("/")
        url = f"{base}/data/plugin/profile/{endpoint}?{query}"
        with urllib.request.urlopen(url, timeout=10.0) as resp:
          raw = resp.read()
          if (
              raw.startswith(b"\x1f\x8b")
              or resp.headers.get("Content-Encoding") == "gzip"
          ):
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))

      tools = fetch_json(
          "run_tools", run="megascale-2host", session_path=str(mega_dir)
      )
      self.assertIn("megascale_stats", tools)

      for host in ("ALL_HOSTS", "mega-node-0"):
        payload = fetch_json(
            "data",
            run="megascale-2host",
            tag="megascale_stats",
            host=host,
            session_path=str(mega_dir),
        )
        self.assertIn("all_reduce_dcn_0", json.dumps(payload))

      # Simulate SxS cache clearing (deleting everything except *.xplane.pb)
      # and verify cold recomputation still yields non-empty DcnSlackSummary.
      for child in mega_dir.iterdir():
        if child.is_file() and not child.name.endswith(".xplane.pb"):
          child.unlink()
      recomputed = fetch_json(
          "data",
          run="megascale-2host",
          tag="megascale_stats",
          host="ALL_HOSTS",
          session_path=str(mega_dir),
      )
      self.assertIn("all_reduce_dcn_0", json.dumps(recomputed))

      # Verify a base trace without /host:CPU still produces /host:CPU and rows.
      tpu_only_base = flat_logdir / "tpu_only_base.xplane.pb"
      ui_helpers.write_perf_utilization_trace(tpu_only_base)
      tpu_only_mega_dir = flat_logdir / "megascale-tpu-base"
      ui_helpers.write_megascale_trace(
          tpu_only_mega_dir / "mega-node-0.xplane.pb", tpu_only_base
      )
      tpu_base_tools = fetch_json(
          "run_tools",
          run="megascale-tpu-base",
          session_path=str(tpu_only_mega_dir),
      )
      self.assertIn("megascale_stats", tpu_base_tools)
      tpu_base_data = fetch_json(
          "data",
          run="megascale-tpu-base",
          tag="megascale_stats",
          host="mega-node-0",
          session_path=str(tpu_only_mega_dir),
      )
      self.assertIn("all_reduce_dcn_0", json.dumps(tpu_base_data))

      serving_dir = flat_logdir / "tpu-serving"
      serving_data = fetch_json(
          "data",
          run="tpu-serving",
          tag="inference_profile",
          host="serving-node-1",
          session_path=str(serving_dir),
      )
      self.assertIsInstance(serving_data, list)
      self.assertEqual(len(serving_data), 3)
      self.assertEqual(serving_data[0].get("p", {}).get("hasBatching"), "true")
      self.assertEqual(
          invariants.check_poison_tokens(json.dumps(serving_data)), []
      )

  def test_no_server_cache_files_checked_into_demo_traces(self) -> None:
    """Verifies no server cache artifact is checked into the demo corpus.

    The server writes `.cached_tools.json` beside the traces it reads. Those
    files are machine-specific and must never be committed, but they are easy
    to pick up accidentally after running the server against the demo data.
    This inspects the staged `demo_traces` filegroup under hermetic test runs,
    or verifies tracked files when running directly in a working tree.
    """
    repo_root = _find_repo_root()
    self.assertIsNotNone(
        repo_root, "Repository root not found in current execution environment."
    )
    demo_profile_dir = repo_root / "demo" / "plugins" / "profile"
    self.assertTrue(
        demo_profile_dir.is_dir(),
        f"Demo profile directory not present at {demo_profile_dir}",
    )
    staged_files = [p for p in demo_profile_dir.rglob("*") if p.is_file()]
    # Guards against the filegroup silently going empty, which would make the
    # assertion below pass without inspecting anything.
    self.assertTrue(
        staged_files, f"No demo trace files staged under {demo_profile_dir}"
    )
    candidates = [
        p for p in staged_files if p.name.endswith(".cached_tools.json")
    ]
    polluting_files: list[str] = []
    if os.environ.get("TEST_SRCDIR"):
      polluting_files = [
          p.relative_to(repo_root).as_posix() for p in candidates
      ]
    else:
      for p in candidates:
        rel_str = p.relative_to(repo_root).as_posix()
        # In local working trees, remove untracked ephemeral server caches
        # unless they have been tracked/staged in VCS.
        is_untracked = False
        for cmd in (
            ["git", "ls-files", "--error-unmatch", str(p)],
            ["hg", "files", str(p)],
        ):
          try:
            res = subprocess.run(
                cmd,
                cwd=repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if res.returncode == 0:
              is_untracked = False
              break
            if res.returncode == 1:
              is_untracked = True
              break
          except OSError:
            continue
        if is_untracked:
          try:
            p.unlink()
          except OSError:
            pass
        else:
          polluting_files.append(rel_str)
    self.assertEqual(
        polluting_files,
        [],
        "Server cache files are checked into the demo trace corpus:"
        f" {polluting_files}",
    )

  def test_proto_interfaces_contract(self) -> None:
    """Verifies .d.ts.gz files are valid archives declared in BUILD.oss."""
    repo_root = _find_repo_root()
    self.assertIsNotNone(
        repo_root, "Repository root not found in current execution environment."
    )
    interfaces_dir = repo_root / "frontend" / "app" / "common" / "interfaces"
    self.assertTrue(
        interfaces_dir.is_dir(),
        f"Interfaces directory not present at {interfaces_dir}",
    )

    build_oss = interfaces_dir / "BUILD.oss"
    self.assertTrue(
        build_oss.is_file(), f"Missing BUILD.oss in {interfaces_dir}"
    )
    build_oss_text = build_oss.read_text(encoding="utf-8")

    gz_files = sorted(interfaces_dir.glob("*.d.ts.gz"))
    self.assertTrue(gz_files, f"No .d.ts.gz archives found in {interfaces_dir}")

    missing_declarations = []
    empty_archives = []
    for gz_file in gz_files:
      with gzip.open(gz_file, "rb") as f:
        content = f.read()
        if not content:
          empty_archives.append(gz_file.name)
      if not re.search(
          rf'["\x27/:]{re.escape(gz_file.name)}["\x27]', build_oss_text
      ):
        missing_declarations.append(gz_file.name)

    self.assertEqual(
        empty_archives,
        [],
        f"Compressed interfaces are empty: {empty_archives}",
    )
    self.assertEqual(
        missing_declarations,
        [],
        f"Interfaces are not declared in {build_oss}: {missing_declarations}",
    )


class InvariantsTest(unittest.TestCase):
  """Tests for generic UI invariants."""

  def test_durations_and_percentages_parsing(self) -> None:
    """Verifies scientific notation and negative lookbehinds in invariants."""
    self.assertEqual(invariants.check_durations_non_negative("1.23e-4 s"), [])
    self.assertEqual(invariants.check_durations_non_negative("worker-1s"), [])
    self.assertEqual(invariants.check_durations_non_negative("step_10ms"), [])
    self.assertEqual(
        invariants.check_durations_non_negative("-1.23e-4 s"),
        ["Negative duration -1.23e-4"],
    )
    self.assertEqual(
        invariants.check_durations_non_negative("Elapsed: -5.0ms"),
        ["Negative duration -5.0"],
    )

    self.assertEqual(invariants.check_percentages("10-20%"), [])
    self.assertEqual(invariants.check_percentages("worker-1%"), [])
    self.assertEqual(invariants.check_percentages("Accuracy: 98.5%"), [])
    self.assertEqual(
        invariants.check_percentages("1.5e2 %"),
        ["Percentage 1.5e2% outside [0.0, 100.0]"],
    )
    self.assertEqual(
        invariants.check_percentages("Over: 150%"),
        ["Percentage 150% outside [0.0, 100.0]"],
    )
    self.assertEqual(
        invariants.check_percentages("Under: -5%"),
        ["Percentage -5% outside [0.0, 100.0]"],
    )

  # pylint: disable=protected-access
  def test_diff_header_matching_whole_tokens(self) -> None:
    """Verifies diff header regex ignores words with substrings like vs."""
    self.assertIsNone(invariants._DIFF_HEADER_RE.search("Recvs"))
    self.assertIsNone(invariants._DIFF_HEADER_RE.search("Convs"))
    self.assertIsNone(invariants._DIFF_HEADER_RE.search("Devs"))
    self.assertIsNone(invariants._DIFF_HEADER_RE.search("Exchange"))
    self.assertIsNone(invariants._DIFF_HEADER_RE.search("Device"))

    self.assertIsNotNone(invariants._DIFF_HEADER_RE.search("Diff"))
    self.assertIsNotNone(invariants._DIFF_HEADER_RE.search("Diff (%)"))
    self.assertIsNotNone(invariants._DIFF_HEADER_RE.search("diff_percent"))
    self.assertIsNotNone(invariants._DIFF_HEADER_RE.search("pct_change"))
    self.assertIsNotNone(invariants._DIFF_HEADER_RE.search("Delta"))
    self.assertIsNotNone(
        invariants._DIFF_HEADER_RE.search("Baseline vs Candidate")
    )
    self.assertIsNotNone(
        invariants._DIFF_HEADER_RE.search("Baseline vs. Candidate")
    )
    self.assertIsNotNone(invariants._DIFF_HEADER_RE.search("Latency Change"))
    self.assertIsNotNone(
        invariants._DIFF_HEADER_RE.search("Speedup / Improvement")
    )

  def test_poison_tokens(self) -> None:
    """Verifies poison token detector surfaces invalid/NaN/undefined values."""
    self.assertEqual(invariants.check_poison_tokens("Clean text 123"), [])
    self.assertTrue(bool(invariants.check_poison_tokens("Cost: NaN ms")))
    self.assertTrue(bool(invariants.check_poison_tokens("Status: undefined")))
    self.assertTrue(bool(invariants.check_poison_tokens("[object Object]")))
    self.assertTrue(bool(invariants.check_poison_tokens("Result: null")))
    self.assertTrue(bool(invariants.check_poison_tokens("Error (null)")))
    self.assertTrue(bool(invariants.check_poison_tokens("Value: INVALID")))

  def test_positive_rendered_content_and_dom_invariants(self) -> None:
    """Verifies each DOM invariant detects its failure state."""

    class _FakeEl:
      """Minimal element double for DOM invariant tests."""

      def __init__(self, box=None, text=""):
        self._box = box
        self._text = text

      def bounding_box(self) -> dict[str, int] | None:
        return self._box

      def inner_text(self) -> str:
        return self._text

      def count(self) -> int:
        return 1

      def locator(self, sel: str) -> "_FakeEl":
        del sel
        return self

    class _FakePage:
      """Minimal page double for DOM invariant tests."""

      def __init__(self, charts, cards, tables):
        self._charts = charts
        self._cards = cards
        self._tables = tables

      def locator(self, selector):
        if "svg" in selector or "canvas" in selector:
          return type("_L", (), {"all": lambda s: self._charts})()
        if "card" in selector:
          return type("_L", (), {"all": lambda s: self._cards})()
        if "table" in selector:
          return type("_L", (), {"all": lambda s: self._tables})()
        return type("_L", (), {"all": lambda s: []})()

      def evaluate(self, script: str, arg: object = None) -> list[str]:
        del script, arg
        return []

    healthy_page = _FakePage(
        charts=[_FakeEl(box={"width": 100, "height": 100})],
        cards=[_FakeEl(text="Performance Card")],
        tables=[_FakeEl()],
    )
    self.assertEqual(
        invariants.check_positive_rendered_content(healthy_page), []
    )
    self.assertEqual(invariants.run_dom_invariants(healthy_page), [])

    collapsed_page = _FakePage(
        charts=[_FakeEl(box={"width": 0, "height": 50})],
        cards=[_FakeEl(text="  ")],
        tables=[],
    )
    violations = invariants.check_positive_rendered_content(collapsed_page)
    self.assertEqual(len(violations), 2)
    self.assertIn("collapsed geometry: 0x50", violations[0])
    self.assertIn("unexpectedly empty", violations[1])

    class _BannerAndLockPage:
      """Page double exposing visible error banners and stuck loading locks."""

      def locator(self, selector: str):
        if "snack-bar" in selector:
          return type(
              "_L", (), {"all": lambda s: [_FakeEl(text="Data fetch failed")]}
          )()
        if "hidden-content" in selector:
          return type("_L", (), {"all": lambda s: [_FakeEl()]})()
        return type("_L", (), {"all": lambda s: []})()

    banner_page = _BannerAndLockPage()
    self.assertEqual(
        invariants.check_no_error_banners(banner_page),
        ["Visible error banner[0]: 'Data fetch failed'"],
    )
    self.assertEqual(
        invariants.check_no_stuck_loading_lock(banner_page),
        [
            "Main page router-outlet is locked inside div.hidden-content (0px"
            " height)"
        ],
    )

  def test_visualization_selector_and_component_geometry(self) -> None:
    """Verifies VISUALIZATION_SELECTOR excludes :scope > * and enforces child geometry."""
    self.assertNotIn(":scope > *", ui_helpers.VISUALIZATION_SELECTOR)
    self.assertIn("op-table-entry .row", ui_helpers.VISUALIZATION_SELECTOR)
    self.assertIn("svg", ui_helpers.VISUALIZATION_SELECTOR)
    self.assertIn("iframe", ui_helpers.VISUALIZATION_SELECTOR)

    class _BoxLocator:
      """Locator double returning a fixed bounding box."""

      def __init__(self, box):
        self._box = box

      @property
      def first(self):
        return self

      def bounding_box(self):
        return self._box

    class _GeometryPage:
      """Page double dispatching outer and child visualization locators."""

      def __init__(self, comp_box, child_box):
        self._comp_box = comp_box
        self._child_box = child_box

      def locator(self, selector):
        if ui_helpers.VISUALIZATION_SELECTOR in selector:
          return _BoxLocator(self._child_box)
        return _BoxLocator(self._comp_box)

    with mock.patch.object(
        ui_helpers.sync_api, "expect", create=True
    ):
      ui_helpers.assert_component_geometry(
          _GeometryPage(
              {"width": 800, "height": 600},
              {"width": 400, "height": 300},
          ),
          "input-pipeline",
          "healthy step",
      )
      with self.assertRaisesRegex(
          AssertionError, "child visualization collapsed"
      ):
        ui_helpers.assert_component_geometry(
            _GeometryPage(
                {"width": 800, "height": 600},
                {"width": 0, "height": 300},
            ),
            "input-pipeline",
            "collapsed child step",
        )

      class _BodyLocator:
        """Locator double returning fixed body text."""

        def __init__(self, text: str):
          self._text = text

        def inner_text(self) -> str:
          return self._text

        def count(self) -> int:
          return 0

        def all(self) -> list[object]:
          return []

      class _HealthyPage:
        """Page double for assert_healthy tests."""

        url = "about:blank"
        frames = ()

        def __init__(self, text: str, cells: list[str] | None = None):
          self._text = text
          self._cells = cells or []

        def inner_text(self, selector: str) -> str:
          del selector
          return self._text

        def evaluate(self, script: str, arg: object = None) -> list[str]:
          del arg
          return self._cells if "diffRe" in script else []

        def locator(self, selector: str) -> _BodyLocator:
          del selector
          return _BodyLocator(self._text)

      ui_helpers.assert_healthy(
          _HealthyPage("Overview Page Metrics", ["12.5 ms", "85.0%"]),
          None,
          "ok",
      )
      with self.assertRaisesRegex(AssertionError, "Empty page body rendered"):
        ui_helpers.assert_healthy(_HealthyPage("   "), None, "blank")
      with self.assertRaisesRegex(AssertionError, "Poison tokens detected"):
        ui_helpers.assert_healthy(_HealthyPage("Value: NaN ms"), None, "nan")
      with self.assertRaisesRegex(AssertionError, "Negative duration -4.2"):
        ui_helpers.assert_healthy(
            _HealthyPage("Overview Page Metrics", ["-4.2 ms"]), None, "cell"
        )

  def test_layout_overlap_and_overflow_invariants(self) -> None:
    """Verifies layout invariants pass on clean pages and flag defects."""

    class _EvalPage:
      """Page double returning scripted JS evaluation outputs."""

      def __init__(
          self,
          response: list[str],
          wait_fn: Callable[..., None] | None = None,
      ):
        self._response = response
        self.last_arg: object = None
        if wait_fn is not None:
          self.wait_for_function = wait_fn

      def evaluate(self, script: str, arg: object = None) -> list[str]:
        del script
        self.last_arg = arg
        return self._response

    self.assertEqual(invariants.check_no_sidenav_overlap(_EvalPage([])), [])
    self.assertEqual(
        invariants.check_no_sidenav_overlap(
            _EvalPage([
                "Open side drawer overlaps mat-sidenav-content by 300px"
            ])
        ),
        ["Open side drawer overlaps mat-sidenav-content by 300px"],
    )
    self.assertEqual(
        invariants.check_no_sidenav_overlap(
            _EvalPage(
                ["Open side drawer overlaps mat-sidenav-content by 12px"],
                wait_fn=lambda script, timeout=600: None,
            )
        ),
        [],
    )

    clean_overflow_page = _EvalPage([])
    self.assertEqual(
        invariants.check_no_horizontal_overflow(
            clean_overflow_page, max_overflow_px=12, check_content_pane=True
        ),
        [],
    )
    self.assertEqual(clean_overflow_page.last_arg, [12, True])
    self.assertEqual(
        invariants.check_no_horizontal_overflow(
            _EvalPage(["Document horizontal overflow: 120px > 16px"])
        ),
        ["Document horizontal overflow: 120px > 16px"],
    )

  def test_synthetic_trace_generators_and_staging(self) -> None:
    """Verifies synthesized traces serialize valid XSpace protos."""
    xspace_cls, _ = ui_helpers._build_xspace_parser()
    with tempfile.TemporaryDirectory() as tmp:
      staged = pathlib.Path(tmp)
      ui_helpers.stage_synthesized_runs(staged)
      expected_runs = (
          "tpu-serving",
          "perf-utilization",
          "perf-v4",
          "megascale-1host",
          "megascale-2host",
          "empty-memory",
      )
      for run_name in expected_runs:
        traces = sorted((staged / run_name).glob("*.xplane.pb"))
        self.assertTrue(traces, f"Missing synthesized trace for {run_name}")
        for trace_file in traces:
          self.assertGreater(trace_file.stat().st_size, 0)
          parsed = xspace_cls.FromString(trace_file.read_bytes())
          self.assertGreater(
              len(parsed.planes), 0, f"Empty XSpace in {run_name}"
          )
      empty_mem_space = xspace_cls.FromString(
          (staged / "empty-memory" / "empty-mem-0.xplane.pb").read_bytes()
      )
      self.assertIn(
          "/host:CPU", {plane.name for plane in empty_mem_space.planes}
      )
      self.assertEqual(
          len(list((staged / "megascale-2host").glob("*.xplane.pb"))), 2
      )
      self.assertEqual(
          list((staged / "megascale-2host").glob("*.dcn_collective_stats.pb")),
          [],
      )
      self.assertFalse(
          (staged / "megascale-2host" / "cache_version.txt").exists()
      )
      mega_space = xspace_cls.FromString(
          (staged / "megascale-2host" / "mega-node-0.xplane.pb").read_bytes()
      )
      mega_planes = {plane.name for plane in mega_space.planes}
      self.assertIn("/host:CPU", mega_planes)
      self.assertIn("/host:metadata", mega_planes)

      fake_repo = staged / "fake_repo"
      gpu_src = (
          fake_repo
          / "xprof"
          / "convert"
          / "test_xplanes"
          / "gpu_training_2.xplane.pb"
      )
      ui_helpers.write_empty_memory_trace(gpu_src)
      empty_staged = staged / "empty_staged"
      empty_staged.mkdir()
      ui_helpers.stage_synthesized_runs(empty_staged, repo_root=fake_repo)
      self.assertTrue((empty_staged / "gpu-training").is_dir())
      self.assertFalse((empty_staged / "multi-host").exists())


_LogEntry = tuple[str, Any, tuple[Any, ...], dict[str, Any]]


class _LoggedAssertions:
  """Logs the assertions made on one `expect` target."""

  def __init__(self, log: list[_LogEntry], target: Any):
    self._log = log
    self._target = target

  def __getattr__(self, name: str) -> Callable[..., None]:
    return lambda *args, **kwargs: self._log.append(
        (name, self._target, args, kwargs)
    )


class _LoggingLocator:
  """Locator double that logs the actions taken on it."""

  def __init__(self, log: list[_LogEntry], selector: str):
    self.log = log
    self.selector = selector
    self.first = self

  def locator(self, selector: str) -> "_LoggingLocator":
    return _LoggingLocator(self.log, f"{self.selector} {selector}")

  def filter(self, **kwargs: Any) -> "_LoggingLocator":
    matches = _LoggingLocator(self.log, self.selector)
    self.log.append(("filter", matches, (self,), kwargs))
    return matches

  def get_attribute(self, name: str) -> str:
    del name
    return "mat-drawer-opened"

  def click(self) -> None:
    self.log.append(("click", self, (), {}))

  def fill(self, value: str) -> None:
    self.log.append(("fill", self, (value,), {}))

  def blur(self) -> None:
    self.log.append(("blur", self, (), {}))


class _LoggingPage:
  """Page double whose locators and mouse write to one log."""

  def __init__(self, log: list[_LogEntry]):
    self.log = log
    self.mouse = mock.Mock()
    self.mouse.move.side_effect = lambda *xy: log.append(("move", None, xy, {}))

  def locator(self, selector: str) -> _LoggingLocator:
    return _LoggingLocator(self.log, selector)

  frame_locator = locator


class UiHelpersTest(unittest.TestCase):
  """Tests that the interaction helpers check the page around each action."""

  def setUp(self) -> None:
    super().setUp()
    self.log: list[_LogEntry] = []
    self.page = _LoggingPage(self.log)
    patcher = mock.patch.object(
        ui_helpers.sync_api,
        "expect",
        create=True,
        side_effect=lambda target: _LoggedAssertions(self.log, target),
    )
    patcher.start()
    self.addCleanup(patcher.stop)

  def _calls(self, name: str) -> list[_LogEntry]:
    return [entry for entry in self.log if entry[0] == name]

  def _actions(self) -> list[str]:
    return [entry[0] for entry in self.log if entry[0] != "filter"]

  def test_dropdowns_pick_and_show_whole_option_texts(self) -> None:
    """Verifies dropdown helpers pick whole option texts and recheck them."""
    ui_helpers.select_session(self.page, "tpu-training")
    ui_helpers.select_module(self.page, "jit_train_step")
    ui_helpers.select_memory_id(self.page, "1")
    ui_helpers.switch_tool(self.page, "Op Profile")
    ui_helpers.select_op_profile_group_by(self.page, "Category")

    patterns = [entry[3]["has_text"] for entry in self._calls("filter")]
    session, module, memory_id, tool, _ = patterns
    self.assertRegex(" tpu-training ", session)
    self.assertNotRegex("tpu-training-2", session)
    self.assertRegex(" jit_train_step(4869159985936022652) ", module)
    self.assertNotRegex("jit_train_step_2", module)
    self.assertNotRegex("prefix_jit_train_step", module)
    self.assertRegex(" 1 ", memory_id)
    self.assertNotRegex("10", memory_id)
    self.assertRegex("HLO Op Profile", tool)
    self.assertRegex("Op Profile", tool)
    shown = [entry[2] for entry in self._calls("to_have_text")]
    self.assertEqual(shown, [(pattern,) for pattern in patterns])

  def test_dropdown_ignores_a_failed_mouse_reset(self) -> None:
    """Verifies a mouse reset error after the dropdown closes is ignored."""
    self.page.mouse.move.side_effect = invariants.PlaywrightError("detached")
    ui_helpers.select_category_filter(self.page, "convolution fusion")

    self.assertEqual(
        self._actions(),
        [
            "to_be_visible",
            "click",
            "to_be_visible",
            "click",
            "to_have_count",
            "to_have_text",
        ],
    )
    (closed,) = self._calls("to_have_count")
    self.assertEqual((closed[1].selector, closed[2]), ("mat-option", (0,)))
    self.page.mouse.move.assert_called_once_with(0, 0)

  def test_expand_op_profile_row_checks_the_rendered_triangle(self) -> None:
    """Verifies a row counts as expanded only by its rendered triangle."""
    ui_helpers.expand_op_profile_row(self.page, "convolution fusion")

    self.assertEqual(
        self._actions(),
        ["to_be_visible", "click", "move", "to_have_text", "to_contain_text"],
    )
    (triangle,) = self._calls("to_have_text")
    self.assertEqual(triangle[2], ("\u25bc",))
    self.assertTrue(triangle[3]["use_inner_text"])
    (details,) = self._calls("to_contain_text")
    self.assertEqual(details[2], ("convolution fusion",))

  def test_plot_graph_node_waits_for_the_url_and_the_graph(self) -> None:
    """Verifies plotting waits for the node in the URL and a drawn node."""
    ui_helpers.plot_graph_node(self.page, "fusion.")

    self.assertEqual(
        self._actions(),
        ["to_be_visible", "click", "to_have_url", "to_be_visible", "move"],
    )
    (url_check,) = self._calls("to_have_url")
    url = url_check[2][0]
    self.assertRegex("/?run=r&node_name=fusion.12&module_name=m", url)
    self.assertNotRegex("/?node_name=fusion.12&module_name=m", url)
    self.assertNotRegex("/?run=r&node_name=add.3&module_name=fusion.12", url)

  def test_filter_table_rows_must_hide_rows_and_keep_matches(self) -> None:
    """Verifies the filter must hide shown rows and keep a matching one."""
    ui_helpers.filter_table_rows(self.page, "all-gather")

    self.assertEqual(
        self._actions(),
        [
            "to_be_visible",
            "fill",
            "blur",
            "to_have_count",
            "to_be_visible",
            "move",
        ],
    )
    (filtering,) = self._calls("filter")
    others, rows = filtering[1], filtering[2][0]
    self.assertRegex("%ALL-GATHER.3", filtering[3]["has_not_text"])
    before, after = self._calls("to_be_visible")
    self.assertIs(before[1], others)
    self.assertIs(after[1], rows)
    (hidden,) = self._calls("to_have_count")
    self.assertIs(hidden[1], others)
    self.assertEqual(hidden[2], (0,))
    (fill,) = self._calls("fill")
    self.assertEqual(fill[2], ("all-gather",))

  def test_sort_table_column_needs_an_unsorted_header(self) -> None:
    """Verifies sorting starts from an unsorted header and ends sorted."""
    ui_helpers.sort_table_column(self.page, "#Occurrences")

    self.assertEqual(
        self._actions(),
        [
            "to_be_visible",
            "not_to_have_class",
            "click",
            "to_have_class",
            "move",
        ],
    )
    (before,) = self._calls("not_to_have_class")
    (after,) = self._calls("to_have_class")
    self.assertEqual(before[2], after[2])
    sorted_class = after[2][0]
    self.assertRegex("header-cell sort-ascending", sorted_class)
    self.assertRegex("header-cell sort-descending", sorted_class)
    self.assertNotRegex("header-cell", sorted_class)

  def test_resolve_server_logdir_strips_plugins_profile_suffix(self) -> None:
    """Verifies server logdir resolves to the parent of plugins/profile."""
    self.assertEqual(
        ui_helpers.resolve_server_logdir("/tmp/xprof/demo/plugins/profile"),
        "/tmp/xprof/demo",
    )
    self.assertEqual(
        ui_helpers.resolve_server_logdir("/tmp/custom_runs"),
        "/tmp/custom_runs",
    )
    self.assertEqual(
        ui_helpers.resolve_server_logdir(
            "/tmp/xprof/demo/plugins/profile/tpu_training"
        ),
        "/tmp/xprof/demo/plugins/profile/tpu_training",
    )

  def test_resolve_artifacts_dir_falls_back_to_kokoro_artifacts_dir(
      self,
  ) -> None:
    """Verifies failure artifact dir resolution across CI and test env vars."""
    self.assertIsNone(ui_helpers.resolve_artifacts_dir({}))
    self.assertEqual(
        ui_helpers.resolve_artifacts_dir(
            {"KOKORO_ARTIFACTS_DIR": "/kokoro/out"}
        ),
        "/kokoro/out/ui_failure_artifacts",
    )
    self.assertEqual(
        ui_helpers.resolve_artifacts_dir({
            "TEST_UNDECLARED_OUTPUTS_DIR": "/ci/undeclared",
            "KOKORO_ARTIFACTS_DIR": "/kokoro/out",
        }),
        "/ci/undeclared",
    )
    self.assertEqual(
        ui_helpers.resolve_artifacts_dir({
            "XPROF_UI_ARTIFACTS_DIR": "/explicit/ui_artifacts",
            "TEST_UNDECLARED_OUTPUTS_DIR": "/ci/undeclared",
            "KOKORO_ARTIFACTS_DIR": "/kokoro/out",
        }),
        "/explicit/ui_artifacts",
    )


class HarnessInvariantsAndConftestTest(unittest.TestCase):
  """Hermetic unit tests for readiness polling, invariants, and conftest."""

  def _mock_http_response(
      self,
      status: int,
      body: bytes,
      headers: dict[str, str] | None = None,
  ) -> mock.MagicMock:
    """Returns a context-manager mock for `urllib.request.urlopen`."""
    resp = mock.MagicMock()
    resp.status = status
    resp.read.return_value = body
    msg = email.message.Message()
    for key, value in (headers or {}).items():
      msg[key] = value
    resp.headers = msg
    cm = mock.MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    return cm

  def test_is_server_ready_edge_cases(self) -> None:
    """Tests readiness polling across HTTP statuses, gzip, and errors."""
    with mock.patch.object(
        urllib.request,
        "urlopen",
        return_value=self._mock_http_response(200, b"ok"),
    ):
      self.assertTrue(ui_helpers.is_server_ready("http://127.0.0.1:9999"))

    http_404 = urllib.error.HTTPError(
        "http://127.0.0.1:9999", 404, "Not Found", email.message.Message(), None
    )
    self.addCleanup(http_404.close)
    with mock.patch.object(urllib.request, "urlopen", side_effect=http_404):
      self.assertTrue(ui_helpers.is_server_ready("http://127.0.0.1:9999"))
      self.assertFalse(
          ui_helpers.is_server_ready(
              "http://127.0.0.1:9999", expected_runs={"run_a"}
          )
      )

    http_500 = urllib.error.HTTPError(
        "http://127.0.0.1:9999",
        500,
        "Server Error",
        email.message.Message(),
        None,
    )
    self.addCleanup(http_500.close)
    with mock.patch.object(urllib.request, "urlopen", side_effect=http_500):
      self.assertFalse(ui_helpers.is_server_ready("http://127.0.0.1:9999"))

    gz_runs = gzip.compress(b'["run_a", "run_b"]')
    with mock.patch.object(
        urllib.request,
        "urlopen",
        return_value=self._mock_http_response(
            200, gz_runs, {"Content-Encoding": "gzip"}
        ),
    ) as mock_open:
      self.assertTrue(
          ui_helpers.is_server_ready(
              "http://127.0.0.1:9999/",
              expected_runs={"run_a"},
              profile_logdir="/tmp/flat",
          )
      )
      called_url = mock_open.call_args.args[0]
      self.assertIn("run_path=%2Ftmp%2Fflat", called_url)

    for bad_side_effect in (
        self._mock_http_response(202, b'["run_a"]'),
        self._mock_http_response(200, b"\x1f\x8bnot-valid-gzip"),
        self._mock_http_response(200, b"{not-json"),
        urllib.error.URLError("connection refused"),
    ):
      with mock.patch.object(
          urllib.request, "urlopen", side_effect=[bad_side_effect]
      ):
        self.assertFalse(
            ui_helpers.is_server_ready(
                "http://127.0.0.1:9999", expected_runs={"run_a"}
            )
        )

  def test_write_megascale_trace_synthesizes_tpu_plane_and_inherits_timestamp(
      self,
  ) -> None:
    """Verifies MegaScale writer synthesizes TPU plane and keeps timestamps."""
    # pylint: disable=protected-access
    xspace_cls, _ = ui_helpers._build_xspace_parser()
    # pylint: enable=protected-access
    base_space = xspace_cls()
    cpu = base_space.planes.add(id=1, name="/host:CPU")
    cpu.lines.add(id=1, name="Main", timestamp_ns=42000)

    with tempfile.TemporaryDirectory() as tmp:
      tmp_dir = pathlib.Path(tmp)
      base_path = tmp_dir / "base.xplane.pb"
      base_path.write_bytes(base_space.SerializeToString())
      out_path = tmp_dir / "mega.xplane.pb"

      ui_helpers.write_megascale_trace(out_path, base_path)
      result = xspace_cls.FromString(out_path.read_bytes())

    planes = {plane.name: plane for plane in result.planes}
    self.assertIn("/host:CPU", planes)
    self.assertIn("/device:TPU:0", planes)
    self.assertIn("/host:metadata", planes)
    cpu_mega = next(
        line for line in planes["/host:CPU"].lines if line.name == "MegaScale"
    )
    self.assertEqual(cpu_mega.timestamp_ns, 42000)
    tpu_lines = {line.name for line in planes["/device:TPU:0"].lines}
    self.assertEqual(tpu_lines, {"MegaScale", "XLA Modules", "XLA Ops"})

  def test_invariants_playwright_error_handling(self) -> None:
    """Verifies navigation errors are ignored while target crashes raise."""
    for msg in (
        "Execution context was destroyed, most likely because of a navigation",
        "Frame was detached",
        "Navigating frame was detached",
    ):
      nav_page = mock.MagicMock()
      nav_page.evaluate.side_effect = invariants.PlaywrightError(msg)
      self.assertEqual(invariants.check_no_sidenav_overlap(nav_page), [])
      self.assertEqual(invariants.check_no_horizontal_overflow(nav_page), [])

    retrying_page = mock.MagicMock()
    retrying_page.evaluate.return_value = ["overlap"]
    retrying_page.wait_for_function.return_value = None
    self.assertEqual(invariants.check_no_sidenav_overlap(retrying_page), [])

    for err_msg in (
        "Target page, context or browser has been closed",
        "Target closed",
    ):
      crash_page = mock.MagicMock()
      crash_page.evaluate.side_effect = invariants.PlaywrightError(err_msg)
      with self.assertRaises(invariants.PlaywrightError):
        invariants.check_no_sidenav_overlap(crash_page)
      with self.assertRaises(invariants.PlaywrightError):
        invariants.check_no_horizontal_overflow(crash_page)

  def _load_isolated_ui_module(
      self,
      filename: str,
      conftest_mod: types.ModuleType | None = None,
  ) -> types.ModuleType:
    """Loads a sibling Playwright UI module hermetically with stubbed deps."""
    module_path = pathlib.Path(__file__).with_name(filename)
    fake_playwright: Any = types.ModuleType("playwright")
    fake_sync_api: Any = types.ModuleType("playwright.sync_api")

    class _StubPage:
      pass

    class _StubCallInfo:

      def __class_getitem__(cls, _: Any) -> type["_StubCallInfo"]:
        return cls

    class _StubPlaywrightError(Exception):
      pass

    fake_sync_api.Error = _StubPlaywrightError
    fake_sync_api.Page = _StubPage
    fake_sync_api.Route = object
    fake_sync_api.expect = mock.MagicMock()
    fake_playwright.sync_api = fake_sync_api

    fake_pytest: Any = types.ModuleType("pytest")
    fake_pytest.fixture = lambda *args, **kwargs: (lambda fn: fn)
    fake_pytest.hookimpl = lambda *args, **kwargs: (lambda fn: fn)
    fake_pytest.mark = types.SimpleNamespace(
        parametrize=lambda *args, **kwargs: (lambda fn: fn)
    )
    fake_pytest.skip = mock.MagicMock()
    fake_pytest.Item = object
    fake_pytest.CallInfo = _StubCallInfo
    fake_pytest.FixtureRequest = object

    patched_modules: dict[str, types.ModuleType] = {
        "playwright": fake_playwright,
        "playwright.sync_api": fake_sync_api,
        "pytest": fake_pytest,
    }
    if conftest_mod is not None:
      fake_tests_pkg: Any = types.ModuleType("tests")
      fake_tests_pkg.__path__ = []
      fake_tests_ui_pkg: Any = types.ModuleType("tests.ui")
      fake_tests_ui_pkg.__path__ = [str(module_path.parent)]
      fake_tests_ui_pkg.conftest = conftest_mod
      fake_tests_ui_pkg.ui_helpers = ui_helpers
      fake_tests_pkg.ui = fake_tests_ui_pkg
      patched_modules.update({
          "tests": fake_tests_pkg,
          "tests.ui": fake_tests_ui_pkg,
          "tests.ui.conftest": conftest_mod,
          "tests.ui.ui_helpers": ui_helpers,
      })

    spec = importlib.util.spec_from_file_location(
        f"_xprof_isolated_{module_path.stem}", module_path
    )
    if spec is None or spec.loader is None:
      self.fail(f"Could not create ModuleSpec for {module_path}")
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, patched_modules):
      spec.loader.exec_module(mod)
    return mod

  def _load_isolated_conftest(self) -> types.ModuleType:
    """Loads sibling `conftest.py` hermetically with stubbed pytest."""
    return self._load_isolated_ui_module("conftest.py")

  def test_conftest_browser_errors_artifacts_and_run_resolution(self) -> None:
    """Tests BrowserErrors, failure artifact dumps, and run resolution."""
    conftest_mod = self._load_isolated_conftest()

    tracker = conftest_mod.BrowserErrors(
        page_errors=["[UNHANDLED] boom"],
        http_errors=["[HTTP 500] GET /data/plugin/profile/data?run=r1"],
    )
    with self.assertRaisesRegex(AssertionError, r"HTTP 500"):
      tracker.assert_clean("tool_render")
    tracker.ignore("HTTP 500", "boom")
    tracker.assert_clean("ignored_case")

    with tempfile.TemporaryDirectory() as tmp:
      tmp_dir = pathlib.Path(tmp)
      art_dir = tmp_dir / "artifacts"
      page = mock.MagicMock()
      page.content.return_value = "<html><body>failed</body></html>"
      item = types.SimpleNamespace(
          nodeid="tests/ui/test_tools.py::test_case[param]",
          funcargs={"page": page, "browser_errors": tracker},
      )
      outcome = types.SimpleNamespace(
          get_result=lambda: types.SimpleNamespace(failed=True, when="call")
      )
      with (
          mock.patch.dict(os.environ, {"XPROF_UI_ARTIFACTS_DIR": str(art_dir)}),
          mock.patch.object(sys, "stderr", new_callable=io.StringIO),
      ):
        for _ in range(2):
          hook_gen = conftest_mod.pytest_runtest_makereport(item, None)
          next(hook_gen)
          with self.assertRaises(StopIteration):
            hook_gen.send(outcome)
      page.screenshot.assert_called_once()
      self.assertEqual(len(list(art_dir.glob("*.html"))), 1)
      console_files = list(art_dir.glob("*.console.log"))
      self.assertEqual(len(console_files), 1)
      self.assertIn("HTTP 500", console_files[0].read_text(encoding="utf-8"))

      src_dir = tmp_dir / "src_runs"
      trace_file = src_dir / "demo_run" / "host.xplane.pb"
      ui_helpers.write_empty_memory_trace(trace_file)

      synth_gen = conftest_mod.synthesized_logdir(str(src_dir))
      self.addCleanup(synth_gen.close)
      staged_dir = pathlib.Path(next(synth_gen))
      linked = staged_dir / "demo_run" / "host.xplane.pb"
      self.assertTrue(linked.is_symlink())
      self.assertEqual(linked.resolve(), trace_file.resolve())
      self.assertTrue((staged_dir / "multi-host").is_dir())
      resolver = conftest_mod.resolve_run_dir(
          str(src_dir), str(staged_dir), lambda name: f"resolved_{name}"
      )
      self.assertEqual(
          resolver("perf-v4"), ("perf-v4", str(staged_dir / "perf-v4"))
      )
      self.assertEqual(
          resolver("demo_run"),
          ("resolved_demo_run", str(src_dir / "resolved_demo_run")),
      )
      synth_gen.close()

      env_repo = tmp_dir / "custom_repo_root"
      env_repo.mkdir()
      rel_env_repo = os.path.relpath(env_repo, pathlib.Path.cwd())
      with mock.patch.dict(os.environ, {"XPROF_REPO_ROOT": rel_env_repo}):
        # pylint: disable=protected-access
        self.assertEqual(conftest_mod._find_repo_root(), env_repo.resolve())
        # pylint: enable=protected-access

      missing_repo = tmp_dir / "missing_repo_root"
      with mock.patch.dict(os.environ, {"XPROF_REPO_ROOT": str(missing_repo)}):
        # pylint: disable=protected-access
        self.assertNotEqual(
            conftest_mod._find_repo_root(), missing_repo.resolve()
        )
        # pylint: enable=protected-access

      fallback_root = tmp_dir / "fallback_srcdir"
      fallback_demo = fallback_root / "xprof" / "demo" / "plugins" / "profile"
      fallback_demo.mkdir(parents=True)
      with (
          mock.patch.object(
              conftest_mod, "__file__", str(tmp_dir / "conftest.py")
          ),
          mock.patch.dict(
              os.environ,
              {
                  "XPROF_LOGDIR": "",
                  "XPROF_REPO_ROOT": str(env_repo),
                  "TEST_SRCDIR": str(fallback_root),
              },
          ),
      ):
        # pylint: disable=protected-access
        self.assertEqual(
            conftest_mod._source_logdir(), str(fallback_demo.resolve())
        )
        # pylint: enable=protected-access

  def test_ui_suite_op_profile_visibility_and_run_tools_route_matching(
      self,
  ) -> None:
    """Verifies Op Profile visible row selector and run_tools slash matching."""
    conftest_mod = self._load_isolated_conftest()
    sweep_mod = self._load_isolated_ui_module(
        "test_tool_sweep.py", conftest_mod
    )
    self.assertEqual(
        sweep_mod.TOOL_CONTENT_SELECTORS["op_profile"],
        "op-table-entry .row:visible",
    )

    edge_mod = self._load_isolated_ui_module(
        "test_data_consistency_and_edge_cases.py", conftest_mod
    )
    # pylint: disable=protected-access
    route_re = edge_mod._RUN_TOOLS_ROUTE_RE
    enable_fn = edge_mod._enable_inference_profile_tool
    # pylint: enable=protected-access
    slash_query_url = (
        "http://127.0.0.1:45983/data/plugin/profile/run_tools"
        "?session_path=/tmp/xprof_logdir/demo/plugins/profile/tpu-serving"
        "&run=tpu-serving"
    )
    self.assertIsNotNone(route_re.search(slash_query_url))
    self.assertIsNotNone(
        route_re.search("http://127.0.0.1:45983/data/plugin/profile/run_tools")
    )
    self.assertIsNotNone(
        route_re.search(
            "http://127.0.0.1:45983/tb/data/plugin/profile/run_tools#frag"
        )
    )
    self.assertIsNone(
        route_re.search(
            "http://127.0.0.1:45983/data/plugin/profile/run_tools_extra"
        )
    )
    self.assertIsNone(
        route_re.search(
            "http://127.0.0.1:45983/data/plugin/profile/hosts"
            "?session_path=/tmp/data/plugin/profile/run_tools"
        )
    )

    fake_route = mock.MagicMock()
    fake_route.fetch.return_value = types.SimpleNamespace(
        status=200, json=lambda: ["overview_page", "op_profile"]
    )
    enable_fn(fake_route)
    fake_route.fulfill.assert_called_once_with(
        status=200,
        content_type="application/json",
        body=json.dumps(["overview_page", "op_profile", "inference_profile"]),
    )

    idempotent_route = mock.MagicMock()
    idempotent_route.fetch.return_value = types.SimpleNamespace(
        status=200, json=lambda: ["overview_page", "inference_profile"]
    )
    enable_fn(idempotent_route)
    idempotent_route.fulfill.assert_called_once_with(
        status=200,
        content_type="application/json",
        body=json.dumps(["overview_page", "inference_profile"]),
    )

    error_route = mock.MagicMock()
    error_route.fetch.return_value = types.SimpleNamespace(
        status=503, json=lambda: {"error": "unavailable"}
    )
    enable_fn(error_route)
    error_route.fulfill.assert_called_once_with(
        status=503,
        content_type="application/json",
        body=json.dumps({"error": "unavailable"}),
    )

    fake_page = mock.MagicMock()
    open_tool = mock.MagicMock(side_effect=RuntimeError("boom"))
    with self.assertRaisesRegex(RuntimeError, "boom"):
      edge_mod.test_inference_profile_renders_serving_tables_and_metrics(
          fake_page, open_tool, conftest_mod.BrowserErrors()
      )
    fake_page.route.assert_called_once_with(route_re, enable_fn)
    fake_page.unroute.assert_called_once_with(route_re)


if __name__ == "__main__":
  if absltest is not None:
    absltest.main()
  else:
    unittest.main()
