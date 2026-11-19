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
  absltest: Any = None

if __package__ in (None, ""):
  # Appended so an installed xprof wheel wins over the source tree's package.
  sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

try:
  from xprof import profile_plugin
  from xprof import profile_plugin_loader
  from xprof import server
  from xprof.standalone import base_plugin
  from xprof.standalone import plugin_event_multiplexer
except ImportError:
  profile_plugin = None
  profile_plugin_loader = None
  server = None
  base_plugin = None
  plugin_event_multiplexer = None
try:
  from tests.ui import invariants
  from tests.ui import journey_capture
  from tests.ui import sxs_diff_engine
  from tests.ui import ui_helpers
except ImportError:
  import invariants
  import journey_capture
  import sxs_diff_engine
  import ui_helpers
# pylint: enable=g-import-not-at-top,g-importing-member


def _find_repo_root() -> pathlib.Path | None:
  """Locates the repository root, or returns None if unavailable."""
  if env_root := os.environ.get("XPROF_REPO_ROOT"):
    candidate = pathlib.Path(env_root).resolve()
    if (candidate / "demo").is_dir():
      return candidate
  if test_srcdir := os.environ.get("TEST_SRCDIR"):
    test_ws = os.environ.get("TEST_WORKSPACE", "")
    base = pathlib.Path(test_srcdir)
    roots = [base / test_ws / "third_party/xprof", base / test_ws]
    for candidate in roots:
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
          "xprof server modules did not import under either the source-tree or"
          " the installed-package module path."
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
      dp: Any = (
          plugin_event_multiplexer.DataProvider("")
          if plugin_event_multiplexer is not None
          else None
      )
      context = base_plugin.TBContext(
          logdir="",
          data_provider=dp,
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

  def test_dotted_run_spa_routes_return_200(self) -> None:
    """Verifies parameterized SPA routes with dots in run names serve index."""
    self.assertIsNotNone(
        self.server_url,
        f"XProf server URL must be available for testing: {self._setup_error}",
    )
    base = self.server_url.rstrip("/")
    for route in (
        "/data/plugin/profile/kernel_viewer/run_v1.0",
        "/data/plugin/profile/megascale_perfetto/run_v1.0",
        "/data/plugin/profile/roofline_model/train/run_v1.0",
        "/data/plugin/profile/roofline_model/.",
        "/data/plugin/profile/stack_trace_page/run_v1.0",
        "/stack_trace_page/run_v1.0",
    ):
      req = urllib.request.Request(f"{base}{route}", method="GET")
      with urllib.request.urlopen(req, timeout=5.0) as resp:
        self.assertEqual(resp.status, 200, f"Unexpected status for {route}")

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
    cache_suffixes = (
        ".cached_tools.json",
        ".SSTABLE",
        ".hlo_proto.pb",
        ".op_stats.pb",
    )
    polluting_files = [
        p.relative_to(repo_root).as_posix()
        for p in staged_files
        if p.name == "cache_version.txt" or p.name.endswith(cache_suffixes)
    ]
    self.assertEqual(
        polluting_files,
        [],
        "Server cache files are checked into the demo trace corpus:"
        f" {polluting_files}",
    )

  def test_proto_interfaces_contract(self) -> None:
    """Verifies .d.ts.gz files are valid archives declared in BUILD(.oss)."""
    repo_root = _find_repo_root()
    self.assertIsNotNone(
        repo_root, "Repository root not found in current execution environment."
    )
    interfaces_dir = repo_root / "frontend" / "app" / "common" / "interfaces"
    self.assertTrue(
        interfaces_dir.is_dir(),
        f"Interfaces directory not present at {interfaces_dir}",
    )

    # Copybara renames BUILD.oss to BUILD in OSS checkouts.
    build_file = interfaces_dir / "BUILD.oss"
    if not build_file.is_file():
      build_file = interfaces_dir / "BUILD"
    self.assertTrue(
        build_file.is_file(),
        f"Missing BUILD.oss or BUILD in {interfaces_dir}",
    )
    build_text = build_file.read_text(encoding="utf-8")

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
          rf'["\x27/:]{re.escape(gz_file.name)}["\x27]', build_text
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
        f"Interfaces are not declared in {build_file}: {missing_declarations}",
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

      def evaluate(self, script: str) -> str:
        del script
        return "visible"

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
    """Verifies VISUALIZATION_SELECTOR skips wrappers and enforces geometry."""
    self.assertNotIn(":scope > *", ui_helpers.VISUALIZATION_SELECTOR)
    self.assertIn("op-table-entry .row", ui_helpers.VISUALIZATION_SELECTOR)
    self.assertIn("svg", ui_helpers.VISUALIZATION_SELECTOR)
    self.assertIn("iframe", ui_helpers.VISUALIZATION_SELECTOR)

    class _BoxLocator:
      """Locator double returning scripted bounding boxes or errors."""

      def __init__(self, box_or_seq):
        if isinstance(box_or_seq, list):
          self._seq = iter(box_or_seq)
        else:
          self._seq = None
        self._box = box_or_seq

      @property
      def first(self):
        return self

      def bounding_box(self):
        if self._seq is None:
          return self._box
        val = next(self._seq, None)
        if isinstance(val, Exception):
          raise val
        return val

    class _GeometryPage:
      """Page double dispatching outer and child visualization locators."""

      def __init__(self, comp_box, child_box):
        self._comp_loc = _BoxLocator(comp_box)
        self._child_loc = _BoxLocator(child_box)

      def locator(self, selector):
        if ui_helpers.VISUALIZATION_SELECTOR in selector:
          return self._child_loc
        return self._comp_loc

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
      # Transient None / detach during Angular DOM swap recovers on retry.
      ui_helpers.assert_component_geometry(
          _GeometryPage(
              {"width": 800, "height": 600},
              [
                  None,
                  invariants.PlaywrightError("detached"),
                  {"width": 120, "height": 32},
              ],
          ),
          "graph-viewer mat-chip-option",
          "re-render recovery",
      )
      with self.assertRaisesRegex(
          AssertionError, "child visualization collapsed"
      ):
        ui_helpers.assert_component_geometry(
            _GeometryPage(
                {"width": 800, "height": 600},
                [None, None, None],
            ),
            "graph-viewer mat-chip-option",
            "persistent None",
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

    # Verify assert_page_invariants runs DOM invariants (kills mutant dropping
    # run_dom_invariants) and forwards check_content_pane=True + allowlist.
    recorded_args: list[object] = []

    class _DomCheckPage:
      """Page double capturing overflow evaluate args and returning locks."""

      url = "http://localhost:8791/overview_page"
      frames = ()

      def __init__(self, *, stuck_lock: bool = False):
        self._stuck_lock = stuck_lock

      def inner_text(self, selector: str) -> str:
        del selector
        return "Overview Page Metrics"

      def locator(self, selector: str) -> Any:
        items = (
            [object()]
            if (self._stuck_lock and "hidden-content" in selector)
            else []
        )
        return type("_L", (), {"all": lambda self, i=items: i})()

      def evaluate(self, script: str, arg: object = None) -> list[str]:
        if "checkContentPane" in script:
          recorded_args.append(arg)
        return []

    invariants.assert_page_invariants(
        _DomCheckPage(), allowed_overflow_selectors=("custom-row .row",)
    )
    self.assertEqual(len(recorded_args), 1)
    self.assertEqual(
        recorded_args[0],
        [
            16,
            True,
            "table, mat-table, .google-visualization-table, custom-row .row",
        ],
    )
    with self.assertRaisesRegex(AssertionError, "div.hidden-content"):
      invariants.assert_page_invariants(_DomCheckPage(stuck_lock=True))

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
        self._wait_fn = wait_fn
        self.last_arg: object = None
        self.wait_arg: object = None

      def wait_for_function(
          self, script: str, *, arg: object = None, timeout: int = 600
      ) -> None:
        self.wait_arg = arg
        if self._wait_fn is None:
          raise invariants.PlaywrightError("Timeout waiting for function")
        self._wait_fn(script, timeout=timeout)

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
    self.assertEqual(
        clean_overflow_page.last_arg,
        [12, True, "table, mat-table, .google-visualization-table"],
    )
    self.assertEqual(
        invariants.check_no_horizontal_overflow(
            _EvalPage(["Document horizontal overflow: 120px > 16px"])
        ),
        ["Document horizontal overflow: 120px > 16px"],
    )
    settling_overflow_page = _EvalPage(
        ["Document horizontal overflow: 24px > 16px"],
        wait_fn=lambda script, timeout=600: None,
    )
    self.assertEqual(
        invariants.check_no_horizontal_overflow(settling_overflow_page), []
    )
    # The settle wait must re-run the check with the same arguments.
    self.assertEqual(
        settling_overflow_page.wait_arg,
        [16, True, "table, mat-table, .google-visualization-table"],
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
    """Verifies dropdown helpers pick whole option texts and verify URLs."""
    ui_helpers.select_session(self.page, "tpu-training")
    ui_helpers.select_module(self.page, "jit_train_step")
    ui_helpers.select_memory_id(self.page, "1")
    ui_helpers.switch_tool(self.page, "Op Profile")
    ui_helpers.select_op_profile_group_by(self.page, "Category")
    ui_helpers.select_host(self.page, "host/0")

    patterns = [entry[3]["has_text"] for entry in self._calls("filter")]
    session, module, memory_id, tool, _, host = patterns
    self.assertRegex(" tpu-training ", session)
    self.assertNotRegex("tpu-training-2", session)
    self.assertRegex(" jit_train_step(4869159985936022652) ", module)
    self.assertNotRegex("jit_train_step_2", module)
    self.assertNotRegex("prefix_jit_train_step", module)
    self.assertRegex(" 1 ", memory_id)
    self.assertNotRegex("10", memory_id)
    self.assertRegex("HLO Op Profile", tool)
    self.assertRegex("Op Profile", tool)
    self.assertRegex(" host/0 ", host)
    shown = [entry[2] for entry in self._calls("to_have_text")]
    self.assertEqual(shown, [(pattern,) for pattern in patterns])

    url_checks = [entry[2][0] for entry in self._calls("to_have_url")]
    self.assertEqual(len(url_checks), 4)
    session_re, module_re, tool_re, host_re = url_checks
    self.assertRegex("?run=tpu-training&tag=overview_page", session_re)
    self.assertNotRegex("?run=tpu-training-2", session_re)
    self.assertRegex("?moduleName=jit_train_step(123)&run=r", module_re)
    self.assertRegex("?moduleName=jit_train_step%28123%29", module_re)
    self.assertNotRegex("?moduleName=jit_train_step_other", module_re)
    self.assertRegex(
        "http://localhost:8791/op_profile?run=r&tag=op_profile", tool_re
    )
    self.assertNotRegex(
        "http://localhost:8791/overview_page?run=r&tag=op_profile", tool_re
    )
    self.assertRegex("?host=host/0&run=r", host_re)
    self.assertRegex("?host=host%2F0", host_re)
    self.assertNotRegex("?host=host/00", host_re)

  def test_select_hosts_toggles_mismatched_options_and_checks_url(
      self,
  ) -> None:
    """Verifies select_hosts toggles only mismatched options and checks URL."""
    clicked: list[str] = []

    class _FakeOption:
      """Fake mat-option supporting inner_text, aria-selected, and click."""

      def __init__(self, label: str, selected: bool):
        self._label = label
        self._selected = selected

      def inner_text(self) -> str:
        return f" {self._label} "

      def get_attribute(self, name: str) -> str:
        return "true" if name == "aria-selected" and self._selected else "false"

      def click(self) -> None:
        clicked.append(self._label)

    options = [
        _FakeOption("worker-0", selected=True),
        _FakeOption("worker-1", selected=True),
        _FakeOption("worker-2", selected=False),
        _FakeOption("worker/3", selected=False),
    ]

    class _MultiHostPage(_LoggingPage):
      """Page double returning scripted mat-option elements."""

      def locator(self, selector: str) -> _LoggingLocator:
        loc = super().locator(selector)
        if selector == "mat-option":
          loc.all = lambda: options  # type: ignore[method-assign]
        return loc

    page = _MultiHostPage(self.log)
    ui_helpers.select_hosts(page, "worker-0", "worker-2", "worker/3")

    self.assertEqual(clicked, ["worker-1", "worker-2", "worker/3"])
    visible_selectors = [
        entry[1].selector for entry in self._calls("to_be_visible")
    ]
    self.assertIn("mat-option", visible_selectors)
    clicks = [entry[1].selector for entry in self._calls("click")]
    self.assertIn(
        "sidenav .item-container:has-text('Hosts') mat-select[multiple]",
        clicks,
    )
    self.assertIn("button:has-text('Apply Host Selection')", clicks)
    (count_check,) = self._calls("to_have_count")
    self.assertEqual(
        (count_check[1].selector, count_check[2]), ("mat-option", (0,))
    )
    (contain_check,) = self._calls("to_contain_text")
    self.assertEqual(contain_check[2], ("Hosts (3)",))
    url_patterns = [entry[2][0] for entry in self._calls("to_have_url")]
    self.assertEqual(len(url_patterns), 3)
    for pat in url_patterns[:2]:
      self.assertRegex("?hosts=worker-0,worker-2&run=r", pat)
      self.assertRegex("?hosts=worker-0%2Cworker-2", pat)
    self.assertNotRegex("?hosts=worker-00,worker-2", url_patterns[0])
    self.assertNotRegex("?hosts=worker-0,worker-20", url_patterns[1])
    self.assertRegex("?hosts=worker-0,worker/3&run=r", url_patterns[2])
    self.assertRegex("?hosts=worker-0%2Cworker%2F3", url_patterns[2])

    with self.assertRaisesRegex(AssertionError, "at least one host"):
      ui_helpers.select_hosts(page)
    with self.assertRaisesRegex(AssertionError, "not found in dropdown"):
      ui_helpers.select_hosts(page, "worker-0", "worker-missing")

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
        str(pathlib.Path("/tmp/xprof/demo")),
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
        os.path.join("/kokoro/out", "ui_failure_artifacts"),
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

    def _passthrough(*args: Any, **kwargs: Any) -> Any:
      if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
      return lambda fn: fn

    fake_sync_api.Browser = object
    fake_sync_api.Error = _StubPlaywrightError
    fake_sync_api.Page = _StubPage
    fake_sync_api.Route = object
    fake_sync_api.expect = mock.MagicMock()
    fake_playwright.sync_api = fake_sync_api

    fake_pytest: Any = types.ModuleType("pytest")
    fake_pytest.fixture = _passthrough
    fake_pytest.hookimpl = _passthrough
    fake_pytest.mark = types.SimpleNamespace(
        parametrize=_passthrough,
        xfail=_passthrough,
        xdist_group=_passthrough,
    )
    fake_pytest.param = lambda *args, **kwargs: args
    fake_pytest.skip = mock.MagicMock()
    fake_pytest.Item = object
    fake_pytest.CallInfo = _StubCallInfo
    fake_pytest.FixtureRequest = object

    fake_report_mod: Any = types.ModuleType("tests.ui.sxs_report_generator")
    fake_report_mod.generate_sxs_html_report = mock.MagicMock()
    fake_report_mod.publish_report_artifact = mock.MagicMock()
    fake_journeys_mod: Any = types.ModuleType("tests.ui.test_user_journeys")
    fake_journeys_mod.dispatch_action = mock.MagicMock()
    fake_journeys_mod.JOURNEY_SCENARIOS = ()
    fake_journeys_mod.JourneyScenario = object

    fake_tests_pkg: Any = types.ModuleType("tests")
    fake_tests_pkg.__path__ = []
    fake_tests_ui_pkg: Any = types.ModuleType("tests.ui")
    fake_tests_ui_pkg.__path__ = [str(module_path.parent)]
    fake_tests_ui_pkg.invariants = invariants
    fake_tests_ui_pkg.journey_capture = journey_capture
    fake_tests_ui_pkg.sxs_diff_engine = sxs_diff_engine
    fake_tests_ui_pkg.sxs_report_generator = fake_report_mod
    fake_tests_ui_pkg.test_user_journeys = fake_journeys_mod
    fake_tests_ui_pkg.ui_helpers = ui_helpers
    fake_tests_pkg.ui = fake_tests_ui_pkg
    patched_modules: dict[str, types.ModuleType] = {
        "playwright": fake_playwright,
        "playwright.sync_api": fake_sync_api,
        "pytest": fake_pytest,
        "tests": fake_tests_pkg,
        "tests.ui": fake_tests_ui_pkg,
        "tests.ui.invariants": invariants,
        "tests.ui.journey_capture": journey_capture,
        "tests.ui.sxs_diff_engine": sxs_diff_engine,
        "tests.ui.sxs_report_generator": fake_report_mod,
        "tests.ui.test_user_journeys": fake_journeys_mod,
        "tests.ui.ui_helpers": ui_helpers,
    }
    if conftest_mod is not None:
      fake_tests_ui_pkg.conftest = conftest_mod
      patched_modules["tests.ui.conftest"] = conftest_mod

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

    # Verify URL-scoped KNOWN_UPSTREAM_BUGS and BrowserErrors.ignoring().
    listeners: dict[str, Callable[[Any], None]] = {}
    scoped_page = types.SimpleNamespace(
        url="http://127.0.0.1:8000/?tag=overview_page",
        on=listeners.__setitem__,
    )
    url_tracker = conftest_mod.BrowserErrors()
    url_tracker.attach(scoped_page)
    listeners["console"](
        types.SimpleNamespace(
            type="error",
            text="Failed to load resource: net::ERR_FAILED",
            location={"url": "https://www.gstatic.com/charts/loader.js"},
        )
    )
    listeners["console"](
        types.SimpleNamespace(
            type="error",
            text=(
                "WARNING: All log messages before absl::InitializeLog() is"
                " called are written to STDERR"
            ),
            location={
                "url": (
                    "http://127.0.0.1:8000/data/plugin/profile/"
                    "trace_viewer_v2.js"
                )
            },
        )
    )
    url_tracker.assert_clean("gstatic_and_oss_absl_allowed")
    with mock.patch.object(conftest_mod.sys, "platform", "linux"):
      linux_args = conftest_mod.browser_type_launch_args({"args": ["--foo"]})
    self.assertIn("--foo", linux_args["args"])
    for flag in (
        "--enable-unsafe-webgpu",
        "--enable-unsafe-swiftshader",
        "--use-vulkan=swiftshader",
        "--disable-vulkan-surface",
        "--disable-gpu-watchdog",
        "--ignore-gpu-blocklist",
    ):
      self.assertIn(flag, linux_args["args"])
    with mock.patch.object(conftest_mod.sys, "platform", "darwin"):
      mac_args = conftest_mod.browser_type_launch_args({"args": ["--foo"]})
    self.assertEqual(mac_args, {"args": ["--foo"]})
    listeners["console"](
        types.SimpleNamespace(
            type="error",
            text="Failed to load resource: net::ERR_FAILED",
            location={"url": "http://127.0.0.1:8000/main.js"},
        )
    )
    with self.assertRaisesRegex(AssertionError, "Failed to load resource"):
      url_tracker.assert_clean("same_origin_blocked")
    data_url = "http://127.0.0.1:8000/data/plugin/profile/data"

    def _respond_503(run: str) -> None:
      listeners["response"](
          types.SimpleNamespace(
              status=503,
              url=f"{data_url}?run={run}",
              request=types.SimpleNamespace(method="GET"),
          )
      )

    with url_tracker.ignoring("HTTP 503", url_pattern="/data/plugin/profile"):
      _respond_503("r1")
    _respond_503("r2")
    self.assertEqual(
        url_tracker.unhandled_errors(),
        [
            "[ERROR] Failed to load resource: net::ERR_FAILED",
            f"[HTTP 503] GET {data_url}?run=r2",
        ],
    )

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

    route_re = ui_helpers.RUN_TOOLS_ROUTE_RE
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

    fake_page = mock.MagicMock()
    with mock.patch.object(ui_helpers.sync_api, "expect", create=True):
      with self.assertRaisesRegex(RuntimeError, "boom"):
        with ui_helpers.enabling_run_tool(fake_page, "inference_profile"):
          raise RuntimeError("boom")
    fake_page.route.assert_called_once()
    registered_re, enable_fn = fake_page.route.call_args.args
    self.assertIs(registered_re, route_re)
    fake_page.goto.assert_called_once_with("about:blank")
    fake_page.unroute.assert_called_once_with(route_re, enable_fn)

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

    sxs_mod = self._load_isolated_ui_module("test_journey_sxs.py", conftest_mod)
    empty_dom = sxs_diff_engine.DomDiff(
        has_changes=False, unified_diff="", added_lines=0, deleted_lines=0
    )
    empty_net = sxs_diff_engine.NetworkDiff(
        has_changes=False, request_count_a=0, request_count_b=0
    )
    mask_wp = sxs_diff_engine.WaypointDiff(
        journey_name="j",
        waypoint_name="00_overview",
        visual=sxs_diff_engine.VisualDiff(
            diff_ratio=0.0,
            total_pixels=100,
            diff_pixels=0,
            mask_error="Mask covers >50% of page (60/100 px)",
        ),
        dom=empty_dom,
        network=empty_net,
        diff_hash="h",
    )
    decode_wp = sxs_diff_engine.WaypointDiff(
        journey_name="j",
        waypoint_name="00_overview",
        visual=sxs_diff_engine.VisualDiff(
            diff_ratio=1.0,
            total_pixels=10,
            diff_pixels=0,
            decode_error="cannot identify image file",
        ),
        dom=empty_dom,
        network=empty_net,
        diff_hash="h",
    )
    # pylint: disable=protected-access
    self.assertIn(
        "mask: Mask covers >50% of page", sxs_mod._describe_divergence(mask_wp)
    )
    self.assertIn(
        "decode: cannot identify image file",
        sxs_mod._describe_divergence(decode_wp),
    )
    # pylint: enable=protected-access


if __name__ == "__main__":
  if absltest is not None:
    absltest.main()
  else:
    unittest.main()
