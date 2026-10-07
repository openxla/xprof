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
import gzip
import os
import pathlib
import re
import subprocess
import threading
import traceback
from typing import Any
import unittest
from unittest import mock
import urllib.error
import urllib.request
from wsgiref import simple_server

# pylint: disable=g-import-not-at-top,g-importing-member
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

      def bounding_box(self):
        return self._box

      def inner_text(self):
        return self._text

      def count(self):
        return 1

      def locator(self, sel):
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
          del script, arg
          return self._cells

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
    self.assertRegex("/?node_name=fusion.12&module_name=m", url)
    self.assertNotRegex("/?node_name=add.3&module_name=fusion.12", url)

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


if __name__ == "__main__":
  unittest.main()
