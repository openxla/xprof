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

import gzip
import os
import pathlib
import re
import subprocess
import threading
import traceback
import unittest
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
  except ImportError:
    profile_plugin = None  # pyrefly: ignore[assignment]
    profile_plugin_loader = None  # pyrefly: ignore[assignment]
    server = None  # pyrefly: ignore[assignment]
    base_plugin = None  # pyrefly: ignore[assignment]
    plugin_event_multiplexer = None  # pyrefly: ignore[assignment]
    import invariants  # pyrefly: ignore[missing-import]
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
    """Verifies DOM invariants detect collapsed charts and empty cards."""

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


if __name__ == "__main__":
  unittest.main()
