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
"""Server HTTP protocol, packaging integrity, and repository cleanliness tests."""

import gzip
import os
import pathlib
import re
import unittest
import urllib.error
import urllib.request


def _find_repo_root() -> pathlib.Path | None:
  """Locates the repository root across local workspace and runfiles."""
  test_srcdir = os.environ.get("TEST_SRCDIR")
  test_workspace = os.environ.get("TEST_WORKSPACE", "google3")
  if test_srcdir:
    candidate = pathlib.Path(test_srcdir) / test_workspace / "third_party/xprof"
    if candidate.is_dir():
      return candidate

  current = pathlib.Path(__file__).parent
  for parent in [current, *current.parents]:
    if (parent / "demo").is_dir() and (parent / "frontend").is_dir():
      return parent

  return None


class ServerAndPackagingContractTest(unittest.TestCase):
  """Tests for server HTTP protocol, packaging integrity, and cleanliness."""

  server_url: str | None = None

  def setUp(self) -> None:
    super().setUp()
    self.server_url = os.environ.get("XPROF_SERVER_URL")

  def test_runtime_js_serves_javascript_mime(self) -> None:
    """Verifies that runtime.js is served with javascript MIME type."""
    if not self.server_url:
      self.skipTest("Live XProf server not running.")
    url = f"{self.server_url.rstrip('/')}/data/plugin/profile/runtime.js"
    req = urllib.request.Request(url, method="GET")
    try:
      with urllib.request.urlopen(req, timeout=5.0) as resp:
        status = resp.status
        content_type = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as err:
      status = err.code
      content_type = ""
    except urllib.error.URLError as err:
      self.fail(f"Failed to connect to server at {url}: {err.reason}")
    self.assertEqual(status, 200)
    self.assertIn("javascript", content_type.lower())

  def test_nonexistent_route_returns_404(self) -> None:
    """Verifies that unmapped static routes return HTTP 404 rather than 200."""
    if not self.server_url:
      self.skipTest("Live XProf server not running.")
    url = f"{self.server_url.rstrip('/')}/data/plugin/profile/nonexistent_route_abc.json"
    req = urllib.request.Request(url, method="GET")
    try:
      with urllib.request.urlopen(req, timeout=5.0) as resp:
        status = resp.status
    except urllib.error.HTTPError as err:
      status = err.code
    except urllib.error.URLError as err:
      self.fail(f"Failed to connect to server at {url}: {err.reason}")
    self.assertEqual(status, 404)

  def test_working_tree_cleanliness_after_server_run(self) -> None:
    """Verifies that server execution does not pollute demo files."""
    repo_root = _find_repo_root()
    if not repo_root:
      self.skipTest(
          "Repository root not found in current execution environment."
      )
    demo_profile_dir = repo_root / "demo" / "plugins" / "profile"
    if not demo_profile_dir.is_dir():
      self.skipTest(f"Demo profile directory not present at {demo_profile_dir}")

    polluting_files = []
    for p in demo_profile_dir.rglob("*"):
      if p.is_file() and p.name.endswith(".cached_tools.json"):
        polluting_files.append(str(p.relative_to(repo_root)))

    self.assertFalse(
        polluting_files,
        "Detected server cache pollution in tracked demo directories:"
        f" {polluting_files}",
    )

  def test_no_raw_google3_internal_imports_in_frontend(self) -> None:
    """Verifies exported TypeScript does not import raw google3 paths."""
    repo_root = _find_repo_root()
    if not repo_root:
      self.skipTest(
          "Repository root not found in current execution environment."
      )
    frontend_dir = repo_root / "frontend" / "app"
    if not frontend_dir.is_dir():
      self.skipTest(f"Frontend app directory not present at {frontend_dir}")

    excluded_patterns = [
        "source_code_editor",
        "syntax_highlight_service",
        "_test.ts",
    ]
    rewritten_imports = {
        "google3/third_party/javascript/ngx_json_viewer/src/ngx-json-viewer.module",
    }

    violations = []
    import_pattern = re.compile(
        r"""['"](google3/third_party/javascript/[^'"]+)['"]"""
    )
    strip_block_pattern = re.compile(
        r"//\s*copy" + r"bara:strip_begin.*?//\s*copy" + r"bara:strip_end",
        re.DOTALL,
    )

    for ts_file in frontend_dir.rglob("*.ts"):
      if any(pat in str(ts_file) for pat in excluded_patterns):
        continue
      try:
        raw_text = ts_file.read_text(encoding="utf-8")
        text = strip_block_pattern.sub("", raw_text)
        for match in import_pattern.finditer(text):
          matched_import = match.group(1)
          if matched_import not in rewritten_imports:
            violations.append(
                f"{ts_file.name}: contains unexported raw google3 import"
                f" {matched_import}"
            )
      except OSError:
        continue

    self.assertFalse(
        violations,
        "Forbidden raw internal google3 imports detected in exported frontend:"
        f" {violations}",
    )

  def test_proto_interfaces_contract(self) -> None:
    """Verifies .d.ts.gz files are valid archives declared in BUILD.oss."""
    repo_root = _find_repo_root()
    if not repo_root:
      self.skipTest(
          "Repository root not found in current execution environment."
      )
    interfaces_dir = repo_root / "frontend" / "app" / "common" / "interfaces"
    if not interfaces_dir.is_dir():
      self.skipTest(f"Interfaces directory not present at {interfaces_dir}")

    build_oss = interfaces_dir / "BUILD.oss"
    self.assertTrue(
        build_oss.is_file(), f"Missing BUILD.oss in {interfaces_dir}"
    )
    build_oss_text = build_oss.read_text(encoding="utf-8")

    gz_files = list(interfaces_dir.glob("*.d.ts.gz"))
    self.assertTrue(gz_files, f"No .d.ts.gz archives found in {interfaces_dir}")

    for gz_file in gz_files:
      with gzip.open(gz_file, "rb") as f:
        content = f.read(1024)
        self.assertTrue(
            content, f"Compressed interface {gz_file.name} is empty"
        )
      self.assertIn(
          gz_file.name,
          build_oss_text,
          f"Interface {gz_file.name} is not declared in {build_oss}",
      )


if __name__ == "__main__":
  unittest.main()
