# Profile Plugin Python Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the profile-plugin Python architecture so domain logic lives in pure services, `plugin.py` is a thin HTTP façade, and convert/I/O are injectable ports—without changing HTTP paths or query semantics.

**Architecture:** Extract Werkzeug-free services (`sessions`, `hosts`, `tool_data`, `tools/options`) behind typed models (`ToolRequest`, `SessionRef`, `HostSelection`, `ToolResult`). `ProfilePlugin` only parses requests, calls services, and maps results through `respond()`. Production wiring stays in the loader/plugin constructor; tests inject fakes.

**Tech Stack:** Python 3.10+, Werkzeug (HTTP edge only), etils.epath, absltest/unittest, existing `profile_io` + `convert.raw_to_tool_data` / pywrap.

**Spec:** `docs/superpowers/specs/2026-07-10-profile-plugin-python-architecture-design.md`

## Global Constraints

- Preserve HTTP routes and query param names (`run`, `tag`, `host`, `hosts`, `use_saved_result`, trace options, etc.).
- Preserve gzip+CSP behavior via `profile_plugin.http.respond.respond`.
- Preserve tools sort order and `trace_viewer@` overriding `trace_viewer`.
- Preserve ALL_HOSTS host aggregation rules from `tools/registry.py`.
- No Werkzeug imports in `services/` or `tools/` (except `http/`).
- Do not ship tests in the production `py_library` glob.
- Behavior-compatible with current master unless a task explicitly documents a change.
- Prefer TDD: failing test → implement → pass → commit.
- Working directory: repo root `xprof`. Run unit tests with:
  `cd plugin && PYTHONPATH=. python3 -m unittest discover -s xprof/profile_plugin/tests -p '*_test.py' -v`
  (or the package runner `python3 xprof/profile_plugin/tests/run_all_tests.py` with stubs as needed).

## File Structure (this plan delivers Phases 0–5)

| Path | Responsibility |
|------|----------------|
| `plugin/xprof/BUILD` | Exclude tests from prod library |
| `plugin/xprof/profile_plugin/models.py` | Domain dataclasses |
| `plugin/xprof/profile_plugin/deps.py` | Protocols for convert/FS/version |
| `plugin/xprof/profile_plugin/http/parse_request.py` | Werkzeug → ToolRequest |
| `plugin/xprof/profile_plugin/services/__init__.py` | Package marker |
| `plugin/xprof/profile_plugin/services/sessions.py` | Session path + run discovery |
| `plugin/xprof/profile_plugin/services/hosts.py` | Host selection |
| `plugin/xprof/profile_plugin/services/tool_data.py` | Convert orchestration |
| `plugin/xprof/profile_plugin/tools/options/` | Per-tool option builders |
| `plugin/xprof/profile_plugin/plugin.py` | Thin façade (progressively) |
| `plugin/xprof/profile_plugin/tests/unit/` | Pure unit tests |
| `plugin/xprof/profile_plugin/tests/fixtures/` | Golden fixtures |

**Out of this plan (Phases 6–11, separate plan later):** cache_generation service extract, capture/hlo/static services, full `plugin.py` ≤400 LOC collapse, public API shrink, CLI adoption, full test suite reorg.

---

### Task 1: BUILD guardrail — exclude tests from production library

**Files:**
- Modify: `plugin/xprof/BUILD`
- Test: manual inspection of `srcs` after change

**Interfaces:**
- Consumes: existing `py_library(name="profile_plugin")`
- Produces: production library without `profile_plugin/tests/**`

- [ ] **Step 1: Update the profile_plugin py_library srcs**

In `plugin/xprof/BUILD`, replace:

```python
py_library(
    name = "profile_plugin",
    srcs = glob(["profile_plugin/**/*.py"]),
```

with:

```python
py_library(
    name = "profile_plugin",
    srcs = glob(
        ["profile_plugin/**/*.py"],
        exclude = ["profile_plugin/tests/**"],
    ),
```

Keep the existing `profile_plugin_package_test` target (it already globs tests separately). Ensure its `deps` still include `:profile_plugin` and any test-only needs.

- [ ] **Step 2: Verify test target still lists test sources**

Confirm `profile_plugin_package_test` (or equivalent) still has:

```python
srcs = glob(["profile_plugin/tests/**/*.py"]),
```

- [ ] **Step 3: Commit**

```bash
git add plugin/xprof/BUILD
git commit -m "build: exclude profile_plugin tests from production py_library"
```

---

### Task 2: Domain models

**Files:**
- Create: `plugin/xprof/profile_plugin/models.py`
- Create: `plugin/xprof/profile_plugin/tests/unit/models_test.py`
- Modify: `plugin/xprof/profile_plugin/tests/run_all_tests.py` only if discover path needs updating (prefer placing tests so existing discover still finds `*_test.py`)

**Interfaces:**
- Produces:
  - `SessionRef(frontend_run: str, directory: str)`
  - `ToolRequest(run, tool, host, hosts, use_saved_result, raw_args)`
  - `HostSelection(selected_hosts: tuple[str, ...], asset_paths: tuple[Any, ...])`
  - `ToolResult(data, content_type, content_encoding=None)`

- [ ] **Step 1: Write failing unit tests**

Create `plugin/xprof/profile_plugin/tests/unit/__init__.py` (empty) and `plugin/xprof/profile_plugin/tests/unit/models_test.py`:

```python
"""Unit tests for profile_plugin.models."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from xprof.profile_plugin.models import (
    HostSelection,
    SessionRef,
    ToolRequest,
    ToolResult,
)


class ModelsTest(unittest.TestCase):

  def test_session_ref_frozen(self):
    ref = SessionRef(frontend_run='s1', directory='/tmp/s1')
    self.assertEqual(ref.frontend_run, 's1')
    with self.assertRaises(Exception):
      ref.frontend_run = 'other'  # type: ignore[misc]

  def test_tool_request_hosts_default_empty(self):
    req = ToolRequest(
        run='r',
        tool='overview_page',
        host=None,
        hosts=(),
        use_saved_result=True,
        raw_args={},
    )
    self.assertEqual(req.hosts, ())
    self.assertTrue(req.use_saved_result)

  def test_host_selection_tuple_paths(self):
    paths = (SimpleNamespace(name='a'),)
    sel = HostSelection(selected_hosts=('h0',), asset_paths=paths)
    self.assertEqual(sel.selected_hosts, ('h0',))
    self.assertEqual(len(sel.asset_paths), 1)

  def test_tool_result_defaults(self):
    result = ToolResult(data='{}', content_type='application/json')
    self.assertIsNone(result.content_encoding)


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 2: Run tests — expect fail**

```bash
cd plugin && PYTHONPATH=. python3 -m unittest xprof.profile_plugin.tests.unit.models_test -v
```

Expected: `ModuleNotFoundError: No module named 'xprof.profile_plugin.models'`

- [ ] **Step 3: Implement models**

Create `plugin/xprof/profile_plugin/models.py`:

```python
# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Domain models for the profile plugin (Werkzeug-free)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SessionRef:
  """Resolved profile session directory for a frontend run name."""

  frontend_run: str
  directory: str


@dataclass(frozen=True)
class ToolRequest:
  """Normalized tool request after HTTP parsing."""

  run: str
  tool: str
  host: str | None
  hosts: tuple[str, ...]
  use_saved_result: bool
  raw_args: Mapping[str, str]


@dataclass(frozen=True)
class HostSelection:
  """Hosts and corresponding XPlane asset paths for a tool request."""

  selected_hosts: tuple[str, ...]
  asset_paths: tuple[Any, ...]  # epath.Path-like; avoid hard epath dep here


@dataclass(frozen=True)
class ToolResult:
  """Payload returned to the HTTP layer."""

  data: bytes | str | None
  content_type: str
  content_encoding: str | None = None
```

- [ ] **Step 4: Run tests — expect pass**

```bash
cd plugin && PYTHONPATH=. python3 -m unittest xprof.profile_plugin.tests.unit.models_test -v
```

Expected: OK

- [ ] **Step 5: Commit**

```bash
git add plugin/xprof/profile_plugin/models.py \
  plugin/xprof/profile_plugin/tests/unit/
git commit -m "feat(profile_plugin): add domain models for sessions and tools"
```

---

### Task 3: Dependency ports

**Files:**
- Create: `plugin/xprof/profile_plugin/deps.py`
- Create: `plugin/xprof/profile_plugin/tests/unit/deps_test.py` (typing-only smoke: Protocol structure)

**Interfaces:**
- Produces:
  - `ConvertPort` Protocol with `xspace_to_tool_data`, `xspace_to_tool_names`, `json_to_csv_string`
  - `FileSystemFactory` Protocol with `get(path: str) -> Any`
  - `VersionProvider` Protocol with `__version__: str`

- [ ] **Step 1: Write deps.py**

```python
# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Injectable ports for convert and filesystem boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConvertPort(Protocol):
  def xspace_to_tool_data(
      self,
      xspace_paths: Sequence[Any],
      tool: str,
      params: Mapping[str, Any],
  ) -> tuple[bytes | str | None, str]:
    ...

  def xspace_to_tool_names(self, xspace_paths: Sequence[str]) -> Sequence[str]:
    ...

  def json_to_csv_string(self, data: Any) -> str:
    ...


@runtime_checkable
class FileSystemFactory(Protocol):
  def get(self, path: str) -> Any:
    """Return a ProfileFileSystem for path."""
    ...


@runtime_checkable
class VersionProvider(Protocol):
  __version__: str
```

- [ ] **Step 2: Smoke test**

```python
"""Smoke tests for deps protocols."""

from __future__ import annotations

import unittest

from xprof.profile_plugin.deps import ConvertPort, FileSystemFactory, VersionProvider


class _FakeConvert:
  def xspace_to_tool_data(self, xspace_paths, tool, params):
    return '{}', 'application/json'

  def xspace_to_tool_names(self, xspace_paths):
    return ['overview_page']

  def json_to_csv_string(self, data):
    return 'a,b\n'


class DepsTest(unittest.TestCase):

  def test_fake_convert_is_convert_port(self):
    self.assertIsInstance(_FakeConvert(), ConvertPort)


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 3: Run and commit**

```bash
cd plugin && PYTHONPATH=. python3 -m unittest xprof.profile_plugin.tests.unit.deps_test -v
git add plugin/xprof/profile_plugin/deps.py plugin/xprof/profile_plugin/tests/unit/deps_test.py
git commit -m "feat(profile_plugin): add ConvertPort and related protocols"
```

---

### Task 4: Parse Werkzeug request → ToolRequest

**Files:**
- Create: `plugin/xprof/profile_plugin/http/parse_request.py`
- Create: `plugin/xprof/profile_plugin/tests/unit/parse_request_test.py`
- Modify: `plugin/xprof/profile_plugin/http/__init__.py` to export `tool_request_from_werkzeug` if desired

**Interfaces:**
- Consumes: `ToolRequest`, `get_bool_arg` from `request_params`
- Produces: `tool_request_from_args(args: Mapping) -> ToolRequest`

Note: implement against a plain Mapping (request.args) so unit tests need no full Werkzeug Request.

- [ ] **Step 1: Failing test**

```python
"""Tests for HTTP → ToolRequest parsing."""

from __future__ import annotations

import unittest

from xprof.profile_plugin.http.parse_request import tool_request_from_args


class ParseRequestTest(unittest.TestCase):

  def test_basic_fields(self):
    req = tool_request_from_args({
        'run': 'session_a',
        'tag': 'overview_page',
        'host': 'ALL_HOSTS',
        'use_saved_result': 'true',
    })
    self.assertEqual(req.run, 'session_a')
    self.assertEqual(req.tool, 'overview_page')
    self.assertEqual(req.host, 'ALL_HOSTS')
    self.assertEqual(req.hosts, ())
    self.assertTrue(req.use_saved_result)

  def test_hosts_csv(self):
    req = tool_request_from_args({
        'run': 'r',
        'tag': 'trace_viewer@',
        'hosts': 'h0,h1',
    })
    self.assertEqual(req.hosts, ('h0', 'h1'))

  def test_use_saved_result_default_true(self):
    req = tool_request_from_args({'run': 'r', 'tag': 'overview_page'})
    self.assertTrue(req.use_saved_result)

  def test_raw_args_preserves_strings(self):
    req = tool_request_from_args({
        'run': 'r',
        'tag': 'trace_viewer@',
        'resolution': '8000',
    })
    self.assertEqual(req.raw_args.get('resolution'), '8000')


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 2: Run — expect fail (import error)**

- [ ] **Step 3: Implement parse_request.py**

```python
# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Parse HTTP query args into domain ToolRequest (edge adapter)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xprof.profile_plugin.http.request_params import get_bool_arg
from xprof.profile_plugin.models import ToolRequest


def tool_request_from_args(args: Mapping[str, Any]) -> ToolRequest:
  """Build ToolRequest from a Werkzeug-like args mapping."""
  hosts_param = args.get('hosts') or ''
  if isinstance(hosts_param, str) and hosts_param.strip():
    hosts = tuple(h.strip() for h in hosts_param.split(',') if h.strip())
  else:
    hosts = ()
  # Normalize raw_args to str values for option builders.
  raw_args = {
      str(k): '' if v is None else str(v) for k, v in args.items()
  }
  return ToolRequest(
      run=str(args.get('run') or ''),
      tool=str(args.get('tag') or ''),
      host=(str(args['host']) if args.get('host') is not None else None),
      hosts=hosts,
      use_saved_result=get_bool_arg(args, 'use_saved_result', True),
      raw_args=raw_args,
  )
```

- [ ] **Step 4: Run tests — pass; commit**

```bash
cd plugin && PYTHONPATH=. python3 -m unittest xprof.profile_plugin.tests.unit.parse_request_test -v
git add plugin/xprof/profile_plugin/http/parse_request.py \
  plugin/xprof/profile_plugin/tests/unit/parse_request_test.py
git commit -m "feat(profile_plugin): parse HTTP args into ToolRequest"
```

---

### Task 5: Session resolution service

**Files:**
- Create: `plugin/xprof/profile_plugin/services/__init__.py`
- Create: `plugin/xprof/profile_plugin/services/sessions.py`
- Create: `plugin/xprof/profile_plugin/tests/unit/sessions_test.py`

**Interfaces:**
- Consumes: `profile_io.get_file_system`, epath, `SessionRef`
- Produces:
  - `SessionResolver.resolve_run_map(session_path, run_path) -> dict[str, str] | None`
  - `SessionResolver.resolve_directory(run, run_map, logdir, run_cache) -> str`
  - Logic must match current precedence: `session_path` > `run_path` > logdir layout

- [ ] **Step 1: Write failing tests for precedence**

```python
"""Unit tests for session path resolution."""

from __future__ import annotations

import os
import tempfile
import unittest

from etils import epath

from xprof.profile_plugin.services.sessions import SessionResolver


class SessionResolverTest(unittest.TestCase):

  def setUp(self):
    self._td = tempfile.TemporaryDirectory()
    self.root = self._td.name
    self.resolver = SessionResolver(epath_module=epath)

  def tearDown(self):
    self._td.cleanup()

  def _touch_xplane(self, session_dir: str, host: str = 'h0'):
    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, f'{host}.xplane.pb')
    with open(path, 'wb') as f:
      f.write(b'x')
    return session_dir

  def test_session_path_map(self):
    session = self._touch_xplane(os.path.join(self.root, 'my_session'))
    m = self.resolver.run_map_from_params(session_path=session, run_path=None)
    self.assertEqual(m, {'my_session': session})

  def test_run_path_lists_sessions(self):
    s1 = self._touch_xplane(os.path.join(self.root, 'runs', 'a'))
    s2 = self._touch_xplane(os.path.join(self.root, 'runs', 'b'))
    m = self.resolver.run_map_from_params(
        session_path=None, run_path=os.path.join(self.root, 'runs')
    )
    self.assertIn('a', m)
    self.assertIn('b', m)
    self.assertEqual(m['a'], s1)


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 2: Implement SessionResolver**

Port behavior from `ProfilePlugin._session_dir_by_run_name_from_request` and path checks:

```python
# services/sessions.py (core methods)

class SessionResolver:
  def __init__(self, epath_module, fs_factory=None):
    self._epath = epath_module
    self._fs_factory = fs_factory  # optional; default profile_io.get_file_system

  def _fs(self, path: str):
    if self._fs_factory is not None:
      return self._fs_factory(path)
    from xprof import profile_io
    return profile_io.get_file_system(path, self._epath)

  def run_map_from_params(
      self, session_path: str | None, run_path: str | None
  ) -> dict[str, str] | None:
    if session_path:
      if self._fs(session_path).dir_has_xplane_files(session_path):
        run_name = self._epath.Path(session_path).name
        return {run_name: session_path}
      return {}
    if run_path:
      return dict(self._fs(run_path).get_session_paths(run_path))
    return None
```

Also move `tb_run_directory` helper and document that full `generate_runs` logdir walk is Task 6 (or continue in this task if small enough).

**Minimum for this task:** `run_map_from_params` + `resolve_run_dir(run, run_map, logdir, cache_dict) -> str` ported from `_run_dir` without the full filesystem walk.

- [ ] **Step 3: Port `_run_dir` resolution when run_map is provided**

```python
  def resolve_run_dir(
      self,
      run: str,
      run_map: dict[str, str] | None,
      logdir: str | None,
      run_dir_cache: dict[str, str],
      cache_lock,
  ) -> str:
    if run_map is not None:
      if run in run_map:
        return run_map[run]
      raise ValueError(f'Run {run} not found in run map: {run_map}')
    with cache_lock:
      if run in run_dir_cache:
        return run_dir_cache[run]
    # Fall back: same split logic as ProfilePlugin._run_dir for logdir
    ...
```

Copy the logdir branch from current `plugin.py` `_run_dir` exactly (including `plugin_asset_util.PluginDirectory`).

- [ ] **Step 4: Tests pass; commit**

```bash
git add plugin/xprof/profile_plugin/services/ \
  plugin/xprof/profile_plugin/tests/unit/sessions_test.py
git commit -m "feat(profile_plugin): add SessionResolver for path precedence"
```

---

### Task 6: Wire SessionResolver into ProfilePlugin (runs path)

**Files:**
- Modify: `plugin/xprof/profile_plugin/plugin.py`
- Test: existing `tests/plugin_integration_test.py` + unit sessions tests

**Interfaces:**
- Consumes: `SessionResolver`
- Produces: `ProfilePlugin.runs_imp` / `_run_dir` / `_session_dir_by_run_name_from_request` delegate to resolver

- [ ] **Step 1: Construct resolver in `__init__`**

```python
from xprof.profile_plugin.services.sessions import SessionResolver

# in __init__:
self._sessions = SessionResolver(epath_module=self._epath)
```

- [ ] **Step 2: Replace `_session_dir_by_run_name_from_request` body**

```python
  def _session_dir_by_run_name_from_request(self, request=None):
    session_path_arg = request.args.get('session_path') if request else None
    run_path_arg = (
        request.args.get('run_path')
        if request and not session_path_arg
        else None
    )
    return self._sessions.run_map_from_params(session_path_arg, run_path_arg)
```

- [ ] **Step 3: Replace `_run_dir` body to call `self._sessions.resolve_run_dir(...)`**

Keep method names for tests that call `_run_dir` / `_session_dir_by_run_name_from_request`.

- [ ] **Step 4: Run package tests + integration**

```bash
cd plugin && PYTHONPATH=. python3 xprof/profile_plugin/tests/run_all_tests.py
```

Expected: OK (or fix regressions).

- [ ] **Step 5: Commit**

```bash
git add plugin/xprof/profile_plugin/plugin.py
git commit -m "refactor(profile_plugin): delegate session path resolution to SessionResolver"
```

---

### Task 7: Host selection service

**Files:**
- Create: `plugin/xprof/profile_plugin/services/hosts.py`
- Create: `plugin/xprof/profile_plugin/tests/unit/hosts_test.py`

**Interfaces:**
- Consumes: `parse_filename`, `ALL_HOSTS`, `supports_multi_host_selection`, `XPLANE_TOOLS_ALL_HOSTS_ONLY`, epath paths
- Produces: `HostSelector.select(run_dir, tool, host, hosts_param, list_xplane_basenames) -> HostSelection`

- [ ] **Step 1: Failing table tests**

Cover:

1. Multi-host + `overview_page` → `selected_hosts` via ALL_HOSTS-only collapse when listing hosts for UI is separate; for select with `host=ALL_HOSTS` all assets returned.
2. `host=h0` single path.
3. `hosts=h0,h1` for `trace_viewer@`.
4. Missing host raises `FileNotFoundError`.

Port assertions from current `plugin_integration_test` host cases.

- [ ] **Step 2: Implement by moving `_get_valid_hosts` logic**

Copy the algorithm from `ProfilePlugin._get_valid_hosts` into `HostSelector.select`, returning `HostSelection` instead of a bare tuple.

```python
class HostSelector:
  def select(
      self,
      run_dir: str,
      tool: str,
      host: str | None,
      hosts_param: str | None,
      xplane_basenames: Sequence[str],
      path_join,
  ) -> HostSelection:
    ...
    return HostSelection(
        selected_hosts=tuple(selected_hosts),
        asset_paths=tuple(asset_paths),
    )
```

- [ ] **Step 3: Tests pass; commit**

```bash
git commit -m "feat(profile_plugin): add HostSelector service"
```

---

### Task 8: Wire HostSelector into ProfilePlugin

**Files:**
- Modify: `plugin/xprof/profile_plugin/plugin.py`

- [ ] **Step 1: `self._hosts = HostSelector()` in `__init__`**

- [ ] **Step 2: `_get_valid_hosts` becomes a thin wrapper**

```python
  def _get_valid_hosts(self, run_dir, run, tool, hosts_param, host):
    basenames = self._get_xplane_basenames(run_dir)
    selection = self._hosts.select(
        run_dir=run_dir,
        tool=tool,
        host=host,
        hosts_param=hosts_param,
        xplane_basenames=basenames,
        path_join=lambda *p: self._epath.Path(os.path.join(*p)),
    )
    return list(selection.selected_hosts), list(selection.asset_paths)
```

Keep return type as today for callers.

- [ ] **Step 3: Run full package tests; commit**

```bash
git commit -m "refactor(profile_plugin): delegate host selection to HostSelector"
```

---

### Task 9: Tool options builders + counter-names handler

**Files:**
- Create: `plugin/xprof/profile_plugin/tools/options/__init__.py`
- Create: `plugin/xprof/profile_plugin/tools/options/base.py`
- Create: `plugin/xprof/profile_plugin/tools/options/trace.py`
- Create: `plugin/xprof/profile_plugin/tools/options/graph.py`
- Create: `plugin/xprof/profile_plugin/tools/options/memory.py`
- Create: `plugin/xprof/profile_plugin/tools/options/default.py`
- Create: `plugin/xprof/profile_plugin/tools/options/registry.py`
- Create: `plugin/xprof/profile_plugin/services/counter_names.py`
- Create: `plugin/xprof/profile_plugin/tests/unit/tool_options_test.py`

**Interfaces:**
- Produces:
  - `build_tool_params(req: ToolRequest) -> dict[str, Any]` merging common fields + family builder
  - `try_counter_names_only(req) -> ToolResult | None` if tool is perf_counters and names_only

- [ ] **Step 1: Tests for trace options format override**

```python
def test_trace_event_name_forces_json_format(self):
  req = ToolRequest(
      run='r',
      tool='trace_viewer@',
      host='h0',
      hosts=(),
      use_saved_result=True,
      raw_args={
          'tag': 'trace_viewer@',
          'run': 'r',
          'event_name': 'op',
          'format': 'proto',
          'resolution': '4000',
      },
  )
  params = build_tool_params(req, use_saved_result=True, graph_options={})
  tv = params['trace_viewer_options']
  self.assertEqual(tv['format'], 'json')
  self.assertEqual(tv['event_name'], 'op')
```

Also test `try_counter_names_only` returns None when not applicable.

- [ ] **Step 2: Implement builders**

Port fields from current `data_impl` and `_get_graph_viewer_options`:

- Common: `tqx`, `perfetto`, `host`, `module_name`, `program_id`, `use_saved_result`, `group_by`, `refresh_suggestion`, `graph_viewer_options`, `memory_space`
- Trace family: resolution, full_dma, enable_legacy_dcn, times, event_name/format, duration, unique_id, search_*
- Memory: `view_memory_allocation_timeline`
- Graph: node_name, module_name, graph_width, show_metadata, merge_fusion, program_id, format, type

- [ ] **Step 3: Counter names service**

```python
def counter_names_result(device_type: str) -> ToolResult:
  import json
  try:
    from xprof.convert import counter_extractor
    names = counter_extractor.get_all_counters(device_type)
  except FileNotFoundError:
    names = []
  return ToolResult(data=json.dumps(names), content_type='application/json')
```

- [ ] **Step 4: Tests pass; commit**

```bash
git commit -m "feat(profile_plugin): tool option builders and counter-names handler"
```

---

### Task 10: ToolDataService (convert orchestration)

**Files:**
- Create: `plugin/xprof/profile_plugin/services/tool_data.py`
- Create: `plugin/xprof/profile_plugin/tests/unit/tool_data_test.py`
- Modify: `plugin/xprof/profile_plugin/cache/result_cache_policy.py` — ensure `write_cache_version_file` is the only write path used by the service

**Interfaces:**
- Consumes: SessionResolver, HostSelector, `build_tool_params`, ConvertPort, VersionProvider, FileSystemFactory
- Produces:
  - `ToolDataService.get_tool_data(req: ToolRequest, *, session_path=None, run_path=None, logdir=None) -> ToolResult`

- [ ] **Step 1: Unit test with fake convert**

```python
class ToolDataServiceTest(unittest.TestCase):
  def test_passes_hosts_and_tool_to_convert(self):
    calls = []
    class FakeConvert:
      def xspace_to_tool_data(self, paths, tool, params):
        calls.append((tool, list(params.get('hosts', [])), params))
        return '{"ok":true}', 'application/json'
      def xspace_to_tool_names(self, paths):
        return []
      def json_to_csv_string(self, data):
        return ''
    # arrange temp session with two xplanes, wire service...
    result = service.get_tool_data(req)
    self.assertIn('ok', result.data)
    self.assertEqual(calls[0][0], 'overview_page')
```

- [ ] **Step 2: Implement ToolDataService**

Algorithm (match `data_impl`):

1. If counter names-only → return handler result.
2. Resolve run_dir via sessions.
3. `use_saved = should_use_saved_result(run_dir, req.use_saved_result, version, epath)`.
4. `params = build_tool_params(req, use_saved_result=use_saved, ...)`.
5. If tool not in TOOLS and not use_xplane(tool): return ToolResult(None, 'application/json').
6. HostSelection via HostSelector.
7. If no assets: return None data.
8. `data, content_type = convert.xspace_to_tool_data(paths, tool, params)`.
9. If not use_saved: `write_cache_version_file(...)`.
10. Return ToolResult.

- [ ] **Step 3: Tests pass; commit**

```bash
git commit -m "feat(profile_plugin): ToolDataService for convert orchestration"
```

---

### Task 11: Wire ToolDataService into ProfilePlugin.data_impl

**Files:**
- Modify: `plugin/xprof/profile_plugin/plugin.py`
- Modify: tests that patch convert / `_write_cache_version_file` if needed

**Interfaces:**
- `data_impl` becomes:

```python
  def data_impl(self, request):
    from xprof.profile_plugin.http.parse_request import tool_request_from_args
    req = tool_request_from_args(request.args)
    result = self._tool_data.get_tool_data(
        req,
        session_path=request.args.get('session_path'),
        run_path=request.args.get('run_path'),
        logdir=self.logdir,
        run_dir_cache=self._run_to_profile_run_dir,
        cache_lock=self._run_dir_cache_lock,
    )
    return result.data, result.content_type, result.content_encoding
```

- [ ] **Step 1: Construct `ToolDataService` in `__init__` with convert adapter**

```python
class _ConvertAdapter:
  def xspace_to_tool_data(self, paths, tool, params):
    return self._fn(paths, tool, params)  # existing injectable fn
  ...
```

Wire existing `xspace_to_tool_data_fn` into the adapter so tests that inject the fn still work.

- [ ] **Step 2: Replace data_impl body; keep exception types raised from service**

- [ ] **Step 3: Run**

```bash
cd plugin && PYTHONPATH=. python3 xprof/profile_plugin/tests/run_all_tests.py
```

Fix any failures (especially cache version write mocks — prefer patching `write_cache_version_file` on the policy module if method is removed later; for now service should call module function, and plugin `_write_cache_version_file` can delegate to same function for old tests).

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor(profile_plugin): data_impl delegates to ToolDataService"
```

---

### Task 12: Tranche verification gate

**Files:** none new (verification only)

- [ ] **Step 1: Run full package unit/integration suite**

```bash
cd plugin && PYTHONPATH=. python3 xprof/profile_plugin/tests/run_all_tests.py
```

Expected: OK

- [ ] **Step 2: LOC check**

```bash
wc -l plugin/xprof/profile_plugin/plugin.py
```

Expected: reduced from ~1309 but **not yet ≤400** (Phases 6–8 finish façade). Record line count in the PR description.

- [ ] **Step 3: Confirm BUILD exclude still present**

```bash
grep -A6 'name = "profile_plugin"' plugin/xprof/BUILD
```

Expected: `exclude = ["profile_plugin/tests/**"]`

- [ ] **Step 4: Grep Werkzeug free zone**

```bash
grep -R "werkzeug" plugin/xprof/profile_plugin/services plugin/xprof/profile_plugin/tools || true
```

Expected: no matches under `services/` or `tools/` (options must not import werkzeug).

- [ ] **Step 5: Final commit only if docs/status notes needed; otherwise done**

Optional:

```bash
git commit --allow-empty -m "chore: verify profile_plugin architecture tranche 0-5"
```

---

## Later work (not in this plan)

Documented in the design spec Phases 6–11:

- Cache generation service extract
- Capture / HLO / static services
- Collapse `plugin.py` to ≤400 LOC
- Shrink public `__init__.py`
- CLI adopts `ToolDataService`
- Test layout `unit/` vs `integration/` + retire dead legacy tests

Write a follow-on plan file when starting that tranche.

---

## Plan self-review (coverage)

| Spec requirement | Task |
|------------------|------|
| BUILD exclude tests | Task 1 |
| Models | Task 2 |
| Ports | Task 3 |
| Parse request | Task 4 |
| Sessions service | Tasks 5–6 |
| Hosts service | Tasks 7–8 |
| Tool options + counter handler | Task 9 |
| ToolDataService | Tasks 10–11 |
| Preserve HTTP semantics | Global constraints + Task 11 |
| Golden fixtures | Partially via unit tables; expand fixtures dir if needed during Task 11 |
| plugin.py ≤400 | Deferred to later plan (explicit) |
| CLI | Deferred |
| Cache gen service | Deferred |

**Placeholder scan:** none intentional.  
**Type consistency:** `ToolRequest`, `HostSelection`, `ToolResult`, `ConvertPort` names used consistently across tasks.

