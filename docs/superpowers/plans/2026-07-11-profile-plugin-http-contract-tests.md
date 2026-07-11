# Profile Plugin HTTP Contract Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lean HTTP contract suite for `xprof.profile_plugin` that proves frontend route wiring after the `api/*` mixin split: route-table bijection, ≥1 happy + ≥1 error per API group via real WSGI apps from `get_plugin_apps()`, and service spies for `/data` and `/runs` only—no browser/server E2E.

**Architecture:** Thin `http_contract/` package under existing tests. A shared `wsgi_client.call_route` exercises apps from `plugin.get_plugin_apps()` (logged middleware + `@Request.application`). Fixtures reuse `make_plugin` + `create_two_host_session`. Convert is injected; no real C++ path.

**Tech Stack:** Python 3.10+, unittest, Werkzeug `EnvironBuilder`, existing `plugin_factory` / `logdir_factory`, `unittest.mock` for spies.

**Spec:** `docs/superpowers/specs/2026-07-11-profile-plugin-http-contract-tests-design.md`

**Branch:** `refactor/profile-plugin-checks-and-thin-facade`

## Global Constraints

- **Discovery name:** `run_all_tests.py` uses `loader.discover(..., pattern='*_test.py')`. Every test module **must** end in `_test.py`. Design names like `test_contracts_static.py` are **invalid**. Use:
  - `route_table_test.py`
  - `contracts_static_test.py`
  - `contracts_runs_test.py`
  - `contracts_hosts_test.py`
  - `contracts_data_test.py`
  - `contracts_capture_test.py`
  - `contracts_cache_test.py`
  - `service_spies_test.py`
  - Helper `wsgi_client.py` is **not** a test module (no `_test` suffix).
- Always call handlers through `plugin.get_plugin_apps()[route]` (never bare methods).
- Reuse `tests/fixtures/plugin_factory.make_plugin` and `logdir_factory.create_two_host_session`. Do not invent a second plugin constructor.
- No real C++ convert: inject `xspace_fn` / patch `xspace_to_tool_names` as in integration tests.
- Cache happy path: assert `202` ACCEPTED only; do not wait on background thread beyond pool submit.
- Capture: error-only (TPU name + `_tf_profiler is None` → 500 JSON). Force `plugin._tf_profiler = None` so CI with TF installed still hits the contract.
- `CACHE_VERSION_FILE` is **not** a route; exclude from bijection expected set.
- Preserve production status/body contracts (copy from handlers below).
- Prefer TDD per task: add test → run → pass → commit.
- Working directory: repo root `/Users/samsi/xprof`.
- Full suite:
  ```bash
  cd /Users/samsi/xprof && python3 plugin/xprof/profile_plugin/tests/run_all_tests.py
  ```
- Focused:
  ```bash
  cd /Users/samsi/xprof && PYTHONPATH=plugin python3 -m unittest discover \
    -s plugin/xprof/profile_plugin/tests/http_contract -p '*_test.py' -v
  ```

## Production contracts (authoritative)

| Case | Status | Body / notes |
|------|--------|--------------|
| data missing (`data is None`) | 404 | plain text `"No Data"` (gzip via `respond`) |
| data success | 200 | gzip JSON |
| config | 200 | gzip JSON keys: `hideCaptureProfileButton`, `srcPathPrefix`, `enableTabNameLabel` |
| static missing file | 404 | text `"Fail to read the files."` |
| capture TPU without TF | 500 | JSON `error` containing TensorFlow not installed |
| cache GET | 405 | `"Method Not Allowed"` |
| cache POST valid session + tools | 202 | JSON `{"status":"ACCEPTED",...}` (patch `xspace_to_tool_names`) |
| hosts overview | 200 | list including `{"hostname":"ALL_HOSTS"}` |
| hosts missing run | 200 | `[]` (no crash) |
| runs with session | 200 | list containing session name |
| runs empty logdir | 200 | `[]` |

## File Structure

| Path | Responsibility |
|------|----------------|
| `plugin/xprof/profile_plugin/tests/http_contract/__init__.py` | Package marker |
| `plugin/xprof/profile_plugin/tests/http_contract/wsgi_client.py` | `call_route` + body decode helpers |
| `plugin/xprof/profile_plugin/tests/http_contract/route_table_test.py` | constants ↔ apps bijection + WSGI dry-run |
| `plugin/xprof/profile_plugin/tests/http_contract/contracts_static_test.py` | config happy + static 404 |
| `plugin/xprof/profile_plugin/tests/http_contract/contracts_runs_test.py` | runs happy + empty logdir |
| `plugin/xprof/profile_plugin/tests/http_contract/contracts_hosts_test.py` | hosts overview + missing run |
| `plugin/xprof/profile_plugin/tests/http_contract/contracts_data_test.py` | data 200 gzip + 404 No Data |
| `plugin/xprof/profile_plugin/tests/http_contract/contracts_capture_test.py` | TPU/no-TF 500 |
| `plugin/xprof/profile_plugin/tests/http_contract/contracts_cache_test.py` | GET 405 + POST 202 |
| `plugin/xprof/profile_plugin/tests/http_contract/service_spies_test.py` | data + runs service spies |
| `plugin/xprof/profile_plugin/README.md` | Mention `http_contract/` |

**Out of scope:** browser E2E, OpenAPI inventory, full per-tool matrices, capture happy path with real TF.

---

### Task 1: WSGI client helper + package init

**Files:**
- Create: `plugin/xprof/profile_plugin/tests/http_contract/__init__.py`
- Create: `plugin/xprof/profile_plugin/tests/http_contract/wsgi_client.py`

**Interfaces:**
- Produces:
  - `WsgiResult(status: str, status_code: int, headers: list[tuple[str, str]], body: bytes)`
  - `call_route(plugin, route, *, method='GET', path=None, query=None) -> WsgiResult`
  - `decode_body(result) -> bytes` (gunzip when `Content-Encoding: gzip`)
  - `json_body(result) -> Any`
  - `text_body(result) -> str`

- [ ] **Step 1: Create package init**

Create empty `plugin/xprof/profile_plugin/tests/http_contract/__init__.py`.

- [ ] **Step 2: Implement wsgi_client**

Create `plugin/xprof/profile_plugin/tests/http_contract/wsgi_client.py`:

```python
# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Minimal WSGI client for profile_plugin HTTP contract tests.

Always resolve apps from ``plugin.get_plugin_apps()`` so logging middleware and
``@wrappers.Request.application`` wiring are exercised.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from typing import Any, Mapping

from werkzeug.test import EnvironBuilder


@dataclass(frozen=True)
class WsgiResult:
  status: str
  status_code: int
  headers: list[tuple[str, str]]
  body: bytes


def _parse_status_code(status: str) -> int:
  return int(status.split(' ', 1)[0])


def call_route(
    plugin,
    route: str,
    *,
    method: str = 'GET',
    path: str | None = None,
    query: Mapping[str, str] | None = None,
) -> WsgiResult:
  """Invoke the registered WSGI app for ``route`` and return status/headers/body."""
  apps = plugin.get_plugin_apps()
  if route not in apps:
    raise KeyError(f'route not registered: {route!r}; known={sorted(apps)}')
  app = apps[route]
  env = EnvironBuilder(
      path=path if path is not None else route,
      method=method,
      query_string=dict(query or {}),
  ).get_environ()
  captured: dict[str, Any] = {}

  def start_response(status, headers, exc_info=None):
    captured['status'] = status
    captured['headers'] = list(headers)

  body = b''.join(app(env, start_response))
  status = captured.get('status', '500 Internal Server Error')
  return WsgiResult(
      status=status,
      status_code=_parse_status_code(status),
      headers=captured.get('headers', []),
      body=body,
  )


def header_value(result: WsgiResult, name: str) -> str | None:
  name_l = name.lower()
  for key, value in result.headers:
    if key.lower() == name_l:
      return value
  return None


def decode_body(result: WsgiResult) -> bytes:
  enc = header_value(result, 'Content-Encoding')
  if enc and 'gzip' in enc.lower():
    return gzip.decompress(result.body)
  return result.body


def json_body(result: WsgiResult) -> Any:
  return json.loads(decode_body(result))


def text_body(result: WsgiResult) -> str:
  return decode_body(result).decode('utf-8')
```

- [ ] **Step 3: Smoke-import**

```bash
cd /Users/samsi/xprof && PYTHONPATH=plugin python3 -c \
  "from xprof.profile_plugin.tests.http_contract.wsgi_client import call_route; print(call_route)"
```

Expected: prints the function repr; no ImportError.

- [ ] **Step 4: Commit**

```bash
git add plugin/xprof/profile_plugin/tests/http_contract/__init__.py \
  plugin/xprof/profile_plugin/tests/http_contract/wsgi_client.py
git commit -m "test(profile_plugin): add http_contract WSGI client helper"
```

---

### Task 2: Route table bijection + WSGI callable dry-run

**Files:**
- Create: `plugin/xprof/profile_plugin/tests/http_contract/route_table_test.py`

**Interfaces:**
- Consumes: `constants.*_ROUTE`, `ProfilePlugin.get_plugin_apps`
- Asserts: set equality of registered routes vs expected `*_ROUTE` list; each app is callable; dry WSGI call does not raise `TypeError` about positional args

- [ ] **Step 1: Write route_table_test.py**

```python
# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Route table integrity: constants.*_ROUTE ↔ get_plugin_apps() bijection."""

from __future__ import annotations

import tempfile
import unittest

from xprof.profile_plugin import constants
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import call_route

# All frontend routes registered in plugin.get_plugin_apps().
# CACHE_VERSION_FILE is intentionally NOT a route.
EXPECTED_ROUTES: frozenset[str] = frozenset({
    constants.BASE_ROUTE,
    constants.INDEX_JS_ROUTE,
    constants.INDEX_HTML_ROUTE,
    constants.BUNDLE_JS_ROUTE,
    constants.STYLES_CSS_ROUTE,
    constants.MATERIALICONS_WOFF2_ROUTE,
    constants.TRACE_VIEWER_INDEX_HTML_ROUTE,
    constants.TRACE_VIEWER_INDEX_JS_ROUTE,
    constants.TRACE_VIEWER_V2_JS_ROUTE,
    constants.TRACE_VIEWER_V2_WASM_ROUTE,
    constants.ZONE_JS_ROUTE,
    constants.RUNS_ROUTE,
    constants.RUN_TOOLS_ROUTE,
    constants.HOSTS_ROUTE,
    constants.DATA_ROUTE,
    constants.DATA_CSV_ROUTE,
    constants.VERSION_ROUTE,
    constants.HLO_MODULE_LIST_ROUTE,
    constants.CAPTURE_ROUTE,
    constants.LOCAL_ROUTE,
    constants.CONFIG_ROUTE,
    constants.GENERATE_CACHE_ROUTE,
})


class RouteTableTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._td = tempfile.TemporaryDirectory()
    create_two_host_session(self._td.name)
    self.plugin = make_plugin(self._td.name)

  def tearDown(self):
    self._td.cleanup()
    super().tearDown()

  def test_route_table_bijection(self):
    apps = self.plugin.get_plugin_apps()
    actual = frozenset(apps.keys())
    self.assertEqual(
        actual,
        EXPECTED_ROUTES,
        msg=(
            f'missing={sorted(EXPECTED_ROUTES - actual)}; '
            f'extra={sorted(actual - EXPECTED_ROUTES)}'
        ),
    )
    self.assertNotIn(constants.CACHE_VERSION_FILE, apps)

  def test_each_app_is_callable(self):
    apps = self.plugin.get_plugin_apps()
    for route, app in sorted(apps.items()):
      self.assertTrue(callable(app), msg=f'{route} app not callable')

  def test_each_app_accepts_wsgi_call_without_typeerror(self):
    """Decorator regression: bare method needs (environ, start_response)."""
    for route in sorted(EXPECTED_ROUTES):
      with self.subTest(route=route):
        try:
          result = call_route(self.plugin, route, method='GET', query={})
        except TypeError as err:
          self.fail(
              f'{route} raised TypeError (likely lost @Request.application): '
              f'{err}'
          )
        # Any HTTP status is fine; we only care the WSGI signature works.
        self.assertIsInstance(result.status_code, int)
        self.assertGreaterEqual(result.status_code, 100)


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 2: Run focused**

```bash
cd /Users/samsi/xprof && PYTHONPATH=plugin python3 -m unittest \
  xprof.profile_plugin.tests.http_contract.route_table_test -v
```

Expected: all 3 tests OK. If bijection fails, fix expected set to match `plugin.py` `get_plugin_apps` (do not drop routes).

- [ ] **Step 3: Commit**

```bash
git add plugin/xprof/profile_plugin/tests/http_contract/route_table_test.py
git commit -m "test(profile_plugin): route table bijection for get_plugin_apps"
```

---

### Task 3: Static + config contracts

**Files:**
- Create: `plugin/xprof/profile_plugin/tests/http_contract/contracts_static_test.py`

**Production anchors:**
- `static_api.config_route` → 200 JSON with `hideCaptureProfileButton`, `srcPathPrefix`
- `static_api.static_file_route` missing → 404 `"Fail to read the files."`

- [ ] **Step 1: Write contracts_static_test.py**

```python
# Copyright 2026 The XProf Authors. All Rights Reserved.
"""HTTP contracts: static assets + /config."""

from __future__ import annotations

import tempfile
import unittest

from xprof.profile_plugin.constants import CONFIG_ROUTE, INDEX_JS_ROUTE
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import (
    call_route,
    header_value,
    json_body,
    text_body,
)


class ContractsStaticTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._td = tempfile.TemporaryDirectory()
    create_two_host_session(self._td.name)
    self.plugin = make_plugin(self._td.name)

  def tearDown(self):
    self._td.cleanup()
    super().tearDown()

  def test_config_happy_json_keys(self):
    result = call_route(self.plugin, CONFIG_ROUTE)
    self.assertEqual(result.status_code, 200)
    self.assertEqual(header_value(result, 'Content-Encoding'), 'gzip')
    payload = json_body(result)
    self.assertIn('hideCaptureProfileButton', payload)
    self.assertIn('srcPathPrefix', payload)
    self.assertIn('enableTabNameLabel', payload)
    self.assertIsInstance(payload['hideCaptureProfileButton'], bool)
    self.assertIsInstance(payload['srcPathPrefix'], str)

  def test_static_missing_file_404(self):
    # Same handler as index.js; path basename is the file looked up under static/.
    result = call_route(
        self.plugin,
        INDEX_JS_ROUTE,
        path='/this_file_does_not_exist_xyz.js',
    )
    self.assertEqual(result.status_code, 404)
    self.assertEqual(text_body(result), 'Fail to read the files.')


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 2: Run**

```bash
cd /Users/samsi/xprof && PYTHONPATH=plugin python3 -m unittest \
  xprof.profile_plugin.tests.http_contract.contracts_static_test -v
```

Expected: OK (2 tests).

- [ ] **Step 3: Commit**

```bash
git add plugin/xprof/profile_plugin/tests/http_contract/contracts_static_test.py
git commit -m "test(profile_plugin): HTTP contracts for config and static 404"
```

---

### Task 4: Runs + hosts contracts

**Files:**
- Create: `plugin/xprof/profile_plugin/tests/http_contract/contracts_runs_test.py`
- Create: `plugin/xprof/profile_plugin/tests/http_contract/contracts_hosts_test.py`

**Production anchors:**
- `runs_route` → 200 JSON list; with session → contains session; empty logdir → `[]`
- `hosts_route` overview → 200 list with `ALL_HOSTS`; missing run → 200 `[]`

- [ ] **Step 1: Write contracts_runs_test.py**

```python
# Copyright 2026 The XProf Authors. All Rights Reserved.
"""HTTP contracts: /runs."""

from __future__ import annotations

import tempfile
import unittest

from xprof.profile_plugin.constants import RUNS_ROUTE
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import (
    call_route,
    json_body,
)


class ContractsRunsTest(unittest.TestCase):

  def test_runs_contains_session(self):
    with tempfile.TemporaryDirectory() as logdir:
      session = 'session_a'
      create_two_host_session(logdir, session=session)
      plugin = make_plugin(logdir)
      result = call_route(plugin, RUNS_ROUTE)
      self.assertEqual(result.status_code, 200)
      runs = json_body(result)
      self.assertIsInstance(runs, list)
      self.assertIn(session, runs)

  def test_runs_empty_logdir(self):
    with tempfile.TemporaryDirectory() as logdir:
      plugin = make_plugin(logdir)
      result = call_route(plugin, RUNS_ROUTE)
      self.assertEqual(result.status_code, 200)
      self.assertEqual(json_body(result), [])


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 2: Write contracts_hosts_test.py**

```python
# Copyright 2026 The XProf Authors. All Rights Reserved.
"""HTTP contracts: /hosts."""

from __future__ import annotations

import tempfile
import unittest

from xprof.profile_plugin.constants import ALL_HOSTS, HOSTS_ROUTE
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import (
    call_route,
    json_body,
)


class ContractsHostsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._td = tempfile.TemporaryDirectory()
    self.logdir = self._td.name
    self.session = 'session_a'
    create_two_host_session(self.logdir, session=self.session)
    self.plugin = make_plugin(self.logdir)

  def tearDown(self):
    self._td.cleanup()
    super().tearDown()

  def test_hosts_overview_includes_all_hosts(self):
    result = call_route(
        self.plugin,
        HOSTS_ROUTE,
        query={'run': self.session, 'tag': 'overview_page'},
    )
    self.assertEqual(result.status_code, 200)
    hosts = json_body(result)
    self.assertIsInstance(hosts, list)
    self.assertIn({'hostname': ALL_HOSTS}, hosts)

  def test_hosts_missing_run_returns_empty_list(self):
    """Missing run_dir logs a warning and returns [] (200), no crash."""
    result = call_route(
        self.plugin,
        HOSTS_ROUTE,
        query={'run': 'no_such_run', 'tag': 'overview_page'},
    )
    self.assertEqual(result.status_code, 200)
    self.assertEqual(json_body(result), [])


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 3: Run**

```bash
cd /Users/samsi/xprof && PYTHONPATH=plugin python3 -m unittest \
  xprof.profile_plugin.tests.http_contract.contracts_runs_test \
  xprof.profile_plugin.tests.http_contract.contracts_hosts_test -v
```

Expected: OK (4 tests).

- [ ] **Step 4: Commit**

```bash
git add plugin/xprof/profile_plugin/tests/http_contract/contracts_runs_test.py \
  plugin/xprof/profile_plugin/tests/http_contract/contracts_hosts_test.py
git commit -m "test(profile_plugin): HTTP contracts for runs and hosts"
```

---

### Task 5: Data contracts

**Files:**
- Create: `plugin/xprof/profile_plugin/tests/http_contract/contracts_data_test.py`

**Production anchors (`data_api.data_route`):**
- success: 200, gzip JSON
- missing: 404, body `"No Data"`

- [ ] **Step 1: Write contracts_data_test.py**

```python
# Copyright 2026 The XProf Authors. All Rights Reserved.
"""HTTP contracts: /data happy + 404 No Data."""

from __future__ import annotations

import json
import tempfile
import unittest

from xprof.profile_plugin.constants import DATA_ROUTE
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import (
    call_route,
    header_value,
    json_body,
    text_body,
)


class ContractsDataTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._td = tempfile.TemporaryDirectory()
    self.logdir = self._td.name
    self.session = 'session_a'
    create_two_host_session(self.logdir, session=self.session)

  def tearDown(self):
    self._td.cleanup()
    super().tearDown()

  def _query(self):
    return {
        'run': self.session,
        'tag': 'overview_page',
        'host': 'ALL_HOSTS',
    }

  def test_data_success_gzip_json(self):
    def ok_convert(paths, tool, params):
      return json.dumps({'ok': True, 'tool': tool}, sort_keys=True), 'application/json'

    plugin = make_plugin(self.logdir, xspace_fn=ok_convert)
    result = call_route(plugin, DATA_ROUTE, query=self._query())
    self.assertEqual(result.status_code, 200)
    self.assertEqual(header_value(result, 'Content-Encoding'), 'gzip')
    self.assertEqual(json_body(result), {'ok': True, 'tool': 'overview_page'})

  def test_data_missing_404_no_data(self):
    def empty_convert(paths, tool, params):
      return None, 'application/json'

    plugin = make_plugin(self.logdir, xspace_fn=empty_convert)
    result = call_route(plugin, DATA_ROUTE, query=self._query())
    self.assertEqual(result.status_code, 404)
    self.assertEqual(text_body(result), 'No Data')


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 2: Run**

```bash
cd /Users/samsi/xprof && PYTHONPATH=plugin python3 -m unittest \
  xprof.profile_plugin.tests.http_contract.contracts_data_test -v
```

Expected: OK (2 tests).

- [ ] **Step 3: Commit**

```bash
git add plugin/xprof/profile_plugin/tests/http_contract/contracts_data_test.py
git commit -m "test(profile_plugin): HTTP contracts for /data happy and 404"
```

---

### Task 6: Capture + cache contracts

**Files:**
- Create: `plugin/xprof/profile_plugin/tests/http_contract/contracts_capture_test.py`
- Create: `plugin/xprof/profile_plugin/tests/http_contract/contracts_cache_test.py`

**Production anchors:**
- capture: `is_tpu_name=true` + `_tf_profiler is None` → 500 JSON error about TensorFlow
- cache GET → 405 `"Method Not Allowed"`
- cache POST with `session_path` + `tools=overview_page` + patched tool names → 202 `status=ACCEPTED`

- [ ] **Step 1: Write contracts_capture_test.py**

```python
# Copyright 2026 The XProf Authors. All Rights Reserved.
"""HTTP contracts: /capture_profile error path (no TF)."""

from __future__ import annotations

import tempfile
import unittest

from xprof.profile_plugin.constants import CAPTURE_ROUTE
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import (
    call_route,
    json_body,
)


class ContractsCaptureTest(unittest.TestCase):

  def test_tpu_capture_without_tf_returns_500_json_error(self):
    with tempfile.TemporaryDirectory() as logdir:
      create_two_host_session(logdir)
      plugin = make_plugin(logdir)
      # Force error contract even when TensorFlow is installed in the env.
      plugin._tf_profiler = None
      result = call_route(
          plugin,
          CAPTURE_ROUTE,
          query={
              'service_addr': 'my-tpu',
              'is_tpu_name': 'true',
              'duration': '1000',
          },
      )
      self.assertEqual(result.status_code, 500)
      payload = json_body(result)
      self.assertIn('error', payload)
      self.assertIn('TensorFlow', payload['error'])
      self.assertIn('not installed', payload['error'].lower())


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 2: Write contracts_cache_test.py**

```python
# Copyright 2026 The XProf Authors. All Rights Reserved.
"""HTTP contracts: /generate_cache GET 405 + POST 202."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from xprof.profile_plugin.constants import GENERATE_CACHE_ROUTE
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import (
    call_route,
    json_body,
    text_body,
)


class ContractsCacheTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._td = tempfile.TemporaryDirectory()
    self.logdir = self._td.name
    self.session = 'session_a'
    self.run_dir = create_two_host_session(self.logdir, session=self.session)
    self.plugin = make_plugin(self.logdir)

  def tearDown(self):
    self._td.cleanup()
    super().tearDown()

  def test_generate_cache_get_method_not_allowed(self):
    result = call_route(self.plugin, GENERATE_CACHE_ROUTE, method='GET')
    self.assertEqual(result.status_code, 405)
    self.assertEqual(text_body(result), 'Method Not Allowed')

  def test_generate_cache_post_accepted(self):
    with mock.patch(
        'xprof.convert.raw_to_tool_data.xspace_to_tool_names',
        return_value=['overview_page', 'trace_viewer@'],
    ):
      result = call_route(
          self.plugin,
          GENERATE_CACHE_ROUTE,
          method='POST',
          query={
              'session_path': self.run_dir,
              'tools': 'overview_page',
          },
      )
    self.assertEqual(result.status_code, 202, msg=text_body(result))
    payload = json_body(result)
    self.assertEqual(payload['status'], 'ACCEPTED')
    self.assertIn('message', payload)


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 3: Run**

```bash
cd /Users/samsi/xprof && PYTHONPATH=plugin python3 -m unittest \
  xprof.profile_plugin.tests.http_contract.contracts_capture_test \
  xprof.profile_plugin.tests.http_contract.contracts_cache_test -v
```

Expected: OK (3 tests). If POST 202 fails with 400 about tools, confirm patch target still hits `run_tools_imp` / catalog path used by the plugin (same as `plugin_integration_test.test_generate_cache_accepts_post`).

- [ ] **Step 4: Commit**

```bash
git add plugin/xprof/profile_plugin/tests/http_contract/contracts_capture_test.py \
  plugin/xprof/profile_plugin/tests/http_contract/contracts_cache_test.py
git commit -m "test(profile_plugin): HTTP contracts for capture and cache"
```

---

### Task 7: Service spies (data + runs)

**Files:**
- Create: `plugin/xprof/profile_plugin/tests/http_contract/service_spies_test.py`

**Spy targets:**
1. **data:** patch `plugin._tool_data.get_tool_data` → return sentinel `ToolResult`; assert status/body and called once with `ToolRequest`-like object (`run`/`tool` from query).
2. **runs:** patch `plugin._run_discovery.iter_frontend_runs` → yield fixed names; assert `/runs` JSON equals `sorted(names, reverse=True)` as `runs_imp` does.

- [ ] **Step 1: Write service_spies_test.py**

```python
# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Service spies: /data and /runs still delegate to domain services."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from xprof.profile_plugin.constants import DATA_ROUTE, RUNS_ROUTE
from xprof.profile_plugin.models import ToolRequest, ToolResult
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import (
    call_route,
    json_body,
)


class ServiceSpiesTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._td = tempfile.TemporaryDirectory()
    self.logdir = self._td.name
    self.session = 'session_a'
    create_two_host_session(self.logdir, session=self.session)
    self.plugin = make_plugin(self.logdir)

  def tearDown(self):
    self._td.cleanup()
    super().tearDown()

  def test_data_route_calls_tool_data_service_once(self):
    sentinel = ToolResult(
        data='{"spy": true}',
        content_type='application/json',
    )
    with mock.patch.object(
        self.plugin._tool_data,
        'get_tool_data',
        return_value=sentinel,
    ) as spy:
      result = call_route(
          self.plugin,
          DATA_ROUTE,
          query={
              'run': self.session,
              'tag': 'overview_page',
              'host': 'ALL_HOSTS',
          },
      )
    self.assertEqual(result.status_code, 200)
    self.assertEqual(json_body(result), {'spy': True})
    spy.assert_called_once()
    req = spy.call_args.args[0]
    self.assertIsInstance(req, ToolRequest)
    self.assertEqual(req.run, self.session)
    self.assertEqual(req.tool, 'overview_page')

  def test_runs_route_uses_run_discovery_sorted_reverse(self):
    fixed = ['run_b', 'run_a', 'run_c']
    with mock.patch.object(
        self.plugin._run_discovery,
        'iter_frontend_runs',
        return_value=iter(fixed),
    ) as spy:
      result = call_route(self.plugin, RUNS_ROUTE)
    self.assertEqual(result.status_code, 200)
    self.assertEqual(json_body(result), sorted(fixed, reverse=True))
    spy.assert_called_once()


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 2: Run**

```bash
cd /Users/samsi/xprof && PYTHONPATH=plugin python3 -m unittest \
  xprof.profile_plugin.tests.http_contract.service_spies_test -v
```

Expected: OK (2 tests). If `get_tool_data` is not called (e.g. path short-circuit), re-check that `data_impl` still delegates to `self._tool_data.get_tool_data`.

- [ ] **Step 3: Commit**

```bash
git add plugin/xprof/profile_plugin/tests/http_contract/service_spies_test.py
git commit -m "test(profile_plugin): service spies for /data and /runs"
```

---

### Task 8: README + full suite green

**Files:**
- Modify: `plugin/xprof/profile_plugin/README.md`

- [ ] **Step 1: Update Tests section**

In `plugin/xprof/profile_plugin/README.md`, update the Tests section to:

```markdown
## Tests

```bash
python3 plugin/xprof/profile_plugin/tests/run_all_tests.py
```

- Unit: `tests/unit/`
- Golden / integration: `tests/golden_*`, `tests/plugin_integration_test.py`
- HTTP contracts (route table + lean per-API happy/error via `get_plugin_apps()`): `tests/http_contract/`
```

- [ ] **Step 2: Run full package suite (with stubs)**

```bash
cd /Users/samsi/xprof && python3 plugin/xprof/profile_plugin/tests/run_all_tests.py
```

Expected: all tests pass, including newly discovered `http_contract/*_test.py` modules. Confirm discovery by watching runner list names like `route_table_test`, `contracts_data_test`, `service_spies_test`.

- [ ] **Step 3: Focused discovery check**

```bash
cd /Users/samsi/xprof && PYTHONPATH=plugin python3 -m unittest discover \
  -s plugin/xprof/profile_plugin/tests/http_contract -p '*_test.py' -v
```

Expected: **≥14** tests (route table 3 + static 2 + runs 2 + hosts 2 + data 2 + capture 1 + cache 2 + spies 2).

- [ ] **Step 4: Final commit**

```bash
git add plugin/xprof/profile_plugin/README.md \
  plugin/xprof/profile_plugin/tests/http_contract/
git commit -m "test(profile_plugin): document http_contract suite; full suite green"
```

(If earlier tasks already committed files, this commit may only touch README.)

---

## Self-review (spec coverage)

| Spec goal | Plan coverage |
|-----------|---------------|
| Route table bijection `*_ROUTE` ↔ `get_plugin_apps()` | Task 2 |
| Per API group ≥1 happy + ≥1 error | Tasks 3–6 |
| Service spies data + runs | Task 7 |
| Discovered by `run_all_tests.py` | `*_test.py` naming; Task 8 |
| No browser/server E2E | Explicit non-goal |
| Reuse fixtures | All tasks |
| Real WSGI apps from `get_plugin_apps` | `wsgi_client.call_route` |
| README | Task 8 |
| Discovery naming vs design | Design `test_*.py` corrected to `*_test.py` |

## Risks / notes for implementers

1. **Naming:** Design doc used `test_contracts_*.py`; this plan **deviates on purpose** so discover works. Do not rename back.
2. **Hosts missing run:** production returns **200 + `[]`**, not 4xx.
3. **Capture with TF installed:** always set `plugin._tf_profiler = None` for the error contract.
4. **JSON string vs dict in `ToolResult`:** `respond` only auto-`json.dumps` dict/list/set/tuple; string payloads must already be JSON text.
5. **`run_all_tests` stubs:** prefer the package runner for full green (native deps stubbed).
6. **Overlap with golden/integration:** intentional lean duplicates of status/body; spies and bijection are new value.

## Success checklist

- [ ] `tests/http_contract/` package with 8 `*_test.py` modules + `wsgi_client.py`
- [ ] Route set exact match to `get_plugin_apps` (22 routes)
- [ ] All 6 API groups covered (capture error-only OK)
- [ ] Spies for `/data` and `/runs` only
- [ ] `python3 plugin/xprof/profile_plugin/tests/run_all_tests.py` green
- [ ] README mentions `http_contract/`
