# Design: Lean HTTP contract tests for profile_plugin APIs

**Date:** 2026-07-11  
**Status:** Approved direction (option 1 — lean)  
**Branch:** `refactor/profile-plugin-checks-and-thin-facade`  
**Scope:** Prove frontend route wiring after the `api/*` mixin split without a heavy E2E suite.

## 1. Problem

After splitting handlers into `api/static_api.py`, `runs_api.py`, `hosts_api.py`, `data_api.py`, `capture_api.py`, and `cache_api.py`, unit tests cover domain services well, but the **HTTP surface** is only spot-checked. Failure modes that unit tests miss:

- Route missing from `get_plugin_apps()`
- Lost `@wrappers.Request.application` (WSGI TypeError: 2 vs 3 args)
- Wrong status mapping on the thin API layer
- Handler no longer delegating to the intended service

There is no HTTP **redirect** matrix; “redirection” here means **routing** (path → handler → status/body).

## 2. Goals

1. **Route table integrity:** bijection between `constants.*_ROUTE` used by the UI and keys in `get_plugin_apps()`.
2. **Lean contract coverage:** every frontend API group has ≥1 happy path and ≥1 error path via real WSGI apps from `get_plugin_apps()`.
3. **Two service spies:** `/data` and `/runs` prove the façade still calls domain services (not a parallel code path).
4. **Discoverable:** suite runs under existing `tests/run_all_tests.py`.
5. **No flaky browser/server E2E** in this tranche.

## 3. Non-goals

- Full per-tool query-param matrices (stay in unit tests).
- Live TensorBoard server or Angular UI.
- Expanding goldens beyond what’s needed to support lean cases.
- Capture/TPU happy path requiring real TensorFlow (use “TF missing → 500 JSON” as error case).

## 4. Architecture

```text
tests/http_contract/
  wsgi_client.py           # call_route(plugin, path|route, method, query) -> Result
  test_route_table.py      # constants ↔ get_plugin_apps bijection + WSGI callable
  test_contracts_static.py
  test_contracts_runs.py
  test_contracts_hosts.py
  test_contracts_data.py
  test_contracts_capture.py
  test_contracts_cache.py
  test_service_spies.py    # data + runs spies only
```

### 4.1 WSGI client

Reuse patterns from `plugin_integration_test.py`:

- Build plugin via existing `fixtures/plugin_factory` or the same `_make_plugin` helpers.
- Resolve app from `plugin.get_plugin_apps()[route]`.
- Call with Werkzeug `EnvironBuilder` / `Client` **or** minimal environ + `start_response` capture.
- Return `status_code`, `headers`, `body` (bytes).

**Must** go through `get_plugin_apps()` (logged apps), not bare methods, so decorator/middleware wiring is exercised.

### 4.2 Route table test

- Collect all `*_ROUTE` string constants from `constants.py` that are intended for `get_plugin_apps` (allowlist internal-only if any).
- Assert `set(apps.keys()) == expected_routes` (or expected ⊆ apps and no unknown extras without allowlist).
- For each app, assert `callable` and a dry WSGI call with empty-ish environ does not raise `TypeError` about positional args (decorator regression).

### 4.3 Lean cases (per API group)

| Group | Happy | Error |
|-------|-------|-------|
| static | GET config → 200 JSON keys | static missing file → 404 text |
| runs | GET runs → 200 list containing session | (optional) empty logdir → 200 `[]` |
| hosts | GET hosts for overview → 200 list | missing run/tool handled without crash (document actual status) |
| data | GET data with mock convert → 200 JSON | missing data → 404 “No Data” |
| capture | — | TPU name without TF → 500 JSON `error` |
| cache | POST generate_cache accepted or documented status | GET → 405 |

Exact status strings should match current production handlers (copy from integration/golden where they exist).

### 4.4 Service spies

1. **data:** patch `ToolDataService.get_tool_data` (or plugin `_tool_data.get_tool_data`) to return a sentinel `ToolResult`; assert response body/status reflects it and spy was called once with a `ToolRequest`-like object.
2. **runs:** patch `RunDiscovery.iter_frontend_runs` (or path used by `generate_runs`) to yield fixed names; assert `/runs` JSON equals sorted reverse list as `runs_imp` does.

### 4.5 Fixtures

Prefer `tests/fixtures/plugin_factory.py` and `logdir_factory.py` if they already create xplane sessions; otherwise thin wrappers in `http_contract` that call those factories. Do not invent a second plugin constructor.

## 5. Success metrics

| Metric | Target |
|--------|--------|
| `run_all_tests.py` | All pass including new http_contract |
| API groups with happy+error | 6/6 (capture may be error-only if happy needs TF) |
| Route table bijection | Yes |
| Service spies | `/data` + `/runs` |
| New flaky tests | 0 |

## 6. Implementation notes

- Keep tests free of real C++ convert (inject `xspace_to_tool_data_fn` like integration tests).
- Cache route may return 202 ACCEPTED — assert that when session_path valid; don’t wait on background thread beyond pool submit (use plugin’s test executor).
- Update `profile_plugin/README.md` “Tests” section to mention `http_contract/`.

## 7. Out of scope follow-ups

- Medium suite: full status matrix for data/cache.
- OpenAPI param inventory.
- Browser E2E.
