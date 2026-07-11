# `xprof.profile_plugin` layout

The HTTP TensorBoard / XProf **frontend** is split so each UI API lives in one place.

## Layers

```text
Browser UI
    │  HTTP
    ▼
api/*          ← one module per frontend concern (routes + request/response)
    │
plugin.py      ← construct services + register route table only
    │
services/*     ← pure domain (no Werkzeug): sessions, hosts, tool_data, runs
tools/*        ← tool taxonomy, filenames, option builders
cache/*        ← tools-list cache, result-cache version policy
http/*         ← respond(), logging middleware, parse helpers
```

## Frontend APIs (`api/`)

| Module | Routes / UI surface |
|--------|---------------------|
| `api/static_api.py` | `/`, static JS/CSS/WASM, `/config` |
| `api/runs_api.py` | `/runs`, `/run_tools` (session list + tools for a run) |
| `api/hosts_api.py` | `/hosts` (hosts for a run+tool) |
| `api/data_api.py` | `/data`, `/data_csv`, HLO module list |
| `api/capture_api.py` | `/capture_profile` |
| `api/cache_api.py` | `/generate_cache` (background warm-up) |

Handlers are **mixins** composed into `ProfilePlugin` so existing call sites
(`plugin.data_impl`, `plugin.runs_imp`, …) keep working.

## Domain services (`services/`)

| Module | Responsibility |
|--------|----------------|
| `sessions.py` | `session_path` / `run_path` / logdir resolution |
| `hosts.py` | Host selection for convert |
| `tool_data.py` | Convert orchestration + cache version policy |
| `runs.py` | Logdir walk → frontend run names + tools-of-run |
| `counter_names.py` | `perf_counters` + `names_only` special case |

## Where to change what

- **New query option for a tool** → `tools/options/<family>.py`
- **Host selection rules** → `services/hosts.py`
- **Session path rules** → `services/sessions.py`
- **Convert / tool data** → `services/tool_data.py`
- **HTTP status / body for a UI call** → matching `api/*_api.py` + `http/respond.py`
- **Route table / wiring** → `plugin.py` only

## Tests

```bash
python3 plugin/xprof/profile_plugin/tests/run_all_tests.py
```

- Unit: `tests/unit/`
- Golden / integration: `tests/golden_*`, `tests/plugin_integration_test.py`
- HTTP contracts (route table + lean per-API happy/error via `get_plugin_apps()`): `tests/http_contract/`
