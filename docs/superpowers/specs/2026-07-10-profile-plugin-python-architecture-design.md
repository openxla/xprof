# Design: Profile Plugin Python Architecture

**Date:** 2026-07-10  
**Status:** Draft for review  
**Related commit:** [bdb4b675](https://github.com/Sam-Si/xprof/commit/bdb4b675) (scaffold package split)  
**Scope:** Python under `plugin/xprof/` (profile plugin package, server, CLI glue, convert Python surface, profile_io). C++ convert/pywrap remains a boundary, not rewritten.

## 1. Problem

### 1.1 Starting point before any split

`plugin/xprof/profile_plugin.py` was a ~1.9k-line module that mixed:

- HTTP routing and responses (Werkzeug)
- Session/run discovery (logdir, `session_path`, `run_path`, GCS)
- Host selection policies
- Tool-specific request parameter packing
- Convert orchestration (XPlane → tool JSON)
- Tools-list and result-cache policy
- Remote capture (optional TensorFlow)
- Static asset serving
- Background cache warm-up

### 1.2 What commit `bdb4b675` achieved

It introduced package `xprof.profile_plugin` and extracted useful leaf modules:

| Module | Responsibility |
|--------|----------------|
| `tools/registry.py`, `filenames.py`, `catalog.py` | Tool taxonomy, filenames, available tools |
| `http/respond.py`, `request_params.py`, `logging_middleware.py` | HTTP edge |
| `cache/tools_cache.py`, `result_cache_policy.py` | Tools list cache + version policy helpers |
| `tensorflow_bridge.py` | Optional TF TPU resolution |
| `constants.py` | Routes and shared names |

Public import path remained: `from xprof import profile_plugin`.

### 1.3 What `bdb4b675` did not achieve

| Issue | Detail |
|-------|--------|
| God object remains | `plugin.py` still ~1309 LOC, ~35 methods |
| Hot paths unextracted | `data_impl`, `_generate_cache_impl`, `generate_runs`, `_get_valid_hosts` still on the class |
| Complexity not deleted | Net line count increased; control flow mostly relocated |
| Public API too wide | `__init__.py` re-exports dozens of symbols |
| BUILD ships tests | `glob(["profile_plugin/**/*.py"])` includes `tests/` in prod library |
| CLI not aligned | CLI often bypasses plugin domain and reimplements concepts |
| Special cases grow mid-pipeline | e.g. `perf_counters` + `names_only` inside `data_impl` |

**Conclusion:** `bdb4b675` is a valid scaffold (Phase 0.5). This design defines the finished architecture and the path to get there.

## 2. Goals and non-goals

### 2.1 Goals

1. **Thin HTTP façade:** `ProfilePlugin` is route wiring + dependency construction only (target ≤ 250–400 LOC).
2. **Pure domain services:** Session discovery, host selection, tool data, capture, cache generation have **no Werkzeug imports**.
3. **Typed request/response models:** Replace unbounded nested dicts for tool options with explicit models + builders.
4. **Single home per concern:** Host policy, cache version policy, tool options, discovery.
5. **Injectable boundaries:** Convert and filesystem accessed via injected ports (testable without C++).
6. **Shared domain for UI and CLI:** CLI tools that mirror UI analysis call the same services.
7. **Safe packaging:** Production libraries and wheels do not ship tests.
8. **Behavior preserved** relative to current master unless a deliberate, documented behavior change is approved.

### 2.2 Non-goals

- Rewriting C++ `xprof/convert` processors or pywrap.
- Redesigning the frontend Angular app.
- Introducing a DI framework or new web framework.
- Changing HTTP paths or query parameter names (unless a follow-up migration is explicit).
- Unifying every CLI tool in the first implementation tranche (alignment is phased; highest-traffic tools first).

### 2.3 Success metrics

| Metric | Target |
|--------|--------|
| `plugin.py` lines | ≤ 400 |
| Werkzeug imports outside `http/` + `plugin.py` | 0 |
| Largest service module | ≤ ~300 LOC |
| Adding a tool-specific query option | Touch only `tools/options/<family>.py` (+ registry entry) |
| Prod `py_library` includes test sources | No |
| Public `__all__` names | Small, intentional (order of 10–15 max) |
| Unit tests for hosts/sessions/options/tool_data | No full TB plugin required |

## 3. Target architecture

### 3.1 Layering

```text
┌─────────────────────────────────────────────────────────────┐
│  HTTP edge (Werkzeug)                                        │
│  profile_plugin/plugin.py, http/*                            │
│  Parse Request → models; map exceptions → respond()          │
└────────────────────────────┬────────────────────────────────┘
                             │ ToolRequest / SessionRef / ...
┌────────────────────────────▼────────────────────────────────┐
│  Domain services (pure Python)                               │
│  services/sessions, hosts, tool_data, options,               │
│  cache_generation, capture, hlo_modules, static_assets       │
└────────────┬─────────────────────────────┬──────────────────┘
             │                             │
             ▼                             ▼
┌────────────────────────┐    ┌────────────────────────────────┐
│  profile_io            │    │  convert ports (Python)        │
│  ProfileFileSystem     │    │  raw_to_tool_data / pywrap     │
└────────────────────────┘    └────────────────────────────────┘
```

CLI and HTTP are both **adapters** over domain services.

### 3.2 Package layout (end state)

```text
plugin/xprof/profile_plugin/
├── __init__.py                 # minimal public API
├── plugin.py                   # ProfilePlugin façade only
├── deps.py                     # Protocols: ConvertPort, FileSystemFactory, VersionProvider
├── models.py                   # SessionRef, ToolRequest, HostSelection, ToolResult
├── constants.py
├── logging_config.py
├── tools/
│   ├── registry.py
│   ├── filenames.py
│   ├── catalog.py
│   └── options/                # per-tool option builders
│       ├── __init__.py         # registry map tool → builder
│       ├── base.py
│       ├── trace.py
│       ├── graph.py
│       ├── memory.py
│       └── default.py
├── services/
│   ├── sessions.py
│   ├── hosts.py
│   ├── tool_data.py
│   ├── hlo_modules.py
│   ├── capture.py
│   ├── cache_generation.py
│   └── static_assets.py
├── http/
│   ├── respond.py
│   ├── request_params.py
│   ├── logging_middleware.py
│   └── parse_request.py        # Werkzeug → ToolRequest
├── cache/
│   ├── tools_cache.py
│   └── result_cache_policy.py  # sole owner of cache_version.txt read/write
├── tensorflow_bridge.py
└── tests/
    ├── unit/
    └── integration/
```

Related outside the package:

- `profile_io.py` — filesystem abstraction (unchanged contract preferred).
- `convert/raw_to_tool_data.py` — convert port implementation details; document option keys.
- `cli/internal/*` — adopt services over time.
- `server.py` — share logging config; remain thin.
- `profile_plugin_loader.py` — constructs plugin with default deps.

### 3.3 Core models

```text
SessionRef
  frontend_run: str
  directory: str          # absolute session path (local or gs://)

ToolRequest
  run: str
  tool: str
  host: str | None
  hosts: tuple[str, ...]  # multi-host selection
  use_saved_result: bool
  # tool-specific structured fields filled by builders, not raw free-form soup
  # residual raw_args only for truly open-ended keys during migration

HostSelection
  selected_hosts: tuple[str, ...]
  asset_paths: tuple[Path, ...]

ToolResult
  data: bytes | str | None
  content_type: str
  content_encoding: str | None
```

### 3.4 Dependency ports

```text
ConvertPort
  xspace_to_tool_data(paths, tool, options) -> (data, content_type)
  xspace_to_tool_names(paths) -> list[str]
  json_to_csv_string(data) -> str

FileSystemFactory
  get(path) -> ProfileFileSystem

VersionProvider
  __version__: str

CapturePort
  trace(service_addr, logdir, workers, ...) -> None
```

Production factory (loader/server) supplies real implementations. Tests supply fakes.

Lazy import of convert/pywrap is allowed **only inside the production factory**, not scattered through service methods.

### 3.5 Tool options strategy

Each tool family has a builder implementing:

```text
build(request: ToolRequest) -> Mapping[str, Any]
```

Registry maps tool name → builder. Unknown tools use `DefaultOptionsBuilder` (common fields only).

**Special handlers** that are not “convert this session” (e.g. `perf_counters` + `names_only`) register as separate handlers and never run through the XPlane convert path.

### 3.6 Data flow (happy path)

```text
GET /data?run=R&tag=overview_page&host=ALL_HOSTS
  → http.parse_request → ToolRequest
  → sessions.resolve(run, session_path|run_path|logdir) → SessionRef
  → cache_policy.decide_use_saved(session, requested_flag)
  → options.build(ToolRequest)
  → hosts.select(session, tool, host, hosts)
  → convert.xspace_to_tool_data(paths, tool, options)
  → optional cache_policy.write_version(session)
  → ToolResult → respond()
```

### 3.7 Error handling

| Domain signal | HTTP |
|---------------|------|
| Unknown tool / no data | 404 text/plain `"No Data"` (preserve current) |
| Missing host file / invalid selection | 500 with message (preserve current unless later change is approved) |
| Convert AttributeError / ValueError | 500 |
| Cache gen bad method | 405 |
| Cache gen accepted | 202 JSON |
| Capture failure | 500 JSON `{error: ...}` |

Services raise typed exceptions where practical (`SessionNotFound`, `HostNotFound`, `InvalidToolRequest`); the HTTP layer maps them. Migration may keep bare built-ins initially if mapping 1:1 to current behavior is safer.

### 3.8 Public API contract

**Stable external:**

- `ProfilePlugin` (loader, server, demo, tests)
- Entry package import: `from xprof import profile_plugin`

**Allowed temporary re-exports during migration** (documented, to be shrunk):

- `ToolsCache`, `make_filename`, tool constant sets used by existing tests

**Not public long-term:**

- Private aliases (`_parse_filename`, etc.)
- HTTP helpers (import from `profile_plugin.http`)
- Internal services

## 4. Service responsibilities

| Service | Owns | Does not own |
|---------|------|--------------|
| `sessions.SessionResolver` | session_path / run_path / logdir precedence, frontend run naming | Convert |
| `sessions.RunDiscovery` | Walk plugins/profile, run map cache | HTTP |
| `hosts.HostSelector` | Multi-host / ALL_HOSTS / missing host rules | Tool options |
| `tools.catalog` | Available tools + ToolsCache integration | HTTP |
| `tools.options` | Query → convert options | I/O |
| `tool_data.ToolDataService` | Convert orchestration + cache version policy usage | Route registration |
| `cache_generation.CacheGenerationService` | Thread pool, warm tools, reuse ToolDataService/Hosts | UI |
| `capture.CaptureService` | Remote capture + TF name resolution via bridge | Static files |
| `hlo_modules.HloModuleService` | List/extract HLO module names | Trace options |
| `static_assets.StaticAssetService` | Read static from `plugin/xprof/static` | Domain analysis |

## 5. Testing strategy

### 5.1 Layers

| Layer | Location | Dependencies |
|-------|----------|--------------|
| Unit | `tests/unit/` | No Werkzeug server; fakes for ConvertPort/FS |
| Integration | `tests/integration/` | Temp dirs, thin ProfilePlugin, fake convert |
| Legacy | `profile_plugin_test.py` | Shrink as behavior moves; do not grow new cases here |

### 5.2 Golden fixtures (Phase 0)

Commit deterministic fixtures for:

- runs listing (logdir layout)
- run_tools ordering (including `trace_viewer@` override)
- hosts for ALL_HOSTS_ONLY vs SUPPORTED
- data_impl param shapes passed to convert (not C++ output)

These gate every extraction PR.

### 5.3 Patching rules

- Prefer fakes injected at construction.
- Do not add new tests that require patching private methods solely for I/O.
- Existing patches may remain until the method is deleted; update in the same PR that removes the method.

## 6. Packaging and BUILD

1. Production `py_library` for `profile_plugin` **excludes** `profile_plugin/tests/**`.
2. Dedicated `py_test` targets for unit and integration suites.
3. Pip packaging: ensure test modules are not installed as runtime package data (verify `build_pip_package` / setuptools package discovery).

## 7. CLI alignment

### 7.1 Principle

CLI is another adapter. High-value tools (`get_overview`, `get_memory_profile`, `get_kpi_metrics`, smart suggestions, utilization) should call `ToolDataService` (or thinner shared helpers) instead of re-deriving session paths and option dicts.

### 7.2 Phasing

- **Early:** Do not block façade extraction on full CLI migration.
- **Later tranche:** Migrate tool-by-tool; delete duplicated discovery/param packing when each tool moves.

## 8. Implementation phases (PR sequence)

Each phase is independently mergeable with green tests and no intentional behavior change.

| Phase | Name | Primary deliverables | Exit criteria |
|-------|------|----------------------|---------------|
| **0** | Guardrails | Golden fixtures; BUILD exclude tests; inventory public symbols | Fixtures green; prod lib has no tests |
| **1** | Models | `models.py`; `http/parse_request.py` | ToolRequest from sample query strings |
| **2** | Sessions | `services/sessions.py`; thin runs_imp/generate_runs | Unit tests for path precedence |
| **3** | Hosts | `services/hosts.py` | Table tests for host modes |
| **4** | Options | `tools/options/*`; counter names handler outside convert path | data_impl loses option towers |
| **5** | Tool data | `services/tool_data.py`; single cache version policy usage | plugin data_impl ≤ ~15 lines |
| **6** | Cache gen | `services/cache_generation.py` | Reuses tool_data + hosts |
| **7** | Capture / HLO / static | dedicated services | No business logic left in those plugin methods |
| **8** | Façade collapse | plugin.py ≤ 400 LOC; route table; clean deps factory | LOC metric met |
| **9** | Public API | Shrink `__init__.py`; remove aliases | Grep shows intentional exports only |
| **10** | CLI | Adopt services for primary tools | Shared domain for listed tools |
| **11** | Test cleanup | unit/integration layout; retire dead legacy cases | Dual-suite debt reduced |

### 8.1 Suggested first implementation tranche

Phases **0 → 5** (through ToolDataService) deliver most of the structural win. Phases 6–11 complete best-possible quality.

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Subtle behavior drift | Golden fixtures + existing tests every PR |
| Upstream openxla conflicts on package layout | Keep public import path; small PRs; rebase frequently |
| Over-abstraction | Protocols only at I/O/convert boundaries; plain dataclasses for models |
| Incomplete migration (second scaffold) | Exit criteria per phase are LOC/placement metrics, not “files exist” |
| CLI scope creep | Phase 10 explicit; not required for façade merge |

## 10. Explicit non-regressions

Must remain true unless separately approved:

- Route paths under `/data/plugin/profile` (standalone prefix handling in `server.py`).
- Query parameter names (`run`, `tag`, `host`, `hosts`, `use_saved_result`, trace options, etc.).
- Response encoding defaults (gzip + CSP via `respond()`).
- Tools sort order and `trace_viewer@` overriding `trace_viewer`.
- Host aggregation rules (`ALL_HOSTS` only / supported).
- Cache generation 202 / method / session_path validation semantics.

## 11. Relationship to commit `bdb4b675`

Treat `bdb4b675` as **accepted scaffolding**:

- **Keep:** package layout, tools/http/cache leaf modules, stable import path, package-local tests foundation.
- **Replace/finish:** god-class methods via services (Phases 2–8), BUILD/test packaging (Phase 0), public API (Phase 9), CLI (Phase 10).

This design is the definition of done for “profile plugin Python architecture is actually clean.”

## 12. Approval checklist

Reviewers should confirm:

- [ ] Layering (HTTP vs services vs ports) is acceptable
- [ ] Phase order and first tranche (0–5) are correct
- [ ] Public API shrinkage is desired
- [ ] CLI alignment can wait until after façade
- [ ] Success metrics (especially `plugin.py` ≤ 400 LOC) are the merge bar for Phase 8

After approval, an implementation plan (task-level PR checklist) will be written via the writing-plans workflow.
