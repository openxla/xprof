# `xprof.profile_plugin` layout

The HTTP TensorBoard/XProf profile UI is split so **no single module owns everything**.

## Layers

| Layer | Package | Responsibility |
|-------|---------|----------------|
| HTTP edge | `http/` | Werkzeug request parse, `respond()`, access logging |
| Façade | `plugin.py` | Route table, construct services, thin handlers |
| Domain services | `services/` | Sessions, hosts, tool data, run discovery (no Werkzeug) |
| Tool taxonomy | `tools/` | Filenames, registry, catalog, option builders |
| Cache | `cache/` | Tools list cache, result-cache version policy |
| Ports | `deps.py` | Convert / filesystem protocols for injection |
| Optional TF | `tensorflow_bridge.py` | Remote TPU capture helpers |

## Where to look

- **Add a tool query option** → `tools/options/<family>.py` + registry
- **Change host selection** → `services/hosts.py`
- **Change session_path / run_path rules** → `services/sessions.py`
- **Change convert orchestration** → `services/tool_data.py`
- **Change logdir run discovery** → `services/runs.py`
- **Change HTTP status/body mapping** → route methods in `plugin.py` + `http/respond.py`

## Tests

```bash
python3 plugin/xprof/profile_plugin/tests/run_all_tests.py
```

Unit tests live under `tests/unit/`; golden HTTP/shape checks under `tests/golden_*` and `tests/fixtures/`.

Target: keep **`plugin.py` as a thin façade** (routes + wiring). Business logic belongs in `services/` / `tools/`.
