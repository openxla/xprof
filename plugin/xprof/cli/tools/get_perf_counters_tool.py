"""Tool to fetch hardware performance counters from XProf."""

import json
import re
from typing import Any

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client


_UINT64_MAX = 0xFFFFFFFFFFFFFFFF


def _parse_counter_value(raw_val: Any) -> tuple[int, str]:
  """Parses a hex or numeric performance counter cell into (int, hex_str)."""
  if raw_val is None:
    return 0, "0x0"
  if isinstance(raw_val, (int, float)):
    int_val = max(0, min(int(raw_val), _UINT64_MAX))
    return int_val, hex(int_val)
  val_str = str(raw_val).strip()
  if not val_str:
    return 0, "0x0"
  try:
    if val_str.lower().startswith("0x"):
      int_val = max(0, min(int(val_str, 16), _UINT64_MAX))
      return int_val, hex(int_val)
    int_val = max(0, min(int(float(val_str)), _UINT64_MAX))
    return int_val, hex(int_val)
  except ValueError:
    return 0, val_str


def _matches_filter(text: str, pattern: str | None) -> bool:
  """Checks whether text matches an optional regex or substring filter."""
  if not pattern:
    return True
  try:
    return bool(re.search(pattern, text, re.IGNORECASE))
  except re.error:
    return pattern.lower() in text.lower()


def _parse_perf_counters_datatable(raw_data: str) -> list[dict[str, Any]]:
  """Parses XProf perf_counters Google DataTable JSON into structured rows."""
  raw_trimmed = raw_data.strip()
  if not raw_trimmed or raw_trimmed == "null":
    return []
  try:
    table = json.loads(raw_trimmed)
  except json.JSONDecodeError:
    return []
  if not isinstance(table, dict):
    return []

  cols = [
      col.get("id") or col.get("label") or f"col_{i}"
      for i, col in enumerate(table.get("cols", []))
  ]
  parsed_rows: list[dict[str, Any]] = []
  for row in table.get("rows", []):
    cells = row.get("c", []) if isinstance(row, dict) else []
    row_map: dict[str, Any] = {}
    for idx, col_name in enumerate(cols):
      cell = cells[idx] if idx < len(cells) else None
      if col_name == "Value" and isinstance(cell, dict):
        # DataTable AddHexCell stores the exact 64-bit hex string in "f" and a
        # 53-bit lossy IEEE-754 double in "v" (which rounds 0xffffffffffffffff
        # up to 2^64 = 18446744073709551616.0). Always prefer "f".
        val = cell.get("f") if cell.get("f") is not None else cell.get("v")
      else:
        val = cell.get("v") if isinstance(cell, dict) else cell
      row_map[col_name] = val

    raw_val = row_map.get("Value")
    int_val, hex_val = _parse_counter_value(raw_val)
    chip_raw = row_map.get("Chip")
    try:
      chip_int = int(chip_raw) if chip_raw is not None else -1
    except (ValueError, TypeError):
      chip_int = -1

    sample_raw = row_map.get("Sample")
    try:
      sample_int = int(sample_raw) if sample_raw is not None else 0
    except (ValueError, TypeError):
      sample_int = 0

    parsed_rows.append({
        "host": str(row_map.get("Host") or ""),
        "chip": chip_int,
        "kernel": str(row_map.get("Kernel") or ""),
        "sample": sample_int,
        "counter": str(row_map.get("Counter") or ""),
        "value": int_val,
        "value_hex": hex_val,
        "description": str(row_map.get("Description") or ""),
        "set": str(row_map.get("Set") or ""),
    })
  return parsed_rows


@decorators.cached(expire=86_400)
def get_perf_counters(
    session_id: str,
    hosts: list[str] | None = None,
    kernel_filter: str | None = None,
    counter_filter: str | None = None,
    set_filter: str | None = None,
    chip_id: int | None = None,
    non_zero_only: bool = True,
    limit: int = 50,
    bypass_cache: bool = False,
) -> str:
  """Fetches hardware performance counters from XProf with filtering.

  Queries the internal `perf_counters` engine endpoint and returns a bounded,
  sorted summary of hardware performance counters without dumping multi-megabyte
  DataTables directly to stdout.

  Args:
    session_id: The XProf session ID or trace directory path.
    hosts: Optional list of hostnames to filter by.
    kernel_filter: Optional case-insensitive regex/substring for Kernel name.
    counter_filter: Optional case-insensitive regex/substring for Counter or
      Description.
    set_filter: Optional case-insensitive regex/substring for Counter Set.
    chip_id: Optional chip ID to filter by (e.g., 0).
    non_zero_only: If True (default), excludes rows whose counter value is 0 or
      the uninitialized all-ones hardware sentinel (0xffffffffffffffff).
    limit: Maximum number of counter rows to return (default: 50).
    bypass_cache: Whether to bypass cache and recompute metrics.

  Returns:
    A JSON string containing `status`, `summary` statistics, and bounded
    `counters` rows sorted descending by integer counter value.
  """
  client = xprof_client.get_client()
  fetch_kwargs: dict[str, Any] = {
      "tool_name": "perf_counters",
      "session_id": session_id,
      "bypass_cache": bypass_cache,
  }
  if hosts is not None:
    fetch_kwargs["hosts"] = hosts
  result = client.fetch(**fetch_kwargs)
  if isinstance(result, tuple) and len(result) == 2:
    _, raw_data = result
  else:
    raw_data = result

  if isinstance(raw_data, bytes):
    raw_data = raw_data.decode("utf-8", errors="replace")

  rows = _parse_perf_counters_datatable(str(raw_data or ""))
  total_rows = len(rows)
  sentinel_rows = sum(1 for r in rows if r["value"] == _UINT64_MAX)
  non_zero_rows = sum(
      1 for r in rows if 0 < r["value"] < _UINT64_MAX
  )
  unique_hosts = sorted({r["host"] for r in rows if r["host"]})
  unique_chips = sorted({r["chip"] for r in rows if r["chip"] >= 0})
  unique_kernels = len({r["kernel"] for r in rows if r["kernel"]})
  unique_counters = len({r["counter"] for r in rows if r["counter"]})

  filtered: list[dict[str, Any]] = []
  for r in rows:
    if non_zero_only and (r["value"] == 0 or r["value"] == _UINT64_MAX):
      continue
    if chip_id is not None and r["chip"] != int(chip_id):
      continue
    if not _matches_filter(r["kernel"], kernel_filter):
      continue
    if counter_filter and not (
        _matches_filter(r["counter"], counter_filter)
        or _matches_filter(r["description"], counter_filter)
    ):
      continue
    if not _matches_filter(r["set"], set_filter):
      continue
    filtered.append(r)

  filtered.sort(
      key=lambda r: (
          r["value"] != _UINT64_MAX,
          r["value"],
          r["counter"],
          r["kernel"],
      ),
      reverse=True,
  )
  bounded = filtered[: max(0, int(limit))] if limit >= 0 else filtered

  status = "OK" if total_rows > 0 else "NO_PERF_COUNTERS"
  return json.dumps(
      {
          "status": status,
          "summary": {
              "total_rows": total_rows,
              "non_zero_rows": non_zero_rows,
              "sentinel_all_ones_rows": sentinel_rows,
              "matched_rows": len(filtered),
              "returned_rows": len(bounded),
              "unique_kernels": unique_kernels,
              "unique_counters": unique_counters,
              "hosts": unique_hosts,
              "chips": unique_chips,
          },
          "counters": bounded,
      },
      indent=2,
  )
