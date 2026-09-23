"""Tool to fetch utilization viewer data from XProf."""

import collections
import csv
import io
import json
import logging
import re
from typing import Any

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client


def _safe_float(val: Any, default: float = 0.0) -> float:
  """Safely converts a value to float, handling 'None' and 'NaN' strings."""
  try:
    if val is None:
      return default
    val_str = str(val).strip().lower()
    if val_str in ("nan", "none", "", "null"):
      return default
    return float(val)
  except ValueError:
    return default


# Hardware counters that failed to read back are reported as an all-ones 64-bit
# value. The backend does not strip these, and a sample where both the busy
# counter and the cycle counter are sentinels yields a literal "100% idle" row.
_COUNTER_SENTINEL = float(0xFFFFFFFFFFFFFFFF)
_COUNTER_SENTINEL_FLOOR = _COUNTER_SENTINEL * 0.999


def _is_sentinel(val: float) -> bool:
  """Returns True for unreadable counter values (all-ones 64-bit sentinel)."""
  return val >= _COUNTER_SENTINEL_FLOOR


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
  """Aggregates utilization rows for a single metric across counter samples.

  The backend emits one row per counter-sampling interval (the ``Sample``
  column), per metric, per node. ``Achieved`` and ``Peak`` are raw counts over
  that interval in the same unit, so the only correct aggregation is
  cycle-weighted: ``sum(Achieved) / sum(Peak)``. Averaging the per-sample
  ratios instead gives an idle interval the same weight as a busy one.

  Args:
    rows: Rows for one metric name, already filtered to a single host/device/
      node.

  Returns:
    A dict with the weighted percentage over the whole capture window, the
    busiest single sample, and the sample count; or None when no usable row
    remains.
  """
  if not rows:
    return None

  total_achieved = 0.0
  total_peak = 0.0
  sample_percents: list[float] = []
  usable = 0
  for row in rows:
    achieved = _safe_float(row.get("Achieved"))
    peak = _safe_float(row.get("Peak"))
    if _is_sentinel(achieved) or _is_sentinel(peak) or peak < 0:
      continue
    usable += 1
    total_achieved += achieved
    total_peak += peak
    if peak > 0:
      sample_percents.append(achieved * 100.0 / peak)

  if not usable:
    return None
  if total_peak <= 0:
    # Every usable sample had a zero denominator. Only a genuine all-zero
    # numerator lets us claim 0%; anything else is unmeasurable.
    if total_achieved == 0:
      return {"percent": 0.0, "peak_sample_percent": 0.0, "samples": usable}
    return None

  return {
      "percent": round(total_achieved * 100.0 / total_peak, 2),
      "peak_sample_percent": (
          round(max(sample_percents), 2) if sample_percents else 0.0
      ),
      "samples": usable,
  }


def _get_percentage_from_rows(rows: list[dict[str, Any]]) -> float | None:
  """Returns the cycle-weighted utilization percentage for one metric."""
  agg = _aggregate_rows(rows)
  return None if agg is None else agg["percent"]


def _rows_for_metric(
    node_rows: list[dict[str, Any]], *metric_names: str
) -> list[dict[str, Any]]:
  """Selects rows whose Name equals, or is a per-core variant of, a metric.

  The backend suffixes some metric names with the core they belong to (for
  example ``HBM Rd+Wr - core 0``), so an exact-match lookup silently returns
  nothing. Matching the ``"<metric> - core N"`` form as well keeps those
  metrics reachable.

  Args:
    node_rows: All rows for the selected host/device/node.
    *metric_names: Accepted metric names, in priority order.

  Returns:
    The matching rows, or an empty list.
  """
  for metric_name in metric_names:
    prefix = f"{metric_name} - core "
    matched = [
        r
        for r in node_rows
        if r.get("Name") == metric_name or str(r.get("Name", "")).startswith(
            prefix
        )
    ]
    if matched:
      return matched
  return []


def _get_metric_percentage(
    node_rows: list[dict[str, Any]], *metric_names: str
) -> float | None:
  """Calculates the percentage for a specific metric."""
  return _get_percentage_from_rows(_rows_for_metric(node_rows, *metric_names))


def _measurement_window_note(samples: int) -> dict[str, Any]:
  """Describes the window the reported percentages are averaged over.

  The utilization_viewer backend applies no time window, no step filter and no
  kernel filter: the denominator is the free-running TensorCore cycle counter,
  which keeps advancing while the device is idle. Every percentage here is
  therefore a fraction of the *entire capture*, not of kernel execution time. A
  short kernel inside a long capture reads as near-zero utilization even when
  it saturates the machine while it runs, so the numbers must not be read as
  "how efficient is my kernel".

  Args:
    samples: Number of usable counter-sampling intervals per metric.

  Returns:
    A dict describing the aggregation basis and pointing at the kernel-scoped
    tool.
  """
  return {
      "basis": "whole_capture",
      "samples_per_metric": samples,
      "aggregation": "cycle_weighted_sum_achieved_over_sum_peak",
      "note": (
          "Percentages are averaged over the whole profiling capture, "
          "including idle time between kernels: the counter denominator is "
          "the free-running TensorCore clock, which is not gated on kernel "
          "execution. A kernel that saturates the device for a small "
          "fraction of the capture will still report near-zero utilization "
          "here, under-reporting by roughly capture_duration / "
          "kernel_duration."
      ),
      "for_per_kernel_utilization_use": "get_kernel_utilization",
      "peak_sample_percent_note": (
          "peak_sample_percent is the busiest single sampling interval for "
          "each metric and is a closer proxy for in-kernel utilization than "
          "the capture-wide average."
      ),
  }


def _calculate_hbm_utilization(node_rows: list[dict[str, Any]]) -> float | None:
  """Calculates HBM bandwidth utilization."""
  return _get_metric_percentage(
      node_rows, "HBM Rd+Wr (per chip)", "HBM Rd+Wr"
  )


def _calculate_xlu_utilization(node_rows: list[dict[str, Any]]) -> float | None:
  """Calculates average XLU utilization."""
  names = {r.get("Name") for r in node_rows if r.get("Name")}
  xlu_names = [name for name in names if re.fullmatch(r"XLU\d+", str(name))]
  xlus = [_get_metric_percentage(node_rows, name) for name in xlu_names]
  xlus_valid = [x for x in xlus if x is not None]
  if xlus_valid:
    return round(sum(xlus_valid) / len(xlus_valid), 2)
  return None


def _calculate_mxu_utilization(node_rows: list[dict[str, Any]]) -> float | None:
  """Calculates average MXU utilization."""
  avg_mxu = _get_metric_percentage(node_rows, "Avg MXU Busy")
  if avg_mxu is not None:
    return avg_mxu

  names = {r.get("Name") for r in node_rows if r.get("Name")}
  mxu_names = [name for name in names if re.fullmatch(r"MXU\d+", str(name))]
  mxus = [_get_metric_percentage(node_rows, name) for name in mxu_names]
  mxus_valid = [m for m in mxus if m is not None]
  if mxus_valid:
    return round(sum(mxus_valid) / len(mxus_valid), 2)
  return None


def _calculate_idleness_percentage(
    node_rows: list[dict[str, Any]],
) -> float | None:
  """Calculates device idleness percentage."""
  no_mxu_busy_rows = _rows_for_metric(node_rows, "No MXU Busy")
  if no_mxu_busy_rows:
    return _get_percentage_from_rows(no_mxu_busy_rows)

  names = {r.get("Name") for r in node_rows if r.get("Name")}
  mxu_names = [name for name in names if re.fullmatch(r"MXU\d+", str(name))]
  mxus = [_get_metric_percentage(node_rows, name) for name in mxu_names]
  mxus_valid = [m for m in mxus if m is not None]
  if mxus_valid:
    return max(0.0, round(100 - max(mxus_valid), 2))

  # Fallback approximation
  avg_mxu = _get_metric_percentage(node_rows, "Avg MXU Busy")
  if avg_mxu is not None:
    return round(100 - avg_mxu, 2)

  return 100.0


def _parse_utilization_data(
    raw_data: str,
) -> tuple[list[dict[str, Any]], list[str]]:
  """Parses utilization data from either Google DataTable JSON or CSV string."""
  raw_trimmed = raw_data.strip()
  if raw_trimmed.startswith("{"):
    try:
      table_json = json.loads(raw_trimmed)
      if isinstance(table_json, dict) and "cols" in table_json:
        cols = [
            c.get("label") or c.get("id", f"col_{i}")
            for i, c in enumerate(table_json.get("cols", []))
        ]
        rows = []
        for row in table_json.get("rows", []):
          cells = row.get("c", [])
          row_dict = {}
          for i, cell in enumerate(cells):
            if i < len(cols):
              val = cell.get("v") if isinstance(cell, dict) else cell
              row_dict[cols[i]] = val
          rows.append(row_dict)
        return rows, cols
    except (ValueError, TypeError, json.JSONDecodeError):
      pass

  reader = csv.DictReader(io.StringIO(raw_data), skipinitialspace=True)
  fieldnames = [f.strip() for f in reader.fieldnames or [] if f]
  rows = [
      {k.strip(): str(v).strip() for k, v in row.items() if k} for row in reader
  ]
  return rows, fieldnames


def _format_utilization_viewer_output(
    raw_data: str,
    session_id: str,
    *,
    host: int = 0,
    device: int = 0,
    node: int = 0,
) -> str:
  """Formats the utilization viewer output."""
  try:
    host = int(host)
    device = int(device)
    node = int(node)

    rows, fieldnames = _parse_utilization_data(raw_data)
    if not rows:
      return json.dumps(
          {
              "status": "NO_DATA",
              "message": (
                  "No hardware performance counter events found in trace"
              ),
          },
          indent=2,
      )

    has_name = "Name" in fieldnames or any("Name" in r for r in rows)
    if not has_name:
      raise ValueError("Missing required column: Name")

    warnings = []
    node_rows = rows

    has_host = any(f and f.strip() == "Host" for f in fieldnames) or any(
        "Host" in r for r in rows
    )
    if has_host:
      node_rows = [
          r
          for r in node_rows
          if r.get("Host") is not None
          and str(r.get("Host")).lower() not in ("nan", "none", "")
          and int(_safe_float(r.get("Host"), -1.0)) == host
      ]
    elif host != 0:
      warnings.append(f"Host column missing; ignoring host={host} filter")

    has_device = any(f and f.strip() == "Device" for f in fieldnames) or any(
        "Device" in r for r in rows
    )
    if has_device:
      node_rows = [
          r
          for r in node_rows
          if r.get("Device") is not None
          and str(r.get("Device")).lower() not in ("nan", "none", "")
          and int(_safe_float(r.get("Device"), -1.0)) == device
      ]
    elif device != 0:
      warnings.append(f"Device column missing; ignoring device={device} filter")

    has_node = any(f and f.strip() == "Node" for f in fieldnames) or any(
        "Node" in r for r in rows
    )
    if has_node:
      node_rows = [
          r
          for r in node_rows
          if r.get("Node") is not None
          and str(r.get("Node")).lower() not in ("nan", "none", "")
          and int(_safe_float(r.get("Node"), -1.0)) == node
      ]
    elif node != 0:
      warnings.append(f"Node column missing; ignoring node={node} filter")

    if not node_rows:
      return json.dumps(
          {
              "status": "NO_DATA",
              "message": (
                  f"No data found for Host {host} Device {device} Node {node}"
              ),
          },
          indent=2,
      )

    ici_read_utilization = _get_metric_percentage(node_rows, "ICI (Read)")
    ici_write_utilization = _get_metric_percentage(node_rows, "ICI (Write)")
    vector_alu_utilization = _get_metric_percentage(node_rows, "Vector ALUs")
    scalar_unit_utilization = _get_metric_percentage(node_rows, "Scalar Unit")
    # The backend emits "Vmem Stores"; "Vmem/Cmem Stores" never matched. The
    # legacy name is kept as a fallback for older traces.
    vmem_cmem_stores_utilization = _get_metric_percentage(
        node_rows, "Vmem Stores", "Vmem/Cmem Stores"
    )
    vmem_loads_utilization = _get_metric_percentage(node_rows, "Vmem Loads")
    cmem_loads_utilization = _get_metric_percentage(node_rows, "Cmem Loads")
    hbm_bandwidth_utilization = _calculate_hbm_utilization(node_rows)
    xlu_utilization = _calculate_xlu_utilization(node_rows)
    mxu_utilization = _calculate_mxu_utilization(node_rows)
    idleness_percentage = _calculate_idleness_percentage(node_rows)

    metrics = {}
    peak_sample_percents = {}
    sample_counts = set()
    rows_by_name = collections.defaultdict(list)
    for r in node_rows:
      name = r.get("Name")
      if name:
        rows_by_name[name].append(r)

    for name, named_rows in rows_by_name.items():
      agg = _aggregate_rows(named_rows)
      if agg is not None:
        metrics[name] = agg["percent"]
        peak_sample_percents[name] = agg["peak_sample_percent"]
        sample_counts.add(agg["samples"])

    results = {
        "hbm_bandwidth_utilization_percent": hbm_bandwidth_utilization,
        "ici_read_utilization_percent": ici_read_utilization,
        "ici_write_utilization_percent": ici_write_utilization,
        "vector_alu_utilization_percent": vector_alu_utilization,
        "scalar_unit_utilization_percent": scalar_unit_utilization,
        "vmem_cmem_stores_utilization_percent": vmem_cmem_stores_utilization,
        "vmem_loads_utilization_percent": vmem_loads_utilization,
        "cmem_loads_utilization_percent": cmem_loads_utilization,
        "xlu_utilization_percent": xlu_utilization,
        "mxu_utilization_percent": mxu_utilization,
        "idleness_percent": idleness_percentage,
    }

    filtered_results = {k: v for k, v in results.items() if v is not None}
    # pyrefly: ignore[unsupported-operation]
    filtered_results["measurement_window"] = _measurement_window_note(
        max(sample_counts) if sample_counts else 0
    )
    if metrics:
      filtered_results["metrics"] = metrics  # pyrefly: ignore[unsupported-operation]
    if peak_sample_percents:
      # pyrefly: ignore[unsupported-operation]
      filtered_results["peak_sample_percent"] = peak_sample_percents
    if warnings:
      filtered_results["warnings"] = warnings  # pyrefly: ignore[unsupported-operation]

    return json.dumps(filtered_results, indent=2)

  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-except
    logging.exception(
        "Error formatting utilization viewer output for session %s:", session_id
    )
    raise RuntimeError(
        "Error formatting utilization viewer output for session"
        f" {session_id}: {e!r}"
    ) from e


@decorators.cached(expire=86_400)
def get_utilization_viewer(
    session_id: str,
    *,
    host: int = 0,
    device: int = 0,
    node: int = 0,
    bypass_cache: bool = False,
) -> str:
  """Fetches and returns key metrics from utilization_viewer data.

  Every percentage is a cycle-weighted average over the *whole* profiling
  capture, including the idle gaps between kernels, because the underlying
  hardware counters are normalized by the free-running TensorCore clock rather
  than by kernel execution time. Short kernels inside a long capture therefore
  report near-zero utilization even when they saturate the device. The
  ``measurement_window`` block in the result states this, and
  ``peak_sample_percent`` gives the busiest single sampling interval per
  metric. For utilization scoped to a kernel, use ``get_kernel_utilization``.

  Args:
      session_id: The XProf session ID.
      host: The host ID to filter by (default is 0).
      device: The device ID to filter by (default is 0).
      node: The node ID to filter by (default is 0).
      bypass_cache: Whether to bypass cache and recompute metrics.

  Returns:
      A JSON string containing key utilization metrics, the
      ``measurement_window`` disclosure, per-metric ``metrics`` and
      ``peak_sample_percent`` maps, or a ``NO_DATA`` envelope.
  """
  client = xprof_client.get_client()
  try:
    result = client.fetch(
        tool_name="utilization_viewer.json",
        session_id=session_id,
        tqx="out:csv",
        bypass_cache=bypass_cache,
    )
  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-except
    logging.exception(
        "Error fetching utilization_viewer.json for session %s", session_id
    )
    raise RuntimeError(
        f"Error fetching utilization_viewer.json for session {session_id}:"
        f" {e!r}"
    ) from e
  else:
    raw_data = (
        result[1]
        if isinstance(result, tuple) and len(result) == 2
        else result
    )

    if not raw_data:
      return json.dumps(
          {
              "status": "NO_DATA",
              "message": f"No data returned for session {session_id}",
          },
          indent=2,
      )

    decoded_data = (
        raw_data.decode("utf-8", errors="replace")
        if isinstance(raw_data, bytes)
        else str(raw_data)
    )

    return _format_utilization_viewer_output(
        decoded_data, session_id, host=host, device=device, node=node
    )
