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


def _get_percentage_from_rows(rows: list[dict[str, Any]]) -> float | None:
  """Calculates percentage from utilization viewer rows."""
  if not rows:
    return None
  rows_with_peak = [r for r in rows if _safe_float(r.get("Peak"), 0.0) > 0]
  if rows_with_peak:
    pcts = [
        _safe_float(r.get("Achieved"), 0.0)
        * 100
        / _safe_float(r.get("Peak"), 1.0)
        for r in rows_with_peak
    ]
    return round(sum(pcts) / len(pcts), 2)
  if all(_safe_float(r.get("Achieved"), 0.0) == 0 for r in rows):
    return 0.0
  return None


def _get_metric_percentage(
    node_rows: list[dict[str, Any]], metric_name: str
) -> float | None:
  """Calculates the percentage for a specific metric."""
  return _get_percentage_from_rows(
      [r for r in node_rows if r.get("Name") == metric_name]
  )


def _calculate_hbm_utilization(node_rows: list[dict[str, Any]]) -> float | None:
  """Calculates HBM bandwidth utilization."""
  hbm_rows = [
      r
      for r in node_rows
      if r.get("Name") in ("HBM Rd+Wr (per chip)", "HBM Rd+Wr")
  ]
  return _get_percentage_from_rows(hbm_rows)


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
  no_mxu_busy_rows = [r for r in node_rows if r.get("Name") == "No MXU Busy"]
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


def _format_utilization_viewer_output(
    csv_data: str,
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

    reader = csv.DictReader(io.StringIO(csv_data), skipinitialspace=True)
    if not reader.fieldnames or "Name" not in [
        f.strip() for f in reader.fieldnames if f
    ]:
      return json.dumps({"error": "Missing required column: Name"}, indent=2)

    rows = [
        {k.strip(): str(v).strip() for k, v in row.items() if k}
        for row in reader
    ]

    warnings = []
    node_rows = rows

    if reader.fieldnames and any(
        f and f.strip() == "Host" for f in reader.fieldnames
    ):
      node_rows = [
          r
          for r in node_rows
          if r.get("Host")
          and str(r.get("Host")).lower() != "nan"
          and int(_safe_float(r.get("Host"), -1.0)) == host
      ]
    elif host != 0:
      warnings.append(f"Host column missing; ignoring host={host} filter")

    if reader.fieldnames and any(
        f and f.strip() == "Device" for f in reader.fieldnames
    ):
      node_rows = [
          r
          for r in node_rows
          if r.get("Device")
          and str(r.get("Device")).lower() != "nan"
          and int(_safe_float(r.get("Device"), -1.0)) == device
      ]
    elif device != 0:
      warnings.append(f"Device column missing; ignoring device={device} filter")

    if reader.fieldnames and any(
        f and f.strip() == "Node" for f in reader.fieldnames
    ):
      node_rows = [
          r
          for r in node_rows
          if r.get("Node")
          and str(r.get("Node")).lower() != "nan"
          and int(_safe_float(r.get("Node"), -1.0)) == node
      ]
    elif node != 0:
      warnings.append(f"Node column missing; ignoring node={node} filter")

    if not node_rows:
      return json.dumps(
          {
              "message": (
                  f"No data found for Host {host} Device {device} Node {node}"
              )
          },
          indent=2,
      )

    ici_read_utilization = _get_metric_percentage(node_rows, "ICI (Read)")
    ici_write_utilization = _get_metric_percentage(node_rows, "ICI (Write)")
    vector_alu_utilization = _get_metric_percentage(node_rows, "Vector ALUs")
    scalar_unit_utilization = _get_metric_percentage(node_rows, "Scalar Unit")
    vmem_cmem_stores_utilization = _get_metric_percentage(
        node_rows, "Vmem/Cmem Stores"
    )
    vmem_loads_utilization = _get_metric_percentage(node_rows, "Vmem Loads")
    cmem_loads_utilization = _get_metric_percentage(node_rows, "Cmem Loads")
    hbm_bandwidth_utilization = _calculate_hbm_utilization(node_rows)
    xlu_utilization = _calculate_xlu_utilization(node_rows)
    mxu_utilization = _calculate_mxu_utilization(node_rows)
    idleness_percentage = _calculate_idleness_percentage(node_rows)

    metrics = {}
    rows_by_name = collections.defaultdict(list)
    for r in node_rows:
      name = r.get("Name")
      if name:
        rows_by_name[name].append(r)

    for name, rows in rows_by_name.items():
      pct = _get_percentage_from_rows(rows)
      if pct is not None:
        metrics[name] = pct

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
    if metrics:
      filtered_results["metrics"] = metrics
    if warnings:
      filtered_results["warnings"] = warnings

    return json.dumps(filtered_results, indent=2)

  except Exception as e:  # pylint: disable=broad-except
    logging.exception(
        "Error formatting utilization viewer output for session %s:", session_id
    )
    return json.dumps(
        {
            "error": (
                "Error formatting utilization viewer output for session"
                f" {session_id}: {e!r}"
            )
        },
        indent=2,
    )


@decorators.cached(expire=86_400)
def get_utilization_viewer(
    session_id: str, *, host: int = 0, device: int = 0, node: int = 0
) -> str:
  """Fetches and returns key metrics from utilization_viewer data.

  Args:
      session_id: The XProf session ID.
      host: The host ID to filter by (default is 0).
      device: The device ID to filter by (default is 0).
      node: The node ID to filter by (default is 0).

  Returns:
      A JSON string containing key utilization metrics or an error message.
  """
  client = xprof_client.get_client()
  try:
    result = client.fetch(
        tool_name="utilization_viewer.json",
        session_id=session_id,
        tqx="out:csv",
    )
  except Exception as e:  # pylint: disable=broad-except
    logging.exception(
        "Error fetching utilization_viewer.json for session %s", session_id
    )
    return json.dumps(
        {
            "error": (
                "Error fetching utilization_viewer.json for session"
                f" {session_id}: {e!r}"
            )
        },
        indent=2,
    )
  else:
    raw_data = (
        result[1]
        if isinstance(result, tuple) and len(result) == 2
        else result
    )

    if not raw_data:
      return json.dumps(
          {"error": f"No data returned for session {session_id}"}, indent=2
      )

    decoded_data = (
        raw_data.decode("utf-8", errors="replace")
        if isinstance(raw_data, bytes)
        else raw_data
    )

    return _format_utilization_viewer_output(
        decoded_data, session_id, host=host, device=device, node=node
    )
