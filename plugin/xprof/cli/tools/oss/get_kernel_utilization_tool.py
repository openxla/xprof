"""Tool to calculate TPU hardware kernel compute utilization in OSS."""

import json
import logging
import pathlib
from typing import Any

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import kernel_stats_tools
from xprof.cli.internal.oss import xprof_client
from xprof.convert import raw_to_tool_data as convert


def _needs_duration_fallback(parsed: dict[str, Any]) -> bool:
  """Returns True if any device kernel has duration_us <= 0."""
  devices = parsed.get("devices", [])
  if not devices:
    return False
  for device in devices:
    for kernel in device.get("kernels", []):
      if (
          "duration_us" in kernel
          and float(kernel["duration_us"] or 0.0) <= 0.0
      ):
        return True
  return False


def _clamp_utilization_metrics(parsed: dict[str, Any]) -> dict[str, Any]:
  """Clamps all percentage metrics to [0.0, 100.0] and normalizes idle counters."""
  for device in parsed.get("devices", []):
    for kernel in device.get("kernels", []):
      active_mxu = 0.0
      if "mxu_utilization" in kernel:
        try:
          mxu_utilization = float(kernel["mxu_utilization"])
          kernel["mxu_utilization"] = min(100.0, max(0.0, mxu_utilization))
          active_mxu = kernel["mxu_utilization"]
        except (ValueError, TypeError):
          pass

      other_metrics = kernel.get("other_metrics", {})
      if isinstance(other_metrics, dict):
        if "Avg MXU Busy" in other_metrics:
          val = other_metrics["Avg MXU Busy"]
          if isinstance(val, (int, float)):
            active_mxu = max(active_mxu, min(100.0, max(0.0, float(val))))
          elif isinstance(val, dict) and "utilization_percent" in val:
            try:
              active_mxu = max(
                  active_mxu,
                  min(100.0, max(0.0, float(val["utilization_percent"]))),
              )
            except (ValueError, TypeError):
              pass

        active_xlu = 0.0
        if "Avg XLU Busy" in other_metrics:
          val = other_metrics["Avg XLU Busy"]
          if isinstance(val, (int, float)):
            active_xlu = min(100.0, max(0.0, float(val)))
          elif isinstance(val, dict) and "utilization_percent" in val:
            try:
              active_xlu = min(
                  100.0, max(0.0, float(val["utilization_percent"]))
              )
            except (ValueError, TypeError):
              pass
        else:
          xlu_1 = other_metrics.get("1 XLU Busy")
          xlu_2 = other_metrics.get("2 XLUs Busy") or other_metrics.get(
              "2 XLU Busy"
          )
          x1 = float(xlu_1) if isinstance(xlu_1, (int, float)) else 0.0
          x2 = float(xlu_2) if isinstance(xlu_2, (int, float)) else 0.0
          if x1 > 0.0 or x2 > 0.0:
            active_xlu = min(100.0, max(0.0, 0.5 * x1 + x2))

        if "No MXU Busy" in other_metrics:
          val = other_metrics["No MXU Busy"]
          norm_no_mxu = max(0.0, min(100.0, 100.0 - active_mxu))
          if isinstance(val, (int, float)):
            other_metrics["No MXU Busy"] = norm_no_mxu
          elif isinstance(val, dict) and "utilization_percent" in val:
            val["utilization_percent"] = norm_no_mxu

        if "No XLU Busy" in other_metrics:
          val = other_metrics["No XLU Busy"]
          norm_no_xlu = max(0.0, min(100.0, 100.0 - active_xlu))
          if isinstance(val, (int, float)):
            other_metrics["No XLU Busy"] = norm_no_xlu
          elif isinstance(val, dict) and "utilization_percent" in val:
            val["utilization_percent"] = norm_no_xlu

        for metric_name, metric_entry in other_metrics.items():
          if isinstance(metric_entry, (int, float)):
            other_metrics[metric_name] = min(
                100.0, max(0.0, float(metric_entry))
            )
          elif isinstance(metric_entry, dict):
            if "utilization_percent" in metric_entry:
              try:
                utilization_percent = float(metric_entry["utilization_percent"])
                metric_entry["utilization_percent"] = min(
                    100.0, max(0.0, utilization_percent)
                )
              except (ValueError, TypeError):
                pass
  return parsed


def _lookup_fallback_duration_us(
    source: Any,
    kernel_name: str | None,
    bypass_cache: bool,
) -> float:
  """Queries kernel_stats for total kernel duration in microseconds."""
  records = None
  if kernel_name:
    try:
      stats = kernel_stats_tools.get_kernel_stats(
          source,
          kernel_name=kernel_name,
          output_format="dict",
          bypass_cache=bypass_cache,
      )
    except (RuntimeError, ValueError, KeyError, OSError, TypeError):
      logging.exception(
          "Failed to look up fallback duration for %r from kernel_stats",
          kernel_name,
      )
      stats = None
    if stats:
      records = (
          stats.get("kernel_records", []) if isinstance(stats, dict) else stats
      )

  if not records:
    try:
      stats = kernel_stats_tools.get_kernel_stats(
          source,
          kernel_name=None,
          output_format="dict",
          bypass_cache=bypass_cache,
      )
    except (RuntimeError, ValueError, KeyError, OSError, TypeError):
      logging.exception("Failed to look up fallback duration from kernel_stats")
      return 0.0
    records = (
        stats.get("kernel_records", []) if isinstance(stats, dict) else stats
    )

  if isinstance(records, list) and records:
    top_record = records[0]
    if isinstance(top_record, dict):
      total_duration_us = float(top_record.get("total_duration_us") or 0.0)
      if total_duration_us > 0.0:
        return total_duration_us
      execution_count = int(top_record.get("execution_count") or 1)
      avg_duration_us = float(top_record.get("avg_duration_us") or 0.0)
      if avg_duration_us > 0.0:
        return avg_duration_us * max(1, execution_count)
  return 0.0


@decorators.cached(expire=86400)
def get_kernel_utilization(
    session_id: str,
    *,
    kernel_name: str | None = None,
    duration_us: float | None = None,
    force_duration: bool = False,
    host: str = "",
    device: int | None = None,
    output_format: str = "json",
    raw_bytes: bytes | None = None,
    bypass_cache: bool = False,
) -> str | dict[str, Any]:
  """Calculates hardware compute utilization from performance counters in OSS.

  Args:
    session_id: The XProf session ID, run name, directory, or file path.
    kernel_name: Optional filter for a specific kernel name.
    duration_us: Optional benchmark duration override in microseconds.
    force_duration: Whether to force duration_us override over hardware cycle
      counters.
    host: Host filter.
    device: Device filter (0-indexed integer).
    output_format: "json" (default) or "dict".
    raw_bytes: Optional raw XSpace protobuf bytes.
    bypass_cache: Whether to bypass cache.

  Returns:
    A JSON string or Python dict containing structured utilization metrics.

  Raises:
    ValueError: If neither session_id nor raw_bytes is provided.
    FileNotFoundError: If no utilization data is found for the session.
    RuntimeError: If computing or fetching utilization fails, or if all
      metrics and hardware counter duration evaluate to 0.0.
  """
  del host
  if not session_id and raw_bytes is None:
    raise ValueError("session_id or raw_bytes must be provided.")

  params: dict[str, Any] = {}
  if kernel_name:
    params["kernel"] = kernel_name
  if duration_us is not None:
    params["duration_us"] = str(duration_us)
  if force_duration:
    params["force_duration"] = True
  if device is not None:
    params["device_id"] = str(device)

  # Mode 1: Direct in-memory proto bytes (e.g. offline analysis)
  if raw_bytes is not None:
    raw_data, _ = convert.xspace_to_tools_data_from_byte_string(
        [raw_bytes], ["trace.pb"], "kernel_utilization", params
    )
    if not raw_data:
      raise RuntimeError(
          "Failed to compute utilization from raw bytes: no data returned."
      )
    if isinstance(raw_data, bytes):
      decoded_str = raw_data.decode("utf-8", errors="replace")
    else:
      decoded_str = str(raw_data)

  # Mode 2: Local file or directory path
  elif session_id.startswith("/") or session_id.startswith("./"):
    file_path = pathlib.Path(session_id)
    if not file_path.exists():
      raise FileNotFoundError(f"Path does not exist: {session_id!r}")

    if file_path.is_dir():
      all_files = sorted(
          set(
              [str(p) for p in file_path.glob("**/*.xplane.pb")]
              + [str(p) for p in file_path.glob("**/*.xspace.pb")]
          )
      )
      if not all_files:
        raise FileNotFoundError(
            "No .xplane.pb or .xspace.pb files found in directory:"
            f" {session_id!r}"
        )
      file_bytes_list = []
      for p in all_files:
        with open(p, "rb") as f:
          file_bytes_list.append(f.read())
      file_paths_list = all_files
    else:
      with open(session_id, "rb") as f:
        file_bytes = f.read()
      file_bytes_list = [file_bytes]
      file_paths_list = [session_id]

    raw_data, _ = convert.xspace_to_tools_data_from_byte_string(
        file_bytes_list, file_paths_list, "kernel_utilization", params
    )
    if not raw_data:
      raise RuntimeError(
          f"Failed to compute utilization from file {session_id!r}: no data"
          " returned."
      )
    if isinstance(raw_data, bytes):
      decoded_str = raw_data.decode("utf-8", errors="replace")
    else:
      decoded_str = str(raw_data)

  # Mode 3: Session ID lookup via OSS xprof_client
  else:
    client = xprof_client.get_client()
    try:
      result = client.fetch(
          tool_name="kernel_utilization.json",
          session_id=session_id,
          bypass_cache=bypass_cache,
          **params,
      )
    except Exception as e:
      logging.exception(
          "Error fetching kernel_utilization.json for session %r", session_id
      )
      raise RuntimeError(
          "Error fetching kernel_utilization.json for session"
          f" {session_id!r}: {e!r}"
      ) from e
    raw_data = (
        result[1]
        if isinstance(result, tuple) and len(result) == 2
        else result
    )
    if not raw_data:
      raise FileNotFoundError(
          f"No utilization data returned for session {session_id!r}."
      )
    if isinstance(raw_data, bytes):
      decoded_str = raw_data.decode("utf-8", errors="replace")
    else:
      decoded_str = str(raw_data)

  parsed = json.loads(decoded_str)
  if _needs_duration_fallback(parsed):
    if duration_us is None:
      source = raw_bytes if raw_bytes is not None else session_id
      target_kernel = kernel_name
      if target_kernel is None:
        for device_entry in parsed.get("devices", []):
          for kernel_entry in device_entry.get("kernels", []):
            kernel_name_found = kernel_entry.get("kernel_name")
            if kernel_name_found and kernel_name_found != "default_kernel":
              target_kernel = kernel_name_found
              break
          if target_kernel is not None:
            break
      fallback_duration_us = _lookup_fallback_duration_us(
          source, target_kernel, bypass_cache
      )
      if fallback_duration_us > 0.0:
        return get_kernel_utilization(
            session_id=session_id,
            kernel_name=kernel_name,
            duration_us=fallback_duration_us,
            force_duration=True,
            device=device,
            output_format=output_format,
            raw_bytes=raw_bytes,
            bypass_cache=bypass_cache,
        )
    raise RuntimeError(
        "Hardware counter duration is 0.0 us. Pass --duration_us=<us> "
        "--force_duration explicitly (obtainable via get_kernel_stats)."
    )

  parsed = _clamp_utilization_metrics(parsed)
  if output_format == "dict":
    return parsed
  return json.dumps(parsed, indent=2)
