"""Tool to fetch memory profile data from XProf."""

import dataclasses
import json
import logging
import traceback

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client

_BYTES_PER_GIB = 1024**3
_BYTES_PER_MIB = 1024**2


@dataclasses.dataclass(frozen=True)
class MemoryProfileMetrics:
  """Raw metrics parsed from a memory profile."""

  memory_capacity: int
  peak_hbm_bytes: int
  stack_bytes: int
  heap_bytes: int
  free_bytes: int
  fragmentation_raw: float


def _parse_profile_data(data: str) -> MemoryProfileMetrics:
  """Parses profile data and returns raw metrics.

  Args:
      data: The JSON string to parse.

  Returns:
      A MemoryProfileMetrics object containing the parsed data.

  Raises:
      json.JSONDecodeError: If JSON is invalid.
      ValueError: If format is unknown or unexpected.
  """
  parsed_data = json.loads(data)

  if isinstance(parsed_data, dict):
    parsed_data = [parsed_data]

  if not isinstance(parsed_data, list) or not parsed_data:
    raise ValueError(
        "Unexpected memory profile data format: not a list or empty"
    )

  peak_hbm_bytes = -1
  memory_capacity = -1
  stack_bytes = -1
  heap_bytes = -1
  free_bytes = -1
  fragmentation_raw = -1.0

  # 1. Handle format: [{"cols":..., "rows":...}]
  if "cols" in parsed_data[0] and "rows" in parsed_data[0]:
    raise ValueError("Table format memory profile not fully supported")

  # 2. Handle format: [{'memoryProfileSummary': ...}]
  elif "memoryProfileSummary" in parsed_data[0]:
    for device_data in parsed_data:
      if not device_data or "memoryProfileSummary" not in device_data:
        continue
      summary = device_data["memoryProfileSummary"]
      mem_cap = int(summary.get("memoryCapacity", -1))
      if mem_cap > memory_capacity:
        memory_capacity = mem_cap

      if not (
          summary
          and "peakStats" in summary
          and "peakBytesUsageHbm" in summary["peakStats"]
      ):
        continue
      try:
        current_peak = int(summary["peakStats"]["peakBytesUsageHbm"])
        if current_peak > peak_hbm_bytes:
          peak_hbm_bytes = current_peak
          stack_bytes = int(summary["peakStats"].get("stackReservedBytes", -1))
          heap_bytes = int(summary["peakStats"].get("heapAllocatedBytes", -1))
          free_bytes = int(summary["peakStats"].get("freeMemoryBytes", -1))
          fragmentation_raw = float(
              summary["peakStats"].get("fragmentation", -1.0)
          )
      except (ValueError, TypeError):
        continue

  # 3. Handle format: [{'peakMemoryUsageMiB': ...}]
  elif "peakMemoryUsageMiB" in parsed_data[0]:
    for item in parsed_data:
      if "peakMemoryUsageMiB" not in item:
        continue
      try:
        current_peak = float(item["peakMemoryUsageMiB"]) * _BYTES_PER_MIB
        if current_peak > peak_hbm_bytes:
          peak_hbm_bytes = int(current_peak)
      except (ValueError, TypeError):
        continue

  # 4. Handle format: [{"memoryProfilePerAllocator": ...}]
  elif "memoryProfilePerAllocator" in parsed_data[0]:
    allocator_data = parsed_data[0]["memoryProfilePerAllocator"]
    for _, stats in allocator_data.items():
      profile_summary = stats.get("profileSummary", {})

      mem_cap = int(profile_summary.get("memoryCapacity", -1))
      if mem_cap > memory_capacity:
        memory_capacity = mem_cap

      peak_stats = profile_summary.get("peakStats", {})
      current_peak = int(peak_stats.get("peakBytesInUse", -1))

      if current_peak > peak_hbm_bytes:
        peak_hbm_bytes = current_peak
        stack_bytes = int(peak_stats.get("stackReservedBytes", -1))
        heap_bytes = int(peak_stats.get("heapAllocatedBytes", -1))
        free_bytes = int(peak_stats.get("freeMemoryBytes", -1))
        fragmentation_raw = float(peak_stats.get("fragmentation", -1.0))
  else:
    raise ValueError("Unknown memory profile JSON format")

  return MemoryProfileMetrics(
      memory_capacity=memory_capacity,
      peak_hbm_bytes=peak_hbm_bytes,
      stack_bytes=stack_bytes,
      heap_bytes=heap_bytes,
      free_bytes=free_bytes,
      fragmentation_raw=fragmentation_raw,
  )


@decorators.cached(expire=86400)
def get_memory_profile(session_id: str) -> str:
  """Fetches a detailed memory profile analysis from an XProf session.

  **Use this** to get the XProf memory profile, which contains peak HBM usage,
  heap allocations, stack reservations, and overall device memory capacity.

  Args:
      session_id: The unique XProf session ID.

  Returns:
      A JSON-formatted string containing memory profile details.
  """
  session_id = str(session_id)
  client = xprof_client.get_client()

  default_output = {
      "memory_capacity_gib": -1.0,
      "peak_memory_usage_gib": -1.0,
      "peak_usage_details": {
          "stack_reservation_gib": -1.0,
          "heap_allocation_gib": -1.0,
          "free_memory_gib": -1.0,
          "fragmentation_percent": -1.0,
          "utilization_percent": -1.0,
      },
  }

  def _try_fetch_and_parse(
      host: str | None = None,
  ) -> tuple[MemoryProfileMetrics | None, str | None]:
    try:
      kwargs = {
          "tool_name": "memory_profile.json",
          "session_id": session_id,
          "format": "json",
      }
      if host:
        kwargs["host"] = host
      result = client.fetch(**kwargs)
      if isinstance(result, tuple) and len(result) == 2:
        _, data = result
      else:
        data = result

      if not data:
        return None, None

      if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")

      return _parse_profile_data(data), None
    except json.JSONDecodeError as e:
      logging.exception("JSON decode error for host %s", host)
      return None, f"Failed to parse JSON for memory profile: {e!r}"
    except ValueError as e:
      logging.warning("Value error during parsing for host %s: %r", host, e)
      return None, str(e)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Unexpected error for host %s", host)
      return None, f"Error fetching memory profile: {e!r}"

  def _try_multi_host_fallback(
      initial_parsed: MemoryProfileMetrics | None,
      initial_error: str | None,
  ) -> tuple[MemoryProfileMetrics | None, str | None]:
    logging.info(
        "Initial memory profile invalid, empty or missing. Trying multi-host"
        " fallback."
    )
    best_parsed, last_error = initial_parsed, initial_error

    try:
      hosts = client.get_hosts(session_id, with_metadata=True)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Failed to get hosts: %r", e)
      return best_parsed, f"Failed to get hosts: {e!r}"

    for host_info in hosts or []:
      host_name = (
          host_info.get("hostname")
          if isinstance(host_info, dict)
          else host_info
      )
      if not host_name:
        continue

      logging.info("Trying to fetch memory profile from host: %s", host_name)
      host_parsed, host_error = _try_fetch_and_parse(host=host_name)

      if host_parsed:
        if host_parsed.memory_capacity > 0 and host_parsed.peak_hbm_bytes > 0:
          logging.info("Found valid memory profile on host: %s", host_name)
          return host_parsed, None
        if not best_parsed:
          best_parsed, last_error = host_parsed, host_error
      elif host_error:
        last_error = host_error

    return best_parsed, last_error

  def _get_best_profile() -> tuple[MemoryProfileMetrics | None, str | None]:
    # 1. Initial fetch without host
    parsed, error = _try_fetch_and_parse()
    if parsed and parsed.memory_capacity > 0 and parsed.peak_hbm_bytes > 0:
      return parsed, None

    # 2. Fallback to multi-host if not valid
    return _try_multi_host_fallback(parsed, error)

  try:
    parsed, last_error = _get_best_profile()

    if parsed is None:
      if last_error:
        return json.dumps(dict(error=last_error), indent=2)
      return json.dumps(default_output, indent=2)

    # Calculate metrics with -1.0 fallback
    memory_capacity_gib = (
        round(parsed.memory_capacity / _BYTES_PER_GIB, 2)
        if parsed.memory_capacity > 0
        else -1.0
    )
    peak_memory_usage_gib = (
        round(parsed.peak_hbm_bytes / _BYTES_PER_GIB, 2)
        if parsed.peak_hbm_bytes > 0
        else -1.0
    )

    utilization_percent = -1.0
    if memory_capacity_gib > 0 and peak_memory_usage_gib > 0:
      utilization_percent = round(
          (peak_memory_usage_gib / memory_capacity_gib) * 100, 2
      )

    stack_reservation_gib = (
        round(parsed.stack_bytes / _BYTES_PER_GIB, 2)
        if parsed.stack_bytes >= 0
        else -1.0
    )
    heap_allocation_gib = (
        round(parsed.heap_bytes / _BYTES_PER_GIB, 2)
        if parsed.heap_bytes >= 0
        else -1.0
    )
    free_memory_gib = (
        round(parsed.free_bytes / _BYTES_PER_GIB, 2)
        if parsed.free_bytes >= 0
        else -1.0
    )
    fragmentation_percent = (
        round(parsed.fragmentation_raw * 100, 2)
        if parsed.fragmentation_raw >= 0
        else -1.0
    )

    output = {
        "memory_capacity_gib": memory_capacity_gib,
        "peak_memory_usage_gib": peak_memory_usage_gib,
        "peak_usage_details": {
            "stack_reservation_gib": stack_reservation_gib,
            "heap_allocation_gib": heap_allocation_gib,
            "free_memory_gib": free_memory_gib,
            "fragmentation_percent": fragmentation_percent,
            "utilization_percent": utilization_percent,
        },
    }

    return json.dumps(output, indent=2)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error fetching memory profile for session %s", session_id
    )
    return json.dumps(
        dict(
            error=f"Error fetching memory profile: {e}",
            traceback=traceback.format_exc(),
        ),
        indent=2,
    )
