"""Tool to calculate TPU hardware kernel compute utilization in 3P."""

import json
import logging
from typing import Any

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.convert import raw_to_tool_data as convert


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
  """Calculates hardware compute utilization from performance counters in 3P.

  Args:
    session_id: The XProf session ID, session path, or file path.
    kernel_name: Optional filter for a specific kernel name.
    duration_us: Optional benchmark duration override in microseconds.
    force_duration: Whether to force duration_us override over hardware cycle
      counters.
    host: Host filter.
    device: Device filter (0-indexed integer).
    output_format: "json" (default) or "dict".
    raw_bytes: Optional raw XSpace or XprofResponse protobuf bytes.
    bypass_cache: Whether to bypass cache.

  Returns:
    A JSON string or Python dict containing structured utilization metrics.

  Raises:
    ValueError: If neither session_id nor raw_bytes is provided.
    FileNotFoundError: If no utilization data is found for the session.
    RuntimeError: If computing or fetching utilization fails.
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

  # Mode 1: Direct in-memory proto bytes (e.g. KernelBench/offline)
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

  # Mode 2: Local, CNS, or x20 file path
  elif (
      session_id.startswith("/")
      or session_id.startswith("cns/")
      or session_id.startswith("x20/")
  ):
    file_path = session_id
    if file_path.startswith("cns/") or file_path.startswith("x20/"):
      file_path = "/" + file_path
    with open(file_path, "rb") as f:
      file_bytes = f.read()
    raw_data, _ = convert.xspace_to_tools_data_from_byte_string(
        [file_bytes], [file_path], "kernel_utilization", params
    )
    if not raw_data:
      raise RuntimeError(
          f"Failed to compute utilization from file {file_path!r}: no data"
          " returned."
      )
    if isinstance(raw_data, bytes):
      decoded_str = raw_data.decode("utf-8", errors="replace")
    else:
      decoded_str = str(raw_data)

  # Mode 3: Session ID lookup via xprof_client
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

  if output_format == "dict":
    return json.loads(decoded_str)
  return decoded_str
