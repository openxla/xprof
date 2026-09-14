"""Tool to calculate TPU hardware kernel compute utilization in OSS."""

import json
import logging
import pathlib
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

  if output_format == "dict":
    return json.loads(decoded_str)
  return decoded_str
