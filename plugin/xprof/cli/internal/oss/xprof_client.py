"""Local XProf Client using OSS xprof converters."""

from collections.abc import Sequence
import hashlib
import logging
import os
import pathlib
import tempfile
from typing import Any

# pylint: disable=g-import-not-at-top
try:
  from xprof.convert import raw_to_tool_data as convert  # pyrefly: ignore[missing-import]
except ImportError:
  from xprof.convert import raw_to_tool_data as convert  # pyrefly: ignore[missing-import]


KNOWN_TOOLS: frozenset[str] = frozenset({
    "overview_page",
    "input_pipeline_analyzer",
    "framework_op_stats",
    "kernel_stats",
    "memory_profile",
    "pod_viewer",
    "op_profile",
    "hlo_op_profile",
    "hlo_stats",
    "roofline_model",
    "graph_viewer",
    "memory_viewer",
    "megascale_stats",
    "inference_profile",
    "perf_counters",
    "utilization_viewer",
    "kernel_utilization",
    "smart_suggestion",
    "trace_viewer",
    "trace_viewer@",
})


class LocalXprofClient:
  """A client for processing local trace files using OSS converters."""

  def __init__(self, logdir: str | None = None):
    """Initializes the instance.

    Args:
      logdir: The base directory where profile runs are stored. Typically, runs
        are in <logdir>/plugins/profile/<run_name>.
    """
    self._logdir = pathlib.Path(logdir).expanduser() if logdir else None

  def set_logdir(self, logdir: str | None):
    """Sets the log directory for the client.

    Args:
      logdir: The base directory where profile runs are stored.
    """
    self._logdir = (
        pathlib.Path(logdir).expanduser() if logdir is not None else None
    )

  @property
  def logdir(self) -> pathlib.Path | None:
    """The current logdir."""
    return self._logdir

  def get_run_dir(self, session_id: str | None = None) -> pathlib.Path:
    """Resolves the run directory for a given session_id (run name).

    Args:
      session_id: The session ID, run name, or direct directory/file path.

    Returns:
      A pathlib.Path to the run directory.

    Raises:
      ValueError: If neither session_id nor logdir is specified.
      FileNotFoundError: If the run directory cannot be found.
    """
    if session_id:
      try:
        session_path = pathlib.Path(str(session_id)).expanduser()
        if session_path.is_file() and session_path.exists():
          return session_path.parent
        if session_path.is_dir() and session_path.exists():
          plugins_dir = session_path / "plugins" / "profile"
          if plugins_dir.is_dir() and plugins_dir.exists():
            subdirs = sorted([d for d in plugins_dir.iterdir() if d.is_dir()])
            if subdirs:
              return subdirs[-1]
          return session_path
      except (ValueError, TypeError, RuntimeError, OSError):
        pass

    if not self._logdir:
      if session_id:
        raise FileNotFoundError(f"Path not found: {session_id}")
      raise ValueError("Logdir not set. Please configure logdir first.")

    if not session_id:
      plugins_dir = self._logdir / "plugins" / "profile"
      if plugins_dir.is_dir() and plugins_dir.exists():
        subdirs = sorted([d for d in plugins_dir.iterdir() if d.is_dir()])
        if subdirs:
          return subdirs[-1]
      return self._logdir

    # Session ID is treated as the run name.
    # Standard TensorBoard structure: <logdir>/plugins/profile/<run>/.
    run_dir = self._logdir / "plugins" / "profile" / str(session_id)
    if not run_dir.exists():
      # Try fallback to formatted date string if fire stripped underscores.
      session_id_str = str(session_id)
      if len(session_id_str) == 14 and session_id_str.isdigit():
        formatted_id = (
            f"{session_id_str[:4]}_{session_id_str[4:6]}_{session_id_str[6:8]}"
            f"_{session_id_str[8:10]}_{session_id_str[10:12]}_{session_id_str[12:14]}"
        )
        formatted_dir = self._logdir / "plugins" / "profile" / formatted_id
        if formatted_dir.exists():
          return formatted_dir

      # Try fallback to direct logdir/run if plugins/profile is missing.
      fallback_dir = self._logdir / str(session_id)
      if fallback_dir.exists():
        return fallback_dir
      raise FileNotFoundError(
          f"Run directory not found for session {session_id!r} in"
          f" {self._logdir}"
      )
    return run_dir

  def get_xspace_paths(self, run_dir: pathlib.Path | str) -> Sequence[str]:
    """Finds all .xplane.pb or .xspace.pb files in the run directory or path.

    Args:
      run_dir: The directory or file path to search within.

    Returns:
      A sorted list of paths to the found files.

    Raises:
      FileNotFoundError: If no .xplane.pb or .xspace.pb files are found.
    """
    p = pathlib.Path(run_dir).expanduser()
    if p.is_file() and (
        p.name.endswith(".xplane.pb") or p.name.endswith(".xspace.pb")
    ):
      return [str(p)]

    paths = []
    for pattern in ("**/*.xplane.pb", "**/*.xspace.pb"):
      paths.extend(str(x) for x in p.glob(pattern))
    if not paths:
      raise FileNotFoundError(
          f"No .xplane.pb or .xspace.pb files found in {run_dir}"
      )
    return sorted(list(set(paths)))

  def fetch(
      self,
      tool_name: str,
      session_id: str,
      rpc_deadline_s: int = 600,
      **kwargs,
  ) -> tuple[Any, Any]:
    """Fetches tool data by converting local traces.

    Args:
      tool_name: e.g. 'overview_page.json', 'memory_profile.json'
      session_id: The run name (directory name under logdir/plugins/profile/)
      rpc_deadline_s: Ignored in local mode.
      **kwargs: Additional tool parameters.

    Returns:
      A tuple (content_type, data), where content_type is the MIME type string
      of the returned data, and data is the tool data payload.

    Raises:
      ValueError: If the logdir has not been set (from `get_run_dir`).
      FileNotFoundError: If the run directory or trace files are not found
        (from `get_run_dir` or `get_xspace_paths`).
    """
    del rpc_deadline_s  # Ignored in local mode.
    logging.info(
        "Fetching profile data locally: tool=%s, run=%s",
        tool_name,
        session_id,
    )

    # Map CLI tool names to TB plugin tool names if needed.
    # Standard tools: overview_page.json, memory_profile.json,
    # hlo_op_profile.json, graph_viewer.
    # Convert accepts: overview_page, memory_profile, op_profile, etc.
    tb_tool = tool_name[:-5] if tool_name.endswith(".json") else tool_name
    if tb_tool == "hlo_op_profile":
      tb_tool = "op_profile"

    if tb_tool not in KNOWN_TOOLS:
      raise ValueError(f"Unknown XProf tool name: {tool_name!r}")

    run_dir = self.get_run_dir(session_id)
    xspace_paths = self.get_xspace_paths(run_dir)

    fetch_params = dict(kwargs)
    bypass_cache = fetch_params.pop("bypass_cache", False)
    try:
      from xprof.cli.internal import decorators  # pyrefly: ignore[missing-import]
    except ImportError:
      from xprof.cli.internal import decorators  # pyrefly: ignore[missing-import]

    if xspace_paths:
      try:
        current_fp = decorators.compute_path_fingerprint(
            run_dir, xspace_paths=xspace_paths
        )
      except Exception:  # pylint: disable=broad-exception-caught
        current_fp = "NO_TRACE_INPUTS"
    else:
      current_fp = "NO_TRACE_INPUTS"

    fp_dir = decorators.get_cache_dir() / "fingerprints"
    fp_dir.mkdir(parents=True, exist_ok=True)
    run_dir_hash = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()[:16]
    fp_file = fp_dir / f"{run_dir_hash}.fp"

    stored_fp = None
    if fp_file.exists():
      try:
        stored_fp = fp_file.read_text(encoding="utf-8").strip()
      except Exception:  # pylint: disable=broad-exception-caught
        pass

    is_fresh = (
        (stored_fp is None)
        or (current_fp == "NO_TRACE_INPUTS")
        or (stored_fp != current_fp)
    )

    if bypass_cache or is_fresh:
      fetch_params["use_saved_result"] = "0"
      try:
        for osp in pathlib.Path(run_dir).glob("*op_stats*.pb"):
          if osp.is_file():
            osp.unlink(missing_ok=True)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Failed to reset stale op_stats files: %s", e)

      if current_fp and current_fp != "NO_TRACE_INPUTS":
        try:
          fd, tmp_path = tempfile.mkstemp(dir=fp_dir, prefix="fp_tmp_")
          with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(current_fp)
          os.replace(tmp_path, fp_file)
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.warning("Failed to write fingerprint file: %s", e)
    else:
      fetch_params["use_saved_result"] = "1"

    data, content_type = convert.xspace_to_tool_data(
        xspace_paths=xspace_paths, tool=tb_tool, params=fetch_params
    )

    return content_type, data

  def get_hosts(
      self,
      session_id: str,
      rpc_deadline_s: int = 600,
      with_metadata: bool = False,
  ) -> Any:
    """Returns hostnames from the trace files in the run directory.

    Args:
      session_id: The run name (directory name under logdir/plugins/profile/).
      rpc_deadline_s: Ignored in local mode.
      with_metadata: If true, returns a list of dictionaries with 'hostname'
        keys. Otherwise, returns a list of hostnames.

    Returns:
      A list of hostnames or hostname metadata.

    Raises:
      ValueError: If the logdir has not been set (from `get_run_dir`).
      FileNotFoundError: If the run directory or trace files are not found
        (from `get_run_dir` or `get_xspace_paths`).
    """
    del rpc_deadline_s  # Ignored in local mode.
    run_dir = self.get_run_dir(session_id)
    xspace_paths = self.get_xspace_paths(run_dir)

    hosts = []
    for path in xspace_paths:
      p = pathlib.Path(path)
      stem = p.name.removesuffix(".xplane.pb")
      parts = stem.split(".")
      hostname = parts[-1] if parts else stem
      hosts.append(hostname)

    hosts = sorted(set(hosts))
    if with_metadata:
      return [{"hostname": h} for h in hosts]
    return hosts

  def get_serialized_xspace(
      self, session_id: str, host: str = "", **kwargs
  ) -> bytes:
    """Returns the raw serialized XSpace data for the session.

    Args:
      session_id: The run name (directory name under logdir/plugins/profile/).
      host: The specific host to fetch data for.
      **kwargs: Additional parameters (ignored in OSS).

    Returns:
      The raw bytes of the serialized XSpace.

    Raises:
      ValueError: If the logdir has not been set (from `get_run_dir`).
      FileNotFoundError: If the run directory or trace files are not found.
      NotImplementedError: If multiple XSpace files are found for the host.
    """
    del kwargs
    run_dir = self.get_run_dir(session_id)
    xspace_paths = self.get_xspace_paths(run_dir)
    if not xspace_paths:
      raise FileNotFoundError(f"No traces found for session {session_id!r}")

    if host:
      filtered_paths = []
      for path in xspace_paths:
        p = pathlib.Path(path)
        stem = p.name.removesuffix(".xplane.pb")
        parts = stem.split(".")
        hostname = parts[-1] if parts else stem
        if hostname == host:
          filtered_paths.append(path)
      xspace_paths = filtered_paths
      if not xspace_paths:
        raise FileNotFoundError(
            f"No traces found for host {host!r} in session {session_id!r}"
        )

    # For single-host, just return the raw file bytes directly.
    if len(xspace_paths) == 1:
      with open(xspace_paths[0], "rb") as f:
        return f.read()

    raise NotImplementedError(
        "Multi-host XSpace serialization is not supported in OSS because"
        " xplane_pb2 is not exposed."
    )


# Global instance
_INSTANCE: LocalXprofClient | None = None


def get_client() -> LocalXprofClient:
  """Gets the global singleton instance of LocalXprofClient.

  Returns:
    A LocalXprofClient instance.
  """
  global _INSTANCE
  if _INSTANCE is None:
    _INSTANCE = XprofAnalysisClient()
  return _INSTANCE


def set_client(client: LocalXprofClient):
  """Sets the global singleton instance of LocalXprofClient.

  Args:
    client: A LocalXprofClient instance.
  """
  global _INSTANCE
  _INSTANCE = client


# Compatibility aliases for Google3 tests migration
CachedXprofClient = LocalXprofClient
XprofAnalysisClient = LocalXprofClient
set_client_override = set_client
