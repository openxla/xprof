"""Local XProf Client using OSS xprof converters."""

from collections.abc import Sequence
import logging
import pathlib
from typing import Any

from xprof.convert import raw_to_tool_data as convert  # pytype: disable=import-error


class LocalXprofClient:
  """A client for processing local trace files using OSS converters."""

  def __init__(self, logdir: str | None = None):
    """Initializes the instance.

    Args:
      logdir: The base directory where profile runs are stored. Typically, runs
        are in <logdir>/plugins/profile/<run_name>.
    """
    self._logdir = pathlib.Path(logdir).expanduser() if logdir else None

  def set_logdir(self, logdir: str):
    """Sets the log directory for the client.

    Args:
      logdir: The base directory where profile runs are stored.
    """
    self._logdir = pathlib.Path(logdir).expanduser()

  @property
  def logdir(self) -> pathlib.Path | None:
    """The current logdir."""
    return self._logdir

  def get_run_dir(self, session_id: str) -> pathlib.Path:
    """Resolves the run directory for a given session_id (run name).

    Args:
      session_id: The session ID, which is treated as the run name.

    Returns:
      A pathlib.Path to the run directory.

    Raises:
      ValueError: If the logdir has not been set.
      FileNotFoundError: If the run directory cannot be found.
    """
    if not self._logdir:
      raise ValueError("Logdir not set. Please configure logdir first.")

    # Session ID is treated as the run name.
    # Standard TensorBoard structure: <logdir>/plugins/profile/<run>/.
    run_dir = self._logdir / "plugins" / "profile" / str(session_id)
    if not run_dir.exists():
      # Try fallback to formatted date string if it looks like fire stripped
      # underscores.
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

  def get_xspace_paths(self, run_dir: pathlib.Path) -> Sequence[str]:
    """Finds all .xplane.pb or .xspace.pb files in the run directory.

    Args:
      run_dir: The directory to search within.

    Returns:
      A sorted list of paths to the found files.

    Raises:
      FileNotFoundError: If no .xplane.pb or .xspace.pb files are found.
    """
    paths = []
    for pattern in ("*.xplane.pb", "*.xspace.pb"):
      paths.extend(str(p) for p in run_dir.glob(pattern))
    if not paths:
      raise FileNotFoundError(
          f"No .xplane.pb or .xspace.pb files found in {run_dir}"
      )
    return sorted(paths)

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

    run_dir = self.get_run_dir(session_id)
    xspace_paths = self.get_xspace_paths(run_dir)

    data, content_type = convert.xspace_to_tool_data(
        xspace_paths=xspace_paths, tool=tb_tool, params=kwargs
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

  def get_serialized_xspace(self, session_id: str) -> bytes:
    """Returns the raw serialized XSpace data for the session.

    Args:
      session_id: The run name (directory name under logdir/plugins/profile/).

    Returns:
      The raw bytes of the serialized XSpace.

    Raises:
      ValueError: If the logdir has not been set (from `get_run_dir`).
      FileNotFoundError: If the run directory or trace files are not found
        (from `get_run_dir` or `get_xspace_paths`).
      NotImplementedError: If multiple XSpace files are found, as multi-host
        serialization is not supported.
    """
    run_dir = self.get_run_dir(session_id)
    xspace_paths = self.get_xspace_paths(run_dir)
    if not xspace_paths:
      raise FileNotFoundError(f"No traces found for session {session_id!r}")

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
