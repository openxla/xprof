"""Local XProf Client using OSS xprof converters."""

from collections.abc import Sequence
import hashlib
import logging
import os
import pathlib
import tempfile
from typing import Any

from xprof.cli.internal import trace_selection

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

# Converters that reject more than one XSpace (`XSpaceSize() != 1` in
# `convert/xplane_to_tools_data.cc`).
SINGLE_HOST_TB_TOOLS: frozenset[str] = frozenset({
    "memory_profile",
    "kernel_utilization",
    "utilization_viewer",
})

trace_stem = trace_selection.trace_stem
trace_host_name = trace_selection.trace_host_name
format_trace_list = trace_selection.format_trace_list


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

  def is_local_session(self, session_id: str) -> bool:
    """Returns True; every OSS session is served from local trace files."""
    del session_id
    return True

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
      run_dir: The session ID, directory, or file path to search within.

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
    # Everything else (including plain directories) is resolved by
    # `get_run_dir`, which maps a logdir root to its latest
    # `plugins/profile/<run>` subdirectory and returns other directories
    # unchanged. Short-circuiting on `p.is_dir()` here would silently merge
    # every run under a logdir root.
    p = self.get_run_dir(str(run_dir))

    paths = []
    for pattern in ("**/*.xplane.pb", "**/*.xspace.pb"):
      paths.extend(str(x) for x in p.glob(pattern))
    if not paths:
      raise FileNotFoundError(
          f"No .xplane.pb or .xspace.pb files found in {run_dir}"
      )
    return _dedup_by_realpath(paths)

  def list_run_traces(self, session_id: str) -> Sequence[str]:
    """Lists every trace in the run directory that `session_id` resolves to.

    Unlike `get_xspace_paths`, a file input lists all of its siblings, so the
    caller can report which traces were left out.

    Args:
      session_id: The session ID, run name, or direct directory/file path.

    Returns:
      A sorted list of trace paths, with symlinked duplicates removed.
    """
    run_dir = self.get_run_dir(session_id)
    paths = []
    for pattern in ("**/*.xplane.pb", "**/*.xspace.pb"):
      paths.extend(str(x) for x in run_dir.glob(pattern))
    return _dedup_by_realpath(paths)

  def resolve_host_path(self, session_id: str, host: str) -> str:
    """Resolves `--host` to exactly one trace file of the session's run.

    A host matches a trace when it equals the file stem or the name that
    `get_hosts` reports for the file.

    Args:
      session_id: The session ID, run name, or direct directory/file path.
      host: The host name or file stem to select.

    Returns:
      The path of the matching trace file.

    Raises:
      ValueError: If no trace or more than one trace matches `host`, or if
        `session_id` names a different trace file.
    """
    candidates = self.list_run_traces(session_id)
    matches = [
        p for p in candidates if host in (trace_stem(p), trace_host_name(p))
    ]
    if len(matches) > 1:
      exact = [p for p in matches if trace_stem(p) == host]
      matches = exact or matches
    if not matches:
      raise ValueError(
          f"Unknown host {host!r}. Available traces (pass a file stem or a"
          f" `get_hosts` name):\n{format_trace_list(candidates)}"
      )
    if len(matches) > 1:
      raise ValueError(
          f"Host {host!r} matches {len(matches)} traces; pass the full file"
          f" stem instead:\n{format_trace_list(matches)}"
      )
    selected = matches[0]
    session_path = pathlib.Path(str(session_id)).expanduser()
    if session_path.is_file() and os.path.realpath(
        session_path
    ) != os.path.realpath(selected):
      raise ValueError(
          f"Conflicting selection: session is {trace_stem(session_path)!r}"
          f" but --host selects {trace_stem(selected)!r}. Pass either a file"
          " path or a directory with --host."
      )
    return selected

  def select_paths(
      self,
      session_id: str,
      host: str | None = None,
      hosts: Sequence[str] | str | None = None,
  ) -> Sequence[str]:
    """Returns the traces a tool reads, honouring `host` / `hosts` filters.

    Args:
      session_id: The session ID, run name, or direct directory/file path.
      host: Optional single host name or file stem.
      hosts: Optional host names or file stems, as a list or comma-separated
        string.

    Returns:
      The sorted list of selected trace paths.

    Raises:
      ValueError: If a requested host does not match any trace.
      FileNotFoundError: If no traces are found.
    """
    names: list[str] = []
    if host:
      names.append(str(host))
    if isinstance(hosts, str):
      # Fire leaves `--hosts=[a-b,c]` unparsed when names contain hyphens.
      names.extend(
          h.strip().strip("'\"")
          for h in hosts.strip().strip("[]").split(",")
          if h.strip().strip("'\"")
      )
    elif hosts:
      names.extend(str(h) for h in hosts if h)
    if not names:
      return self.get_xspace_paths(session_id)
    return sorted({self.resolve_host_path(session_id, n) for n in names})

  def describe_capture(
      self,
      session_id: str,
      paths: Sequence[str] | None = None,
      combine: str = "merged",
  ) -> dict[str, Any]:
    """Describes which traces a local tool call reads.

    Args:
      session_id: The session ID, run name, or direct directory/file path.
      paths: The traces the tool reads. Defaults to `get_xspace_paths`.
      combine: How the tool combines several traces: "summed" (Python plane
        tools add values), "listed" (rows from every trace are concatenated),
        "merged" (C++ all-hosts view; totals across hosts), or "none" (no
        combine warning, e.g. `get_hosts`).

    Returns:
      A dict with `input`, `run`, `files_used`, `files_available`,
      `combined` and `warnings`.
    """
    run_dir = self.get_run_dir(session_id)
    used = (
        list(paths) if paths is not None else self.get_xspace_paths(session_id)
    )
    available = self.list_run_traces(session_id)
    warnings: list[str] = []

    input_path = pathlib.Path(str(session_id)).expanduser()
    plugins_dir = input_path / "plugins" / "profile"
    if input_path.is_dir() and plugins_dir.is_dir():
      other_runs = sorted(
          d.name
          for d in plugins_dir.iterdir()
          if d.is_dir() and d.resolve() != run_dir.resolve()
      )
      if other_runs:
        warnings.append(
            f"Used latest run {run_dir.name!r}; ignored"
            f" {len(other_runs)} other run(s): {other_runs}. Pass a run"
            " directory to choose a different run."
        )

    combined = len(used) > 1
    if combined and combine == "summed":
      warnings.append(
          f"Values combined across {len(used)} traces (summed for counts"
          " and durations). Pass one .xplane.pb file or --host=<name> to"
          " analyze a single host/rank."
      )
    elif combined and combine == "listed":
      warnings.append(
          f"Events listed from {len(used)} traces; rows from different"
          " hosts/ranks are mixed. Pass one .xplane.pb file or"
          " --host=<name> for a single host/rank."
      )
    elif combined and combine == "merged":
      warnings.append(
          f"Results merged across {len(used)} traces (XProf all-hosts"
          " view): times and counts are totals across hosts, not per-host"
          " values. Pass one .xplane.pb file or --host=<name> for a single"
          " host/rank."
      )

    return {
        "input": str(session_id),
        "run": run_dir.name,
        "files_used": [trace_stem(p) for p in used],
        "files_available": [trace_stem(p) for p in available],
        "combined": combined,
        "warnings": warnings,
    }

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
    xspace_paths = self.select_paths(
        session_id, host=kwargs.get("host"), hosts=kwargs.get("hosts")
    )
    if tb_tool in SINGLE_HOST_TB_TOOLS and len(xspace_paths) > 1:
      raise ValueError(
          f"{tb_tool} needs exactly one trace, but {len(xspace_paths)} were"
          f" found in {run_dir}:\n{format_trace_list(xspace_paths)}\nPass one"
          " .xplane.pb file path or --host=<name>."
      )

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
    # Key on the resolved trace scope, not just the containing directory.
    # Sibling ranks of a multi-host capture live in one directory, so a
    # directory-only key would make rank0 and rank2 share freshness state.
    scope_key = "|".join([str(run_dir), *xspace_paths])
    scope_hash = hashlib.sha256(scope_key.encode("utf-8")).hexdigest()[:16]
    fp_file = fp_dir / f"{scope_hash}.fp"

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
    xspace_paths = self.get_xspace_paths(session_id)

    hosts = sorted({trace_host_name(p) for p in xspace_paths})
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
      ValueError: If the logdir has not been set (from `get_run_dir`), if
        `host` is unknown, or if more than one trace is selected.
      FileNotFoundError: If the run directory or trace files are not found.
    """
    del kwargs
    xspace_paths = self.select_paths(session_id, host=host)
    if not xspace_paths:
      raise FileNotFoundError(f"No traces found for session {session_id!r}")

    # For single-host, just return the raw file bytes directly.
    if len(xspace_paths) == 1:
      with open(xspace_paths[0], "rb") as f:
        return f.read()

    raise ValueError(
        f"Raw XSpace export needs exactly one trace, but {len(xspace_paths)}"
        f" were found:\n{format_trace_list(xspace_paths)}\nPass one"
        " .xplane.pb file path or --host=<name>."
    )


def _dedup_by_realpath(paths: Sequence[str]) -> list[str]:
  """Sorts paths and drops later entries that resolve to the same file."""
  seen: set[str] = set()
  result: list[str] = []
  for path in sorted(set(paths)):
    real = os.path.realpath(path)
    if real not in seen:
      seen.add(real)
      result.append(path)
  return result


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
