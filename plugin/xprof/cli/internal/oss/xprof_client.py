"""Local XProf Client using OSS xprof converters."""

from collections.abc import Iterator, Sequence
import contextlib
import dataclasses
import hashlib
import logging
import os
import pathlib
import tempfile
from typing import Any

from xprof.cli.internal import trace_selection

# pylint: disable=g-import-not-at-top
try:
  import fcntl
except ImportError:  # pragma: no cover
  fcntl = None  # type: ignore[assignment]

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

# Converters that read or write shared per-directory `*op_stats*.pb` (or
# `*smart_suggestion*.pb`) files in the run directory.
OP_STATS_TB_TOOLS: frozenset[str] = frozenset({
    "overview_page",
    "input_pipeline_analyzer",
    "framework_op_stats",
    "kernel_stats",
    "pod_viewer",
    "op_profile",
    "hlo_stats",
    "roofline_model",
    "smart_suggestion",
})

trace_stem = trace_selection.trace_stem
trace_host_name = trace_selection.trace_host_name
trace_labels = trace_selection.trace_labels
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

  def _resolve_op_stats_scope(
      self,
      session_id: str,
      xspace_paths: Sequence[str] | None = None,
  ) -> tuple[pathlib.Path, "_OpStatsScope"] | None:
    """Resolves the run directory and op_stats scope metadata for `session_id`."""
    try:
      run_dir = self.get_run_dir(session_id)
      paths = (
          list(xspace_paths)
          if xspace_paths is not None
          else list(self.select_paths(session_id))
      )
    except (OSError, ValueError):
      return None
    if not paths:
      return None
    run_dir_path = pathlib.Path(run_dir)
    return run_dir_path, _compute_op_stats_scope(run_dir_path, paths)

  def get_owned_op_stats_path(
      self,
      session_id: str,
      xspace_paths: Sequence[str] | None = None,
  ) -> pathlib.Path | None:
    """Returns `ALL_HOSTS.op_stats_v2.pb` if owned by the given trace scope.

    Verifies under the run-directory lock that the saved
    `ALL_HOSTS.op_stats_v2.pb` in the session's run directory was written for
    the exact `xspace_paths` scope and current trace fingerprint.

    Args:
      session_id: The session ID, run name, or direct directory/file path.
      xspace_paths: Selected trace paths. Defaults to
        `select_paths(session_id)`.

    Returns:
      Path to `ALL_HOSTS.op_stats_v2.pb` when owned by this scope, or None.
    """
    resolved = self._resolve_op_stats_scope(session_id, xspace_paths)
    if resolved is None:
      return None
    run_dir_path, scope = resolved
    with _lock_run_dir(scope.owner_dir, scope.run_dir_hash):
      return _find_owned_op_stats_unlocked(run_dir_path, scope)

  def read_owned_op_stats_bytes(
      self,
      session_id: str,
      xspace_paths: Sequence[str] | None = None,
  ) -> bytes | None:
    """Reads `ALL_HOSTS.op_stats_v2.pb` under the run-directory lock if owned.

    Holds `_lock_run_dir` across both the ownership check and file read so a
    concurrent CLI invocation cannot swap the file between check and read.

    Args:
      session_id: The session ID, run name, or direct directory/file path.
      xspace_paths: Selected trace paths. Defaults to
        `select_paths(session_id)`.

    Returns:
      Raw bytes of `ALL_HOSTS.op_stats_v2.pb` when owned by this scope, or None.
    """
    resolved = self._resolve_op_stats_scope(session_id, xspace_paths)
    if resolved is None:
      return None
    run_dir_path, scope = resolved
    with _lock_run_dir(scope.owner_dir, scope.run_dir_hash):
      op_stats_path = _find_owned_op_stats_unlocked(run_dir_path, scope)
      if op_stats_path is None:
        return None
      try:
        return op_stats_path.read_bytes()
      except OSError:
        logging.exception("Failed to read %s", op_stats_path)
        return None

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
    run_dir_path = pathlib.Path(run_dir)
    scope = _compute_op_stats_scope(run_dir_path, xspace_paths)
    scope.fp_dir.mkdir(parents=True, exist_ok=True)

    if tb_tool not in OP_STATS_TB_TOOLS:
      stored_fp = _read_text_or_none(scope.fp_file)
      is_fresh = (
          (stored_fp is None)
          or (scope.current_fp == "NO_TRACE_INPUTS")
          or (stored_fp != scope.current_fp)
      )
      if (
          is_fresh
          and scope.current_fp
          and scope.current_fp != "NO_TRACE_INPUTS"
      ):
        _atomic_write_text(scope.fp_dir, scope.fp_file, scope.current_fp)
      if bypass_cache:
        fetch_params["use_saved_result"] = "0"
        _unlink_quietly(scope.owner_file)
        _remove_op_stats_files(run_dir_path)
      else:
        # Non-op_stats tools do not read `ALL_HOSTS.op_stats_v2.pb`, and
        # passing `use_saved_result='0'` to C++ calls `ClearCacheFiles()` on
        # the whole run directory.
        fetch_params["use_saved_result"] = "1"
      data, content_type = convert.xspace_to_tool_data(
          xspace_paths=xspace_paths, tool=tb_tool, params=fetch_params
      )
      return content_type, data

    # The native converter saves one set of `*op_stats*.pb` files per run
    # directory (e.g. `ALL_HOSTS.op_stats_v2.pb`), whatever the selected
    # scope. Serialize conversions per run directory and reuse the saved
    # files only when this scope and trace fingerprint wrote the exact files
    # currently on disk.
    with _lock_run_dir(scope.owner_dir, scope.run_dir_hash):
      stored_fp = _read_text_or_none(scope.fp_file)
      is_fresh = (
          (stored_fp is None)
          or (scope.current_fp == "NO_TRACE_INPUTS")
          or (stored_fp != scope.current_fp)
      )
      reuse_saved_result = (
          not bypass_cache
          and not is_fresh
          and _has_valid_op_stats_owner(scope, run_dir_path)
      )

      claim_saved_result = False
      if reuse_saved_result:
        fetch_params["use_saved_result"] = "1"
      else:
        fetch_params["use_saved_result"] = "0"
        _unlink_quietly(scope.owner_file)
        # Claim the saved files after conversion only if no foreign op_stats
        # file survived the reset (e.g. in a read-only directory); then any
        # op_stats file in run_dir comes from this conversion.
        claim_saved_result = _remove_op_stats_files(run_dir_path)

        if scope.current_fp and scope.current_fp != "NO_TRACE_INPUTS":
          _atomic_write_text(scope.fp_dir, scope.fp_file, scope.current_fp)
        else:
          claim_saved_result = False

      data, content_type = convert.xspace_to_tool_data(
          xspace_paths=xspace_paths, tool=tb_tool, params=fetch_params
      )

      if claim_saved_result:
        post_sig = _op_stats_signature(run_dir_path)
        if post_sig:
          _atomic_write_text(
              scope.owner_dir,
              scope.owner_file,
              f"{scope.scope_hash}:{scope.current_fp}:{post_sig}",
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


@dataclasses.dataclass(frozen=True)
class _OpStatsScope:
  """Resolved fingerprint and ownership metadata for a run directory scope."""

  current_fp: str
  fp_dir: pathlib.Path
  fp_file: pathlib.Path
  scope_hash: str
  owner_dir: pathlib.Path
  run_dir_hash: str
  owner_file: pathlib.Path


def _compute_op_stats_scope(
    run_dir: pathlib.Path, xspace_paths: Sequence[str]
) -> _OpStatsScope:
  """Computes trace fingerprint and owner-file paths for a local trace scope."""
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
  # Canonicalize paths so `./run`, `/abs/run`, and symlinks share one key,
  # and key on the resolved trace scope rather than the directory alone.
  real_run_dir = os.path.realpath(run_dir)
  real_xspace_paths = [os.path.realpath(p) for p in xspace_paths]
  scope_key = "|".join([real_run_dir, *real_xspace_paths])
  scope_hash = hashlib.sha256(scope_key.encode("utf-8")).hexdigest()[:16]
  fp_file = fp_dir / f"{scope_hash}.fp"

  owner_dir = decorators.get_cache_dir() / "op_stats_owners"
  run_dir_hash = hashlib.sha256(real_run_dir.encode("utf-8")).hexdigest()[:16]
  owner_file = owner_dir / f"{run_dir_hash}.owner"
  return _OpStatsScope(
      current_fp=current_fp,
      fp_dir=fp_dir,
      fp_file=fp_file,
      scope_hash=scope_hash,
      owner_dir=owner_dir,
      run_dir_hash=run_dir_hash,
      owner_file=owner_file,
  )


def _has_valid_op_stats_owner(
    scope: _OpStatsScope, run_dir_path: pathlib.Path
) -> bool:
  """Returns True if `scope.owner_file` matches the current scope and files."""
  if not scope.current_fp or scope.current_fp == "NO_TRACE_INPUTS":
    return False
  saved_sig = _op_stats_signature(run_dir_path)
  return (
      bool(saved_sig)
      and _read_text_or_none(scope.owner_file)
      == f"{scope.scope_hash}:{scope.current_fp}:{saved_sig}"
  )


def _find_owned_op_stats_unlocked(
    run_dir_path: pathlib.Path, scope: _OpStatsScope
) -> pathlib.Path | None:
  """Returns `ALL_HOSTS.op_stats_v2.pb` in `run_dir_path` if owned by `scope`."""
  op_stats_path = next(
      (
          p
          for p in _op_stats_files(run_dir_path) or []
          if p.name == "ALL_HOSTS.op_stats_v2.pb"
      ),
      None,
  )
  if op_stats_path is None or not _has_valid_op_stats_owner(
      scope, run_dir_path
  ):
    return None
  return op_stats_path


def _read_text_or_none(path: pathlib.Path) -> str | None:
  """Returns the stripped text of `path`, or None if it cannot be read."""
  try:
    return path.read_text(encoding="utf-8").strip()
  except FileNotFoundError:
    return None
  except (OSError, UnicodeDecodeError):
    logging.exception("Failed to read %s", path)
    return None


def _unlink_quietly(path: pathlib.Path) -> None:
  """Removes `path` if it exists; logs instead of raising on failure."""
  try:
    path.unlink(missing_ok=True)
  except OSError:
    logging.exception("Failed to remove %s", path)


def _op_stats_files(run_dir: pathlib.Path) -> list[pathlib.Path] | None:
  """Lists saved op_stats and smart_suggestion files in `run_dir`."""
  try:
    files: list[pathlib.Path] = []
    for pattern in ("**/*op_stats*.pb", "**/*smart_suggestion*.pb"):
      files.extend(p for p in run_dir.glob(pattern) if p.is_file())
    return sorted(set(files))
  except OSError:
    logging.exception("Failed to list op_stats files in %s", run_dir)
    return None


def _op_stats_signature(run_dir: pathlib.Path) -> str:
  """Returns a stat and content digest of saved op_stats files in `run_dir`."""
  files = _op_stats_files(run_dir)
  if not files:
    return ""
  hasher = hashlib.sha256()
  for path in files:
    try:
      st = path.stat()
      hasher.update(
          f"{path.name}:{st.st_size}:{st.st_mtime_ns}:{st.st_ino}:".encode(
              "utf-8"
          )
      )
      with open(path, "rb") as f:
        hasher.update(f.read(64 * 1024))
        if st.st_size > 128 * 1024:
          f.seek(-64 * 1024, os.SEEK_END)
          hasher.update(f.read(64 * 1024))
    except OSError:
      logging.exception("Failed to stat op_stats file %s", path)
      return ""
  return hasher.hexdigest()[:16]


def _remove_op_stats_files(run_dir: pathlib.Path) -> bool:
  """Removes saved `*op_stats*.pb` and `*smart_suggestion*.pb` files.

  Args:
    run_dir: The run directory that the native converter writes into.

  Returns:
    True if no op_stats file remains in `run_dir`. False if a file could not
    be removed, e.g. because the directory is read-only.
  """
  stale_files = _op_stats_files(run_dir)
  if stale_files is None:
    return False
  if not stale_files:
    return True
  if not os.access(run_dir, os.W_OK):
    logging.warning(
        "Run directory %s is read-only; cannot remove %d stale op_stats"
        " file(s).",
        run_dir,
        len(stale_files),
    )
    return False
  all_removed = True
  for path in stale_files:
    try:
      path.unlink(missing_ok=True)
    except OSError:
      logging.exception("Failed to reset stale op_stats file %s", path)
      all_removed = False
  return all_removed


@contextlib.contextmanager
def _lock_run_dir(
    lock_dir: pathlib.Path, run_dir_hash: str
) -> Iterator[None]:
  """Serializes op_stats conversions for the same run directory."""
  if fcntl is None:
    yield
    return
  lock_file = None
  try:
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{run_dir_hash}.lock"
    lock_file = open(lock_path, "a+", encoding="utf-8")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
  except OSError:
    logging.exception("Failed to lock %s", lock_dir)
    if lock_file is not None:
      lock_file.close()
      lock_file = None
  try:
    yield
  finally:
    if lock_file is not None:
      try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
      except OSError:
        logging.exception("Failed to unlock %s", lock_dir)
      finally:
        lock_file.close()


def _atomic_write_text(
    tmp_dir: pathlib.Path, path: pathlib.Path, text: str
) -> None:
  """Writes `text` to `path` via a temp file in `tmp_dir`; logs on failure."""
  tmp_path = None
  try:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, prefix="fp_tmp_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      f.write(text)
    os.replace(tmp_path, path)
  except OSError:
    logging.exception("Failed to write %s", path)
    if tmp_path is not None:
      _unlink_quietly(pathlib.Path(tmp_path))


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
