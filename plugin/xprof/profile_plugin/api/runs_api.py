# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Frontend run listing and per-run tools (/runs, /run_tools)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from werkzeug import wrappers

from xprof.profile_plugin.http.respond import respond


class RunsApiMixin:
  """HTTP handlers for this frontend API group (mixed into ProfilePlugin)."""

  @wrappers.Request.application
  def runs_route(self, request: wrappers.Request) -> wrappers.Response:
    # pytype: enable=wrong-arg-types
    runs = self.runs_imp(request)
    return respond(runs, 'application/json')


  def _session_dir_by_run_name_from_request(
      self, request: wrappers.Request | None = None
  ) -> Mapping[str, str] | None:
    """Returns a map of run names to session directories from the request.

    Args:
      request: Optional; werkzeug request used for grabbing session_path and
        run_path arguments.
    """
    session_path_arg = request.args.get('session_path') if request else None
    run_path_arg = (
        request.args.get('run_path')
        if request and not session_path_arg
        else None
    )
    return self._sessions.run_map_from_params(session_path_arg, run_path_arg)


  def _run_dir(
      self, run: str, request: wrappers.Request | None = None
  ) -> str | None:
    """Helper that maps a frontend run name to a profile "run" directory.

    The frontend run name consists of the TensorBoard run name (aka the relative
    path from the logdir root to the directory containing the data) path-joined
    to the Profile plugin's "run" concept (which is a subdirectory of the
    plugins/profile directory representing an individual run of the tool), with
    the special case that TensorBoard run is the logdir root (which is the run
    named '.') then only the Profile plugin "run" name is used, for backwards
    compatibility.

    Args:
      run: the frontend run name, as described above, e.g. train/run1.
      request: Optional; werkzeug request used for grabbing session_path and
        run_path arguments.

    Returns:
      The resolved directory path, e.g. /logdir/train/plugins/profile/run1.

    Raises:
      ValueError: If the run is not found in the run map.
      RuntimeError: If the run directory is not found.
    """
    session_dir_by_run_name = self._session_dir_by_run_name_from_request(
        request
    )
    return self._sessions.resolve_run_dir(
        run,
        session_dir_by_run_name,
        self.logdir,
        self._run_to_profile_run_dir,
        self._run_dir_cache_lock,
    )


  def runs_imp(self, request: wrappers.Request | None = None) -> list[str]:
    """Returns a list all runs for the profile plugin.

    Args:
      request: Optional; werkzeug request used for grabbing ctx and experiment
        id for other host implementations
    """
    session_dir_by_run_name = self._session_dir_by_run_name_from_request(
        request
    )
    if session_dir_by_run_name is not None:
      runs = session_dir_by_run_name.keys()
    else:
      runs = self.generate_runs()
    return sorted(runs, reverse=True)


  @wrappers.Request.application
  def run_tools_route(self, request: wrappers.Request) -> wrappers.Response:
    # pytype: enable=wrong-arg-types
    run = request.args.get('run')
    run_tools = self.run_tools_imp(run, request)
    return respond(run_tools, 'application/json')


  def run_tools_imp(
      self, run, request: wrappers.Request | None = None
  ) -> list[str]:
    """Returns a list of tools given a single run.

    Args:
      run: the frontend run name, item is list returned by runs_imp
      request: Optional; werkzeug request used for grabbing ctx and experiment
        id for other host implementations
    """
    run_dir = self._run_dir(run, request)
    return list(self.generate_tools_of_run(run, run_dir))


  def generate_runs(self) -> Iterator[str]:
    """Yield frontend run names under logdir (delegates to RunDiscovery)."""
    yield from self._run_discovery.iter_frontend_runs()


  def generate_tools_of_run(self, run: str, run_dir: str) -> Iterator[str]:
    """Yield tools for a run (delegates to RunDiscovery)."""
    yield from self._run_discovery.tools_of_run(run, run_dir)


