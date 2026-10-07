"""Trace selection for local multi-host and multi-rank captures.

A run directory may hold one `.xplane.pb` per host or rank. This module decides
which trace files a CLI tool call reads (`--host` / `--hosts`), rejects
selections a tool cannot honour, and describes the result in a `capture` block
so users can see which traces a number covers.
"""

from collections.abc import Callable, Sequence
import dataclasses
import inspect
import json
import logging
import os
import pathlib
import sys
import types
from typing import Any, Literal

import fire

_TRACE_SUFFIXES: tuple[str, ...] = (".xplane.pb", ".xspace.pb")

Combine = Literal["summed", "listed", "merged", "none"]


@dataclasses.dataclass(frozen=True)
class ToolTraits:
  """How a CLI tool treats a run with several trace files.

  Attributes:
    combine: How values from several traces are combined: "summed" (Python plane
      tools add counts and durations), "listed" (rows from every trace are
      concatenated), "merged" (XProf all-hosts view; times and counts are totals
      across hosts), or "none" (no values are combined).
    single_trace: Whether the tool reads exactly one trace file.
    host_kind: What the tool's own `host` parameter means: "name" (a host or
      file name, routed by `--host=<name>`) or "index" (an integer index that is
      passed through unchanged).
  """

  combine: Combine = "merged"
  single_trace: bool = False
  host_kind: Literal["name", "index"] = "name"


_DEFAULT_TRAITS = ToolTraits()

# Tools whose behaviour differs from the default (merged, multi-trace, name).
TOOL_TRAITS: types.MappingProxyType[str, ToolTraits] = types.MappingProxyType({
    # keep-sorted start
    "aggregate_xplane_events": ToolTraits(combine="summed"),
    "create_events_db": ToolTraits(combine="listed"),
    "get_hlo_module_content": ToolTraits(combine="none"),
    "get_hlo_neighborhood": ToolTraits(combine="none"),
    "get_hlo_text": ToolTraits(combine="none"),
    "get_hosts": ToolTraits(combine="none"),
    "get_kernel_stats": ToolTraits(combine="summed"),
    "get_kernel_utilization": ToolTraits(single_trace=True),
    "get_llo_analysis": ToolTraits(single_trace=True),
    "get_llo_debug_string": ToolTraits(single_trace=True),
    "get_memory_profile": ToolTraits(single_trace=True),
    "get_utilization_viewer": ToolTraits(single_trace=True, host_kind="index"),
    "get_xspace_proto": ToolTraits(single_trace=True),
    "list_xplane_events": ToolTraits(combine="listed"),
    "query_events_db": ToolTraits(combine="listed"),
    # keep-sorted end
})


def traits_for(tool_name: str) -> ToolTraits:
  """Returns the traits of `tool_name`, or the defaults if it has none."""
  return TOOL_TRAITS.get(tool_name, _DEFAULT_TRAITS)


def trace_stem(path: str | os.PathLike[str]) -> str:
  """Returns the file name of a trace without its `.xplane.pb` suffix.

  Args:
    path: Path to a `.xplane.pb` or `.xspace.pb` file.

  Returns:
    The file name with the trace suffix removed.
  """
  name = pathlib.Path(path).name
  for suffix in _TRACE_SUFFIXES:
    if name.endswith(suffix):
      return name.removesuffix(suffix)
  return name


def trace_host_name(path: str | os.PathLike[str]) -> str:
  """Returns the host name that `get_hosts` reports for a trace file.

  Args:
    path: Path to a `.xplane.pb` or `.xspace.pb` file.

  Returns:
    The last dot-separated component of the trace stem.
  """
  return trace_stem(path).split(".")[-1]


def format_trace_list(paths: Sequence[str | os.PathLike[str]]) -> str:
  """Formats trace paths as an indented list of stems for error messages."""
  return "\n".join(f"  {trace_stem(path)}" for path in paths)


def attach_capture(result: Any, capture: dict[str, Any]) -> Any:
  """Adds `capture` to a tool result without changing its shape.

  JSON-object results (dicts or JSON strings) get `capture` as the first key,
  so it survives truncated reads of large outputs. Other results are returned
  unchanged and `capture` is written to stderr as one line prefixed with
  `xprof-capture:`.

  Args:
    result: The tool result.
    capture: The capture description from `describe_capture`.

  Returns:
    The result with `capture` attached where possible.
  """
  if isinstance(result, dict):
    return {"capture": capture, **result}
  if isinstance(result, str) and result.lstrip().startswith("{"):
    try:
      parsed = json.loads(result)
    except ValueError:
      parsed = None
    if isinstance(parsed, dict):
      parsed.pop("capture", None)
      return json.dumps({"capture": capture, **parsed}, indent=2)
  sys.stderr.write(f"xprof-capture: {json.dumps(capture)}\n")
  return result


def _has_empty_host_default(param: inspect.Parameter) -> bool:
  """Returns whether `param` is a `host` parameter with an empty default."""
  return (
      param.name == "host"
      and isinstance(param.default, str)
      and not param.default
  )


class TraceSelector:
  """Applies `--host` / `--hosts` and builds `capture` for one CLI tool.

  Built once per tool from its signature. `extend_params` adjusts the
  signature Fire sees; `resolve` runs on every call before the tool.
  """

  def __init__(
      self,
      tool_name: str,
      signature: inspect.Signature,
      get_client: Callable[[], Any],
  ):
    """Initializes the selector.

    Args:
      tool_name: The CLI tool name, used to look up `ToolTraits`.
      signature: The tool's signature.
      get_client: Returns the XProf client that resolves local traces.
    """
    self._tool_name = tool_name
    self._signature = signature
    self._get_client = get_client
    self._traits = traits_for(tool_name)
    params = signature.parameters
    self._session_params = [
        name for name in ("source", "session_id") if name in params
    ]
    self._reads_trace = (
        bool(self._session_params)
        and "destination" not in params
        and "run_name" not in params
    )
    self._native_host = "host" in params
    self._added_host = self._reads_trace and not self._native_host
    self._name_host = self._native_host and self._traits.host_kind == "name"
    self._accepts_hosts = "hosts" in params
    self._has_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    # Fire passes named flags positionally for ordinary parameters (e.g.
    # `--host=x` arrives as args[1]), so positional args are mapped to names.
    self._bindable = not any(
        param.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.POSITIONAL_ONLY)
        for param in params.values()
    )
    self._positional_names = [
        param.name
        for param in params.values()
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]

  def extend_params(
      self, params: list[inspect.Parameter]
  ) -> list[inspect.Parameter]:
    """Returns `params` adjusted for `--host`.

    Adds a keyword-only `host` to trace tools that lack one. For tools with
    their own `host=""`, shows Fire a `None` default instead: Fire passes
    every declared default, so `--host=` would otherwise look like an omitted
    flag. `resolve` drops `None` so the tool's own default still applies.

    Args:
      params: The wrapper's parameters, without any `**kwargs` parameter.

    Returns:
      The adjusted parameters.
    """
    if self._added_host:
      return params + [
          inspect.Parameter(
              "host",
              inspect.Parameter.KEYWORD_ONLY,
              default=None,
              annotation=str | None,
          )
      ]
    if self._reads_trace and self._name_host:
      return [
          param.replace(default=None)
          if _has_empty_host_default(param)
          else param
          for param in params
      ]
    return params

  def resolve(
      self, args: tuple[Any, ...], kwargs: dict[str, Any]
  ) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any] | None]:
    """Applies `--host`, the single-trace check, and builds `capture`.

    Args:
      args: Positional arguments for the tool.
      kwargs: Keyword arguments for the tool.

    Returns:
      The (possibly rewritten) args and kwargs, and the capture description
      for local sessions, or None when the session is not local.

    Raises:
      ValueError: If `--host` / `--hosts` cannot be honoured, or a
        single-trace tool is given more than one trace.
      fire.core.FireError: If `--host` / `--hosts` is given without a value.
      TypeError: If an argument is given both positionally and by name.
    """
    if not self._reads_trace:
      return args, kwargs, None

    if self._bindable and args:
      for name, value in zip(self._positional_names, args):
        if name in kwargs:
          raise TypeError(
              f"{self._tool_name}() got multiple values for {name!r}"
          )
        kwargs[name] = value
      args = ()

    host = self._pop_host(kwargs)
    hosts = kwargs.get("hosts") if self._accepts_hosts else None
    if isinstance(hosts, bool) or (isinstance(hosts, str) and not hosts):
      raise fire.core.FireError("The --hosts flag requires a value.")
    if self._has_var_keyword and not self._accepts_hosts and "hosts" in kwargs:
      raise ValueError(
          f"{self._tool_name} does not support --hosts. Pass one .xplane.pb"
          " file path or --host=<name>."
      )

    position, session_key, session_value = self._find_session(args, kwargs)
    if session_value is None:
      if host is not None and self._added_host:
        raise ValueError("--host needs a session or trace path.")
      return args, kwargs, None

    def set_session(value: str) -> None:
      nonlocal args
      if 0 <= position < len(args):
        args_list = list(args)
        args_list[position] = value
        args = tuple(args_list)
      else:
        kwargs[session_key] = value

    client = self._get_client()
    is_local = getattr(client, "is_local_session", None)
    local = bool(is_local(session_value)) if callable(is_local) else False

    if host is not None:
      if local:
        session_value = str(client.resolve_host_path(session_value, host))
        set_session(session_value)
        if self._native_host:
          kwargs["host"] = trace_host_name(session_value)
      elif self._added_host:
        raise ValueError(
            f"{self._tool_name} supports --host only for local trace paths."
            " For a remote session, pass a host-specific session or omit"
            " --host."
        )

    if not local:
      return args, kwargs, None

    try:
      paths = list(client.select_paths(session_value, hosts=hosts))
    except FileNotFoundError:
      # Let the tool report missing sessions in its own words.
      return args, kwargs, None

    if self._traits.single_trace and len(paths) > 1:
      hint = (
          "Pass one .xplane.pb file path."
          if self._traits.host_kind == "index"
          else "Pass one .xplane.pb file path or --host=<name>."
      )
      raise ValueError(
          f"{self._tool_name} needs exactly one trace, but {len(paths)} were"
          f" found in {session_value}:\n{format_trace_list(paths)}\n{hint}"
      )

    try:
      capture = client.describe_capture(
          session_value, paths=paths, combine=self._traits.combine
      )
    except (OSError, ValueError):
      logging.exception("Failed to describe capture for %s", session_value)
      return args, kwargs, None
    return args, kwargs, capture if isinstance(capture, dict) else None

  def _pop_host(self, kwargs: dict[str, Any]) -> str | None:
    """Returns the `--host=<name>` value to route, or None.

    Removes the added `host` from `kwargs`, and an omitted native `host`
    (`None`) so the tool's own default applies.

    Args:
      kwargs: Keyword arguments for the tool; modified in place.

    Returns:
      The host name to route, or None when no host name was given or the
      tool's `host` is not a name.

    Raises:
      fire.core.FireError: If `--host` is given without a value.
    """
    if self._added_host:
      host = kwargs.pop("host", None)
    elif self._name_host:
      host = kwargs.get("host")
      if host is None:
        kwargs.pop("host", None)
    else:
      return None
    if isinstance(host, bool) or (isinstance(host, str) and not host):
      raise fire.core.FireError("The --host flag requires a value.")
    return None if host is None else str(host)

  def _find_session(
      self, args: tuple[Any, ...], kwargs: dict[str, Any]
  ) -> tuple[int, str, str | None]:
    """Finds the session argument, positional or keyword.

    Some tools declare both `source` and a `session_id` alias; whichever
    holds a non-empty string is used.

    Args:
      args: Positional arguments for the tool.
      kwargs: Keyword arguments for the tool.

    Returns:
      The positional index (-1 if passed by name), the parameter name, and
      the session value, or None if no session was given.
    """
    names = [
        param.name
        for param in self._signature.parameters.values()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    for candidate in self._session_params:
      index = names.index(candidate) if candidate in names else -1
      value = args[index] if 0 <= index < len(args) else kwargs.get(candidate)
      if isinstance(value, str) and value:
        return index, candidate, value
    return -1, self._session_params[0], None
