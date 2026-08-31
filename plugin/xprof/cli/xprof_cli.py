"""CLI for XProf tools."""

import functools
import inspect
import json
import pathlib
import re
import sys
from typing import Any

from absl import app
from absl import flags
import fire

from xprof import server
from xprof.cli.internal import xprof_data
from xprof.cli.internal.oss import hlo_tools
from xprof.cli.internal.oss import xplane_tools
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import check_host_boundness_tool
from xprof.cli.tools import get_graph_viewer_tool
from xprof.cli.tools import get_hlo_stats_tool
from xprof.cli.tools import get_kernel_stats_tool
from xprof.cli.tools import get_kernel_utilization_tool
from xprof.cli.tools import get_kpi_metrics_tool
from xprof.cli.tools import get_llo_analysis_tool
from xprof.cli.tools import get_llo_debug_string_tool
from xprof.cli.tools import get_memory_profile_tool
from xprof.cli.tools import get_overview_tool
from xprof.cli.tools import get_peak_allocations_tool
from xprof.cli.tools import get_roofline_model_tool
from xprof.cli.tools import get_top_hlo_ops_tool
from xprof.cli.tools import get_utilization_viewer_tool
from xprof.cli.tools import verify_numerical_parity_tool


def cli_main() -> dict[str, Any]:
  """Initializes the CLI and returns the available tools.

  Returns:
    A dictionary of tool names to functions.
  """
  return {
      # 27 Core Tools (Available in both 1P and 3P):
      # keep-sorted start
      "aggregate_xplane_events": xplane_tools.aggregate_xplane_events,
      "check_host_boundness": check_host_boundness_tool.check_host_boundness,
      "compute_utilization": get_kernel_utilization_tool.get_kernel_utilization,
      "get_avg_step_time": get_kernel_stats_tool.get_avg_step_time,
      "get_device_information": xprof_data.get_device_information,
      "get_graph_viewer": get_graph_viewer_tool.get_graph_viewer,
      "get_hlo_module_content": hlo_tools.get_hlo_module_content,
      "get_hlo_neighborhood": hlo_tools.get_hlo_neighborhood,
      "get_hlo_op_profile": xprof_data.get_hlo_op_profile,
      "get_hlo_stats": get_hlo_stats_tool.get_hlo_stats,
      "get_hlo_text": hlo_tools.get_hlo_text,
      "get_hosts": xprof_data.get_hosts,
      "get_kernel_stats": get_kernel_stats_tool.get_kernel_stats,
      "get_kernel_utilization": (
          get_kernel_utilization_tool.get_kernel_utilization
      ),
      "get_kpi_metrics": get_kpi_metrics_tool.get_kpi_metrics,
      "get_llo_analysis": get_llo_analysis_tool.get_llo_analysis,
      "get_llo_debug_string": get_llo_debug_string_tool.get_llo_debug_string,
      "get_memory_profile": get_memory_profile_tool.get_memory_profile,
      "get_overview": get_overview_tool.get_overview,
      "get_peak_allocations": get_peak_allocations_tool.get_peak_allocations,
      "get_profile_summary": xprof_data.get_profile_summary,
      "get_roofline_model": get_roofline_model_tool.get_roofline_model,
      "get_top_hlo_ops": get_top_hlo_ops_tool.get_top_hlo_ops,
      "get_utilization_viewer": (
          get_utilization_viewer_tool.get_utilization_viewer
      ),
      "get_xspace_proto": xplane_tools.get_xspace_proto,
      "list_hlo_modules": hlo_tools.list_hlo_modules,
      "list_xplane_events": xplane_tools.list_xplane_events,
      "verify_numerical_parity": (
          verify_numerical_parity_tool.verify_numerical_parity
      ),
      # keep-sorted end
  }


def _is_oss() -> bool:
  """Returns True if running in OSS."""
  return True


def _wrap_with_logdir(tool_func):
  """Wraps a tool to natively accept logdir and bypass_cache in Fire."""
  sig = inspect.signature(tool_func)

  params = []
  kwargs_param = None
  for p in sig.parameters.values():
    if p.kind == inspect.Parameter.VAR_KEYWORD:
      kwargs_param = p
    else:
      params.append(p)

  if "logdir" not in sig.parameters:
    params.append(
        inspect.Parameter(
            "logdir",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=str | None,
        )
    )
  if "bypass_cache" not in sig.parameters:
    params.append(
        inspect.Parameter(
            "bypass_cache",
            inspect.Parameter.KEYWORD_ONLY,
            default=False,
            annotation=bool,
        )
    )

  if kwargs_param is not None:
    params.append(kwargs_param)

  @functools.wraps(tool_func)
  def wrapper(
      *args,
      logdir: str | None = None,
      **kwargs,
  ):
    if isinstance(logdir, bool):
      raise fire.core.FireError("The --logdir flag requires a value.")

    # Reject empty session_id when --logdir is given or as positional arg
    if args and not args[0]:
      raise ValueError("session_id cannot be an empty string.")
    if "session_id" in kwargs and not kwargs["session_id"]:
      raise ValueError("session_id cannot be an empty string.")

    target_path = None
    if logdir is not None:
      target_path = str(logdir)
    elif args and "session_id" in sig.parameters:
      first_arg = args[0]
      if isinstance(first_arg, str) and (
          "/" in first_arg or "\\" in first_arg or "." in first_arg
      ):
        target_path = first_arg

    if target_path is not None:
      if (
          "destination" not in sig.parameters
          and "run_name" not in sig.parameters
      ):
        skip_local_check = target_path.startswith("gs://")
        if not skip_local_check:
          p = pathlib.Path(target_path).expanduser()
          if not p.exists():
            raise FileNotFoundError(
                f"Trace path '{target_path}' does not exist."
            )
          if p.is_dir():
            has_trace = any(p.glob("**/*.xplane.pb")) or any(
                p.glob("**/*.xspace.pb")
            )
            if not has_trace:
              raise FileNotFoundError(
                  "No .xplane.pb or .xspace.pb files found in directory"
                  f" '{target_path}' (DATA_ABSENT)."
              )
      if logdir is not None:
        xprof_client.get_client().set_logdir(str(logdir))
        if (
            "session_id" in sig.parameters
            and "session_id" not in kwargs
            and not args
        ):
          kwargs["session_id"] = str(logdir)
        elif (
            "source" in sig.parameters
            and "source" not in kwargs
            and not args
        ):
          kwargs["source"] = str(logdir)

    args_list = list(args)
    for i, p in enumerate(sig.parameters.values()):
      if i < len(args_list) and p.name in {
          "session_id",
          "source",
          "baseline_session_id",
          "optimized_session_id",
          "run_name",
          "module_name",
          "instruction_name",
          "func_name",
          "kernel_name",
          "host_name",
          "host",
      }:
        if isinstance(args_list[i], (int, float)):
          args_list[i] = str(args_list[i])
    args = tuple(args_list)

    for k in (
        "session_id",
        "source",
        "baseline_session_id",
        "optimized_session_id",
        "run_name",
        "module_name",
        "instruction_name",
        "func_name",
        "kernel_name",
        "host_name",
        "host",
    ):
      if k in kwargs and isinstance(kwargs[k], (int, float)):
        kwargs[k] = str(kwargs[k])

    if "bypass_cache" not in sig.parameters:
      if not (
          hasattr(tool_func, "__wrapped__")
          or getattr(tool_func, "_is_cached", False)
          or kwargs_param is not None
      ):
        kwargs.pop("bypass_cache", None)

    res = tool_func(*args, **kwargs)

    # Enforce volume spill guard (X-6) if output exceeds 10 MB.
    if isinstance(res, (str, bytes)):
      byte_len = (
          len(res) if isinstance(res, bytes) else len(res.encode("utf-8"))
      )
      if byte_len > 10 * 1024 * 1024:
        import tempfile  # pylint: disable=g-import-not-at-top
        import uuid  # pylint: disable=g-import-not-at-top

        tool_name_safe = getattr(tool_func, "__name__", "output")
        spill_file = (
            pathlib.Path(tempfile.gettempdir())
            / f"xprof_spill_{tool_name_safe}_{uuid.uuid4().hex[:8]}.json"
        )
        if isinstance(res, bytes):
          with open(spill_file, "wb") as f:
            f.write(res)
        else:
          with open(spill_file, "w", encoding="utf-8") as f:
            f.write(res)
        return json.dumps(
            {
                "status": "SAVED_TO_FILE",
                "size_bytes": byte_len,
                "size_mib": round(byte_len / (1024 * 1024), 2),
                "file_path": str(spill_file),
                "message": (
                    f"Output payload ({round(byte_len / (1024 * 1024), 2)} MB)"
                    " exceeded 10 MB threshold. Saved to file to prevent"
                    " terminal buffer overflow."
                ),
            },
            indent=2,
        )

    return res

  wrapper.__signature__ = sig.replace(parameters=params)  # pyrefly: ignore[missing-attribute]
  return wrapper


wrap_with_logdir = _wrap_with_logdir


class XProfCli:
  """XProf CLI to be invoked by fire.Fire."""

  if _is_oss():

    @classmethod
    def server(
        cls,
        logdir: str | None = None,
        port: int = 8791,
        hide_capture_profile_button: bool = False,
        enable_tab_name_label: bool = False,
        worker_service_address: str | None = None,
        grpc_port: int = 50051,
        src_prefix: str | None = None,
        max_concurrent_worker_requests: int = 1,
    ):
      """Starts the XProf web server.

      Args:
        logdir: Path to the TensorBoard log directory root.
        port: Port to run the main server on.
        hide_capture_profile_button: Whether to hide the capture profile button.
        enable_tab_name_label: Whether to enable tab name label.
        worker_service_address: Address for the worker service.
        grpc_port: Port for the gRPC server.
        src_prefix: Prefix for source paths.
        max_concurrent_worker_requests: Maximum concurrent worker requests.
      """
      del cls
      if isinstance(logdir, bool):
        raise fire.core.FireError("The --logdir flag requires a value.")
      xprof_client.get_client().set_logdir(logdir)
      try:
        server.start_server(
            default_logdir=logdir,
            port=port,
            hide_capture_profile_button=hide_capture_profile_button,
            enable_tab_name_label=enable_tab_name_label,
            worker_service_address=worker_service_address,
            grpc_port=grpc_port,
            src_prefix=src_prefix,
            max_concurrent_worker_requests=max_concurrent_worker_requests,
        )
      except ValueError as e:
        raise fire.core.FireError(str(e))

  else:

    @classmethod
    def server(
        cls,
        logdir: str | None = None,
        port: int = 8791,
        hide_capture_profile_button: bool = False,
        enable_tab_name_label: bool = False,
        worker_service_address: str | None = None,
        grpc_port: int = 50051,
        src_prefix: str | None = None,
        max_concurrent_worker_requests: int = 1,
    ):
      """Starts the XProf web server.

      Args:
        logdir: Path to the TensorBoard log directory root.
        port: Port to run the main server on.
        hide_capture_profile_button: Whether to hide the capture profile button.
        enable_tab_name_label: Whether to enable tab name label.
        worker_service_address: Address for the worker service.
        grpc_port: Port for the gRPC server.
        src_prefix: Prefix for source paths.
        max_concurrent_worker_requests: Maximum concurrent worker requests.
      """
      del cls
      try:
        server.start_server(
            logdir=logdir,
            port=port,
            hide_capture_profile_button=hide_capture_profile_button,
            enable_tab_name_label=enable_tab_name_label,
            worker_service_address=worker_service_address,
            grpc_port=grpc_port,
            src_prefix=src_prefix,
            max_concurrent_worker_requests=max_concurrent_worker_requests,
        )
      except ValueError as e:
        raise fire.core.FireError(str(e))

  def __call__(self, *args, **kwargs):
    # This triggers on: `xprof`
    # (or `xprof --logdir .` without a command name)
    return self.server(*args, **kwargs)


for _name, _tool in cli_main().items():
  setattr(XProfCli, _name, staticmethod(_wrap_with_logdir(_tool)))


def _check_xprof_version() -> None:
  """Checks that the installed xprof package meets minimum version 2.23.0."""
  try:
    import importlib.metadata  # pylint: disable=g-import-not-at-top

    version_str = importlib.metadata.version("xprof")
    parts = [int(p) for p in version_str.split(".") if p.isdigit()]
    if len(parts) >= 2 and (parts[0], parts[1]) < (2, 23):
      sys.stderr.write(
          f"WARNING: xprof version {version_str} is installed, but xprof >="
          " 2.23.0 is recommended for complete tool support.\n"
      )
  except Exception:  # pylint: disable=broad-exception-caught
    pass


def _emit_error(reason: str, message: str, exit_code: int) -> None:
  """Emits structured JSON on stdout and human-readable header on stderr."""
  payload = {
      "status": "ERROR",
      "reason": reason,
      "error": message,
  }
  sys.stdout.write(json.dumps(payload, indent=2) + "\n")
  sys.stderr.write(f"{reason}: {message}\n")
  sys.exit(exit_code)


_UNDERSCORE_NUM_PATTERN = re.compile(r"^\d+(_\d+)+$")


def _preprocess_argv(argv: list[str] | None) -> list[str] | None:
  """Preprocesses CLI arguments so timestamp session IDs with underscores are preserved as strings in Fire."""
  if not argv:
    return argv
  processed = []
  for arg in argv:
    if arg.startswith("--") and "=" in arg:
      key, val = arg.split("=", 1)
      if _UNDERSCORE_NUM_PATTERN.match(val) and not (
          val.startswith(('"', "'")) and val.endswith(('"', "'"))
      ):
        processed.append(f'{key}="{val}"')
      else:
        processed.append(arg)
    elif arg.startswith("-"):
      processed.append(arg)
    else:
      if _UNDERSCORE_NUM_PATTERN.match(arg) and not (
          arg.startswith(('"', "'")) and arg.endswith(('"', "'"))
      ):
        processed.append(f'"{arg}"')
      else:
        processed.append(arg)
  return processed


def main(argv=None) -> None:
  """Main function for the xprof CLI."""
  import logging  # pylint: disable=g-import-not-at-top

  _check_xprof_version()

  processed_command = _preprocess_argv(argv[1:] if argv else None)
  try:
    fire.Fire(XProfCli(), command=processed_command, name="xprof")
  except (fire.core.FireError, TypeError) as e:
    _emit_error("USAGE_ERROR", str(e), 2)
  except FileNotFoundError as e:
    _emit_error("PATH_ERROR", str(e), 3)
  except ValueError as e:
    _emit_error("INVALID_VALUE", str(e), 4)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Unhandled defect in xprof_cli")
    _emit_error("INTERNAL_ERROR", f"{e}\nPlease report to b/547935083", 1)


if __name__ == "__main__":
  import sys
  main(sys.argv)
