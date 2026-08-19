"""CLI for XProf tools."""

import functools
import inspect
from typing import Any

from absl import app
from absl import flags
import fire

from xprof import server
from xprof.cli.internal import xprof_data

from xprof.cli.internal.oss import hlo_tools
from xprof.cli.internal.oss import xplane_tools
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_graph_viewer_tool
from xprof.cli.tools import get_kernel_stats_tool
from xprof.cli.tools import get_kpi_metrics_tool
from xprof.cli.tools import get_llo_analysis_tool
from xprof.cli.tools import get_llo_debug_string_tool
from xprof.cli.tools import get_memory_profile_tool
from xprof.cli.tools import get_overview_tool
from xprof.cli.tools import get_peak_allocations_tool
from xprof.cli.tools import get_top_hlo_ops_tool
from xprof.cli.tools import get_utilization_viewer_tool
from xprof.cli.tools import verify_numerical_parity_tool
from xprof.cli.tools.oss import upload_trace_tool


def cli_main() -> dict[str, Any]:
  """Initializes the CLI and returns the available tools.

  Returns:
    A dictionary of tool names to functions.
  """
  return {
      # 23 Core Tools (Available in both 1P and 3P):
      # keep-sorted start
      "aggregate_xplane_events": xplane_tools.aggregate_xplane_events,
      "get_device_information": xprof_data.get_device_information,
      "get_graph_viewer": get_graph_viewer_tool.get_graph_viewer,
      "get_hlo_module_content": hlo_tools.get_hlo_module_content,
      "get_hlo_neighborhood": hlo_tools.get_hlo_neighborhood,
      "get_hlo_op_profile": xprof_data.get_hlo_op_profile,
      "get_hlo_text": hlo_tools.get_hlo_text,
      "get_hosts": xprof_data.get_hosts,
      "get_kernel_stats": get_kernel_stats_tool.get_kernel_stats,
      "get_kpi_metrics": get_kpi_metrics_tool.get_kpi_metrics,
      "get_llo_analysis": get_llo_analysis_tool.get_llo_analysis,
      "get_llo_debug_string": get_llo_debug_string_tool.get_llo_debug_string,
      "get_memory_profile": get_memory_profile_tool.get_memory_profile,
      "get_overview": get_overview_tool.get_overview,
      "get_peak_allocations": get_peak_allocations_tool.get_peak_allocations,
      "get_profile_summary": xprof_data.get_profile_summary,
      "get_top_hlo_ops": get_top_hlo_ops_tool.get_top_hlo_ops,
      "get_utilization_viewer": (
          get_utilization_viewer_tool.get_utilization_viewer
      ),
      "get_xspace_proto": xplane_tools.get_xspace_proto,
      "list_hlo_modules": hlo_tools.list_hlo_modules,
      "list_xplane_events": xplane_tools.list_xplane_events,
      "upload_trace": upload_trace_tool.upload_trace,
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
    if logdir is not None:
      if _is_oss():
        if (
            "destination" not in sig.parameters
            and "run_name" not in sig.parameters
        ):
          import pathlib  # pylint: disable=g-import-not-at-top

          p = pathlib.Path(str(logdir))
          if not p.exists() and not str(logdir).startswith("gs://"):
            raise FileNotFoundError(f"Trace path '{logdir}' does not exist.")
        xprof_client.get_client().set_logdir(str(logdir))
      if (
          "session_id" in sig.parameters
          and "session_id" not in kwargs
          and not args
      ):
        kwargs["session_id"] = str(logdir)
    elif args and "session_id" in sig.parameters and _is_oss():
      first_arg = args[0]
      if isinstance(first_arg, str) and (
          "/" in first_arg or "\\" in first_arg or "." in first_arg
      ):
        if (
            "destination" not in sig.parameters
            and "run_name" not in sig.parameters
        ):
          import pathlib  # pylint: disable=g-import-not-at-top

          p = pathlib.Path(first_arg)
          if not p.exists() and not first_arg.startswith("gs://"):
            raise FileNotFoundError(f"Trace path '{first_arg}' does not exist.")

    if "bypass_cache" not in sig.parameters:
      if not (
          hasattr(tool_func, "__wrapped__")
          or getattr(tool_func, "_is_cached", False)
          or kwargs_param is not None
      ):
        kwargs.pop("bypass_cache", None)

    return tool_func(*args, **kwargs)

  wrapper.__signature__ = sig.replace(parameters=params)  # pyrefly: ignore[missing-attribute]
  return wrapper


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


def main(argv=None) -> None:
  """Main function for the xprof CLI."""
  import sys  # pylint: disable=g-import-not-at-top
  import logging  # pylint: disable=g-import-not-at-top

  try:
    fire.Fire(XProfCli(), command=argv[1:] if argv else None, name="xprof")
  except fire.core.FireError as e:
    sys.stderr.write(f"USAGE ERROR: {e}\n")
    sys.exit(2)
  except FileNotFoundError as e:
    sys.stderr.write(f"PATH ERROR: {e}\n")
    sys.exit(2)
  except ValueError as e:
    sys.stderr.write(f"INVALID ARGUMENT: {e}\n")
    sys.exit(2)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Unhandled defect in xprof_cli")
    sys.stderr.write(f"INTERNAL ERROR: {e}\nPlease report to b/547935083\n")
    sys.exit(1)


if __name__ == "__main__":
  import sys
  main(sys.argv)
