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
from xprof.cli.tools import detect_layout_mismatch_copies_tool
from xprof.cli.tools import detect_unfused_reshapes_tool
from xprof.cli.tools import detect_unnecessary_convert_reduce_tool
from xprof.cli.tools import get_graph_viewer_tool
from xprof.cli.tools import get_kpi_metrics_tool
from xprof.cli.tools import get_memory_profile_tool
from xprof.cli.tools import get_overview_tool
from xprof.cli.tools import get_peak_allocations_tool
from xprof.cli.tools import get_smart_suggestions_tool
from xprof.cli.tools import get_top_hlo_ops_tool
from xprof.cli.tools import get_utilization_viewer_tool
from xprof.cli.tools.oss import upload_trace_tool


def cli_main() -> dict[str, Any]:
  """Initializes the CLI and returns the available tools.

  Returns:
    A dictionary of tool names to functions.
  """
  return {
      # Standard tools to be exposed as skills through CLI.
      # keep-sorted start
      "detect_layout_mismatch_copies": (
          detect_layout_mismatch_copies_tool.detect_layout_mismatch_copies
      ),
      "detect_unnecessary_convert_reduce": (
          detect_unnecessary_convert_reduce_tool.detect_unnecessary_convert_reduce
      ),
      "detect_unfused_reshapes": (
          detect_unfused_reshapes_tool.detect_unfused_reshapes
      ),
      "get_graph_viewer": get_graph_viewer_tool.get_graph_viewer,
      "get_hlo_neighborhood": hlo_tools.get_hlo_neighborhood,
      "get_hlo_text": hlo_tools.get_hlo_text,
      "get_kpi_metrics": get_kpi_metrics_tool.get_kpi_metrics,
      "get_memory_profile": get_memory_profile_tool.get_memory_profile,
      "get_overview": get_overview_tool.get_overview,
      "get_peak_allocations": get_peak_allocations_tool.get_peak_allocations,
      "get_smart_suggestions": get_smart_suggestions_tool.get_smart_suggestions,
      "get_top_hlo_ops": get_top_hlo_ops_tool.get_top_hlo_ops,
      "get_utilization_viewer": (
          get_utilization_viewer_tool.get_utilization_viewer
      ),
      "upload_trace": upload_trace_tool.upload_trace,
      # keep-sorted end
      # Tools to be migrated or consolidated to the above standard list.
      "get_hlo_module_content": hlo_tools.get_hlo_module_content,
      "list_hlo_modules": hlo_tools.list_hlo_modules,
      "get_hlo_op_profile": xprof_data.get_hlo_op_profile,
      "list_xplane_events": xplane_tools.list_xplane_events,
      "aggregate_xplane_events": xplane_tools.aggregate_xplane_events,
      "get_xspace_proto": xplane_tools.get_xspace_proto,
      "get_profile_summary": xprof_data.get_profile_summary,
      "get_hosts": xprof_data.get_hosts,
      "get_device_information": xprof_data.get_device_information,
  }


def _is_oss() -> bool:
  """Returns True if running in OSS."""
  return True


def _wrap_with_logdir(tool_func):
  """Wraps a tool to natively accept logdir as its first argument in Fire."""
  if not _is_oss():
    return tool_func

  sig = inspect.signature(tool_func)
  if "logdir" in sig.parameters:
    return tool_func

  new_params = [
      inspect.Parameter(
          "logdir", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str
      )
  ]
  new_params.extend(sig.parameters.values())

  @functools.wraps(tool_func)
  def wrapper(logdir: str, *args, **kwargs):
    if isinstance(logdir, bool):
      raise fire.core.FireError("The --logdir flag requires a value.")
    xprof_client.get_client().set_logdir(logdir)
    return tool_func(*args, **kwargs)

  wrapper.__signature__ = sig.replace(parameters=new_params)
  return wrapper


class XProfCli:
  """XProf CLI to be invoked by fire.Fire."""

  if _is_oss():

    @staticmethod
    def server(
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

    @staticmethod
    def server(
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
  fire.Fire(XProfCli(), command=argv[1:] if argv else None, name="xprof")


if __name__ == "__main__":
  app.run(main, flags_parser=lambda argv: flags.FLAGS(argv, known_only=True))
