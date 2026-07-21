"""Tool to generate structured LLO runtime report from XProf session."""

import json
import logging
import tempfile
import traceback

from tensorflow.tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import llo_parser
from xprof.cli.internal.oss import llo_report_generator
from xprof.cli.internal.oss import xprof_client


@decorators.cached(expire=86400)
def get_llo_report(
    session_id: str,
    host: str = "",
    format: str = "json",  # pylint: disable=redefined-builtin
    kernel_filter: str = "",
) -> str:
  """Fetches serialized xspace, loads LLO events, and runs report generator.

  Args:
      session_id: The unique XProf session ID.
      host: The host to get the xspace for.
      format: Output format ('json', 'markdown', or 'html').
      kernel_filter: Optional substring filter across scopes and events.

  Returns:
      JSON or Markdown formatted string containing the LLO report.
  """
  session_id = str(session_id)
  client = xprof_client.get_client()
  try:
    hosts = client.get_hosts(session_id, with_metadata=False)
    available_hosts = hosts if hosts else []

    if not host:
      if available_hosts:
        host = available_hosts[0]
      else:
        host = ""
    elif host not in available_hosts:
      return json.dumps(
          dict(
              error=f"Invalid host: '{host}'.",
              available_hosts=available_hosts,
          ),
          indent=2,
      )

    serialized_xspace = client.get_serialized_xspace(session_id, host)
    if not serialized_xspace:
      return json.dumps(
          dict(
              error=(
                  f"No serialized xspace found for session {session_id}, host"
                  f" {host}"
              )
          ),
          indent=2,
      )

    xspace = xplane_pb2.XSpace()
    xspace.ParseFromString(serialized_xspace)

    with tempfile.NamedTemporaryFile(suffix=".db") as temp_db:
      llo_parser.parse_and_load_llo_events(xspace, temp_db.name)
      report_dict = llo_report_generator.generate_structured_report(
          temp_db.name,
          kernel_filter=kernel_filter if kernel_filter else None,
          session_id=session_id,
      )

    if format.lower() in ("markdown", "md", "html"):
      return llo_report_generator.generate_markdown_html_report(report_dict)
    return json.dumps(report_dict, indent=2)

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error fetching/analyzing LLO report for session %s", session_id
    )
    return json.dumps(
        dict(
            error=f"Error generating LLO report: {e}",
            traceback=traceback.format_exc(),
        ),
        indent=2,
    )
