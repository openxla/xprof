"""Tool to analyze LLO from XProf xspace and extract debug string."""

import json
import logging
import tempfile
import traceback

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.convert import _pywrap_profiler_plugin


@decorators.cached(expire=86400)
def get_llo_debug_string(session_id: str, host: str = "") -> str:
  """Fetches xspace and runs LLO analysis to return the debug string.

  Args:
      session_id: The unique XProf session ID.
      host: The host to get the xspace for.

  Returns:
      A JSON-formatted string containing LLO debug string.
  """
  session_id = str(session_id)
  client = xprof_client.get_client()
  try:
    if not _pywrap_profiler_plugin.built_with_embedded():
      return json.dumps(
          dict(
              error=(
                  "LLO debug string is not supported in this build. Ensure"
                  " xprof is installed via pypi."
              )
          ),
          indent=2,
      )

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

    with tempfile.NamedTemporaryFile() as temp_file:
      temp_file.write(serialized_xspace)
      temp_file.flush()

      debug_str = _pywrap_profiler_plugin.get_llo_debug_string(temp_file.name)

      if not debug_str:
        return json.dumps(
            dict(error="Failed to extract LLO debug string from xspace"),
            indent=2,
        )

      return json.dumps({"debug_string": debug_str}, indent=2)

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error fetching/analyzing LLO data for session %s", session_id
    )
    return json.dumps(
        dict(
            error=f"Error analyzing LLO data: {e}",
            traceback=traceback.format_exc(),
        ),
        indent=2,
    )
