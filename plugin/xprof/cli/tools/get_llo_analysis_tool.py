"""Tool to analyze LLO from XProf xspace."""

import json
import logging
import os
import tempfile
import traceback

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.convert import _pywrap_profiler_plugin


@decorators.cached(expire=86400)
def get_llo_analysis(session_id: str, host: str = "", kernel: str = "") -> str:
  """Fetches xspace and runs LLO analysis.

  Args:
      session_id: The unique XProf session ID or xplane file path.
      host: The host to get the xspace for.
      kernel: Optional kernel name filter (e.g. 'mm.1' or 'custom-call').

  Returns:
      A JSON-formatted string containing LLO analysis details.
  """
  session_id = str(session_id)
  client = xprof_client.get_client()
  try:
    if not _pywrap_profiler_plugin.built_with_embedded():
      return json.dumps(
          dict(
              status="UNAVAILABLE",
              reason="LLO_ANALYSIS_UNSUPPORTED_IN_OSS",
              message=(
                  "LLO analysis is not supported in this standard OSS build"
                  " (requires a TPU profiler binary with embedded LLO analysis"
                  " support)."
              ),
          ),
          indent=2,
      )

    session_str = str(session_id)
    if (
        os.path.exists(session_str)
        or session_str.startswith(("/", ".", "\\"))
        or session_str.endswith((".xplane.pb", ".xspace.pb"))
    ):
      target_file = session_str
      if os.path.isdir(target_file):
        for root, _, files in os.walk(target_file):
          for f in files:
            if f.endswith((".xplane.pb", ".xspace.pb")):
              target_file = os.path.join(root, f)
              break
          if target_file != session_str:
            break
      analysis = _pywrap_profiler_plugin.analyze_llo(target_file, kernel=kernel)
      if not analysis.get("success", False):
        return json.dumps(
            dict(
                status="UNAVAILABLE",
                reason="LLO_DATA_ABSENT",
                error=(
                    "Failed to analyze LLO from xspace (LLO trace data is not"
                    " available in this session)."
                ),
                remediation=(
                    "To enable LLO tracing, ensure the workload is executed"
                    " with"
                    ' LIBTPU_INIT_ARGS="--xla_xprof_enable_custom_call_tracing=true'
                    ' --xla_xprof_register_llo_debug_info=true" exported'
                    " strictly BEFORE 'import jax'. Prerequisites: Python 3.11+"
                    " (Python 3.12 recommended via uv), JAX >= 0.11.0 (default"
                    " Cloud TPU VM images running Python 3.10 cap JAX at 0.6.2"
                    " and lack LLO flag support), and xprof-nightly."
                ),
            ),
            indent=2,
        )
      return json.dumps(analysis, indent=2)

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

      analysis = _pywrap_profiler_plugin.analyze_llo(
          temp_file.name, kernel=kernel
      )

      if not analysis.get("success", False):
        return json.dumps(
            dict(
                status="UNAVAILABLE",
                reason="LLO_DATA_ABSENT",
                error=(
                    "Failed to analyze LLO from xspace (LLO trace data is not"
                    " available in this session)."
                ),
                remediation=(
                    "To enable LLO tracing, ensure the workload is executed"
                    " with"
                    ' LIBTPU_INIT_ARGS="--xla_xprof_enable_custom_call_tracing=true'
                    ' --xla_xprof_register_llo_debug_info=true" exported'
                    " strictly BEFORE 'import jax'. Prerequisites: Python 3.11+"
                    " (Python 3.12 recommended via uv), JAX >= 0.11.0 (default"
                    " Cloud TPU VM images running Python 3.10 cap JAX at 0.6.2"
                    " and lack LLO flag support), and xprof-nightly."
                ),
            ),
            indent=2,
        )

      return json.dumps(analysis, indent=2)

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
