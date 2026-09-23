"""OSS xprof_cli tool exposing LLO static-schedule analyses.

The 1P counterpart (`cli/tools/google/llo_static_analysis_tool.py`) renders the
analyses in Python from the `LloModuleProto` embedded in an XSpace. OSS builds
delegate instead to the C API in the profiler plugin pywrap binding, which is
only compiled when embedded features are enabled; without them the binding
raises `NotImplementedError`, which is surfaced here as a structured error.
"""

import glob
import json
import os

from xprof.convert import _pywrap_profiler_plugin


def _error(mode: str, message: str) -> str:
  """Returns an error payload matching the 1P JSON output contract."""
  return json.dumps(
      {
          "status": "ERROR",
          "semantics": "static_modelled_schedule",
          "mode": mode,
          "error": message,
      },
      indent=2,
  )


def get_llo_static_analysis(
    session_id: str,
    mode: str = "region_tree",
    hlo_op: str = "",
    bundle: int = 0,
    host: str = "",
) -> str:
  """Renders an LLO static analysis from a local XSpace file or directory.

  Args:
    session_id: A local `.xplane.pb` file path, or a directory holding one.
    mode: One of list_ops, region_tree, bundle_util, opcodes, register_pressure,
      spills, bdi_stalls, target_info, source_for_bundle.
    hlo_op: Substring of the HLO instruction name to select an LLO module.
    bundle: Bundle index (only for source_for_bundle).
    host: Unsupported in OSS; a non-empty value is reported as an error. The
      parameter is kept so the CLI surface matches the 1P tool.

  Returns:
    A JSON string complying with the JSON output contract.
  """
  if host:
    return _error(mode, "--host filtering is not supported in OSS builds.")
  target_path = str(session_id)
  if os.path.isdir(target_path):
    xplane_files = sorted(
        glob.glob(
            os.path.join(target_path, "**", "*.xplane.pb"), recursive=True
        )
    )
    if not xplane_files:
      return _error(mode, f"No *.xplane.pb files found in {target_path}.")
    target_path = xplane_files[0]
  try:
    return _pywrap_profiler_plugin.get_llo_static_analysis_json(
        target_path, mode=mode, hlo_op=hlo_op, bundle=bundle
    )
  except NotImplementedError as e:
    return _error(mode, str(e))
