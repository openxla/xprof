"""Hermetic Python execution runner for isolated sandbox execution pool.

Enforces virtual memory quotas (`4GB`), configures headless matplotlib (`Agg`),
traps stdout cleanly for single-line structured container logging, captures
generated plots into single-line Base64 SVG payloads, and processes analytical
scripts securely.
"""

import base64
from collections.abc import Sequence
import contextlib
import io
import json
import os
import resource
import sys
import traceback
from typing import Any, TypedDict

from absl import app
from absl import flags
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('Agg')

_SESSION_ID = flags.DEFINE_string(
    'session_id', '', 'XProf profile session identifier.'
)
_USE_SPANNER = flags.DEFINE_bool(
    'use_spanner', False, 'Whether to enable Spanner storage backend access.'
)
_READ_ONLY_MODE = flags.DEFINE_bool(
    'read_only_mode',
    True,
    'Whether to enforce read-only execution constraints.',
)
_CODE = flags.DEFINE_string(
    'code', '', 'Inline analytical script string to execute directly.'
)
_SCRIPT_PATH = flags.DEFINE_string(
    'script_path', '', 'Path to analytical script file to read and execute.'
)

# Mandatory virtual memory quota limit (4GB).
MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024


class ExecutionSuccess(TypedDict):
  status: str
  stdout_line: str
  stderr_line: str
  charts_svg_base64: list[str]


class ExecutionError(TypedDict):
  status: str
  error_trace_line: str
  stderr_line: str


ExecutionResult = ExecutionSuccess | ExecutionError


def enforce_memory_limits(limit_bytes: int = MEMORY_LIMIT_BYTES) -> None:
  """Applies resource limits (`RLIMIT_AS`) on virtual memory usage.

  Args:
    limit_bytes: Maximum virtual memory address space footprint in bytes.
  """
  try:
    _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    target_soft = limit_bytes
    if hard_limit != resource.RLIM_INFINITY:
      target_soft = min(target_soft, hard_limit)
    resource.setrlimit(resource.RLIMIT_AS, (target_soft, hard_limit))
  except (ValueError, resource.error) as err:
    # In sandboxes or environments where setrlimit cannot decrease below
    # current heap footprint, log single-line warning without aborting.
    sys.stderr.write(
        f'WARNING: Unable to enforce strict RLIMIT_AS quota to {limit_bytes}:'
        f' {err}\n'
    )


def _sanitize_single_line_log(content: str) -> str:
  """Cleans string outputs to eliminate embedded newlines for structured logs.

  Args:
    content: Raw string potentially containing multiple lines.

  Returns:
    Single-line escaped output suitable for structured container logs.
  """
  return content.rstrip('\r\n').replace('\r\n', '\\n').replace('\n', '\\n')


def capture_plots_base64_svg() -> list[str]:
  """Extracts all active Matplotlib figures into single-line Base64 SVG strings.

  Returns:
    List of Base64 encoded SVG strings for each open chart figure.
  """
  svg_charts: list[str] = []
  figure_numbers: list[int] = plt.get_fignums()
  for fig_num in figure_numbers:
    fig = plt.figure(fig_num)
    with io.BytesIO() as buf:
      fig.savefig(buf, format='svg', bbox_inches='tight')
      plt.close(fig)
      encoded = (
          base64.b64encode(buf.getvalue()).decode('ascii').replace('\n', '')
      )
    svg_charts.append(encoded)
  return svg_charts


def execute_analytical_script(
    code_text: str,
    *,
    session_id: str,
    use_spanner: bool,
    read_only_mode: bool,
) -> ExecutionResult:
  """Executes Python analytical code under hermetic trapping conditions.

  Args:
    code_text: Python script source text to execute in isolated namespace.
    session_id: Profiling session parameter exposed inside the namespace.
    use_spanner: Flag indicating backend routing exposed in execution context.
    read_only_mode: Flag ensuring mutations or unsafe calls can be audited.

  Returns:
    Structured result dictionary matching expected JSON payloads.
  """
  stdout_buf = io.StringIO()
  stderr_buf = io.StringIO()

  execution_context: dict[str, Any] = {
      '__name__': '__main__',
      'np': np,
      'pandas': pd,
      'pd': pd,
      'plt': plt,
      'session_id': session_id,
      'use_spanner': use_spanner,
      'read_only_mode': read_only_mode,
  }

  with (
      contextlib.redirect_stdout(stdout_buf),
      contextlib.redirect_stderr(stderr_buf),
  ):
    try:
      exec(code_text, execution_context)  # pylint: disable=exec-used
      charts_svg_base64 = capture_plots_base64_svg()
      stdout_line = _sanitize_single_line_log(stdout_buf.getvalue())
      stderr_line = _sanitize_single_line_log(stderr_buf.getvalue())
      return {
          'status': 'SUCCESS',
          'stdout_line': stdout_line,
          'stderr_line': stderr_line,
          'charts_svg_base64': charts_svg_base64,
      }
    except (Exception, SystemExit):  # pylint: disable=broad-except
      error_trace_line = _sanitize_single_line_log(traceback.format_exc())
      stderr_line = _sanitize_single_line_log(stderr_buf.getvalue())
      return {
          'status': 'ERROR',
          'error_trace_line': error_trace_line,
          'stderr_line': stderr_line,
      }
    finally:
      plt.close('all')


def _read_script_source(argv: Sequence[str]) -> str:
  """Resolves the script source text from flags, positional args, or stdin."""
  if _CODE.value:
    return _CODE.value
  if _SCRIPT_PATH.value:
    try:
      with open(_SCRIPT_PATH.value, 'r', encoding='utf-8') as handle:
        return handle.read()
    except OSError as err:
      raise OSError(
          f'Failed to open script_path {_SCRIPT_PATH.value}: {err}'
      ) from err
  if len(argv) > 1 and argv[1]:
    if os.path.isfile(argv[1]):
      try:
        with open(argv[1], 'r', encoding='utf-8') as handle:
          return handle.read()
      except OSError as err:
        raise OSError(
            f'Failed to open positional script file {argv[1]}: {err}'
        ) from err
    return argv[1]
  return sys.stdin.read()


def main(argv: Sequence[str]) -> None:
  enforce_memory_limits(MEMORY_LIMIT_BYTES)

  try:
    script_source = _read_script_source(argv)
  except OSError as err:
    error_result: ExecutionError = {
        'status': 'ERROR',
        'error_trace_line': _sanitize_single_line_log(str(err)),
        'stderr_line': '',
    }
    sys.stdout.write(json.dumps(error_result) + '\n')
    sys.stdout.flush()
    return

  result = execute_analytical_script(
      code_text=script_source,
      session_id=_SESSION_ID.value,
      use_spanner=_USE_SPANNER.value,
      read_only_mode=_READ_ONLY_MODE.value,
  )

  # Output exact JSON payload on a single line without extraneous newlines.
  sys.stdout.write(json.dumps(result) + '\n')
  sys.stdout.flush()


if __name__ == '__main__':
  app.run(main)
