"""Tests for the hermetic Python runner (`xprof_py_runner.py`)."""

import base64
import io
import json
import os
import resource
import sys
import tempfile
from typing import Any

from absl.testing import absltest
from absl.testing import flagsaver
import matplotlib.pyplot as plt

from google3.third_party.xprof.labs.jetski import xprof_py_runner


class XprofPyRunnerTest(absltest.TestCase):

  def test_enforce_memory_limits_executes(self) -> None:
    original_soft, original_hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
      # Verifies enforce_memory_limits runs cleanly inside test environments.
      xprof_py_runner.enforce_memory_limits(xprof_py_runner.MEMORY_LIMIT_BYTES)
      soft_limit, _ = resource.getrlimit(resource.RLIMIT_AS)
      self.assertIsInstance(soft_limit, int)
    finally:
      try:
        resource.setrlimit(resource.RLIMIT_AS, (original_soft, original_hard))
      except (ValueError, resource.error):
        pass

  def test_execute_analytical_script_success_and_plots(self) -> None:
    script = (
        'print("Initializing metrics analysis...")\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'import matplotlib.pyplot as plt\n'
        'x = np.array([1, 2, 3, 4])\n'
        'y = x * 2\n'
        'plt.plot(x, y)\n'
        'print("Plot generated successfully.")\n'
    )
    result = xprof_py_runner.execute_analytical_script(
        code_text=script,
        session_id='sess-888',
        use_spanner=True,
        read_only_mode=True,
    )
    self.assertEqual(result['status'], 'SUCCESS')
    self.assertIn('Initializing metrics analysis...', result['stdout_line'])
    self.assertIn('Plot generated successfully.', result['stdout_line'])
    self.assertNotIn('\n', result['stdout_line'])

    charts = result['charts_svg_base64']
    self.assertLen(charts, 1)
    # Ensure single-line formatting inside charts.
    self.assertNotIn('\n', charts[0])
    decoded_svg = base64.b64decode(charts[0]).decode('utf-8')
    self.assertIn('<svg', decoded_svg)

  def test_execute_analytical_script_zero_plots(self) -> None:
    script = 'print("Calculation without charts")'
    result = xprof_py_runner.execute_analytical_script(
        code_text=script,
        session_id='sess-no-plots',
        use_spanner=False,
        read_only_mode=True,
    )
    self.assertEqual(result['status'], 'SUCCESS')
    self.assertEqual(result['charts_svg_base64'], [])

  def test_execute_analytical_script_multiple_plots(self) -> None:
    script = (
        'plt.figure(1)\n'
        'plt.plot([1, 2], [3, 4])\n'
        'plt.figure(2)\n'
        'plt.plot([5, 6], [7, 8])\n'
    )
    result = xprof_py_runner.execute_analytical_script(
        code_text=script,
        session_id='sess-multi-plots',
        use_spanner=False,
        read_only_mode=True,
    )
    self.assertEqual(result['status'], 'SUCCESS')
    self.assertLen(result['charts_svg_base64'], 2)

  def test_execute_analytical_script_gke_single_line_formatting(self) -> None:
    script = 'print("Line A\\nLine B\\nLine C")'
    result = xprof_py_runner.execute_analytical_script(
        code_text=script,
        session_id='sess-999',
        use_spanner=False,
        read_only_mode=True,
    )
    self.assertEqual(result['status'], 'SUCCESS')
    self.assertEqual(result['stdout_line'], 'Line A\\nLine B\\nLine C')
    self.assertNotIn('\n', result['stdout_line'])

  def test_execute_analytical_script_captures_error_trace(self) -> None:
    script = (
        'def failing_operation():\n'
        '  raise RuntimeError("Simulated calculation error")\n'
        'failing_operation()\n'
    )
    result = xprof_py_runner.execute_analytical_script(
        code_text=script,
        session_id='sess-1010',
        use_spanner=False,
        read_only_mode=True,
    )
    self.assertEqual(result['status'], 'ERROR')
    self.assertIn(
        'RuntimeError: Simulated calculation error', result['error_trace_line']
    )
    self.assertNotIn('\n', result['error_trace_line'])

  def test_execute_analytical_script_handles_system_exit(self) -> None:
    script = 'import sys\nprint("Before exit")\nsys.exit(1)\n'
    result = xprof_py_runner.execute_analytical_script(
        code_text=script,
        session_id='sess-exit',
        use_spanner=False,
        read_only_mode=True,
    )
    self.assertEqual(result['status'], 'ERROR')
    self.assertIn('SystemExit: 1', result['error_trace_line'])
    self.assertNotIn('\n', result['error_trace_line'])

  def test_execute_analytical_script_captures_stderr(self) -> None:
    script = (
        'import sys\n'
        'sys.stderr.write("Warning: low memory condition\\n")\n'
        'print("OK")\n'
    )
    result = xprof_py_runner.execute_analytical_script(
        code_text=script,
        session_id='sess-stderr',
        use_spanner=False,
        read_only_mode=True,
    )
    self.assertEqual(result['status'], 'SUCCESS')
    self.assertEqual(result['stdout_line'], 'OK')
    self.assertIn('Warning: low memory condition', result['stderr_line'])
    self.assertNotIn('\n', result['stderr_line'])

  def test_execute_analytical_script_injects_execution_context(self) -> None:
    script = (
        'assert session_id == "context-test-123"\n'
        'assert use_spanner is True\n'
        'assert read_only_mode is True\n'
        'assert np is not None\n'
        'assert pd is not None\n'
        'assert plt is not None\n'
        'print("Context verified.")\n'
    )
    result = xprof_py_runner.execute_analytical_script(
        code_text=script,
        session_id='context-test-123',
        use_spanner=True,
        read_only_mode=True,
    )
    self.assertEqual(result['status'], 'SUCCESS')
    self.assertEqual(result['stdout_line'], 'Context verified.')

  def test_execute_analytical_script_cleans_up_figures(self) -> None:
    script = 'plt.figure()\nplt.plot([1, 2], [3, 4])\n'
    result = xprof_py_runner.execute_analytical_script(
        code_text=script,
        session_id='sess-cleanup',
        use_spanner=False,
        read_only_mode=True,
    )
    self.assertEqual(result['status'], 'SUCCESS')
    # Confirm no figure handles leak after execution.
    self.assertEmpty(plt.get_fignums())

  @flagsaver.flagsaver(
      session_id='cli-sess-456',
      use_spanner=True,
      read_only_mode=True,
      code='print(f"CLI session: {session_id}, spanner={use_spanner}")',
  )
  def test_main_cli_code_flag_execution(self) -> None:
    captured_stdout = io.StringIO()
    old_stdout = sys.stdout
    original_soft, original_hard = resource.getrlimit(resource.RLIMIT_AS)
    sys.stdout = captured_stdout
    try:
      xprof_py_runner.main(['xprof_py_runner'])
    finally:
      sys.stdout = old_stdout
      try:
        resource.setrlimit(resource.RLIMIT_AS, (original_soft, original_hard))
      except (ValueError, resource.error):
        pass

    output_lines = [
        line for line in captured_stdout.getvalue().split('\n') if line
    ]
    self.assertLen(output_lines, 1)
    data: dict[str, Any] = json.loads(output_lines[0])
    self.assertEqual(data['status'], 'SUCCESS')
    self.assertIn(
        'CLI session: cli-sess-456, spanner=True', data['stdout_line']
    )

  def test_main_cli_script_path_execution(self) -> None:
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False
    ) as temp_script:
      temp_script.write('print("Executed from temp file.")\n')
      temp_path = temp_script.name

    try:
      captured_stdout = io.StringIO()
      old_stdout = sys.stdout
      sys.stdout = captured_stdout
      with flagsaver.flagsaver(
          session_id='cli-file-sess',
          code='',
          script_path=temp_path,
      ):
        try:
          xprof_py_runner.main(['xprof_py_runner'])
        finally:
          sys.stdout = old_stdout

      output_lines = [
          line for line in captured_stdout.getvalue().split('\n') if line
      ]
      self.assertLen(output_lines, 1)
      data: dict[str, Any] = json.loads(output_lines[0])
      self.assertEqual(data['status'], 'SUCCESS')
      self.assertEqual(data['stdout_line'], 'Executed from temp file.')
    finally:
      if os.path.exists(temp_path):
        os.remove(temp_path)

  def test_main_cli_missing_script_file_error_json(self) -> None:
    captured_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_stdout
    nonexistent_path = '/tmp/nonexistent_xprof_test_script_12345.py'
    with flagsaver.flagsaver(
        session_id='cli-missing-file',
        code='',
        script_path=nonexistent_path,
    ):
      try:
        xprof_py_runner.main(['xprof_py_runner'])
      finally:
        sys.stdout = old_stdout

    output_lines = [
        line for line in captured_stdout.getvalue().split('\n') if line
    ]
    self.assertLen(output_lines, 1)
    data: dict[str, Any] = json.loads(output_lines[0])
    self.assertEqual(data['status'], 'ERROR')
    self.assertIn('Failed to open script_path', data['error_trace_line'])

  def test_main_cli_stdin_execution(self) -> None:
    captured_stdout = io.StringIO()
    old_stdout = sys.stdout
    old_stdin = sys.stdin
    sys.stdout = captured_stdout
    sys.stdin = io.StringIO('print("Executed from stdin pipe.")')
    with flagsaver.flagsaver(
        session_id='cli-stdin-sess',
        code='',
        script_path='',
    ):
      try:
        xprof_py_runner.main(['xprof_py_runner'])
      finally:
        sys.stdout = old_stdout
        sys.stdin = old_stdin

    output_lines = [
        line for line in captured_stdout.getvalue().split('\n') if line
    ]
    self.assertLen(output_lines, 1)
    data: dict[str, Any] = json.loads(output_lines[0])
    self.assertEqual(data['status'], 'SUCCESS')
    self.assertEqual(data['stdout_line'], 'Executed from stdin pipe.')

  def test_main_cli_positional_arg_execution(self) -> None:
    captured_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_stdout
    with flagsaver.flagsaver(
        session_id='cli-pos-sess',
        code='',
        script_path='',
    ):
      try:
        xprof_py_runner.main(
            ['xprof_py_runner', 'print("Executed from positional arg.")']
        )
      finally:
        sys.stdout = old_stdout

    output_lines = [
        line for line in captured_stdout.getvalue().split('\n') if line
    ]
    self.assertLen(output_lines, 1)
    data: dict[str, Any] = json.loads(output_lines[0])
    self.assertEqual(data['status'], 'SUCCESS')
    self.assertEqual(data['stdout_line'], 'Executed from positional arg.')

  def test_main_cli_positional_arg_file_execution(self) -> None:
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False
    ) as temp_script:
      temp_script.write('print("Executed from positional file path.")\n')
      temp_path = temp_script.name

    try:
      captured_stdout = io.StringIO()
      old_stdout = sys.stdout
      sys.stdout = captured_stdout
      with flagsaver.flagsaver(
          session_id='cli-pos-file-sess',
          code='',
          script_path='',
      ):
        try:
          xprof_py_runner.main(['xprof_py_runner', temp_path])
        finally:
          sys.stdout = old_stdout

      output_lines = [
          line for line in captured_stdout.getvalue().split('\n') if line
      ]
      self.assertLen(output_lines, 1)
      data: dict[str, Any] = json.loads(output_lines[0])
      self.assertEqual(data['status'], 'SUCCESS')
      self.assertEqual(
          data['stdout_line'], 'Executed from positional file path.'
      )
    finally:
      if os.path.exists(temp_path):
        os.remove(temp_path)


if __name__ == '__main__':
  absltest.main()
