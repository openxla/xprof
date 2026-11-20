"""End-to-end tests for the static HLO detectors on the bundled demo traces.

The detectors read HLO protos through `cli/internal/oss/hlo_tools.py`, which
parses them with a private descriptor pool, so these tests need no TensorFlow.
"""

import json
import os
import pathlib
import shutil
import tempfile
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from xprof.cli.internal.oss import hlo_tools as oss_hlo_tools
from xprof.cli.tools import detect_layout_mismatch_copies_tool
from xprof.cli.tools import detect_unfused_reshapes_tool
from xprof.cli.tools import detect_unfused_updates_tool
from xprof.cli.tools import detect_unnecessary_convert_dynamic_scale_tool
from xprof.cli.tools import detect_unnecessary_convert_reduce_tool

_TRACES = {
    "tpu-training": "tpu-training/gke-tpu-b309f56b-rq5s.xplane.pb",
    "v6e-4-training": "v6e-4-training/t1v-n-9bfa07b4-w-0.xplane.pb",
}

_TOOL_MODULES = (
    detect_layout_mismatch_copies_tool,
    detect_unfused_reshapes_tool,
    detect_unfused_updates_tool,
    detect_unnecessary_convert_dynamic_scale_tool,
    detect_unnecessary_convert_reduce_tool,
)

_DETECTORS = {
    "layout_mismatch_copies": (
        detect_layout_mismatch_copies_tool.detect_layout_mismatch_copies
    ),
    "unfused_reshapes": detect_unfused_reshapes_tool.detect_unfused_reshapes,
    "unfused_updates": detect_unfused_updates_tool.detect_unfused_updates,
    "unnecessary_convert_dynamic_scale": (
        detect_unnecessary_convert_dynamic_scale_tool.detect_unnecessary_convert_dynamic_scale
    ),
    "unnecessary_convert_reduce": (
        detect_unnecessary_convert_reduce_tool.detect_unnecessary_convert_reduce
    ),
}


def _demo_trace_path(rel_path: str) -> str:
  """Returns the path of a bundled demo trace.

  Args:
    rel_path: Path of the trace relative to the demo profile directory.

  Raises:
    FileNotFoundError: If the trace is not found.
  """
  candidates = (
      os.path.join(
          os.path.dirname(__file__),
          "../../../../../demo/plugins/profile",
          rel_path,
      ),
  )
  for candidate in candidates:
    if os.path.exists(candidate):
      return candidate
  raise FileNotFoundError(f"Demo trace not found: {rel_path}")


def _run_detector(name: str, session_id: str) -> dict[str, Any]:
  """Runs the `detect_<name>` tool on `session_id` and parses its JSON."""
  return json.loads(_DETECTORS[name](session_id))


def _instructions(result: dict[str, Any]) -> list[str]:
  """Returns the sorted instruction names reported by a detector."""
  names = []
  for op in result["inefficient_ops"]:
    name = op.get("instruction") or op.get("instruction_name")
    names.append(name or op["name"].rsplit("/", 1)[-1])
  return sorted(names)


class DetectToolsE2ETest(parameterized.TestCase):
  """Runs the five detectors on real traces and checks their findings."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Copy each trace to its own writable run directory, since generating the
    # HLO protos writes next to the trace.
    cls._tmp_dir = tempfile.mkdtemp()
    cls._sessions = {}
    for trace_name, rel_path in _TRACES.items():
      src = _demo_trace_path(rel_path)
      run_dir = pathlib.Path(cls._tmp_dir) / trace_name
      run_dir.mkdir()
      shutil.copy(src, run_dir / os.path.basename(rel_path))
      cls._sessions[trace_name] = str(run_dir)

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls._tmp_dir, ignore_errors=True)
    super().tearDownClass()

  def setUp(self):
    super().setUp()
    # Make every detector read HLO protos through the open-source loader, as in
    # the released package.
    for module in _TOOL_MODULES:
      self.enter_context(mock.patch.object(module, "hlo_tools", oss_hlo_tools))

  @parameterized.named_parameters(
      # Findings on the bundled traces with the current detector logic.
      (
          "tpu_layout_mismatch_copies",
          "tpu-training",
          "layout_mismatch_copies",
          [
              "copy.1511",
              "copy.1512",
              "copy.1516",
              "copy.1534",
              "copy.1536",
              "copy.1543",
          ],
      ),
      (
          "tpu_unfused_reshapes",
          "tpu-training",
          "unfused_reshapes",
          [
              "copy.1511",
              "copy.1512",
              "copy.1513",
              "copy.1516",
              "copy.1535",
              "copy.1536",
              "copy.1543",
              "copy.1546",
          ],
      ),
      (
          "tpu_unfused_updates",
          "tpu-training",
          "unfused_updates",
          [
              "copy.1511",
              "copy.1512",
              "copy.1516",
              "copy.1534",
              "copy.1536",
              "copy.1543",
          ],
      ),
      (
          "tpu_unnecessary_convert_dynamic_scale",
          "tpu-training",
          "unnecessary_convert_dynamic_scale",
          [],
      ),
      # `jit_train_step` is a training program, where reductions keep f32.
      (
          "tpu_unnecessary_convert_reduce",
          "tpu-training",
          "unnecessary_convert_reduce",
          [],
      ),
      (
          "v6e_layout_mismatch_copies",
          "v6e-4-training",
          "layout_mismatch_copies",
          [],
      ),
      (
          "v6e_unfused_reshapes",
          "v6e-4-training",
          "unfused_reshapes",
          ["convert_element_type.658"],
      ),
      # The only small fusions here are async collectives, which are skipped.
      (
          "v6e_unfused_updates",
          "v6e-4-training",
          "unfused_updates",
          [],
      ),
      (
          "v6e_unnecessary_convert_dynamic_scale",
          "v6e-4-training",
          "unnecessary_convert_dynamic_scale",
          [],
      ),
      (
          "v6e_unnecessary_convert_reduce",
          "v6e-4-training",
          "unnecessary_convert_reduce",
          [],
      ),
  )
  def test_detector_findings(self, trace_name, detector, expected):
    """Checks the instructions each detector reports on a demo trace."""
    result = _run_detector(detector, self._sessions[trace_name])

    self.assertEqual(result["bottlenecks_found"], bool(expected))
    self.assertEqual(_instructions(result), expected)

  def test_unfused_updates_findings_are_in_profiled_program(self):
    result = _run_detector("unfused_updates", self._sessions["tpu-training"])

    self.assertNotEmpty(result["inefficient_ops"])
    for op in result["inefficient_ops"]:
      self.assertStartsWith(
          op["name"], "by_program/jit_train_step(4869159985936022652)/"
      )

  def test_layout_mismatch_copy_reports_layouts(self):
    result = _run_detector(
        "layout_mismatch_copies", self._sessions["tpu-training"]
    )

    ops = {op["instruction_name"]: op for op in result["inefficient_ops"]}
    self.assertEqual(
        ops["copy.1516"]["source_shape"], "bf16[1, 8, 16384, 6144]{2,1,0,3}"
    )
    self.assertEqual(
        ops["copy.1516"]["target_shape"], "bf16[1, 8, 16384, 6144]{3,2,1,0}"
    )


if __name__ == "__main__":
  absltest.main()
