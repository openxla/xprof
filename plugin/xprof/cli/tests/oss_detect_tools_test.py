"""Tests the static HLO detectors with the OSS HloProto loader.

Each test writes real `xla.HloProto` files built from the private descriptor
pool of `cli/internal/oss/hlo_tools.py` and makes the detectors read them
through that module, as they do in the open-source package.
"""

from collections.abc import Callable, Sequence
import json
import pathlib
import tempfile
from typing import Any
from unittest import mock

from absl.testing import parameterized

from absl.testing import absltest
from xprof.cli.internal.oss import hlo_tools as oss_hlo_tools
from xprof.cli.tools import detect_layout_mismatch_copies_tool
from xprof.cli.tools import detect_unfused_reshapes_tool
from xprof.cli.tools import detect_unfused_updates_tool
from xprof.cli.tools import detect_unnecessary_convert_dynamic_scale_tool
from xprof.cli.tools import detect_unnecessary_convert_reduce_tool

# xla.PrimitiveType values.
_BF16 = 16
_F32 = 11
_F8E4M3FN = 20

_MB = 1024 * 1024

_DETECTORS: dict[str, Callable[..., str]] = {
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

_TOOL_MODULES = (
    detect_layout_mismatch_copies_tool,
    detect_unfused_reshapes_tool,
    detect_unfused_updates_tool,
    detect_unnecessary_convert_dynamic_scale_tool,
    detect_unnecessary_convert_reduce_tool,
)


def _add_instr(
    computation: Any,
    instr_id: int,
    name: str,
    opcode: str,
    element_type: int = _F32,
    dimensions: Sequence[int] = (),
    minor_to_major: Sequence[int] | None = None,
    operand_ids: Sequence[int] = (),
    called_computation_ids: Sequence[int] = (),
    **kwargs: Any,
) -> Any:
  """Adds an instruction with the given shape to `computation`."""
  instr = computation.instructions.add(
      id=instr_id,
      name=name,
      opcode=opcode,
      operand_ids=operand_ids,
      called_computation_ids=called_computation_ids,
      **kwargs,
  )
  instr.shape.element_type = element_type
  instr.shape.dimensions.extend(dimensions)
  if minor_to_major is not None:
    instr.shape.layout.minor_to_major.extend(minor_to_major)
  return instr


def _new_module(name: str) -> tuple[Any, Any]:
  """Returns a new HloProto and its module named `name`."""
  hlo = oss_hlo_tools._hlo_proto_class()()  # pylint: disable=protected-access
  hlo.hlo_module.name = name
  return hlo, hlo.hlo_module


def _layout_mismatch_module() -> Any:
  """A copy that only changes the layout between a dot and a reduce."""
  hlo, module = _new_module("jit_layout")
  main = module.computations.add(name="main", id=1)
  module.entry_computation_id = 1
  _add_instr(main, 100, "dot_op", "dot", _F32, [128, 256], [1, 0])
  _add_instr(
      main, 200, "copy_op", "copy", _F32, [128, 256], [0, 1], operand_ids=[100]
  )
  _add_instr(main, 300, "reduce_op", "reduce", _F32, operand_ids=[200])
  main.root_id = 300
  return hlo


def _unfused_reshape_module() -> Any:
  """A standalone reshape that feeds a dot."""
  hlo, module = _new_module("jit_reshape")
  main = module.computations.add(name="main", id=1)
  module.entry_computation_id = 1
  _add_instr(main, 1, "p0", "parameter", _F32, [100, 100], parameter_number=0)
  _add_instr(main, 2, "reshape.1", "reshape", _F32, [100, 100], operand_ids=[1])
  _add_instr(main, 3, "dot.1", "dot", _F32, [100, 100], operand_ids=[2])
  main.root_id = 3
  return hlo


def _convert_reduce_module() -> Any:
  """A bf16 -> f32 -> abs -> reduce(add) -> bf16 fusion."""
  hlo, module = _new_module("jit_convert_reduce")
  add_comp = module.computations.add(name="add_comp", id=2)
  _add_instr(add_comp, 20, "x", "parameter", _F32, parameter_number=0)
  _add_instr(add_comp, 21, "y", "parameter", _F32, parameter_number=1)
  _add_instr(add_comp, 22, "add", "add", _F32, operand_ids=[20, 21])
  add_comp.root_id = 22

  fused = module.computations.add(name="my_fusion", id=3)
  _add_instr(fused, 30, "param_0.1", "parameter", _BF16, [100])
  _add_instr(fused, 31, "convert.1", "convert", _F32, [100], operand_ids=[30])
  _add_instr(fused, 32, "abs.1", "abs", _F32, [100], operand_ids=[31])
  _add_instr(fused, 33, "constant.0", "constant", _F32)
  _add_instr(
      fused,
      34,
      "reduce.1",
      "reduce",
      _F32,
      operand_ids=[32, 33],
      called_computation_ids=[2],
      dimensions=[0],
  )
  _add_instr(fused, 35, "convert.2", "convert", _BF16, operand_ids=[34])
  fused.root_id = 35

  main = module.computations.add(name="main", id=1)
  module.entry_computation_id = 1
  _add_instr(main, 10, "param_0", "parameter", _BF16, [100])
  _add_instr(
      main,
      11,
      "fusion.1",
      "fusion",
      _BF16,
      operand_ids=[10],
      called_computation_ids=[3],
  )
  main.root_id = 11
  return hlo


def _dynamic_scale_module() -> Any:
  """A bf16 activation upcast to f32 for both its scale and its fp8 cast."""
  hlo, module = _new_module("jit_dynamic_scale")
  max_comp = module.computations.add(name="max_reducer", id=2)
  _add_instr(max_comp, 20, "x", "parameter", _F32, parameter_number=0)
  _add_instr(max_comp, 21, "y", "parameter", _F32, parameter_number=1)
  _add_instr(max_comp, 22, "max", "maximum", _F32, operand_ids=[20, 21])
  max_comp.root_id = 22

  main = module.computations.add(name="main", id=1)
  module.entry_computation_id = 1
  shape = [128, 256]
  _add_instr(main, 10, "act", "parameter", _BF16, shape, parameter_number=0)
  # Scale branch: max over the f32 upcast of the activation.
  _add_instr(main, 11, "convert.1", "convert", _F32, shape, operand_ids=[10])
  _add_instr(main, 12, "c_inf", "constant", _F32)
  reduce = _add_instr(
      main,
      13,
      "raw_scale",
      "reduce",
      _F32,
      [128],
      operand_ids=[11, 12],
      called_computation_ids=[2],
  )
  reduce.dimensions.append(1)
  _add_instr(
      main, 14, "scale_bcast", "broadcast", _F32, shape, operand_ids=[13]
  )
  # Quantization branch: divide a second f32 upcast by the scale, cast to fp8.
  _add_instr(main, 15, "convert.2", "convert", _F32, shape, operand_ids=[10])
  _add_instr(
      main, 16, "scaled_act", "divide", _F32, shape, operand_ids=[15, 14]
  )
  _add_instr(
      main, 17, "convert.3", "convert", _F8E4M3FN, shape, operand_ids=[16]
  )
  main.root_id = 17
  return hlo


def _unfused_chain_module() -> Any:
  """A chain of three small elementwise ops that are not fused."""
  hlo, module = _new_module("jit_chain")
  main = module.computations.add(name="main", id=1)
  module.entry_computation_id = 1
  _add_instr(main, 1, "p0", "parameter", _F32, [16], parameter_number=0)
  _add_instr(main, 2, "add.1", "add", _F32, [16], operand_ids=[1, 1])
  _add_instr(main, 3, "add.2", "add", _F32, [16], operand_ids=[2, 2])
  _add_instr(main, 4, "mul.3", "multiply", _F32, [16], operand_ids=[3, 3])
  main.root_id = 4
  return hlo


def _clean_module() -> Any:
  """A module with nothing for any detector to flag."""
  hlo, module = _new_module("jit_clean")
  main = module.computations.add(name="main", id=1)
  module.entry_computation_id = 1
  _add_instr(main, 1, "p0", "parameter", _F32, [1024], parameter_number=0)
  _add_instr(main, 2, "abs.1", "abs", _F32, [1024], operand_ids=[1])
  main.root_id = 2
  return hlo


def _top_ops_fn(ops: Sequence[dict[str, Any]]) -> Callable[..., str]:
  """Returns a `get_top_hlo_ops` replacement reporting `ops`."""
  payload = json.dumps(
      {"top_by_time": list(ops), "top_by_bytes_accessed": list(ops)}
  )

  def get_top_hlo_ops(
      session_id: str, limit: int = 50, bypass_cache: bool = False
  ) -> str:
    del session_id, limit, bypass_cache
    return payload

  return get_top_hlo_ops


def _op(
    module: str, instruction: str, category: str = "non-fusion elementwise"
) -> dict[str, Any]:
  """Returns a top op of `module`, named as the profiler names it."""
  return {
      "name": f"by_program/{module}/{category}/{instruction}",
      "category": category,
      "total_self_time_ms": 10.0,
      "occurrences": 1,
      "bytes_accessed": 20 * _MB,
  }


class OssDetectToolsTest(parameterized.TestCase):
  """Runs the detectors on HloProto files parsed by the OSS loader."""

  def setUp(self):
    super().setUp()
    tmp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(tmp_dir.cleanup)
    self.session_dir = pathlib.Path(tmp_dir.name)
    self.hlo_files: list[pathlib.Path] = []
    self.enter_context(
        mock.patch.object(
            oss_hlo_tools,
            "_get_hlo_proto_files",
            side_effect=lambda session_id: list(self.hlo_files),
        )
    )
    for module in _TOOL_MODULES:
      self.enter_context(mock.patch.object(module, "hlo_tools", oss_hlo_tools))

  def _write(self, hlo: Any, program_id: int = 1) -> None:
    path = (
        self.session_dir / f"{hlo.hlo_module.name}({program_id}).hlo_proto.pb"
    )
    path.write_bytes(hlo.SerializeToString())
    self.hlo_files.append(path)

  def _detect(self, name: str, ops: Sequence[dict[str, Any]]) -> dict[str, Any]:
    raw = _DETECTORS[name](
        str(self.session_dir), get_top_hlo_ops_fn=_top_ops_fn(ops)
    )
    return json.loads(raw)

  def test_layout_mismatch_copies_detected(self):
    self._write(_layout_mismatch_module())

    result = self._detect(
        "layout_mismatch_copies", [_op("jit_layout(1)", "copy_op", "copy")]
    )

    self.assertTrue(result["bottlenecks_found"], msg=result)
    (op,) = result["inefficient_ops"]
    self.assertEqual(op["instruction_name"], "copy_op")
    self.assertEqual(op["source_shape"], "f32[128, 256]{1,0}")
    self.assertEqual(op["target_shape"], "f32[128, 256]{0,1}")

  def test_unfused_reshapes_detected(self):
    self._write(_unfused_reshape_module())

    result = self._detect(
        "unfused_reshapes", [_op("jit_reshape(1)", "reshape.1", "reshape")]
    )

    self.assertTrue(result["bottlenecks_found"], msg=result)
    (op,) = result["inefficient_ops"]
    self.assertEqual(op["downstream_compute"], "dot")

  def test_unfused_updates_detected_in_profiled_program_only(self):
    # Two compilations of the same function; only program 9 was profiled.
    self._write(_unfused_chain_module(), program_id=5)
    self._write(_unfused_chain_module(), program_id=9)

    result = self._detect(
        "unfused_updates",
        [
            _op("jit_chain(9)", "add.1"),
            _op("jit_chain(9)", "add.2"),
            _op("jit_chain(9)", "mul.3"),
        ],
    )

    self.assertTrue(result["bottlenecks_found"], msg=result)
    self.assertCountEqual(
        [op["name"] for op in result["inefficient_ops"]],
        [
            "by_program/jit_chain(9)/add.1",
            "by_program/jit_chain(9)/add.2",
            "by_program/jit_chain(9)/mul.3",
        ],
    )
    for op in result["inefficient_ops"]:
      self.assertEqual(op["unfused_chain"], "add.1 -> add.2 -> mul.3")

  def test_unnecessary_convert_reduce_detected(self):
    self._write(_convert_reduce_module())

    result = self._detect(
        "unnecessary_convert_reduce",
        [_op("jit_convert_reduce(1)", "fusion.1", "loop fusion")],
    )

    self.assertTrue(result["bottlenecks_found"], msg=result)
    (op,) = result["inefficient_ops"]
    self.assertEqual(op["instruction"], "convert.1")
    self.assertEqual(op["fusion_name"], "fusion.1")

  def test_unnecessary_convert_dynamic_scale_detected(self):
    self._write(_dynamic_scale_module())

    result = self._detect(
        "unnecessary_convert_dynamic_scale",
        [_op("jit_dynamic_scale(1)", "convert.1")],
    )

    self.assertTrue(result["bottlenecks_found"], msg=result)
    (op,) = result["inefficient_ops"]
    self.assertEqual(op["instruction"], "convert.1")

  @parameterized.parameters(*_DETECTORS)
  def test_clean_module_has_no_findings(self, name):
    self._write(_clean_module())

    result = self._detect(name, [_op("jit_clean(1)", "abs.1")])

    self.assertFalse(result["bottlenecks_found"], msg=result)
    self.assertEmpty(result["inefficient_ops"])

  @parameterized.parameters(*_DETECTORS)
  def test_missing_hlo_protos_raises(self, name):
    with self.assertRaises(FileNotFoundError):
      self._detect(name, [_op("jit_clean(1)", "copy.1", "copy")])


if __name__ == "__main__":
  absltest.main()
