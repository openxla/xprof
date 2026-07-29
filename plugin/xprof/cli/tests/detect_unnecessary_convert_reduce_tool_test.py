"""Tests for the detect_unnecessary_convert_reduce_tool HLO parser suggestions."""

import json
from unittest import mock

from absl.testing import absltest
from tensorflow.compiler.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.xla.python import xla_client  # pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.xla.service import hlo_pb2  # pylint: disable=g-direct-tensorflow-import
from xprof.cli.internal.oss import hlo_tools
from xprof.cli.tools import detect_unnecessary_convert_reduce_tool


def _parse_hlo_text_to_proto(hlo_text: str) -> hlo_pb2.HloModuleProto:
  """Compiles HLO text to C++ module, serializes, and deserializes to proto."""
  module = xla_client.hlo.hlo_module_from_text(hlo_text)
  serialized_bytes = module.as_serialized_hlo_module_proto()
  return hlo_pb2.HloModuleProto.FromString(serialized_bytes)


class MockHloProto:

  def __init__(self, hlo_module):
    self.hlo_module = hlo_module


class MockDebugInfoCollection:

  def __init__(self, hlo_protos, program_ids):
    self.hlo_proto = hlo_protos
    self.program_id = program_ids


class DetectReduceConvertToolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Mock list_hlo_modules and _fetch_debug_info to prevent network calls
    self.mock_list_modules = self.enter_context(
        mock.patch.object(
            hlo_tools, "list_hlo_modules", return_value="No HLO modules found."
        )
    )
    self.mock_fetch_debug = self.enter_context(
        mock.patch.object(
            hlo_tools,
            "_fetch_debug_info",
            side_effect=RuntimeError("Network isolated"),
        )
    )

  def test_parse_hlo_text_to_proto(self):
    hlo_text = """
    HloModule simple_module
    ENTRY entry_computation {
      p0 = f32[128] parameter(0)
      ROOT abs = f32[128] abs(p0)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    self.assertEqual(proto.name, "simple_module")
    self.assertLen(proto.computations, 1)
    self.assertEqual(proto.computations[0].name, "entry_computation")

  def test_build_module_indices(self):
    hlo_text = """
    HloModule test_module

    %fused_computation (p0.1: f32[100]) -> f32[100] {
      %p0.1 = f32[100] parameter(0)
      ROOT %abs.1 = f32[100] abs(%p0.1)
    }

    ENTRY %entry_computation {
      %p0 = f32[100] parameter(0)
      ROOT %fusion.1 = f32[100] fusion(%p0), kind=kLoop, calls=%fused_computation
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_reduce_tool._HloModuleTracer(proto)

    self.assertLen(tracer.computations, 2)
    self.assertLen(tracer.instructions, 4)

    fusion_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "fusion"
    )
    abs_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "abs"
    )
    entry_comp = next(
        c for c in tracer.computations.values() if c.name == "entry_computation"
    )
    fused_comp = next(
        c for c in tracer.computations.values() if c.name == "fused_computation"
    )

    self.assertEqual(
        tracer.instruction_to_computation[fusion_instr.id], entry_comp.id
    )
    self.assertEqual(
        tracer.instruction_to_computation[abs_instr.id], fused_comp.id
    )

    self.assertIn(fused_comp.id, tracer.fusion_callers)
    parent_comp_id, caller_instr = tracer.fusion_callers[fused_comp.id]
    self.assertEqual(parent_comp_id, entry_comp.id)
    self.assertEqual(caller_instr.id, fusion_instr.id)

    param_inside_fusion = next(
        i for i in tracer.instructions.values() if i.name == "p0.1"
    )
    self.assertIn(param_inside_fusion.id, tracer.consumers)
    self.assertLen(tracer.consumers[param_inside_fusion.id], 1)
    self.assertEqual(
        tracer.consumers[param_inside_fusion.id][0].id, abs_instr.id
    )

  def test_classify_execution_phase(self):
    # Case A: Training keywords present (case-insensitive)
    hlo_training = """
    HloModule test_prog
    %adam_optimizer {
      p0 = f32[10] parameter(0)
      ROOT r = f32[10] abs(p0)
    }
    ENTRY %main {
      p0 = f32[10] parameter(0)
      ROOT r = f32[10] abs(p0)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_training)
    self.assertEqual(
        detect_unnecessary_convert_reduce_tool._classify_execution_phase(proto),
        "TRAINING",
    )

    # Case B: Only inference computations present
    hlo_inference = """
    HloModule test_prog
    ENTRY %entry_computation {
      p0 = f32[10] parameter(0)
      ROOT r = f32[10] abs(p0)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_inference)
    self.assertEqual(
        detect_unnecessary_convert_reduce_tool._classify_execution_phase(proto),
        "INFERENCE",
    )

  def test_no_inefficient_ops(self):
    def mock_get_top_ops(session_id, limit):
      del session_id, limit
      return json.dumps({"top_by_time": [], "top_by_bytes_accessed": []})

    result_json = detect_unnecessary_convert_reduce_tool.detect_unnecessary_convert_reduce(
        "session_123",
        get_top_hlo_ops_fn=mock_get_top_ops,
    )
    result = json.loads(result_json)
    self.assertFalse(result["bottlenecks_found"])

  def test_trace_upcast_upstream_direct(self):
    # Setup HLO: %convert = f32[] convert(%p0), %abs = f32[] abs(%convert)
    hlo_text = """
    HloModule test_module
    ENTRY %entry_computation {
      %p0 = bf16[100] parameter(0)
      %convert.1 = f32[100] convert(%p0)
      ROOT %abs.1 = f32[100] abs(%convert.1)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_reduce_tool._HloModuleTracer(proto)
    abs_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "abs"
    )
    convert_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "convert"
    )

    # Trace upstream from abs.1
    found_upcast = tracer.trace_upcast(abs_instr.id)
    self.assertIsNotNone(found_upcast)
    self.assertEqual(found_upcast.id, convert_instr.id)

  def test_trace_upcast_upstream_through_fusion_entry(self):
    # Setup HLO: entry calls fusion. fusion contains convert.
    hlo_text = """
    HloModule test_module
    %fused_comp (p0.1: bf16[100]) -> f32[100] {
      %p0.1 = bf16[100] parameter(0)
      %convert.1 = f32[100] convert(%p0.1)
      ROOT %abs.1 = f32[100] abs(%convert.1)
    }
    ENTRY %entry_computation {
      %p0 = bf16[100] parameter(0)
      ROOT %fusion.1 = f32[100] fusion(%p0), kind=kLoop, calls=%fused_comp
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_reduce_tool._HloModuleTracer(proto)
    fusion_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "fusion"
    )
    convert_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "convert"
    )

    # Trace upstream starting from fusion.1.
    found_upcast = tracer.trace_upcast(fusion_instr.id)
    self.assertIsNotNone(found_upcast)
    self.assertEqual(found_upcast.id, convert_instr.id)

  def test_trace_upcast_upstream_fusion_exit(
      self,
  ):
    # Setup HLO: convert is outside fusion. Inside fusion we have abs
    # and parameter.
    hlo_text = """
    HloModule test_module
    %fused_comp (p0.1: f32[100]) -> f32[100] {
      %p0.1 = f32[100] parameter(0)
      ROOT %abs.1 = f32[100] abs(%p0.1)
    }
    ENTRY %entry_computation {
      %p0 = bf16[100] parameter(0)
      %convert.1 = f32[100] convert(%p0)
      ROOT %fusion.1 = f32[100] fusion(%convert.1), kind=kLoop, calls=%fused_comp
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_reduce_tool._HloModuleTracer(proto)
    abs_instr = next(
        i
        for i in tracer.instructions.values()
        if i.opcode == "abs" and i.name == "abs.1"
    )
    convert_instr = next(
        i
        for i in tracer.instructions.values()
        if i.opcode == "convert" and i.name == "convert.1"
    )

    # Trace upstream from abs.1 inside fusion.
    found_upcast = tracer.trace_upcast(abs_instr.id)
    self.assertIsNotNone(found_upcast)
    self.assertEqual(found_upcast.id, convert_instr.id)

  def test_trace_upcast_upstream_aborts_on_heavy_math(self):
    # Setup HLO: convert -> dot -> abs
    hlo_text = """
    HloModule test_module
    ENTRY %entry_computation {
      %p0 = bf16[100,100] parameter(0)
      %convert.1 = f32[100,100] convert(%p0)
      %dot.1 = f32[100,100] dot(%convert.1, %convert.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT %abs.1 = f32[100,100] abs(%dot.1)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_reduce_tool._HloModuleTracer(proto)
    abs_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "abs"
    )

    # Trace upstream from abs.1.
    found_upcast = tracer.trace_upcast(abs_instr.id)
    self.assertIsNone(found_upcast)

  def test_detect_unnecessary_convert_reduce_integration(self):
    hlo_content = """
    HloModule jit_my_entry_comp

    %add_comp (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(%x, %y)
    }

    %my_fusion (param_0.1: bf16[100]) -> bf16[] {
      %param_0.1 = bf16[100] parameter(0)
      %convert.1 = f32[100] convert(%param_0.1)
      %abs.1 = f32[100] abs(%convert.1)
      %constant.0 = f32[] constant(0)
      %reduce.1 = f32[] reduce(%abs.1, %constant.0), dimensions={0}, to_apply=add_comp
      ROOT %convert.2 = bf16[] convert(%reduce.1)
    }

    ENTRY my_entry_comp (param_0: bf16[100]) -> bf16[] {
      %param_0 = bf16[100] parameter(0)
      ROOT %fusion.1 = bf16[] fusion(%param_0), kind=kLoop, calls=my_fusion
    }
    """

    result = self._run_detect(hlo_content)
    self.assertTrue(result["bottlenecks_found"], msg=result)
    self.assertLen(result["inefficient_ops"], 1)
    self.assertEqual(result["inefficient_ops"][0]["instruction"], "convert.1")
    self.assertEqual(result["inefficient_ops"][0]["fusion_name"], "fusion.1")
    self.assertIn(
        "Detected candidate unnecessary promotion",
        result["inefficient_ops"][0]["recommendation"],
    )

  def test_detect_unnecessary_convert_reduce_mismatched_param_names(self):
    # Setup HLO where parameter variables inside the body do not match
    # signature names

    hlo_content = """
    HloModule jit_my_entry_comp

    %add_comp (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(%x, %y)
    }

    %my_fusion (Arg_0: bf16[100]) -> bf16[] {
      %param_0 = bf16[100] parameter(0)
      %convert.1 = f32[100] convert(%param_0)
      %abs.1 = f32[100] abs(%convert.1)
      %constant.0 = f32[] constant(0)
      %reduce.1 = f32[] reduce(%abs.1, %constant.0), dimensions={0}, to_apply=add_comp
      ROOT %convert.2 = bf16[] convert(%reduce.1)
    }

    ENTRY my_entry_comp (param_0: bf16[100]) -> bf16[] {
      %p0 = bf16[100] parameter(0)
      ROOT %fusion.1 = bf16[] fusion(%p0), kind=kLoop, calls=my_fusion
    }
    """

    result = self._run_detect(hlo_content)
    self.assertTrue(
        result["bottlenecks_found"],
        "Failed to detect bottleneck due to mismatched parameter names!"
        f" Result: {result}",
    )
    self.assertLen(result["inefficient_ops"], 1)
    self.assertEqual(result["inefficient_ops"][0]["instruction"], "convert.1")
    self.assertEqual(result["inefficient_ops"][0]["fusion_name"], "fusion.1")

  def _run_detect(
      self, hlo_content, target="fusion.1", module="jit_my_entry_comp"
  ):
    def mock_get_top_ops(session_id, limit):
      del session_id, limit
      return json.dumps({
          "top_by_time": [{
              "name": f"by_program/{module}/{target}",
              "total_self_time_ms": 10.0,
          }],
          "top_by_bytes_accessed": [],
      })

    def mock_fetch_debug_info(session_id):
      del session_id
      hlo_proto_mock = MockHloProto(_parse_hlo_text_to_proto(hlo_content))
      return MockDebugInfoCollection([hlo_proto_mock], [123])

    self.enter_context(
        mock.patch.object(
            hlo_tools, "_fetch_debug_info", side_effect=mock_fetch_debug_info
        )
    )
    return json.loads(
        detect_unnecessary_convert_reduce_tool.detect_unnecessary_convert_reduce(
            "session_123", get_top_hlo_ops_fn=mock_get_top_ops
        )
    )

  def test_detect_skips_fan_out_f32_result(self):
    # Regression: the reduce result is downcast to bf16 AND also output as f32.
    # The f32 is not discarded, so the upcast must NOT be flagged.
    hlo_content = """
    HloModule jit_my_entry_comp

    %add_comp (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(%x, %y)
    }

    %my_fusion (param_0.1: bf16[100]) -> (bf16[], f32[]) {
      %param_0.1 = bf16[100] parameter(0)
      %convert.1 = f32[100] convert(%param_0.1)
      %sq = f32[100] multiply(%convert.1, %convert.1)
      %constant.0 = f32[] constant(0)
      %reduce.1 = f32[] reduce(%sq, %constant.0), dimensions={0}, to_apply=add_comp
      %convert.2 = bf16[] convert(%reduce.1)
      ROOT %t = (bf16[], f32[]) tuple(%convert.2, %reduce.1)
    }

    ENTRY my_entry_comp (param_0: bf16[100]) -> (bf16[], f32[]) {
      %param_0 = bf16[100] parameter(0)
      ROOT %fusion.1 = (bf16[], f32[]) fusion(%param_0), kind=kLoop, calls=my_fusion
    }
    """
    result = self._run_detect(hlo_content)
    self.assertFalse(result["bottlenecks_found"], msg=result)

  def test_detect_skips_signed_reduction(self):
    # A plain signed sum (no square/abs/exp) may need f32 accumulation; skip it.
    hlo_content = """
    HloModule jit_my_entry_comp

    %add_comp (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(%x, %y)
    }

    %my_fusion (param_0.1: bf16[100]) -> bf16[] {
      %param_0.1 = bf16[100] parameter(0)
      %convert.1 = f32[100] convert(%param_0.1)
      %constant.0 = f32[] constant(0)
      %reduce.1 = f32[] reduce(%convert.1, %constant.0), dimensions={0}, to_apply=add_comp
      ROOT %convert.2 = bf16[] convert(%reduce.1)
    }

    ENTRY my_entry_comp (param_0: bf16[100]) -> bf16[] {
      %param_0 = bf16[100] parameter(0)
      ROOT %fusion.1 = bf16[] fusion(%param_0), kind=kLoop, calls=my_fusion
    }
    """
    result = self._run_detect(hlo_content)
    self.assertFalse(result["bottlenecks_found"], msg=result)

  def test_detect_skips_training(self):
    # Even a matchable non-negative reduction is skipped in training modules.
    hlo_content = """
    HloModule jit_my_entry_comp

    %add_comp (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(%x, %y)
    }

    %my_fusion_backward (param_0.1: bf16[100]) -> bf16[] {
      %param_0.1 = bf16[100] parameter(0)
      %convert.1 = f32[100] convert(%param_0.1)
      %abs.1 = f32[100] abs(%convert.1)
      %constant.0 = f32[] constant(0)
      %reduce.1 = f32[] reduce(%abs.1, %constant.0), dimensions={0}, to_apply=add_comp
      ROOT %convert.2 = bf16[] convert(%reduce.1)
    }

    ENTRY my_entry_comp (param_0: bf16[100]) -> bf16[] {
      %param_0 = bf16[100] parameter(0)
      ROOT %fusion.1 = bf16[] fusion(%param_0), kind=kLoop, calls=my_fusion_backward
    }
    """
    result = self._run_detect(hlo_content)
    self.assertFalse(result["bottlenecks_found"], msg=result)

  def test_calculate_reduction_size(self):
    shape = xla_data_pb2.ShapeProto()
    shape.dimensions.extend([10, 20, 30])
    # Collapsing dimensions 0 and 2: size = 10 * 30 = 300
    size = detect_unnecessary_convert_reduce_tool._calculate_reduction_size(
        shape, [0, 2]
    )
    self.assertEqual(size, 300)

    # Dynamic shapes or invalid sizes <= 0
    shape_dynamic = xla_data_pb2.ShapeProto()
    shape_dynamic.dimensions.extend([-1, 20, 30])
    size_dynamic = (
        detect_unnecessary_convert_reduce_tool._calculate_reduction_size(
            shape_dynamic, [0, 1]
        )
    )
    self.assertEqual(size_dynamic, 0)

    # Out of bounds dimension index should be skipped safely
    size_oob = detect_unnecessary_convert_reduce_tool._calculate_reduction_size(
        shape, [0, 5]
    )
    self.assertEqual(size_oob, 10)

  def test_classify_reducer(self):
    tool = detect_unnecessary_convert_reduce_tool
    add_reducer = """
      %add (x: f32[], y: f32[]) -> f32[] {
        %x = f32[] parameter(0)
        %y = f32[] parameter(1)
        ROOT %a = f32[] add(%x, %y)
      }
    """

    def _reduce_class(entry_body):
      hlo = f"HloModule m\n{add_reducer}\nENTRY e {{\n{entry_body}\n}}"
      tracer = tool._HloModuleTracer(_parse_hlo_text_to_proto(hlo))
      reduce_instr = next(
          i for i in tracer.instructions.values() if i.opcode == "reduce"
      )
      return tracer.classify_reducer(reduce_instr)

    # x * x -> sum of squares.
    self.assertEqual(
        _reduce_class(
            "  p0 = f32[100] parameter(0)\n"
            "  sq = f32[100] multiply(p0, p0)\n"
            "  c0 = f32[] constant(0)\n"
            "  ROOT r = f32[] reduce(sq, c0), dimensions={0}, to_apply=%add"
        ),
        tool._REDUCER_SUM_OF_SQUARES,
    )
    # abs -> non-negative.
    self.assertEqual(
        _reduce_class(
            "  p0 = f32[100] parameter(0)\n"
            "  a = f32[100] abs(p0)\n"
            "  c0 = f32[] constant(0)\n"
            "  ROOT r = f32[] reduce(a, c0), dimensions={0}, to_apply=%add"
        ),
        tool._REDUCER_SUM_NONNEG,
    )
    # exponential -> non-negative.
    self.assertEqual(
        _reduce_class(
            "  p0 = f32[100] parameter(0)\n"
            "  e = f32[100] exponential(p0)\n"
            "  c0 = f32[] constant(0)\n"
            "  ROOT r = f32[] reduce(e, c0), dimensions={0}, to_apply=%add"
        ),
        tool._REDUCER_SUM_NONNEG,
    )
    # Raw (signed) input -> signed sum (not matched by the tool).
    self.assertEqual(
        _reduce_class(
            "  p0 = f32[100] parameter(0)\n"
            "  c0 = f32[] constant(0)\n"
            "  ROOT r = f32[] reduce(p0, c0), dimensions={0}, to_apply=%add"
        ),
        tool._REDUCER_SUM_SIGNED,
    )

    # Product reducer.
    hlo_prod = """
    HloModule m
    %mul (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %m = f32[] multiply(%x, %y)
    }
    ENTRY e {
      p0 = f32[100] parameter(0)
      c1 = f32[] constant(1)
      ROOT r = f32[] reduce(p0, c1), dimensions={0}, to_apply=%mul
    }
    """
    tracer = tool._HloModuleTracer(_parse_hlo_text_to_proto(hlo_prod))
    reduce_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "reduce"
    )
    self.assertEqual(
        tracer.classify_reducer(reduce_instr), tool._REDUCER_PRODUCT
    )

  def test_result_escapes_as_f32(self):
    tool = detect_unnecessary_convert_reduce_tool
    reducer = """
    %add (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %a = f32[] add(%x, %y)
    }
    """

    # Fully discarded: the reduce result is only downcast back to bf16.
    hlo_ok = f"""
    HloModule m
    {reducer}
    ENTRY e (p: bf16[100]) -> bf16[] {{
      %p = bf16[100] parameter(0)
      %c = f32[100] convert(%p)
      %sq = f32[100] multiply(%c, %c)
      %z = f32[] constant(0)
      %r = f32[] reduce(%sq, %z), dimensions={{0}}, to_apply=%add
      ROOT %d = bf16[] convert(%r)
    }}
    """
    tracer = tool._HloModuleTracer(_parse_hlo_text_to_proto(hlo_ok))
    r = next(i for i in tracer.instructions.values() if i.opcode == "reduce")
    self.assertFalse(tracer.result_escapes_as_f32(r.id))

    # Fan-out: the reduce result also leaves the module as f32 (reported bug).
    hlo_escape = f"""
    HloModule m
    {reducer}
    ENTRY e (p: bf16[100]) -> (bf16[], f32[]) {{
      %p = bf16[100] parameter(0)
      %c = f32[100] convert(%p)
      %sq = f32[100] multiply(%c, %c)
      %z = f32[] constant(0)
      %r = f32[] reduce(%sq, %z), dimensions={{0}}, to_apply=%add
      %d = bf16[] convert(%r)
      ROOT %t = (bf16[], f32[]) tuple(%d, %r)
    }}
    """
    tracer = tool._HloModuleTracer(_parse_hlo_text_to_proto(hlo_escape))
    r = next(i for i in tracer.instructions.values() if i.opcode == "reduce")
    self.assertTrue(tracer.result_escapes_as_f32(r.id))

  def test_result_escapes_via_shared_computation(self):
    tool = detect_unnecessary_convert_reduce_tool
    # %shared is called by BOTH %fb and %fa. Through %fa the result is downcast
    # to bf16 (safe), but through %fb it escapes the module as f32. %fa is
    # defined last, so a single-caller (overwriting) map would only follow the
    # safe path and wrongly report no escape. All callers must be traced.
    hlo = """
    HloModule m
    %add (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %a = f32[] add(%x, %y)
    }
    %shared (p: bf16[100]) -> f32[] {
      %p = bf16[100] parameter(0)
      %c = f32[100] convert(%p)
      %sq = f32[100] multiply(%c, %c)
      %z = f32[] constant(0)
      ROOT %r = f32[] reduce(%sq, %z), dimensions={0}, to_apply=%add
    }
    ENTRY e (a: bf16[100], b: bf16[100]) -> (bf16[], f32[]) {
      %a = bf16[100] parameter(0)
      %b = bf16[100] parameter(1)
      %fb = f32[] fusion(%b), kind=kLoop, calls=%shared
      %fa = f32[] fusion(%a), kind=kLoop, calls=%shared
      %da = bf16[] convert(%fa)
      ROOT %t = (bf16[], f32[]) tuple(%da, %fb)
    }
    """
    tracer = tool._HloModuleTracer(_parse_hlo_text_to_proto(hlo))
    shared_comp = next(
        c for c in tracer.computations.values() if c.name == "shared"
    )
    # Both call sites are recorded.
    self.assertLen(tracer.computation_callers[shared_comp.id], 2)
    r = next(i for i in tracer.instructions.values() if i.opcode == "reduce")
    self.assertTrue(tracer.result_escapes_as_f32(r.id))

  def test_upcast_serves_only_reduce(self):
    tool = detect_unnecessary_convert_reduce_tool
    reducer = """
    %add (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %a = f32[] add(%x, %y)
    }
    """

    def _f32_convert_and_reduce(hlo):
      tracer = tool._HloModuleTracer(_parse_hlo_text_to_proto(hlo))
      upcast = next(
          i
          for i in tracer.instructions.values()
          if i.opcode == "convert" and i.shape.element_type == tool._F32_TYPE
      )
      reduce_instr = next(
          i for i in tracer.instructions.values() if i.opcode == "reduce"
      )
      return tracer, upcast, reduce_instr

    # Exclusive: the upcast only feeds the reduction.
    hlo_excl = f"""
    HloModule m
    {reducer}
    ENTRY e (p: bf16[100]) -> bf16[] {{
      %p = bf16[100] parameter(0)
      %c = f32[100] convert(%p)
      %sq = f32[100] multiply(%c, %c)
      %z = f32[] constant(0)
      %r = f32[] reduce(%sq, %z), dimensions={{0}}, to_apply=%add
      ROOT %d = bf16[] convert(%r)
    }}
    """
    tracer, upcast, reduce_instr = _f32_convert_and_reduce(hlo_excl)
    self.assertTrue(
        tracer.upcast_serves_only_reduce(upcast.id, reduce_instr.id)
    )

    # Shared: the upcast also escapes the module as f32.
    hlo_shared = f"""
    HloModule m
    {reducer}
    ENTRY e (p: bf16[100]) -> (bf16[], f32[100]) {{
      %p = bf16[100] parameter(0)
      %c = f32[100] convert(%p)
      %sq = f32[100] multiply(%c, %c)
      %z = f32[] constant(0)
      %r = f32[] reduce(%sq, %z), dimensions={{0}}, to_apply=%add
      %d = bf16[] convert(%r)
      ROOT %t = (bf16[], f32[100]) tuple(%d, %c)
    }}
    """
    tracer, upcast, reduce_instr = _f32_convert_and_reduce(hlo_shared)
    self.assertFalse(
        tracer.upcast_serves_only_reduce(upcast.id, reduce_instr.id)
    )

  def test_deep_graph_recursion_limit(self):
    # Generates a very deep computation chain to verify the iterative
    # implementation does not hit RecursionError (which recursive DFS would
    # do for depth > 1000).
    lines = [
        "HloModule deep_module",
        "ENTRY entry_computation {",
        "  p0 = f16[10] parameter(0)",
        "  c0 = f32[10] convert(p0)",
    ]

    # Add 1100 intermediate dummy element-wise operations (abs) to create depth
    prev = "c0"
    for i in range(1, 1100):
      curr = f"x_{i}"
      lines.append(f"  {curr} = f32[10] abs({prev})")
      prev = curr

    lines.append(f"  ROOT r = f32[10] abs({prev})")
    lines.append("}")

    hlo_text = "\n".join(lines)
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_reduce_tool._HloModuleTracer(proto)

    # We trace from the ROOT instruction id
    root_id = proto.computations[0].root_id

    # This should run without throwing RecursionError!
    upcast_instr = tracer.trace_upcast(root_id)
    self.assertIsNotNone(upcast_instr)
    self.assertEqual(upcast_instr.opcode, "convert")


if __name__ == "__main__":
  absltest.main()
