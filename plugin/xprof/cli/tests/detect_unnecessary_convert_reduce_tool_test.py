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
    tracer = detect_unnecessary_convert_reduce_tool.HloModuleTracer(proto)

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

  def test_verify_reducer(self):
    # Case A: Valid reducer (Add)
    hlo_add = """
    HloModule test_prog
    %add_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(%x, %y)
    }
    ENTRY %main {
      %p0 = f32[10,10] parameter(0)
      %c0 = f32[] constant(0)
      ROOT %reduce = f32[10] reduce(%p0, %c0), dimensions={1}, to_apply=%add_reducer
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_add)
    tracer = detect_unnecessary_convert_reduce_tool.HloModuleTracer(proto)
    reduce_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "reduce"
    )
    self.assertTrue(tracer.verify_reducer(reduce_instr))

    # Case B: Invalid reducer (Maximum)
    hlo_max = """
    HloModule test_prog
    %max_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %maximum = f32[] maximum(%x, %y)
    }
    ENTRY %main {
      %p0 = f32[10,10] parameter(0)
      %c0 = f32[] constant(0)
      ROOT %reduce = f32[10] reduce(%p0, %c0), dimensions={1}, to_apply=%max_reducer
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_max)
    tracer = detect_unnecessary_convert_reduce_tool.HloModuleTracer(proto)
    reduce_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "reduce"
    )
    self.assertFalse(tracer.verify_reducer(reduce_instr))

  def test_verify_reducer_collective(self):
    # Case A: Valid all-reduce
    hlo_all_reduce = """
    HloModule test_prog
    %add_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(%x, %y)
    }
    ENTRY %main {
      %p0 = f32[10,10] parameter(0)
      ROOT %all-reduce = f32[10,10] all-reduce(%p0),
        replica_groups={}, to_apply=%add_reducer
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_all_reduce)
    tracer = detect_unnecessary_convert_reduce_tool.HloModuleTracer(proto)
    ar_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "all-reduce"
    )
    self.assertTrue(tracer.verify_reducer(ar_instr))

    # Case B: Valid reduce-scatter
    hlo_reduce_scatter = """
    HloModule test_prog
    %add_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(%x, %y)
    }
    ENTRY %main {
      %p0 = f32[10,10] parameter(0)
      ROOT %reduce-scatter = f32[5,10] reduce-scatter(%p0),
        dimensions={0}, replica_groups={}, to_apply=%add_reducer
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_reduce_scatter)
    tracer = detect_unnecessary_convert_reduce_tool.HloModuleTracer(proto)
    rs_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "reduce-scatter"
    )
    self.assertTrue(tracer.verify_reducer(rs_instr))

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
    tracer = detect_unnecessary_convert_reduce_tool.HloModuleTracer(proto)
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
    tracer = detect_unnecessary_convert_reduce_tool.HloModuleTracer(proto)
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
    tracer = detect_unnecessary_convert_reduce_tool.HloModuleTracer(proto)
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
    tracer = detect_unnecessary_convert_reduce_tool.HloModuleTracer(proto)
    abs_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "abs"
    )

    # Trace upstream from abs.1.
    found_upcast = tracer.trace_upcast(abs_instr.id)
    self.assertIsNone(found_upcast)

  def test_trace_downcast_downstream_direct(self):
    # Setup HLO: %abs -> %convert (f32 -> bf16)
    hlo_text = """
    HloModule test_module
    ENTRY %entry_computation {
      %p0 = f32[100] parameter(0)
      %abs.1 = f32[100] abs(%p0)
      ROOT %convert.1 = bf16[100] convert(%abs.1)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_reduce_tool.HloModuleTracer(proto)
    abs_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "abs"
    )

    # Trace downstream from abs.1.
    found_downcast = tracer.trace_downcast(abs_instr.id)
    self.assertTrue(found_downcast)

  def test_trace_downcast_downstream_through_fusion_entry(self):
    # Setup HLO: entry passes f32 to fusion. Inside fusion, abs -> convert.
    hlo_text = """
    HloModule test_module
    %fused_comp (p0.1: f32[100]) -> bf16[100] {
      %p0.1 = f32[100] parameter(0)
      %abs.1 = f32[100] abs(%p0.1)
      ROOT %convert.1 = bf16[100] convert(%abs.1)
    }
    ENTRY %entry_computation {
      %p0 = f32[100] parameter(0)
      ROOT %fusion.1 = bf16[100] fusion(%p0), kind=kLoop, calls=%fused_comp
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_reduce_tool.HloModuleTracer(proto)
    p0_instr = next(
        i
        for i in tracer.instructions.values()
        if i.opcode == "parameter" and i.name == "p0"
    )

    # Trace downstream starting from parameter p0.
    found_downcast = tracer.trace_downcast(p0_instr.id)
    self.assertTrue(found_downcast)

  def test_trace_downcast_downstream_through_fusion_exit(self):
    # Setup HLO: inside fusion is abs (f32). convert is outside fusion.
    hlo_text = """
    HloModule test_module
    %fused_comp (p0.1: f32[100]) -> f32[100] {
      %p0.1 = f32[100] parameter(0)
      ROOT %abs.1 = f32[100] abs(%p0.1)
    }
    ENTRY %entry_computation {
      %p0 = f32[100] parameter(0)
      %fusion.1 = f32[100] fusion(%p0), kind=kLoop, calls=%fused_comp
      ROOT %convert.1 = bf16[100] convert(%fusion.1)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_reduce_tool.HloModuleTracer(proto)
    abs_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "abs"
    )

    # Trace downstream from abs.1 inside fusion.
    found_downcast = tracer.trace_downcast(abs_instr.id)
    self.assertTrue(found_downcast)

  def test_trace_downcast_downstream_aborts_on_heavy_math(self):
    # Setup HLO: abs -> dot -> convert
    hlo_text = """
    HloModule test_module
    ENTRY %entry_computation {
      %p0 = f32[100,100] parameter(0)
      %abs.1 = f32[100,100] abs(%p0)
      %dot.1 = f32[100,100] dot(%abs.1, %abs.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT %convert.1 = bf16[100,100] convert(%dot.1)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_reduce_tool.HloModuleTracer(proto)
    abs_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "abs"
    )

    # Trace downstream starting from abs.1.
    found_downcast = tracer.trace_downcast(abs_instr.id)
    self.assertFalse(found_downcast)

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

    def mock_get_top_ops(session_id, limit):
      del session_id, limit
      return json.dumps({
          "top_by_time": [{
              "name": "by_program/jit_my_entry_comp/fusion.1",
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

    result_json = detect_unnecessary_convert_reduce_tool.detect_unnecessary_convert_reduce(
        "session_123",
        get_top_hlo_ops_fn=mock_get_top_ops,
    )
    result = json.loads(result_json)
    if not result.get("bottlenecks_found"):
      print(f"DEBUG: result = {result_json}")
    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)
    self.assertEqual(result["inefficient_ops"][0]["instruction"], "convert.1")
    self.assertEqual(result["inefficient_ops"][0]["fusion_name"], "fusion.1")
    self.assertIn(
        "Detected unnecessary promotion pattern",
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

    def mock_get_top_ops(session_id, limit):
      del session_id, limit
      return json.dumps({
          "top_by_time": [{
              "name": "by_program/jit_my_entry_comp/fusion.1",
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

    result_json = detect_unnecessary_convert_reduce_tool.detect_unnecessary_convert_reduce(
        "session_123",
        get_top_hlo_ops_fn=mock_get_top_ops,
    )
    result = json.loads(result_json)
    self.assertTrue(
        result["bottlenecks_found"],
        "Failed to detect bottleneck due to mismatched parameter names!"
        f" Result: {result_json}",
    )
    self.assertLen(result["inefficient_ops"], 1)
    self.assertEqual(result["inefficient_ops"][0]["instruction"], "convert.1")
    self.assertEqual(result["inefficient_ops"][0]["fusion_name"], "fusion.1")

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

  def test_classify_context(self):
    # Setup HLO module to parse
    hlo_text = """
    HloModule test_module
    ENTRY %entry_computation {
      p0 = f32[10] parameter(0)
      ROOT r = f32[10] abs(p0)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_reduce_tool.HloModuleTracer(proto)
    instr = next(i for i in tracer.instructions.values() if i.opcode == "abs")

    # Case 1: Metadata op_name contains LOSS
    instr.metadata.op_name = "layer_26.block/mlp/loss_fn/reduce_sum"
    self.assertEqual(tracer.classify_context(instr), "LOSS")

    # Case 2: Metadata op_name contains NORM
    instr.metadata.op_name = "layer_26.block/mlp/pre_ffw_norm/reduce_sum"
    self.assertEqual(tracer.classify_context(instr), "NORM")

    # Case 3: Metadata op_name contains SOFTMAX
    instr.metadata.op_name = "layer_26.block/mlp/attention/softmax/reduce_sum"
    self.assertEqual(tracer.classify_context(instr), "SOFTMAX")

    # Case 4: Upstream exponential check
    hlo_exp = """
    HloModule test_module
    %add_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(%x, %y)
    }
    ENTRY %entry_computation {
      p0 = f32[100] parameter(0)
      exp1 = f32[100] exponential(p0)
      ROOT reduce1 = f32[] reduce(exp1, exp1), dimensions={0}, to_apply=%add_reducer
    }
    """
    proto_exp = _parse_hlo_text_to_proto(hlo_exp)
    tracer_exp = detect_unnecessary_convert_reduce_tool.HloModuleTracer(
        proto_exp
    )
    reduce_instr = next(
        i for i in tracer_exp.instructions.values() if i.opcode == "reduce"
    )
    self.assertEqual(tracer_exp.classify_context(reduce_instr), "SOFTMAX")

    # Case 5: GENERAL default
    hlo_gen = """
    HloModule test_module
    %add_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(%x, %y)
    }
    ENTRY %pooling_computation {
      p0 = f32[100] parameter(0)
      ROOT reduce1 = f32[] reduce(p0, p0), dimensions={0}, to_apply=%add_reducer
    }
    """
    proto_gen = _parse_hlo_text_to_proto(hlo_gen)
    tracer_gen = detect_unnecessary_convert_reduce_tool.HloModuleTracer(
        proto_gen
    )
    reduce_instr_gen = next(
        i for i in tracer_gen.instructions.values() if i.opcode == "reduce"
    )
    self.assertEqual(tracer_gen.classify_context(reduce_instr_gen), "GENERAL")

  def test_evaluate_optimization(self):
    # 1. TRAINING phase
    # General context gets low priority alert
    is_ineff, rec, warn, is_low = (
        detect_unnecessary_convert_reduce_tool._evaluate_optimization(
            "TRAINING", "GENERAL", 500
        )
    )
    self.assertTrue(is_ineff)
    self.assertTrue(is_low)
    self.assertIn("Keep intermediate reduction calculation", rec)
    self.assertIn("Low priority", warn)

    # Norm context in training gets silently skipped
    is_ineff, _, _, _ = (
        detect_unnecessary_convert_reduce_tool._evaluate_optimization(
            "TRAINING", "NORM", 500
        )
    )
    self.assertFalse(is_ineff)

    # 2. INFERENCE phase
    # General context in inference gets standard alert, no precision warning
    is_ineff, rec, warn, is_low = (
        detect_unnecessary_convert_reduce_tool._evaluate_optimization(
            "INFERENCE", "GENERAL", 2048
        )
    )
    self.assertTrue(is_ineff)
    self.assertFalse(is_low)
    self.assertIn("Keep intermediate reduction calculation", rec)
    self.assertEqual(warn, "")

    # Softmax context in inference: size <= 1024 gets no warning
    is_ineff, rec, warn, _ = (
        detect_unnecessary_convert_reduce_tool._evaluate_optimization(
            "INFERENCE", "SOFTMAX", 500
        )
    )
    self.assertTrue(is_ineff)
    self.assertIn("Keep intermediate reduction calculation", rec)
    self.assertEqual(warn, "")

    # Softmax context in inference: size > 1024 gets extra precision warning
    is_ineff, rec, warn, _ = (
        detect_unnecessary_convert_reduce_tool._evaluate_optimization(
            "INFERENCE", "SOFTMAX", 2048
        )
    )
    self.assertTrue(is_ineff)
    self.assertIn("Keep intermediate reduction calculation", rec)
    self.assertIn("Warning: The reduction size is large", warn)


if __name__ == "__main__":
  absltest.main()
