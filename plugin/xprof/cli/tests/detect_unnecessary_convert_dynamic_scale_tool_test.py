import json
import types
from unittest import mock

from absl.testing import absltest
from tensorflow.compiler.xla.python import xla_client  # pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.xla.service import hlo_pb2  # pylint: disable=g-direct-tensorflow-import
from xprof.cli.internal.oss import hlo_tools
from xprof.cli.tools import detect_unnecessary_convert_dynamic_scale_tool


def _parse_hlo_text_to_proto(hlo_text: str) -> hlo_pb2.HloModuleProto:
  """Compiles HLO text to C++ module, serializes, and deserializes to proto."""
  module = xla_client.hlo.hlo_module_from_text(hlo_text)
  serialized_bytes = module.as_serialized_hlo_module_proto()
  return hlo_pb2.HloModuleProto.FromString(serialized_bytes)


class DetectUnnecessaryConvertDynamicScaleToolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
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

  def test_build_module_indices(self):
    hlo_text = """
    HloModule test_module

    %fused_comp (p0.1: bf16[100]) -> f32[100] {
      %p0.1 = bf16[100] parameter(0)
      ROOT %convert.1 = f32[100] convert(%p0.1)
    }

    ENTRY %entry_computation {
      %p0 = bf16[100] parameter(0)
      ROOT %fusion.1 = f32[100] fusion(%p0), kind=kLoop, calls=%fused_comp
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_dynamic_scale_tool._HloModuleTracer(
        proto
    )
    self.assertLen(tracer.computations, 2)
    self.assertLen(tracer.instructions, 4)

    convert_instr = next(
        i for i in tracer.instructions.values() if i.opcode == "convert"
    )
    self.assertIn("convert.1", tracer.instructions_by_name)
    self.assertEqual(
        tracer.instructions_by_name["convert.1"].id, convert_instr.id
    )

  def test_trace_upstream_activation_direct_parameter(self):
    hlo_text = """
    HloModule test_module
    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)
      ROOT %convert.1 = f32[128,256] convert(%act)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_dynamic_scale_tool._HloModuleTracer(
        proto
    )
    convert_1 = tracer.instructions_by_name["convert.1"]
    param_act = tracer.instructions_by_name["act"]

    source_id, call_stack = tracer.trace_upstream_activation(
        convert_1.operand_ids[0]
    )
    self.assertEqual(source_id, param_act.id)
    self.assertEqual(call_stack, ())

  def test_trace_upstream_activation_intermediate_transparent_ops(self):
    hlo_text = """
    HloModule test_module
    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)
      %reshaped = bf16[128,256] reshape(%act)
      %transposed = bf16[256,128] transpose(%reshaped), dimensions={1,0}
      ROOT %convert.1 = f32[256,128] convert(%transposed)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_dynamic_scale_tool._HloModuleTracer(
        proto
    )
    convert_1 = tracer.instructions_by_name["convert.1"]
    param_act = tracer.instructions_by_name["act"]

    source_id, call_stack = tracer.trace_upstream_activation(
        convert_1.operand_ids[0]
    )
    self.assertEqual(source_id, param_act.id)
    self.assertEqual(call_stack, ())

  def test_trace_upstream_activation_fusion_parameter(self):
    hlo_text = """
    HloModule test_module
    %fused_scale (p0.1: bf16[128,256]) -> f32[128,256] {
      %p0.1 = bf16[128,256] parameter(0)
      %reshaped = bf16[128,256] reshape(%p0.1)
      ROOT %convert.1 = f32[128,256] convert(%reshaped)
    }
    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)
      ROOT %fusion.1 = f32[128,256] fusion(%act), kind=kLoop, calls=%fused_scale
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_dynamic_scale_tool._HloModuleTracer(
        proto
    )
    convert_1 = tracer.instructions_by_name["convert.1"]
    param_act = tracer.instructions_by_name["act"]

    source_id, call_stack = tracer.trace_upstream_activation(
        convert_1.operand_ids[0]
    )
    self.assertEqual(source_id, param_act.id)
    self.assertEqual(call_stack, ())

  def test_trace_branch1_scale_and_branch2_quantization_positive(self):
    hlo_text = """
    HloModule dynamic_scale_module

    %max_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %max = f32[] maximum(%x, %y)
    }

    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)
      %reshaped_act = bf16[128,256] reshape(%act)

      // Branch 1: Scale Calculation
      %convert.1 = f32[128,256] convert(%reshaped_act)
      %abs.1 = f32[128,256] abs(%convert.1)
      %c_inf = f32[] constant(-inf)
      %raw_scale = f32[128] reduce(%abs.1, %c_inf), dimensions={1}, to_apply=%max_reducer
      %scale_bcast = f32[128,256] broadcast(%raw_scale), dimensions={0}

      // Branch 2: Quantization Scaling
      %convert.2 = f32[128,256] convert(%act)
      %scaled_act = f32[128,256] divide(%convert.2, %scale_bcast)
      ROOT %convert.3 = f8e4m3fn[128,256] convert(%scaled_act)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_dynamic_scale_tool._HloModuleTracer(
        proto
    )
    convert_1 = tracer.instructions_by_name["convert.1"]
    convert_2 = tracer.instructions_by_name["convert.2"]
    raw_scale = tracer.instructions_by_name["raw_scale"]
    scale_bcast = tracer.instructions_by_name["scale_bcast"]
    scaled_act = tracer.instructions_by_name["scaled_act"]
    convert_3 = tracer.instructions_by_name["convert.3"]

    scale_node_id = tracer.trace_branch1_scale(convert_1.id)
    self.assertEqual(scale_node_id, raw_scale.id)
    assert scale_node_id is not None

    self.assertTrue(
        tracer.trace_upstream_to_node(scale_bcast.id, scale_node_id)
    )

    quant_res = tracer.trace_branch2_quantization(convert_2.id, scale_node_id)
    self.assertEqual(quant_res, (scaled_act.id, convert_3.id))

  def test_trace_branch1_scale_epsilon_clamping(self):
    hlo_text = """
    HloModule dynamic_scale_module

    %max_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %max = f32[] maximum(%x, %y)
    }

    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)

      // Branch 1: Scale Calculation with Epsilon Clamping
      %convert.1 = f32[128,256] convert(%act)
      %c_inf = f32[] constant(-inf)
      %raw_scale = f32[128] reduce(%convert.1, %c_inf), dimensions={1}, to_apply=%max_reducer
      %eps = f32[] constant(1e-7)
      %eps_bcast = f32[128] broadcast(%eps), dimensions={}
      %clamped_scale = f32[128] maximum(%raw_scale, %eps_bcast)
      %scale_bcast = f32[128,256] broadcast(%clamped_scale), dimensions={0}

      // Branch 2: Quantization Scaling
      %convert.2 = f32[128,256] convert(%act)
      %scaled_act = f32[128,256] divide(%convert.2, %scale_bcast)
      ROOT %convert.3 = f8e4m3fn[128,256] convert(%scaled_act)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_dynamic_scale_tool._HloModuleTracer(
        proto
    )
    convert_1 = tracer.instructions_by_name["convert.1"]
    convert_2 = tracer.instructions_by_name["convert.2"]
    raw_scale = tracer.instructions_by_name["raw_scale"]
    scale_bcast = tracer.instructions_by_name["scale_bcast"]
    scaled_act = tracer.instructions_by_name["scaled_act"]
    convert_3 = tracer.instructions_by_name["convert.3"]

    scale_node_id = tracer.trace_branch1_scale(convert_1.id)
    self.assertEqual(scale_node_id, raw_scale.id)
    assert scale_node_id is not None

    self.assertTrue(
        tracer.trace_upstream_to_node(scale_bcast.id, scale_node_id)
    )

    quant_res = tracer.trace_branch2_quantization(convert_2.id, scale_node_id)
    self.assertEqual(quant_res, (scaled_act.id, convert_3.id))

  def test_trace_branch2_quantization_invalid_downcast_type(self):
    hlo_text = """
    HloModule dynamic_scale_module

    %max_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %max = f32[] maximum(%x, %y)
    }

    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)

      %convert.1 = f32[128,256] convert(%act)
      %c_inf = f32[] constant(-inf)
      %raw_scale = f32[128] reduce(%convert.1, %c_inf), dimensions={1}, to_apply=%max_reducer
      %scale_bcast = f32[128,256] broadcast(%raw_scale), dimensions={0}

      %convert.2 = f32[128,256] convert(%act)
      %scaled_act = f32[128,256] divide(%convert.2, %scale_bcast)
      ROOT %convert.3 = f32[128,256] convert(%scaled_act)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_dynamic_scale_tool._HloModuleTracer(
        proto
    )
    convert_1 = tracer.instructions_by_name["convert.1"]
    convert_2 = tracer.instructions_by_name["convert.2"]

    scale_node_id = tracer.trace_branch1_scale(convert_1.id)
    self.assertIsNotNone(scale_node_id)
    assert scale_node_id is not None

    quant_res = tracer.trace_branch2_quantization(convert_2.id, scale_node_id)
    self.assertIsNone(quant_res)

  def test_trace_branch2_quantization_unconnected_scale(self):
    hlo_text = """
    HloModule dynamic_scale_module

    %max_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %max = f32[] maximum(%x, %y)
    }

    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)
      %other = f32[128,256] parameter(1)

      %convert.1 = f32[128,256] convert(%act)
      %c_inf = f32[] constant(-inf)
      %raw_scale = f32[128] reduce(%convert.1, %c_inf), dimensions={1}, to_apply=%max_reducer

      %convert.2 = f32[128,256] convert(%act)
      %scaled_act = f32[128,256] divide(%convert.2, %other)
      ROOT %convert.3 = f8e4m3fn[128,256] convert(%scaled_act)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_dynamic_scale_tool._HloModuleTracer(
        proto
    )
    convert_1 = tracer.instructions_by_name["convert.1"]
    convert_2 = tracer.instructions_by_name["convert.2"]
    other = tracer.instructions_by_name["other"]

    scale_node_id = tracer.trace_branch1_scale(convert_1.id)
    self.assertIsNotNone(scale_node_id)
    assert scale_node_id is not None

    self.assertFalse(tracer.trace_upstream_to_node(other.id, scale_node_id))

    quant_res = tracer.trace_branch2_quantization(convert_2.id, scale_node_id)
    self.assertIsNone(quant_res)

  def test_analyze_hlo_module_clamped_fp8_downcast(self):
    hlo_text = """
    HloModule dynamic_scale_clamped_module

    %max_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %max = f32[] maximum(%x, %y)
    }

    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)
      %reshaped_act = bf16[128,256] reshape(%act)

      // Branch 1: Scale Calculation
      %convert.1 = f32[128,256] convert(%reshaped_act)
      %abs.1 = f32[128,256] abs(%convert.1)
      %c_inf = f32[] constant(-inf)
      %raw_scale = f32[128] reduce(%abs.1, %c_inf), dimensions={1}, to_apply=%max_reducer
      %scale_bcast = f32[128,256] broadcast(%raw_scale), dimensions={0}

      // Branch 2: Quantization Scaling with Clamping
      %convert.2 = f32[128,256] convert(%act)
      %scaled_act = f32[128,256] divide(%convert.2, %scale_bcast)
      %c_min = f32[] constant(-448.0)
      %min_bcast = f32[128,256] broadcast(%c_min), dimensions={}
      %c_max = f32[] constant(448.0)
      %max_bcast = f32[128,256] broadcast(%c_max), dimensions={}
      %clamped_act = f32[128,256] clamp(%min_bcast, %scaled_act, %max_bcast)
      ROOT %convert.3 = f8e4m3fn[128,256] convert(%clamped_act)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    bottlenecks = (
        detect_unnecessary_convert_dynamic_scale_tool.analyze_hlo_module(proto)
    )
    self.assertLen(bottlenecks, 1)
    self.assertEqual(bottlenecks[0]["instruction"], "convert.1")

  def test_analyze_hlo_module_asymmetric_top_ops(self):
    hlo_text = """
    HloModule dynamic_scale_module

    %max_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %max = f32[] maximum(%x, %y)
    }

    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)

      %convert.1 = f32[128,256] convert(%act)
      %c_inf = f32[] constant(-inf)
      %raw_scale = f32[128] reduce(%convert.1, %c_inf), dimensions={1}, to_apply=%max_reducer
      %scale_bcast = f32[128,256] broadcast(%raw_scale), dimensions={0}

      %convert.2 = f32[128,256] convert(%act)
      %scaled_act = f32[128,256] divide(%convert.2, %scale_bcast)
      ROOT %convert.3 = f8e4m3fn[128,256] convert(%scaled_act)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    bottlenecks = (
        detect_unnecessary_convert_dynamic_scale_tool.analyze_hlo_module(
            proto, seed_convert_names={"convert.1"}
        )
    )
    self.assertLen(bottlenecks, 1)

    bottlenecks_c2 = (
        detect_unnecessary_convert_dynamic_scale_tool.analyze_hlo_module(
            proto, seed_convert_names={"convert.2"}
        )
    )
    self.assertLen(bottlenecks_c2, 1)

    no_bottlenecks = (
        detect_unnecessary_convert_dynamic_scale_tool.analyze_hlo_module(
            proto, seed_convert_names={"other_op"}
        )
    )
    self.assertEmpty(no_bottlenecks)

  def test_same_convert_guard_no_finding(self):
    hlo_text = """
    HloModule same_convert_module

    %max_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %max = f32[] maximum(%x, %y)
    }

    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)
      %convert.1 = f32[128,256] convert(%act)

      %c_inf = f32[] constant(-inf)
      %raw_scale = f32[128] reduce(%convert.1, %c_inf), dimensions={1}, to_apply=%max_reducer
      %scale_bcast = f32[128,256] broadcast(%raw_scale), dimensions={0}

      %scaled_act = f32[128,256] divide(%convert.1, %scale_bcast)
      ROOT %convert.3 = f8e4m3fn[128,256] convert(%scaled_act)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    bottlenecks = (
        detect_unnecessary_convert_dynamic_scale_tool.analyze_hlo_module(proto)
    )
    self.assertEmpty(bottlenecks)

  def test_cli_entry_point_with_mocked_backend(self):
    hlo_text = """
    HloModule dynamic_scale_module

    %max_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %max = f32[] maximum(%x, %y)
    }

    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)

      %convert.1 = f32[128,256] convert(%act)
      %c_inf = f32[] constant(-inf)
      %raw_scale = f32[128] reduce(%convert.1, %c_inf), dimensions={1}, to_apply=%max_reducer
      %scale_bcast = f32[128,256] broadcast(%raw_scale), dimensions={0}

      %convert.2 = f32[128,256] convert(%act)
      %scaled_act = f32[128,256] divide(%convert.2, %scale_bcast)
      ROOT %convert.3 = f8e4m3fn[128,256] convert(%scaled_act)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)

    mock_proto_wrapper = types.SimpleNamespace(hlo_module=proto)
    mock_debug_info = types.SimpleNamespace(
        hlo_proto=[mock_proto_wrapper],
        program_id=[12345],
    )

    top_ops_payload = {
        "top_by_time": [{
            "name": "by_program/dynamic_scale_module/convert.1",
            "total_self_time_ms": 15.5,
        }]
    }

    def mock_get_top_ops(session_id: str, limit: int = 50) -> str:
      del session_id, limit  # Unused; signature mirrors the real API.
      return json.dumps(top_ops_payload)

    with mock.patch.object(
        hlo_tools, "_fetch_debug_info", return_value=mock_debug_info
    ):
      result_json = detect_unnecessary_convert_dynamic_scale_tool.detect_unnecessary_convert_dynamic_scale(
          "test_session", get_top_hlo_ops_fn=mock_get_top_ops, limit=50
      )

    res = json.loads(result_json)
    self.assertTrue(res["bottlenecks_found"])
    self.assertLen(res["inefficient_ops"], 1)
    self.assertEqual(res["inefficient_ops"][0]["instruction"], "convert.1")
    self.assertEqual(res["inefficient_ops"][0]["total_self_time_ms"], 15.5)

  def test_int8_downcast_target_rejected(self):
    """FP8-only scope: an INT8 (s8) quantization target must NOT be flagged."""
    hlo_text = """
    HloModule int8_target_module

    %max_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %max = f32[] maximum(%x, %y)
    }

    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)
      %convert.1 = f32[128,256] convert(%act)
      %c_inf = f32[] constant(-inf)
      %raw_scale = f32[128] reduce(%convert.1, %c_inf), dimensions={1}, to_apply=%max_reducer
      %scale_bcast = f32[128,256] broadcast(%raw_scale), dimensions={0}
      %convert.2 = f32[128,256] convert(%act)
      %scaled_act = f32[128,256] divide(%convert.2, %scale_bcast)
      ROOT %convert.3 = s8[128,256] convert(%scaled_act)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    bottlenecks = (
        detect_unnecessary_convert_dynamic_scale_tool.analyze_hlo_module(proto)
    )
    self.assertEmpty(bottlenecks)

  def test_bf16_scale_path_rejected_g2(self):
    """G2 purity: if the scale path drops to bf16 it is not "useless f32"."""
    hlo_text = """
    HloModule bf16_scale_module

    %max_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %max = f32[] maximum(%x, %y)
    }

    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)
      %convert.1 = f32[128,256] convert(%act)
      %c_inf = f32[] constant(-inf)
      %raw_scale = f32[128] reduce(%convert.1, %c_inf), dimensions={1}, to_apply=%max_reducer
      %scale_bf16 = bf16[128] convert(%raw_scale)
      %scale_f32 = f32[128] convert(%scale_bf16)
      %scale_bcast = f32[128,256] broadcast(%scale_f32), dimensions={0}
      %convert.2 = f32[128,256] convert(%act)
      %scaled_act = f32[128,256] divide(%convert.2, %scale_bcast)
      ROOT %convert.3 = f8e4m3fn[128,256] convert(%scaled_act)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_dynamic_scale_tool._HloModuleTracer(
        proto
    )
    convert_1 = tracer.instructions_by_name["convert.1"]
    scale_bcast = tracer.instructions_by_name["scale_bcast"]

    scale_node_id = tracer.trace_branch1_scale(convert_1.id)
    self.assertIsNotNone(scale_node_id)
    assert scale_node_id is not None

    # The bf16 hop on the scale path breaks f32 purity (G2), so tracing from the
    # scaling op's scale operand must NOT reach the reduction node.
    self.assertFalse(
        tracer.trace_upstream_to_node(scale_bcast.id, scale_node_id)
    )

    bottlenecks = (
        detect_unnecessary_convert_dynamic_scale_tool.analyze_hlo_module(proto)
    )
    self.assertEmpty(bottlenecks)

  def test_scale_via_multiply_constant_positive(self):
    """Generality: scale applied via multiply-by-constant (1/qscale) is detected.

    This used to be missed because `multiply` was not in the traversal
    allowlist; the f32-purity (G2) logic now traverses it.
    """
    hlo_text = """
    HloModule multiply_scale_module

    %max_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %max = f32[] maximum(%x, %y)
    }

    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)
      %convert.1 = f32[128,256] convert(%act)
      %abs.1 = f32[128,256] abs(%convert.1)
      %c_inf = f32[] constant(-inf)
      %raw_scale = f32[128] reduce(%abs.1, %c_inf), dimensions={1}, to_apply=%max_reducer
      %c_recip = f32[] constant(0.00223214)
      %recip_bcast = f32[128] broadcast(%c_recip), dimensions={}
      %scale = f32[128] multiply(%raw_scale, %recip_bcast)
      %scale_bcast = f32[128,256] broadcast(%scale), dimensions={0}
      %convert.2 = f32[128,256] convert(%act)
      %scaled_act = f32[128,256] divide(%convert.2, %scale_bcast)
      ROOT %convert.3 = f8e4m3fn[128,256] convert(%scaled_act)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_dynamic_scale_tool._HloModuleTracer(
        proto
    )
    convert_1 = tracer.instructions_by_name["convert.1"]
    raw_scale = tracer.instructions_by_name["raw_scale"]
    scale_bcast = tracer.instructions_by_name["scale_bcast"]

    scale_node_id = tracer.trace_branch1_scale(convert_1.id)
    self.assertEqual(scale_node_id, raw_scale.id)
    assert scale_node_id is not None

    self.assertTrue(
        tracer.trace_upstream_to_node(scale_bcast.id, scale_node_id)
    )

    bottlenecks = (
        detect_unnecessary_convert_dynamic_scale_tool.analyze_hlo_module(proto)
    )
    self.assertLen(bottlenecks, 1)
    self.assertEqual(bottlenecks[0]["instruction"], "convert.1")

  def test_reduce_body_agnostic_product_reducer(self):
    """Reduce-body relaxation: any reduce family op is node S, regardless of body."""
    hlo_text = """
    HloModule product_reduce_module

    %prod_reducer (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %prod = f32[] multiply(%x, %y)
    }

    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)
      %convert.1 = f32[128,256] convert(%act)
      %c_one = f32[] constant(1.0)
      %raw_scale = f32[128] reduce(%convert.1, %c_one), dimensions={1}, to_apply=%prod_reducer
      %scale_bcast = f32[128,256] broadcast(%raw_scale), dimensions={0}
      %convert.2 = f32[128,256] convert(%act)
      %scaled_act = f32[128,256] divide(%convert.2, %scale_bcast)
      ROOT %convert.3 = f8e4m3fn[128,256] convert(%scaled_act)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    tracer = detect_unnecessary_convert_dynamic_scale_tool._HloModuleTracer(
        proto
    )
    convert_1 = tracer.instructions_by_name["convert.1"]
    raw_scale = tracer.instructions_by_name["raw_scale"]

    # A product-reduction body would have been rejected by the old max/min/add
    # body check; the relaxed logic treats any reduce as the Scale Node S.
    scale_node_id = tracer.trace_branch1_scale(convert_1.id)
    self.assertEqual(scale_node_id, raw_scale.id)
    assert scale_node_id is not None

    bottlenecks = (
        detect_unnecessary_convert_dynamic_scale_tool.analyze_hlo_module(proto)
    )
    self.assertLen(bottlenecks, 1)

  def test_cross_fusion_multi_output_tuple_positive(self):
    """Real-world topology: scale/activation paths split across 3 fusions.

    Faithful replica of the jit_prefill fusion.8915/8916/8917 example:
      * fusion.8915 upcasts the bf16 source X to f32 (convert_1), computes the
        reduce (Node S), and outputs a TUPLE of (f32 scale-source, bf16 X).
      * fusion.8916 consumes the f32 tuple element and does further f32 scale
        computation.
      * fusion.8917 consumes the bf16 tuple element (convert_2), consumes the
        f32 scale from fusion.8916, applies the scaling divide, and converts to
        FP8.
    The convergence (G1) / f32-purity (G2) check must cross the 8917->8916->8915
    fusion boundaries and correctly walk through the multi-output tuple (pruning
    the bf16 element, following the f32 element back to Node S).
    """
    hlo_text = """
    HloModule jit_prefill_multi_fusion

    %max_reducer (mx: f32[], my: f32[]) -> f32[] {
      %mx = f32[] parameter(0)
      %my = f32[] parameter(1)
      ROOT %mmax = f32[] maximum(%mx, %my)
    }

    %comp_8915 (a_p: bf16[128,256]) -> (f32[128], bf16[128,256]) {
      %a_p = bf16[128,256] parameter(0)
      %scale_convert = f32[128,256] convert(%a_p)
      %a_abs = f32[128,256] abs(%scale_convert)
      %a_cinf = f32[] constant(-inf)
      %a_reduce = f32[128] reduce(%a_abs, %a_cinf), dimensions={1}, to_apply=%max_reducer
      ROOT %a_tup = (f32[128], bf16[128,256]) tuple(%a_reduce, %a_p)
    }

    %comp_8916 (b_p: f32[128]) -> f32[128] {
      %b_p = f32[128] parameter(0)
      ROOT %b_scale = f32[128] rsqrt(%b_p)
    }

    %comp_8917 (c_p0: bf16[128,256], c_p1: f32[128]) -> f8e4m3fn[128,256] {
      %c_p0 = bf16[128,256] parameter(0)
      %c_p1 = f32[128] parameter(1)
      %act_convert = f32[128,256] convert(%c_p0)
      %c_bcast = f32[128,256] broadcast(%c_p1), dimensions={0}
      %c_div = f32[128,256] divide(%act_convert, %c_bcast)
      ROOT %c_q = f8e4m3fn[128,256] convert(%c_div)
    }

    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)
      %fus8915 = (f32[128], bf16[128,256]) fusion(%act), kind=kLoop, calls=%comp_8915
      %gte0 = f32[128] get-tuple-element(%fus8915), index=0
      %gte1 = bf16[128,256] get-tuple-element(%fus8915), index=1
      %fus8916 = f32[128] fusion(%gte0), kind=kLoop, calls=%comp_8916
      ROOT %fus8917 = f8e4m3fn[128,256] fusion(%gte1, %fus8916), kind=kLoop, calls=%comp_8917
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    bottlenecks = (
        detect_unnecessary_convert_dynamic_scale_tool.analyze_hlo_module(proto)
    )
    self.assertLen(bottlenecks, 1)
    self.assertEqual(bottlenecks[0]["instruction"], "scale_convert")

  def test_fully_split_across_fusions_positive(self):
    """Every key op lives in its own fusion; detection must still match.

    Layout (each op isolated in a distinct single-output fusion):
      compA: convert_1 (bf16->f32)          -> fusA
      compB: abs + reduce (Node S)          -> fusB
      compC: broadcast of the scale         -> fusC
      compD: convert_2 (bf16->f32)          -> fusD
      compE: divide + FP8 convert           -> fusE
    Branch-1 (convert_1 -> reduce) crosses fusA->fusB; the scale broadcast is in
    fusC; branch-2 (convert_2 -> divide -> FP8) crosses fusD->fusE; and the G1
    convergence walks fusE->fusC->fusB. This only passes if every downstream
    tracer can both exit and enter fusions.
    """
    hlo_text = """
    HloModule fully_split

    %max_reducer (mx: f32[], my: f32[]) -> f32[] {
      %mx = f32[] parameter(0)
      %my = f32[] parameter(1)
      ROOT %mmax = f32[] maximum(%mx, %my)
    }

    %compA (a_p: bf16[128,256]) -> f32[128,256] {
      %a_p = bf16[128,256] parameter(0)
      ROOT %convert_1 = f32[128,256] convert(%a_p)
    }

    %compB (b_p: f32[128,256]) -> f32[128] {
      %b_p = f32[128,256] parameter(0)
      %b_abs = f32[128,256] abs(%b_p)
      %b_cinf = f32[] constant(-inf)
      ROOT %b_reduce = f32[128] reduce(%b_abs, %b_cinf), dimensions={1}, to_apply=%max_reducer
    }

    %compC (c_p: f32[128]) -> f32[128,256] {
      %c_p = f32[128] parameter(0)
      ROOT %c_bcast = f32[128,256] broadcast(%c_p), dimensions={0}
    }

    %compD (d_p: bf16[128,256]) -> f32[128,256] {
      %d_p = bf16[128,256] parameter(0)
      ROOT %convert_2 = f32[128,256] convert(%d_p)
    }

    %compE (e_p0: f32[128,256], e_p1: f32[128,256]) -> f8e4m3fn[128,256] {
      %e_p0 = f32[128,256] parameter(0)
      %e_p1 = f32[128,256] parameter(1)
      %e_div = f32[128,256] divide(%e_p0, %e_p1)
      ROOT %e_q = f8e4m3fn[128,256] convert(%e_div)
    }

    ENTRY %entry {
      %act = bf16[128,256] parameter(0)
      %fusA = f32[128,256] fusion(%act), kind=kLoop, calls=%compA
      %fusB = f32[128] fusion(%fusA), kind=kLoop, calls=%compB
      %fusC = f32[128,256] fusion(%fusB), kind=kLoop, calls=%compC
      %fusD = f32[128,256] fusion(%act), kind=kLoop, calls=%compD
      ROOT %fusE = f8e4m3fn[128,256] fusion(%fusD, %fusC), kind=kLoop, calls=%compE
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    bottlenecks = (
        detect_unnecessary_convert_dynamic_scale_tool.analyze_hlo_module(proto)
    )
    self.assertLen(bottlenecks, 1)
    self.assertEqual(bottlenecks[0]["instruction"], "convert_1")

  def test_scale_fusion_multi_output_bf16_sibling_positive(self):
    """Faithful jit_prefill replica where the scale fusion (8916) emits a TUPLE.

    Distinct from test_cross_fusion_multi_output_tuple_positive: here the scale
    fusion (analog of fusion.8916) is itself multi-output, emitting BOTH a bf16
    downcast of the f32 scale (for later dequant) and the f32 scale that feeds
    the quantization divide. The G1/G2 upstream walk must follow the f32 tuple
    element back to Node S while pruning the sibling bf16 element, so that
    dequant bf16 downcast must NOT break f32-purity (G2).

    Topology (mirrors the real fusion.8915/8916/8917):
      * comp_8915: source X = bf16 multiply; convert_1 (scale_convert) ->
        bitcast -> abs -> reduce over {0,2} (Node S); ROOT tuple(reduce, X).
      * comp_8916: multiply the f32 scale-source by a constant, downcast the
        result to bf16 (the dequant sibling), ROOT tuple(bf16, f32).
      * comp_8917: convert_2 (act_convert) on bf16 X; bitcast; divide by the
        broadcast f32 scale; convert to FP8; bitcast + copy tail.
    """
    hlo_text = """
    HloModule jit_prefill_scale_tuple

    %max_reducer (mx: f32[], my: f32[]) -> f32[] {
      %mx = f32[] parameter(0)
      %my = f32[] parameter(1)
      ROOT %mmax = f32[] maximum(%mx, %my)
    }

    %comp_8915 (a_p: bf16[128,256]) -> (f32[128], bf16[128,256]) {
      %a_p = bf16[128,256] parameter(0)
      %a_src = bf16[128,256] multiply(%a_p, %a_p)
      %scale_convert = f32[128,256] convert(%a_src)
      %a_bitcast = f32[1,128,256] bitcast(%scale_convert)
      %a_abs = f32[1,128,256] abs(%a_bitcast)
      %a_cinf = f32[] constant(-inf)
      %a_reduce = f32[128] reduce(%a_abs, %a_cinf), dimensions={0,2}, to_apply=%max_reducer
      ROOT %a_tup = (f32[128], bf16[128,256]) tuple(%a_reduce, %a_src)
    }

    %comp_8916 (b_p: f32[128]) -> (bf16[128], f32[128]) {
      %b_p = f32[128] parameter(0)
      %b_c = f32[] constant(0.00223214296)
      %b_bcast = f32[128] broadcast(%b_c), dimensions={}
      %b_scale = f32[128] multiply(%b_p, %b_bcast)
      %b_bf16 = bf16[128] convert(%b_scale)
      ROOT %b_tup = (bf16[128], f32[128]) tuple(%b_bf16, %b_scale)
    }

    %comp_8917 (c_p0: bf16[128,256], c_p1: f32[128]) -> f8e4m3fn[128,256] {
      %c_p0 = bf16[128,256] parameter(0)
      %c_p1 = f32[128] parameter(1)
      %act_convert = f32[128,256] convert(%c_p0)
      %c_bitcast = f32[1,128,256] bitcast(%act_convert)
      %c_bcast = f32[1,128,256] broadcast(%c_p1), dimensions={1}
      %c_div = f32[1,128,256] divide(%c_bitcast, %c_bcast)
      %c_q = f8e4m3fn[1,128,256] convert(%c_div)
      %c_bitcast2 = f8e4m3fn[128,256] bitcast(%c_q)
      ROOT %c_copy = f8e4m3fn[128,256] copy(%c_bitcast2)
    }

    ENTRY %entry_computation {
      %act = bf16[128,256] parameter(0)
      %fus8915 = (f32[128], bf16[128,256]) fusion(%act), kind=kLoop, calls=%comp_8915
      %gte0 = f32[128] get-tuple-element(%fus8915), index=0
      %gte1 = bf16[128,256] get-tuple-element(%fus8915), index=1
      %fus8916 = (bf16[128], f32[128]) fusion(%gte0), kind=kLoop, calls=%comp_8916
      %gte_scale = f32[128] get-tuple-element(%fus8916), index=1
      ROOT %fus8917 = f8e4m3fn[128,256] fusion(%gte1, %gte_scale), kind=kLoop, calls=%comp_8917
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    bottlenecks = (
        detect_unnecessary_convert_dynamic_scale_tool.analyze_hlo_module(proto)
    )
    self.assertLen(bottlenecks, 1)
    # convert_1 (scale branch) is the primary instruction; convert_2 (quant
    # branch) is surfaced as a first-class field. Both are the unnecessary
    # bf16->f32 upcasts we want to flag, and they live in different fusions.
    self.assertEqual(bottlenecks[0]["instruction"], "scale_convert")
    self.assertEqual(bottlenecks[0]["fusion_name"], "fus8915")
    self.assertEqual(bottlenecks[0]["quant_instruction"], "act_convert")
    self.assertEqual(bottlenecks[0]["quant_fusion_name"], "fus8917")

  def test_trace_upstream_activation_cyclic_fusion_safety(self):
    """Verifies graph traversal terminates safely on cyclic fusion references (CWE-835)."""
    hlo_text = """
    HloModule cyclic_module

    %comp_a (p: bf16[128]) -> bf16[128] {
      %p = bf16[128] parameter(0)
      ROOT %out = bf16[128] copy(%p)
    }

    ENTRY %entry_computation {
      %x = bf16[128] parameter(0)
      %fus = bf16[128] fusion(%x), kind=kLoop, calls=%comp_a
      ROOT %conv = f32[128] convert(%fus)
    }
    """
    proto = _parse_hlo_text_to_proto(hlo_text)
    # Simulate a malformed/cyclic HLO proto by making comp_a call itself:
    comp_a = proto.computations[0]
    comp_a.instructions[1].opcode = "fusion"
    comp_a.instructions[1].called_computation_ids.append(comp_a.id)

    tracer = detect_unnecessary_convert_dynamic_scale_tool._HloModuleTracer(
        proto
    )
    x = tracer.instructions_by_name["x"]
    conv = tracer.instructions_by_name["conv"]
    source_id, call_stack = tracer.trace_upstream_activation(
        conv.operand_ids[0]
    )
    # Should terminate safely and return without infinite recursion or hanging.
    self.assertEqual(source_id, x.id)
    self.assertEqual(call_stack, ())


if __name__ == "__main__":
  absltest.main()
