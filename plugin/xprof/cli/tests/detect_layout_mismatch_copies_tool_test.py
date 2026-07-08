import collections
import json
from unittest import mock

from xprof.cli.internal.oss import hlo_tools
from xprof.cli.tools import detect_layout_mismatch_copies_tool
from absl.testing import parameterized
from tensorflow.compiler.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import


def _get_top_hlo_ops_spec(session_id: str, limit: int = 100) -> str:
  del session_id, limit
  return ""


class DetectLayoutMismatchCopiesToolTest(parameterized.TestCase):

  @mock.patch.object(hlo_tools, "_fetch_debug_info", autospec=True)
  def test_detect_no_bottlenecks(self, mock_fetch):
    # Create empty hlo module
    debug_info = hlo_tools.hlo_proto_dump_pb2.DebugInfoCollection()
    hlo_proto = debug_info.hlo_proto.add()
    comp = hlo_proto.hlo_module.computations.add(name="main", id=1)

    # Just a standalone copy that does not connect to any compute op
    comp.instructions.add(id=10, name="copy_op", opcode="copy")

    mock_fetch.return_value = debug_info

    mock_top_ops_fn = mock.create_autospec(_get_top_hlo_ops_spec, spec_set=True)
    mock_top_ops_fn.return_value = "{}"

    result_json = (
        detect_layout_mismatch_copies_tool.detect_layout_mismatch_copies(
            "session_123", get_top_hlo_ops_fn=mock_top_ops_fn
        )
    )
    result = json.loads(result_json)

    self.assertFalse(result["bottlenecks_found"])
    self.assertEqual(
        result["message"], "No layout mismatch copy bottlenecks detected."
    )
    self.assertEmpty(result["inefficient_ops"])

  @mock.patch.object(hlo_tools, "_fetch_debug_info", autospec=True)
  def test_detect_sandwiched_copy_with_layout_mismatch(self, mock_fetch):
    debug_info = hlo_tools.hlo_proto_dump_pb2.DebugInfoCollection()
    hlo_proto = debug_info.hlo_proto.add()
    comp = hlo_proto.hlo_module.computations.add(name="main", id=1)

    # 1. Upstream compute op (dot)
    comp.instructions.add(
        id=100,
        name="dot_op",
        opcode="dot",
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.F32,
            dimensions=[128, 256],
            layout=xla_data_pb2.LayoutProto(minor_to_major=[1, 0]),
        ),
    )  # Minor dimension size 256 (optimal)

    # 2. Sandwiched copy op (operand is dot_op)
    comp.instructions.add(
        id=200,
        name="copy_op",
        opcode="copy",
        operand_ids=[100],
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.F32,
            dimensions=[128, 256],
            layout=xla_data_pb2.LayoutProto(minor_to_major=[0, 1]),
        ),
    )  # Layout mismatch! Target minor size 128 (optimal)

    # 3. Downstream compute op (reduce)
    comp.instructions.add(
        id=300, name="reduce_op", opcode="reduce", operand_ids=[200]
    )

    mock_fetch.return_value = debug_info

    # Mock top HLO ops function
    mock_top_ops_fn = mock.create_autospec(_get_top_hlo_ops_spec, spec_set=True)
    mock_top_ops_fn.return_value = json.dumps({
        "top_by_bytes_accessed": [{
            "name": "main/copy_op",
            "category": "Copy",
            "total_self_time_ms": 15.5,
            "bytes_accessed": 102400,
        }]
    })

    result_json = (
        detect_layout_mismatch_copies_tool.detect_layout_mismatch_copies(
            "session_123", get_top_hlo_ops_fn=mock_top_ops_fn
        )
    )
    result = json.loads(result_json)

    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)

    op = result["inefficient_ops"][0]
    # Single assertion to verify all performance dictionary values (readability)
    self.assertEqual(
        {
            "instruction_name": op["instruction_name"],
            "source_shape": op["source_shape"],
            "target_shape": op["target_shape"],
            "layout_mismatch": op["layout_mismatch"],
            "source_minor_dim_optimal": op["source_minor_dim_optimal"],
            "target_minor_dim_optimal": op["target_minor_dim_optimal"],
            "total_self_time_ms": op["total_self_time_ms"],
            "bytes_accessed": op["bytes_accessed"],
        },
        {
            "instruction_name": "copy_op",
            "source_shape": "f32[128, 256]{1,0}",
            "target_shape": "f32[128, 256]{0,1}",
            "layout_mismatch": True,
            "source_minor_dim_optimal": True,
            "target_minor_dim_optimal": True,
            "total_self_time_ms": 15.5,
            "bytes_accessed": 102400,
        },
    )

    self.assertIn("Layout mismatch detected!", op["recommendation"])
    self.assertIn("sandwiched between upstream producers", op["recommendation"])

  @mock.patch.object(hlo_tools, "_fetch_debug_info", autospec=True)
  def test_detect_non_optimal_minor_dimensions(self, mock_fetch):
    debug_info = hlo_tools.hlo_proto_dump_pb2.DebugInfoCollection()
    hlo_proto = debug_info.hlo_proto.add()
    comp = hlo_proto.hlo_module.computations.add(name="main", id=1)

    # 1. Upstream compute op (dot)
    comp.instructions.add(
        id=100,
        name="dot_op",
        opcode="dot",
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.F32,
            dimensions=[128, 50],
            layout=xla_data_pb2.LayoutProto(minor_to_major=[1, 0]),
        ),
    )  # Minor dim size 50 (not multiple of 128!)

    # 2. Sandwiched copy op (operand is dot_op)
    comp.instructions.add(
        id=200,
        name="copy_op",
        opcode="copy",
        operand_ids=[100],
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.F32,
            dimensions=[128, 50],
            layout=xla_data_pb2.LayoutProto(minor_to_major=[1, 0]),
        ),
    )  # Matching layout structurally

    # 3. Downstream compute op (reduce)
    comp.instructions.add(
        id=300, name="reduce_op", opcode="reduce", operand_ids=[200]
    )

    mock_fetch.return_value = debug_info

    mock_top_ops_fn = mock.create_autospec(_get_top_hlo_ops_spec, spec_set=True)
    mock_top_ops_fn.return_value = "{}"

    result_json = (
        detect_layout_mismatch_copies_tool.detect_layout_mismatch_copies(
            "session_123", get_top_hlo_ops_fn=mock_top_ops_fn
        )
    )
    result = json.loads(result_json)

    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)

    op = result["inefficient_ops"][0]
    self.assertEqual(op["instruction_name"], "copy_op")
    self.assertFalse(op["layout_mismatch"])
    self.assertFalse(op["source_minor_dim_optimal"])
    self.assertFalse(op["target_minor_dim_optimal"])
    self.assertIn(
        "Non-optimal dimension lane sizes found for TPU", op["recommendation"]
    )
    self.assertIn("expected multiple of 128", op["recommendation"])

  @mock.patch.object(hlo_tools, "_fetch_debug_info", autospec=True)
  def test_detect_custom_call_compute_op(self, mock_fetch):
    debug_info = hlo_tools.hlo_proto_dump_pb2.DebugInfoCollection()
    hlo_proto = debug_info.hlo_proto.add()
    comp = hlo_proto.hlo_module.computations.add(name="main", id=1)

    # 1. Upstream compute custom-call (update_slice)
    comp.instructions.add(
        id=100,
        name="update_slice.1",
        opcode="custom-call",
        custom_call_target="update_slice",
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.F32,
            dimensions=[128, 256],
            layout=xla_data_pb2.LayoutProto(minor_to_major=[1, 0]),
        ),
    )

    # 2. Sandwiched copy op
    comp.instructions.add(
        id=200,
        name="copy_op",
        opcode="copy",
        operand_ids=[100],
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.F32,
            dimensions=[128, 256],
            layout=xla_data_pb2.LayoutProto(minor_to_major=[0, 1]),
        ),
    )

    # 3. Downstream compute custom-call (update_kv_cache)
    comp.instructions.add(
        id=300,
        name="update_kv_cache.1",
        opcode="custom-call",
        custom_call_target="update_kv_cache",
        operand_ids=[200],
    )

    mock_fetch.return_value = debug_info

    mock_top_ops_fn = mock.create_autospec(_get_top_hlo_ops_spec, spec_set=True)
    mock_top_ops_fn.return_value = "{}"

    result_json = (
        detect_layout_mismatch_copies_tool.detect_layout_mismatch_copies(
            "session_123", get_top_hlo_ops_fn=mock_top_ops_fn
        )
    )
    result = json.loads(result_json)

    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)
    op = result["inefficient_ops"][0]
    self.assertEqual(op["instruction_name"], "copy_op")
    self.assertTrue(op["layout_mismatch"])

  @mock.patch.object(hlo_tools, "_fetch_debug_info", autospec=True)
  def test_detect_custom_call_non_compute_op(self, mock_fetch):
    debug_info = hlo_tools.hlo_proto_dump_pb2.DebugInfoCollection()
    hlo_proto = debug_info.hlo_proto.add()
    comp = hlo_proto.hlo_module.computations.add(name="main", id=1)

    # 1. Upstream non-compute custom-call (Sharding)
    comp.instructions.add(
        id=100,
        name="sharding_op",
        opcode="custom-call",
        custom_call_target="Sharding",
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.F32,
            dimensions=[128, 256],
            layout=xla_data_pb2.LayoutProto(minor_to_major=[1, 0]),
        ),
    )

    # 2. Sandwiched copy op
    comp.instructions.add(
        id=200,
        name="copy_op",
        opcode="copy",
        operand_ids=[100],
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.F32,
            dimensions=[128, 256],
            layout=xla_data_pb2.LayoutProto(minor_to_major=[0, 1]),
        ),
    )

    # 3. Downstream non-compute custom-call (AllocateBuffer)
    comp.instructions.add(
        id=300,
        name="allocate_op",
        opcode="custom-call",
        custom_call_target="AllocateBuffer",
        operand_ids=[200],
    )

    mock_fetch.return_value = debug_info

    mock_top_ops_fn = mock.create_autospec(_get_top_hlo_ops_spec, spec_set=True)
    mock_top_ops_fn.return_value = "{}"

    result_json = (
        detect_layout_mismatch_copies_tool.detect_layout_mismatch_copies(
            "session_123", get_top_hlo_ops_fn=mock_top_ops_fn
        )
    )
    result = json.loads(result_json)

    # Should NOT find any bottlenecks because Sharding & AllocateBuffer are
    # non-compute
    self.assertFalse(result["bottlenecks_found"])
    self.assertEmpty(result["inefficient_ops"])

  @mock.patch.object(hlo_tools, "_fetch_debug_info", autospec=True)
  def test_detect_tuple_layout_mismatch(self, mock_fetch):
    debug_info = hlo_tools.hlo_proto_dump_pb2.DebugInfoCollection()
    hlo_proto = debug_info.hlo_proto.add()
    comp = hlo_proto.hlo_module.computations.add(name="main", id=1)

    # Upstream compute op (dot) producing tuple output
    dot_instr = comp.instructions.add(id=100, name="dot_op", opcode="dot")
    # Source shape is a nested tuple: (f32[128, 256]{1,0}, f32[64, 128]{1,0})
    dot_instr.shape.tuple_shapes.add(
        element_type=xla_data_pb2.F32,
        dimensions=[128, 256],
        layout=xla_data_pb2.LayoutProto(minor_to_major=[1, 0]),
    )
    dot_instr.shape.tuple_shapes.add(
        element_type=xla_data_pb2.F32,
        dimensions=[64, 128],
        layout=xla_data_pb2.LayoutProto(minor_to_major=[1, 0]),
    )

    # Sandwiched copy op copying the tuple
    copy_instr = comp.instructions.add(
        id=200, name="copy_op", opcode="copy", operand_ids=[100]
    )
    # Target shape has layout mismatch on one of the tuple elements:
    # (f32[128, 256]{0,1}, f32[64, 128]{1,0})
    copy_instr.shape.tuple_shapes.add(
        element_type=xla_data_pb2.F32,
        dimensions=[128, 256],
        layout=xla_data_pb2.LayoutProto(minor_to_major=[0, 1]),
    )  # Mismatch!
    copy_instr.shape.tuple_shapes.add(
        element_type=xla_data_pb2.F32,
        dimensions=[64, 128],
        layout=xla_data_pb2.LayoutProto(minor_to_major=[1, 0]),
    )

    # Downstream compute op (reduce) consuming the copy
    comp.instructions.add(
        id=300, name="reduce_op", opcode="reduce", operand_ids=[200]
    )

    mock_fetch.return_value = debug_info

    mock_top_ops_fn = mock.create_autospec(_get_top_hlo_ops_spec, spec_set=True)
    mock_top_ops_fn.return_value = "{}"

    result_json = (
        detect_layout_mismatch_copies_tool.detect_layout_mismatch_copies(
            "session_123", get_top_hlo_ops_fn=mock_top_ops_fn
        )
    )
    result = json.loads(result_json)

    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)
    op = result["inefficient_ops"][0]
    self.assertEqual(op["instruction_name"], "copy_op")
    self.assertTrue(op["layout_mismatch"])

  @mock.patch.object(hlo_tools, "_fetch_debug_info", autospec=True)
  def test_detect_boundary_crossing_fusion(self, mock_fetch):
    debug_info = hlo_tools.hlo_proto_dump_pb2.DebugInfoCollection()
    hlo_proto = debug_info.hlo_proto.add()

    # Main computation
    main_comp = hlo_proto.hlo_module.computations.add(name="main", id=1)
    hlo_proto.hlo_module.entry_computation_id = 1

    # 1. Upstream compute op (dot) in main computation
    main_comp.instructions.add(
        id=10,
        name="dot_op",
        opcode="dot",
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.F32,
            dimensions=[128, 256],
            layout=xla_data_pb2.LayoutProto(minor_to_major=[1, 0]),
        ),
    )

    # 2. Fusion caller op in main computation
    main_comp.instructions.add(
        id=20,
        name="fusion_op",
        opcode="fusion",
        operand_ids=[10],  # Operand is dot_op
        called_computation_ids=[2],
    )  # Calls computation 2

    # 3. Downstream compute op (reduce) in main computation
    main_comp.instructions.add(
        id=30, name="reduce_op", opcode="reduce", operand_ids=[20]
    )  # Operand is fusion_op

    # Fusion sub-computation
    fusion_comp = hlo_proto.hlo_module.computations.add(
        name="fusion_comp", id=2
    )

    # parameter of fusion computation
    fusion_comp.instructions.add(
        id=201, name="param_0", opcode="parameter", parameter_number=0
    )

    # sandwiched copy inside fusion
    fusion_comp.instructions.add(
        id=202,
        name="copy_op",
        opcode="copy",
        operand_ids=[201],
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.F32,
            dimensions=[128, 256],
            layout=xla_data_pb2.LayoutProto(minor_to_major=[0, 1]),
        ),
    )  # Layout Mismatch!

    fusion_comp.root_id = 202  # Root is copy_op

    mock_fetch.return_value = debug_info

    mock_top_ops_fn = mock.create_autospec(_get_top_hlo_ops_spec, spec_set=True)
    mock_top_ops_fn.return_value = "{}"

    result_json = (
        detect_layout_mismatch_copies_tool.detect_layout_mismatch_copies(
            "session_123", get_top_hlo_ops_fn=mock_top_ops_fn
        )
    )
    result = json.loads(result_json)

    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)
    op = result["inefficient_ops"][0]
    self.assertEqual(op["instruction_name"], "copy_op")
    self.assertTrue(op["layout_mismatch"])
    # Verify upstream compute is correctly found outside fusion!
    upstream_opcodes = [u["opcode"] for u in op["upstream_stages"]]
    self.assertIn("dot", upstream_opcodes)
    # Verify downstream compute is correctly found outside fusion!
    downstream_opcodes = [d["opcode"] for d in op["downstream_stages"]]
    self.assertIn("reduce", downstream_opcodes)

  @mock.patch.object(hlo_tools, "_fetch_debug_info", autospec=True)
  def test_detect_datatype_aware_tpu_optimality(self, mock_fetch):
    debug_info = hlo_tools.hlo_proto_dump_pb2.DebugInfoCollection()
    hlo_proto = debug_info.hlo_proto.add()
    comp = hlo_proto.hlo_module.computations.add(name="main", id=1)

    # Case A: F64 with minor size 64 (optimal for F64)
    # 1. Upstream
    comp.instructions.add(
        id=100,
        name="dot_op_f64",
        opcode="dot",
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.F64,
            dimensions=[128, 64],
            layout=xla_data_pb2.LayoutProto(minor_to_major=[1, 0]),
        ),
    )

    # 2. Sandwiched copy (should be detected as OPTIMAL since 64-bit requires
    # 64 elements alignment)
    comp.instructions.add(
        id=200,
        name="copy_op_f64",
        opcode="copy",
        operand_ids=[100],
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.F64,
            dimensions=[128, 64],
            layout=xla_data_pb2.LayoutProto(minor_to_major=[1, 0]),
        ),
    )

    # 3. Downstream
    comp.instructions.add(
        id=300, name="reduce_op_f64", opcode="reduce", operand_ids=[200]
    )

    mock_fetch.return_value = debug_info

    mock_top_ops_fn = mock.create_autospec(_get_top_hlo_ops_spec, spec_set=True)
    mock_top_ops_fn.return_value = "{}"

    result_json = (
        detect_layout_mismatch_copies_tool.detect_layout_mismatch_copies(
            "session_123", get_top_hlo_ops_fn=mock_top_ops_fn
        )
    )
    result = json.loads(result_json)

    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)
    op = result["inefficient_ops"][0]
    # F64 shape with 64 element minor dimension must be checked as optimal
    self.assertTrue(op["source_minor_dim_optimal"])
    self.assertTrue(op["target_minor_dim_optimal"])

  @mock.patch.object(hlo_tools, "_fetch_debug_info", autospec=True)
  def test_detect_missing_layout_mismatch(self, mock_fetch):
    debug_info = hlo_tools.hlo_proto_dump_pb2.DebugInfoCollection()
    hlo_proto = debug_info.hlo_proto.add()
    comp = hlo_proto.hlo_module.computations.add(name="main", id=1)

    # Upstream: explicit [0, 1] layout
    comp.instructions.add(
        id=100,
        name="dot_op",
        opcode="dot",
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.F32,
            dimensions=[128, 256],
            layout=xla_data_pb2.LayoutProto(minor_to_major=[0, 1]),
        ),
    )

    # Sandwiched copy: layout is missing (None) which defaults to
    # major-to-minor [1, 0]
    comp.instructions.add(
        id=200,
        name="copy_op",
        opcode="copy",
        operand_ids=[100],
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.F32,
            dimensions=[128, 256],
        ),
    )
    # layout remains None

    # Downstream
    comp.instructions.add(
        id=300, name="reduce_op", opcode="reduce", operand_ids=[200]
    )

    mock_fetch.return_value = debug_info

    mock_top_ops_fn = mock.create_autospec(_get_top_hlo_ops_spec, spec_set=True)
    mock_top_ops_fn.return_value = "{}"

    result_json = (
        detect_layout_mismatch_copies_tool.detect_layout_mismatch_copies(
            "session_123", get_top_hlo_ops_fn=mock_top_ops_fn
        )
    )
    result = json.loads(result_json)

    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)
    op = result["inefficient_ops"][0]
    # Should flag layout mismatch between explicit [0, 1] and default [1, 0]
    self.assertTrue(op["layout_mismatch"])

  @parameterized.named_parameters(
      ("f32_packing_32", "F32", 32, 128),
      ("pred_packing_32", "PRED", 32, 512),
      ("s1_packing_32", "S1", 32, 4096),
      ("s1_packing_8", "S1", 8, 1024),
      ("s2_packing_8", "S2", 8, 1024),
  )
  def test_tpu_datatype_lane_size_packing(
      self, type_name, max_packing, expected_lane_size
  ):
    val = xla_data_pb2.PrimitiveType.Value(type_name)
    self.assertEqual(
        detect_layout_mismatch_copies_tool.get_tpu_lane_size(
            val, max_packing_factor=max_packing
        ),
        expected_lane_size,
    )

  @parameterized.named_parameters(
      ("pred", "PRED", 8),
      ("bf16", "BF16", 16),
      ("c128", "C128", 128),
      ("f8e5m2", "F8E5M2", 8),
      ("s4", "S4", 4),
      ("u2", "U2", 2),
  )
  def test_parse_bit_width_from_name(self, type_name, expected_bit_width):
    self.assertEqual(
        detect_layout_mismatch_copies_tool._parse_bit_width_from_name(
            type_name
        ),
        expected_bit_width,
    )

  @parameterized.named_parameters(
      ("tpu_custom_call", "tpu_custom_call", True),
      ("triton_gpu", "__gpu$xla.gpu.triton.foo", True),
      ("pallas_kernel", "edge_tpu_pallas_kernel", True),
      ("allocate_buffer", "AllocateBuffer", False),
      ("funcresultsharding", "xla.sdy.funcresultsharding", False),
      ("control_dep", "control_dep", False),
  )
  def test_custom_call_heuristics(self, custom_call_target, expected_compute):
    instr = mock.Mock(spec_set=["custom_call_target"])
    instr.custom_call_target = custom_call_target
    self.assertEqual(
        detect_layout_mismatch_copies_tool.is_compute_custom_call(instr),
        expected_compute,
    )

  @mock.patch.object(hlo_tools, "_fetch_debug_info", autospec=True)
  def test_gte_tuple_path_tracking(self, mock_fetch):
    debug_info = hlo_tools.hlo_proto_dump_pb2.DebugInfoCollection()
    hlo_proto = debug_info.hlo_proto.add()
    comp = hlo_proto.hlo_module.computations.add(name="main", id=1)

    comp.instructions.add(id=10, name="dot0", opcode="dot")
    comp.instructions.add(id=20, name="dot1", opcode="dot")
    comp.instructions.add(
        id=30, name="tuple_op", opcode="tuple", operand_ids=[10, 20]
    )

    copy_instr = comp.instructions.add(
        id=40, name="copy_op", opcode="copy", operand_ids=[30]
    )
    copy_instr.shape.tuple_shapes.add(
        element_type=xla_data_pb2.F32,
        dimensions=[128, 256],
        layout=xla_data_pb2.LayoutProto(minor_to_major=[0, 1]),
    )
    copy_instr.shape.tuple_shapes.add(
        element_type=xla_data_pb2.F32,
        dimensions=[64, 64],
        layout=xla_data_pb2.LayoutProto(minor_to_major=[1, 0]),
    )

    comp.instructions.add(
        id=50,
        name="gte0",
        opcode="get-tuple-element",
        tuple_index=0,
        operand_ids=[40],
    )
    comp.instructions.add(
        id=60,
        name="gte1",
        opcode="get-tuple-element",
        tuple_index=1,
        operand_ids=[40],
    )
    comp.instructions.add(
        id=70, name="reduce_op", opcode="reduce", operand_ids=[50]
    )

    mock_fetch.return_value = debug_info
    mock_top_ops_fn = mock.create_autospec(_get_top_hlo_ops_spec, spec_set=True)
    mock_top_ops_fn.return_value = "{}"

    id_to_instr = {instr.id: instr for instr in comp.instructions}
    id_to_comp = {comp.id: comp}
    id_to_users = {40: [50, 60], 50: [70]}

    comp_id_by_instr_id = {instr.id: comp.id for instr in comp.instructions}
    callers_by_comp_id = collections.defaultdict(list)
    root_id_by_comp_id = {comp.id: comp.root_id}

    upstream = detect_layout_mismatch_copies_tool.find_upstream_compute_stages(
        40,
        id_to_instr,
        id_to_comp,
        comp_id_by_instr_id,
        callers_by_comp_id,
        max_depth=5,
    )
    upstream_names = [u.name for u, _ in upstream]
    self.assertIn("dot0", upstream_names)
    self.assertIn("dot1", upstream_names)

    downstream = (
        detect_layout_mismatch_copies_tool.find_downstream_compute_stages(
            40,
            id_to_instr,
            id_to_users,
            id_to_comp,
            comp_id_by_instr_id,
            callers_by_comp_id,
            root_id_by_comp_id,
            max_depth=5,
        )
    )
    downstream_names = [d.name for d, _ in downstream]
    self.assertIn("reduce_op", downstream_names)


if __name__ == "__main__":
  parameterized.absltest.main()
