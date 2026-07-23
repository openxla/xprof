"""Tests for the detect_unfused_updates_tool HLO parser suggestions."""

import json
import types
from unittest import mock

from absl.testing import absltest

from xprof.cli.tools import detect_unfused_updates_tool


def _make_instr(
    id_val, name, opcode, operands=None, calls=None, fake_shape=False
):
  if operands is None:
    operands = []
  if calls is None:
    calls = []
  return types.SimpleNamespace(
      id=id_val,
      name=name,
      opcode=opcode,
      operand_ids=operands,
      called_computation_ids=calls,
      shape=types.SimpleNamespace(),
      parameter_number=0,
      HasField=lambda f: fake_shape and f == "shape",
  )


class DetectUnfusedUpdatesToolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_fetch_debug = self.enter_context(
        mock.patch.object(
            detect_unfused_updates_tool.hlo_tools,
            "_fetch_debug_info",
            side_effect=RuntimeError("Network isolated in unit test"),
        )
    )

  def test_detect_unfused_updates_success_topological_chain(self):
    mock_top_ops = {
        "top_by_time": [
            {
                "name": "by_program/jit_func/add.1",
                "category": "elementwise",
                "occurrences": 10,
                "flops": 100,
                "bytes_accessed": 400,
                "total_self_time_ms": 1.0,
            },
            {
                "name": "by_program/jit_func/add.2",
                "category": "elementwise",
                "occurrences": 10,
                "flops": 100,
                "bytes_accessed": 400,
                "total_self_time_ms": 1.5,
            },
            {
                "name": "by_program/jit_func/mul.3",
                "category": "elementwise",
                "occurrences": 10,
                "flops": 100,
                "bytes_accessed": 400,
                "total_self_time_ms": 2.0,
            },
        ],
        "top_by_flops": [],
        "top_by_bytes_accessed": [],
    }

    def mock_get_top_hlo_ops(session_id, limit=50):
      del session_id, limit
      return json.dumps(mock_top_ops)

    main_comp = types.SimpleNamespace(
        id=10,
        name="main",
        instructions=[
            _make_instr(1, "p1", "parameter"),
            _make_instr(11, "add.1", "add", [1]),
            _make_instr(12, "add.2", "add", [11]),
            _make_instr(13, "mul.3", "multiply", [12]),
        ],
    )
    module_proto = types.SimpleNamespace(
        name="jit_func", computations=[main_comp]
    )
    debug_info = types.SimpleNamespace(
        hlo_proto=[types.SimpleNamespace(hlo_module=module_proto)],
        program_id=[None],
    )

    def mock_fetch_debug_info(session_id):
      del session_id
      return debug_info

    result_str = detect_unfused_updates_tool.detect_unfused_updates(
        session_id="test_session",
        get_top_hlo_ops_fn=mock_get_top_hlo_ops,
        fetch_debug_info_fn=mock_fetch_debug_info,
    )
    result = json.loads(result_str)
    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 3)

    op1 = result["inefficient_ops"][1]
    self.assertEqual(op1["name"], "by_program/jit_func/add.2")
    self.assertEqual(op1["unfused_type"], "topological_chain")
    self.assertEqual(op1["unfused_chain"], "add.1 -> add.2 -> mul.3")
    self.assertEqual(op1["group_members"], ["add.1", "add.2", "mul.3"])

  def test_detect_unfused_updates_parallel_group(self):
    mock_top_ops = {
        "top_by_time": [
            {
                "name": "by_program/jit_func/sub.1",
                "category": "elementwise",
                "occurrences": 10,
                "flops": 100,
                "bytes_accessed": 400,
                "total_self_time_ms": 1.5,
            },
            {
                "name": "by_program/jit_func/sub.2",
                "category": "elementwise",
                "occurrences": 10,
                "flops": 100,
                "bytes_accessed": 400,
                "total_self_time_ms": 1.5,
            },
        ],
        "top_by_flops": [],
        "top_by_bytes_accessed": [],
    }

    def mock_get_top_hlo_ops(session_id, limit=50):
      del session_id, limit
      return json.dumps(mock_top_ops)

    main_comp = types.SimpleNamespace(
        id=10,
        name="main",
        instructions=[
            _make_instr(11, "sub.1", "subtract"),
            _make_instr(12, "sub.2", "subtract"),
        ],
    )
    module_proto = types.SimpleNamespace(
        name="jit_func", computations=[main_comp]
    )
    debug_info = types.SimpleNamespace(
        hlo_proto=[types.SimpleNamespace(hlo_module=module_proto)],
        program_id=[None],
    )

    def mock_fetch_debug_info(session_id):
      del session_id
      return debug_info

    result_str = detect_unfused_updates_tool.detect_unfused_updates(
        session_id="test_session",
        get_top_hlo_ops_fn=mock_get_top_hlo_ops,
        fetch_debug_info_fn=mock_fetch_debug_info,
    )
    result = json.loads(result_str)
    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 2)

    op1 = result["inefficient_ops"][0]
    self.assertEqual(op1["unfused_type"], "parallel_group")
    self.assertEqual(op1["group_members"], ["sub.1", "sub.2"])
    self.assertIn("parallel/sibling", op1["recommendation"].lower())

  def test_detect_unfused_updates_no_bottlenecks(self):
    mock_top_ops = {
        "top_by_time": [{
            "name": "by_program/jit_func/fusion.1",
            "occurrences": 10,
            "total_self_time_ms": 1.5,
        }]
    }

    def mock_get_top_hlo_ops(session_id, limit=50):
      del session_id, limit
      return json.dumps(mock_top_ops)

    fusion_body = types.SimpleNamespace(
        id=100,
        name="fusion_body",
        instructions=[
            _make_instr(1, "p0", "parameter"),
            _make_instr(2, "add", "add", [1]),
            _make_instr(3, "sub", "subtract", [2]),
            _make_instr(4, "mul", "multiply", [3]),
            _make_instr(5, "div", "divide", [4]),
            _make_instr(6, "exp", "exponential", [5]),
        ],
    )
    main_comp = types.SimpleNamespace(
        id=10,
        name="main",
        instructions=[
            _make_instr(11, "fusion.1", "fusion", calls=[100]),
        ],
    )
    module_proto = types.SimpleNamespace(
        name="jit_func", computations=[main_comp, fusion_body]
    )
    debug_info = types.SimpleNamespace(
        hlo_proto=[types.SimpleNamespace(hlo_module=module_proto)],
        program_id=[None],
    )

    result_str = detect_unfused_updates_tool.detect_unfused_updates(
        session_id="test",
        get_top_hlo_ops_fn=mock_get_top_hlo_ops,
        fetch_debug_info_fn=lambda s: debug_info,
    )
    result = json.loads(result_str)
    self.assertFalse(result["bottlenecks_found"])

  def test_detect_unfused_updates_hierarchical_module_name(self):
    mock_top_ops = {
        "top_by_time": [
            {
                "name": "by_category/Loop fusion/jit_clip/%copy.1",
                "total_self_time_ms": 1.5,
            },
            {
                "name": (
                    "by_category/Loop fusion/jit_clip/%maximum_minimum_fusion"
                ),
                "total_self_time_ms": 1.5,
            },
            {
                "name": "by_category/Loop fusion/jit_clip/%add.1",
                "total_self_time_ms": 1.5,
            },
        ]
    }

    def mock_get_top_hlo_ops(session_id, limit=50):
      del session_id, limit
      return json.dumps(mock_top_ops)

    main_comp = types.SimpleNamespace(
        id=10,
        name="main",
        instructions=[
            _make_instr(11, "copy.1", "copy"),
            _make_instr(12, "maximum_minimum_fusion", "fusion", operands=[11]),
            _make_instr(13, "add.1", "add", operands=[12]),
        ],
    )
    module_proto = types.SimpleNamespace(
        name="jit_clip", computations=[main_comp]
    )
    debug_info = types.SimpleNamespace(
        hlo_proto=[types.SimpleNamespace(hlo_module=module_proto)],
        program_id=[None],
    )

    result_str = detect_unfused_updates_tool.detect_unfused_updates(
        session_id="test",
        get_top_hlo_ops_fn=mock_get_top_hlo_ops,
        fetch_debug_info_fn=lambda s: debug_info,
    )
    result = json.loads(result_str)
    self.assertTrue(result["bottlenecks_found"])
    self.assertEqual(
        result["inefficient_ops"][0]["unfused_chain"],
        "copy.1 -> maximum_minimum_fusion -> add.1",
    )

  def test_detect_unfused_updates_micro_fusion_in_fused_computation(self):
    mock_top_ops = {
        "top_by_time": [
            {
                "name": "by_program/jit_func/max.0",
                "total_self_time_ms": 1.5,
            },
            {
                "name": "by_program/jit_func/min.0",
                "total_self_time_ms": 1.5,
            },
            {
                "name": "by_program/jit_func/add.0",
                "total_self_time_ms": 1.5,
            },
        ]
    }

    def mock_get_top_hlo_ops(session_id, limit=50):
      del session_id, limit
      return json.dumps(mock_top_ops)

    main_comp = types.SimpleNamespace(
        id=10,
        name="main",
        instructions=[
            _make_instr(11, "max.0", "maximum"),
            _make_instr(12, "min.0", "minimum", operands=[11]),
            _make_instr(13, "add.0", "add", operands=[12]),
        ],
    )
    module_proto = types.SimpleNamespace(
        name="jit_func", computations=[main_comp]
    )
    debug_info = types.SimpleNamespace(
        hlo_proto=[types.SimpleNamespace(hlo_module=module_proto)],
        program_id=[None],
    )

    result_str = detect_unfused_updates_tool.detect_unfused_updates(
        session_id="test",
        get_top_hlo_ops_fn=mock_get_top_hlo_ops,
        fetch_debug_info_fn=lambda s: debug_info,
    )
    result = json.loads(result_str)
    self.assertTrue(result["bottlenecks_found"])
    self.assertEqual(
        result["inefficient_ops"][0]["unfused_chain"],
        "max.0 -> min.0 -> add.0",
    )

  def test_detect_unfused_updates_eager_topological_chain_across_modules(self):
    mock_top_ops = {
        "top_by_time": [
            {
                "name": "by_program/jit_sub/sub.1",
                "occurrences": 10,
                "total_self_time_ms": 1.5,
            },
            {
                "name": "by_program/jit_div/div.1",
                "occurrences": 10,
                "total_self_time_ms": 1.5,
            },
        ]
    }

    def mock_get_top_hlo_ops(session_id, limit=50):
      del session_id, limit
      return json.dumps(mock_top_ops)

    comp_sub = types.SimpleNamespace(
        id=10,
        name="main",
        instructions=[_make_instr(11, "sub.1", "subtract", fake_shape=True)],
    )
    mod_sub = types.SimpleNamespace(name="jit_sub", computations=[comp_sub])
    comp_div = types.SimpleNamespace(
        id=20,
        name="main",
        instructions=[_make_instr(21, "div.1", "divide", fake_shape=True)],
    )
    mod_div = types.SimpleNamespace(name="jit_div", computations=[comp_div])

    debug_info = types.SimpleNamespace(
        hlo_proto=[
            types.SimpleNamespace(hlo_module=mod_sub),
            types.SimpleNamespace(hlo_module=mod_div),
        ],
        program_id=[None, None],
    )

    with mock.patch.object(
        detect_unfused_updates_tool,
        "format_shape_proto",
        return_value="f32[100]",
    ):
      result_str = detect_unfused_updates_tool.detect_unfused_updates(
          session_id="test",
          get_top_hlo_ops_fn=mock_get_top_hlo_ops,
          fetch_debug_info_fn=lambda s: debug_info,
      )
    result = json.loads(result_str)
    self.assertTrue(result["bottlenecks_found"])
    self.assertEqual(
        result["inefficient_ops"][0]["unfused_type"], "eager_unfused_group"
    )
    self.assertIn(
        "across distinct eager modules",
        result["inefficient_ops"][0]["recommendation"],
    )

  def test_detect_unfused_updates_dag_cluster(self):
    mock_top_ops = {
        "top_by_time": [
            {
                "name": "by_program/jit_func/add.1",
                "occurrences": 10,
                "total_self_time_ms": 1.5,
            },
            {
                "name": "by_program/jit_func/mul.2",
                "occurrences": 10,
                "total_self_time_ms": 1.5,
            },
            {
                "name": "by_program/jit_func/sub.3",
                "occurrences": 10,
                "total_self_time_ms": 1.5,
            },
        ]
    }

    def mock_get_top_hlo_ops(session_id, limit=50):
      del session_id, limit
      return json.dumps(mock_top_ops)

    main_comp = types.SimpleNamespace(
        id=10,
        name="main",
        instructions=[
            _make_instr(11, "add.1", "add"),
            _make_instr(12, "mul.2", "multiply", operands=[11]),
            _make_instr(13, "sub.3", "subtract", operands=[11]),
        ],
    )
    module_proto = types.SimpleNamespace(
        name="jit_func", computations=[main_comp]
    )
    debug_info = types.SimpleNamespace(
        hlo_proto=[types.SimpleNamespace(hlo_module=module_proto)],
        program_id=[None],
    )

    result_str = detect_unfused_updates_tool.detect_unfused_updates(
        session_id="test",
        get_top_hlo_ops_fn=mock_get_top_hlo_ops,
        fetch_debug_info_fn=lambda s: debug_info,
    )
    result = json.loads(result_str)
    self.assertTrue(result["bottlenecks_found"])
    for op in result["inefficient_ops"]:
      self.assertEqual(op["unfused_type"], "unfused_cluster")
      self.assertIn("{", op["unfused_cluster"])

  def test_detect_unfused_updates_skips_zero_cost_and_metadata_ops(self):
    mock_top_ops = {
        "top_by_time": [
            {
                "name": "by_program/jit_func/bitcast.1",
                "total_self_time_ms": 1.5,
            },
            {
                "name": "by_program/jit_func/tuple.2",
                "total_self_time_ms": 1.5,
            },
            {
                "name": "by_program/jit_func/add.3",
                "total_self_time_ms": 0.0001,
            },
        ]
    }

    def mock_get_top_hlo_ops(session_id, limit=50):
      del session_id, limit
      return json.dumps(mock_top_ops)

    main_comp = types.SimpleNamespace(
        id=10,
        name="main",
        instructions=[
            _make_instr(11, "bitcast.1", "bitcast"),
            _make_instr(12, "tuple.2", "tuple", operands=[11]),
        ],
    )
    module_proto = types.SimpleNamespace(
        name="jit_func", computations=[main_comp]
    )
    debug_info = types.SimpleNamespace(
        hlo_proto=[types.SimpleNamespace(hlo_module=module_proto)],
        program_id=[None],
    )

    result_str = detect_unfused_updates_tool.detect_unfused_updates(
        session_id="test",
        get_top_hlo_ops_fn=mock_get_top_hlo_ops,
        fetch_debug_info_fn=lambda s: debug_info,
    )
    result = json.loads(result_str)
    self.assertFalse(result["bottlenecks_found"])

  def test_detect_unfused_updates_handles_missing_debug_info(self):
    def mock_get_top_hlo_ops(session_id, limit=50):
      del session_id, limit
      return json.dumps(
          {"top_by_time": [{"name": "add.1", "total_self_time_ms": 2.0}]}
      )

    def mock_fetch_debug_info(session_id):
      del session_id
      return None

    result_str = detect_unfused_updates_tool.detect_unfused_updates(
        session_id="test",
        get_top_hlo_ops_fn=mock_get_top_hlo_ops,
        fetch_debug_info_fn=mock_fetch_debug_info,
    )
    result = json.loads(result_str)
    self.assertIn("error", result)
    self.assertIn("unavailable", result["error"])

  def test_detect_unfused_updates_backend_type_error_handling(self):
    def mock_get_top_hlo_ops(session_id, limit=50):
      del session_id, limit
      return json.dumps({"error": "Unexpected data type: NoneType"})

    result_str = detect_unfused_updates_tool.detect_unfused_updates(
        session_id="invalid_session",
        get_top_hlo_ops_fn=mock_get_top_hlo_ops,
    )
    result = json.loads(result_str)
    self.assertIn("error", result)
    self.assertIn(
        "the profiler backend returned no HLO op profile data",
        result["error"],
    )
    self.assertEqual(
        result.get("original_error"), "Unexpected data type: NoneType"
    )

  def test_find_parallel_eager_groups(self):
    unflagged = {
        ("add.1", "jit_module_a"): {
            "shape": None,
            "opcode": "add",
            "comp_id": 10,
        },
        ("add.2", "jit_module_a"): {
            "shape": None,
            "opcode": "add",
            "comp_id": 10,
        },
        ("sub.1", "jit_module_b"): {
            "shape": "f32[100]",
            "opcode": "subtract",
            "comp_id": 20,
        },
        ("div.2", "jit_module_c"): {
            "shape": "f32[100]",
            "opcode": "divide",
            "comp_id": 30,
        },
    }
    profile = {
        ("jit_module_a", "add.1"): {
            "occurrences": 10,
            "total_self_time_ms": 1.0,
        },
        ("jit_module_a", "add.2"): {
            "occurrences": 10,
            "total_self_time_ms": 1.0,
        },
        ("jit_module_b", "sub.1"): {
            "occurrences": 10,
            "total_self_time_ms": 1.0,
        },
        ("jit_module_c", "div.2"): {
            "occurrences": 10,
            "total_self_time_ms": 1.0,
        },
    }
    bottlenecks = detect_unfused_updates_tool.find_parallel_eager_groups(
        unflagged, profile
    )
    self.assertLen(bottlenecks, 4)
    types_found = [b["unfused_type"] for b in bottlenecks]
    self.assertEqual(types_found.count("parallel_group"), 2)
    self.assertEqual(types_found.count("eager_unfused_group"), 2)

  def test_hlo_module_indexer(self):
    param1 = _make_instr(1, "p1", "parameter")
    param2 = _make_instr(2, "p2", "parameter")
    add1 = _make_instr(3, "add1", "add", operands=[1, 2])
    fusion_call = _make_instr(4, "fusion1", "fusion", operands=[3], calls=[100])

    main_comp = types.SimpleNamespace(
        id=10,
        name="main",
        instructions=[param1, param2, add1, fusion_call],
    )

    fused_p = _make_instr(101, "fp1", "parameter")
    fused_mul = _make_instr(102, "fmul", "multiply", operands=[101])
    fusion_comp = types.SimpleNamespace(
        id=100,
        name="fused_computation",
        instructions=[fused_p, fused_mul],
    )

    module_proto = types.SimpleNamespace(computations=[main_comp, fusion_comp])

    indexer = detect_unfused_updates_tool._HloModuleIndexer(module_proto)

    self.assertEqual(indexer.instructions[3].name, "add1")
    self.assertEqual(indexer.instructions_by_name["fmul"].id, 102)

    operands_of_add = indexer.get_operand_instructions(3)
    self.assertEqual([op.id for op in operands_of_add], [1, 2])

    consumers_of_p1 = indexer.get_consumer_instructions(1)
    self.assertEqual([c.id for c in consumers_of_p1], [3])

    consumers_of_add = indexer.get_consumer_instructions(3)
    self.assertEqual([c.id for c in consumers_of_add], [4])

    self.assertFalse(indexer.is_fused_instruction(3))
    self.assertFalse(indexer.is_fused_instruction(4))
    self.assertTrue(indexer.is_fused_instruction(102))

  def test_hlo_module_indexer_non_fusion_caller_not_fused(self):
    # A non-fusion instruction referencing a computation must NOT mark that
    # computation's instructions as fused.
    p0 = _make_instr(1, "p0", "parameter")
    call_op = _make_instr(2, "call_op", "call", operands=[1], calls=[50])
    main_comp = types.SimpleNamespace(
        id=10, name="main", instructions=[p0, call_op]
    )

    cp0 = _make_instr(500, "cp0", "parameter")
    cadd = _make_instr(501, "cadd", "add", operands=[500])
    called_comp = types.SimpleNamespace(
        id=50, name="called_computation", instructions=[cp0, cadd]
    )
    module_proto = types.SimpleNamespace(computations=[main_comp, called_comp])

    indexer = detect_unfused_updates_tool._HloModuleIndexer(module_proto)
    self.assertFalse(indexer.is_fused_instruction(501))
    self.assertEmpty(indexer.fusion_callers)

  def test_extract_dag_subgraphs_linear_chain_vs_branching_cluster(self):
    nodes_chain = ["A", "B", "C"]
    edges_chain = {("A", "B"), ("B", "C")}
    subgraphs_chain = detect_unfused_updates_tool.extract_dag_subgraphs(
        nodes_chain, edges_chain
    )
    self.assertLen(subgraphs_chain, 1)
    self.assertEqual(subgraphs_chain[0]["type"], "topological_chain")
    self.assertEqual(subgraphs_chain[0]["nodes"], ["A", "B", "C"])

    nodes_fanout = ["A", "B", "C"]
    edges_fanout = {("A", "B"), ("A", "C")}
    subgraphs_fanout = detect_unfused_updates_tool.extract_dag_subgraphs(
        nodes_fanout, edges_fanout
    )
    self.assertLen(subgraphs_fanout, 1)
    self.assertEqual(subgraphs_fanout[0]["type"], "unfused_cluster")
    self.assertEqual(subgraphs_fanout[0]["nodes"][0], "A")
    self.assertCountEqual(subgraphs_fanout[0]["nodes"][1:], ["B", "C"])

  def test_analyze_hlo_module_proto_full_static_scan(self):
    p1 = _make_instr(1, "p1", "parameter")
    p2 = _make_instr(2, "p2", "parameter")
    add1 = _make_instr(3, "add1", "add", operands=[1, 2])
    mul1 = _make_instr(4, "mul1", "multiply", operands=[3])
    sub1 = _make_instr(5, "sub1", "subtract", operands=[4])

    bcast = _make_instr(6, "bitcast1", "bitcast", operands=[5])
    tup = _make_instr(7, "tup1", "tuple", operands=[6])

    fusion_op = _make_instr(8, "fused_op", "fusion", operands=[5], calls=[99])

    main_comp = types.SimpleNamespace(
        id=10,
        name="main",
        instructions=[p1, p2, add1, mul1, sub1, bcast, tup, fusion_op],
    )

    fp1 = _make_instr(100, "fp1", "parameter")
    fused_add = _make_instr(101, "fused_add", "add", operands=[100])
    fusion_comp = types.SimpleNamespace(
        id=99, name="fusion_body", instructions=[fp1, fused_add]
    )

    module_proto = types.SimpleNamespace(computations=[main_comp, fusion_comp])

    dummy_profile = {
        ("test_module", "add1"): {"total_self_time_ms": 1.0},
        ("test_module", "mul1"): {"total_self_time_ms": 1.0},
        ("test_module", "sub1"): {"total_self_time_ms": 1.0},
        ("test_module", "fused_op"): {"total_self_time_ms": 1.0},
    }

    bottlenecks, _ = detect_unfused_updates_tool.analyze_hlo_module_proto(
        module_proto, "test_module", dummy_profile
    )

    self.assertLen(bottlenecks, 4)
    b_add1 = next(b for b in bottlenecks if b["instruction"] == "add1")
    self.assertEqual(b_add1["unfused_type"], "topological_chain")
    self.assertEqual(
        b_add1["unfused_chain"], "add1 -> mul1 -> sub1 -> fused_op"
    )
    self.assertEqual(
        b_add1["group_members"], ["add1", "mul1", "sub1", "fused_op"]
    )

  def test_get_profile_info_lookup_and_fallbacks(self):
    profile_map = {
        ("jit_func", "add.1"): {"total_self_time_ms": 1.5, "occurrences": 10},
        ("jit_module", "sub.2"): {"total_self_time_ms": 2.0, "occurrences": 5},
    }
    # Exact match
    res1 = detect_unfused_updates_tool._get_profile_info(
        profile_map, "jit_func", "add.1"
    )
    self.assertEqual(res1.get("total_self_time_ms"), 1.5)

    # Fallback with program ID in module name: "jit_func(12345)"
    res2 = detect_unfused_updates_tool._get_profile_info(
        profile_map, "jit_func(12345)", "add.1"
    )
    self.assertEqual(res2.get("total_self_time_ms"), 1.5)

    # Missing op returns empty dict
    res3 = detect_unfused_updates_tool._get_profile_info(
        profile_map, "jit_func", "non_existent"
    )
    self.assertEqual(res3, {})

  def test_profile_filtering_excludes_cold_and_unprofiled_ops(self):
    p1 = _make_instr(1, "p1", "parameter")
    add1 = _make_instr(2, "add1", "add", operands=[1])
    mul1 = _make_instr(3, "mul1", "multiply", operands=[2])
    main_comp = types.SimpleNamespace(
        id=10, name="main", instructions=[p1, add1, mul1]
    )
    module_proto = types.SimpleNamespace(
        name="jit_func", computations=[main_comp]
    )

    # mul1 is not in profile_map (unprofiled)
    profile_map = {
        ("jit_func", "add1"): {"total_self_time_ms": 1.0},
    }
    bottlenecks, unflagged = (
        detect_unfused_updates_tool.analyze_hlo_module_proto(
            module_proto, "jit_func", profile_map
        )
    )
    self.assertEmpty(bottlenecks)
    self.assertIn(("add1", "jit_func"), unflagged)
    self.assertNotIn(("mul1", "jit_func"), unflagged)

  def test_zero_cost_op_bfs_traversal(self):
    p1 = _make_instr(1, "p1", "parameter")
    add1 = _make_instr(2, "add1", "add", operands=[1])
    bcast = _make_instr(3, "bitcast1", "bitcast", operands=[2])
    tup = _make_instr(4, "tuple1", "tuple", operands=[3])
    mul1 = _make_instr(5, "mul1", "multiply", operands=[4])

    main_comp = types.SimpleNamespace(
        id=10, name="main", instructions=[p1, add1, bcast, tup, mul1]
    )
    module_proto = types.SimpleNamespace(
        name="jit_func", computations=[main_comp]
    )

    profile_map = {
        ("jit_func", "add1"): {"total_self_time_ms": 1.0},
        ("jit_func", "mul1"): {"total_self_time_ms": 1.0},
    }
    bottlenecks, _ = detect_unfused_updates_tool.analyze_hlo_module_proto(
        module_proto, "jit_func", profile_map
    )
    self.assertLen(bottlenecks, 2)
    self.assertEqual(bottlenecks[0]["unfused_type"], "topological_chain")
    self.assertEqual(bottlenecks[0]["unfused_chain"], "add1 -> mul1")

  def test_micro_fusion_filtering_threshold(self):
    # Micro fusion with <= 4 instructions (unique IDs 10..13)
    micro_insts = [_make_instr(10 + i, f"m{i}", "add") for i in range(4)]
    micro_comp = types.SimpleNamespace(
        id=101, name="micro_comp", instructions=micro_insts
    )

    # Large fusion with > 4 instructions (unique IDs 20..24)
    large_insts = [_make_instr(20 + i, f"l{i}", "add") for i in range(5)]
    large_comp = types.SimpleNamespace(
        id=102, name="large_comp", instructions=large_insts
    )

    f_micro = _make_instr(1, "f_micro", "fusion", calls=[101])
    f_large = _make_instr(2, "f_large", "fusion", calls=[102])
    main_comp = types.SimpleNamespace(
        id=10, name="main", instructions=[f_micro, f_large]
    )
    module_proto = types.SimpleNamespace(
        name="jit_func", computations=[main_comp, micro_comp, large_comp]
    )

    profile_map = {
        ("jit_func", "f_micro"): {"total_self_time_ms": 1.0},
        ("jit_func", "f_large"): {"total_self_time_ms": 1.0},
    }
    _, unflagged = detect_unfused_updates_tool.analyze_hlo_module_proto(
        module_proto, "jit_func", profile_map
    )
    self.assertIn(("f_micro", "jit_func"), unflagged)
    self.assertNotIn(("f_large", "jit_func"), unflagged)

  def test_format_shape_proto_primitive(self):
    shape = mock.Mock(
        spec=["element_type", "dimensions"],
        element_type=11,
        dimensions=[2, 3],
    )
    self.assertEqual(
        detect_unfused_updates_tool.format_shape_proto(shape), "f32[2,3]"
    )

  def test_format_shape_proto_tuple(self):
    shape1 = mock.Mock(
        spec=["element_type", "dimensions"],
        element_type=11,
        dimensions=[2, 3],
    )
    shape2 = mock.Mock(
        spec=["element_type", "dimensions"], element_type=4, dimensions=[5]
    )
    tuple_shape = mock.Mock(
        spec=["element_type", "tuple_shapes"],
        element_type=13,
        tuple_shapes=[shape1, shape2],
    )
    self.assertEqual(
        detect_unfused_updates_tool.format_shape_proto(tuple_shape),
        "(f32[2,3], s32[5])",
    )

  def test_format_shape_proto_token_opaque(self):
    token_shape = mock.Mock(spec=["element_type"], element_type=15)
    self.assertEqual(
        detect_unfused_updates_tool.format_shape_proto(token_shape), "token[]"
    )

    opaque_shape = mock.Mock(spec=["element_type"], element_type=14)
    self.assertEqual(
        detect_unfused_updates_tool.format_shape_proto(opaque_shape), "opaque[]"
    )

  def test_format_shape_proto_empty(self):
    self.assertIsNone(detect_unfused_updates_tool.format_shape_proto(None))

    # Missing element_type
    empty_shape = mock.Mock(spec=[])
    self.assertIsNone(
        detect_unfused_updates_tool.format_shape_proto(empty_shape)
    )

  def test_format_shape_proto_unknown_primitive(self):
    shape = mock.Mock(
        spec=["element_type", "dimensions"], element_type=999, dimensions=[1]
    )
    self.assertEqual(
        detect_unfused_updates_tool.format_shape_proto(shape),
        "primitive_999[1]",
    )

  def test_format_shape_proto_non_integer_element_type(self):
    shape = mock.Mock(
        spec=["element_type", "dimensions"],
        element_type="CustomType",
        dimensions=[5, 10],
    )
    self.assertEqual(
        detect_unfused_updates_tool.format_shape_proto(shape),
        "customtype[5,10]",
    )

  def test_format_shape_proto_primitive_s32_no_dimensions(self):
    shape = mock.Mock(
        spec=["element_type", "dimensions"], element_type=4, dimensions=[]
    )
    self.assertEqual(
        detect_unfused_updates_tool.format_shape_proto(shape), "s32[]"
    )


if __name__ == "__main__":
  absltest.main()
