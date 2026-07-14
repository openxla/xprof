"""Tests for the detect unfused updates tool and its subcomponents."""

import json
import types

from absl.testing import absltest

from xprof.cli.tools import detect_unfused_updates_tool_copy_3 as detect_unfused_updates_tool


class DetectUnfusedUpdatesToolTest(absltest.TestCase):

  def test_detect_unfused_updates_no_hlo_proto(self):
    mock_top_ops = {
        "top_by_time": [
            {"name": "by_program/jit_func/add.1", "total_self_time_ms": 1.5}
        ]
    }

    def mock_get_top_hlo_ops(session_id, limit=50):
      del session_id, limit
      return json.dumps(mock_top_ops)

    def mock_fetch_debug_info(session_id):
      del session_id
      return None

    result_str = detect_unfused_updates_tool.detect_unfused_updates(
        session_id="test_session",
        get_top_hlo_ops_fn=mock_get_top_hlo_ops,
        fetch_debug_info_fn=mock_fetch_debug_info,
    )
    result = json.loads(result_str)
    self.assertIn("error", result)
    self.assertIn("HloProto debug info unavailable", result["error"])

  def test_detect_unfused_updates_end_to_end(self):
    # Mock top ops
    mock_top_ops = {
        "top_by_time": [
            {"name": "by_program/jit_func/add1", "total_self_time_ms": 1.5},
            {"name": "by_program/jit_func/mul1", "total_self_time_ms": 2.0},
        ],
    }

    def mock_get_top_hlo_ops(session_id, limit=50):
      del session_id, limit
      return json.dumps(mock_top_ops)

    # Mock module proto
    p1 = types.SimpleNamespace(
        id=1,
        name="p1",
        opcode="parameter",
        operand_ids=[],
        parameter_number=0,
        called_computation_ids=[],
    )
    p2 = types.SimpleNamespace(
        id=2,
        name="p2",
        opcode="parameter",
        operand_ids=[],
        parameter_number=1,
        called_computation_ids=[],
    )
    add1 = types.SimpleNamespace(
        id=3,
        name="add1",
        opcode="add",
        operand_ids=[1, 2],
        called_computation_ids=[],
        HasField=lambda f: False,
    )
    mul1 = types.SimpleNamespace(
        id=4,
        name="mul1",
        opcode="multiply",
        operand_ids=[3],
        called_computation_ids=[],
        HasField=lambda f: False,
    )
    main_comp = types.SimpleNamespace(
        id=10, name="main", instructions=[p1, p2, add1, mul1]
    )
    module_proto = types.SimpleNamespace(
        name="jit_func", computations=[main_comp]
    )
    hlo_proto_msg = types.SimpleNamespace(hlo_module=module_proto)
    debug_info = types.SimpleNamespace(
        hlo_proto=[hlo_proto_msg], program_id=[123]
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

    op = result["inefficient_ops"][0]
    self.assertEqual(op["unfused_type"], "topological_chain")
    self.assertEqual(op["unfused_chain"], "add1 -> mul1")
    self.assertEqual(op["total_self_time_ms"], 1.5)

  def test_hlo_module_indexer(self):
    # Construct mock proto objects for main computation and fusion computation
    param1 = types.SimpleNamespace(
        id=1,
        name="p1",
        opcode="parameter",
        operand_ids=[],
        parameter_number=0,
        called_computation_ids=[],
    )
    param2 = types.SimpleNamespace(
        id=2,
        name="p2",
        opcode="parameter",
        operand_ids=[],
        parameter_number=1,
        called_computation_ids=[],
    )
    add1 = types.SimpleNamespace(
        id=3,
        name="add1",
        opcode="add",
        operand_ids=[1, 2],
        called_computation_ids=[],
    )
    fusion_call = types.SimpleNamespace(
        id=4,
        name="fusion1",
        opcode="fusion",
        operand_ids=[3],
        called_computation_ids=[100],
    )

    main_comp = types.SimpleNamespace(
        id=10, name="main", instructions=[param1, param2, add1, fusion_call]
    )

    # Inner fusion computation
    fused_p = types.SimpleNamespace(
        id=101,
        name="fp1",
        opcode="parameter",
        operand_ids=[],
        parameter_number=0,
        called_computation_ids=[],
    )
    fused_mul = types.SimpleNamespace(
        id=102,
        name="fmul",
        opcode="multiply",
        operand_ids=[101],
        called_computation_ids=[],
    )
    fusion_comp = types.SimpleNamespace(
        id=100, name="fused_computation", instructions=[fused_p, fused_mul]
    )

    module_proto = types.SimpleNamespace(computations=[main_comp, fusion_comp])

    indexer = detect_unfused_updates_tool._HloModuleIndexer(module_proto)

    # 1. Verify instruction lookup
    self.assertEqual(indexer.instructions[3].name, "add1")
    self.assertEqual(indexer.instructions_by_name["fmul"].id, 102)

    # 2. Verify operands and consumers
    operands_of_add = indexer.get_operand_instructions(3)
    self.assertEqual([op.id for op in operands_of_add], [1, 2])

    consumers_of_p1 = indexer.get_consumer_instructions(1)
    self.assertEqual([c.id for c in consumers_of_p1], [3])

    consumers_of_add = indexer.get_consumer_instructions(3)
    self.assertEqual([c.id for c in consumers_of_add], [4])

    # 3. Verify fusion classification
    self.assertFalse(
        indexer.is_fused_instruction(3)
    )  # add1 is in main, standalone
    # fusion1 is opcode="fusion" (root call is NOT considered fused internally)
    self.assertFalse(
        indexer.is_fused_instruction(4)
    )
    # fmul is inside computation 100, which is called by fusion1
    self.assertTrue(
        indexer.is_fused_instruction(102)
    )

  def test_extract_dag_subgraphs_linear_chain_vs_branching_cluster(self):
    # 1. Pure linear chain: A -> B -> C
    nodes_chain = ["A", "B", "C"]
    edges_chain = {("A", "B"), ("B", "C")}
    subgraphs_chain = detect_unfused_updates_tool.extract_dag_subgraphs(
        nodes_chain, edges_chain
    )
    self.assertLen(subgraphs_chain, 1)
    self.assertEqual(subgraphs_chain[0]["type"], "topological_chain")
    self.assertEqual(subgraphs_chain[0]["nodes"], ["A", "B", "C"])

    # 2. Branching DAG cluster (Fan-out): A -> B and A -> C
    nodes_fanout = ["A", "B", "C"]
    edges_fanout = {("A", "B"), ("A", "C")}
    subgraphs_fanout = detect_unfused_updates_tool.extract_dag_subgraphs(
        nodes_fanout, edges_fanout
    )
    self.assertLen(subgraphs_fanout, 1)
    self.assertEqual(subgraphs_fanout[0]["type"], "unfused_cluster")
    self.assertEqual(
        subgraphs_fanout[0]["nodes"][0], "A"
    )  # A must be topological root
    self.assertCountEqual(subgraphs_fanout[0]["nodes"][1:], ["B", "C"])

  def test_analyze_hlo_module_proto_full_static_scan(self):
    # Construct an HloModuleProto with an unfused 3-op chain and an inner fusion
    p1 = types.SimpleNamespace(
        id=1,
        name="p1",
        opcode="parameter",
        operand_ids=[],
        parameter_number=0,
        called_computation_ids=[],
    )
    p2 = types.SimpleNamespace(
        id=2,
        name="p2",
        opcode="parameter",
        operand_ids=[],
        parameter_number=1,
        called_computation_ids=[],
    )
    # Unfused chain: add1 -> mul1 -> sub1
    add1 = types.SimpleNamespace(
        id=3,
        name="add1",
        opcode="add",
        operand_ids=[1, 2],
        called_computation_ids=[],
        HasField=lambda f: False,
    )
    mul1 = types.SimpleNamespace(
        id=4,
        name="mul1",
        opcode="multiply",
        operand_ids=[3],
        called_computation_ids=[],
        HasField=lambda f: False,
    )
    sub1 = types.SimpleNamespace(
        id=5,
        name="sub1",
        opcode="subtract",
        operand_ids=[4],
        called_computation_ids=[],
        HasField=lambda f: False,
    )

    # Zero-cost ops (bitcast, tuple) that must be ignored
    bcast = types.SimpleNamespace(
        id=6,
        name="bitcast1",
        opcode="bitcast",
        operand_ids=[5],
        called_computation_ids=[],
        HasField=lambda f: False,
    )
    tup = types.SimpleNamespace(
        id=7,
        name="tup1",
        opcode="tuple",
        operand_ids=[6],
        called_computation_ids=[],
        HasField=lambda f: False,
    )

    # Fused operation call
    fusion_op = types.SimpleNamespace(
        id=8,
        name="fused_op",
        opcode="fusion",
        operand_ids=[5],
        called_computation_ids=[99],
        HasField=lambda f: False,
    )

    main_comp = types.SimpleNamespace(
        id=10,
        name="main",
        instructions=[p1, p2, add1, mul1, sub1, bcast, tup, fusion_op],
    )

    # Fusion computation body
    fp1 = types.SimpleNamespace(
        id=100,
        name="fp1",
        opcode="parameter",
        operand_ids=[],
        parameter_number=0,
        called_computation_ids=[],
    )
    fused_add = types.SimpleNamespace(
        id=101,
        name="fused_add",
        opcode="add",
        operand_ids=[100],
        called_computation_ids=[],
        HasField=lambda f: False,
    )
    fusion_comp = types.SimpleNamespace(
        id=99, name="fusion_body", instructions=[fp1, fused_add]
    )

    module_proto = types.SimpleNamespace(computations=[main_comp, fusion_comp])

    bottlenecks, _ = detect_unfused_updates_tool.analyze_hlo_module_proto(
        module_proto, "test_module", {}
    )

    # Verify that:
    # 1. Unfused chain (add1 -> mul1 -> sub1) is detected
    # 2. Bitcast/tuple zero-cost ops are ignored
    # 3. Operations inside fusion_body are ignored
    self.assertLen(bottlenecks, 4)
    b_add1 = next(b for b in bottlenecks if b["instruction"] == "add1")
    self.assertEqual(b_add1["unfused_type"], "topological_chain")
    self.assertEqual(
        b_add1["unfused_chain"], "add1 -> mul1 -> sub1 -> fused_op"
    )
    self.assertEqual(
        b_add1["group_members"], ["add1", "mul1", "sub1", "fused_op"]
    )


if __name__ == "__main__":
  absltest.main()
