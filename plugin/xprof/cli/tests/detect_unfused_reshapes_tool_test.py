"""Tests for the detect_unfused_reshapes_tool.

This suite combines the strengths of the two earlier drafts and adds coverage
for behavior that neither exercised, including:
  * Error paths (empty response, malformed JSON, unexpected exceptions).
  * The ``list_hlo_modules`` fallback when the direct neighborhood lookup
  misses.
  * Fusion-context exclusion (fused ops must not be flagged as standalone).
  * Negative combinations (standalone-only, downstream-compute-only).
  * Instruction-name cleanup ("%" prefix and " and its duplicate(s)" suffix).
  * The configurable ``limit`` and ``min_bytes_accessed`` parameters.
  * Breadth over all recognized formatting keywords and compute opcodes.
  * The presence of output metadata and log metrics.
"""

from collections.abc import Sequence
import json
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from xprof.cli.tools import detect_unfused_reshapes_tool

_MB = 1024 * 1024


def _make_instr(
    id_val: int,
    name: str,
    opcode: str,
    operands: Sequence[int] | None = None,
    calls: Sequence[int] | None = None,
) -> types.SimpleNamespace:
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
  )


def _top_ops_json(ops: Sequence[dict[str, Any]]) -> str:
  """Wraps a list of op dicts in the JSON envelope the tool expects."""
  return json.dumps({"top_by_bytes_accessed": ops})


def _op(
    name: str,
    category: str = "other",
    bytes_accessed: int = 20 * _MB,
) -> dict[str, Any]:
  return {
      "name": name,
      "category": category,
      "bytes_accessed": bytes_accessed,
  }


def _neighborhood(
    instr_name: str,
    downstream_op: str,
    computation: str = "main.1",
) -> str:
  """Builds a two-line neighborhood: a standalone def feeding a compute op."""
  return (
      f"[dist=0] [{computation}] %{instr_name} = f32[100,100]"
      " reshape(p0)\n"
      f"[dist=1] [{computation}] %out = f32[100,100]"
      f" {downstream_op}(%{instr_name})\n"
  )


class DetectUnfusedReshapesToolTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Guard against any accidental real network call for module listing and
    # debug-info paths. Individual tests override the return value as needed.
    self.mock_list_modules = self.enter_context(
        mock.patch.object(
            detect_unfused_reshapes_tool.hlo_tools,
            "list_hlo_modules",
            return_value="No HLO modules found.",
        )
    )
    self.mock_fetch_debug_info = self.enter_context(
        mock.patch.object(
            detect_unfused_reshapes_tool.hlo_tools,
            "_fetch_debug_info",
            return_value=None,
        )
    )

  # ---------------------------------------------------------------------------
  # Constants and parameter plumbing.
  # ---------------------------------------------------------------------------

  def test_min_bytes_accessed_constant(self):
    self.assertEqual(detect_unfused_reshapes_tool.MIN_BYTES_ACCESSED, 10 * _MB)

  def test_default_limit_is_forwarded(self):
    """The default limit (75) must be forwarded unchanged."""
    captured = {}

    def capturing_get_top_hlo_ops(session_id, limit):
      del session_id
      captured["limit"] = limit
      return _top_ops_json([])

    detect_unfused_reshapes_tool.detect_unfused_reshapes(
        "s", get_top_hlo_ops_fn=capturing_get_top_hlo_ops
    )
    self.assertEqual(captured["limit"], 75)

  def test_custom_limit_is_forwarded(self):
    captured = {}

    def capturing_get_top_hlo_ops(session_id, limit):
      del session_id
      captured["limit"] = limit
      return _top_ops_json([])

    detect_unfused_reshapes_tool.detect_unfused_reshapes(
        "s", get_top_hlo_ops_fn=capturing_get_top_hlo_ops, limit=123
    )
    self.assertEqual(captured["limit"], 123)

  # ---------------------------------------------------------------------------
  # Threshold filtering.
  # ---------------------------------------------------------------------------

  def test_below_threshold_filtered_out(self):
    """Ops under MIN_BYTES_ACCESSED are dropped before graph analysis."""
    ops = [_op("by_program/jit_func/reshape.1", "reshape", 5 * _MB)]

    result = self._run(
        ops, neighborhood_fn=lambda *a, **k: _neighborhood("reshape.1", "dot")
    )
    self.assertFalse(result["bottlenecks_found"])
    self.assertEqual(result["message"], "No formatting operations found.")
    self.assertEmpty(result["inefficient_ops"])

  def test_above_threshold_detected(self):
    ops = [_op("by_program/jit_func/reshape.2", "reshape", 15 * _MB)]

    result = self._run(
        ops, neighborhood_fn=lambda *a, **k: _neighborhood("reshape.2", "dot")
    )
    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)
    op = result["inefficient_ops"][0]
    self.assertEqual(op["name"], "by_program/jit_func/reshape.2")
    self.assertEqual(op["downstream_compute"], "dot")
    self.assertTrue(op["hbm_materialization_overhead"])
    self.assertIn(
        "Standalone formatting op 'reshape.2' feeds into compute op 'dot'",
        op["recommendation"],
    )

  def test_custom_min_bytes_accessed_param(self):
    """A caller-supplied threshold overrides the module default."""
    ops = [_op("by_program/jit_func/reshape.1", "reshape", 20 * _MB)]

    result = self._run(
        ops,
        neighborhood_fn=lambda *a, **k: _neighborhood("reshape.1", "dot"),
        min_bytes_accessed=30 * _MB,
    )
    self.assertFalse(result["bottlenecks_found"])
    self.assertEqual(result["message"], "No formatting operations found.")

  # ---------------------------------------------------------------------------
  # Candidate detection: categories and name keywords.
  # ---------------------------------------------------------------------------

  @parameterized.parameters("data formatting", "copy", "reshape", "transpose")
  def test_category_based_detection(self, category):
    """Formatting categories are detected even when the name has no keyword."""
    ops = [_op("by_program/jit_func/op.1", category, 20 * _MB)]

    result = self._run(
        ops, neighborhood_fn=lambda *a, **k: _neighborhood("op.1", "dot")
    )
    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)

  @parameterized.parameters(
      "reshape",
      "transpose",
      "copy",
      "broadcast",
      "slice",
      "pad",
      "convert",
  )
  def test_name_keyword_detection(self, keyword):
    """Every formatting keyword in the name triggers candidate selection."""
    instr = f"{keyword}.7"
    ops = [_op(f"by_program/jit_func/{instr}", "other", 20 * _MB)]

    result = self._run(
        ops, neighborhood_fn=lambda *a, **k: _neighborhood(instr, "dot")
    )
    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)

  # ---------------------------------------------------------------------------
  # Downstream compute-op recognition.
  # ---------------------------------------------------------------------------

  @parameterized.named_parameters(
      ("dot", "dot", "dot"),
      ("einsum", "einsum", "einsum"),
      ("custom_call", "custom-call", "custom-call"),
      ("fusion", "fusion", "fusion"),
      ("convolution", "convolution", "convolution"),
      ("reduce", "reduce", "reduce"),
      # "reduce-window" matches "reduce-window(" opcode.
      ("reduce_window", "reduce-window", "reduce-window"),
      ("fft", "fft", "fft"),
      ("cholesky", "cholesky", "cholesky"),
      ("triangular_solve", "triangular-solve", "triangular-solve"),
      ("sort", "sort", "sort"),
      ("topk", "topk", "topk"),
      ("bn_training", "batch-norm-training", "batch-norm-training"),
      ("bn_inference", "batch-norm-inference", "batch-norm-inference"),
      ("bn_grad", "batch-norm-grad", "batch-norm-grad"),
  )
  def test_compute_op_recognition(self, downstream_op, expected_target):
    ops = [_op("by_program/jit_func/reshape.1", "reshape", 20 * _MB)]

    result = self._run(
        ops,
        neighborhood_fn=lambda *a, **k: _neighborhood(
            "reshape.1", downstream_op
        ),
    )
    self.assertTrue(result["bottlenecks_found"])
    self.assertEqual(
        result["inefficient_ops"][0]["downstream_compute"], expected_target
    )

  # ---------------------------------------------------------------------------
  # Multiple ops and negative cases.
  # ---------------------------------------------------------------------------

  def test_multiple_ops_with_distinct_neighborhoods(self):
    ops = [
        _op("by_program/jit_func/slice.1", "other", 20 * _MB),
        _op("by_program/jit_func/broadcast.2", "other", 25 * _MB),
    ]

    def neighborhood_fn(session_id, instruction_name, radius, module_name):
      del session_id, radius, module_name
      if instruction_name == "slice.1":
        return _neighborhood("slice.1", "batch-norm-training")
      if instruction_name == "broadcast.2":
        return _neighborhood("broadcast.2", "reduce")
      return ""

    result = self._run(ops, neighborhood_fn=neighborhood_fn)
    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 2)
    self.assertCountEqual(
        [op["downstream_compute"] for op in result["inefficient_ops"]],
        ["batch-norm-training", "reduce"],
    )

  def test_no_formatting_ops_found(self):
    ops = [_op("by_program/jit_func/dot.1", "dot", 50 * _MB)]

    result = self._run(ops, neighborhood_fn=lambda *a, **k: "")
    self.assertFalse(result["bottlenecks_found"])
    self.assertEqual(result["message"], "No formatting operations found.")

  def test_standalone_without_downstream_compute_not_flagged(self):
    """A formatting op that only feeds another formatting op is not a bottleneck."""
    ops = [_op("by_program/jit_func/reshape.1", "reshape", 20 * _MB)]

    def neighborhood_fn(*a, **k):
      del a, k
      return (
          "[dist=0] [main.1] %reshape.1 = f32[100,100] reshape(p0)\n[dist=1]"
          " [main.1] %transpose.2 = f32[100,100] transpose(%reshape.1)\n"
      )

    result = self._run(ops, neighborhood_fn=neighborhood_fn)
    self.assertFalse(result["bottlenecks_found"])
    self.assertEqual(
        result["message"], "No unfused reshape bottlenecks detected."
    )
    self.assertEmpty(result["inefficient_ops"])

  def test_fused_op_not_flagged(self):
    """An op defined inside a fused computation must not be marked standalone."""
    ops = [_op("by_program/jit_func/reshape.1", "reshape", 20 * _MB)]

    def neighborhood_fn(*a, **k):
      del a, k
      return (
          "[dist=0] [fused_computation.3] %reshape.1 = f32[100,100]"
          " reshape(p0)\n"
          "[dist=1] [fused_computation.3] %out = f32[100,100]"
          " dot(%reshape.1)\n"
      )

    result = self._run(ops, neighborhood_fn=neighborhood_fn)
    self.assertFalse(result["bottlenecks_found"])
    self.assertEmpty(result["inefficient_ops"])

  def test_standalone_definition_without_bracket_context(self):
    """Definition lines lacking a bracketed computation are still standalone."""
    ops = [_op("by_program/jit_func/reshape.1", "reshape", 20 * _MB)]

    def neighborhood_fn(*a, **k):
      del a, k
      return (
          "%reshape.1 = f32[100,100] reshape(p0)\n"
          "[dist=1] [main.1] %out = f32[100,100] dot(%reshape.1)\n"
      )

    result = self._run(ops, neighborhood_fn=neighborhood_fn)
    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)

  # ---------------------------------------------------------------------------
  # Instruction-name extraction and cleanup.
  # ---------------------------------------------------------------------------

  def test_instruction_name_strips_percent_and_duplicate_suffix(self):
    ops = [
        _op(
            "by_program/jit_func/%reshape.1 and its duplicate(s)",
            "reshape",
            20 * _MB,
        )
    ]
    captured = {}

    def neighborhood_fn(session_id, instruction_name, radius, module_name):
      del session_id, radius
      captured["instruction_name"] = instruction_name
      captured["module_name"] = module_name
      return _neighborhood("reshape.1", "dot")

    result = self._run(ops, neighborhood_fn=neighborhood_fn)
    self.assertEqual(captured["instruction_name"], "reshape.1")
    self.assertEqual(captured["module_name"], "jit_func")
    self.assertTrue(result["bottlenecks_found"])

  # ---------------------------------------------------------------------------
  # list_hlo_modules fallback path.
  # ---------------------------------------------------------------------------

  def test_module_fallback_when_direct_lookup_not_found(self):
    """When the module-scoped lookup misses, we scan modules from list_hlo_modules."""
    ops = [_op("by_program/jit_func/reshape.1", "reshape", 20 * _MB)]
    self.mock_list_modules.return_value = (
        "1. jit_func(111)\n2. real_module(222)\n"
    )

    def neighborhood_fn(session_id, instruction_name, radius, module_name):
      del session_id, instruction_name, radius
      if module_name == "real_module":
        return _neighborhood("reshape.1", "dot")
      return "Instruction not found in module."

    result = self._run(ops, neighborhood_fn=neighborhood_fn)
    self.mock_list_modules.assert_called_once()
    self.assertTrue(result["bottlenecks_found"])
    self.assertEqual(result["inefficient_ops"][0]["downstream_compute"], "dot")

  def test_name_without_module_segment_uses_fallback(self):
    """A bare instruction name (no '/module/') goes straight to the fallback."""
    ops = [_op("reshape.1", "reshape", 20 * _MB)]
    self.mock_list_modules.return_value = "1. real_module(222)\n"

    def neighborhood_fn(session_id, instruction_name, radius, module_name):
      del session_id, instruction_name, radius
      if module_name == "real_module":
        return _neighborhood("reshape.1", "dot")
      return "not found"

    result = self._run(ops, neighborhood_fn=neighborhood_fn)
    self.mock_list_modules.assert_called_once()
    self.assertTrue(result["bottlenecks_found"])

  def test_instruction_not_found_in_any_module_is_skipped(self):
    ops = [_op("by_program/jit_func/reshape.1", "reshape", 20 * _MB)]
    self.mock_list_modules.return_value = "1. some_module(333)\n"

    result = self._run(ops, neighborhood_fn=lambda *a, **k: "not found")
    self.assertFalse(result["bottlenecks_found"])
    self.assertEqual(
        result["message"], "No unfused reshape bottlenecks detected."
    )
    self.assertEmpty(result["inefficient_ops"])

  # ---------------------------------------------------------------------------
  # Error handling.
  # ---------------------------------------------------------------------------

  def test_empty_top_ops_response_returns_error(self):
    result = json.loads(
        detect_unfused_reshapes_tool.detect_unfused_reshapes(
            "s", get_top_hlo_ops_fn=lambda *a, **k: ""
        )
    )
    self.assertEqual(result["error"], "Could not fetch top HLO ops.")

  def test_malformed_json_returns_error(self):
    result = json.loads(
        detect_unfused_reshapes_tool.detect_unfused_reshapes(
            "s", get_top_hlo_ops_fn=lambda *a, **k: "{not valid json"
        )
    )
    self.assertIn("Malformed JSON data from backend", result["error"])

  def test_unexpected_exception_returns_error(self):
    def boom(*a, **k):
      del a, k
      raise ValueError("kaboom")

    result = json.loads(
        detect_unfused_reshapes_tool.detect_unfused_reshapes(
            "s", get_top_hlo_ops_fn=boom
        )
    )
    self.assertIn("Internal error during detection", result["error"])
    self.assertIn("kaboom", result["error"])

  def test_missing_top_by_bytes_key(self):
    result = json.loads(
        detect_unfused_reshapes_tool.detect_unfused_reshapes(
            "s", get_top_hlo_ops_fn=lambda *a, **k: json.dumps({})
        )
    )
    self.assertFalse(result["bottlenecks_found"])
    self.assertEqual(result["message"], "No formatting operations found.")

  # ---------------------------------------------------------------------------
  # Output shape / metadata.
  # ---------------------------------------------------------------------------

  def test_output_contains_metadata_and_message(self):
    ops = [_op("by_program/jit_func/reshape.1", "reshape", 20 * _MB)]

    with self.assertLogs(level="INFO") as log_output:
      result = self._run(
          ops, neighborhood_fn=lambda *a, **k: _neighborhood("reshape.1", "dot")
      )

    for key in (
        "bottlenecks_found",
        "inefficient_ops",
        "message",
    ):
      self.assertIn(key, result)

    op = result["inefficient_ops"][0]
    for op_key in (
        "name",
        "category",
        "bytes_accessed",
        "hbm_materialization_overhead",
        "downstream_compute",
        "recommendation",
    ):
      self.assertIn(op_key, op)

    self.assertIn(
        "Detected 1 standalone formatting operations", result["message"]
    )
    self.assertTrue(
        any(
            "Unfused reshapes detection metrics" in log_msg
            for log_msg in log_output.output
        )
    )

  # ---------------------------------------------------------------------------
  # Helper.
  # ---------------------------------------------------------------------------

  def _run(
      self,
      ops: Sequence[dict[str, Any]],
      neighborhood_fn: Any,
      **kwargs: Any,
  ) -> dict[str, Any]:
    """Runs the tool with mocked dependencies and returns the parsed result."""
    result_json = detect_unfused_reshapes_tool.detect_unfused_reshapes(
        "test_session",
        get_top_hlo_ops_fn=lambda session_id, limit: _top_ops_json(ops),
        get_hlo_neighborhood_fn=neighborhood_fn,
        **kwargs,
    )
    return json.loads(result_json)

  def test_proto_based_analysis_detects_unfused_reshapes(self):
    p0 = _make_instr(1, "p0", "parameter")
    reshape1 = _make_instr(2, "reshape.1", "reshape", operands=[1])
    dot1 = _make_instr(3, "dot.1", "dot", operands=[2])
    main_comp = types.SimpleNamespace(
        id=10, name="main", instructions=[p0, reshape1, dot1]
    )
    module_proto = types.SimpleNamespace(
        name="jit_func", computations=[main_comp]
    )
    debug_info = types.SimpleNamespace(
        hlo_proto=[types.SimpleNamespace(hlo_module=module_proto)],
        program_id=[None],
    )

    ops = [_op("by_program/jit_func/reshape.1", "reshape", 20 * _MB)]
    result_str = detect_unfused_reshapes_tool.detect_unfused_reshapes(
        "s",
        get_top_hlo_ops_fn=lambda s, limit: _top_ops_json(ops),
        fetch_debug_info_fn=lambda s: debug_info,
    )
    result = json.loads(result_str)
    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)
    op = result["inefficient_ops"][0]
    self.assertEqual(op["downstream_compute"], "dot")
    self.assertTrue(op["hbm_materialization_overhead"])

  def test_proto_based_analysis_ignores_fused_reshapes(self):
    p0 = _make_instr(1, "p0", "parameter")
    reshape1 = _make_instr(2, "reshape.1", "reshape", operands=[1])
    dot1 = _make_instr(3, "dot.1", "dot", operands=[2])
    fused_comp = types.SimpleNamespace(
        id=101, name="fused_computation", instructions=[p0, reshape1, dot1]
    )
    fusion_instr = _make_instr(10, "fusion.1", "fusion", calls=[101])
    main_comp = types.SimpleNamespace(
        id=10, name="main", instructions=[fusion_instr]
    )
    module_proto = types.SimpleNamespace(
        name="jit_func", computations=[main_comp, fused_comp]
    )
    debug_info = types.SimpleNamespace(
        hlo_proto=[types.SimpleNamespace(hlo_module=module_proto)],
        program_id=[None],
    )

    ops = [_op("by_program/jit_func/reshape.1", "reshape", 20 * _MB)]
    result_str = detect_unfused_reshapes_tool.detect_unfused_reshapes(
        "s",
        get_top_hlo_ops_fn=lambda s, limit: _top_ops_json(ops),
        fetch_debug_info_fn=lambda s: debug_info,
    )
    result = json.loads(result_str)
    self.assertFalse(result["bottlenecks_found"])
    self.assertEmpty(result["inefficient_ops"])

  def test_proto_based_analysis_multi_module_lookup(self):
    # Module 1 (unrelated)
    p0 = _make_instr(1, "p0", "parameter")
    comp1 = types.SimpleNamespace(id=1, name="main", instructions=[p0])
    mod1 = types.SimpleNamespace(name="module_a", computations=[comp1])

    # Module 2 (contains candidate)
    p1 = _make_instr(10, "p1", "parameter")
    transpose1 = _make_instr(11, "transpose.1", "transpose", operands=[10])
    einsum1 = _make_instr(12, "einsum.1", "einsum", operands=[11])
    comp2 = types.SimpleNamespace(
        id=2, name="main", instructions=[p1, transpose1, einsum1]
    )
    mod2 = types.SimpleNamespace(name="module_b", computations=[comp2])

    debug_info = types.SimpleNamespace(
        hlo_proto=[
            types.SimpleNamespace(hlo_module=mod1),
            types.SimpleNamespace(hlo_module=mod2),
        ],
        program_id=["111", "222"],
    )

    # Candidate specifies module name with program ID: module_b(222)
    ops = [_op("by_program/module_b(222)/transpose.1", "transpose", 25 * _MB)]
    result_str = detect_unfused_reshapes_tool.detect_unfused_reshapes(
        "s",
        get_top_hlo_ops_fn=lambda s, limit: _top_ops_json(ops),
        fetch_debug_info_fn=lambda s: debug_info,
    )
    result = json.loads(result_str)
    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)
    self.assertEqual(
        result["inefficient_ops"][0]["downstream_compute"], "einsum"
    )

  def test_large_proto_performance(self):
    modules = []
    for mod_idx in range(20):
      instructions = [
          _make_instr(
              mod_idx * 1000 + i,
              f"instr_{mod_idx}_{i}",
              "reshape" if i == 10 else ("dot" if i == 11 else "add"),
              operands=[mod_idx * 1000 + i - 1] if i > 0 else [],
          )
          for i in range(1000)
      ]
      comp = types.SimpleNamespace(
          id=mod_idx, name="main", instructions=instructions
      )
      mod = types.SimpleNamespace(name=f"module_{mod_idx}", computations=[comp])
      modules.append(types.SimpleNamespace(hlo_module=mod))

    debug_info = types.SimpleNamespace(
        hlo_proto=modules, program_id=[None] * 20
    )

    ops = [
        _op("by_program/module_0/instr_0_10", "reshape", 20 * _MB),
        _op("by_program/module_1/instr_1_10", "reshape", 20 * _MB),
    ]

    result_str = detect_unfused_reshapes_tool.detect_unfused_reshapes(
        session_id="perf_test",
        get_top_hlo_ops_fn=lambda s, limit=75: _top_ops_json(ops),
        fetch_debug_info_fn=lambda s: debug_info,
    )
    result = json.loads(result_str)
    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 2)

  def test_proto_based_analysis_unnamed_module_no_false_match(self):
    # Module 1 (unnamed module, does not match candidate)
    p0 = _make_instr(1, "p0", "parameter")
    # Instruction name matches candidate, but opcode is not formatting
    add1 = _make_instr(2, "reshape.1", "add", operands=[1])
    comp1 = types.SimpleNamespace(id=1, name="main", instructions=[p0, add1])
    mod1 = types.SimpleNamespace(name="", computations=[comp1])

    # Module 2 (named module, matches candidate correctly)
    p1 = _make_instr(10, "p1", "parameter")
    reshape1 = _make_instr(11, "reshape.1", "reshape", operands=[10])
    dot1 = _make_instr(12, "dot.1", "dot", operands=[11])
    comp2 = types.SimpleNamespace(
        id=2, name="main", instructions=[p1, reshape1, dot1]
    )
    mod2 = types.SimpleNamespace(name="target_module", computations=[comp2])

    debug_info = types.SimpleNamespace(
        hlo_proto=[
            types.SimpleNamespace(hlo_module=mod1),
            types.SimpleNamespace(hlo_module=mod2),
        ],
        program_id=[None, None],
    )

    ops = [_op("by_program/target_module/reshape.1", "reshape", 20 * _MB)]
    result_str = detect_unfused_reshapes_tool.detect_unfused_reshapes(
        "s",
        get_top_hlo_ops_fn=lambda s, limit: _top_ops_json(ops),
        fetch_debug_info_fn=lambda s: debug_info,
    )
    result = json.loads(result_str)
    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)
    self.assertEqual(result["inefficient_ops"][0]["downstream_compute"], "dot")


if __name__ == "__main__":
  absltest.main()
