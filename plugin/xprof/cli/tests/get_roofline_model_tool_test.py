import json
from typing import Any
from unittest import mock
from absl.testing import absltest
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_roofline_model_tool


class GetRooflineModelToolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_cache = mock.create_autospec(
        decorators.Cache, instance=True, spec_set=True
    )
    mock_cache.get.return_value = decorators.Cache.UNKNOWN
    self.enter_context(
        mock.patch.object(
            decorators,
            "get_cache",
            return_value=mock_cache,
            autospec=True,
            spec_set=True,
        )
    )
    self.mock_client = mock.create_autospec(xprof_client.CachedXprofClient)
    self.enter_context(
        mock.patch.object(
            xprof_client,
            "get_client",
            return_value=self.mock_client,
            autospec=True,
        )
    )

  def test_get_roofline_model_success(self):
    roofline_raw_data = [{
        "cols": [
            {"id": "step", "type": "string"},
            {"id": "rank", "type": "number"},
            {"id": "category", "type": "string"},
            {"id": "operation", "type": "string"},
            {"id": "occurrences", "type": "number"},
            {"id": "total_time", "type": "number"},
            {"id": "total_self_time", "type": "number"},
            {"id": "total_self_time_percent", "type": "number"},
            {"id": "measured_flop_rate", "type": "number"},
            {"id": "model_flop_rate", "type": "number"},
            {"id": "measured_memory_bw", "type": "number"},
            {"id": "hbm_bw", "type": "number"},
            {"id": "operational_intensity", "type": "number"},
            {"id": "bound_by", "type": "string"},
            {"id": "roofline_efficiency", "type": "number"},
            {"id": "compute_efficiency", "type": "number"},
            {"id": "max_mem_bw_utilization", "type": "number"},
            {"id": "hlo_module_id", "type": "string"},
            {"id": "source_info", "type": "string"},
        ],
        "p": {
            "device_type": "TPU v6 Lite",
            "peak_flop_rate": "946700",
            "peak_hbm_bw": "1525.5",
            "hbm_ridge_point": "577.96",
        },
        "rows": [
            {
                "c": [
                    {"v": "Total"},
                    {"v": 0.0},
                    {"v": "Program"},
                    {"v": "Program"},
                    {"v": 1.0},
                    {"v": 100000.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": 7181.72},
                    {"v": 6440.89},
                    {"v": 258.31},
                    {"v": 205.79},
                    {"v": 25.89},
                    {"v": "HBM"},
                    {"v": 0.1349},
                    {"v": 0.0076},
                    {"v": 0.1349},
                    {"v": "0"},
                    {"v": ""},
                ]
            },
            {
                "c": [
                    {"v": "Total"},
                    {"v": 1.0},
                    {"v": "all-reduce"},
                    {"v": "psum.118"},
                    {"v": 10.0},
                    {"v": 50000.0},
                    {"v": 50000.0},
                    {"v": 0.50},
                    {"v": 100.0},
                    {"v": 90.0},
                    {"v": 200.0},
                    {"v": 180.0},
                    {"v": 0.25},
                    {"v": "HBM"},
                    {"v": 0.1631},
                    {"v": 0.0001},
                    {"v": 0.1631},
                    {"v": "12345"},
                    {"v": "<div title='file.py:10'>file.py:10</div>"},
                ]
            },
        ],
    }]

    roofline_json = json.dumps(roofline_raw_data).encode("utf-8")
    self.mock_client.fetch.return_value = ("application/json", roofline_json)

    result = get_roofline_model_tool.get_roofline_model("test_session", top_n=5)
    parsed = json.loads(result)

    self.assertEqual(parsed["program"]["bound_by"], "HBM")
    self.assertEqual(parsed["program"]["roofline_efficiency_percent"], "13.49%")
    self.assertEqual(parsed["program"]["compute_efficiency_percent"], "0.76%")
    self.assertEqual(
        parsed["program"]["max_mem_bw_utilization_percent"], "13.49%"
    )
    self.assertEqual(
        parsed["program"]["operational_intensity_flop_per_byte"], 25.89
    )
    self.assertEqual(parsed["program"]["measured_flop_rate_gflops"], 7181.72)
    self.assertEqual(parsed["program"]["hbm_bw_gibs"], 205.79)
    self.assertEqual(parsed["program"]["flops_provenance"], "xla_cost_model")
    self.assertEqual(parsed["program"]["derived_flops_fraction"], 0.0)
    self.assertFalse(parsed["program"]["has_opaque_custom_calls"])

    self.assertEqual(parsed["device_info"]["device_type"], "TPU v6 Lite")
    self.assertEqual(parsed["device_info"]["peak_flop_rate"], 946700.0)
    self.assertEqual(parsed["device_info"]["peak_hbm_bw_gibs"], 1525.5)
    self.assertNotIn("peak_hbm_bw", parsed["device_info"])
    self.assertEqual(
        parsed["device_info"]["units"]["peak_hbm_bw_gibs"], "GiB/s"
    )
    self.assertNotIn("peak_hbm_bw", parsed["device_info"]["units"])

    self.assertLen(parsed["top_operations"], 1)
    op = parsed["top_operations"][0]
    self.assertEqual(op["name"], "psum.118")
    self.assertEqual(op["category"], "all-reduce")
    self.assertEqual(op["total_self_time_ms"], 50.0)
    self.assertEqual(op["bound_by"], "HBM")
    self.assertEqual(op["roofline_efficiency_percent"], "16.31%")
    self.assertEqual(op["source_info"], "file.py:10")

  def test_get_roofline_model_custom_call_opaque(self):
    roofline_raw_data = [{
        "cols": [
            {"id": "step", "type": "string"},
            {"id": "rank", "type": "number"},
            {"id": "category", "type": "string"},
            {"id": "operation", "type": "string"},
            {"id": "occurrences", "type": "number"},
            {"id": "total_time", "type": "number"},
            {"id": "total_self_time", "type": "number"},
            {"id": "total_self_time_percent", "type": "number"},
            {"id": "measured_flop_rate", "type": "number"},
            {"id": "model_flop_rate", "type": "number"},
            {"id": "measured_memory_bw", "type": "number"},
            {"id": "hbm_bw", "type": "number"},
            {"id": "operational_intensity", "type": "number"},
            {"id": "bound_by", "type": "string"},
            {"id": "roofline_efficiency", "type": "number"},
            {"id": "compute_efficiency", "type": "number"},
            {"id": "max_mem_bw_utilization", "type": "number"},
            {"id": "hlo_module_id", "type": "string"},
            {"id": "source_info", "type": "string"},
        ],
        "p": {
            "device_type": "TPU v6 Lite",
            "peak_flop_rate": "946700",
            "peak_hbm_bw": "1525.5",
            "hbm_ridge_point": "577.96",
        },
        "rows": [
            {
                "c": [
                    {"v": "Total"},
                    {"v": 0.0},
                    {"v": "Program"},
                    {"v": "Program"},
                    {"v": 1.0},
                    {"v": 100000.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": "Unknown"},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": "0"},
                    {"v": ""},
                ]
            },
            {
                "c": [
                    {"v": "Total"},
                    {"v": 1.0},
                    {"v": "custom-call"},
                    {"v": "custom-call.1"},
                    {"v": 1.0},
                    {"v": 50000.0},
                    {"v": 50000.0},
                    {"v": 1.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": "Unknown"},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": "12345"},
                    {"v": ""},
                ]
            },
        ],
    }]

    roofline_json = json.dumps(roofline_raw_data).encode("utf-8")
    self.mock_client.fetch.return_value = ("application/json", roofline_json)

    result = get_roofline_model_tool.get_roofline_model("test_session", top_n=5)
    parsed = json.loads(result)

    self.assertEqual(parsed["program"]["bound_by"], "CustomCall (opaque)")
    self.assertEqual(
        parsed["program"]["flops_provenance"], "opaque_custom_call"
    )
    self.assertEqual(parsed["program"]["derived_flops_fraction"], 0.0)
    self.assertTrue(parsed["program"]["has_opaque_custom_calls"])
    self.assertLen(parsed["top_operations"], 1)
    self.assertEqual(
        parsed["top_operations"][0]["bound_by"], "CustomCall (opaque)"
    )
    self.assertEqual(parsed["custom_call_share_pct"], 100.0)
    self.assertIn("custom_call_warning", parsed)
    self.assertIn("0 FLOPs", parsed["custom_call_warning"])
    self.assertIn("guidance", parsed)
    self.assertIn("Pallas kernels", parsed["guidance"])

  def test_get_roofline_model_no_data(self):
    self.mock_client.fetch.return_value = ("application/json", None)
    result = get_roofline_model_tool.get_roofline_model("test_session")
    parsed = json.loads(result)
    self.assertEqual(parsed["status"], "NO_DATA")

  def test_get_roofline_model_invalid_data(self):
    self.mock_client.fetch.return_value = ("application/json", b"{}")
    with self.assertRaises(ValueError):
      get_roofline_model_tool.get_roofline_model("test_session")

  def test_get_roofline_model_bypass_cache(self):
    self.mock_client.fetch.return_value = ("application/json", None)
    get_roofline_model_tool.get_roofline_model(
        "test_session", bypass_cache=True
    )
    self.mock_client.fetch.assert_any_call(
        tool_name="roofline_model.json",
        session_id="test_session",
        bypass_cache=True,
    )

  def test_get_roofline_model_derives_custom_call_flops_from_shapes(self):
    roofline_raw_data = [{
        "cols": [
            {"id": "step_id", "type": "string"},
            {"id": "rank", "type": "number"},
            {"id": "category", "type": "string"},
            {"id": "operation", "type": "string"},
            {"id": "occurrences", "type": "number"},
            {"id": "total_time", "type": "number"},
            {"id": "total_self_time", "type": "number"},
            {"id": "total_self_time_percent", "type": "number"},
            {"id": "measured_flop_rate", "type": "number"},
            {"id": "model_flop_rate", "type": "number"},
            {"id": "measured_memory_bw", "type": "number"},
            {"id": "hbm_bw", "type": "number"},
            {"id": "operational_intensity", "type": "number"},
            {"id": "bound_by", "type": "string"},
            {"id": "roofline_efficiency", "type": "number"},
            {"id": "compute_efficiency", "type": "number"},
            {"id": "max_mem_bw_utilization", "type": "number"},
            {"id": "hlo_module_id", "type": "string"},
            {"id": "source_info", "type": "string"},
        ],
        "p": {
            "device_type": "TPU v7x",
            "peak_flop_rate": "918000",
            "peak_hbm_bw": "1638.0",
            "hbm_ridge_point": "560.44",
        },
        "rows": [
            {
                "c": [
                    {"v": "Total"},
                    {"v": 0.0},
                    {"v": "Program"},
                    {"v": "Program"},
                    {"v": 1.0},
                    {"v": 396.32},
                    {"v": 396.32},
                    {"v": 1.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": 77.15},
                    {"v": 77.15},
                    {"v": 0.0},
                    {"v": "HBM"},
                    {"v": 0.0471},
                    {"v": 0.0},
                    {"v": 0.0471},
                    {"v": "0"},
                    {"v": ""},
                ]
            },
            {
                "c": [
                    {"v": "Total"},
                    {"v": 1.0},
                    {"v": "custom-call"},
                    {"v": "custom-call.1"},
                    {"v": 1.0},
                    {"v": 396.32},
                    {"v": 396.32},
                    {"v": 1.0},
                    {"v": 0.0},
                    {"v": 0.0},
                    {"v": 77.15},
                    {"v": 77.15},
                    {"v": 0.0},
                    {"v": "Unknown"},
                    {"v": 0.0471},
                    {"v": 0.0},
                    {"v": 0.0471},
                    {"v": "12345"},
                    {"v": ""},
                ]
            },
        ],
    }]
    op_profile_json = json.dumps({
        "by_program": {
            "name": "main",
            "children": [{
                "name": "custom-call.1",
                "xla": {
                    "expression": (
                        "%custom-call.1 = bf16[4,4096,4096] custom-call("
                        "bf16[4,4096,2048] %p0, bf16[2048,4096] %p1)"
                    )
                },
            }],
        }
    }).encode("utf-8")

    def fake_fetch(tool_name, session_id, bypass_cache=False):
      del session_id, bypass_cache
      if tool_name == "roofline_model.json":
        return (
            "application/json",
            json.dumps(roofline_raw_data).encode("utf-8"),
        )
      if tool_name == "hlo_op_profile.json":
        return ("application/json", op_profile_json)
      return ("application/json", None)

    self.mock_client.fetch.side_effect = fake_fetch

    result = get_roofline_model_tool.get_roofline_model(
        "test_session", bypass_cache=True
    )
    parsed = json.loads(result)

    self.assertEqual(
        parsed["program"]["flops_provenance"], "derived_from_shapes"
    )
    self.assertEqual(parsed["program"]["bound_by"], "Compute")
    self.assertEqual(parsed["program"]["compute_efficiency_percent"], "75.55%")
    self.assertEqual(parsed["program"]["roofline_efficiency_percent"], "75.55%")
    self.assertLen(parsed["top_operations"], 1)
    op = parsed["top_operations"][0]
    self.assertEqual(op["flops_provenance"], "derived_from_shapes")
    self.assertEqual(op["bound_by"], "Compute")
    self.assertEqual(op["compute_efficiency_percent"], "75.55%")
    self.assertEqual(op["roofline_efficiency_percent"], "75.55%")
    self.assertEqual(parsed["program"]["derived_flops_fraction"], 1.0)

  def _mock_roofline_session(
      self,
      program_row: dict[str, Any],
      op_rows: list[dict[str, Any]],
  ) -> None:
    """Mocks a session from rows keyed by roofline column id.

    Only custom-call.1 has an op_profile expression. It computes
    bf16[4,4096,2048] x bf16[2048,4096], which is 2 * 4 * 4096 * 4096 * 2048
    = 2**38 shape-derived FLOPs. Any other custom call is opaque.

    Args:
      program_row: Program row values by column id. Missing columns are None.
      op_rows: Per-op row values by column id. Missing columns are None.
    """
    cols = [
        "step_id",
        "rank",
        "category",
        "operation",
        "total_time",
        "total_self_time",
        "total_self_time_percent",
        "measured_flop_rate",
        "model_flop_rate",
        "measured_memory_bw",
        "hbm_bw",
        "operational_intensity",
        "hbm_operational_intensity",
        "bottleneck_operational_intensity",
        "bound_by",
        "compute_efficiency",
        "max_mem_bw_utilization",
        "optimal_flop_rate",
    ]

    def row(values: dict[str, Any]) -> dict[str, Any]:
      return {"c": [{"v": values.get(c)} for c in cols]}

    program = {"step_id": "Total", "rank": 0.0, "category": "Program"}
    program["operation"] = "Program"
    program.update(program_row)
    roofline_raw_data = [{
        "cols": [{"id": c} for c in cols],
        "p": {
            "device_type": "TPU v6 Lite",
            "peak_flop_rate": "946700.0",
            "peak_hbm_bw": "1525.5",
            "hbm_ridge_point": "577.963",
        },
        "rows": (
            [row(program)] + [row({"step_id": "Total", **op}) for op in op_rows]
        ),
    }]
    op_profile_json = json.dumps({
        "by_program": {
            "name": "main",
            "children": [{
                "name": "custom-call.1",
                "xla": {
                    "expression": (
                        "%custom-call.1 = bf16[4,4096,4096] custom-call("
                        "bf16[4,4096,2048] %p0, bf16[2048,4096] %p1)"
                    )
                },
            }],
        }
    }).encode("utf-8")

    def fake_fetch(tool_name, session_id, bypass_cache=False):
      del session_id, bypass_cache
      if tool_name == "roofline_model.json":
        return (
            "application/json",
            json.dumps(roofline_raw_data).encode("utf-8"),
        )
      if tool_name == "hlo_op_profile.json":
        return ("application/json", op_profile_json)
      return ("application/json", None)

    self.mock_client.fetch.side_effect = fake_fetch

  def _mock_derived_custom_call_session(
      self,
      prog_total_time_us: float,
      prog_flop_rate: float,
      prog_mem_bw_gibs: float,
      custom_call_flop_rate: float,
      *,
      prog_model_flop_rate: float = 0.0,
      custom_call_model_flop_rate: float = 0.0,
      prog_operational_intensity: float = 0.0,
  ) -> None:
    """Mocks a session whose only custom call has shape-derived FLOPs.

    Args:
      prog_total_time_us: Program total time in microseconds.
      prog_flop_rate: Program measured FLOP rate in GFLOP/s.
      prog_mem_bw_gibs: Program measured memory bandwidth in GiB/s.
      custom_call_flop_rate: XLA cost-model measured FLOP rate of the custom
        call in GFLOP/s.
      prog_model_flop_rate: Program model FLOP rate in GFLOP/s.
      custom_call_model_flop_rate: Model FLOP rate of the custom call in
        GFLOP/s.
      prog_operational_intensity: Backend program operational intensity.
    """
    custom_call_time_us = 396.32
    self._mock_roofline_session(
        program_row={
            "total_time": prog_total_time_us,
            "total_self_time": prog_total_time_us,
            "total_self_time_percent": 1.0,
            "measured_flop_rate": prog_flop_rate,
            "model_flop_rate": prog_model_flop_rate,
            "measured_memory_bw": prog_mem_bw_gibs,
            "hbm_bw": 752.54,
            "operational_intensity": prog_operational_intensity,
            "bound_by": "HBM",
            "compute_efficiency": 0.0,
            "max_mem_bw_utilization": 0.4933,
        },
        op_rows=[{
            "rank": 1.0,
            "category": "custom-call",
            "operation": "custom-call.1",
            "total_time": custom_call_time_us,
            "total_self_time": custom_call_time_us,
            "total_self_time_percent": (
                custom_call_time_us / prog_total_time_us
            ),
            "measured_flop_rate": custom_call_flop_rate,
            "model_flop_rate": custom_call_model_flop_rate,
            "measured_memory_bw": 77.15,
            "hbm_bw": 77.15,
            "bound_by": "Unknown",
            "compute_efficiency": 0.0,
            "max_mem_bw_utilization": 0.0471,
        }],
    )

  def _get_program(self, top_n: int = 15) -> dict[str, Any]:
    return json.loads(
        get_roofline_model_tool.get_roofline_model(
            "test_session", top_n=top_n, bypass_cache=True
        )
    )["program"]

  def test_program_operational_intensity_converts_gibs_to_gb(self):
    # Pick the cost-model remainder so the recomputed program rate is
    # 312,992.93 GFLOP/s over 1000 us: 312992.93e6 - 2**38 FLOPs.
    cost_model_flops = 312992.93e6 - 2**38
    self._mock_derived_custom_call_session(
        prog_total_time_us=1000.0,
        prog_flop_rate=cost_model_flops / 1e6,
        prog_mem_bw_gibs=1011.98,
        custom_call_flop_rate=0.0,
    )

    parsed = json.loads(
        get_roofline_model_tool.get_roofline_model(
            "test_session", bypass_cache=True
        )
    )

    program = parsed["program"]
    self.assertAlmostEqual(
        program["measured_flop_rate_gflops"], 312992.93, places=2
    )
    # 312992.93 GFLOP/s / (1011.98 GiB/s * 2**30 / 1e9) = 288.04 FLOP/byte.
    # Dividing by GiB/s directly would give 309.29 (7.37 % too high).
    self.assertAlmostEqual(
        program["operational_intensity_flop_per_byte"], 288.04, delta=0.01
    )

  def test_program_flops_provenance_is_mixed_with_cost_model_remainder(self):
    cost_model_flops = 312992.93e6 - 2**38
    self._mock_derived_custom_call_session(
        prog_total_time_us=1000.0,
        prog_flop_rate=cost_model_flops / 1e6,
        prog_mem_bw_gibs=1011.98,
        custom_call_flop_rate=0.0,
    )

    parsed = json.loads(
        get_roofline_model_tool.get_roofline_model(
            "test_session", bypass_cache=True
        )
    )

    program = parsed["program"]
    self.assertEqual(program["flops_provenance"], "mixed")
    self.assertAlmostEqual(
        program["derived_flops_fraction"], 2**38 / 312992.93e6, places=4
    )
    self.assertEqual(
        parsed["top_operations"][0]["flops_provenance"], "derived_from_shapes"
    )

  def test_program_flops_provenance_is_derived_when_all_flops_replaced(self):
    # The custom call carries all of the program's cost-model FLOPs, so
    # replacing them leaves no cost-model remainder.
    self._mock_derived_custom_call_session(
        prog_total_time_us=396.32,
        prog_flop_rate=1234.5,
        prog_mem_bw_gibs=77.15,
        custom_call_flop_rate=1234.5,
    )

    parsed = json.loads(
        get_roofline_model_tool.get_roofline_model(
            "test_session", bypass_cache=True
        )
    )

    program = parsed["program"]
    self.assertEqual(program["flops_provenance"], "derived_from_shapes")
    self.assertEqual(program["derived_flops_fraction"], 1.0)

  def test_cost_model_remainder_just_under_tolerance_is_derived(self):
    # The remainder is 0.999e-3 of the program FLOPs, inside the 1e-3
    # tolerance, so it is clamped to 0 before the totals are computed.
    self._mock_derived_custom_call_session(
        prog_total_time_us=396.32,
        prog_flop_rate=1000.0,
        prog_mem_bw_gibs=77.15,
        custom_call_flop_rate=1000.0 * (1.0 - 0.999e-3),
    )

    program = self._get_program()

    self.assertEqual(program["flops_provenance"], "derived_from_shapes")
    self.assertEqual(program["derived_flops_fraction"], 1.0)
    self.assertAlmostEqual(
        program["measured_flop_rate_gflops"], 2**38 / 396.32e3, places=2
    )

  def test_cost_model_remainder_just_over_tolerance_is_mixed(self):
    self._mock_derived_custom_call_session(
        prog_total_time_us=396.32,
        prog_flop_rate=1000.0,
        prog_mem_bw_gibs=77.15,
        custom_call_flop_rate=1000.0 * (1.0 - 1.001e-3),
    )

    program = self._get_program()

    remainder = 1000.0 * 1.001e-3 * 396.32e3
    self.assertEqual(program["flops_provenance"], "mixed")
    self.assertAlmostEqual(
        program["derived_flops_fraction"],
        2**38 / (2**38 + remainder),
        places=4,
    )

  def test_replaced_flops_use_program_rate_column(self):
    # The program has measured FLOPs, so the custom call's replaced FLOPs
    # must also come from measured_flop_rate (0 here), not from its
    # model_flop_rate. All program measured FLOPs then belong to other ops.
    self._mock_derived_custom_call_session(
        prog_total_time_us=396.32,
        prog_flop_rate=1000.0,
        prog_mem_bw_gibs=77.15,
        custom_call_flop_rate=0.0,
        prog_model_flop_rate=1000.0,
        custom_call_model_flop_rate=1000.0,
    )

    program = self._get_program()

    cost_model_flops = 1000.0 * 396.32e3
    self.assertEqual(program["flops_provenance"], "mixed")
    self.assertAlmostEqual(
        program["measured_flop_rate_gflops"],
        (cost_model_flops + 2**38) / 396.32e3,
        places=2,
    )

  def test_program_without_measured_flops_uses_model_column(self):
    self._mock_derived_custom_call_session(
        prog_total_time_us=396.32,
        prog_flop_rate=0.0,
        prog_mem_bw_gibs=77.15,
        custom_call_flop_rate=0.0,
        prog_model_flop_rate=1000.0,
        custom_call_model_flop_rate=500.0,
    )

    program = self._get_program()

    cost_model_flops = 500.0 * 396.32e3
    self.assertEqual(program["flops_provenance"], "mixed")
    self.assertAlmostEqual(
        program["derived_flops_fraction"],
        2**38 / (2**38 + cost_model_flops),
        places=4,
    )

  def test_opaque_and_derived_custom_calls_are_mixed_for_any_top_n(self):
    self._mock_roofline_session(
        program_row={
            "total_time": 1000.0,
            "measured_flop_rate": 0.0,
            "measured_memory_bw": 77.15,
            "hbm_bw": 77.15,
            "bound_by": "HBM",
            "max_mem_bw_utilization": 0.0471,
        },
        op_rows=[
            {
                "rank": 1.0,
                "category": "custom-call",
                "operation": "custom-call.1",
                "total_self_time": 600.0,
                "total_self_time_percent": 0.6,
                "bound_by": "Unknown",
                "max_mem_bw_utilization": 0.0471,
            },
            {
                "rank": 2.0,
                "category": "custom-call",
                "operation": "custom-call.2",
                "total_self_time": 400.0,
                "total_self_time_percent": 0.4,
                "bound_by": "Unknown",
            },
        ],
    )

    program_top_1 = self._get_program(top_n=1)
    program_top_5 = self._get_program(top_n=5)

    self.assertEqual(program_top_1["flops_provenance"], "mixed")
    self.assertTrue(program_top_1["has_opaque_custom_calls"])
    self.assertEqual(program_top_1, program_top_5)

  def test_bottleneck_intensity_recomputed_when_bound_flips_to_hbm(self):
    prog_time_us = 1000.0
    # The custom call carries 5e11 of the program's 6e11 cost-model FLOPs;
    # replacing them with 2**38 shape-derived FLOPs drops compute efficiency
    # below the 49.33% HBM utilization.
    self._mock_roofline_session(
        program_row={
            "total_time": prog_time_us,
            "measured_flop_rate": 600000.0,
            "measured_memory_bw": 1011.98,
            "hbm_bw": 752.54,
            "operational_intensity": 555.0,
            "hbm_operational_intensity": 745.0,
            "bottleneck_operational_intensity": 555.0,
            "bound_by": "Compute",
            "compute_efficiency": 600000.0 / 946700.0,
            "max_mem_bw_utilization": 0.4933,
            "optimal_flop_rate": 946700.0,
        },
        op_rows=[{
            "rank": 1.0,
            "category": "custom-call",
            "operation": "custom-call.1",
            "total_self_time": 396.32,
            "total_self_time_percent": 0.39632,
            "measured_flop_rate": 5e11 / 396.32e3,
            "measured_memory_bw": 77.15,
            "hbm_bw": 77.15,
            "bottleneck_operational_intensity": 123.0,
            "bound_by": "Compute",
            "max_mem_bw_utilization": 0.0471,
            "optimal_flop_rate": 999.0,
        }],
    )

    parsed = json.loads(
        get_roofline_model_tool.get_roofline_model(
            "test_session", bypass_cache=True
        )
    )
    program = parsed["program"]

    new_rate = (1e11 + 2**38) / (prog_time_us * 1e3)
    peak_hbm_gb = 1525.5 * 2**30 / 1e9
    self.assertEqual(program["bound_by"], "HBM")
    self.assertEqual(program["bound_by_ridge_point_flop_per_byte"], 577.963)
    self.assertAlmostEqual(
        program["bottleneck_operational_intensity_flop_per_byte"],
        new_rate / (752.54 * 2**30 / 1e9),
        places=3,
    )
    self.assertLess(
        program["bottleneck_operational_intensity_flop_per_byte"],
        program["bound_by_ridge_point_flop_per_byte"],
    )
    self.assertAlmostEqual(
        program["optimal_flop_rate_gflops"], new_rate / 0.4933, places=1
    )
    # optimal_flop_rate = bottleneck_OI x peak_hbm_bw_GB, to within the
    # rounding of max_mem_bw_utilization (752.54 / 1525.5 = 0.49331).
    self.assertAlmostEqual(
        program["optimal_flop_rate_gflops"]
        / (
            program["bottleneck_operational_intensity_flop_per_byte"]
            * peak_hbm_gb
        ),
        1.0,
        places=3,
    )
    op = parsed["top_operations"][0]
    self.assertEqual(op["bound_by"], "Compute")
    self.assertEqual(
        op["bottleneck_operational_intensity_flop_per_byte"],
        op["operational_intensity_flop_per_byte"],
    )
    self.assertAlmostEqual(op["optimal_flop_rate_gflops"], 946700.0, places=1)

  def test_intensity_fallback_scales_backend_value_without_memory_bw(self):
    self._mock_derived_custom_call_session(
        prog_total_time_us=1000.0,
        prog_flop_rate=1000.0,
        prog_mem_bw_gibs=0.0,
        custom_call_flop_rate=500.0,
        prog_operational_intensity=100.0,
    )

    program = self._get_program()

    orig_flops = 1000.0 * 1000.0e3
    new_flops = orig_flops - 500.0 * 396.32e3 + 2**38
    self.assertEqual(program["flops_provenance"], "mixed")
    self.assertAlmostEqual(
        program["operational_intensity_flop_per_byte"],
        100.0 * new_flops / orig_flops,
        places=3,
    )

  def test_intensity_fallback_uses_derived_bytes_when_fully_derived(self):
    self._mock_derived_custom_call_session(
        prog_total_time_us=396.32,
        prog_flop_rate=0.0,
        prog_mem_bw_gibs=0.0,
        custom_call_flop_rate=0.0,
    )

    program = self._get_program()

    # Operand and result bytes: 2 * (4*4096*2048 + 2048*4096 + 4*4096*4096).
    derived_bytes = 2 * (4 * 4096 * 2048 + 2048 * 4096 + 4 * 4096 * 4096)
    self.assertEqual(program["flops_provenance"], "derived_from_shapes")
    self.assertAlmostEqual(
        program["operational_intensity_flop_per_byte"],
        2**38 / derived_bytes,
        places=3,
    )

  def test_intensity_fallback_keeps_backend_value_when_mixed(self):
    # Mixed program, no memory bandwidth and no backend intensity: the
    # custom call's bytes are not the program bytes, so keep the backend 0.
    self._mock_derived_custom_call_session(
        prog_total_time_us=1000.0,
        prog_flop_rate=1000.0,
        prog_mem_bw_gibs=0.0,
        custom_call_flop_rate=0.0,
    )

    program = self._get_program()

    self.assertEqual(program["flops_provenance"], "mixed")
    self.assertEqual(program["operational_intensity_flop_per_byte"], 0.0)

  def test_replaced_flops_use_total_time_when_different_from_self_time(self):
    # Backend measured_flop_rate divides FLOPs by total_time (1000 us), not
    # total_self_time (400 us). When the custom call carries all program FLOPs,
    # multiplying by total_time recovers the exact program FLOPs so no phantom
    # cost_model_flops remainder appears.
    self._mock_roofline_session(
        program_row={
            "total_time": 1000.0,
            "total_self_time": 1000.0,
            "total_self_time_percent": 1.0,
            "measured_flop_rate": 500.0,
            "measured_memory_bw": 100.0,
            "hbm_bw": 100.0,
            "bound_by": "HBM",
            "max_mem_bw_utilization": 0.1,
        },
        op_rows=[{
            "rank": 1.0,
            "category": "custom-call",
            "operation": "custom-call.1",
            "total_time": 1000.0,
            "total_self_time": 400.0,
            "total_self_time_percent": 0.4,
            "measured_flop_rate": 500.0,
            "measured_memory_bw": 100.0,
            "hbm_bw": 100.0,
            "bound_by": "Unknown",
            "max_mem_bw_utilization": 0.1,
        }],
    )

    program = self._get_program()

    self.assertEqual(program["flops_provenance"], "derived_from_shapes")
    self.assertEqual(program["derived_flops_fraction"], 1.0)

  def test_negative_total_time_falls_back_to_self_time(self):
    # A malformed negative total_time must not produce negative replaced
    # FLOPs; total_self_time is used instead, so the custom call's replaced
    # FLOPs equal the program FLOPs and leave no cost-model remainder.
    self._mock_roofline_session(
        program_row={
            "total_time": 396.32,
            "total_self_time": 396.32,
            "total_self_time_percent": 1.0,
            "measured_flop_rate": 1000.0,
            "measured_memory_bw": 77.15,
            "hbm_bw": 77.15,
            "bound_by": "HBM",
            "max_mem_bw_utilization": 0.0471,
        },
        op_rows=[{
            "rank": 1.0,
            "category": "custom-call",
            "operation": "custom-call.1",
            "total_time": -5.0,
            "total_self_time": 396.32,
            "total_self_time_percent": 1.0,
            "measured_flop_rate": 1000.0,
            "measured_memory_bw": 77.15,
            "hbm_bw": 77.15,
            "bound_by": "Unknown",
            "max_mem_bw_utilization": 0.0471,
        }],
    )

    program = self._get_program()

    self.assertEqual(program["flops_provenance"], "derived_from_shapes")
    self.assertEqual(program["derived_flops_fraction"], 1.0)
    self.assertAlmostEqual(
        program["measured_flop_rate_gflops"], 2**38 / 396.32e3, places=2
    )

  def test_custom_call_share_matches_name_case_and_expression(self):
    self._mock_roofline_session(
        program_row={
            "total_time": 1000.0,
            "total_self_time": 1000.0,
            "total_self_time_percent": 1.0,
            "bound_by": "HBM",
        },
        op_rows=[
            {
                "rank": 1.0,
                "category": "other",
                "operation": "my_flash_attn",
                "total_time": 300.0,
                "total_self_time": 300.0,
                "total_self_time_percent": 0.3,
                "bound_by": "Unknown",
            },
            {
                "rank": 2.0,
                "category": "other",
                "operation": "CUSTOM_CALL.3",
                "total_time": 200.0,
                "total_self_time": 200.0,
                "total_self_time_percent": 0.2,
                "bound_by": "Unknown",
            },
            {
                "rank": 3.0,
                "category": "loop fusion",
                "operation": "fusion.1",
                "total_time": 500.0,
                "total_self_time": 500.0,
                "total_self_time_percent": 0.5,
                "bound_by": "HBM",
            },
        ],
    )
    self.enter_context(
        mock.patch.object(
            get_roofline_model_tool,
            "_fetch_op_profile_expressions",
            autospec=True,
            return_value={
                "my_flash_attn": (
                    "%my_flash_attn = f32[8] custom-call(f32[8] %p0)"
                ),
            },
        )
    )

    parsed = json.loads(
        get_roofline_model_tool.get_roofline_model(
            "test_session", bypass_cache=True
        )
    )

    self.assertEqual(parsed["custom_call_share_pct"], 50.0)
    self.assertIn("custom_call_warning", parsed)

  def test_is_custom_call_op(self):
    self.assertTrue(get_roofline_model_tool.is_custom_call_op("%custom-call.5"))
    self.assertTrue(get_roofline_model_tool.is_custom_call_op("CUSTOM_CALL.3"))
    self.assertTrue(
        get_roofline_model_tool.is_custom_call_op("my_custom_call_kernel")
    )
    self.assertTrue(
        get_roofline_model_tool.is_custom_call_op("tpu_kernel", "Custom-Call")
    )
    self.assertTrue(
        get_roofline_model_tool.is_custom_call_op(
            "my_flash_attn",
            "other",
            "%my_flash_attn = f32[8] custom-call(f32[8] %p0)",
        )
    )
    self.assertFalse(
        get_roofline_model_tool.is_custom_call_op(
            "fusion.1", "loop fusion", "%fusion.1 = f32[8] fusion(%p0)"
        )
    )


if __name__ == "__main__":
  absltest.main()
