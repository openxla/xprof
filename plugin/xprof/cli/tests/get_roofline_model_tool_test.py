import json
from unittest import mock
from absl.testing import absltest
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_roofline_model_tool


class GetRooflineModelToolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
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

    self.assertEqual(parsed["device_info"]["device_type"], "TPU v6 Lite")
    self.assertEqual(parsed["device_info"]["peak_flop_rate"], 946700.0)

    self.assertLen(parsed["top_operations"], 1)
    op = parsed["top_operations"][0]
    self.assertEqual(op["name"], "psum.118")
    self.assertEqual(op["category"], "all-reduce")
    self.assertEqual(op["total_self_time_ms"], 50.0)
    self.assertEqual(op["bound_by"], "HBM")
    self.assertEqual(op["roofline_efficiency_percent"], "16.31%")
    self.assertEqual(op["source_info"], "file.py:10")

  def test_get_roofline_model_no_data(self):
    self.mock_client.fetch.return_value = ("application/json", None)
    result = get_roofline_model_tool.get_roofline_model("test_session")
    parsed = json.loads(result)
    self.assertEqual(parsed["status"], "NO_DATA")


if __name__ == "__main__":
  absltest.main()
