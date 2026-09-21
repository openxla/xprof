"""Unit tests for get_perf_counters_tool."""

import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_perf_counters_tool


class GetPerfCountersToolTest(parameterized.TestCase):

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
        )
    )
    self.mock_client = mock.create_autospec(
        xprof_client.CachedXprofClient, instance=True
    )
    self.enter_context(
        mock.patch.object(
            xprof_client,
            "get_client",
            return_value=self.mock_client,
            autospec=True,
        )
    )

  def _make_datatable_json(self) -> str:
    table = {
        "cols": [
            {"id": "Host", "label": "Host", "type": "string"},
            {"id": "Chip", "label": "Chip", "type": "number"},
            {"id": "Kernel", "label": "Kernel", "type": "string"},
            {"id": "Sample", "label": "Sample", "type": "number"},
            {"id": "Counter", "label": "Counter", "type": "string"},
            {"id": "Value", "label": "Value (Hex)", "type": "number"},
            {"id": "Description", "label": "Description", "type": "string"},
            {"id": "Set", "label": "Set", "type": "string"},
        ],
        "rows": [
            {
                "c": [
                    {"v": "host0"},
                    {"v": 0},
                    {"v": "fusion.1"},
                    {"v": 1},
                    {"v": "mxu_busy_cycles"},
                    {"v": "0x1000"},
                    {"v": "MXU busy cycles"},
                    {"v": "TPU_TC_SET"},
                ]
            },
            {
                "c": [
                    {"v": "host1"},
                    {"v": 1},
                    {"v": "fusion.2"},
                    {"v": 2},
                    {"v": "hbm_read_bytes"},
                    {"v": "0x2000"},
                    {"v": "HBM read bytes"},
                    {"v": "TPU_CMN_SET"},
                ]
            },
            {
                "c": [
                    {"v": "host0"},
                    {"v": 0},
                    {"v": "fusion.1"},
                    {"v": 1},
                    {"v": "zero_counter"},
                    {"v": "0x0"},
                    {"v": "Zero counter"},
                    {"v": "TPU_TC_SET"},
                ]
            },
        ],
    }
    return json.dumps(table)

  def test_get_perf_counters_decodes_hex_and_sorts_descending(self):
    self.mock_client.fetch.return_value = (81, self._make_datatable_json())
    raw = get_perf_counters_tool.get_perf_counters("session_1", limit=10)
    res = json.loads(raw) if isinstance(raw, str) else raw

    self.assertEqual(res["status"], "OK")
    self.assertEqual(res["summary"]["total_rows"], 3)
    self.assertEqual(res["summary"]["non_zero_rows"], 2)
    self.assertEqual(res["summary"]["returned_rows"], 2)
    self.assertEqual(res["summary"]["hosts"], ["host0", "host1"])
    self.assertEqual(res["summary"]["chips"], [0, 1])

    self.assertEqual(res["counters"][0]["counter"], "hbm_read_bytes")
    self.assertEqual(res["counters"][0]["value"], 8192)
    self.assertEqual(res["counters"][0]["value_hex"], "0x2000")
    self.assertEqual(res["counters"][1]["counter"], "mxu_busy_cycles")
    self.assertEqual(res["counters"][1]["value"], 4096)

  def test_get_perf_counters_filtering(self):
    self.mock_client.fetch.return_value = (81, self._make_datatable_json())
    raw = get_perf_counters_tool.get_perf_counters(
        "session_1",
        kernel_filter="fusion.1",
        chip_id=0,
        non_zero_only=False,
        limit=10,
    )
    res = json.loads(raw) if isinstance(raw, str) else raw
    self.assertEqual(res["summary"]["matched_rows"], 2)
    self.assertLen(res["counters"], 2)

  def test_get_perf_counters_prefers_formatted_hex_and_filters_sentinels(self):
    table = {
        "cols": [
            {"id": "Host", "label": "Host", "type": "string"},
            {"id": "Chip", "label": "Chip", "type": "number"},
            {"id": "Kernel", "label": "Kernel", "type": "string"},
            {"id": "Sample", "label": "Sample", "type": "number"},
            {"id": "Counter", "label": "Counter", "type": "string"},
            {"id": "Value", "label": "Value (Hex)", "type": "number"},
            {"id": "Description", "label": "Description", "type": "string"},
            {"id": "Set", "label": "Set", "type": "string"},
        ],
        "rows": [
            {
                "c": [
                    {"v": "host0"},
                    {"v": 0},
                    {"v": "fusion.1"},
                    {"v": 1},
                    {"v": "sentinel_counter"},
                    {"v": 18446744073709551616.0, "f": "0xffffffffffffffff"},
                    {"v": "Uninitialized CSR sentinel"},
                    {"v": "TPU_SET"},
                ]
            },
            {
                "c": [
                    {"v": "host0"},
                    {"v": 0},
                    {"v": "fusion.1"},
                    {"v": 1},
                    {"v": "large_64bit_counter"},
                    {"v": 9007199254740996.0, "f": "0x20000000000003"},
                    {"v": "64-bit counter exceeding 53-bit double precision"},
                    {"v": "TPU_SET"},
                ]
            },
        ],
    }
    self.mock_client.fetch.return_value = (81, json.dumps(table))
    raw = get_perf_counters_tool.get_perf_counters("session_1", limit=10)
    res = json.loads(raw) if isinstance(raw, str) else raw
    self.assertEqual(res["summary"]["total_rows"], 2)
    self.assertEqual(res["summary"]["sentinel_all_ones_rows"], 1)
    self.assertEqual(res["summary"]["non_zero_rows"], 1)
    self.assertLen(res["counters"], 1)
    self.assertEqual(res["counters"][0]["counter"], "large_64bit_counter")
    self.assertEqual(res["counters"][0]["value"], 0x20000000000003)
    self.assertEqual(res["counters"][0]["value_hex"], "0x20000000000003")


if __name__ == "__main__":
  absltest.main()
