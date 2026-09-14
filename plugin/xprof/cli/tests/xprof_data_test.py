"""Unit tests for XprofData HLO operation profile extraction and summarization."""

# pylint: disable=g-redundant-test-module-docstring

import json
from unittest import mock

from absl.testing import absltest
from xprof.cli.internal import decorators
from xprof.cli.internal import xprof_data
from xprof.cli.internal.oss import xprof_client
from xprof.protobuf import op_profile_pb2


class XprofDataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_cache = mock.create_autospec(
        decorators.Cache, instance=True, spec_set=True
    )
    mock_cache.get.return_value = decorators.Cache.UNKNOWN
    self.mock_cache_patcher = mock.patch.object(
        decorators,
        "get_cache",
        return_value=mock_cache,
        autospec=True,
    )
    self.mock_cache_patcher.start()
    # Mock the client directly to avoid internal-only string patching
    self.mock_client = mock.MagicMock(spec=xprof_client.CachedXprofClient)
    xprof_client.set_client_override(self.mock_client)

  def tearDown(self):
    self.mock_cache_patcher.stop()
    xprof_client.set_client_override(None)
    super().tearDown()

  def _create_mock_profile(self) -> op_profile_pb2.Profile:
    return op_profile_pb2.Profile(
        by_category=op_profile_pb2.Node(
            name="by_category",
            metrics=op_profile_pb2.Metrics(
                raw_time=100000000000, occurrences=15, raw_flops=1500
            ),
            children=[
                op_profile_pb2.Node(
                    name="MatMul",
                    category=op_profile_pb2.Node.InstructionCategory(),
                    metrics=op_profile_pb2.Metrics(
                        raw_time=60000000000,
                        occurrences=10,
                        raw_flops=1000,
                        raw_bytes_accessed_array=[100, 200],
                    ),
                ),
                op_profile_pb2.Node(
                    name="Fusion",
                    xla=op_profile_pb2.Node.XLAInstruction(
                        category="FusionCategory"
                    ),
                    metrics=op_profile_pb2.Metrics(
                        raw_time=40000000000, occurrences=5, raw_flops=500
                    ),
                ),
            ],
        )
    )

  def test_get_hlo_op_profile_grouped_default(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())
    result = xprof_data.get_hlo_op_profile("session_op")
    result_json = json.loads(result)

    self.assertIn("category_summary", result_json)
    self.assertIn("grouped_operations", result_json)
    self.assertIn("navigation_hints", result_json)

    categories = [c["category"] for c in result_json["category_summary"]]
    self.assertIn("Category: MatMul", categories)
    self.assertIn("FusionCategory", categories)

    self.assertIn("total_profile_time_ms", result_json)
    self.assertEqual(result_json["total_profile_time_ms"], 100.0)
    self.assertIn("drill_down_category", result_json["navigation_hints"])
    self.assertIn("available_categories", result_json["navigation_hints"])
    self.assertIn(
        "--instruction_name=",
        result_json["navigation_hints"]["inspect_op_neighborhood"],
    )
    self.assertNotIn(
        "--op_name=",
        result_json["navigation_hints"]["inspect_op_neighborhood"],
    )

  def test_get_hlo_op_profile_category_view(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())
    result = xprof_data.get_hlo_op_profile("session_op", view="category")
    result_json = json.loads(result)

    self.assertIn("category_summary", result_json)
    self.assertIn("total_profile_time_ms", result_json)
    self.assertEqual(result_json["total_profile_time_ms"], 100.0)
    self.assertNotIn("grouped_operations", result_json)
    self.assertLen(result_json["category_summary"], 2)
    self.assertEqual(
        result_json["category_summary"][0]["category"], "Category: MatMul"
    )
    self.assertEqual(
        result_json["category_summary"][0]["total_self_time_ms"], 60.0
    )
    total_frac = sum(
        c["fraction_of_total_time"] for c in result_json["category_summary"]
    )
    self.assertAlmostEqual(total_frac, 1.0, places=4)

  def test_get_hlo_op_profile_nested_fusions_category_rollup_unity(self):
    """Verifies that nested fusions do not cause rollup double-counting."""
    profile = op_profile_pb2.Profile(
        by_category=op_profile_pb2.Node(
            name="by_category",
            metrics=op_profile_pb2.Metrics(
                raw_time=100000000000, occurrences=15, raw_flops=1500
            ),
            children=[
                op_profile_pb2.Node(
                    name="MatMul",
                    category=op_profile_pb2.Node.InstructionCategory(),
                    metrics=op_profile_pb2.Metrics(
                        raw_time=60000000000,
                        occurrences=10,
                        raw_flops=1000,
                    ),
                ),
                op_profile_pb2.Node(
                    name="FusionParent",
                    xla=op_profile_pb2.Node.XLAInstruction(
                        category="FusionCategory"
                    ),
                    metrics=op_profile_pb2.Metrics(
                        raw_time=40000000000, occurrences=5, raw_flops=500
                    ),
                    children=[
                        op_profile_pb2.Node(
                            name="child_add",
                            xla=op_profile_pb2.Node.XLAInstruction(
                                category="FusionCategory"
                            ),
                            metrics=op_profile_pb2.Metrics(
                                raw_time=15000000000,
                                occurrences=5,
                                raw_flops=200,
                            ),
                        ),
                        op_profile_pb2.Node(
                            name="child_mul",
                            xla=op_profile_pb2.Node.XLAInstruction(
                                category="FusionCategory"
                            ),
                            metrics=op_profile_pb2.Metrics(
                                raw_time=25000000000,
                                occurrences=5,
                                raw_flops=300,
                            ),
                        ),
                    ],
                ),
            ],
        )
    )
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())
    result = xprof_data.get_hlo_op_profile("session_nested", view="category")
    result_json = json.loads(result)

    total_frac = sum(
        c["fraction_of_total_time"] for c in result_json["category_summary"]
    )
    self.assertAlmostEqual(total_frac, 1.0, places=4)

    # In flat view, only true leaves (child_add, child_mul, MatMul) appear,
    # never the intermediate FusionParent container.
    flat_result = json.loads(
        xprof_data.get_hlo_op_profile("session_nested", view="flat")
    )
    flat_names = [op["name"] for op in flat_result]
    self.assertIn("by_category/MatMul", flat_names)
    self.assertIn("by_category/FusionParent/child_add", flat_names)
    self.assertIn("by_category/FusionParent/child_mul", flat_names)
    self.assertNotIn("by_category/FusionParent", flat_names)

  def test_get_hlo_op_profile_fusion_with_zero_time_children_preserved(self):
    """Verifies that fusion ops with zero-time children are preserved as leaves."""
    profile = op_profile_pb2.Profile(
        by_category=op_profile_pb2.Node(
            name="by_category",
            metrics=op_profile_pb2.Metrics(
                raw_time=100000000000, occurrences=15, raw_flops=1500
            ),
            children=[
                op_profile_pb2.Node(
                    name="MatMul",
                    category=op_profile_pb2.Node.InstructionCategory(),
                    metrics=op_profile_pb2.Metrics(
                        raw_time=60000000000,
                        occurrences=10,
                        raw_flops=1000,
                    ),
                ),
                op_profile_pb2.Node(
                    name="fusion.668",
                    xla=op_profile_pb2.Node.XLAInstruction(
                        category="convolution fusion"
                    ),
                    metrics=op_profile_pb2.Metrics(
                        raw_time=40000000000, occurrences=5, raw_flops=500
                    ),
                    # In real HLO traces, fusion sub-instructions carry 0
                    # raw_time
                    children=[
                        op_profile_pb2.Node(
                            name="sub_add",
                            xla=op_profile_pb2.Node.XLAInstruction(
                                category="convolution fusion"
                            ),
                            metrics=op_profile_pb2.Metrics(
                                raw_time=0,
                                occurrences=5,
                                raw_flops=0,
                            ),
                        ),
                        op_profile_pb2.Node(
                            name="sub_mul",
                            xla=op_profile_pb2.Node.XLAInstruction(
                                category="convolution fusion"
                            ),
                            metrics=op_profile_pb2.Metrics(
                                raw_time=0,
                                occurrences=5,
                                raw_flops=0,
                            ),
                        ),
                    ],
                ),
            ],
        )
    )
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())
    result = xprof_data.get_hlo_op_profile("session_fusion", view="category")
    result_json = json.loads(result)

    categories = {c["category"]: c for c in result_json["category_summary"]}
    self.assertIn("convolution fusion", categories)
    self.assertEqual(
        categories["convolution fusion"]["total_self_time_ms"], 40.0
    )
    total_frac = sum(
        c["fraction_of_total_time"] for c in result_json["category_summary"]
    )
    self.assertAlmostEqual(total_frac, 1.0, places=4)

    # In flat view, fusion.668 itself is the leaf op emitted
    flat_result = json.loads(
        xprof_data.get_hlo_op_profile("session_fusion", view="flat")
    )
    flat_names = [op["name"] for op in flat_result]
    self.assertIn("by_category/MatMul", flat_names)
    self.assertIn("by_category/fusion.668", flat_names)
    self.assertNotIn("by_category/fusion.668/sub_add", flat_names)
    self.assertNotIn("by_category/fusion.668/sub_mul", flat_names)

  def test_get_hlo_op_profile_category_filter(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())
    result = xprof_data.get_hlo_op_profile("session_op", category="Fusion")
    result_json = json.loads(result)

    self.assertEqual(result_json["category"], "FusionCategory")
    self.assertEqual(result_json["total_self_time_ms"], 40.0)
    self.assertLen(result_json["operations"], 1)
    self.assertEqual(result_json["operations"][0]["name"], "by_category/Fusion")
    self.assertIn(
        "--instruction_name=",
        result_json["navigation_hints"]["inspect_top_op_ast"],
    )
    self.assertNotIn(
        "--op_name=",
        result_json["navigation_hints"]["inspect_top_op_ast"],
    )

  def test_get_hlo_op_profile_category_not_found(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())

    with self.assertRaises(FileNotFoundError):
      xprof_data.get_hlo_op_profile("session_op", category="NonExistent")

  def test_get_hlo_op_profile_flat_view(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())
    result = xprof_data.get_hlo_op_profile("session_op", view="flat")
    result_json = json.loads(result)

    self.assertIsInstance(result_json, list)
    self.assertLen(result_json, 2)
    self.assertEqual(result_json[0]["name"], "by_category/MatMul")
    self.assertEqual(result_json[0]["total_self_time_ms"], 60.0)
    self.assertEqual(result_json[1]["name"], "by_category/Fusion")
    self.assertEqual(result_json[1]["total_self_time_ms"], 40.0)

  def test_get_hlo_op_profile_tree_view(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())
    result = xprof_data.get_hlo_op_profile("session_op", view="tree", depth=2)
    result_json = json.loads(result)

    self.assertIn("tree", result_json)
    self.assertIn("current_path", result_json)
    self.assertIn("navigation_hints", result_json)
    self.assertEqual(result_json["current_path"], "by_category")
    self.assertLen(result_json["tree"]["children"], 2)

  def test_get_hlo_op_profile_tree_path_not_found(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())

    with self.assertRaises(FileNotFoundError):
      xprof_data.get_hlo_op_profile(
          "session_op", view="tree", path="invalid/path"
      )

  def test_get_hlo_op_profile_invalid_view(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())

    with self.assertRaises(ValueError):
      xprof_data.get_hlo_op_profile("session_op", view="unsupported_view")

  def test_get_hlo_op_profile_invalid_sort_by(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())

    with self.assertRaises(ValueError):
      xprof_data.get_hlo_op_profile("session_op", sort_by="unsupported_sort")

  def test_get_profile_summary_success(self):
    profile = op_profile_pb2.Profile(
        by_category=op_profile_pb2.Node(
            name="root", metrics=op_profile_pb2.Metrics(raw_time=100e12)
        )
    )
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())

    result = xprof_data.get_profile_summary("session_summary")

    self.mock_client.fetch.assert_called_with(
        tool_name="op_profile",
        session_id="session_summary",
        format="json",
        bypass_cache=False,
    )
    parsed = json.loads(result)
    self.assertEqual(parsed["status"], "SUCCESS")
    self.assertAlmostEqual(parsed["total_time_s"], 100.0)
    self.assertIn("Profile Summary", parsed["summary_markdown"])
    self.assertIn("Total Time:", parsed["summary_markdown"])

  def test_get_profile_summary_bypass_cache(self):
    profile = op_profile_pb2.Profile(
        by_category=op_profile_pb2.Node(
            name="root", metrics=op_profile_pb2.Metrics(raw_time=100e12)
        )
    )
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())

    result = xprof_data.get_profile_summary(
        "session_summary", bypass_cache=True
    )

    self.mock_client.fetch.assert_called_with(
        tool_name="op_profile",
        session_id="session_summary",
        format="json",
        bypass_cache=True,
    )
    parsed = json.loads(result)
    self.assertEqual(parsed["status"], "SUCCESS")
    self.assertIn("Profile Summary", parsed["summary_markdown"])

  def test_get_hosts(self):
    self.mock_client.get_hosts.return_value = [
        {"hostname": "host1", "ip": "1.2.3.4"},
        {"hostname": "host2", "ip": "5.6.7.8"},
    ]

    result = xprof_data.get_hosts("session_hosts")
    result_json = json.loads(result)

    self.assertIn("hosts", result_json)
    self.assertEqual(
        result_json["hosts"],
        [
            {"hostname": "host1", "ip": "1.2.3.4"},
            {"hostname": "host2", "ip": "5.6.7.8"},
        ],
    )
    self.mock_client.get_hosts.assert_called_with(
        "session_hosts", with_metadata=True
    )

  def test_get_hosts_error(self):
    self.mock_client.get_hosts.side_effect = Exception("RPC Fail")

    with self.assertRaises(RuntimeError):
      xprof_data.get_hosts("session_hosts")

  def test_get_hosts_empty(self):
    self.mock_client.get_hosts.return_value = []

    with self.assertRaises(FileNotFoundError):
      xprof_data.get_hosts("session_hosts")

  def test_get_device_information_success(self):
    roofline_json = json.dumps([{
        "p": {
            "device_type": "TPU v5p",
            "peak_flop_rate": "1234.5",
            "peak_hbm_bw": "678",
            "peak_vmem_bw": "910",
            "ridge_point": "not_a_number",
        }
    }]).encode("utf-8")
    self.mock_client.fetch.return_value = (81, roofline_json)

    result = xprof_data.get_device_information("session_device")
    result_json = json.loads(result)

    self.assertEqual(result_json["device_type"], "TPU v5p")
    self.assertEqual(result_json["peak_flop_rate"], 1234.5)
    # Bandwidth fields are renamed with an explicit `_gibs` suffix; the raw
    # unsuffixed names must no longer be present.
    self.assertEqual(result_json["peak_hbm_bw_gibs"], 678.0)
    self.assertEqual(result_json["peak_vmem_bw_gibs"], 910.0)
    self.assertNotIn("peak_hbm_bw", result_json)
    self.assertNotIn("peak_vmem_bw", result_json)
    self.assertEqual(result_json["ridge_point"], "not_a_number")
    # A `units` metadata dict documents the bandwidth units unambiguously.
    self.assertEqual(
        result_json["units"],
        {"peak_hbm_bw_gibs": "GiB/s", "peak_vmem_bw_gibs": "GiB/s"},
    )

  def test_get_device_information_error(self):
    self.mock_client.fetch.side_effect = Exception("RPC Fail")

    with self.assertRaises(RuntimeError):
      xprof_data.get_device_information("session_device")

  def test_get_device_information_empty(self):
    self.mock_client.fetch.return_value = (81, b"")

    with self.assertRaises(FileNotFoundError):
      xprof_data.get_device_information("session_device")

  def test_get_profile_summary_missing_data(self):
    self.mock_client.fetch.return_value = (None, None)

    with self.assertRaises(FileNotFoundError):
      xprof_data.get_profile_summary("session_missing")

  def test_get_hlo_op_profile_missing_data(self):
    self.mock_client.fetch.return_value = (None, None)

    with self.assertRaises(FileNotFoundError):
      xprof_data.get_hlo_op_profile("session_missing")

  def test_sanitize_and_bound_json(self):
    payload = {
        "a": [1, 2, 3, 4, 5],
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        (1, 2): b"raw_bytes",
    }
    bounded, truncated, notes = xprof_data.sanitize_and_bound_json(
        payload, max_depth=5, max_nodes=3
    )
    self.assertTrue(truncated)
    self.assertTrue(any("max_nodes" in n for n in notes))
    self.assertIn("a", bounded)
    bounded_full, truncated_full, _ = xprof_data.sanitize_and_bound_json(
        payload, max_depth=5, max_nodes=100
    )
    self.assertFalse(truncated_full)
    self.assertIsNone(bounded_full["nan_val"])
    self.assertIsNone(bounded_full["inf_val"])
    self.assertEqual(bounded_full["(1, 2)"], "raw_bytes")
    # Ensure json.dumps succeeds without error on the sanitized output.
    json.dumps(bounded_full)

  def test_get_profile_summary_bounded_json(self):
    profile = self._create_mock_profile()
    self.mock_client.fetch.return_value = (None, profile.SerializeToString())
    res_str = xprof_data.get_profile_summary("session_summary")
    res_json = json.loads(res_str)
    self.assertIn("markdown_summary", res_json)
    self.assertIn("top_operations", res_json)
    self.assertIn("truncated", res_json)
    self.assertFalse(res_json["truncated"])


if __name__ == "__main__":
  absltest.main()
