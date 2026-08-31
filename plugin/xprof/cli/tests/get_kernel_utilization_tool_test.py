"""Unit tests for get_kernel_utilization_tool CLI interface in 3P."""

import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tensorflow.tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_kernel_utilization_tool


def _create_sample_xspace() -> xplane_pb2.XSpace:
  space = xplane_pb2.XSpace()
  plane = space.planes.add()
  plane.name = "/device:TPU:0"

  # Stat metadata
  stat_dev_id = plane.stat_metadata[0]
  stat_dev_id.name = "device_id"
  stat_dev_id.id = 0
  stat_dev_type = plane.stat_metadata[1]
  stat_dev_type.name = "device_type_string"
  stat_dev_type.id = 1
  stat_counter_id = plane.stat_metadata[2]
  stat_counter_id.name = "performance_counter_id"
  stat_counter_id.id = 2
  stat_counter_val = plane.stat_metadata[3]
  stat_counter_val.name = "counter_value"
  stat_counter_val.id = 3

  # Plane stats
  s0 = plane.stats.add()
  s0.metadata_id = 0
  s0.int64_value = 0
  s1 = plane.stats.add()
  s1.metadata_id = 1
  s1.str_value = "TPU v7x"

  # Line and events
  line = plane.lines.add()
  line.id = 0
  line.name = "counters_matmul"

  # Event 0: CYCLES (UNPRIVILEGED_CYCLE_COUNT)
  ev0 = line.events.add()
  ev0.metadata_id = 0
  ev_s0 = ev0.stats.add()
  ev_s0.metadata_id = 2
  ev_s0.uint64_value = 2847490056
  ev_s1 = ev0.stats.add()
  ev_s1.metadata_id = 3
  ev_s1.double_value = 2000.0

  # Event 1: MXU_BUSY_1
  ev1 = line.events.add()
  ev1.metadata_id = 1
  ev_mxu_0 = ev1.stats.add()
  ev_mxu_0.metadata_id = 2
  ev_mxu_0.uint64_value = 2791875024
  ev_mxu_1 = ev1.stats.add()
  ev_mxu_1.metadata_id = 3
  ev_mxu_1.double_value = 800.0

  # Event 2: MATMUL_VREG_BF16_MXU_0
  ev2 = line.events.add()
  ev2.metadata_id = 2
  ev_bf16_0 = ev2.stats.add()
  ev_bf16_0.metadata_id = 2
  ev_bf16_0.uint64_value = 2791874968
  ev_bf16_1 = ev2.stats.add()
  ev_bf16_1.metadata_id = 3
  ev_bf16_1.double_value = 400.0

  return space


class GetKernelUtilizationToolTest(parameterized.TestCase):

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
    self.mock_client = mock.create_autospec(
        xprof_client.CachedXprofClient, instance=True, spec_set=True
    )
    self.enter_context(
        mock.patch.object(
            xprof_client,
            "get_client",
            return_value=self.mock_client,
            autospec=True,
            spec_set=True,
        )
    )

  def test_get_kernel_utilization_from_raw_bytes(self):
    space = _create_sample_xspace()
    raw_bytes = space.SerializeToString()

    result = get_kernel_utilization_tool.get_kernel_utilization(
        session_id="",
        raw_bytes=raw_bytes,
        output_format="json",
        bypass_cache=True,
    )
    parsed = json.loads(result)
    self.assertEqual(parsed["status"], "SUCCESS")
    self.assertIn("devices", parsed)
    self.assertLen(parsed["devices"], 1)
    device = parsed["devices"][0]
    self.assertEqual(device["device_type"], "TPU v7x")
    self.assertEqual(device["device_id"], 0)
    self.assertLen(device["kernels"], 1)
    kernel = device["kernels"][0]
    self.assertEqual(kernel["kernel_name"], "matmul")
    self.assertGreater(kernel["mxu_utilization"], 0.0)
    self.assertEqual(kernel["mxu_cycles_breakdown"]["BF16"], 100.0)
    self.assertFalse(kernel["mxu_is_anomaly"])

  def test_get_kernel_utilization_from_local_file(self):
    space = _create_sample_xspace()
    temp_file = self.create_tempfile(
        "test_trace.xplane.pb", content=space.SerializeToString()
    )

    result = get_kernel_utilization_tool.get_kernel_utilization(
        session_id=temp_file.full_path,
        output_format="dict",
        bypass_cache=True,
    )
    self.assertIsInstance(result, dict)
    self.assertEqual(result["status"], "SUCCESS")
    self.assertIn("devices", result)
    self.assertEqual(
        result["devices"][0]["kernels"][0]["kernel_name"], "matmul"
    )

  def test_get_kernel_utilization_from_session_id_with_client(self):
    mock_json_response = json.dumps({
        "status": "SUCCESS",
        "devices": [{
            "device_id": 0,
            "device_type": "TPU v7x",
            "kernels": [{
                "kernel_name": "matmul",
                "mxu_utilization": 25.0,
                "mxu_is_anomaly": False,
                "mxu_cycles_breakdown": {
                    "BF16": 100.0,
                    "Int8": 0.0,
                    "Int4": 0.0,
                    "FP8": 0.0,
                },
                "other_metrics": {},
            }],
        }],
    })
    self.mock_client.fetch.return_value = (
        None,
        mock_json_response.encode("utf-8"),
    )

    result = get_kernel_utilization_tool.get_kernel_utilization(
        session_id="session_12345",
        kernel_name="matmul",
        output_format="dict",
        bypass_cache=True,
    )
    self.mock_client.fetch.assert_called_once_with(
        tool_name="kernel_utilization.json",
        session_id="session_12345",
        bypass_cache=True,
        kernel="matmul",
    )
    self.assertEqual(result["status"], "SUCCESS")
    self.assertEqual(
        result["devices"][0]["kernels"][0]["kernel_name"], "matmul"
    )

  def test_get_kernel_utilization_duration_override(self):
    space = _create_sample_xspace()
    result = get_kernel_utilization_tool.get_kernel_utilization(
        session_id="",
        raw_bytes=space.SerializeToString(),
        duration_us=20.0,
        force_duration=True,
        output_format="dict",
        bypass_cache=True,
    )
    self.assertEqual(result["status"], "SUCCESS")
    self.assertAlmostEqual(
        result["devices"][0]["kernels"][0]["duration_us"], 20.0, places=3
    )

  def test_get_kernel_utilization_empty_session_and_raw_bytes_raises_value_error(
      self,
  ):
    with self.assertRaises(ValueError):
      get_kernel_utilization_tool.get_kernel_utilization(
          session_id="",
          raw_bytes=None,
          bypass_cache=True,
      )

  def test_get_kernel_utilization_client_no_data_raises_file_not_found_error(
      self,
  ):
    self.mock_client.fetch.return_value = (None, None)
    with self.assertRaises(FileNotFoundError):
      get_kernel_utilization_tool.get_kernel_utilization(
          session_id="non_existent_session",
          bypass_cache=True,
      )

  def test_get_kernel_utilization_client_error_raises_runtime_error(self):
    self.mock_client.fetch.side_effect = ConnectionError("Network error")
    with self.assertRaises(RuntimeError):
      get_kernel_utilization_tool.get_kernel_utilization(
          session_id="session_err",
          bypass_cache=True,
      )


if __name__ == "__main__":
  absltest.main()
