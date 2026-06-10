import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_memory_profile_tool


class GetMemoryProfileToolTest(parameterized.TestCase):

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
    self.mock_client = mock.create_autospec(xprof_client.CachedXprofClient)
    self.mock_client.get_hosts.return_value = None  # Default to no hosts
    # Patch the get_client function to return our mock
    self.enter_context(
        mock.patch.object(
            xprof_client,
            "get_client",
            return_value=self.mock_client,
            autospec=True,
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="per_allocator",
          mock_response_data=[{
              "memoryProfilePerAllocator": {
                  "0": {
                      "profileSummary": {
                          "memoryCapacity": "17179869184",  # 16 GiB
                          "peakStats": {
                              "peakBytesInUse": "8589934592",  # 8 GiB
                              "stackReservedBytes": "1073741824",  # 1 GiB
                              "heapAllocatedBytes": "7516192768",  # 7 GiB
                              "freeMemoryBytes": "8589934592",  # 8 GiB
                              "fragmentation": 0.05,
                          },
                      }
                  }
              }
          }],
          fetch_returns_tuple=True,
          expected={
              "memory_capacity_gib": 16.0,
              "peak_memory_usage_gib": 8.0,
              "peak_usage_details": {
                  "stack_reservation_gib": 1.0,
                  "heap_allocation_gib": 7.0,
                  "free_memory_gib": 8.0,
                  "fragmentation_percent": 5.0,
                  "utilization_percent": 50.0,
              },
          },
      ),
      dict(
          testcase_name="summary_format",
          mock_response_data=[{
              "memoryProfileSummary": {
                  "memoryCapacity": "34359738368",  # 32 GiB
                  "peakStats": {
                      "peakBytesUsageHbm": "17179869184",  # 16 GiB
                      "stackReservedBytes": "2147483648",  # 2 GiB
                      "heapAllocatedBytes": "15032385536",  # 14 GiB
                      "freeMemoryBytes": "17179869184",  # 16 GiB
                      "fragmentation": 0.10,
                  },
              }
          }],
          fetch_returns_tuple=False,
          expected={
              "memory_capacity_gib": 32.0,
              "peak_memory_usage_gib": 16.0,
              "peak_usage_details": {
                  "stack_reservation_gib": 2.0,
                  "heap_allocation_gib": 14.0,
                  "free_memory_gib": 16.0,
                  "fragmentation_percent": 10.0,
                  "utilization_percent": 50.0,
              },
          },
      ),
      dict(
          testcase_name="peak_memory_usage_mib_format",
          mock_response_data=[{"peakMemoryUsageMiB": "4096.0"}],  # 4 GiB
          fetch_returns_tuple=False,
          expected={
              "memory_capacity_gib": -1.0,
              "peak_memory_usage_gib": 4.0,
              "peak_usage_details": {
                  "stack_reservation_gib": -1.0,
                  "heap_allocation_gib": -1.0,
                  "free_memory_gib": -1.0,
                  "fragmentation_percent": -1.0,
                  "utilization_percent": -1.0,
              },
          },
      ),
      dict(
          testcase_name="uninitialized_summary",
          mock_response_data=[{
              "memoryProfileSummary": {
                  "memoryCapacity": "0",
                  "peakStats": {
                      "peakBytesUsageHbm": "0",
                  },
              }
          }],
          fetch_returns_tuple=False,
          expected={
              "memory_capacity_gib": -1.0,
              "peak_memory_usage_gib": -1.0,
              "peak_usage_details": {
                  "stack_reservation_gib": -1.0,
                  "heap_allocation_gib": -1.0,
                  "free_memory_gib": -1.0,
                  "fragmentation_percent": -1.0,
                  "utilization_percent": -1.0,
              },
          },
      ),
  )
  def test_get_memory_profile(
      self, mock_response_data, fetch_returns_tuple, expected
  ):
    mock_response = json.dumps(mock_response_data).encode("utf-8")
    if fetch_returns_tuple:
      self.mock_client.fetch.return_value = ("json", mock_response)
    else:
      self.mock_client.fetch.return_value = mock_response

    result_json = get_memory_profile_tool.get_memory_profile("test_session")
    result = json.loads(result_json)

    self.assertNotIn("error", result)
    self.assertEqual(result, expected)

  def test_get_memory_profile_no_data(self):
    self.mock_client.fetch.return_value = None

    result_json = get_memory_profile_tool.get_memory_profile("test_session")
    result = json.loads(result_json)

    expected = {
        "memory_capacity_gib": -1.0,
        "peak_memory_usage_gib": -1.0,
        "peak_usage_details": {
            "stack_reservation_gib": -1.0,
            "heap_allocation_gib": -1.0,
            "free_memory_gib": -1.0,
            "fragmentation_percent": -1.0,
            "utilization_percent": -1.0,
        },
    }
    self.assertEqual(result, expected)

  def test_get_memory_profile_fallback(self):
    self.mock_client.get_hosts.return_value = [
        {"hostname": "host-primary"},
        {"hostname": "host-secondary"},
    ]

    def fetch_side_effect(**kwargs):
      host = kwargs.get("host")
      if host is None:
        return b'[{"memoryProfileSummary": {"memoryCapacity": "0"}}]'
      elif host == "host-primary":
        return b'[{"memoryProfileSummary": {"memoryCapacity": "0"}}]'
      elif host == "host-secondary":
        return b"""[{
            "memoryProfileSummary": {
                "memoryCapacity": "17179869184",
                "peakStats": {
                    "peakBytesUsageHbm": "8589934592",
                    "stackReservedBytes": "1073741824",
                    "heapAllocatedBytes": "7516192768",
                    "freeMemoryBytes": "8589934592",
                    "fragmentation": 0.05
                }
            }
        }]"""
      return None

    self.mock_client.fetch.side_effect = fetch_side_effect

    result_json = get_memory_profile_tool.get_memory_profile("test_session")
    result = json.loads(result_json)

    expected = {
        "memory_capacity_gib": 16.0,
        "peak_memory_usage_gib": 8.0,
        "peak_usage_details": {
            "stack_reservation_gib": 1.0,
            "heap_allocation_gib": 7.0,
            "free_memory_gib": 8.0,
            "fragmentation_percent": 5.0,
            "utilization_percent": 50.0,
        },
    }
    self.assertNotIn("error", result)
    self.assertEqual(result, expected)

    self.mock_client.fetch.assert_has_calls([
        mock.call(
            tool_name="memory_profile.json",
            session_id="test_session",
            format="json",
        ),
        mock.call(
            tool_name="memory_profile.json",
            session_id="test_session",
            format="json",
            host="host-primary",
        ),
        mock.call(
            tool_name="memory_profile.json",
            session_id="test_session",
            format="json",
            host="host-secondary",
        ),
    ])

  def test_get_memory_profile_invalid_json(self):
    self.mock_client.fetch.return_value = b"invalid json data"

    result_json = get_memory_profile_tool.get_memory_profile("test_session")
    result = json.loads(result_json)

    self.assertIn("error", result)
    self.assertTrue(result["error"].startswith("Failed to parse JSON"))


if __name__ == "__main__":
  absltest.main()
