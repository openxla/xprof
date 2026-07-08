import json
import textwrap
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_peak_allocations_tool


class GetPeakAllocationsToolTest(parameterized.TestCase):

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

  @parameterized.named_parameters(
      dict(
          testcase_name="success_logical_buffers",
          mock_side_effect=[
              b"module1,module2",
              (
                  b'{"totalBufferAllocationMib": 100.0, "bufferAssignment":'
                  b' {"logicalBuffers": [{"size": "1048576", "definedAt":'
                  b' {"instructionName": "op1"}}]}}'
              ),
              (
                  b'{"totalBufferAllocationMib": 200.0, "bufferAssignment":'
                  b' {"logicalBuffers": [{"size": "2097152", "definedAt":'
                  b' {"instructionName": "op2"}}]}}'
              ),
          ],
          expected_data=[
              {
                  "module_name": "module2",
                  "total_hbm_mib": 200.0,
                  "top_buffers": [{"instruction": "op2", "size_mib": 2.0}],
              },
              {
                  "module_name": "module1",
                  "total_hbm_mib": 100.0,
                  "top_buffers": [{"instruction": "op1", "size_mib": 1.0}],
              },
          ],
          expected_calls=[
              mock.call(
                  tool_name="memory_viewer.json",
                  session_id="session_123",
                  format="json",
              ),
              mock.call(
                  tool_name="memory_viewer.json",
                  session_id="session_123",
                  format="json",
                  module_name="module1",
              ),
              mock.call(
                  tool_name="memory_viewer.json",
                  session_id="session_123",
                  format="json",
                  module_name="module2",
              ),
          ],
      ),
      dict(
          testcase_name="fallback_max_heap",
          mock_side_effect=[
              b"module1",
              (
                  b'{"totalBufferAllocationMib": 100.0, "maxHeap":'
                  b' [{"logicalBufferSizeMib": 1.0, "instructionName": "op1"}]}'
              ),
          ],
          expected_data=[{
              "module_name": "module1",
              "total_hbm_mib": 100.0,
              "top_buffers": [{"instruction": "op1", "size_mib": 1.0}],
          }],
          expected_calls=[
              mock.call(
                  tool_name="memory_viewer.json",
                  session_id="session_123",
                  format="json",
              ),
              mock.call(
                  tool_name="memory_viewer.json",
                  session_id="session_123",
                  format="json",
                  module_name="module1",
              ),
          ],
      ),
  )
  def test_get_peak_allocations_success_variants(
      self, mock_side_effect, expected_data, expected_calls
  ):
    self.mock_client.fetch.side_effect = mock_side_effect

    result = get_peak_allocations_tool.get_peak_allocations(
        "session_123", include_summary=False
    )

    parsed_result = json.loads(result)
    self.assertEqual(expected_data, parsed_result)
    self.assertSequenceEqual(
        expected_calls, self.mock_client.fetch.call_args_list
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="no_data",
          fetch_return=b"",
          expected_error=(
              "Failed to get peak allocations for session session_123:"
              " ValueError: No memory viewer data returned for the session"
          ),
      ),
      dict(
          testcase_name="empty_list",
          fetch_return=b"  ",
          expected_error=(
              "Failed to get peak allocations for session session_123:"
              " ValueError: No HLO modules found in memory viewer data for"
              " the session"
          ),
      ),
  )
  def test_get_peak_allocations_empty_data_variants(
      self, fetch_return, expected_error
  ):
    self.mock_client.fetch.return_value = fetch_return

    result = get_peak_allocations_tool.get_peak_allocations("session_123")

    parsed_result = json.loads(result)
    self.assertIn("error", parsed_result)
    self.assertEqual(expected_error, parsed_result["error"])

  def test_get_peak_allocations_error_markdown(self):
    self.mock_client.fetch.return_value = b""

    result = get_peak_allocations_tool.get_peak_allocations(
        "session_123", output_format="markdown"
    )

    expected_markdown = (
        "# Error\nFailed to get peak allocations for session session_123:"
        " ValueError: No memory viewer data returned for the session\n"
    )
    self.assertEqual(expected_markdown, result)

  def test_get_peak_allocations_aggregation(self):

    self.mock_client.fetch.side_effect = [
        b"module1",  # First call returns list of modules
        (  # Module1 with buffers that should be aggregated
            b'{"totalBufferAllocationMib": 300.0, "bufferAssignment":'
            b' {"logicalBuffers": [{"size": "67108864", "definedAt":'
            b' {"instructionName": "param.1"}},{"size": "67108864",'
            b' "definedAt": {"instructionName": "param.2"}},{"size": "1048576",'
            b' "definedAt": {"instructionName": "op1"}}]}}'
        ),
    ]

    result = get_peak_allocations_tool.get_peak_allocations(
        "session_123", include_summary=False
    )

    expected_data = [
        {
            "module_name": "module1",
            "total_hbm_mib": 300.0,
            "top_buffers": [
                {
                    "instruction": "param.* (2 occurrences of size 64 MiB)",
                    "size_mib": 128.0,
                },
                {"instruction": "op1", "size_mib": 1.0},
            ],
        },
    ]

    parsed_result = json.loads(result)
    self.assertEqual(expected_data, parsed_result)
    expected_calls = [
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
        ),
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
            module_name="module1",
        ),
    ]
    self.assertSequenceEqual(
        expected_calls, self.mock_client.fetch.call_args_list
    )

  @parameterized.named_parameters(
      dict(testcase_name="with_limit", limit=2, expected_len=2),
      dict(testcase_name="no_limit", limit=0, expected_len=3),
  )
  def test_get_peak_allocations_limit_variants(self, limit, expected_len):

    self.mock_client.fetch.side_effect = [
        b"module1,module2,module3",
        b'{"totalBufferAllocationMib": 100.0}',
        b'{"totalBufferAllocationMib": 300.0}',
        b'{"totalBufferAllocationMib": 200.0}',
    ]

    result = get_peak_allocations_tool.get_peak_allocations(
        "session_123", limit=limit, include_summary=False
    )

    parsed_result = json.loads(result)
    self.assertLen(parsed_result, expected_len)

    # Sorted order should be module2, module3, module1
    expected_order = ["module2", "module3", "module1"]
    self.assertEqual(
        expected_order[:expected_len],
        [mod["module_name"] for mod in parsed_result],
    )
    expected_calls = [
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
        ),
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
            module_name="module1",
        ),
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
            module_name="module2",
        ),
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
            module_name="module3",
        ),
    ]
    self.assertSequenceEqual(
        expected_calls, self.mock_client.fetch.call_args_list
    )

  def test_get_peak_allocations_size_threshold(self):

    self.mock_client.fetch.side_effect = [
        b"module1",
        (
            b'{"totalBufferAllocationMib": 100.0, "bufferAssignment":'
            b' {"logicalBuffers": [{"size": "2097152", "definedAt":'
            b' {"instructionName": "large_op"}},{"size": "524288", "definedAt":'
            b' {"instructionName": "small_op1"}},{"size": "262144",'
            b' "definedAt": {"instructionName": "small_op2"}}]}}'
        ),
    ]

    result = get_peak_allocations_tool.get_peak_allocations(
        "session_123", min_size_mib=1.0, include_summary=False
    )

    expected_data = [
        {
            "module_name": "module1",
            "total_hbm_mib": 100.0,
            "top_buffers": [
                {"instruction": "large_op", "size_mib": 2.0},
                {"instruction": "Others (< 1.0 MiB)", "size_mib": 0.75},
            ],
        },
    ]

    parsed_result = json.loads(result)
    self.assertEqual(expected_data, parsed_result)
    expected_calls = [
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
        ),
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
            module_name="module1",
        ),
    ]
    self.assertSequenceEqual(
        expected_calls, self.mock_client.fetch.call_args_list
    )

  def test_get_peak_allocations_markdown(self):

    self.mock_client.fetch.side_effect = [
        b"module1",
        (
            b'{"totalBufferAllocationMib": 100.0, "bufferAssignment":'
            b' {"logicalBuffers": ['
            b'{"size": "2097152", "definedAt": {"instructionName": "op1"}}'
            b"]}}"
        ),
    ]

    result = get_peak_allocations_tool.get_peak_allocations(
        "session_123", output_format="markdown", include_summary=False
    )

    expected_markdown = textwrap.dedent("""\
        # Peak Memory Allocations by Module

        > [!NOTE]
        > **Aggregation Logic:**
        > - Buffers with similar names (e.g., `name.1`, `name.2`) and identical sizes are aggregated into `name.*`.
        > - Buffers smaller than the threshold are aggregated into 'Others'.

        ## Module: `module1`
        Total HBM: 100.00 MiB

        | Instruction | Size (MiB) |
        | :--- | ---: |
        | `op1` | 2.00 |
        """)

    self.assertEqual(expected_markdown, result)
    expected_calls = [
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
        ),
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
            module_name="module1",
        ),
    ]
    self.assertSequenceEqual(
        expected_calls, self.mock_client.fetch.call_args_list
    )

  def test_get_peak_allocations_no_aggregation(self):
    self.mock_client.fetch.side_effect = [
        b"module1",
        (
            b'{"totalBufferAllocationMib": 100.0, "bufferAssignment":'
            b' {"logicalBuffers": ['
            b'{"size": "1048576", "definedAt": {"instructionName": "param.1"}},'
            b'{"size": "1048576", "definedAt": {"instructionName": "param.2"}},'
            b'{"size": "524288", "definedAt": {"instructionName": "small_op"}}'
            b"]}}"
        ),
    ]

    result = get_peak_allocations_tool.get_peak_allocations(
        "session_123", aggregate_instructions=False, include_summary=False
    )

    expected_data = [
        {
            "module_name": "module1",
            "total_hbm_mib": 100.0,
            "top_buffers": [
                {"instruction": "param.1", "size_mib": 1.0},
                {"instruction": "param.2", "size_mib": 1.0},
                {"instruction": "small_op", "size_mib": 0.5},
            ],
        },
    ]

    parsed_result = json.loads(result)
    self.assertEqual(expected_data, parsed_result)

  def test_get_peak_allocations_summary_json(self):

    self.mock_client.fetch.side_effect = [
        b"module1",
        (
            b'{"totalBufferAllocationMib": 100.0, "bufferAssignment":'
            b' {"logicalBuffers": ['
            b'{"size": "2097152", "definedAt": {"instructionName": "op1"}}'
            b"]}}"
        ),
    ]

    result = get_peak_allocations_tool.get_peak_allocations(
        "session_123", include_summary=True
    )

    expected_data = {
        "summary": {
            "total_modules": 1,
            "top_modules": [{"module_name": "module1", "total_hbm_mib": 100.0}],
        },
        "modules": [{
            "module_name": "module1",
            "total_hbm_mib": 100.0,
            "top_buffers": [{"instruction": "op1", "size_mib": 2.0}],
        }],
    }

    parsed_result = json.loads(result)
    self.assertEqual(expected_data, parsed_result)
    expected_calls = [
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
        ),
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
            module_name="module1",
        ),
    ]
    self.assertSequenceEqual(
        expected_calls, self.mock_client.fetch.call_args_list
    )

  def test_get_peak_allocations_summary_markdown(self):

    self.mock_client.fetch.side_effect = [
        b"module1",
        (
            b'{"totalBufferAllocationMib": 100.0, "bufferAssignment":'
            b' {"logicalBuffers": ['
            b'{"size": "2097152", "definedAt": {"instructionName": "op1"}}'
            b"]}}"
        ),
    ]

    result = get_peak_allocations_tool.get_peak_allocations(
        "session_123", output_format="markdown", include_summary=True
    )

    expected_markdown = textwrap.dedent("""\
        # Peak Memory Allocations by Module

        > [!NOTE]
        > **Aggregation Logic:**
        > - Buffers with similar names (e.g., `name.1`, `name.2`) and identical sizes are aggregated into `name.*`.
        > - Buffers smaller than the threshold are aggregated into 'Others'.

        ## Session Summary
        - Total Modules: 1

        | Module | Total HBM (MiB) |
        | :--- | ---: |
        | `module1` | 100.00 |

        ## Module: `module1`
        Total HBM: 100.00 MiB

        | Instruction | Size (MiB) |
        | :--- | ---: |
        | `op1` | 2.00 |
        """)

    self.assertEqual(expected_markdown, result)
    expected_calls = [
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
        ),
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
            module_name="module1",
        ),
    ]
    self.assertSequenceEqual(
        expected_calls, self.mock_client.fetch.call_args_list
    )


class InternalFunctionsTest(parameterized.TestCase):

  def test_get_module_names_success(self):
    mock_client = mock.create_autospec(
        xprof_client.CachedXprofClient, instance=True, spec_set=True
    )
    mock_client.fetch.return_value = b"module1,module2"
    module_names = get_peak_allocations_tool._get_module_names(
        mock_client, "session_123"
    )
    self.assertEqual(["module1", "module2"], module_names)
    expected_calls = [
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
        ),
    ]
    self.assertSequenceEqual(expected_calls, mock_client.fetch.call_args_list)

  @parameterized.named_parameters(
      dict(
          testcase_name="empty_data",
          fetch_return=b"",
          expected_regex="No memory viewer data returned",
      ),
      dict(
          testcase_name="no_modules",
          fetch_return=b",",
          expected_regex="No HLO modules found",
      ),
  )
  def test_get_module_names_failure_variants(
      self, fetch_return, expected_regex
  ):
    mock_client = mock.create_autospec(
        xprof_client.CachedXprofClient, instance=True, spec_set=True
    )
    mock_client.fetch.return_value = fetch_return
    with self.assertRaisesRegex(ValueError, expected_regex):
      get_peak_allocations_tool._get_module_names(mock_client, "session_123")

  def test_parse_and_aggregate_buffers_logical_buffers(self):
    parsed_mod_data = {
        "bufferAssignment": {
            "logicalBuffers": [
                {
                    "size": "1048576",
                    "definedAt": {"instructionName": "param.1"},
                },
                {
                    "size": "1048576",
                    "definedAt": {"instructionName": "param.2"},
                },
                {"size": "2097152", "definedAt": {"instructionName": "op1"}},
            ]
        }
    }
    buffers = get_peak_allocations_tool._parse_and_aggregate_buffers(
        parsed_mod_data, 1.0
    )
    self.assertEqual(
        [
            get_peak_allocations_tool.BufferAllocation(
                instruction="param.* (2 occurrences of size 1 MiB)",
                size_mib=2.0,
            ),
            get_peak_allocations_tool.BufferAllocation(
                instruction="op1", size_mib=2.0
            ),
        ],
        buffers,
    )

  def test_parse_and_aggregate_buffers_max_heap_fallback(self):
    parsed_mod_data = {
        "maxHeap": [
            {"logicalBufferSizeMib": 1.0, "instructionName": "param.1"},
            {"logicalBufferSizeMib": 1.0, "instructionName": "param.2"},
            {"logicalBufferSizeMib": 2.0, "instructionName": "op1"},
        ]
    }
    buffers = get_peak_allocations_tool._parse_and_aggregate_buffers(
        parsed_mod_data, 1.0
    )
    self.assertEqual(
        [
            get_peak_allocations_tool.BufferAllocation(
                instruction="param.* (2 occurrences of size 1 MiB)",
                size_mib=2.0,
            ),
            get_peak_allocations_tool.BufferAllocation(
                instruction="op1", size_mib=2.0
            ),
        ],
        buffers,
    )

  def test_parse_and_aggregate_buffers_no_aggregation(self):
    parsed_mod_data = {
        "bufferAssignment": {
            "logicalBuffers": [
                {
                    "size": "1048576",
                    "definedAt": {"instructionName": "param.1"},
                },
                {
                    "size": "1048576",
                    "definedAt": {"instructionName": "param.2"},
                },
                {"size": "2097152", "definedAt": {"instructionName": "op1"}},
                {
                    "size": "524288",
                    "definedAt": {"instructionName": "small_op"},
                },
            ]
        }
    }
    buffers = get_peak_allocations_tool._parse_and_aggregate_buffers(
        parsed_mod_data, 1.0, aggregate_instructions=False
    )
    self.assertEqual(
        [
            get_peak_allocations_tool.BufferAllocation(
                instruction="op1", size_mib=2.0
            ),
            get_peak_allocations_tool.BufferAllocation(
                instruction="param.1", size_mib=1.0
            ),
            get_peak_allocations_tool.BufferAllocation(
                instruction="param.2", size_mib=1.0
            ),
            get_peak_allocations_tool.BufferAllocation(
                instruction="small_op", size_mib=0.5
            ),
        ],
        buffers,
    )

  def test_fetch_modules_data(self):
    mock_client = mock.create_autospec(
        xprof_client.CachedXprofClient, instance=True, spec_set=True
    )
    mock_client.fetch.side_effect = [
        (
            b'{"totalBufferAllocationMib": 100.0, "bufferAssignment":'
            b' {"logicalBuffers": [{"size": "1048576", "definedAt":'
            b' {"instructionName": "op1"}}]}}'
        ),
        (
            b'{"totalBufferAllocationMib": 200.0, "bufferAssignment":'
            b' {"logicalBuffers": [{"size": "2097152", "definedAt":'
            b' {"instructionName": "op2"}}]}}'
        ),
    ]
    modules_data = get_peak_allocations_tool._fetch_modules_data(
        mock_client, "session_123", ["module1", "module2"], 1.0
    )
    self.assertEqual(
        [
            get_peak_allocations_tool.ModulePeakAllocations(
                module_name="module1",
                total_hbm_mib=100.0,
                top_buffers=[
                    get_peak_allocations_tool.BufferAllocation(
                        instruction="op1", size_mib=1.0
                    )
                ],
            ),
            get_peak_allocations_tool.ModulePeakAllocations(
                module_name="module2",
                total_hbm_mib=200.0,
                top_buffers=[
                    get_peak_allocations_tool.BufferAllocation(
                        instruction="op2", size_mib=2.0
                    )
                ],
            ),
        ],
        modules_data,
    )
    expected_calls = [
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
            module_name="module1",
        ),
        mock.call(
            tool_name="memory_viewer.json",
            session_id="session_123",
            format="json",
            module_name="module2",
        ),
    ]
    self.assertSequenceEqual(expected_calls, mock_client.fetch.call_args_list)


if __name__ == "__main__":
  absltest.main()
