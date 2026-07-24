"""Unit tests for tool option builders and counter-names handler."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from xprof.profile_plugin.models import ToolRequest
from xprof.profile_plugin.services.counter_names import (
    counter_names_result,
    try_counter_names_only,
)
from xprof.profile_plugin.tools.options import build_tool_params
from xprof.profile_plugin.tools.options.graph import build_graph_viewer_options


def _req(
    tool: str,
    raw_args: dict | None = None,
    *,
    host: str | None = 'h0',
    use_saved_result: bool = True,
) -> ToolRequest:
  args = dict(raw_args or {})
  args.setdefault('tag', tool)
  args.setdefault('run', 'r')
  return ToolRequest(
      run='r',
      tool=tool,
      host=host,
      hosts=(),
      use_saved_result=use_saved_result,
      raw_args=args,
  )


class ToolOptionsTest(unittest.TestCase):

  def test_trace_event_name_forces_json_format(self):
    req = ToolRequest(
        run='r',
        tool='trace_viewer@',
        host='h0',
        hosts=(),
        use_saved_result=True,
        raw_args={
            'tag': 'trace_viewer@',
            'run': 'r',
            'event_name': 'op',
            'format': 'proto',
            'resolution': '4000',
        },
    )
    params = build_tool_params(req, use_saved_result=True, graph_options={})
    tv = params['trace_viewer_options']
    self.assertEqual(tv['format'], 'json')
    self.assertEqual(tv['event_name'], 'op')

  def test_trace_format_without_event_name(self):
    req = _req(
        'trace_viewer@',
        {'format': 'proto', 'resolution': '1000'},
    )
    params = build_tool_params(req, use_saved_result=True, graph_options={})
    tv = params['trace_viewer_options']
    self.assertEqual(tv['format'], 'proto')
    self.assertNotIn('event_name', tv)
    self.assertEqual(tv['resolution'], '1000')

  def test_common_params_and_memory_space_default(self):
    req = _req('overview_page', {'tqx': 'out:json', 'perfetto': 'true'})
    params = build_tool_params(req, use_saved_result=False, graph_options={})
    self.assertEqual(params['tqx'], 'out:json')
    self.assertTrue(params['perfetto'])
    self.assertFalse(params['use_saved_result'])
    self.assertEqual(params['host'], 'h0')
    self.assertEqual(params['memory_space'], '0')
    self.assertNotIn('trace_viewer_options', params)
    self.assertNotIn('view_memory_allocation_timeline', params)

  def test_memory_allocation_timeline(self):
    req = _req(
        'memory_viewer',
        {'view_memory_allocation_timeline': '1', 'memory_space': '2'},
    )
    params = build_tool_params(req, graph_options={})
    self.assertTrue(params['view_memory_allocation_timeline'])
    self.assertEqual(params['memory_space'], '2')

  def test_graph_viewer_options_defaults(self):
    opts = build_graph_viewer_options({})
    self.assertEqual(opts['graph_width'], 3)
    self.assertEqual(opts['show_metadata'], 0)
    self.assertEqual(opts['merge_fusion'], 0)
    self.assertIsNone(opts['node_name'])

  def test_graph_viewer_options_from_args(self):
    opts = build_graph_viewer_options({
        'node_name': 'n1',
        'module_name': 'm',
        'graph_width': '5',
        'show_metadata': 'true',
        'merge_fusion': 'true',
        'program_id': 'p1',
        'format': 'json',
        'type': 'hlo',
    })
    self.assertEqual(opts['node_name'], 'n1')
    self.assertEqual(opts['graph_width'], 5)
    self.assertEqual(opts['show_metadata'], 1)
    self.assertEqual(opts['merge_fusion'], 1)
    self.assertEqual(opts['format'], 'json')
    self.assertEqual(opts['type'], 'hlo')

  def test_build_tool_params_builds_graph_options_when_omitted(self):
    req = _req('graph_viewer', {'node_name': 'op0', 'graph_width': '7'})
    params = build_tool_params(req)
    self.assertEqual(params['graph_viewer_options']['node_name'], 'op0')
    self.assertEqual(params['graph_viewer_options']['graph_width'], 7)

  def test_group_by_and_refresh_suggestion_optional(self):
    req = _req(
        'op_profile',
        {'group_by': 'category', 'refresh_suggestion': '1'},
    )
    params = build_tool_params(req, graph_options={})
    self.assertEqual(params['group_by'], 'category')
    self.assertEqual(params['refresh_suggestion'], '1')


class CounterNamesTest(unittest.TestCase):

  def test_try_counter_names_only_returns_none_when_not_applicable(self):
    req = _req('overview_page')
    self.assertIsNone(try_counter_names_only(req))

    req = _req('perf_counters', {'device_type': 'tpu'})
    self.assertIsNone(try_counter_names_only(req))

    req = _req('perf_counters', {'names_only': '0', 'device_type': 'tpu'})
    self.assertIsNone(try_counter_names_only(req))

  def test_try_counter_names_only_requires_device_type(self):
    req = _req('perf_counters', {'names_only': '1'})
    with self.assertRaises(ValueError) as ctx:
      try_counter_names_only(req)
    self.assertIn('device_type', str(ctx.exception))

  @mock.patch(
      'xprof.convert.counter_extractor.get_all_counters',
      create=True,
  )
  def test_try_counter_names_only_success(self, mock_get):
    mock_get.return_value = ['cycles', 'flops']
    # Ensure import path works via counter_extractor module mock.
    import sys
    import types

    fake = types.ModuleType('xprof.convert.counter_extractor')
    fake.get_all_counters = mock_get
    with mock.patch.dict(
        sys.modules,
        {
            'xprof.convert.counter_extractor': fake,
            'xprof.convert': sys.modules.get(
                'xprof.convert', types.ModuleType('xprof.convert')
            ),
        },
    ):
      # Re-bind convert package attribute if needed.
      convert_mod = sys.modules['xprof.convert']
      convert_mod.counter_extractor = fake
      req = _req(
          'perf_counters',
          {'names_only': '1', 'device_type': 'tpu_v5e'},
      )
      result = try_counter_names_only(req)
    self.assertIsNotNone(result)
    self.assertEqual(result.content_type, 'application/json')
    self.assertEqual(json.loads(result.data), ['cycles', 'flops'])
    mock_get.assert_called_once_with('tpu_v5e')

  def test_counter_names_result_file_not_found(self):
    import sys
    import types

    def _raise(_device_type):
      raise FileNotFoundError('missing')

    fake = types.ModuleType('xprof.convert.counter_extractor')
    fake.get_all_counters = _raise
    convert_mod = sys.modules.get(
        'xprof.convert', types.ModuleType('xprof.convert')
    )
    convert_mod.counter_extractor = fake
    with mock.patch.dict(
        sys.modules,
        {
            'xprof.convert': convert_mod,
            'xprof.convert.counter_extractor': fake,
        },
    ):
      result = counter_names_result('unknown')
    self.assertEqual(result.content_type, 'application/json')
    self.assertEqual(json.loads(result.data), [])


if __name__ == '__main__':
  unittest.main()
