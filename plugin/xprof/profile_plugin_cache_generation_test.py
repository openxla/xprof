# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for cache generation in ProfilePlugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import concurrent.futures
import gzip
import os
from unittest import mock

from absl.testing import absltest
from etils import epath
from werkzeug import wrappers

from xprof import profile_plugin
from xprof import version


class ProfilePluginCacheGenerationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.logdir = self.create_tempdir().full_path
    self.context = mock.Mock()
    self.context.logdir = self.logdir
    self.context.data_provider = mock.Mock()
    self.context.flags = mock.Mock()
    self.context.flags.master_tpu_unsecure_channel = ''

    # Mock executor
    self.mock_executor = mock.create_autospec(
        concurrent.futures.ThreadPoolExecutor
    )
    self.mock_executor.submit.return_value = None

    # Mock xspace_to_tool_data_fn
    self.mock_xspace_converter = mock.Mock()

    self.plugin = profile_plugin.ProfilePlugin(
        self.context,
        epath_module=epath,
        xspace_to_tool_data_fn=self.mock_xspace_converter,
        version_module=version,
        cache_generation_executor=self.mock_executor,
    )

    # Helper to create dummy xplane files
    self.session_dir = os.path.join(self.logdir, 'session_1')
    epath.Path(self.session_dir).mkdir()
    self.xplane_file = os.path.join(self.session_dir, 'host1.xplane.pb')
    epath.Path(self.xplane_file).write_text('dummy data')

  def _decode_response(self, response):
    """Decodes the response data, handling gzip compression if necessary."""
    data = response.data
    if response.headers.get('Content-Encoding') == 'gzip':
      data = gzip.decompress(data)
    if isinstance(data, bytes):
      return data.decode('utf-8')
    return data

  def test_generate_cache_route_method_not_allowed(self):
    request = wrappers.Request.from_values(method='GET')
    response = self.plugin.generate_cache_impl(request)
    self.assertEqual(response.status_code, 405)

  def test_generate_cache_route_missing_session_path(self):
    request = wrappers.Request.from_values(method='POST', query_string={})
    response = self.plugin.generate_cache_impl(request)
    self.assertEqual(response.status_code, 400)
    self.assertIn('Missing "session_path"', self._decode_response(response))

  def test_generate_cache_route_no_xplane_files(self):
    empty_session_dir = os.path.join(self.logdir, 'empty_session')
    epath.Path(empty_session_dir).mkdir()
    request = wrappers.Request.from_values(
        method='POST', query_string={'session_path': empty_session_dir}
    )
    response = self.plugin.generate_cache_impl(request)
    self.assertEqual(response.status_code, 404)
    self.assertIn('No XPlane files found', self._decode_response(response))

  def test_generate_cache_route_file_listing_error(self):
    # Mock self.plugin._epath.Path to return a mock that raises OSError on glob
    with mock.patch.object(self.plugin._epath, 'Path') as mock_path_cls:
      mock_path_instance = mock_path_cls.return_value
      mock_path_instance.glob.side_effect = OSError('glob error')

      request = wrappers.Request.from_values(
          method='POST', query_string={'session_path': self.session_dir}
      )
      response = self.plugin.generate_cache_impl(request)
      self.assertEqual(response.status_code, 500)
      self.assertIn('Error listing files', self._decode_response(response))

  def test_generate_cache_route_no_runs_found(self):
    # Mock runs_imp to return empty list
    with mock.patch.object(self.plugin, 'runs_imp', return_value=[]):
      request = wrappers.Request.from_values(
          method='POST', query_string={'session_path': self.session_dir}
      )
      response = self.plugin.generate_cache_impl(request)
      self.assertEqual(response.status_code, 404)
      self.assertIn('No runs found', self._decode_response(response))

  def test_generate_cache_route_no_valid_tools(self):
    # Mock runs_imp and run_tools_imp
    with mock.patch.object(self.plugin, 'runs_imp', return_value=['run1']):
      with mock.patch.object(
          self.plugin, 'run_tools_imp', return_value=['invalid_tool']
      ):
        request = wrappers.Request.from_values(
            method='POST',
            query_string={
                'session_path': self.session_dir,
                'tools': 'requested_tool',
            },
        )
        response = self.plugin.generate_cache_impl(request)
        self.assertEqual(response.status_code, 400)
        self.assertIn(
            'No valid XPlane tools found', self._decode_response(response)
        )

  def test_generate_cache_route_success_default_tools(self):
    # Mock runs_imp and run_tools_imp.
    # trace_viewer@ is in DEFAULT_CACHE_TOOLS and XPLANE_TOOLS_SET
    with mock.patch.object(self.plugin, 'runs_imp', return_value=['run1']):
      with mock.patch.object(
          self.plugin,
          'run_tools_imp',
          return_value=['trace_viewer@', 'overview_page'],
      ):
        request = wrappers.Request.from_values(
            method='POST', query_string={'session_path': self.session_dir}
        )
        response = self.plugin.generate_cache_impl(request)
        self.assertEqual(response.status_code, 202)

        # Verify executor.submit called
        self.mock_executor.submit.assert_called_once()
        args, kwargs = self.mock_executor.submit.call_args
        self.assertEqual(args[0], self.plugin._generate_cache_task)
        self.assertEqual(kwargs['session_path'], self.session_dir)
        # Check tool list contains defaults intersected with available
        self.assertCountEqual(
            kwargs['tool_list'], ['trace_viewer@', 'overview_page']
        )

  def test_generate_cache_route_success_specific_tools(self):
    with mock.patch.object(self.plugin, 'runs_imp', return_value=['run1']):
      with mock.patch.object(
          self.plugin,
          'run_tools_imp',
          return_value=['trace_viewer@', 'op_profile'],
      ):
        request = wrappers.Request.from_values(
            method='POST',
            query_string={
                'session_path': self.session_dir,
                'tools': 'trace_viewer@, op_profile',
            },
        )
        response = self.plugin.generate_cache_impl(request)
        self.assertEqual(response.status_code, 202)

        _, kwargs = self.mock_executor.submit.call_args
        self.assertCountEqual(
            kwargs['tool_list'], ['trace_viewer@', 'op_profile']
        )

  def test_generate_cache_route_filters_tools(self):
    # Request A and B. Available A. B not available.
    with mock.patch.object(self.plugin, 'runs_imp', return_value=['run1']):
      with mock.patch.object(
          self.plugin,
          'run_tools_imp',
          return_value=['trace_viewer@'],
      ):
        request = wrappers.Request.from_values(
            method='POST',
            query_string={
                'session_path': self.session_dir,
                'tools': 'trace_viewer@, op_profile',
            },
        )
        response = self.plugin.generate_cache_impl(request)
        self.assertEqual(response.status_code, 202)

        _, kwargs = self.mock_executor.submit.call_args
        self.assertCountEqual(kwargs['tool_list'], ['trace_viewer@'])

  def test_generate_cache_route_executor_failure(self):
    self.mock_executor.submit.side_effect = RuntimeError('executor failed')
    with mock.patch.object(self.plugin, 'runs_imp', return_value=['run1']):
      with mock.patch.object(
          self.plugin,
          'run_tools_imp',
          return_value=['trace_viewer@'],
      ):
        request = wrappers.Request.from_values(
            method='POST', query_string={'session_path': self.session_dir}
        )
        response = self.plugin.generate_cache_impl(request)
        self.assertEqual(response.status_code, 500)
        self.assertIn(
            'Failed to schedule task', self._decode_response(response)
        )

  def test_generate_cache_task_success(self):
    tool_list = ['trace_viewer@', 'overview_page']
    asset_paths = [self.xplane_file]
    params = {'foo': 'bar'}

    self.plugin._generate_cache_task(
        asset_paths=asset_paths,
        tool_list=tool_list,
        params=params,
        session_path=self.session_dir,
    )

    # Verify cache version file written
    cache_version_file = (
        epath.Path(self.session_dir) / profile_plugin.CACHE_VERSION_FILE
    )
    self.assertTrue(cache_version_file.exists())
    self.assertEqual(cache_version_file.read_text(), version.__version__)

    # Verify converter called for each tool
    self.assertEqual(self.mock_xspace_converter.call_count, 2)
    # Check call args
    calls = self.mock_xspace_converter.call_args_list
    tools_called = [c[0][1] for c in calls]
    self.assertCountEqual(tools_called, tool_list)

    # Check params passed (hosts should be added)
    passed_params = calls[0][0][2]
    self.assertEqual(passed_params['foo'], 'bar')
    self.assertIn('hosts', passed_params)

  def test_generate_cache_task_partial_failure(self):
    tool_list = ['trace_viewer@', 'overview_page']
    asset_paths = [self.xplane_file]

    # Make converter fail for first tool, succeed for second
    def side_effect(tool):
      if tool == 'trace_viewer@':
        raise ValueError('Conversion failed')
      return ('data', 'json')

    self.mock_xspace_converter.side_effect = side_effect

    # Should not raise
    self.plugin._generate_cache_task(
        asset_paths=asset_paths,
        tool_list=tool_list,
        params={},
        session_path=self.session_dir,
    )

    # Both attempted
    self.assertEqual(self.mock_xspace_converter.call_count, 2)

  def test_generate_cache_task_unexpected_error(self):
    self.mock_xspace_converter.side_effect = Exception('Unexpected')
    # Should not raise
    self.plugin._generate_cache_task(
        asset_paths=[self.xplane_file],
        tool_list=['trace_viewer@'],
        params={},
        session_path=self.session_dir,
    )


if __name__ == '__main__':
  absltest.main()
