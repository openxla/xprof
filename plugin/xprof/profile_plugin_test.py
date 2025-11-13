# -*- coding: utf-8 -*-
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the Profile plugin."""

# pylint: disable=missing-function-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import inspect
import logging
import os
import shutil
import tempfile
from unittest import mock

from absl.testing import absltest

from xprof import profile_plugin
from xprof import profile_plugin_test_utils as utils
from xprof import version
from xprof.convert import raw_to_tool_data as convert
from xprof.protobuf import trace_events_pb2
from xprof.standalone.tensorboard_shim import plugin_asset_util
from xprof.standalone.tensorboard_shim import plugin_event_multiplexer

RUN_TO_TOOLS = {
    'foo': ['trace_viewer', 'trace_viewer@'],
    'bar': ['unsupported'],
    'baz': ['overview_page', 'op_profile', 'trace_viewer', 'trace_viewer@'],
    'qux': [
        'overview_page',
        'input_pipeline_analyzer',
        'trace_viewer',
        'trace_viewer@',
    ],
    'abc': ['xplane'],
    'empty': [],
}

RUN_TO_HOSTS = {
    'foo': ['host0', 'host1'],
    'bar': ['host1'],
    'baz': ['host2'],
    'qux': [None],
    'abc': ['host1', 'host2'],
    'empty': [],
}

EXPECTED_TRACE_DATA = dict(
    displayTimeUnit='ns',
    metadata={'highres-ticks': True},
    traceEvents=[
        dict(ph='M', pid=0, name='process_name', args=dict(name='foo')),
        dict(ph='M', pid=0, name='process_sort_index', args=dict(sort_index=0)),
        dict(),
    ],
)

# Suffix for the empty eventfile to write. Should be kept in sync with TF
# profiler kProfileEmptySuffix constant defined in:
#   tensorflow/core/profiler/rpc/client/capture_profile.cc.
EVENT_FILE_SUFFIX = '.profile-empty'


# TODO(muditgokhale): Add support for xplane test generation.
def generate_testdata(logdir):
  plugin_logdir = plugin_asset_util.PluginDirectory(
      logdir, profile_plugin.ProfilePlugin.plugin_name)
  os.makedirs(plugin_logdir)
  for run in RUN_TO_TOOLS:
    run_dir = os.path.join(plugin_logdir, run)
    os.mkdir(run_dir)
    for tool in RUN_TO_TOOLS[run]:
      if (
          tool not in profile_plugin.XPLANE_TOOLS
          and tool not in profile_plugin.HLO_TOOLS
          and tool not in profile_plugin.TOOLS
      ):
        continue
      for host in RUN_TO_HOSTS[run]:
        filename = profile_plugin.make_filename(host, tool)
        tool_file = os.path.join(run_dir, filename)
        if tool in ('trace_viewer', 'trace_viewer@'):
          trace = trace_events_pb2.Trace()
          trace.devices[0].name = run
          data = trace.SerializeToString()
        else:
          data = tool.encode('utf-8')
        with open(tool_file, 'wb') as f:
          f.write(data)
  with open(os.path.join(plugin_logdir, 'noise'), 'w') as f:
    f.write('Not a dir, not a run.')


def write_empty_event_file(logdir):
  os.makedirs(logdir, exist_ok=True)
  open(os.path.join(logdir, 'events.out.tfevents.profile-empty'), 'a').close()


class ProfilePluginTest(absltest.TestCase):

  def setUp(self):
    super(ProfilePluginTest, self).setUp()
    self._temp_dir = None
    self.logdir = self.get_temp_dir()
    self.multiplexer = plugin_event_multiplexer.EventMultiplexer()
    self.multiplexer.AddRunsFromDirectory(self.logdir)
    self.plugin = utils.create_profile_plugin(self.logdir, self.multiplexer)

  def get_temp_dir(self):
    """Return a temporary directory for tests to use."""
    if not self._temp_dir:
      # If the test is running on Forge, use the TEST_UNDECLARED_OUTPUTS_DIR
      # environment variable to store the temporary directory for Sponge.
      if os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR'):
        temp_dir = tempfile.mkdtemp(
            dir=os.environ['TEST_UNDECLARED_OUTPUTS_DIR']
        )
      else:
        frame = inspect.stack()[-1]
        filename = frame.filename
        base_filename = os.path.basename(filename)
        temp_dir_prefix = os.path.join(
            tempfile.gettempdir(), base_filename.removesuffix('.py')
        )
        temp_dir = tempfile.mkdtemp(prefix=temp_dir_prefix)

      def delete_temp_dir(dirname=temp_dir):
        try:
          shutil.rmtree(dirname)
        except OSError as e:
          logging.error('Error removing %s: %s', dirname, e)

      atexit.register(delete_temp_dir)

      self._temp_dir = temp_dir

    return self._temp_dir

  def _set_up_side_effect(self):  # pylint: disable=g-unreachable-test-method
    # Fail if we call PluginDirectory with a non-normalized logdir path, since
    # that won't work on GCS, as a regression test for b/235606632.
    original_plugin_directory = plugin_asset_util.PluginDirectory
    plugin_directory_patcher = mock.patch.object(plugin_asset_util,
                                                 'PluginDirectory')
    mock_plugin_directory = plugin_directory_patcher.start()
    self.addCleanup(plugin_directory_patcher.stop)

    def plugin_directory_spy(logdir, plugin_name):
      if os.path.normpath(logdir) != logdir:
        self.fail(
            'PluginDirectory called with a non-normalized logdir path: %r' %
            logdir)
      return original_plugin_directory(logdir, plugin_name)

    mock_plugin_directory.side_effect = plugin_directory_spy

  def testRuns_logdirWithoutEventFile(self):
    generate_testdata(self.logdir)
    self.multiplexer.Reload()
    all_runs = list(self.plugin.generate_runs())
    self.assertSetEqual(frozenset(all_runs), frozenset(RUN_TO_HOSTS.keys()))

    self.assertEmpty(list(self.plugin.generate_tools_of_run('bar')))
    self.assertEmpty(list(self.plugin.generate_tools_of_run('empty')))

  def testRuns_logdirWithEventFIle(self):
    write_empty_event_file(self.logdir)
    generate_testdata(self.logdir)
    self.multiplexer.Reload()
    all_runs = self.plugin.generate_runs()
    self.assertSetEqual(frozenset(all_runs), frozenset(RUN_TO_HOSTS.keys()))

  def testRuns_withSubdirectories(self):
    subdir_a = os.path.join(self.logdir, 'a')
    subdir_b = os.path.join(self.logdir, 'b')
    subdir_b_c = os.path.join(subdir_b, 'c')
    generate_testdata(self.logdir)
    generate_testdata(subdir_a)
    generate_testdata(subdir_b)
    generate_testdata(subdir_b_c)
    write_empty_event_file(self.logdir)
    write_empty_event_file(subdir_a)
    # Skip writing an event file for subdir_b, so that we can test that it is
    # included in the runs regardless of tfevents file presence.
    write_empty_event_file(subdir_b_c)
    self.multiplexer.AddRunsFromDirectory(self.logdir)
    self.multiplexer.Reload()
    all_runs = list(self.plugin.generate_runs())
    # Expect runs for the logdir root, 'a', and 'b/c' but not for 'b'
    # because it doesn't contain a tfevents file.
    expected = set(RUN_TO_TOOLS.keys())
    expected.update(set('a/' + run for run in RUN_TO_TOOLS.keys()))
    expected.update(set('b/' + run for run in RUN_TO_TOOLS.keys()))
    expected.update(set('b/c/' + run for run in RUN_TO_TOOLS.keys()))
    self.assertSetEqual(frozenset(all_runs), expected)

  def testRuns_withoutEvents(self):
    generate_testdata(self.logdir)
    self.multiplexer.AddRunsFromDirectory(self.logdir)
    self.multiplexer.Reload()
    all_runs = list(self.plugin.generate_runs())
    expected = set(RUN_TO_TOOLS.keys())
    self.assertSetEqual(frozenset(all_runs), expected)

  def testRuns_withNestedRuns(self):
    subdir_date = os.path.join(self.logdir, '2024-12-19')
    subdir_train = os.path.join(subdir_date, 'train')
    subdir_validation = os.path.join(subdir_date, 'validation')
    # Write the plugin directory for the subdir_date directory.
    generate_testdata(subdir_date)
    # Write events files for the subdir_train and subdir_validation directories.
    write_empty_event_file(subdir_train)
    write_empty_event_file(subdir_validation)
    self.multiplexer.AddRunsFromDirectory(self.logdir)
    self.multiplexer.Reload()
    all_runs = list(self.plugin.generate_runs())
    # Expect runs for the subdir_date because it contains the plugin directory
    # and is the parent directory of the subdir_train and subdir_validation
    # directories.
    expected = set(set('2024-12-19/' + run for run in RUN_TO_TOOLS.keys()))
    self.assertSetEqual(frozenset(all_runs), expected)

  def testHosts(self):
    generate_testdata(self.logdir)
    subdir_a = os.path.join(self.logdir, 'a')
    generate_testdata(subdir_a)
    write_empty_event_file(subdir_a)
    self.multiplexer.AddRunsFromDirectory(self.logdir)
    self.multiplexer.Reload()
    expected_hosts_abc = [{'hostname': 'host1'}, {'hostname': 'host2'}]
    expected_all_hosts_only = [{'hostname': 'ALL_HOSTS'}]
    hosts_q = self.plugin.host_impl('qux', 'framework_op_stats')
    self.assertEmpty(hosts_q)
    hosts_abc_tf_stats = self.plugin.host_impl('abc', 'framework_op_stats')
    self.assertListEqual(
        expected_all_hosts_only + expected_hosts_abc, hosts_abc_tf_stats
    )
    # TraceViewer and MemoryProfile does not support all hosts.
    hosts_abc_trace_viewer = self.plugin.host_impl('abc', 'trace_viewer')
    self.assertListEqual(expected_hosts_abc, hosts_abc_trace_viewer)
    hosts_abc_memory_profile = self.plugin.host_impl('abc', 'memory_profile')
    self.assertListEqual(expected_hosts_abc, hosts_abc_memory_profile)
    # OverviewPage supports all hosts only.
    hosts_abc_overview_page = self.plugin.host_impl('abc', 'overview_page')
    self.assertListEqual(expected_all_hosts_only, hosts_abc_overview_page)
    # PodViewer supports all hosts only.
    hosts_abc_pod_viewer = self.plugin.host_impl('abc', 'pod_viewer')
    self.assertListEqual(expected_all_hosts_only, hosts_abc_pod_viewer)

  def testData(self):
    generate_testdata(self.logdir)
    subdir_a = os.path.join(self.logdir, 'a')
    generate_testdata(subdir_a)
    write_empty_event_file(subdir_a)
    self.multiplexer.AddRunsFromDirectory(self.logdir)
    self.multiplexer.Reload()

    # Invalid tool/run/host.
    data, _, _ = self.plugin.data_impl(
        utils.make_data_request(
            utils.DataRequestOptions(
                run='foo', tool='invalid_tool', host='host0'
            )
        )
    )
    self.assertIsNone(data)
    data, _, _ = self.plugin.data_impl(
        utils.make_data_request(
            utils.DataRequestOptions(
                run='foo', tool='memory_viewer', host='host0'
            )
        )
    )
    self.assertIsNone(data)
    with self.assertRaises(FileNotFoundError):
      self.plugin.data_impl(
          utils.make_data_request(
              utils.DataRequestOptions(run='foo', tool='trace_viewer', host='')
          )
      )
    data, _, _ = self.plugin.data_impl(
        utils.make_data_request(
            utils.DataRequestOptions(
                run='bar', tool='unsupported', host='host1'
            )
        )
    )
    self.assertIsNone(data)
    with self.assertRaises(FileNotFoundError):
      self.plugin.data_impl(
          utils.make_data_request(
              utils.DataRequestOptions(
                  run='bar', tool='trace_viewer', host='host0'
              )
          )
      )
    with self.assertRaises(FileNotFoundError):
      self.plugin.data_impl(
          utils.make_data_request(
              utils.DataRequestOptions(
                  run='qux', tool='trace_viewer', host='host'
              )
          )
      )
    with self.assertRaises(FileNotFoundError):
      self.plugin.data_impl(
          utils.make_data_request(
              utils.DataRequestOptions(
                  run='empty', tool='trace_viewer', host=''
              )
          )
      )
    with self.assertRaises(FileNotFoundError):
      self.plugin.data_impl(
          utils.make_data_request(
              utils.DataRequestOptions(
                  run='a/foo', tool='trace_viewer', host=''
              )
          )
      )

  def testDataWithCache(self):
    generate_testdata(self.logdir)
    self.multiplexer.AddRunsFromDirectory(self.logdir)
    self.multiplexer.Reload()
    run_dir = os.path.join(
        plugin_asset_util.PluginDirectory(
            self.logdir, profile_plugin.ProfilePlugin.plugin_name
        ),
        'baz',
    )
    cache_version_file_path = os.path.join(
        run_dir, profile_plugin.CACHE_VERSION_FILE
    )

    # Check if the cache_version.txt file doesn't exists.
    self.assertFalse(os.path.exists(cache_version_file_path))

    # Check if first run generates a cache file.
    _, _, _ = self.plugin.data_impl(
        utils.make_data_request(
            utils.DataRequestOptions(
                run='baz', tool='overview_page', host='host2'
            )
        )
    )
    self.assertTrue(os.path.exists(cache_version_file_path))
    with open(cache_version_file_path, 'r') as f:
      self.assertEqual(f.read(), version.__version__)
    cache_file_first_run_timestamp = os.path.getmtime(cache_version_file_path)

    # Check if the second run generates a cache file.
    _, _, _ = self.plugin.data_impl(
        utils.make_data_request(
            utils.DataRequestOptions(
                run='baz', tool='overview_page', host='host2'
            )
        )
    )
    self.assertTrue(os.path.exists(cache_version_file_path))
    with open(cache_version_file_path, 'r') as f:
      self.assertEqual(f.read(), version.__version__)
    self.assertEqual(
        cache_file_first_run_timestamp,
        os.path.getmtime(cache_version_file_path),
    )

    # Check if the use_saved_result=False generates a cache file.
    _, _, _ = self.plugin.data_impl(
        utils.make_data_request(
            utils.DataRequestOptions(
                run='baz',
                tool='overview_page',
                host='host2',
                use_saved_result='False',
            )
        )
    )
    self.assertTrue(os.path.exists(cache_version_file_path))
    with open(cache_version_file_path, 'r') as f:
      self.assertEqual(f.read(), version.__version__)
    self.assertLess(
        cache_file_first_run_timestamp,
        os.path.getmtime(cache_version_file_path),
    )

    # Overwrite the cache_version.txt file with an old version.
    with open(cache_version_file_path, 'w') as f:
      f.write('1.0.0')
    _, _, _ = self.plugin.data_impl(
        utils.make_data_request(
            utils.DataRequestOptions(
                run='baz', tool='overview_page', host='host2'
            )
        )
    )
    self.assertTrue(os.path.exists(cache_version_file_path))
    with open(cache_version_file_path, 'r') as f:
      self.assertEqual(f.read(), version.__version__)

  @mock.patch.object(convert, 'xspace_to_tool_data', autospec=True)
  def testDataImplTraceViewerOptions(self, mock_xspace_to_tool_data):
    generate_testdata(self.logdir)
    self.multiplexer.AddRunsFromDirectory(self.logdir)
    self.multiplexer.Reload()
    # Mock the return value to avoid errors during the call
    mock_xspace_to_tool_data.return_value = ('mocked_data', 'application/json')
    expected_asset_path = os.path.join(
        self.plugin._run_dir('foo'),
        profile_plugin.make_filename('host1', 'trace_viewer@'),
    )
    expected_params = {
        'graph_viewer_options': {
            'node_name': None,
            'module_name': None,
            'graph_width': 3,
            'show_metadata': 0,
            'merge_fusion': 0,
            'format': None,
            'type': None,
        },
        'tqx': None,
        'host': 'host1',
        'module_name': None,
        'use_saved_result': False,
        'memory_space': '0',
        'trace_viewer_options': {
            'resolution': '10000',
            'full_dma': True,
            'start_time_ms': '100',
            'end_time_ms': '200',
        },
        'hosts': ['host1'],
    }

    _, _, _ = self.plugin.data_impl(
        utils.make_data_request(
            utils.DataRequestOptions(
                run='foo',
                tool='trace_viewer@',
                host='host1',
                full_dma='true',
                resolution='10000',
                start_time_ms='100',
                end_time_ms='200',
            )
        )
    )

    mock_xspace_to_tool_data.assert_called_once_with(
        [mock.ANY], 'trace_viewer@', expected_params
    )
    args, _ = mock_xspace_to_tool_data.call_args
    actual_path_list = args[0]
    self.assertLen(actual_path_list, 1)
    self.assertEqual(str(actual_path_list[0]), expected_asset_path)

  def testActive(self):

    def wait_for_thread():
      with self.plugin._is_active_lock:
        pass

    # Launch thread to check if active.
    self.plugin.is_active()
    wait_for_thread()
    # Should be false since there's no data yet.
    self.assertFalse(self.plugin.is_active())
    wait_for_thread()
    generate_testdata(self.logdir)
    self.multiplexer.Reload()
    # Launch a new thread to check if active.
    self.plugin.is_active()
    wait_for_thread()
    # Now that there's data, this should be active.
    self.assertTrue(self.plugin.is_active())

  def testActive_subdirectoryOnly(self):

    def wait_for_thread():
      with self.plugin._is_active_lock:
        pass

    # Launch thread to check if active.
    self.plugin.is_active()
    wait_for_thread()
    # Should be false since there's no data yet.
    self.assertFalse(self.plugin.is_active())
    wait_for_thread()
    subdir_a = os.path.join(self.logdir, 'a')
    generate_testdata(subdir_a)
    write_empty_event_file(subdir_a)
    self.multiplexer.AddRunsFromDirectory(self.logdir)
    self.multiplexer.Reload()
    # Launch a new thread to check if active.
    self.plugin.is_active()
    wait_for_thread()
    # Now that there's data, this should be active.
    self.assertTrue(self.plugin.is_active())

  def test_generate_runs_from_path_params_with_session(self):
    session_path = os.path.join(self.logdir, 'session_run')
    os.mkdir(session_path)
    with open(os.path.join(session_path, 'host.xplane.pb'), 'w') as f:
      f.write('dummy xplane data')
    runs = list(
        self.plugin._generate_runs_from_path_params(session_path=session_path)
    )
    self.assertListEqual(['session_run'], runs)
    self.assertEqual(self.logdir, self.plugin.logdir)

  def test_generate_runs_no_logdir(self):
    self.plugin.logdir = None
    self.plugin.basedir = None
    runs = list(self.plugin.generate_runs())
    self.assertEmpty(runs)

  def test_generate_runs_from_path_params_with_run_path(self):
    run_path = os.path.join(self.logdir, 'base')
    os.mkdir(run_path)
    run1_path = os.path.join(run_path, 'run1')
    os.mkdir(run1_path)
    with open(os.path.join(run1_path, 'host.xplane.pb'), 'w') as f:
      f.write('dummy xplane data')
    run2_path = os.path.join(run_path, 'run2')
    os.mkdir(run2_path)
    # run3 is a file, not a directory, and should be ignored.
    with open(os.path.join(run_path, 'run3'), 'w') as f:
      f.write('dummy file')
    with open(os.path.join(run2_path, 'host2.xplane.pb'), 'w') as f:
      f.write('dummy xplane data for run2')
    runs = list(self.plugin._generate_runs_from_path_params(run_path=run_path))
    self.assertListEqual(['run1', 'run2'], sorted(runs))
    self.assertEqual(run_path, self.plugin.logdir)

  def test_runs_impl_with_session(self):
    session_path = os.path.join(self.logdir, 'session_run')
    os.mkdir(session_path)
    with open(os.path.join(session_path, 'host.xplane.pb'), 'w') as f:
      f.write('dummy xplane data')
    request = utils.make_data_request(
        utils.DataRequestOptions(session_path=session_path)
    )
    runs = self.plugin.runs_imp(request)
    self.assertListEqual(['session_run'], runs)
    self.assertEqual(os.path.dirname(session_path), self.plugin.logdir)

  def test_runs_impl_with_run_path(self):
    run_path = os.path.join(self.logdir, 'base')
    os.mkdir(run_path)
    run1_path = os.path.join(run_path, 'run1')
    os.mkdir(run1_path)
    with open(os.path.join(run1_path, 'host.xplane.pb'), 'w') as f:
      f.write('dummy xplane data')
    request = utils.make_data_request(
        utils.DataRequestOptions(run_path=run_path)
    )
    runs = self.plugin.runs_imp(request)
    self.assertListEqual(['run1'], runs)
    self.assertEqual(run_path, self.plugin.logdir)

  def test_run_dir_no_logdir(self):
    self.plugin.logdir = None
    with self.assertRaisesRegex(
        RuntimeError, 'No matching run directory for run foo'
    ):
      self.plugin._run_dir('foo')

  def test_run_dir_invalid_profile_run_directory(self):
    # This test verifies that no error is raised if the TB run directory exists,
    # even if the specific profile run subfolder does not.
    expected_path = os.path.join(
        self.logdir, 'plugins', 'profile', 'non_existent_run'
    )
    run_dir = self.plugin._run_dir('non_existent_run')
    self.assertEqual(run_dir, expected_path)

  def test_run_dir_invalid_tb_run_directory(self):
    with self.assertRaisesRegex(
        RuntimeError,
        'No matching run directory for run non_existent_tb_run/run1',
    ):
      self.plugin._run_dir('non_existent_tb_run/run1')

  def test_run_dir_with_custom_session(self):
    self.plugin.custom_session_path = os.path.join(self.logdir, 'session_run')
    os.mkdir(self.plugin.custom_session_path)
    run_dir = self.plugin._run_dir('session_run')
    self.assertEqual(
        run_dir, os.path.join(self.logdir, 'session_run')
    )

  def test_run_dir_with_custom_run_path(self):
    self.plugin.custom_run_path = os.path.join(self.logdir, 'base')
    os.mkdir(self.plugin.custom_run_path)
    run_dir = self.plugin._run_dir('base/run1')
    self.assertEqual(run_dir, os.path.join(self.logdir, 'base', 'run1'))

  def test_run_dir_default(self):
    run_path = os.path.join(self.logdir, 'train')
    os.mkdir(run_path)
    plugin_dir = os.path.join(run_path, 'plugins', 'profile')
    os.makedirs(plugin_dir)
    run1_path = os.path.join(plugin_dir, 'run1')
    os.mkdir(run1_path)
    run_dir = self.plugin._run_dir('train/run1')
    self.assertEqual(run_dir, run1_path)


if __name__ == '__main__':
  absltest.main()
