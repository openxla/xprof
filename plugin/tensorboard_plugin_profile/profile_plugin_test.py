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

from tensorboard_plugin_profile import profile_plugin
from tensorboard_plugin_profile import profile_plugin_test_utils as utils
from tensorboard_plugin_profile.protobuf import trace_events_pb2
from tensorboard_plugin_profile.standalone.tensorboard_shim import plugin_asset_util
from tensorboard_plugin_profile.standalone.tensorboard_shim import plugin_event_multiplexer

RUN_TO_TOOLS = {
    'foo': ['trace_viewer'],
    'bar': ['unsupported'],
    'baz': ['overview_page', 'op_profile', 'trace_viewer'],
    'qux': ['overview_page', 'input_pipeline_analyzer', 'trace_viewer'],
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
      if tool not in profile_plugin.TOOLS:
        continue
      for host in RUN_TO_HOSTS[run]:
        filename = profile_plugin.make_filename(host, tool)
        tool_file = os.path.join(run_dir, filename)
        if tool == 'trace_viewer':
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
      if os.environ.get('TEST_TMPDIR'):
        temp_dir = tempfile.mkdtemp(prefix=os.environ['TEST_TMPDIR'])
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
    self.assertEmpty(
        list(self.plugin.generate_tools_of_run('baz')), RUN_TO_TOOLS['baz'])
    self.assertEmpty(
        list(self.plugin.generate_tools_of_run('qux')), RUN_TO_TOOLS['qux'])
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
    data, _, _ = self.plugin.data_impl(utils.make_data_request(
        run='foo', tool='invalid_tool', host='host0'))
    self.assertIsNone(data)
    data, _, _ = self.plugin.data_impl(utils.make_data_request(
        run='foo', tool='memory_viewer', host='host0'))
    self.assertIsNone(data)
    with self.assertRaises(FileNotFoundError):
      self.plugin.data_impl(utils.make_data_request(
          run='foo', tool='trace_viewer', host=''))
    data, _, _ = self.plugin.data_impl(utils.make_data_request(
        run='bar', tool='unsupported', host='host1'))
    self.assertIsNone(data)
    with self.assertRaises(FileNotFoundError):
      self.plugin.data_impl(utils.make_data_request(
          run='bar', tool='trace_viewer', host='host0'))
    with self.assertRaises(FileNotFoundError):
      self.plugin.data_impl(utils.make_data_request(
          run='qux', tool='trace_viewer', host='host'))
    with self.assertRaises(FileNotFoundError):
      self.plugin.data_impl(utils.make_data_request(
          run='empty', tool='trace_viewer', host=''))
    with self.assertRaises(FileNotFoundError):
      self.plugin.data_impl(utils.make_data_request(
          run='a', tool='trace_viewer', host=''))

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


if __name__ == '__main__':
  absltest.main()
