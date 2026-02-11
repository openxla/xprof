# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

from unittest import mock
from absl.testing import absltest
from xprof.api import continuous_profiling_snapshot
from xprof.convert import _pywrap_profiler_plugin


class ContinuousProfilingSnapshotTest(absltest.TestCase):

  def test_start_continuous_profiling_with_options(self):
    mock_plugin = mock.create_autospec(_pywrap_profiler_plugin, spec_set=True)
    profiler = continuous_profiling_snapshot.ContinuousProfilingSnapshot(
        mock_plugin
    )

    profiler.start_continuous_profiling('localhost:1234', {'duration_ms': 1000})
    mock_plugin.start_continuous_profiling.assert_called_once_with(
        'localhost:1234', {'duration_ms': 1000}
    )

  def test_start_continuous_profiling_with_default_options(self):
    mock_plugin = mock.create_autospec(_pywrap_profiler_plugin, spec_set=True)
    profiler = continuous_profiling_snapshot.ContinuousProfilingSnapshot(
        mock_plugin
    )

    profiler.start_continuous_profiling('localhost:1234', {})
    mock_plugin.start_continuous_profiling.assert_called_once_with(
        'localhost:1234', {}
    )

  def test_stop_continuous_profiling(self):
    mock_plugin = mock.create_autospec(_pywrap_profiler_plugin, spec_set=True)
    profiler = continuous_profiling_snapshot.ContinuousProfilingSnapshot(
        mock_plugin
    )
    profiler.stop_continuous_profiling('localhost:1234')
    mock_plugin.stop_continuous_profiling.assert_called_once_with(
        'localhost:1234'
    )

  def test_get_snapshot(self):
    mock_plugin = mock.create_autospec(_pywrap_profiler_plugin, spec_set=True)
    profiler = continuous_profiling_snapshot.ContinuousProfilingSnapshot(
        mock_plugin
    )
    profiler.get_snapshot('localhost:1234', '/tmp/logdir')
    mock_plugin.get_snapshot.assert_called_once_with(
        'localhost:1234', '/tmp/logdir'
    )

  def test_module_level_functions(self):
    # Verify that module-level functions use the default profiler and plugin.
    # We patch the plugin instance on the default profiler to verify calls.
    with mock.patch.object(
        continuous_profiling_snapshot._default_profiler,
        '_profiler_plugin',
        autospec=True,
        spec_set=True,
    ) as mock_plugin:
      continuous_profiling_snapshot.start_continuous_profiling(
          'localhost:1234', {'duration_ms': 1000}
      )
      mock_plugin.start_continuous_profiling.assert_called_once_with(
          'localhost:1234', {'duration_ms': 1000}
      )

      continuous_profiling_snapshot.stop_continuous_profiling('localhost:1234')
      mock_plugin.stop_continuous_profiling.assert_called_once_with(
          'localhost:1234'
      )

      continuous_profiling_snapshot.get_snapshot(
          'localhost:1234', '/tmp/logdir'
      )
      mock_plugin.get_snapshot.assert_called_once_with(
          'localhost:1234', '/tmp/logdir'
      )


if __name__ == '__main__':
  absltest.main()
