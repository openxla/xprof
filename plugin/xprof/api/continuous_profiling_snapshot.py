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
"""Continuous profiling Snapshot APIs."""

from collections.abc import Mapping
import dataclasses
import os
from typing import Any

from xprof.convert import _pywrap_profiler_plugin


@dataclasses.dataclass
class ContinuousProfilingSnapshot:
  """A handler for continuous profiling."""

  _profiler_plugin: Any = dataclasses.field(
      default_factory=lambda: _pywrap_profiler_plugin
  )

  def start_continuous_profiling(
      self, service_addr: str, options: Mapping[str, Any]
  ) -> None:
    """Starts continuous profiling on the profiling service.

    Args:
      service_addr: Address of the profiling service (e.g. localhost:6006).
      options: A dictionary of profiling options derived from
        tensorflow.compiler.xla.tsl.profiler ProfileOptions. Expected keys
        include 'include_dataset_ops','repository_path',
        'raise_error_on_start_failure', 'advanced_configuration', etc. Example:
        `{'duration_ms': 1000}`.
    """

    self._profiler_plugin.start_continuous_profiling(service_addr, options)

  def stop_continuous_profiling(self, service_addr: str) -> None:
    """Stops continuous profiling on the profiling service.

    Args:
      service_addr: Address of the profiling service (e.g. localhost:6006).
    """
    self._profiler_plugin.stop_continuous_profiling(service_addr)

  def get_snapshot(self, service_addr: str, logdir: os.PathLike[str]) -> None:
    """Gets a snapshot of the profiling result.

    Args:
      service_addr: Address of the profiling service (e.g. localhost:6006).
      logdir: Directory to save the profiling result (e.g.
        /tmp/profile_snapshot).
    """
    self._profiler_plugin.get_snapshot(service_addr, logdir)


_default_profiler = ContinuousProfilingSnapshot()


start_continuous_profiling = _default_profiler.start_continuous_profiling
stop_continuous_profiling = _default_profiler.stop_continuous_profiling
get_snapshot = _default_profiler.get_snapshot
