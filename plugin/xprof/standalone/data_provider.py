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
"""A standalone version of Tensorboard's context and utils."""


class MultiplexerDataProvider:
  """Data provider that reads data from a multiplexer."""

  def __init__(self, multiplexer, logdir):
    """Creates a new MultiplexerDataProvider.

    Args:
      multiplexer: A multiplexer object.
      logdir: The log directory.
    """
    self.multiplexer = multiplexer
    self.logdir = logdir

  def list_runs(self, *args, **kwargs):  # pylint: disable=invalid-name
    return self.multiplexer.list_runs(*args, **kwargs)
