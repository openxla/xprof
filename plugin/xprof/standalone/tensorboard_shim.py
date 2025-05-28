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
"""A shim to handle whether or not TensorBoard is available."""
# pylint: disable=g-import-not-at-top,unused-import,line-too-long

try:
  from tensorboard import context
  from tensorboard.backend.event_processing import data_provider
  from tensorboard.backend.event_processing import plugin_asset_util
  from tensorboard.backend.event_processing import plugin_event_multiplexer
  from tensorboard.plugins import base_plugin
  IS_TENSORBOARD_AVAILABLE = True
except ImportError:
  from xprof.standalone import base_plugin
  from xprof.standalone import context
  from xprof.standalone import data_provider
  from xprof.standalone import plugin_asset_util
  from xprof.standalone import plugin_event_multiplexer
  IS_TENSORBOARD_AVAILABLE = False
