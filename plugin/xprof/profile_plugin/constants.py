# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Route paths and shared constants for the profile plugin."""

from __future__ import annotations

# The prefix of routes provided by this plugin.
TB_NAME = 'plugins'
PLUGIN_NAME = 'profile'

BASE_ROUTE = '/'
INDEX_JS_ROUTE = '/index.js'
INDEX_HTML_ROUTE = '/index.html'
BUNDLE_JS_ROUTE = '/bundle.js'
STYLES_CSS_ROUTE = '/styles.css'
MATERIALICONS_WOFF2_ROUTE = '/materialicons.woff2'
TRACE_VIEWER_INDEX_HTML_ROUTE = '/trace_viewer_index.html'
TRACE_VIEWER_INDEX_JS_ROUTE = '/trace_viewer_index.js'
TRACE_VIEWER_V2_JS_ROUTE = '/trace_viewer_v2.js'
TRACE_VIEWER_V2_WASM_ROUTE = '/trace_viewer_v2.wasm'
ZONE_JS_ROUTE = '/zone.js'
DATA_ROUTE = '/data'
DATA_CSV_ROUTE = '/data_csv'
VERSION_ROUTE = '/version'
RUNS_ROUTE = '/runs'
RUN_TOOLS_ROUTE = '/run_tools'
HOSTS_ROUTE = '/hosts'
HLO_MODULE_LIST_ROUTE = '/module_list'
CAPTURE_ROUTE = '/capture_profile'
LOCAL_ROUTE = '/local'
CONFIG_ROUTE = '/config'
CACHE_VERSION_FILE = 'cache_version.txt'
GENERATE_CACHE_ROUTE = '/generate_cache'

ALL_HOSTS = 'ALL_HOSTS'

# Rate limiter constants, the GCS quota defined below
# https://cloud.google.com/storage/quotas#rate-quotas.
# currently set to 1000 request per minute.
# TODO(kcai): The assumption on the average number of subdirs is not
# always true. If this is not sufficient, we can consider a token-based
# approach that counts the number of subdirs after calling iterdir.
MAX_GCS_REQUESTS = 1000
LIMIT_WINDOW_SECONDS = 60
AVERAGE_SUBDIR_NUMBER = 10
