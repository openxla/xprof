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
"""XProf TensorBoard profile plugin package.

This package is the standalone home of the Profile plugin. Prefer importing
submodules for new code; the re-exports below preserve
``from xprof import profile_plugin`` for existing callers and tests.

Package layout:
  plugin.py       ProfilePlugin HTTP façade
  constants.py    routes and shared constants
  tools/          tool registry, filenames, catalog
  http/           responses, request params, logging middleware
  cache/          tools list cache and result-cache policy
  tensorflow_bridge.py  optional TF helpers for remote capture
"""

from __future__ import annotations

from xprof.profile_plugin.cache.tools_cache import ToolsCache
from xprof.profile_plugin.constants import (
    ALL_HOSTS,
    AVERAGE_SUBDIR_NUMBER,
    BASE_ROUTE,
    BUNDLE_JS_ROUTE,
    CACHE_VERSION_FILE,
    CAPTURE_ROUTE,
    CONFIG_ROUTE,
    DATA_CSV_ROUTE,
    DATA_ROUTE,
    GENERATE_CACHE_ROUTE,
    HLO_MODULE_LIST_ROUTE,
    HOSTS_ROUTE,
    INDEX_HTML_ROUTE,
    INDEX_JS_ROUTE,
    LIMIT_WINDOW_SECONDS,
    LOCAL_ROUTE,
    MATERIALICONS_WOFF2_ROUTE,
    MAX_GCS_REQUESTS,
    PLUGIN_NAME,
    RUNS_ROUTE,
    RUN_TOOLS_ROUTE,
    STYLES_CSS_ROUTE,
    TB_NAME,
    TRACE_VIEWER_INDEX_HTML_ROUTE,
    TRACE_VIEWER_INDEX_JS_ROUTE,
    TRACE_VIEWER_V2_JS_ROUTE,
    TRACE_VIEWER_V2_WASM_ROUTE,
    VERSION_ROUTE,
    ZONE_JS_ROUTE,
)
from xprof.profile_plugin.http.logging_middleware import logging_wrapper
from xprof.profile_plugin.http.request_params import (
    generate_csv_filename,
    get_bool_arg,
)
from xprof.profile_plugin.http.respond import respond, version_route
from xprof.profile_plugin.plugin import HostMetadata, ProfilePlugin
from xprof.profile_plugin.tensorflow_bridge import (
    TfProfiler,
    create_tf_profiler,
    tf,
)
from xprof.profile_plugin.tools.catalog import (
    get_active_tools,
    get_tools_from_filenames,
)
from xprof.profile_plugin.tools.filenames import (
    get_hosts,
    hosts_from_xplane_filenames,
    make_filename,
    parse_filename,
)
from xprof.profile_plugin.tools.registry import (
    DEFAULT_CACHE_TOOLS,
    HLO_TOOLS,
    TOOLS,
    XPLANE_TOOLS,
    XPLANE_TOOLS_ALL_HOSTS_ONLY,
    XPLANE_TOOLS_ALL_HOSTS_SUPPORTED,
    XPLANE_TOOLS_SET,
    sort_tools,
    supports_multi_host_selection,
    use_hlo,
    use_xplane,
)

# Legacy private names still referenced by a few unit tests.
_parse_filename = parse_filename

__all__ = [
    'ALL_HOSTS',
    'AVERAGE_SUBDIR_NUMBER',
    'BASE_ROUTE',
    'BUNDLE_JS_ROUTE',
    'CACHE_VERSION_FILE',
    'CAPTURE_ROUTE',
    'CONFIG_ROUTE',
    'DATA_CSV_ROUTE',
    'DATA_ROUTE',
    'DEFAULT_CACHE_TOOLS',
    'GENERATE_CACHE_ROUTE',
    'HLO_MODULE_LIST_ROUTE',
    'HLO_TOOLS',
    'HOSTS_ROUTE',
    'HostMetadata',
    'INDEX_HTML_ROUTE',
    'INDEX_JS_ROUTE',
    'LIMIT_WINDOW_SECONDS',
    'LOCAL_ROUTE',
    'MATERIALICONS_WOFF2_ROUTE',
    'MAX_GCS_REQUESTS',
    'PLUGIN_NAME',
    'ProfilePlugin',
    'RUNS_ROUTE',
    'RUN_TOOLS_ROUTE',
    'STYLES_CSS_ROUTE',
    'TB_NAME',
    'TOOLS',
    'TRACE_VIEWER_INDEX_HTML_ROUTE',
    'TRACE_VIEWER_INDEX_JS_ROUTE',
    'TRACE_VIEWER_V2_JS_ROUTE',
    'TRACE_VIEWER_V2_WASM_ROUTE',
    'ToolsCache',
    'TfProfiler',
    'VERSION_ROUTE',
    'XPLANE_TOOLS',
    'XPLANE_TOOLS_ALL_HOSTS_ONLY',
    'XPLANE_TOOLS_ALL_HOSTS_SUPPORTED',
    'XPLANE_TOOLS_SET',
    'ZONE_JS_ROUTE',
    'create_tf_profiler',
    'generate_csv_filename',
    'get_active_tools',
    'get_bool_arg',
    'get_hosts',
    'get_tools_from_filenames',
    'hosts_from_xplane_filenames',
    'logging_wrapper',
    'make_filename',
    'parse_filename',
    'respond',
    'sort_tools',
    'supports_multi_host_selection',
    'tf',
    'use_hlo',
    'use_xplane',
    'version_route',
]
