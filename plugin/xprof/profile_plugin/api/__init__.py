# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Frontend HTTP APIs grouped by UI concern.

Mixins are composed into ``ProfilePlugin`` so method names stay stable for tests
while each file owns one API surface (static → runs → hosts → data → capture → cache).
"""

from xprof.profile_plugin.api.cache_api import CacheApiMixin
from xprof.profile_plugin.api.capture_api import CaptureApiMixin
from xprof.profile_plugin.api.data_api import DataApiMixin
from xprof.profile_plugin.api.hosts_api import HostsApiMixin
from xprof.profile_plugin.api.runs_api import RunsApiMixin
from xprof.profile_plugin.api.static_api import StaticApiMixin

__all__ = [
    'StaticApiMixin',
    'RunsApiMixin',
    'HostsApiMixin',
    'DataApiMixin',
    'CaptureApiMixin',
    'CacheApiMixin',
]
