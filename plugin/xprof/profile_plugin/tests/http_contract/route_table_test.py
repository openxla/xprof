# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Route table integrity: constants.*_ROUTE ↔ get_plugin_apps() bijection."""

from __future__ import annotations

import tempfile
import unittest

from xprof.profile_plugin import constants
from xprof.profile_plugin.tests.fixtures.logdir_factory import (
    create_two_host_session,
)
from xprof.profile_plugin.tests.fixtures.plugin_factory import make_plugin
from xprof.profile_plugin.tests.http_contract.wsgi_client import call_route

# All frontend routes registered in plugin.get_plugin_apps().
# CACHE_VERSION_FILE is intentionally NOT a route.
EXPECTED_ROUTES: frozenset[str] = frozenset({
    constants.BASE_ROUTE,
    constants.INDEX_JS_ROUTE,
    constants.INDEX_HTML_ROUTE,
    constants.BUNDLE_JS_ROUTE,
    constants.STYLES_CSS_ROUTE,
    constants.MATERIALICONS_WOFF2_ROUTE,
    constants.TRACE_VIEWER_INDEX_HTML_ROUTE,
    constants.TRACE_VIEWER_INDEX_JS_ROUTE,
    constants.TRACE_VIEWER_V2_JS_ROUTE,
    constants.TRACE_VIEWER_V2_WASM_ROUTE,
    constants.ZONE_JS_ROUTE,
    constants.RUNS_ROUTE,
    constants.RUN_TOOLS_ROUTE,
    constants.HOSTS_ROUTE,
    constants.DATA_ROUTE,
    constants.DATA_CSV_ROUTE,
    constants.VERSION_ROUTE,
    constants.HLO_MODULE_LIST_ROUTE,
    constants.CAPTURE_ROUTE,
    constants.LOCAL_ROUTE,
    constants.CONFIG_ROUTE,
    constants.GENERATE_CACHE_ROUTE,
})


class RouteTableTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._td = tempfile.TemporaryDirectory()
    create_two_host_session(self._td.name)
    self.plugin = make_plugin(self._td.name)

  def tearDown(self):
    self._td.cleanup()
    super().tearDown()

  def test_route_table_bijection(self):
    apps = self.plugin.get_plugin_apps()
    actual = frozenset(apps.keys())
    self.assertEqual(
        actual,
        EXPECTED_ROUTES,
        msg=(
            f'missing={sorted(EXPECTED_ROUTES - actual)}; '
            f'extra={sorted(actual - EXPECTED_ROUTES)}'
        ),
    )
    self.assertNotIn(constants.CACHE_VERSION_FILE, apps)

  def test_each_app_is_callable(self):
    apps = self.plugin.get_plugin_apps()
    for route, app in sorted(apps.items()):
      self.assertTrue(callable(app), msg=f'{route} app not callable')

  def test_each_app_accepts_wsgi_call_without_typeerror(self):
    """Decorator regression: bare method needs (environ, start_response).

    Empty query may raise handler-level errors (e.g. missing ``run``); those are
    out of scope for this dry-run. Only TypeError indicates a lost
    ``@Request.application`` decorator (wrong WSGI arity).
    """
    for route in sorted(EXPECTED_ROUTES):
      with self.subTest(route=route):
        try:
          result = call_route(self.plugin, route, method='GET', query={})
        except TypeError as err:
          self.fail(
              f'{route} raised TypeError (likely lost @Request.application): '
              f'{err}'
          )
        except Exception:
          # Handler body failed on empty query; WSGI signature still valid.
          continue
        # Any HTTP status is fine; we only care the WSGI signature works.
        self.assertIsInstance(result.status_code, int)
        self.assertGreaterEqual(result.status_code, 100)


if __name__ == '__main__':
  unittest.main()
