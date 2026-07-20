"""Unit tests for result cache version policy."""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from etils import epath

from xprof.profile_plugin.cache.result_cache_policy import (
    should_use_saved_result,
    write_cache_version_file,
)
from xprof.profile_plugin.constants import CACHE_VERSION_FILE


class ResultCachePolicyTest(unittest.TestCase):

  def test_missing_file_disables_saved(self):
    with tempfile.TemporaryDirectory() as d:
      self.assertFalse(
          should_use_saved_result(
              d, True, SimpleNamespace(__version__='2.0.0'), epath
          )
      )

  def test_older_version_disables_saved(self):
    with tempfile.TemporaryDirectory() as d:
      with open(os.path.join(d, CACHE_VERSION_FILE), 'w') as f:
        f.write('1.0.0')
      self.assertFalse(
          should_use_saved_result(
              d, True, SimpleNamespace(__version__='2.0.0'), epath
          )
      )

  def test_same_version_keeps_saved(self):
    with tempfile.TemporaryDirectory() as d:
      write_cache_version_file(
          d, SimpleNamespace(__version__='2.0.0'), epath
      )
      self.assertTrue(
          should_use_saved_result(
              d, True, SimpleNamespace(__version__='2.0.0'), epath
          )
      )

  def test_requested_false_stays_false_when_version_ok(self):
    with tempfile.TemporaryDirectory() as d:
      write_cache_version_file(
          d, SimpleNamespace(__version__='2.0.0'), epath
      )
      self.assertFalse(
          should_use_saved_result(
              d, False, SimpleNamespace(__version__='2.0.0'), epath
          )
      )


if __name__ == '__main__':
  unittest.main()
