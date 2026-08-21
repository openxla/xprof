"""Unit tests for cli/internal/decorators.py."""

import json
import pathlib
import shutil
import tempfile
import time
import unittest

# pylint: disable=g-import-not-at-top
from xprof.cli.internal import decorators


class DecoratorsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
    self.cache = decorators.Cache(self.temp_dir)

  def test_sqlite_cache_records_set_time(self):
    """Verifies Cache.set() writes a float timestamp into the set_time column."""
    test_key = "test_key_123"
    payload = {"status": "ok", "value": 42}
    before_set = time.time()
    self.cache.set(test_key, payload)
    after_set = time.time()

    with decorators.sqlite3.connect(self.cache.db_path) as conn:
      cursor = conn.cursor()
      cursor.execute(
          "SELECT value, set_time FROM cache WHERE key = ?", (test_key,)
      )
      row = cursor.fetchone()

    self.assertIsNotNone(row)
    val_str, set_time = row
    self.assertEqual(json.loads(val_str), payload)
    self.assertIsNotNone(set_time)
    self.assertGreaterEqual(set_time, before_set - 0.1)
    self.assertLessEqual(set_time, after_set + 0.1)

  def test_sqlite_cache_handles_legacy_null_set_time(self):
    """Verifies Cache.get() and get_with_metadata() handle legacy rows with NULL set_time."""
    legacy_key = "legacy_key_456"
    legacy_payload = {"legacy": True}

    with decorators.sqlite3.connect(self.cache.db_path) as conn:
      conn.execute(
          "INSERT INTO cache (key, value, expire, set_time) VALUES (?, ?, ?,"
          " NULL)",
          (legacy_key, json.dumps(legacy_payload), None),
      )
      conn.commit()

    # Regular get
    val = self.cache.get(legacy_key)
    self.assertEqual(val, legacy_payload)

    # get_with_metadata
    val, set_time = self.cache.get_with_metadata(legacy_key)
    self.assertEqual(val, legacy_payload)
    self.assertIsNone(set_time)

  def test_cache_indicator_includes_age(self):
    """Verifies _add_cache_indicator includes __cached__: True and __cache_age_s__."""
    data = {"metric": 100.0}
    t0 = time.time() - 15.5  # 15.5 seconds ago

    indicated_dict = decorators._add_cache_indicator(data, set_time=t0)
    self.assertTrue(indicated_dict["__cached__"])
    self.assertAlmostEqual(indicated_dict["__cache_age_s__"], 15.5, delta=1.0)

    # JSON string handling
    json_str = json.dumps(data)
    indicated_str = decorators._add_cache_indicator(json_str, set_time=t0)
    parsed = json.loads(indicated_str)
    self.assertTrue(parsed["__cached__"])
    self.assertAlmostEqual(parsed["__cache_age_s__"], 15.5, delta=1.0)

  def test_compute_path_fingerprint_no_trace_inputs_sentinel(self):
    """Verifies empty directory emits NO_TRACE_INPUTS."""
    empty_dir = self.temp_dir / "empty_dir"
    empty_dir.mkdir()
    fp = decorators._compute_path_fingerprint(empty_dir)
    self.assertEqual(fp, "NO_TRACE_INPUTS")

  def test_error_payload_not_cached(self):
    """Verifies error dictionaries are not stored in cache."""
    error_key = "error_key"
    self.cache.set(error_key, {"error": "Trace file not found"})
    self.assertIs(self.cache.get(error_key), decorators.Cache.UNKNOWN)


if __name__ == "__main__":
  unittest.main()
