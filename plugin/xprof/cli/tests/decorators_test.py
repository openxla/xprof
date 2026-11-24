"""Unit tests for cli/internal/decorators.py."""

import io
import json
import pathlib
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

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
    self.assertNotIn("\n", indicated_str)

    # Pretty-printed JSON string preserves indentation on cache hits.
    pretty_str = json.dumps(data, indent=2)
    indicated_pretty = decorators._add_cache_indicator(pretty_str, set_time=t0)
    self.assertIn("\n  ", indicated_pretty)
    self.assertTrue(json.loads(indicated_pretty)["__cached__"])

  def test_cached_replays_stderr_notes_on_cache_hit(self):
    """Verifies emit_stderr_note replays notes and clears stale ones."""
    emit = True

    @decorators.cached(cache=self.cache)
    def noisy_tool(session_id: str, bypass_cache: bool = False) -> str:
      del bypass_cache
      if emit:
        decorators.emit_stderr_note(f"xprof-note: truncated {session_id}\n")
      return json.dumps([{"k": 1}], indent=2)

    with mock.patch.object(sys, "stderr", new_callable=io.StringIO) as err1:
      res1 = noisy_tool("run1")
    with mock.patch.object(sys, "stderr", new_callable=io.StringIO) as err2:
      res2 = noisy_tool("run1")

    self.assertEqual(res1, res2)
    self.assertEqual(err1.getvalue(), "xprof-note: truncated run1\n")
    self.assertEqual(err2.getvalue(), "xprof-note: truncated run1\n")

    # Replacing the cached result with a run that produces no note must clear
    # the old saved note so later cache hits do not replay it.
    emit = False
    with mock.patch.object(sys, "stderr", new_callable=io.StringIO) as err3:
      noisy_tool("run1", bypass_cache=True)
    with mock.patch.object(sys, "stderr", new_callable=io.StringIO) as err4:
      noisy_tool("run1")
    self.assertEqual(err3.getvalue(), "")
    self.assertEqual(err4.getvalue(), "")

  def test_build_truncation_fields_standardizes_rounding(self):
    """Verifies build_truncation_fields produces consistent fields and rounding."""
    fields = decorators.build_truncation_fields(
        total_records=5,
        returned_records=2,
        top_level_records=4,
        total_field="total_self_time_ms_all_records",
        total_value=12.3456789,
    )
    self.assertEqual(
        fields,
        {
            "truncated": True,
            "total_records": 5,
            "returned_records": 2,
            "top_level_records": 4,
            "total_self_time_ms_all_records": 12.3457,
        },
    )

  def test_compute_path_fingerprint_no_trace_inputs_sentinel(self):
    """Verifies empty directory emits NO_TRACE_INPUTS."""
    empty_dir = self.temp_dir / "empty_dir"
    empty_dir.mkdir()
    fp = decorators._compute_path_fingerprint(empty_dir)
    self.assertEqual(fp, "NO_TRACE_INPUTS")

  def test_content_addressable_fingerprint_relative_vs_absolute(self):
    """Verifies relative and absolute paths produce identical content signatures."""
    trace_dir = self.temp_dir / "trace_rel_abs"
    trace_dir.mkdir()
    (trace_dir / "trace.xplane.pb").write_bytes(b"TRACE_DATA_12345" * 100)

    fp_abs = decorators._compute_path_fingerprint(str(trace_dir.resolve()))
    fp_rel = decorators._compute_path_fingerprint(str(trace_dir))
    self.assertEqual(fp_abs, fp_rel)
    self.assertNotIn(fp_abs, ("NO_TRACE_INPUTS", "NONEXISTENT"))

  def test_content_addressable_fingerprint_copied_directory(self):
    """Verifies copied directories with identical content share signatures."""
    dir1 = self.temp_dir / "dir1"
    dir2 = self.temp_dir / "dir2_copy"
    dir1.mkdir()
    (dir1 / "node.xplane.pb").write_bytes(b"IDENTICAL_DATA" * 500)
    shutil.copytree(dir1, dir2)

    fp1 = decorators._compute_path_fingerprint(str(dir1))
    fp2 = decorators._compute_path_fingerprint(str(dir2))
    self.assertEqual(fp1, fp2)

  def test_content_addressable_fingerprint_in_place_edit(self):
    """Verifies in-place edits to trace files invalidate the fingerprint."""
    dir1 = self.temp_dir / "dir_edit"
    dir1.mkdir()
    trace_file = dir1 / "node.xplane.pb"
    trace_file.write_bytes(b"INITIAL_DATA" * 500)
    fp1 = decorators._compute_path_fingerprint(str(dir1))

    trace_file.write_bytes(b"MODIFIED_DATA" * 500)
    fp2 = decorators._compute_path_fingerprint(str(dir1))
    self.assertNotEqual(fp1, fp2)

  def test_error_payload_not_cached(self):
    """Verifies error dictionaries are not stored in cache."""
    error_key = "error_key"
    self.cache.set(error_key, {"error": "Trace file not found"})
    self.assertIs(self.cache.get(error_key), decorators.Cache.UNKNOWN)

  def test_extract_bypass_cache_positional_and_keyword(self):
    """Verifies bypass_cache is read from either position or keyword."""
    # Signature: (session_id, limit, bypass_cache) -> index 2.
    idx = 2

    kwargs = {}
    bypass, args_for_key = decorators._extract_bypass_cache(
        ("run", 10, True), kwargs, idx
    )
    self.assertTrue(bypass)
    self.assertEqual(args_for_key, ("run", 10))
    self.assertEqual(kwargs, {})

    bypass, args_for_key = decorators._extract_bypass_cache(
        ("run", 10, False), {}, idx
    )
    self.assertFalse(bypass)
    self.assertEqual(args_for_key, ("run", 10))

    # A declared bypass_cache parameter must stay in kwargs so the wrapped
    # function still receives it.
    kwargs = {"bypass_cache": True}
    bypass, args_for_key = decorators._extract_bypass_cache(
        ("run", 10), kwargs, idx
    )
    self.assertTrue(bypass)
    self.assertEqual(args_for_key, ("run", 10))
    self.assertEqual(kwargs, {"bypass_cache": True})

    # Omitted entirely.
    bypass, args_for_key = decorators._extract_bypass_cache(
        ("run", 10), {}, idx
    )
    self.assertFalse(bypass)
    self.assertEqual(args_for_key, ("run", 10))

  def test_extract_bypass_cache_undeclared_param_is_popped(self):
    """Verifies bypass_cache is stripped when the function does not accept it."""
    kwargs = {"bypass_cache": True, "limit": 5}
    bypass, args_for_key = decorators._extract_bypass_cache(
        ("run",), kwargs, None
    )
    self.assertTrue(bypass)
    self.assertEqual(args_for_key, ("run",))
    self.assertEqual(kwargs, {"limit": 5})

  def test_cached_key_is_stable_across_bypass_cache_forms(self):
    """Verifies positional, keyword and default bypass_cache share a cache key."""
    calls = []

    @decorators.cached(cache=self.cache)
    def tool(session_id: str, bypass_cache: bool = False) -> str:
      calls.append((session_id, bypass_cache))
      return f"result:{session_id}"

    # Populate the cache.
    self.assertEqual(tool("run"), "result:run")
    self.assertEqual(len(calls), 1)

    # Cache hit: bypass_cache=False in either form must reuse the same key.
    tool("run", False)
    tool("run", bypass_cache=False)
    self.assertEqual(len(calls), 1)

    # Bypass in either form re-invokes and forwards the flag unchanged.
    tool("run", True)
    self.assertEqual(calls[-1], ("run", True))
    tool("run", bypass_cache=True)
    self.assertEqual(calls[-1], ("run", True))
    self.assertEqual(len(calls), 3)

  def test_rate_limiter_stores_tokens_without_blocking_itself(self):
    """Verifies the limiter writes its row inside its own transaction."""
    limiter = decorators._SharedRateLimiter(self.cache, "tool", 1.0, 2)

    start = time.time()
    limiter.sleep_and_advance()
    elapsed = time.time() - start

    # A write on another connection would wait 5 s for the transaction's lock.
    self.assertLess(elapsed, 2.0)
    with decorators.sqlite3.connect(self.cache.db_path) as conn:
      row = conn.execute(
          "SELECT value FROM cache WHERE key = ?", ("ratelimit:tool",)
      ).fetchone()
    self.assertIsNotNone(row)
    tokens, last_update = json.loads(row[0])
    self.assertEqual(tokens, 1.0)
    self.assertAlmostEqual(last_update, start, delta=2.0)

  def test_rate_limiter_shares_tokens_across_cache_instances(self):
    """Verifies a second process sees the token the first one consumed."""
    decorators._SharedRateLimiter(
        self.cache, "tool", 0.01, 1
    ).sleep_and_advance()
    other_cache = decorators.Cache(self.temp_dir)
    self.addCleanup(other_cache.close)
    limiter = decorators._SharedRateLimiter(other_cache, "tool", 0.01, 1)

    # The bucket is empty, so the second caller has to wait for a token.
    with mock.patch.object(decorators.time, "sleep", side_effect=TimeoutError):
      with self.assertRaises(TimeoutError):
        limiter.sleep_and_advance()

  def test_init_db_tolerates_locked_database(self):
    """Verifies start-up skips the prune when another process holds the lock."""
    blocker = decorators.sqlite3.connect(self.cache.db_path)
    self.addCleanup(blocker.close)
    blocker.execute("BEGIN IMMEDIATE")

    other_cache = decorators.Cache(self.temp_dir)
    self.addCleanup(other_cache.close)

    blocker.rollback()
    other_cache.set("key", {"value": 1})
    self.assertEqual(self.cache.get("key"), {"value": 1})

  def test_cache_start_up_while_rate_limiters_run(self):
    """Verifies a process can start while two others take rate-limit tokens."""
    threads = []
    for _ in range(2):
      limiter_cache = decorators.Cache(self.temp_dir)
      self.addCleanup(limiter_cache.close)
      limiter = decorators._SharedRateLimiter(limiter_cache, "tool", 1.0, 2)
      threads.append(threading.Thread(target=limiter.sleep_and_advance))
    for thread in threads:
      thread.start()
    time.sleep(0.5)

    other_cache = decorators.Cache(self.temp_dir)
    self.addCleanup(other_cache.close)

    for thread in threads:
      thread.join()
    # Both limiters took a token from the bucket of two.
    self.assertLess(other_cache.get("ratelimit:tool")[0], 1.0)


if __name__ == "__main__":
  unittest.main()
