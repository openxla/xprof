"""Tests for persisting LLO static analysis into SQLite."""

import os
import sqlite3

from absl.testing import absltest

from xprof.cli.internal.llo_static_analysis import llo_static_db
from xprof.cli.internal.llo_static_analysis import llo_test_fixtures


class LloStaticDbTest(absltest.TestCase):

  def _db(self) -> str:
    return os.path.join(self.create_tempdir().full_path, "test.xprofdb")

  def test_writes_expected_row_counts(self):
    module = llo_test_fixtures.make_sample_module()
    db_path = self._db()

    written = llo_static_db.write_static_analysis([module], db_path)

    # The fixture has six bundles, one spill pseudo, and one BDI-annotated
    # instruction carrying two stall codes (R:2 and O:1).
    self.assertEqual(written["llo_modules"], 1)
    self.assertEqual(written["llo_bundles"], 6)
    self.assertEqual(written["llo_spill_fill"], 1)
    self.assertGreater(written["llo_bdi_stalls"], 0)

  def test_bundle_rows_carry_used_and_avail(self):
    module = llo_test_fixtures.make_sample_module()
    db_path = self._db()
    llo_static_db.write_static_analysis([module], db_path)

    conn = sqlite3.connect(db_path)
    try:
      rows = conn.execute(
          "SELECT bundle_number, mxu_used, mxu_avail, spill FROM llo_bundles"
          " ORDER BY bundle_number"
      ).fetchall()
    finally:
      conn.close()

    self.assertLen(rows, 6)
    # The fixture puts the MXU to work at bundle 3 and spills at bundle 4.
    self.assertEqual(rows[3], (3, 1, 1, 0))
    self.assertEqual(rows[4], (4, 0, 1, 1))

  def test_target_capacity_is_present_but_empty(self):
    # Consumers LEFT JOIN this table today and must get NULL, not an error.
    module = llo_test_fixtures.make_sample_module()
    db_path = self._db()
    llo_static_db.write_static_analysis([module], db_path)

    conn = sqlite3.connect(db_path)
    try:
      row = conn.execute(
          "SELECT m.module_key, c.capacity_value"
          " FROM llo_modules m"
          " LEFT JOIN llo_target_capacity c ON m.module_key = c.module_key"
      ).fetchone()
    finally:
      conn.close()

    self.assertIsNone(row[1])

  def test_reload_is_idempotent(self):
    module = llo_test_fixtures.make_sample_module()
    db_path = self._db()

    llo_static_db.write_static_analysis([module], db_path)
    llo_static_db.write_static_analysis([module], db_path)

    conn = sqlite3.connect(db_path)
    try:
      modules = conn.execute("SELECT COUNT(*) FROM llo_modules").fetchone()[0]
      bundles = conn.execute("SELECT COUNT(*) FROM llo_bundles").fetchone()[0]
      spills = conn.execute("SELECT COUNT(*) FROM llo_spill_fill").fetchone()[0]
    finally:
      conn.close()

    self.assertEqual(modules, 1)
    self.assertEqual(bundles, 6)
    self.assertEqual(spills, 1)

  def test_distinct_modules_coexist(self):
    db_path = self._db()
    first = llo_test_fixtures.make_sample_module("kScan")
    second = llo_test_fixtures.make_sample_module("kFlashAttention")

    llo_static_db.write_static_analysis([first, second], db_path)

    conn = sqlite3.connect(db_path)
    try:
      keys = [
          r[0]
          for r in conn.execute(
              "SELECT module_key FROM llo_modules ORDER BY module_key"
          )
      ]
    finally:
      conn.close()

    self.assertEqual(keys, ["7:kFlashAttention", "7:kScan"])

  def test_target_fields_are_persisted(self):
    module = llo_test_fixtures.make_sample_module()
    db_path = self._db()
    llo_static_db.write_static_analysis([module], db_path)

    conn = sqlite3.connect(db_path)
    try:
      row = conn.execute(
          "SELECT chip_config_name, variant, replica_count,"
          " reserved_hbm_usage_bytes FROM llo_modules"
      ).fetchone()
    finally:
      conn.close()

    self.assertEqual(row, ("df", "pf", 4, 1048576))

  def test_coexists_with_an_existing_llo_events_table(self):
    # The whole point of the design is one file, so writing static tables must
    # not disturb a database that already holds the runtime trace.
    db_path = self._db()
    conn = sqlite3.connect(db_path)
    try:
      conn.execute("CREATE TABLE llo_events (bundle_number INTEGER)")
      conn.execute("INSERT INTO llo_events VALUES (3)")
      conn.commit()
    finally:
      conn.close()

    llo_static_db.write_static_analysis(
        [llo_test_fixtures.make_sample_module()], db_path
    )

    conn = sqlite3.connect(db_path)
    try:
      joined = conn.execute(
          "SELECT e.bundle_number, b.mxu_used"
          " FROM llo_events e"
          " JOIN llo_bundles b ON e.bundle_number = b.bundle_number"
      ).fetchall()
    finally:
      conn.close()

    self.assertEqual(joined, [(3, 1)])


if __name__ == "__main__":
  absltest.main()
