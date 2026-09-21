"""Persists LLO static-schedule analysis into the xprof_cli SQLite database.

`llo_parser.py` already materializes the *runtime* LLO trace into an
`llo_events` table. This module writes the *static*, compiler-modelled view of
the same kernel -- per-bundle issue-slot utilization, spill and fill sites,
allocations, and BDI stall annotations -- into the same database file.

The point of writing both into one file is that a single `.xprofdb` can then
answer "what did the hardware actually do" and "what did the compiler plan for"
in one query, instead of requiring a second tool and a second artifact to
correlate by hand.

Rows are keyed by `module_key`, a `<hlo_module_id>:<hlo_instruction_name>`
string, because one capture routinely contains many LLO modules.

Two limits are deliberate and should not be worked around by callers:

  * `llo_events` carries no module identifier today, so a static/runtime join is
    only unambiguous when the capture holds a single LLO module. Adding a module
    discriminator to `llo_events` is follow-up work; until it lands, do not
    assume the join is safe on a multi-module capture.
  * Hardware capacity constants -- register-file sizes, FIFO depths -- are not
    written here, because this proto does not carry them and guessing them would
    be worse than leaving them absent. `llo_target_capacity` is created empty on
    purpose so consumers can LEFT JOIN against it now and receive NULL rather
    than an error, and start returning real numbers when the target catalog
    lands without any consumer having to change its SQL.

`*_avail` comes from the `RationalVectorProto` denominator the compiler emitted,
not from a hardware table, so it is exactly as trustworthy as the schedule
profile that produced it.
"""

import sqlite3
from typing import Any, Iterable

from xprof.cli.internal.llo_static_analysis import (
    bdi_stalls,
)
from xprof.cli.internal.llo_static_analysis import (
    llo_allocations,
)
from xprof.cli.internal.llo_static_analysis import (
    llo_bundle_utilization,
)
from xprof.cli.internal.llo_static_analysis import (
    target_info,
)
from xprof.protobuf import llo_lite_pb2

# Per-unit columns are generated from the extractor's field tuple rather than
# spelled out, so a new functional unit cannot appear in the analysis without
# also appearing in the schema.
_UNIT_COLUMN_DDL = ",\n        ".join(
    f"{f}_used INTEGER,\n        {f}_avail INTEGER"
    for f in llo_bundle_utilization.FIELDS
)


def _bundle_columns() -> list[str]:
  """Returns the llo_bundles column order, matching _UNIT_COLUMN_DDL."""
  columns = ["module_key", "bundle_number"]
  for field in llo_bundle_utilization.FIELDS:
    columns.append(f"{field}_used")
    columns.append(f"{field}_avail")
  columns.extend(["spill", "fill", "util_pct"])
  return columns


_BUNDLE_COLUMNS = _bundle_columns()

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS llo_modules (
        module_key TEXT PRIMARY KEY,
        hlo_instruction_name TEXT,
        hlo_module_name TEXT,
        hlo_module_id INTEGER,
        total_bundles INTEGER,
        has_static_utilization INTEGER,
        tpu_version TEXT,
        variant TEXT,
        chip_config_name TEXT,
        platform_type TEXT,
        replica_count INTEGER,
        reserved_hbm_usage_bytes INTEGER
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS llo_bundles (
        module_key TEXT,
        bundle_number INTEGER,
        {_UNIT_COLUMN_DDL},
        spill INTEGER,
        fill INTEGER,
        util_pct INTEGER,
        PRIMARY KEY (module_key, bundle_number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS llo_allocations (
        module_key TEXT,
        ordinal INTEGER,
        size INTEGER,
        space INTEGER,
        is_spill INTEGER,
        is_scoped INTEGER,
        is_remote INTEGER,
        is_virtual INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS llo_spill_fill (
        module_key TEXT,
        ordinal INTEGER,
        bundle_number INTEGER,
        kind TEXT
    )
    """,
    # One row per (instruction, stall code) rather than a codes blob, so the
    # histogram the JSON renderer produces is reachable as a GROUP BY.
    """
    CREATE TABLE IF NOT EXISTS llo_bdi_stalls (
        module_key TEXT,
        ordinal INTEGER,
        bundle_number INTEGER,
        code TEXT
    )
    """,
    # Intentionally empty until the target catalog lands. See module docstring.
    """
    CREATE TABLE IF NOT EXISTS llo_target_capacity (
        module_key TEXT,
        capacity_key TEXT,
        capacity_value INTEGER,
        PRIMARY KEY (module_key, capacity_key)
    )
    """,
)

_INDEXES = (
    (
        "CREATE INDEX IF NOT EXISTS idx_llo_bundles_bundle ON"
        " llo_bundles(bundle_number)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_llo_spill_fill_bundle ON"
        " llo_spill_fill(bundle_number)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_llo_bdi_stalls_bundle ON"
        " llo_bdi_stalls(bundle_number)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_llo_bdi_stalls_code ON"
        " llo_bdi_stalls(code)"
    ),
)

# Every table that stores per-module rows, cleared before a module is rewritten
# so that loading the same capture twice is idempotent.
_MODULE_SCOPED_TABLES = (
    "llo_modules",
    "llo_bundles",
    "llo_allocations",
    "llo_spill_fill",
    "llo_bdi_stalls",
    "llo_target_capacity",
)


def module_key(module: llo_lite_pb2.LloModuleProto) -> str:
  """Returns the stable key identifying `module` within a capture."""
  return f"{module.hlo_module_id}:{module.hlo_instruction_name}"


def _insert_sql(table: str, columns: list[str]) -> str:
  placeholders = ", ".join("?" * len(columns))
  return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"


def _module_row(module: llo_lite_pb2.LloModuleProto) -> list[Any]:
  """Flattens the proto-resident module and target fields into one row."""
  fields = target_info.extract_key_fields(module)
  return [
      module_key(module),
      module.hlo_instruction_name,
      module.hlo_module_name,
      int(module.hlo_module_id),
      llo_bundle_utilization.num_bundles(module.static_utilization),
      int(llo_bundle_utilization.has_static_utilization(module)),
      fields.get("version"),
      fields.get("variant"),
      fields.get("chip_config_name"),
      fields.get("platform_type"),
      fields.get("replica_count"),
      fields.get("reserved_hbm_usage_bytes"),
  ]


def _bundle_rows(module: llo_lite_pb2.LloModuleProto) -> list[list[Any]]:
  key = module_key(module)
  rows = []
  for rec in llo_bundle_utilization.extract_bundle_utilization(module):
    row: list[Any] = [key, rec["bundle"]]
    for f in llo_bundle_utilization.FIELDS:
      row.append(rec[f"{f}_used"])
      row.append(rec[f"{f}_avail"])
    row.extend([rec["spill"], rec["fill"], rec["util_pct"]])
    rows.append(row)
  return rows


def _bdi_rows(module: llo_lite_pb2.LloModuleProto) -> list[list[Any]]:
  key = module_key(module)
  rows = []
  for ordinal, bundle, codes in bdi_stalls.iter_bdi_instructions(module):
    for code in codes:
      rows.append([key, ordinal, bundle, code])
  return rows


def write_static_analysis(
    modules: Iterable[llo_lite_pb2.LloModuleProto], db_path: str
) -> dict[str, int]:
  """Writes the static analysis of `modules` into the SQLite DB at `db_path`.

  Creates the schema if absent, so this composes with a database that already
  holds `llo_events`. Rewriting a module that is already present replaces its
  rows rather than duplicating them.

  Args:
    modules: The LLO modules to persist.
    db_path: Path to the SQLite database file, created if it does not exist.

  Returns:
    A per-table count of the rows written.
  """
  written = {
      "llo_modules": 0,
      "llo_bundles": 0,
      "llo_allocations": 0,
      "llo_spill_fill": 0,
      "llo_bdi_stalls": 0,
  }
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    for statement in _SCHEMA:
      cursor.execute(statement)
    for statement in _INDEXES:
      cursor.execute(statement)

    for module in modules:
      key = module_key(module)
      for table in _MODULE_SCOPED_TABLES:
        cursor.execute(f"DELETE FROM {table} WHERE module_key = ?", (key,))

      cursor.execute(
          _insert_sql(
              "llo_modules",
              [
                  "module_key",
                  "hlo_instruction_name",
                  "hlo_module_name",
                  "hlo_module_id",
                  "total_bundles",
                  "has_static_utilization",
                  "tpu_version",
                  "variant",
                  "chip_config_name",
                  "platform_type",
                  "replica_count",
                  "reserved_hbm_usage_bytes",
              ],
          ),
          _module_row(module),
      )
      written["llo_modules"] += 1

      bundle_rows = _bundle_rows(module)
      cursor.executemany(
          _insert_sql("llo_bundles", _BUNDLE_COLUMNS), bundle_rows
      )
      written["llo_bundles"] += len(bundle_rows)

      alloc_rows = [
          [
              key,
              a["ordinal"],
              a["size"],
              a["space"],
              a["is_spill"],
              a["is_scoped"],
              a["is_remote"],
              a["is_virtual"],
          ]
          for a in llo_allocations.extract_allocations(module)
      ]
      cursor.executemany(
          _insert_sql(
              "llo_allocations",
              [
                  "module_key",
                  "ordinal",
                  "size",
                  "space",
                  "is_spill",
                  "is_scoped",
                  "is_remote",
                  "is_virtual",
              ],
          ),
          alloc_rows,
      )
      written["llo_allocations"] += len(alloc_rows)

      spill_rows = [
          [key, s["ordinal"], s["bundle"], s["kind"]]
          for s in llo_allocations.extract_spill_fill_instructions(module)
      ]
      cursor.executemany(
          _insert_sql(
              "llo_spill_fill",
              ["module_key", "ordinal", "bundle_number", "kind"],
          ),
          spill_rows,
      )
      written["llo_spill_fill"] += len(spill_rows)

      bdi_rows = _bdi_rows(module)
      cursor.executemany(
          _insert_sql(
              "llo_bdi_stalls",
              ["module_key", "ordinal", "bundle_number", "code"],
          ),
          bdi_rows,
      )
      written["llo_bdi_stalls"] += len(bdi_rows)

    conn.commit()
  finally:
    conn.close()
  return written
