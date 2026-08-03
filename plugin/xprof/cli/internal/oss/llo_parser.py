"""LLO Parser and DB Module for XProf CLI."""

import json
import logging
import os
import re
import sqlite3
from typing import Any
import urllib.request


_LLO_LINE_NAMES = {
    "Pallas Primitives",
    "Source Lines",
    "Source code",
    "LLO Instructions",
    "SALU Instructions",
    "VALU Instructions",
    "EUP Instructions",
    "XLU Instructions",
    "VLD Instructions",
    "VST Instructions",
}
_MXU_PATTERN = re.compile(r"^MXU.* Instructions$")
_TPU_DEVICE_PATTERN = re.compile(r"device:TPU:\d+")


def is_llo_line(line_name: str) -> bool:
  """Checks if the line name matches LLO line criteria."""
  if line_name in _LLO_LINE_NAMES:
    return True
  if _MXU_PATTERN.match(line_name):
    return True
  return False


def _is_numeric_or_id(name: str) -> bool:
  """Checks if the name is a numeric string or starts with ID:."""
  if name.startswith("ID:"):
    return True
  try:
    float(name)
    return True
  except ValueError:
    return False


def _get_stat_value_str(stat, stat_metadata_map) -> str | None:
  """Helper to extract stat value as a string."""
  if stat.HasField("str_value"):
    return stat.str_value
  if stat.HasField("ref_value"):
    return stat_metadata_map.get(stat.ref_value)
  if stat.HasField("int64_value"):
    return str(stat.int64_value)
  if stat.HasField("uint64_value"):
    return str(stat.uint64_value)
  if stat.HasField("double_value"):
    return str(stat.double_value)
  if stat.HasField("bytes_value"):
    try:
      return stat.bytes_value.decode("utf-8")
    except UnicodeDecodeError:
      return stat.bytes_value.hex()
  return None


def _parse_int_stat(val: str | None) -> int | None:
  """Helper to convert a stat string/value to int if numeric."""
  if val is None:
    return None
  try:
    return int(val)
  except ValueError:
    try:
      return int(float(val))
    except ValueError:
      return None


def parse_and_load_llo_events(
    xspace_proto: Any, db_path: str
) -> None:
  """Parses XSpace for TPU planes, extracts LLO events, and loads them into SQLite DB.

  Args:
      xspace_proto: The input XSpace protobuf containing the profiles.
      db_path: Path to the SQLite database file.
  """
  events_to_insert = []

  for plane in xspace_proto.planes:
    # Match any plane representing a TPU device
    if not _TPU_DEVICE_PATTERN.search(plane.name):
      continue

    # Build stat metadata map for quick lookups within the plane
    stat_metadata_map = {}
    relevant_stat_ids = set()
    target_stat_names = {
        "details",
        "source_info",
        "llo_scopes",
        "pallas_primitive",
        "source_call_stack",
        "msg",
        "message",
        "annotation",
        "label",
        "bundle_number",
        "instruction_ordinal",
        "unit_id",
    }

    for k, v in plane.stat_metadata.items():
      stat_metadata_map[k] = v.name
      if v.name in target_stat_names:
        relevant_stat_ids.add(k)

    metadata_cache = {}

    for line in plane.lines:
      line_name = line.display_name or line.name
      if not line_name:
        continue
      if not is_llo_line(line_name):
        continue

      for event in line.events:
        metadata_id = event.metadata_id
        if metadata_id not in metadata_cache:
          metadata = plane.event_metadata.get(metadata_id)
          metadata_stats_dict = {}
          if metadata:
            for stat in metadata.stats:
              mid = stat.metadata_id
              if mid in relevant_stat_ids:
                name = stat_metadata_map.get(mid)
                if name:
                  val = _get_stat_value_str(stat, stat_metadata_map)
                  if val is not None:
                    metadata_stats_dict[name] = val
            display_name = (
                metadata.display_name or metadata.name or f"ID:{metadata.id}"
            )
          else:
            display_name = f"ID:{metadata_id}"
          metadata_cache[metadata_id] = (display_name, metadata_stats_dict)

        event_name, metadata_stats_dict = metadata_cache[metadata_id]

        event_stats_dict = {}
        for stat in event.stats:
          mid = stat.metadata_id
          if mid in relevant_stat_ids:
            name = stat_metadata_map.get(mid)
            if name:
              val = _get_stat_value_str(stat, stat_metadata_map)
              if val is not None:
                event_stats_dict[name] = val

        # If name is a simple number, fallback to stat labels if available
        if _is_numeric_or_id(event_name):
          found_name = None
          for name_stat in ("msg", "message", "annotation", "label"):
            if name_stat in event_stats_dict:
              found_name = event_stats_dict[name_stat]
              break
          if found_name is None:
            for name_stat in ("msg", "message", "annotation", "label"):
              if name_stat in metadata_stats_dict:
                found_name = metadata_stats_dict[name_stat]
                break
          if found_name is not None:
            event_name = found_name

        # Extract stats of interest
        details = event_stats_dict.get("details")
        if details is None:
          details = metadata_stats_dict.get("details")

        source_info = event_stats_dict.get("source_info")
        if source_info is None:
          source_info = metadata_stats_dict.get("source_info")

        llo_scopes = event_stats_dict.get("llo_scopes")
        if llo_scopes is None:
          llo_scopes = metadata_stats_dict.get("llo_scopes")

        pallas_primitive = event_stats_dict.get("pallas_primitive")
        if pallas_primitive is None:
          pallas_primitive = metadata_stats_dict.get("pallas_primitive")

        source_call_stack = event_stats_dict.get("source_call_stack")
        if source_call_stack is None:
          source_call_stack = metadata_stats_dict.get("source_call_stack")

        bundle_number_str = event_stats_dict.get("bundle_number")
        if bundle_number_str is None:
          bundle_number_str = metadata_stats_dict.get("bundle_number")
        bundle_number = _parse_int_stat(bundle_number_str)

        instruction_ordinal_str = event_stats_dict.get("instruction_ordinal")
        if instruction_ordinal_str is None:
          instruction_ordinal_str = metadata_stats_dict.get(
              "instruction_ordinal"
          )
        instruction_ordinal = _parse_int_stat(instruction_ordinal_str)

        unit_id_str = event_stats_dict.get("unit_id")
        if unit_id_str is None:
          unit_id_str = metadata_stats_dict.get("unit_id")
        unit_id = _parse_int_stat(unit_id_str)

        events_to_insert.append((
            plane.name,
            line_name,
            event_name,
            event.offset_ps,
            event.duration_ps,
            details,
            source_info,
            llo_scopes,
            pallas_primitive,
            source_call_stack,
            bundle_number,
            instruction_ordinal,
            unit_id,
        ))

  # Establish connection and populate the SQLite Database
  db_dir = os.path.dirname(db_path)
  if db_dir and not os.path.exists(db_dir):
    os.makedirs(db_dir, exist_ok=True)

  conn = sqlite3.connect(db_path)
  try:
    with conn:
      cursor = conn.cursor()
      cursor.execute("""
        CREATE TABLE IF NOT EXISTS llo_events (
            device TEXT,
            line_name TEXT,
            event_name TEXT,
            offset_ps INTEGER,
            duration_ps INTEGER,
            details TEXT,
            source_info TEXT,
            llo_scopes TEXT,
            pallas_primitive TEXT,
            source_call_stack TEXT,
            bundle_number INTEGER,
            instruction_ordinal INTEGER,
            unit_id INTEGER
        )
      """)
      # Add indexes to speed up typical filtering and grouping queries
      cursor.execute(
          "CREATE INDEX IF NOT EXISTS idx_llo_events_line ON"
          " llo_events(line_name)"
      )
      cursor.execute(
          "CREATE INDEX IF NOT EXISTS idx_llo_events_event ON"
          " llo_events(event_name)"
      )
      cursor.execute(
          "CREATE INDEX IF NOT EXISTS idx_llo_events_duration ON"
          " llo_events(duration_ps DESC)"
      )
      cursor.execute(
          "CREATE INDEX IF NOT EXISTS idx_llo_events_bundle ON"
          " llo_events(bundle_number)"
      )

      insert_sql = """
        INSERT INTO llo_events (
          device, line_name, event_name, offset_ps, duration_ps,
          details, source_info, llo_scopes, pallas_primitive, source_call_stack,
          bundle_number, instruction_ordinal, unit_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      """
      cursor.executemany(insert_sql, events_to_insert)
      logging.info(
          "Successfully loaded %d LLO events into database %s",
          len(events_to_insert),
          db_path,
      )
  finally:
    conn.close()


PREBAKED_QUERIES = {
    "instruction_mix": (
        """
        SELECT event_name, COUNT(*) as count
        FROM llo_events
        WHERE line_name LIKE '%Instructions'
        GROUP BY event_name
        ORDER BY count DESC
        LIMIT 10
    """
    ),
    "top_pallas_primitives": (
        """
        SELECT event_name, COUNT(*) as count, SUM(duration_ps) as total_duration_ps
        FROM llo_events
        WHERE line_name = 'Pallas Primitives'
        GROUP BY event_name
        ORDER BY total_duration_ps DESC
    """
    ),
    "longest_events": (
        """
        SELECT line_name, event_name, offset_ps, duration_ps, details
        FROM llo_events
        ORDER BY duration_ps DESC
        LIMIT 20
    """
    ),
}


def execute_llo_query(db_path: str, query: str) -> str:
  """Executes SQL query on the SQLite DB and returns results as JSON.

  Args:
      db_path: Path to the SQLite database.
      query: The raw SQL query, or a pre-baked query name (e.g.
        'instruction_mix').

  Returns:
      A JSON string of query results.
  """
  if not db_path:
    return json.dumps(
        {"status": "error", "message": f"Database file not found: {db_path}"}
    )

  if db_path != ":memory:" and not os.path.exists(db_path):
    return json.dumps(
        {"status": "error", "message": f"Database file not found: {db_path}"}
    )

  sql_query = PREBAKED_QUERIES.get(query, query)

  db_uri = f"file:{urllib.request.pathname2url(db_path)}?mode=ro"
  conn = None
  try:
    conn = sqlite3.connect(db_uri, uri=True)
    cursor = conn.cursor()
    cursor.execute(sql_query)

    if cursor.description is None:
      # If query did not produce rows (e.g. UPDATE, INSERT, CREATE)
      conn.commit()
      return json.dumps({
          "status": "success",
          "rows_affected": cursor.rowcount,
      })

    # For SELECT queries, build a list of dicts with column names as keys
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    results = [dict(zip(columns, row)) for row in rows]
    return json.dumps(results, indent=2)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error executing LLO query")
    msg = str(e)
    if "disk i/o error" in msg.lower() or os.path.isdir(db_path):
      msg = f"unable to open database file: {msg}"
    return json.dumps({
        "status": "error",
        "message": msg,
    })
  finally:
    if conn is not None:
      conn.close()
