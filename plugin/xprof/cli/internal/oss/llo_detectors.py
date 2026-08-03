"""Runtime LLO Detectors Module for XProf CLI.

This module implements algorithms for detecting scheduling bubbles, blocked
zero-initialization phases, and memory/compute concurrency ratios directly
from empirical execution timelines in the llo_events SQLite database.
"""

import json
import re
import sqlite3
from typing import Any


def _build_filter_sql(kernel_filter: str | None) -> tuple[str, list[Any]]:
  """Builds a SQL WHERE snippet and params array for filtering events."""
  if not kernel_filter:
    return "", []
  return (
      " AND (llo_scopes LIKE ? OR details LIKE ? OR event_name LIKE ?)",
      [f"%{kernel_filter}%", f"%{kernel_filter}%", f"%{kernel_filter}%"],
  )


def parse_source_locations(
    source_infos: list[str | None],
) -> list[dict[str, Any]]:
  """Parses a list of raw source_info strings into structured location objects."""
  locations_map = {}
  for info in source_infos:
    if not info:
      continue
    # Process each line or segment if separated by newline
    for line in str(info).split("\n"):
      line = line.strip()
      if not line:
        continue
      # Match file:line patterns like "main.py:10" or "main.py:L10 (op)"
      pattern = r"^([^:]+):[L|l]?(\d+)(?::\s*(.*)|\s*\((.*)\))?$"
      m = re.fullmatch(pattern, line)
      if m:
        file_name = m.group(1).strip()
        line_num = int(m.group(2))
        line_name = (m.group(3) or m.group(4) or "").strip()
        key = (file_name, line_num, line_name)
      else:
        key = (line, 0, "")
        file_name, line_num, line_name = line, 0, ""
      if key not in locations_map:
        locations_map[key] = {
            "file": file_name,
            "line": line_num,
            "line_name": line_name,
        }
  # Sort by file, then line number
  sorted_keys = sorted(locations_map.keys(), key=lambda k: (k[0], k[1], k[2]))
  return [locations_map[k] for k in sorted_keys]


def parse_scopes(scopes_list: list[str | None]) -> list[str]:
  """Parses and deduplicates llo_scopes strings."""
  unique_scopes = set()
  for scopes_str in scopes_list:
    if not scopes_str:
      continue
    scopes_str = str(scopes_str).strip()
    if not scopes_str or scopes_str == "[]":
      continue
    try:
      parsed = json.loads(scopes_str)
      if isinstance(parsed, list):
        formatted = " > ".join(str(p) for p in parsed)
        unique_scopes.add(formatted)
        continue
    except (json.JSONDecodeError, TypeError):
      pass
    unique_scopes.add(scopes_str)
  return sorted(list(unique_scopes))


def get_scopes_and_locations_for_range(
    c: sqlite3.Cursor,
    start_ps: int,
    end_ps: int,
    kernel_filter: str | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
  """Retrieves unique llo_scopes and source_locations active in [start_ps, end_ps]."""
  filter_sql, filter_params = _build_filter_sql(kernel_filter)
  query = f"""
    SELECT llo_scopes, source_info
    FROM llo_events
    WHERE offset_ps < ? AND (offset_ps + duration_ps) > ?
      AND line_name LIKE '%Instructions'{filter_sql}
  """
  params = [end_ps, start_ps] + filter_params
  c.execute(query, params)
  rows = c.fetchall()
  scopes_raw = [r[0] for r in rows if r[0] is not None]
  source_raw = [r[1] for r in rows if r[1] is not None]
  return parse_scopes(scopes_raw), parse_source_locations(source_raw)


def analyze_runtime_concurrency_and_overlap(
    c: sqlite3.Cursor,
    time_slice_ps: int = 1000,
    window_size_ps: int = 50000,
    kernel_filter: str | None = None,
) -> dict[str, Any]:
  """Buckets execution into time slices to calculate concurrency ratios and find the lowest overlap region.

  Args:
    c: SQLite cursor to the database containing llo_events.
    time_slice_ps: Width of each discretization bucket in picoseconds.
    window_size_ps: Size of sliding window in picoseconds for lowest overlap
      finding.
    kernel_filter: Optional substring filter across
      llo_scopes/details/event_name.

  Returns:
    Dictionary of concurrency durations, percentages, and lowest overlap region.
  """
  filter_sql, filter_params = _build_filter_sql(kernel_filter)
  query = f"""
    SELECT MIN(offset_ps), MAX(offset_ps + duration_ps)
    FROM llo_events
    WHERE line_name LIKE '%Instructions'{filter_sql}
  """
  c.execute(query, filter_params)
  row = c.fetchone()
  if not row or row[0] is None or row[1] is None or row[1] <= row[0]:
    return {
        "total_duration_ps": 0,
        "overlapped_mxu_mem_duration_ps": 0,
        "pure_compute_duration_ps": 0,
        "pure_memory_duration_ps": 0,
        "exposed_vector_duration_ps": 0,
        "idle_duration_ps": 0,
        "overlapped_percentage": 0.0,
        "pure_compute_percentage": 0.0,
        "pure_memory_percentage": 0.0,
        "idle_percentage": 0.0,
        "lowest_overlap_region": {
            "start_offset_ps": 0,
            "end_offset_ps": 0,
            "duration_ps": 0,
            "overlap_percentage": 0.0,
            "attributed_scopes": [],
            "source_locations": [],
        },
    }

  min_offset = int(row[0])
  max_offset = int(row[1])
  total_duration_ps = max_offset - min_offset

  # Ensure bounded number of slices for memory and performance
  num_slices = max(1, (total_duration_ps + time_slice_ps - 1) // time_slice_ps)
  if num_slices > 1_000_000:
    time_slice_ps = max(1000, (total_duration_ps + 999_999) // 1_000_000)
    num_slices = max(
        1, (total_duration_ps + time_slice_ps - 1) // time_slice_ps
    )

  has_mxu = bytearray(num_slices)
  has_valu = bytearray(num_slices)
  has_xlu = bytearray(num_slices)
  has_mem = bytearray(num_slices)

  query_events = f"""
    SELECT line_name, offset_ps, duration_ps
    FROM llo_events
    WHERE line_name LIKE '%Instructions'{filter_sql}
  """
  c.execute(query_events, filter_params)
  for line_name, offset_ps, duration_ps in c.fetchall():
    if offset_ps is None or duration_ps is None or duration_ps <= 0:
      continue
    ev_start = int(offset_ps)
    ev_end = ev_start + int(duration_ps)
    start_idx = max(0, (ev_start - min_offset) // time_slice_ps)
    end_idx = min(num_slices - 1, (ev_end - 1 - min_offset) // time_slice_ps)

    is_mxu = line_name == "MXU Instructions" or "MXU" in line_name
    is_mem = line_name in ("VLD Instructions", "VST Instructions")
    is_valu = line_name in ("VALU Instructions", "EUP Instructions")
    is_xlu = line_name == "XLU Instructions"

    for idx in range(start_idx, end_idx + 1):
      if is_mxu:
        has_mxu[idx] = 1
      if is_mem:
        has_mem[idx] = 1
      if is_valu:
        has_valu[idx] = 1
      if is_xlu:
        has_xlu[idx] = 1

  overlapped_ps = 0
  pure_compute_ps = 0
  pure_memory_ps = 0
  exposed_vector_ps = 0
  idle_ps = 0

  for idx in range(num_slices):
    slice_dur = (
        time_slice_ps
        if idx < num_slices - 1
        else (total_duration_ps - (num_slices - 1) * time_slice_ps)
    )
    if slice_dur <= 0:
      continue

    mxu_flag = bool(has_mxu[idx])
    mem_flag = bool(has_mem[idx])
    valu_flag = bool(has_valu[idx])
    xlu_flag = bool(has_xlu[idx])
    compute_flag = mxu_flag or valu_flag or xlu_flag

    if mxu_flag and mem_flag:
      overlapped_ps += slice_dur
    elif compute_flag and not mem_flag:
      pure_compute_ps += slice_dur
    elif mem_flag and not compute_flag:
      pure_memory_ps += slice_dur
    elif not compute_flag and not mem_flag:
      idle_ps += slice_dur

    if valu_flag and not mxu_flag:
      exposed_vector_ps += slice_dur

  # Sliding window to find lowest overlap region
  w_slices = max(1, window_size_ps // time_slice_ps)
  if w_slices > num_slices:
    w_slices = max(1, int(0.10 * num_slices))

  # Compute prefix sums for compute, mem, and overlap
  pref_compute = [0] * (num_slices + 1)
  pref_mem = [0] * (num_slices + 1)
  pref_overlap = [0] * (num_slices + 1)

  for i in range(num_slices):
    c_active = 1 if (has_mxu[i] or has_valu[i] or has_xlu[i]) else 0
    m_active = 1 if has_mem[i] else 0
    o_active = (
        1 if ((has_mxu[i] or has_valu[i] or has_xlu[i]) and has_mem[i]) else 0
    )

    pref_compute[i + 1] = pref_compute[i] + c_active
    pref_mem[i + 1] = pref_mem[i] + m_active
    pref_overlap[i + 1] = pref_overlap[i] + o_active

  best_start_idx = 0
  min_overlap_ratio = 1.0
  found_co_occurring = False

  for s in range(num_slices - w_slices + 1):
    e = s + w_slices
    c_cnt = pref_compute[e] - pref_compute[s]
    m_cnt = pref_mem[e] - pref_mem[s]
    o_cnt = pref_overlap[e] - pref_overlap[s]

    if c_cnt > 0 and m_cnt > 0:
      denom = c_cnt + m_cnt - o_cnt
      ratio = (o_cnt / denom) if denom > 0 else 0.0
      if not found_co_occurring or ratio < min_overlap_ratio:
        min_overlap_ratio = ratio
        best_start_idx = s
        found_co_occurring = True
    elif not found_co_occurring and (c_cnt > 0 or m_cnt > 0):
      denom = c_cnt + m_cnt
      ratio = (o_cnt / denom) if denom > 0 else 0.0
      if ratio < min_overlap_ratio:
        min_overlap_ratio = ratio
        best_start_idx = s

  region_start_ps = min_offset + best_start_idx * time_slice_ps
  region_end_ps = min(max_offset, region_start_ps + w_slices * time_slice_ps)
  region_dur_ps = region_end_ps - region_start_ps
  region_scopes, region_locations = get_scopes_and_locations_for_range(
      c, region_start_ps, region_end_ps, kernel_filter
  )

  def _pct(val: int, total: int) -> float:
    return round((val / total * 100.0), 2) if total > 0 else 0.0

  return {
      "total_duration_ps": total_duration_ps,
      "overlapped_mxu_mem_duration_ps": overlapped_ps,
      "pure_compute_duration_ps": pure_compute_ps,
      "pure_memory_duration_ps": pure_memory_ps,
      "exposed_vector_duration_ps": exposed_vector_ps,
      "idle_duration_ps": idle_ps,
      "overlapped_percentage": _pct(overlapped_ps, total_duration_ps),
      "pure_compute_percentage": _pct(pure_compute_ps, total_duration_ps),
      "pure_memory_percentage": _pct(pure_memory_ps, total_duration_ps),
      "idle_percentage": _pct(idle_ps, total_duration_ps),
      "lowest_overlap_region": {
          "start_offset_ps": region_start_ps,
          "end_offset_ps": region_end_ps,
          "duration_ps": region_dur_ps,
          "overlap_percentage": round(min_overlap_ratio * 100.0, 2),
          "attributed_scopes": region_scopes,
          "source_locations": region_locations,
      },
  }


def detect_runtime_scheduling_bubbles(
    c: sqlite3.Cursor,
    util_threshold_pct: float = 25.0,
    min_consecutive_bundles: int = 10,
    min_duration_ps: int = 6666,
    kernel_filter: str | None = None,
) -> list[dict[str, Any]]:
  """Detects time intervals and bundle ranges where all execution units drop below utilization threshold.

  Args:
    c: SQLite cursor.
    util_threshold_pct: Threshold percentage below which all units must drop.
    min_consecutive_bundles: Minimum consecutive bundle count to flag a bubble.
    min_duration_ps: Minimum duration in picoseconds to flag a bubble.
    kernel_filter: Optional substring filter across
      llo_scopes/details/event_name.

  Returns:
    List of detected scheduling bubble dictionary objects.
  """
  filter_sql, filter_params = _build_filter_sql(kernel_filter)
  bubbles = []

  # 1. Check consecutive bundle runs if bundle_number is present,
  # splitting by time jumps across iterations.
  query_bundle = f"""
    SELECT bundle_number,
           SUM(CASE WHEN line_name LIKE '%MXU%' THEN duration_ps ELSE 0 END) AS mxu_ps,
           SUM(CASE WHEN line_name = 'VALU Instructions' THEN duration_ps ELSE 0 END) AS valu_ps,
           SUM(CASE WHEN line_name = 'VLD Instructions' THEN duration_ps ELSE 0 END) AS vld_ps,
           SUM(CASE WHEN line_name = 'VST Instructions' THEN duration_ps ELSE 0 END) AS vst_ps,
           SUM(CASE WHEN line_name = 'XLU Instructions' THEN duration_ps ELSE 0 END) AS xlu_ps,
           MIN(offset_ps) as start_ps,
           MAX(offset_ps + duration_ps) as end_ps
    FROM llo_events
    WHERE bundle_number IS NOT NULL AND bundle_number > 0
      AND line_name LIKE '%Instructions'{filter_sql}
    GROUP BY bundle_number, (offset_ps / 50000)
    ORDER BY MIN(offset_ps), bundle_number
  """
  c.execute(query_bundle, filter_params)
  bundle_rows = c.fetchall()

  if bundle_rows:
    total_bundles = len(bundle_rows)
    curr_start_idx = None

    for i, row in enumerate(bundle_rows):
      start_ps = row[6]
      end_ps = row[7]
      dur_ps = (
          end_ps - start_ps
          if (end_ps and start_ps and end_ps > start_ps)
          else 1000
      )
      if (
          i < total_bundles - 1
          and bundle_rows[i + 1][6]
          and start_ps
          and bundle_rows[i + 1][6] > start_ps
      ):
        delta_t = bundle_rows[i + 1][6] - start_ps
        if (
            delta_t < 50_000
        ):  # If consecutive bundles are within 50 ns (same iteration)
          dur_ps = max(dur_ps, delta_t)
      elif (
          i > 0
          and bundle_rows[i - 1][6]
          and start_ps
          and start_ps > bundle_rows[i - 1][6]
      ):
        delta_t = start_ps - bundle_rows[i - 1][6]
        if delta_t < 50_000:
          dur_ps = max(dur_ps, delta_t)

      if dur_ps <= 0:
        dur_ps = 1000

      mxu_pct = (row[1] / dur_ps) * 100.0
      valu_pct = (row[2] / dur_ps) * 100.0
      vld_pct = (row[3] / dur_ps) * 100.0
      vst_pct = (row[4] / dur_ps) * 100.0
      xlu_pct = (row[5] / dur_ps) * 100.0

      is_under = (
          mxu_pct < util_threshold_pct
          and valu_pct < util_threshold_pct
          and vld_pct < util_threshold_pct
          and vst_pct < util_threshold_pct
          and xlu_pct < util_threshold_pct
      )

      # Check if we crossed a time jump / loop iteration boundary
      time_jump = False
      if i > 0 and start_ps and bundle_rows[i - 1][7]:
        if start_ps - bundle_rows[i - 1][7] > 50_000:
          time_jump = True

      if time_jump and curr_start_idx is not None:
        num_b = i - curr_start_idx
        b_start = bundle_rows[curr_start_idx][0]
        b_end = bundle_rows[i - 1][0]
        t_start = bundle_rows[curr_start_idx][6]
        t_end = bundle_rows[i - 1][7]
        t_dur = t_end - t_start if (t_end and t_start) else 0
        if num_b >= min_consecutive_bundles or t_dur >= min_duration_ps:
          scopes, locations, preceding_op, following_op, _, _ = (
              _enrich_bubble_context(c, t_start, t_end, kernel_filter)
          )
          bubbles.append({
              "start_offset_ps": int(t_start),
              "end_offset_ps": int(t_end),
              "duration_ps": int(t_dur),
              "start_bundle": int(b_start),
              "end_bundle": int(b_end),
              "attributed_scopes": scopes,
              "source_locations": locations,
              "preceding_op": preceding_op,
              "following_op": following_op,
          })
        curr_start_idx = None

      if is_under:
        if curr_start_idx is None:
          curr_start_idx = i
      else:
        if curr_start_idx is not None:
          num_b = i - curr_start_idx
          b_start = bundle_rows[curr_start_idx][0]
          b_end = bundle_rows[i - 1][0]
          t_start = bundle_rows[curr_start_idx][6]
          t_end = bundle_rows[i - 1][7]
          t_dur = t_end - t_start if (t_end and t_start) else 0

          is_prologue = total_bundles > 100 and curr_start_idx <= 10 and i <= 25
          is_epilogue = (
              total_bundles > 100
              and (total_bundles - curr_start_idx <= 20)
              and (i >= total_bundles - 15)
          )

          if (
              num_b >= min_consecutive_bundles or t_dur >= min_duration_ps
          ) and not (is_prologue or is_epilogue):
            scopes, locations, preceding_op, following_op, _, _ = (
                _enrich_bubble_context(c, t_start, t_end, kernel_filter)
            )
            bubbles.append({
                "start_offset_ps": int(t_start),
                "end_offset_ps": int(t_end),
                "duration_ps": int(t_dur),
                "start_bundle": int(b_start),
                "end_bundle": int(b_end),
                "attributed_scopes": scopes,
                "source_locations": locations,
                "preceding_op": preceding_op,
                "following_op": following_op,
            })
          curr_start_idx = None

    if curr_start_idx is not None:
      i = len(bundle_rows)
      num_b = i - curr_start_idx
      b_start = bundle_rows[curr_start_idx][0]
      b_end = bundle_rows[i - 1][0]
      t_start = bundle_rows[curr_start_idx][6]
      t_end = bundle_rows[i - 1][7]
      t_dur = t_end - t_start if (t_end and t_start) else 0

      is_prologue = total_bundles > 100 and curr_start_idx <= 10 and i <= 25
      is_epilogue = total_bundles > 100 and (
          total_bundles - curr_start_idx <= 20
      )

      if (
          num_b >= min_consecutive_bundles or t_dur >= min_duration_ps
      ) and not (is_prologue or is_epilogue):
        scopes, locations, preceding_op, following_op, _, _ = (
            _enrich_bubble_context(c, t_start, t_end, kernel_filter)
        )
        bubbles.append({
            "start_offset_ps": int(t_start),
            "end_offset_ps": int(t_end),
            "duration_ps": int(t_dur),
            "start_bundle": int(b_start),
            "end_bundle": int(b_end),
            "attributed_scopes": scopes,
            "source_locations": locations,
            "preceding_op": preceding_op,
            "following_op": following_op,
        })

  # 2. Check time slices across the entire timeline for gaps / bubbles
  query_range = f"""
    SELECT MIN(offset_ps), MAX(offset_ps + duration_ps)
    FROM llo_events
    WHERE line_name LIKE '%Instructions'{filter_sql}
  """
  c.execute(query_range, filter_params)
  range_row = c.fetchone()
  if range_row and range_row[0] is not None and range_row[1] is not None:
    min_ps = int(range_row[0])
    max_ps = int(range_row[1])
    if max_ps > min_ps:
      slice_ps = 500  # 0.5 ns resolution for exact bubble boundaries
      n_slices = (max_ps - min_ps + slice_ps - 1) // slice_ps
      if n_slices > 2_000_000:
        slice_ps = max(500, (max_ps - min_ps + 1_999_999) // 2_000_000)
        n_slices = (max_ps - min_ps + slice_ps - 1) // slice_ps

      active_ps = bytearray(n_slices)
      query_ev = f"""
        SELECT offset_ps, duration_ps
        FROM llo_events
        WHERE line_name LIKE '%Instructions'{filter_sql}
      """
      c.execute(query_ev, filter_params)
      for off, dur in c.fetchall():
        if off is None or dur is None or dur <= 0:
          continue
        s_idx = max(0, (int(off) - min_ps) // slice_ps)
        e_idx = min(
            n_slices - 1, (int(off) + int(dur) - 1 - min_ps) // slice_ps
        )
        for idx in range(s_idx, e_idx + 1):
          active_ps[idx] = 1

      curr_s = None
      for idx in range(n_slices):
        if active_ps[idx] == 0:
          if curr_s is None:
            curr_s = idx
        else:
          if curr_s is not None:
            t_start = min_ps + curr_s * slice_ps
            t_end = min_ps + idx * slice_ps
            t_dur = t_end - t_start
            is_prologue = n_slices > 100 and curr_s <= 10 and idx <= 25
            is_epilogue = (
                n_slices > 100
                and (n_slices - curr_s <= 20)
                and (idx >= n_slices - 15)
            )
            if t_dur >= min_duration_ps and not (is_prologue or is_epilogue):
              # Check if already covered by bundle bubbles
              if not any(
                  b["start_offset_ps"] <= t_start
                  and b["end_offset_ps"] >= t_end
                  for b in bubbles
              ):
                (
                    scopes,
                    locations,
                    preceding_op,
                    following_op,
                    b_start,
                    b_end,
                ) = _enrich_bubble_context(c, t_start, t_end, kernel_filter)
                bubbles.append({
                    "start_offset_ps": int(t_start),
                    "end_offset_ps": int(t_end),
                    "duration_ps": int(t_dur),
                    "start_bundle": int(b_start),
                    "end_bundle": int(b_end),
                    "attributed_scopes": scopes,
                    "source_locations": locations,
                    "preceding_op": preceding_op,
                    "following_op": following_op,
                })
            curr_s = None

      if curr_s is not None:
        t_start = min_ps + curr_s * slice_ps
        t_end = max_ps
        t_dur = t_end - t_start
        is_prologue = n_slices > 100 and curr_s <= 10 and n_slices <= 25
        is_epilogue = n_slices > 100 and (n_slices - curr_s <= 20)
        if t_dur >= min_duration_ps and not (is_prologue or is_epilogue):
          if not any(
              b["start_offset_ps"] <= t_start and b["end_offset_ps"] >= t_end
              for b in bubbles
          ):
            scopes, locations, preceding_op, following_op, b_start, b_end = (
                _enrich_bubble_context(c, t_start, t_end, kernel_filter)
            )
            bubbles.append({
                "start_offset_ps": int(t_start),
                "end_offset_ps": int(t_end),
                "duration_ps": int(t_dur),
                "start_bundle": int(b_start),
                "end_bundle": int(b_end),
                "attributed_scopes": scopes,
                "source_locations": locations,
                "preceding_op": preceding_op,
                "following_op": following_op,
            })

  bubbles.sort(key=lambda b: b["start_offset_ps"])
  return bubbles


def _enrich_bubble_context(
    c: sqlite3.Cursor,
    start_ps: int,
    end_ps: int,
    kernel_filter: str | None,
) -> tuple[list[str], list[dict[str, Any]], str, str, int, int]:
  """Queries surrounding instructions and inside-gap scalar operations to attach scopes, locations, and bundle boundaries."""
  filter_sql, filter_params = _build_filter_sql(kernel_filter)
  query = f"""
    SELECT llo_scopes, source_info
    FROM llo_events
    WHERE ((offset_ps + duration_ps >= ? AND offset_ps <= ?)
           OR (offset_ps >= ? AND offset_ps <= ? AND line_name IN ('SALU Instructions', 'Pallas Primitives')))
      AND line_name LIKE '%Instructions'{filter_sql}
  """
  margin_ps = 5000
  params = [
      start_ps - margin_ps,
      end_ps + margin_ps,
      start_ps,
      end_ps,
  ] + filter_params
  c.execute(query, params)
  rows = c.fetchall()
  scopes_raw = [r[0] for r in rows if r[0] is not None]
  source_raw = [r[1] for r in rows if r[1] is not None]
  scopes = parse_scopes(scopes_raw)
  locations = parse_source_locations(source_raw)

  # Find the exact preceding instruction right before start_ps
  query_prev = f"""
    SELECT line_name, event_name, bundle_number, offset_ps
    FROM llo_events
    WHERE offset_ps < ? AND line_name LIKE '%Instructions'{filter_sql}
    ORDER BY offset_ps DESC LIMIT 1
  """
  c.execute(query_prev, [start_ps] + filter_params)
  prev_row = c.fetchone()
  preceding_op = (
      f"{prev_row[0]} ({prev_row[1]}) at {prev_row[3]:,} ps"
      if prev_row
      else "None"
  )
  b_start = int(prev_row[2]) if prev_row and prev_row[2] is not None else 0

  # Find the exact following instruction right after end_ps
  query_next = f"""
    SELECT line_name, event_name, bundle_number, offset_ps
    FROM llo_events
    WHERE offset_ps >= ? AND line_name LIKE '%Instructions'{filter_sql}
    ORDER BY offset_ps ASC LIMIT 1
  """
  c.execute(query_next, [end_ps] + filter_params)
  next_row = c.fetchone()
  following_op = (
      f"{next_row[0]} ({next_row[1]}) at {next_row[3]:,} ps"
      if next_row
      else "None"
  )
  b_end = int(next_row[2]) if next_row and next_row[2] is not None else 0

  return scopes, locations, preceding_op, following_op, b_start, b_end


def detect_runtime_blocked_zero_init(
    c: sqlite3.Cursor,
    max_start_delay_pct: float = 5.0,
    min_vst_ratio_pct: float = 30.0,
    kernel_filter: str | None = None,
) -> dict[str, Any]:
  """Detects if kernel start is delayed by pre-compute memory store writes (zero-initialization).

  Args:
    c: SQLite cursor.
    max_start_delay_pct: Percentage threshold of total duration before first
      compute op.
    min_vst_ratio_pct: Percentage threshold of VST writes among memory ops in
      delay period.
    kernel_filter: Optional substring filter.

  Returns:
    Dictionary of blocked zero-init detection results and attributed
    scopes/locations.
  """
  filter_sql, filter_params = _build_filter_sql(kernel_filter)

  query_bounds = f"""
    SELECT MIN(offset_ps), MAX(offset_ps + duration_ps)
    FROM llo_events
    WHERE line_name LIKE '%Instructions'{filter_sql}
  """
  c.execute(query_bounds, filter_params)
  bounds = c.fetchone()
  if (
      not bounds
      or bounds[0] is None
      or bounds[1] is None
      or bounds[1] <= bounds[0]
  ):
    return {
        "detected": False,
        "start_delay_ps": 0,
        "start_delay_pct": 0.0,
        "first_compute_op": "",
        "write_ratio_pct": 0.0,
        "attributed_scopes": [],
        "source_locations": [],
    }

  kernel_start_ps = int(bounds[0])
  kernel_end_ps = int(bounds[1])
  total_duration_ps = kernel_end_ps - kernel_start_ps

  # Find the first computational instruction (MXU, VALU, XLU, SALU, EUP
  # excluding VLD/VST and sync/nop/wait)
  query_compute = f"""
    SELECT offset_ps, event_name
    FROM llo_events
    WHERE offset_ps IS NOT NULL
      AND line_name IN ('MXU Instructions', 'VALU Instructions', 'XLU Instructions', 'SALU Instructions', 'EUP Instructions')
      AND (line_name LIKE '%MXU%' OR event_name NOT LIKE '%sync%' AND event_name NOT LIKE '%wait%' AND event_name NOT LIKE '%nop%'){filter_sql}
    ORDER BY offset_ps ASC
    LIMIT 1
  """
  c.execute(query_compute, filter_params)
  comp_row = c.fetchone()
  if not comp_row or comp_row[0] is None:
    return {
        "detected": False,
        "start_delay_ps": 0,
        "start_delay_pct": 0.0,
        "first_compute_op": "",
        "write_ratio_pct": 0.0,
        "attributed_scopes": [],
        "source_locations": [],
    }

  compute_start_ps = int(comp_row[0])
  first_compute_op = str(comp_row[1] or "")
  start_delay_ps = compute_start_ps - kernel_start_ps
  start_delay_pct = (
      round((start_delay_ps / total_duration_ps * 100.0), 2)
      if total_duration_ps > 0
      else 0.0
  )

  if start_delay_pct <= max_start_delay_pct or start_delay_ps <= 0:
    return {
        "detected": False,
        "start_delay_ps": start_delay_ps,
        "start_delay_pct": start_delay_pct,
        "first_compute_op": first_compute_op,
        "write_ratio_pct": 0.0,
        "attributed_scopes": [],
        "source_locations": [],
    }

  # Check memory operations in [kernel_start_ps, compute_start_ps]
  query_mem = f"""
    SELECT
      SUM(CASE WHEN line_name = 'VST Instructions' THEN 1 ELSE 0 END) AS vst_count,
      COUNT(*) AS total_mem_count
    FROM llo_events
    WHERE offset_ps < ? AND offset_ps >= ?
      AND line_name IN ('VLD Instructions', 'VST Instructions'){filter_sql}
  """
  params_mem = [compute_start_ps, kernel_start_ps] + filter_params
  c.execute(query_mem, params_mem)
  mem_row = c.fetchone()
  vst_count = (mem_row[0] if mem_row else 0) or 0
  total_mem = (mem_row[1] if mem_row else 0) or 0

  write_ratio_pct = (
      round((vst_count / total_mem * 100.0), 2) if total_mem > 0 else 0.0
  )
  detected = write_ratio_pct > min_vst_ratio_pct

  scopes = []
  locations = []
  if detected:
    query_vst = f"""
      SELECT llo_scopes, source_info
      FROM llo_events
      WHERE offset_ps < ? AND offset_ps >= ?
        AND line_name = 'VST Instructions'{filter_sql}
    """
    c.execute(query_vst, params_mem)
    vst_rows = c.fetchall()
    scopes_raw = [r[0] for r in vst_rows if r[0] is not None]
    source_raw = [r[1] for r in vst_rows if r[1] is not None]
    scopes = parse_scopes(scopes_raw)
    locations = parse_source_locations(source_raw)

  return {
      "detected": detected,
      "start_delay_ps": start_delay_ps,
      "start_delay_pct": start_delay_pct,
      "first_compute_op": first_compute_op,
      "write_ratio_pct": write_ratio_pct,
      "attributed_scopes": scopes,
      "source_locations": locations,
  }


def synthesize_findings(
    concurrency_report: dict[str, Any],
    bubbles: list[dict[str, Any]],
    zero_init: dict[str, Any],
) -> list[str]:
  """Synthesizes categorical finding tags from detector outputs.

  Args:
    concurrency_report: Output dict from
      analyze_runtime_concurrency_and_overlap.
    bubbles: Output list from detect_runtime_scheduling_bubbles.
    zero_init: Output dict from detect_runtime_blocked_zero_init.

  Returns:
    List of unique categorical finding tags.
  """
  findings = []
  if bubbles:
    findings.append("runtime_scheduling_bubbles")
  if zero_init and zero_init.get("detected"):
    findings.append("blocked_zero_init")

  if concurrency_report:
    # If pure memory duration > 20% of total duration, flag exposed memory
    if concurrency_report.get("pure_memory_percentage", 0.0) > 20.0:
      findings.append("exposed_memory_bubbles")
    # If pure memory is high or overlapped is low when memory is active
    mem_ps = concurrency_report.get(
        "pure_memory_duration_ps", 0
    ) + concurrency_report.get("overlapped_mxu_mem_duration_ps", 0)
    total_ps = concurrency_report.get("total_duration_ps", 0)
    if total_ps > 0 and mem_ps > 0:
      overlap_pct_of_mem = (
          concurrency_report.get("overlapped_mxu_mem_duration_ps", 0) / mem_ps
      ) * 100.0
      if overlap_pct_of_mem < 50.0 and (mem_ps / total_ps) > 0.15:
        findings.append("low_memory_compute_overlap")

    if concurrency_report.get("exposed_vector_percentage", 0.0) > 25.0 or (
        total_ps > 0
        and (
            concurrency_report.get("exposed_vector_duration_ps", 0)
            / total_ps
            * 100.0
        )
        > 25.0
    ):
      findings.append("exposed_vector_bubbles")

  return sorted(list(set(findings)))
