"""Shared Playwright UI interaction helpers for XProf frontend tests."""

from collections.abc import Mapping
import functools
import gzip
import http
import importlib
import json
import logging
import os
import pathlib
import re
import shutil
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import zlib

# pylint: disable=g-import-not-at-top
try:
  from google3.third_party.xprof.tests.ui import invariants
except ImportError:
  try:
    from tests.ui import invariants  # pyrefly: ignore[missing-import]
  except ImportError:
    import invariants  # pyrefly: ignore[missing-import]
# pylint: enable=g-import-not-at-top

# invariants owns the optional Playwright import and its hermetic stub.
sync_api = invariants.sync_api

# Genuine rendered visualization and data-bearing elements inside XProf tools.
# Deliberately excludes wrappers that exist before anything is drawn (e.g.
# :scope > *, div, the <chart> host) and the decorative svgs Angular Material
# marks focusable="false" or aria-hidden="true" (select arrows, icons,
# spinners, switches), so child geometry checks cannot pass on page chrome.
# An iframe counts once it is visible; its document is not inspected, so an
# iframe that has not drawn anything yet (Graph Viewer's graph) still counts.
VISUALIZATION_SELECTOR = (
    "svg:not([focusable='false'], [aria-hidden='true']), canvas, table,"
    " mat-card, .mat-mdc-card, iframe, op-table-entry .row"
)


def assert_component_geometry(
    page: sync_api.Page,
    selector: str,
    context: str = "",
) -> None:
  """Asserts that a component and its visualization have positive geometry."""
  ctx = f" at {context}" if context else ""
  comp = page.locator(f":is({selector}):visible").first
  sync_api.expect(comp).to_be_visible(timeout=20000)
  bbox = comp.bounding_box()
  assert (
      bbox is not None and bbox["width"] > 0 and bbox["height"] > 0
  ), f"Component {selector!r} collapsed{ctx}: {bbox}"

  vis_selector = (
      f":is({selector}):is({VISUALIZATION_SELECTOR}):visible, "
      f":is({selector}) :is({VISUALIZATION_SELECTOR}):visible"
  )
  child = page.locator(vis_selector).first
  sync_api.expect(child).to_be_visible(timeout=20000)
  child_bbox = child.bounding_box()
  assert (
      child_bbox is not None
      and child_bbox["width"] > 0
      and child_bbox["height"] > 0
  ), f"Component {selector!r} child visualization collapsed{ctx}: {child_bbox}"


def assert_healthy(
    page: sync_api.Page,
    browser_errors: object = None,
    context: str = "",
) -> None:
  """Asserts a non-empty body, clean page invariants and clean browser logs."""
  ctx = f" at {context}" if context else ""
  body = page.locator("body")
  sync_api.expect(body).to_contain_text(re.compile(r"\S"), timeout=20000)
  text = body.inner_text()
  assert text.strip(), f"Empty page body rendered{ctx}"
  violations = invariants.run_content_invariants(text)
  assert not violations, f"Poison tokens detected{ctx}: {violations}"
  if page.locator("main-page mat-sidenav-content").count() > 0:
    sync_api.expect(
        page.locator("main-page mat-sidenav-content > div.full-height")
    ).to_be_visible(timeout=20000)
  invariants.assert_page_invariants(page)
  if browser_errors is not None and hasattr(browser_errors, "assert_clean"):
    browser_errors.assert_clean(context)


def build_tool_url(
    server_url: str,
    session_path: str,
    run: str,
    tag: str,
    **extra_params: str,
) -> str:
  """Constructs a normalized, URL-encoded XProf tool URL."""
  base = server_url.rstrip("/")
  params = {
      "session_path": pathlib.Path(session_path).as_posix(),
      "run": run,
      "tag": tag,
      **extra_params,
  }
  return f"{base}/?{urllib.parse.urlencode(params)}"


def ensure_sidenav_open(page: sync_api.Page) -> None:
  """Ensures the navigation drawer is open and ready for user interactions."""
  drawer = page.locator("mat-sidenav:has(sidenav)")
  sync_api.expect(drawer).to_be_attached(timeout=10000)
  drawer_classes = drawer.get_attribute("class") or ""
  if "mat-drawer-opened" not in drawer_classes:
    toggle_btn = page.locator("button.sidenav-toggle-button")
    toggle_btn.click()
  sync_api.expect(
      page.locator("mat-sidenav.mat-drawer-opened:has(sidenav)")
  ).to_be_visible(timeout=5000)


def _select_option(
    page: sync_api.Page, dropdown: sync_api.Locator, *labels: str
) -> None:
  """Picks the option of a mat-select whose text is one of `labels`.

  An option may end in a parenthesized number, such as an HLO module's program
  id, which differs between profiles.

  Args:
    page: Page holding the dropdown.
    dropdown: The mat-select to open.
    *labels: Accepted option texts.
  """
  pattern = re.compile(
      rf"^\s*(?:{'|'.join(re.escape(label) for label in labels)})"
      r"(?:\(\d+\))?\s*$"
  )
  sync_api.expect(dropdown).to_be_visible(timeout=10000)
  dropdown.click()
  option = page.locator("mat-option").filter(has_text=pattern).first
  sync_api.expect(option).to_be_visible(timeout=5000)
  option.click()
  try:
    page.mouse.move(0, 0)
  except invariants.PlaywrightError as err:
    logging.debug("Ignored mouse reset error after dropdown close: %s", err)
  sync_api.expect(page.locator("mat-option")).to_have_count(0, timeout=5000)
  sync_api.expect(dropdown).to_have_text(pattern, timeout=5000)


def _select_sidenav_dropdown_option(
    page: sync_api.Page, label: str, option_text: str
) -> None:
  """Opens sidenav and clicks an option within the specified dropdown."""
  ensure_sidenav_open(page)
  aliases = [option_text]
  if option_text in ("Op Profile", "HLO Op Profile"):
    aliases = ["Op Profile", "HLO Op Profile"]
  _select_option(
      page,
      page.locator(f"sidenav .item-container:has-text('{label}') mat-select"),
      *aliases,
  )


def select_session(page: sync_api.Page, session_name: str) -> None:
  """Opens the navigation drawer and selects a profile run from dropdown."""
  _select_sidenav_dropdown_option(page, "Sessions", session_name)


def switch_tool(page: sync_api.Page, tool_name: str) -> None:
  """Opens the navigation drawer and switches tools via dropdown."""
  _select_sidenav_dropdown_option(page, "Tools", tool_name)


def select_host(page: sync_api.Page, host_name: str) -> None:
  """Opens the navigation drawer and selects a worker host from dropdown."""
  _select_sidenav_dropdown_option(page, "Hosts", host_name)


def select_module(page: sync_api.Page, module_name: str) -> None:
  """Opens the navigation drawer and selects an HLO module from dropdown."""
  _select_sidenav_dropdown_option(page, "Hlo Modules", module_name)


def select_op_profile_group_by(page: sync_api.Page, group_by: str) -> None:
  """Selects an Op Profile grouping: Program, Category or Provenance."""
  _select_option(
      page,
      page.locator(
          "op-profile-base mat-form-field:has(mat-label:text-is('Group by'))"
          " mat-select"
      ),
      group_by,
  )


def expand_op_profile_row(page: sync_api.Page, row_text: str) -> None:
  """Expands the first Op Profile row containing `row_text` and selects it."""
  row = (
      page.locator("op-table-entry .row:visible")
      .filter(has_text=row_text)
      .first
  )
  sync_api.expect(row).to_be_visible(timeout=10000)
  row.click()
  page.mouse.move(0, 0)
  # Both triangles stay in the DOM; only the rendered one is the state.
  sync_api.expect(row.locator(".disclosure")).to_have_text(
      "\u25bc", use_inner_text=True, timeout=5000
  )
  sync_api.expect(
      page.locator("op-details .info-header-title")
  ).to_contain_text(row_text, timeout=10000)


def plot_graph_node(page: sync_api.Page, node_text: str) -> None:
  """Clicks a Graph Viewer quick-option chip to plot its HLO graph."""
  chip = (
      page.locator("graph-viewer mat-chip-option:visible")
      .filter(has_text=node_text)
      .first
  )
  sync_api.expect(chip).to_be_visible(timeout=15000)
  chip.click()
  sync_api.expect(page).to_have_url(
      re.compile(
          rf"(?=.*[?&]run=[^&#]+).*[?&]node_name=[^&#]*{re.escape(node_text)}"
      ),
      timeout=20000,
  )
  graph_node = (
      page.frame_locator("iframe#graph-html").locator("svg g.node").first
  )
  sync_api.expect(graph_node).to_be_visible(timeout=20000)
  page.mouse.move(0, 0)


def select_memory_id(page: sync_api.Page, memory_id: str) -> None:
  """Selects a memory ID once Memory Profile has loaded its data."""
  sync_api.expect(page.locator("memory-breakdown-table")).to_be_visible(
      timeout=20000
  )
  _select_option(
      page, page.locator("#memory-id-selector mat-select"), memory_id
  )


def select_category_filter(page: sync_api.Page, category: str) -> None:
  """Selects an operation category in HLO Op Stats or Roofline Model."""
  _select_option(
      page,
      page.locator(
          "category-filter:has(mat-label:has-text('Category')) mat-select"
      ),
      category,
  )


def filter_table_rows(page: sync_api.Page, text: str) -> None:
  """Filters the tool's op table by `text` and checks that only matches stay.

  The table must first show a row without `text`, so a filter that has
  nothing to hide cannot pass.

  Args:
    page: Page showing HLO Op Stats, Framework Op Stats or Memory Profile.
    text: Text to filter by, matched case-insensitively.
  """
  table = page.locator(
      "hlo-stats:visible, stats-table:visible, memory-breakdown-table:visible"
  ).first
  rows = table.locator(".google-visualization-table-table tbody tr")
  others = rows.filter(has_not_text=re.compile(re.escape(text), re.IGNORECASE))
  sync_api.expect(others.first).to_be_visible(timeout=20000)
  filter_input = table.locator(
      "string-filter[column='hlo_op_expression'] input,"
      " mat-form-field:has(mat-label:text-is('Operation Type')) input,"
      " input[placeholder='Operation']"
  )
  filter_input.fill(text)
  # Blurring commits the value where the table filters on change, not input.
  filter_input.blur()
  sync_api.expect(others).to_have_count(0, timeout=10000)
  sync_api.expect(rows.first).to_be_visible()
  page.mouse.move(0, 0)


def sort_table_column(page: sync_api.Page, column_name: str) -> None:
  """Clicks a Google Visualization table header to sort by `column_name`."""
  header = (
      page.locator(".google-visualization-table-table thead th:visible")
      .filter(has_text=column_name)
      .first
  )
  sorted_class = re.compile(r"\bsort-(?:ascending|descending)\b")
  sync_api.expect(header).to_be_visible(timeout=10000)
  sync_api.expect(header).not_to_have_class(sorted_class)
  header.click()
  sync_api.expect(header).to_have_class(sorted_class, timeout=5000)
  page.mouse.move(0, 0)


_XPLANE_FILE_DESCRIPTOR_TEXTPROTO = """
name: "xplane.proto"
package: "tensorflow.profiler"
syntax: "proto3"
message_type {
  name: "XSpace"
  field {
    name: "planes"
    number: 1
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".tensorflow.profiler.XPlane"
  }
}
message_type {
  name: "XPlane"
  field { name: "id" number: 1 label: LABEL_OPTIONAL type: TYPE_INT64 }
  field { name: "name" number: 2 label: LABEL_OPTIONAL type: TYPE_STRING }
  field {
    name: "lines"
    number: 3
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".tensorflow.profiler.XLine"
  }
  field {
    name: "event_metadata"
    number: 4
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".tensorflow.profiler.XPlane.EventMetadataEntry"
  }
  field {
    name: "stat_metadata"
    number: 5
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".tensorflow.profiler.XPlane.StatMetadataEntry"
  }
  field {
    name: "stats"
    number: 6
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".tensorflow.profiler.XStat"
  }
  nested_type {
    name: "EventMetadataEntry"
    field { name: "key" number: 1 label: LABEL_OPTIONAL type: TYPE_INT64 }
    field {
      name: "value"
      number: 2
      label: LABEL_OPTIONAL
      type: TYPE_MESSAGE
      type_name: ".tensorflow.profiler.XEventMetadata"
    }
    options { map_entry: true }
  }
  nested_type {
    name: "StatMetadataEntry"
    field { name: "key" number: 1 label: LABEL_OPTIONAL type: TYPE_INT64 }
    field {
      name: "value"
      number: 2
      label: LABEL_OPTIONAL
      type: TYPE_MESSAGE
      type_name: ".tensorflow.profiler.XStatMetadata"
    }
    options { map_entry: true }
  }
}
message_type {
  name: "XLine"
  field { name: "id" number: 1 label: LABEL_OPTIONAL type: TYPE_INT64 }
  field { name: "name" number: 2 label: LABEL_OPTIONAL type: TYPE_STRING }
  field {
    name: "timestamp_ns"
    number: 3
    label: LABEL_OPTIONAL
    type: TYPE_INT64
  }
  field {
    name: "events"
    number: 4
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".tensorflow.profiler.XEvent"
  }
}
message_type {
  name: "XEvent"
  field { name: "metadata_id" number: 1 label: LABEL_OPTIONAL type: TYPE_INT64 }
  field { name: "offset_ps" number: 2 label: LABEL_OPTIONAL type: TYPE_INT64 }
  field { name: "duration_ps" number: 3 label: LABEL_OPTIONAL type: TYPE_INT64 }
  field {
    name: "stats"
    number: 4
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".tensorflow.profiler.XStat"
  }
}
message_type {
  name: "XEventMetadata"
  field { name: "id" number: 1 label: LABEL_OPTIONAL type: TYPE_INT64 }
  field { name: "name" number: 2 label: LABEL_OPTIONAL type: TYPE_STRING }
  field {
    name: "display_name"
    number: 4
    label: LABEL_OPTIONAL
    type: TYPE_STRING
  }
  field {
    name: "stats"
    number: 5
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".tensorflow.profiler.XStat"
  }
}
message_type {
  name: "XStatMetadata"
  field { name: "id" number: 1 label: LABEL_OPTIONAL type: TYPE_INT64 }
  field { name: "name" number: 2 label: LABEL_OPTIONAL type: TYPE_STRING }
}
message_type {
  name: "XStat"
  field { name: "metadata_id" number: 1 label: LABEL_OPTIONAL type: TYPE_INT64 }
  field {
    name: "double_value"
    number: 2
    label: LABEL_OPTIONAL
    type: TYPE_DOUBLE
  }
  field {
    name: "uint64_value"
    number: 3
    label: LABEL_OPTIONAL
    type: TYPE_UINT64
  }
  field { name: "int64_value" number: 4 label: LABEL_OPTIONAL type: TYPE_INT64 }
  field { name: "str_value" number: 5 label: LABEL_OPTIONAL type: TYPE_STRING }
  field { name: "bytes_value" number: 6 label: LABEL_OPTIONAL type: TYPE_BYTES }
}
"""

_INFERENCE_SERVING_XSPACE_TEXTPROTO = """
planes {
  id: 1
  name: "/host:CPU"
  lines {
    id: 1
    name: "RequestThread"
    events {
      metadata_id: 1
      offset_ps: 100
      duration_ps: 20000
      stats { metadata_id: 1 int64_value: 1 }
      stats { metadata_id: 2 str_value: "/models/ranking_model:1" }
    }
    events {
      metadata_id: 2
      offset_ps: 200
      duration_ps: 200
      stats { metadata_id: 3 int64_value: 4 }
      stats { metadata_id: 4 int64_value: 1 }
      stats { metadata_id: 5 int64_value: 8 }
    }
  }
  lines {
    id: 2
    name: "BatchThread"
    events {
      metadata_id: 3
      offset_ps: 500
      duration_ps: 15000
      stats { metadata_id: 6 int64_value: 4 }
      stats { metadata_id: 7 int64_value: 1 }
      stats { metadata_id: 1 int64_value: 2 }
      stats { metadata_id: 3 int64_value: 12 }
      stats { metadata_id: 4 int64_value: 42 }
    }
    events {
      metadata_id: 4
      offset_ps: 600
      duration_ps: 100
      stats { metadata_id: 8 int64_value: 0 }
      stats { metadata_id: 9 int64_value: 8 }
    }
  }
  event_metadata { key: 1 value { id: 1 name: "SessionRun" } }
  event_metadata { key: 2 value { id: 2 name: "ScheduleWithoutSplit" } }
  event_metadata { key: 3 value { id: 3 name: "ProcessBatch" } }
  event_metadata { key: 4 value { id: 4 name: "ConcatInputTensors" } }
  stat_metadata { key: 1 value { id: 1 name: "_r" } }
  stat_metadata { key: 2 value { id: 2 name: "model_id" } }
  stat_metadata { key: 3 value { id: 3 name: "_pt" } }
  stat_metadata { key: 4 value { id: 4 name: "_p" } }
  stat_metadata { key: 5 value { id: 5 name: "batching_input_task_size" } }
  stat_metadata { key: 6 value { id: 6 name: "_ct" } }
  stat_metadata { key: 7 value { id: 7 name: "_c" } }
  stat_metadata { key: 8 value { id: 8 name: "padding_amount" } }
  stat_metadata { key: 9 value { id: 9 name: "batch_size_after_padding" } }
}
planes {
  id: 2
  name: "/device:TPU:0"
  lines {
    id: 1
    name: "XLA Modules"
    events {
      metadata_id: 1
      offset_ps: 1000
      duration_ps: 10000
      stats { metadata_id: 4 int64_value: 12 }
      stats { metadata_id: 5 int64_value: 42 }
    }
  }
  lines {
    id: 2
    name: "XLA Ops"
    events {
      metadata_id: 2
      offset_ps: 2000
      duration_ps: 8000
    }
  }
  event_metadata {
    key: 1
    value { id: 1 name: "jit_serve(5621224425449793704)" }
  }
  event_metadata {
    key: 2
    value {
      id: 2
      name: "fusion.1"
      stats { metadata_id: 6 int64_value: 5621224425449793704 }
      stats { metadata_id: 7 int64_value: 1 }
      stats { metadata_id: 8 str_value: "fusion" }
      stats { metadata_id: 9 int64_value: 1000000 }
      stats { metadata_id: 10 int64_value: 100000 }
    }
  }
  stat_metadata { key: 1 value { id: 1 name: "peak_teraflops_per_second" } }
  stat_metadata {
    key: 2
    value { id: 2 name: "peak_hbm_bw_gigabytes_per_second" }
  }
  stat_metadata { key: 3 value { id: 3 name: "device_type_string" } }
  stat_metadata { key: 4 value { id: 4 name: "_ct" } }
  stat_metadata { key: 5 value { id: 5 name: "_c" } }
  stat_metadata { key: 6 value { id: 6 name: "program_id" } }
  stat_metadata { key: 7 value { id: 7 name: "symbol_id" } }
  stat_metadata { key: 8 value { id: 8 name: "hlo_category" } }
  stat_metadata { key: 9 value { id: 9 name: "flops" } }
  stat_metadata { key: 10 value { id: 10 name: "bytes_accessed" } }
  stats { metadata_id: 1 double_value: 275.0 }
  stats { metadata_id: 2 double_value: 1200.0 }
  stats { metadata_id: 3 str_value: "TPU v4" }
}
"""


_PERF_UTILIZATION_XSPACE_TEMPLATE = """
planes {{
  id: 1
  name: "/host:CPU"
  lines {{
    id: 1
    name: "MainThread"
    events {{
      metadata_id: 1
      offset_ps: 0
      duration_ps: 2000000
    }}
  }}
  event_metadata {{ key: 1 value {{ id: 1 name: "TrainStep" }} }}
}}
planes {{
  id: 2
  name: "/device:TPU:0"
  stats {{ metadata_id: 1 int64_value: 0 }}
  stats {{ metadata_id: 2 str_value: "{device_type}" }}
  lines {{
    id: 1
    name: "XLA Ops"
    events {{
      metadata_id: 10
      offset_ps: 1000
      duration_ps: 1000000000
    }}
  }}
  lines {{
    id: 2
    name: "counters_test_matmul"
    events {{
      metadata_id: 101
      offset_ps: 1000
      duration_ps: 1000000000
      stats {{ metadata_id: 3 uint64_value: 3257466888 }}
      stats {{ metadata_id: 4 uint64_value: 1750000 }}
      stats {{ metadata_id: 5 str_value: "TC unprivileged cycles" }}
      stats {{ metadata_id: 6 str_value: "TC" }}
    }}
    events {{
      metadata_id: 102
      offset_ps: 1000
      duration_ps: 1000000000
      stats {{ metadata_id: 3 uint64_value: 3257467408 }}
      stats {{ metadata_id: 4 uint64_value: 875000 }}
      stats {{ metadata_id: 5 str_value: "MXU 1-core busy cycles" }}
      stats {{ metadata_id: 6 str_value: "MXU TC" }}
    }}
    events {{
      metadata_id: 103
      offset_ps: 1000
      duration_ps: 1000000000
      stats {{ metadata_id: 3 uint64_value: 3257467352 }}
      stats {{ metadata_id: 4 uint64_value: 400000 }}
      stats {{ metadata_id: 5 str_value: "MXU 0 BF16 LMR" }}
      stats {{ metadata_id: 6 str_value: "MXU TC" }}
    }}
    events {{
      metadata_id: 104
      offset_ps: 1000
      duration_ps: 1000000000
      stats {{ metadata_id: 3 uint64_value: 3257467136 }}
      stats {{ metadata_id: 4 uint64_value: 250000 }}
      stats {{ metadata_id: 5 str_value: "Vector ALU 0 ops" }}
      stats {{ metadata_id: 6 str_value: "VPU TC" }}
    }}
    events {{
      metadata_id: 105
      offset_ps: 1000
      duration_ps: 1000000000
      stats {{ metadata_id: 3 uint64_value: 3257466968 }}
      stats {{ metadata_id: 4 uint64_value: 100000 }}
      stats {{ metadata_id: 5 str_value: "Scalar ALU 0 ops" }}
      stats {{ metadata_id: 6 str_value: "Scalar TC" }}
    }}
    events {{
      metadata_id: 106
      offset_ps: 1000
      duration_ps: 1000000000
      stats {{ metadata_id: 3 uint64_value: 3257467536 }}
      stats {{ metadata_id: 4 uint64_value: 0 }}
      stats {{ metadata_id: 5 str_value: "XLU 0 busy zero test" }}
      stats {{ metadata_id: 6 str_value: "XLU TC" }}
    }}
  }}
  event_metadata {{ key: 10 value {{ id: 10 name: "test_matmul" }} }}
  event_metadata {{ key: 101 value {{ id: 101 name: "tcs_count_cycles" }} }}
  event_metadata {{ key: 102 value {{ id: 102 name: "mxu_busy_1" }} }}
  event_metadata {{
    key: 103
    value {{ id: 103 name: "matmul_lmr_bf16_mxu_0" }}
  }}
  event_metadata {{ key: 104 value {{ id: 104 name: "vector_alu_0" }} }}
  event_metadata {{ key: 105 value {{ id: 105 name: "scalar_alu_0" }} }}
  event_metadata {{ key: 106 value {{ id: 106 name: "xlu_busy_0" }} }}
  stat_metadata {{ key: 1 value {{ id: 1 name: "device_id" }} }}
  stat_metadata {{ key: 2 value {{ id: 2 name: "device_type_string" }} }}
  stat_metadata {{ key: 3 value {{ id: 3 name: "performance_counter_id" }} }}
  stat_metadata {{ key: 4 value {{ id: 4 name: "counter_value" }} }}
  stat_metadata {{
    key: 5
    value {{ id: 5 name: "performance_counter_description" }}
  }}
  stat_metadata {{ key: 6 value {{ id: 6 name: "performance_counter_sets" }} }}
}}
"""

_EMPTY_MEMORY_XSPACE_TEXTPROTO = """
planes {
  id: 1
  name: "/device:TPU:0"
  lines {
    id: 1
    name: "XLA Ops"
    events { metadata_id: 1 offset_ps: 1000 duration_ps: 5000 }
  }
  event_metadata { key: 1 value { id: 1 name: "noop" } }
  stat_metadata { key: 1 value { id: 1 name: "device_id" } }
  stat_metadata { key: 2 value { id: 2 name: "device_type_string" } }
  stats { metadata_id: 1 int64_value: 0 }
  stats { metadata_id: 2 str_value: "TPU v6 Lite" }
}
planes {
  id: 2
  name: "/host:CPU"
  lines {
    id: 1
    name: "MainThread"
    events { metadata_id: 1 offset_ps: 0 duration_ps: 5000 }
  }
  event_metadata { key: 1 value { id: 1 name: "TrainStep" } }
}
"""

_DCN_COLLECTIVE_INFO_DESCRIPTOR_TEXTPROTO = """
name: "dcn_collective_info.proto"
package: "tensorflow.profiler"
syntax: "proto3"
message_type {
  name: "DcnCollectiveInfoProto"
  field {
    name: "transfer_type"
    number: 1
    label: LABEL_OPTIONAL
    type: TYPE_INT32
  }
  field {
    name: "endpoint_groups"
    number: 2
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".tensorflow.profiler.DcnCollectiveInfoProto.EndpointGroup"
  }
  nested_type {
    name: "Endpoint"
    field { name: "slice_id" number: 1 label: LABEL_OPTIONAL type: TYPE_INT32 }
    field { name: "device_id" number: 2 label: LABEL_OPTIONAL type: TYPE_INT32 }
  }
  nested_type {
    name: "EndpointGroup"
    field {
      name: "endpoints"
      number: 1
      label: LABEL_REPEATED
      type: TYPE_MESSAGE
      type_name: ".tensorflow.profiler.DcnCollectiveInfoProto.Endpoint"
    }
  }
}
"""

_DEFAULT_DCN_COLLECTIVE_INFO_TEXTPROTO = """
transfer_type: 5
endpoint_groups {
  endpoints { slice_id: 0 device_id: 0 }
  endpoints { slice_id: 1 device_id: 0 }
  endpoints { slice_id: 2 device_id: 0 }
  endpoints { slice_id: 3 device_id: 0 }
}
"""

_MINIMAL_HLO_FILE_DESCRIPTOR_TEXTPROTO = """
name: "minimal_hlo.proto"
package: "xla"
syntax: "proto3"
message_type {
  name: "LayoutProto"
  field {
    name: "minor_to_major"
    number: 1
    label: LABEL_REPEATED
    type: TYPE_INT64
  }
}
message_type {
  name: "ShapeProto"
  field {
    name: "element_type"
    number: 2
    label: LABEL_OPTIONAL
    type: TYPE_INT32
  }
  field {
    name: "dimensions"
    number: 3
    label: LABEL_REPEATED
    type: TYPE_INT64
  }
  field {
    name: "tuple_shapes"
    number: 4
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".xla.ShapeProto"
  }
  field {
    name: "layout"
    number: 5
    label: LABEL_OPTIONAL
    type: TYPE_MESSAGE
    type_name: ".xla.LayoutProto"
  }
}
message_type {
  name: "ProgramShapeProto"
  field {
    name: "parameters"
    number: 1
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".xla.ShapeProto"
  }
  field {
    name: "result"
    number: 2
    label: LABEL_OPTIONAL
    type: TYPE_MESSAGE
    type_name: ".xla.ShapeProto"
  }
  field {
    name: "parameter_names"
    number: 3
    label: LABEL_REPEATED
    type: TYPE_STRING
  }
}
message_type {
  name: "FrontendAttributes"
  field {
    name: "map"
    number: 1
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".xla.FrontendAttributes.MapEntry"
  }
  nested_type {
    name: "MapEntry"
    field { name: "key" number: 1 label: LABEL_OPTIONAL type: TYPE_STRING }
    field { name: "value" number: 2 label: LABEL_OPTIONAL type: TYPE_STRING }
    options { map_entry: true }
  }
}
message_type {
  name: "HloInstructionProto"
  field { name: "name" number: 1 label: LABEL_OPTIONAL type: TYPE_STRING }
  field { name: "opcode" number: 2 label: LABEL_OPTIONAL type: TYPE_STRING }
  field {
    name: "shape"
    number: 3
    label: LABEL_OPTIONAL
    type: TYPE_MESSAGE
    type_name: ".xla.ShapeProto"
  }
  field {
    name: "parameter_number"
    number: 9
    label: LABEL_OPTIONAL
    type: TYPE_INT64
  }
  field {
    name: "channel_id"
    number: 26
    label: LABEL_OPTIONAL
    type: TYPE_INT64
  }
  field { name: "id" number: 35 label: LABEL_OPTIONAL type: TYPE_INT64 }
  field {
    name: "operand_ids"
    number: 36
    label: LABEL_REPEATED
    type: TYPE_INT64
  }
  field {
    name: "is_host_transfer"
    number: 47
    label: LABEL_OPTIONAL
    type: TYPE_BOOL
  }
  field {
    name: "frontend_attributes"
    number: 68
    label: LABEL_OPTIONAL
    type: TYPE_MESSAGE
    type_name: ".xla.FrontendAttributes"
  }
}
message_type {
  name: "HloComputationProto"
  field { name: "name" number: 1 label: LABEL_OPTIONAL type: TYPE_STRING }
  field {
    name: "instructions"
    number: 2
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".xla.HloInstructionProto"
  }
  field {
    name: "program_shape"
    number: 4
    label: LABEL_OPTIONAL
    type: TYPE_MESSAGE
    type_name: ".xla.ProgramShapeProto"
  }
  field { name: "id" number: 5 label: LABEL_OPTIONAL type: TYPE_INT64 }
  field { name: "root_id" number: 6 label: LABEL_OPTIONAL type: TYPE_INT64 }
}
message_type {
  name: "HloModuleProto"
  field { name: "name" number: 1 label: LABEL_OPTIONAL type: TYPE_STRING }
  field {
    name: "entry_computation_name"
    number: 2
    label: LABEL_OPTIONAL
    type: TYPE_STRING
  }
  field {
    name: "computations"
    number: 3
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".xla.HloComputationProto"
  }
  field {
    name: "host_program_shape"
    number: 4
    label: LABEL_OPTIONAL
    type: TYPE_MESSAGE
    type_name: ".xla.ProgramShapeProto"
  }
  field { name: "id" number: 5 label: LABEL_OPTIONAL type: TYPE_INT64 }
  field {
    name: "entry_computation_id"
    number: 6
    label: LABEL_OPTIONAL
    type: TYPE_INT64
  }
}
message_type {
  name: "HloProto"
  field {
    name: "hlo_module"
    number: 1
    label: LABEL_OPTIONAL
    type: TYPE_MESSAGE
    type_name: ".xla.HloModuleProto"
  }
}
"""

_MEGASCALE_HLO_TEXTPROTO = """
hlo_module {
  name: "megascale_dcn_module"
  entry_computation_name: "main"
  entry_computation_id: 1
  id: 9001
  host_program_shape {
    parameters {
      element_type: 11
      dimensions: 1024
      layout { minor_to_major: 0 }
    }
    parameter_names: "p0"
    result {
      element_type: 13
      tuple_shapes {
        element_type: 11
        dimensions: 1024
        layout { minor_to_major: 0 }
      }
      tuple_shapes { element_type: 17 }
    }
  }
  computations {
    name: "main"
    id: 1
    root_id: 6
    program_shape {
      parameters {
        element_type: 11
        dimensions: 1024
        layout { minor_to_major: 0 }
      }
      parameter_names: "p0"
      result {
        element_type: 13
        tuple_shapes {
          element_type: 11
          dimensions: 1024
          layout { minor_to_major: 0 }
        }
        tuple_shapes { element_type: 17 }
      }
    }
    instructions {
      id: 1
      name: "p0"
      opcode: "parameter"
      parameter_number: 0
      shape {
        element_type: 11
        dimensions: 1024
        layout { minor_to_major: 0 }
      }
    }
    instructions {
      id: 2
      name: "tok0"
      opcode: "after-all"
      shape { element_type: 17 }
    }
    instructions {
      id: 3
      name: "send.1"
      opcode: "send"
      operand_ids: 1
      operand_ids: 2
      channel_id: 1
      is_host_transfer: true
      shape {
        element_type: 13
        tuple_shapes {
          element_type: 11
          dimensions: 1024
          layout { minor_to_major: 0 }
        }
        tuple_shapes { element_type: 8 layout {} }
        tuple_shapes { element_type: 17 }
      }
      frontend_attributes {
        map { key: "_xla_host_transfer_rendezvous" value: "all_reduce_dcn_0" }
        map { key: "_xla_megascale_transfer_type" value: "ALL_REDUCE" }
      }
    }
    instructions {
      id: 4
      name: "send-done.1"
      opcode: "send-done"
      operand_ids: 3
      channel_id: 1
      is_host_transfer: true
      shape { element_type: 17 }
    }
    instructions {
      id: 5
      name: "recv.1"
      opcode: "recv"
      operand_ids: 4
      channel_id: 2
      is_host_transfer: true
      shape {
        element_type: 13
        tuple_shapes {
          element_type: 11
          dimensions: 1024
          layout { minor_to_major: 0 }
        }
        tuple_shapes { element_type: 8 layout {} }
        tuple_shapes { element_type: 17 }
      }
      frontend_attributes {
        map { key: "_xla_host_transfer_rendezvous" value: "all_reduce_dcn_0" }
      }
    }
    instructions {
      id: 6
      name: "recv-done.1"
      opcode: "recv-done"
      operand_ids: 5
      channel_id: 2
      is_host_transfer: true
      shape {
        element_type: 13
        tuple_shapes {
          element_type: 11
          dimensions: 1024
          layout { minor_to_major: 0 }
        }
        tuple_shapes { element_type: 17 }
      }
    }
  }
}
"""

SYNTHESIZED_RUNS: frozenset[str] = frozenset({
    "empty-memory",
    "gpu-training",
    "megascale-1host",
    "megascale-2host",
    "multi-host",
    "perf-utilization",
    "perf-v4",
    "tpu-serving",
})


@functools.cache
def _build_proto_parser(
    descriptor_textproto: str, message_name: str
) -> tuple[Any, Any]:
  """Builds and caches a runtime protobuf message class for `message_name`.

  Args:
    descriptor_textproto: Text-format `FileDescriptorProto` schema.
    message_name: Fully-qualified protobuf message name to look up.

  Returns:
    Tuple of `(message_class, text_format_module)`.
  """
  descriptor_pb2 = importlib.import_module("google.protobuf.descriptor_pb2")
  descriptor_pool = importlib.import_module("google.protobuf.descriptor_pool")
  message_factory = importlib.import_module("google.protobuf.message_factory")
  text_format = importlib.import_module("google.protobuf.text_format")
  pool = descriptor_pool.DescriptorPool()
  pool.Add(
      text_format.Parse(
          descriptor_textproto, descriptor_pb2.FileDescriptorProto()
      )
  )
  msg_cls = message_factory.GetMessageClass(
      pool.FindMessageTypeByName(message_name)
  )
  return msg_cls, text_format


def _build_xspace_parser() -> tuple[Any, Any]:
  """Builds a runtime protobuf parser for `tensorflow.profiler.XSpace`.

  Returns:
    Tuple of `(xspace_class, text_format_module)`.
  """
  return _build_proto_parser(
      _XPLANE_FILE_DESCRIPTOR_TEXTPROTO, "tensorflow.profiler.XSpace"
  )


def _write_xspace_textproto(path: pathlib.Path, textproto: str) -> None:
  """Serializes `textproto` as an `XSpace` binary proto to `path`.

  Args:
    path: Destination `.xplane.pb` file path.
    textproto: Text-format `XSpace` proto payload.
  """
  xspace_cls, text_format = _build_xspace_parser()
  msg = text_format.Parse(textproto, xspace_cls())
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(msg.SerializeToString())


def write_inference_serving_trace(path: pathlib.Path) -> None:
  """Writes a minimal TPU inference serving `.xplane.pb` trace to `path`.

  Args:
    path: Destination `.xplane.pb` file path.
  """
  _write_xspace_textproto(path, _INFERENCE_SERVING_XSPACE_TEXTPROTO)


def write_perf_utilization_trace(
    path: pathlib.Path, device_type: str = "TPU v6 Lite"
) -> None:
  """Writes a TPU performance-counter and utilization trace to `path`.

  Args:
    path: Destination `.xplane.pb` file path.
    device_type: Target TPU device string recorded on the device plane.
  """
  _write_xspace_textproto(
      path, _PERF_UTILIZATION_XSPACE_TEMPLATE.format(device_type=device_type)
  )


def write_empty_memory_trace(path: pathlib.Path) -> None:
  """Writes a minimal TPU trace with zero host memory allocators to `path`.

  Args:
    path: Destination `.xplane.pb` file path.
  """
  _write_xspace_textproto(path, _EMPTY_MEMORY_XSPACE_TEXTPROTO)


@functools.cache
def _serialize_megascale_hlo_and_dcn_info() -> tuple[bytes, bytes]:
  """Serializes the cached `HloProto` and `DcnCollectiveInfoProto` payloads.

  Returns:
    Tuple of serialized `(hlo_proto_bytes, dcn_collective_info_bytes)`.
  """
  hlo_cls, text_format = _build_proto_parser(
      _MINIMAL_HLO_FILE_DESCRIPTOR_TEXTPROTO, "xla.HloProto"
  )
  dcn_info_cls, _ = _build_proto_parser(
      _DCN_COLLECTIVE_INFO_DESCRIPTOR_TEXTPROTO,
      "tensorflow.profiler.DcnCollectiveInfoProto",
  )
  hlo_bytes = text_format.Parse(
      _MEGASCALE_HLO_TEXTPROTO, hlo_cls()
  ).SerializeToString()
  dcn_info_bytes = text_format.Parse(
      _DEFAULT_DCN_COLLECTIVE_INFO_TEXTPROTO, dcn_info_cls()
  ).SerializeToString()
  return hlo_bytes, dcn_info_bytes


def _get_or_create_stat_metadata_id(plane: Any, stat_name: str) -> int:
  """Returns the stat metadata ID for `stat_name`, creating one if absent.

  Args:
    plane: Mutable `XPlane` protobuf object.
    stat_name: Metadata entry name to find or add.

  Returns:
    Integer stat metadata ID on `plane`.
  """
  for stat_id, meta in plane.stat_metadata.items():
    if meta.name == stat_name:
      return stat_id
  next_id = max(plane.stat_metadata.keys(), default=0) + 1
  meta = plane.stat_metadata[next_id]
  meta.id = next_id
  meta.name = stat_name
  return next_id


def _get_or_create_line(
    plane: Any, line_name: str, timestamp_ns: int | None = None
) -> Any:
  """Gets or creates an `XLine` named `line_name` on `plane`.

  Args:
    plane: Mutable `XPlane` protobuf object.
    line_name: Name of the line to locate or create.
    timestamp_ns: Optional start timestamp in nanoseconds. When omitted (`None`)
      and `line_name` is created, inherits the earliest non-zero sibling line
      timestamp on `plane` (or `1000` when `plane` has no timestamped lines).

  Returns:
    Existing or newly created `XLine` message.
  """
  for line in plane.lines:
    if line.name == line_name:
      return line
  next_id = max((line.id for line in plane.lines), default=0) + 1
  if timestamp_ns is None:
    timestamp_ns = min(
        (line.timestamp_ns for line in plane.lines if line.timestamp_ns),
        default=1000,
    )
  return plane.lines.add(id=next_id, name=line_name, timestamp_ns=timestamp_ns)


def write_megascale_trace(
    path: pathlib.Path, base_trace_path: pathlib.Path | None = None
) -> None:
  """Writes an `.xplane.pb` with MegaScale DCN ops and HloProto.

  Args:
    path: Destination `.xplane.pb` file path.
    base_trace_path: Optional base `.xplane.pb` trace to augment. When `None`
      or not an existing file, starts from the built-in TPU serving fixture.
  """
  xspace_cls, text_format = _build_xspace_parser()
  if base_trace_path is not None and base_trace_path.is_file():
    mega_space = xspace_cls.FromString(base_trace_path.read_bytes())
  else:
    mega_space = text_format.Parse(
        _INFERENCE_SERVING_XSPACE_TEXTPROTO, xspace_cls()
    )

  cpu_plane = next(
      (p for p in mega_space.planes if p.name == "/host:CPU"), None
  )
  if cpu_plane is None:
    next_plane_id = max((p.id for p in mega_space.planes), default=0) + 1
    cpu_plane = mega_space.planes.add(id=next_plane_id, name="/host:CPU")

  tpu_planes = [
      p for p in mega_space.planes if p.name.startswith("/device:TPU:")
  ]
  if not tpu_planes:
    next_plane_id = max((p.id for p in mega_space.planes), default=0) + 1
    tpu_plane = mega_space.planes.add(id=next_plane_id, name="/device:TPU:0")
    dev_id_stat = _get_or_create_stat_metadata_id(tpu_plane, "device_id")
    dev_type_stat = _get_or_create_stat_metadata_id(
        tpu_plane, "device_type_string"
    )
    tpu_plane.stats.add(metadata_id=dev_id_stat, int64_value=0)
    tpu_plane.stats.add(metadata_id=dev_type_stat, str_value="TPU v4")
    tpu_planes = [tpu_plane]

  for plane in [cpu_plane, *tpu_planes]:
    mega_em_id = max(plane.event_metadata.keys(), default=999998) + 1
    em = plane.event_metadata[mega_em_id]
    em.id = mega_em_id
    em.name = "MegaScale:send_dcn"
    mega_line = _get_or_create_line(plane, "MegaScale")
    mega_line.events.add(
        metadata_id=mega_em_id, offset_ps=1000, duration_ps=50000
    )

  hlo_bytes, dcn_info_bytes = _serialize_megascale_hlo_and_dcn_info()
  meta_plane = next(
      (p for p in mega_space.planes if p.name == "/host:metadata"), None
  )
  if meta_plane is None:
    next_plane_id = max((p.id for p in mega_space.planes), default=0) + 1
    meta_plane = mega_space.planes.add(id=next_plane_id, name="/host:metadata")
  hlo_stat_id = _get_or_create_stat_metadata_id(meta_plane, "Hlo Proto")
  prog_stat_id = _get_or_create_stat_metadata_id(meta_plane, "program_id")
  hlo_em_id = max(meta_plane.event_metadata.keys(), default=9000) + 1
  hlo_em = meta_plane.event_metadata[hlo_em_id]
  hlo_em.id = hlo_em_id
  hlo_em.name = "megascale_dcn_module(9001)"
  del hlo_em.stats[:]
  hlo_em.stats.add(metadata_id=hlo_stat_id, bytes_value=hlo_bytes)
  hlo_em.stats.add(metadata_id=prog_stat_id, int64_value=9001)

  tpu0 = next(
      (p for p in tpu_planes if p.name == "/device:TPU:0"), tpu_planes[0]
  )
  cat_stat_id = _get_or_create_stat_metadata_id(tpu0, "hlo_category")
  dcn_stat_id = _get_or_create_stat_metadata_id(tpu0, "dcn_collective_info")
  mod_line = _get_or_create_line(tpu0, "XLA Modules")
  ops_line = _get_or_create_line(tpu0, "XLA Ops")
  max_end_ps = 0
  for ln in (mod_line, ops_line):
    for ev in ln.events:
      max_end_ps = max(max_end_ps, ev.offset_ps + ev.duration_ps)
  base_ps = max_end_ps + 10_000_000

  next_em_id = max(tpu0.event_metadata.keys(), default=1000) + 1
  mod_em = tpu0.event_metadata[next_em_id]
  mod_em.id = next_em_id
  mod_em.name = "megascale_dcn_module(9001)"
  mod_line.events.add(
      metadata_id=next_em_id, offset_ps=base_ps, duration_ps=20_000_000
  )

  dcn_ops = (
      ("send.1", "host send", 2_000_000, 2_000_000, dcn_info_bytes),
      ("send-done.1", "host send", 4_000_000, 1_000_000, b""),
      ("recv.1", "host recv", 8_000_000, 1_000_000, b""),
      ("recv-done.1", "host recv", 9_000_000, 3_000_000, b""),
  )
  for idx, (op_name, category, rel_start, dur, info_payload) in enumerate(
      dcn_ops, start=1
  ):
    em_id = next_em_id + idx
    op_em = tpu0.event_metadata[em_id]
    op_em.id = em_id
    op_em.name = op_name
    op_em.display_name = op_name
    op_em.stats.add(metadata_id=cat_stat_id, str_value=category)
    if info_payload:
      op_em.stats.add(metadata_id=dcn_stat_id, bytes_value=info_payload)
    ops_line.events.add(
        metadata_id=em_id, offset_ps=base_ps + rel_start, duration_ps=dur
    )

  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(mega_space.SerializeToString())


def stage_synthesized_runs(
    staged_dir: pathlib.Path, repo_root: pathlib.Path | None = None
) -> None:
  """Populates multi-tool and edge-case runs inside a writable `staged_dir`.

  Args:
    staged_dir: Writable profile logdir directory to populate.
    repo_root: Optional repository root for locating `gpu_training_2.xplane.pb`.
      When `None`, only `/tmp/convert_test_xplanes` is checked for GPU traces.
  """
  base_tpu_trace = next(
      iter(
          sorted(staged_dir.glob("tpu[-_]training/*.xplane.pb"))
          or [
              p
              for p in sorted(staged_dir.glob("*/*.xplane.pb"))
              if p.parent.name != "gpu-training"
          ]
      ),
      None,
  )

  gpu_candidates: list[pathlib.Path] = [
      pathlib.Path("/tmp/convert_test_xplanes/gpu_training_2.xplane.pb"),
  ]
  if repo_root is not None:
    gpu_candidates.append(
        repo_root / "convert" / "test_xplanes" / "gpu_training_2.xplane.pb"
    )
    gpu_candidates.append(
        repo_root
        / "xprof"
        / "convert"
        / "test_xplanes"
        / "gpu_training_2.xplane.pb"
    )
  for gpu_trace in gpu_candidates:
    if gpu_trace.is_file() and not (staged_dir / "gpu-training").exists():
      (staged_dir / "gpu-training").mkdir(parents=True, exist_ok=True)
      (staged_dir / "gpu-training" / "gpu-node-1.xplane.pb").symlink_to(
          gpu_trace.resolve()
      )
      break

  if not (staged_dir / "tpu-serving").exists():
    write_inference_serving_trace(
        staged_dir / "tpu-serving" / "serving-node-1.xplane.pb"
    )

  if base_tpu_trace is not None and not (staged_dir / "multi-host").exists():
    mh_dir = staged_dir / "multi-host"
    mh_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_tpu_trace, mh_dir / "worker-0.xplane.pb")
    shutil.copy2(base_tpu_trace, mh_dir / "worker-1.xplane.pb")

  if not (staged_dir / "perf-utilization").exists():
    write_perf_utilization_trace(
        staged_dir / "perf-utilization" / "perf-node-1.xplane.pb",
        device_type="TPU v6 Lite",
    )
  if not (staged_dir / "perf-v4").exists():
    write_perf_utilization_trace(
        staged_dir / "perf-v4" / "perf-v4-node-1.xplane.pb",
        device_type="TPU v4",
    )
  if not (staged_dir / "megascale-1host").exists():
    write_megascale_trace(
        staged_dir / "megascale-1host" / "mega-node-0.xplane.pb",
        base_tpu_trace,
    )
  if not (staged_dir / "megascale-2host").exists():
    write_megascale_trace(
        staged_dir / "megascale-2host" / "mega-node-0.xplane.pb",
        base_tpu_trace,
    )
    shutil.copy2(
        staged_dir / "megascale-2host" / "mega-node-0.xplane.pb",
        staged_dir / "megascale-2host" / "mega-node-1.xplane.pb",
    )
  if not (staged_dir / "empty-memory").exists():
    write_empty_memory_trace(
        staged_dir / "empty-memory" / "empty-mem-0.xplane.pb"
    )


def resolve_server_logdir(profile_logdir: str) -> str:
  """Returns the TensorBoard root logdir for a `.../plugins/profile` path.

  Args:
    profile_logdir: Profile plugin directory or standalone run directory.

  Returns:
    Parent root logdir when `profile_logdir` ends in `plugins/profile`,
    otherwise `profile_logdir` unchanged.
  """
  path = pathlib.Path(profile_logdir)
  if path.parts[-2:] == ("plugins", "profile"):
    return str(path.parent.parent)
  return profile_logdir


def is_server_ready(
    url: str,
    expected_runs: set[str] | None = None,
    profile_logdir: str | None = None,
) -> bool:
  """Polls the HTTP server until its profile plugin responds with runs.

  Args:
    url: Base HTTP URL of the XProf server.
    expected_runs: Optional run names to check in `/runs`. When `None` or empty,
      checks only that the root URL responds with status below 500.
    profile_logdir: Optional profile logdir passed via `run_path` when polling a
      flat directory (`None` or `.../plugins/profile` omits `run_path`).

  Returns:
    True once the server responds and at least one of `expected_runs` (if
    given) is present.
  """
  try:
    if expected_runs:
      runs_url = f"{url.rstrip('/')}/data/plugin/profile/runs"
      if (
          profile_logdir
          and pathlib.Path(profile_logdir).parts[-2:] != ("plugins", "profile")
      ):
        query = urllib.parse.urlencode({"run_path": profile_logdir})
        runs_url = f"{runs_url}?{query}"
      with urllib.request.urlopen(runs_url, timeout=0.5) as resp:
        if resp.status != http.HTTPStatus.OK:
          return False
        payload = resp.read()
        if (
            payload.startswith(b"\x1f\x8b")
            or resp.headers.get("Content-Encoding") == "gzip"
        ):
          payload = gzip.decompress(payload)
        runs = json.loads(payload.decode("utf-8"))
        return isinstance(runs, list) and bool(expected_runs.intersection(runs))
    with urllib.request.urlopen(url, timeout=0.5) as resp:
      return resp.status < http.HTTPStatus.INTERNAL_SERVER_ERROR
  except urllib.error.HTTPError as e:
    return (
        not expected_runs
        and e.code < http.HTTPStatus.INTERNAL_SERVER_ERROR
    )
  except (urllib.error.URLError, OSError, ValueError, zlib.error):
    return False


def resolve_artifacts_dir(
    env: Mapping[str, str] | None = None,
) -> str | None:
  """Resolves the output directory for Playwright failure diagnostic artifacts.

  Args:
    env: Environment mapping to inspect (`None` defaults to `os.environ`).

  Returns:
    Resolved directory path string, or `None` when no artifact env var is set.
  """
  environ = os.environ if env is None else env
  if out_dir := environ.get("XPROF_UI_ARTIFACTS_DIR"):
    return out_dir
  if undeclared_dir := environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
    return undeclared_dir
  if kokoro_dir := environ.get("KOKORO_ARTIFACTS_DIR"):
    return os.path.join(kokoro_dir, "ui_failure_artifacts")
  return None


def ensure_inference_serving_run(
    logdir: str, fallback_dir: pathlib.Path
) -> pathlib.Path:
  """Returns the staged `tpu-serving` run path or writes one in `fallback_dir`.

  Args:
    logdir: Primary profile logdir to check for an existing `tpu-serving` run.
    fallback_dir: Writable directory used if `tpu-serving` is absent.

  Returns:
    Path to the ready `tpu-serving` directory.
  """
  staged_dir = pathlib.Path(logdir) / "tpu-serving"
  if staged_dir.is_dir():
    return staged_dir
  serving_dir = fallback_dir / "tpu-serving"
  write_inference_serving_trace(serving_dir / "serving-node-1.xplane.pb")
  return serving_dir
