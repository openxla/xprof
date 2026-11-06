"""Shared Playwright UI interaction helpers for XProf frontend tests."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import importlib
import logging
import pathlib
import re
import shutil
from typing import Any
import urllib.parse

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
      stats {{ metadata_id: 3 uint64_value: 3257467400 }}
      stats {{ metadata_id: 4 uint64_value: 875000 }}
      stats {{ metadata_id: 5 str_value: "MXU 0 busy cycles" }}
      stats {{ metadata_id: 6 str_value: "MXU TC" }}
    }}
    events {{
      metadata_id: 103
      offset_ps: 1000
      duration_ps: 1000000000
      stats {{ metadata_id: 3 uint64_value: 3257467288 }}
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
  event_metadata {{ key: 102 value {{ id: 102 name: "mxu_busy_0" }} }}
  event_metadata {{ key: 103 value {{ id: 103 name: "matmul_lmr_bf16_mxu_0" }} }}
  event_metadata {{ key: 104 value {{ id: 104 name: "vector_alu_0" }} }}
  event_metadata {{ key: 105 value {{ id: 105 name: "scalar_alu_0" }} }}
  event_metadata {{ key: 106 value {{ id: 106 name: "xlu_busy_0" }} }}
  stat_metadata {{ key: 1 value {{ id: 1 name: "device_id" }} }}
  stat_metadata {{ key: 2 value {{ id: 2 name: "device_type_string" }} }}
  stat_metadata {{ key: 3 value {{ id: 3 name: "performance_counter_id" }} }}
  stat_metadata {{ key: 4 value {{ id: 4 name: "counter_value" }} }}
  stat_metadata {{ key: 5 value {{ id: 5 name: "performance_counter_description" }} }}
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
"""

_DCN_SLACK_FILE_DESCRIPTOR_TEXTPROTO = """
name: "dcn_slack_analysis.proto"
package: "tensorflow.profiler"
syntax: "proto3"
message_type {
  name: "DcnSlackSummary"
  field { name: "rendezvous" number: 1 label: LABEL_OPTIONAL type: TYPE_STRING }
  field { name: "slack_us" number: 2 label: LABEL_OPTIONAL type: TYPE_UINT64 }
  field {
    name: "occurrences"
    number: 3
    label: LABEL_OPTIONAL
    type: TYPE_UINT64
  }
  field {
    name: "bytes_transmitted_over_network"
    number: 4
    label: LABEL_OPTIONAL
    type: TYPE_UINT64
  }
  field {
    name: "stall_duration_us"
    number: 5
    label: LABEL_OPTIONAL
    type: TYPE_UINT64
  }
  field {
    name: "observed_duration_us"
    number: 6
    label: LABEL_OPTIONAL
    type: TYPE_UINT64
  }
  field {
    name: "recv_op_name"
    number: 7
    label: LABEL_OPTIONAL
    type: TYPE_STRING
  }
  field {
    name: "send_op_name"
    number: 8
    label: LABEL_OPTIONAL
    type: TYPE_STRING
  }
  field {
    name: "transfer_type"
    number: 13
    label: LABEL_OPTIONAL
    type: TYPE_STRING
  }
}
message_type {
  name: "DcnSlackAnalysis"
  field {
    name: "dcn_slack_summary"
    number: 2
    label: LABEL_REPEATED
    type: TYPE_MESSAGE
    type_name: ".tensorflow.profiler.DcnSlackSummary"
  }
}
"""

_DEFAULT_DCN_SLACK_TEXTPROTO = """
dcn_slack_summary {
  rendezvous: "all_reduce_dcn_0"
  slack_us: 1200
  occurrences: 4
  bytes_transmitted_over_network: 16777216
  stall_duration_us: 650
  observed_duration_us: 1850
  recv_op_name: "recv-done.1"
  send_op_name: "send.1"
  transfer_type: "ALL_REDUCE"
}
"""


def _build_xspace_parser() -> tuple[Any, Any]:
  """Builds a runtime protobuf parser for `tensorflow.profiler.XSpace`."""
  descriptor_pb2 = importlib.import_module("google.protobuf.descriptor_pb2")
  descriptor_pool = importlib.import_module("google.protobuf.descriptor_pool")
  message_factory = importlib.import_module("google.protobuf.message_factory")
  text_format = importlib.import_module("google.protobuf.text_format")
  pool = descriptor_pool.DescriptorPool()
  pool.Add(
      text_format.Parse(
          _XPLANE_FILE_DESCRIPTOR_TEXTPROTO,
          descriptor_pb2.FileDescriptorProto(),
      )
  )
  xspace_cls = message_factory.GetMessageClass(
      pool.FindMessageTypeByName("tensorflow.profiler.XSpace")
  )
  return xspace_cls, text_format


def _write_xspace_textproto(path: pathlib.Path, textproto: str) -> None:
  """Serializes `textproto` as an `XSpace` binary proto to `path`."""
  xspace_cls, text_format = _build_xspace_parser()
  msg = text_format.Parse(textproto, xspace_cls())
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(msg.SerializeToString())


def write_inference_serving_trace(path: pathlib.Path) -> None:
  """Writes a minimal TPU inference serving `.xplane.pb` trace to `path`."""
  _write_xspace_textproto(path, _INFERENCE_SERVING_XSPACE_TEXTPROTO)


def write_perf_utilization_trace(
    path: pathlib.Path, device_type: str = "TPU v6 Lite"
) -> None:
  """Writes a TPU performance-counter and utilization trace to `path`."""
  _write_xspace_textproto(
      path, _PERF_UTILIZATION_XSPACE_TEMPLATE.format(device_type=device_type)
  )


def write_empty_memory_trace(path: pathlib.Path) -> None:
  """Writes a minimal TPU trace with zero host memory allocators to `path`."""
  _write_xspace_textproto(path, _EMPTY_MEMORY_XSPACE_TEXTPROTO)


def _serialize_dcn_slack_analysis() -> bytes:
  """Serializes a non-empty `DcnSlackAnalysis` binary proto payload."""
  descriptor_pb2 = importlib.import_module("google.protobuf.descriptor_pb2")
  descriptor_pool = importlib.import_module("google.protobuf.descriptor_pool")
  message_factory = importlib.import_module("google.protobuf.message_factory")
  text_format = importlib.import_module("google.protobuf.text_format")
  pool = descriptor_pool.DescriptorPool()
  pool.Add(
      text_format.Parse(
          _DCN_SLACK_FILE_DESCRIPTOR_TEXTPROTO,
          descriptor_pb2.FileDescriptorProto(),
      )
  )
  dcn_cls = message_factory.GetMessageClass(
      pool.FindMessageTypeByName("tensorflow.profiler.DcnSlackAnalysis")
  )
  return text_format.Parse(
      _DEFAULT_DCN_SLACK_TEXTPROTO, dcn_cls()
  ).SerializeToString()


def write_megascale_trace(
    path: pathlib.Path, base_trace_path: pathlib.Path | None = None
) -> None:
  """Writes a trace containing `MegaScale:send_dcn` DCN events to `path`."""
  xspace_cls, text_format = _build_xspace_parser()
  if base_trace_path is not None and base_trace_path.is_file():
    mega_space = xspace_cls.FromString(base_trace_path.read_bytes())
  else:
    mega_space = text_format.Parse(
        _INFERENCE_SERVING_XSPACE_TEXTPROTO, xspace_cls()
    )
  injected = False
  for plane in mega_space.planes:
    if plane.name == "/host:CPU":
      em = plane.event_metadata[999999]
      em.id = 999999
      em.name = "MegaScale:send_dcn"
      line = plane.lines[0] if plane.lines else plane.lines.add(id=1, name="T")
      ev = line.events.add()
      ev.metadata_id = 999999
      ev.offset_ps = 1000
      ev.duration_ps = 50000
      injected = True
      break
  if not injected:
    plane = mega_space.planes.add(id=99, name="/host:CPU")
    em = plane.event_metadata[999999]
    em.id = 999999
    em.name = "MegaScale:send_dcn"
    line = plane.lines.add(id=1, name="MegaScale")
    ev = line.events.add()
    ev.metadata_id = 999999
    ev.offset_ps = 1000
    ev.duration_ps = 50000
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(mega_space.SerializeToString())
  dcn_bytes = _serialize_dcn_slack_analysis()
  host = path.name.removesuffix(".xplane.pb")
  (path.parent / f"{host}.dcn_collective_stats.pb").write_bytes(dcn_bytes)
  (path.parent / "ALL_HOSTS.dcn_collective_stats.pb").write_bytes(dcn_bytes)


def stage_synthesized_runs(
    staged_dir: pathlib.Path, repo_root: pathlib.Path | None = None
) -> None:
  """Populates multi-tool and edge-case runs inside a writable `staged_dir`."""
  gpu_candidates: list[pathlib.Path] = []
  if repo_root is not None:
    gpu_candidates.append(
        repo_root / "convert" / "test_xplanes" / "gpu_training_2.xplane.pb"
    )
  gpu_candidates.append(
      pathlib.Path("/tmp/convert_test_xplanes/gpu_training_2.xplane.pb")
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

  base_tpu_trace = next(
      iter(
          sorted(staged_dir.glob("tpu[-_]training/*.xplane.pb"))
          or sorted(staged_dir.glob("*/*.xplane.pb"))
      ),
      None,
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
    shutil.copy2(
        staged_dir
        / "megascale-2host"
        / "mega-node-0.dcn_collective_stats.pb",
        staged_dir
        / "megascale-2host"
        / "mega-node-1.dcn_collective_stats.pb",
    )
  if not (staged_dir / "empty-memory").exists():
    write_empty_memory_trace(
        staged_dir / "empty-memory" / "empty-mem-0.xplane.pb"
    )


def resolve_server_logdir(profile_logdir: str) -> str:
  """Returns the TensorBoard root logdir for a `.../plugins/profile` path."""
  path = pathlib.Path(profile_logdir)
  if path.parts[-2:] == ("plugins", "profile"):
    return str(path.parent.parent)
  return profile_logdir


def ensure_inference_serving_run(
    logdir: str, fallback_dir: pathlib.Path
) -> pathlib.Path:
  """Returns the staged `tpu-serving` run path or writes one in `fallback_dir`."""
  staged_dir = pathlib.Path(logdir) / "tpu-serving"
  if staged_dir.is_dir():
    return staged_dir
  serving_dir = fallback_dir / "tpu-serving"
  write_inference_serving_trace(serving_dir / "serving-node-1.xplane.pb")
  return serving_dir
