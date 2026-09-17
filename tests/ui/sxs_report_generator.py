# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generates standalone visual HTML certification reports for A/B diff runs."""

import base64
import html
import json
import os
import pathlib
import shutil
import string

# pylint: disable=g-import-not-at-top
try:
  from google3.third_party.xprof.tests.ui import sxs_diff_engine
except ModuleNotFoundError as err:
  if not (err.name or "").startswith("google3"):
    raise
  try:
    from tests.ui import sxs_diff_engine  # pyrefly: ignore[missing-import]
  except ModuleNotFoundError as sub_err:
    if not (sub_err.name or "").startswith("tests"):
      raise
    import sxs_diff_engine  # pyrefly: ignore[missing-import]

_NETWORK_ITEM_HTML = '<li class="network-item network-item-mismatch">{}</li>'


def _get_default_template_dir() -> pathlib.Path:
  """Resolves template directory across local and test environments."""
  local_dir = pathlib.Path(__file__).resolve().parent / "templates"
  if (local_dir / "sxs_report_template.html").is_file():
    return local_dir

  runfiles_dir = (
      pathlib.Path(os.environ.get("TEST_SRCDIR", ""))
      / os.environ.get("TEST_WORKSPACE", "")
      / "third_party/xprof/tests/ui/templates"
  )
  if (runfiles_dir / "sxs_report_template.html").is_file():
    return runfiles_dir

  return local_dir


class _TemplateRegistry:
  """Loads and caches template files for HTML report generation."""

  def __init__(self, template_dir: pathlib.Path):
    self._dir = template_dir
    self._cache: dict[str, str] = {}

  def get(self, filename: str) -> str:
    """Retrieves raw content of a template file."""
    if filename not in self._cache:
      path = self._dir / filename
      with open(path, "r", encoding="utf-8") as f:
        self._cache[filename] = f.read()
    return self._cache[filename]

  def get_section(self, filename: str) -> str:
    """Retrieves a section template, or a placeholder if it cannot be read.

    Report generation runs after the verdict has been decided and printed. An
    unreadable section template used to raise out of here and take the whole
    report with it, so a run that had already failed lost the evidence for why.
    Losing one section is recoverable; losing the report is not.

    Args:
      filename: Template file name relative to the template directory.

    Returns:
      The template text, or a self-describing placeholder in its place.
    """
    try:
      return self.get(filename)
    except OSError as err:
      escaped_err = html.escape(str(err)).replace("$", "$$")
      return (
          '<div class="diff-section">Section unavailable'
          f" ({escaped_err})</div>"
      )


def _render_dom_diff(
    templates: _TemplateRegistry, dom: sxs_diff_engine.DomDiff
) -> str:
  """Renders a formatted DOM diff section using template."""
  diff_lines: list[str] = []
  for line in dom.unified_diff.splitlines():
    escaped_line = html.escape(line)
    if line.startswith("+") and not line.startswith("+++"):
      diff_lines.append(f'<span class="diff-added">{escaped_line}</span>')
    elif line.startswith("-") and not line.startswith("---"):
      diff_lines.append(f'<span class="diff-deleted">{escaped_line}</span>')
    else:
      diff_lines.append(escaped_line)

  tmpl = string.Template(templates.get_section("dom_diff_section.html"))
  return tmpl.safe_substitute(
      added_lines=str(dom.added_lines),
      deleted_lines=str(dom.deleted_lines),
      diff_html="\n".join(diff_lines),
  )


def _data_uri_payload(png_bytes: bytes | None) -> str:
  """Encodes PNG bytes for an inline data: URI, or "" when there are none."""
  return base64.b64encode(png_bytes).decode("ascii") if png_bytes else ""


def _render_visual_diff(
    templates: _TemplateRegistry,
    visual: sxs_diff_engine.VisualDiff,
    card_id: str,
) -> str:
  """Renders the side-by-side, swipe slider, and heatmap views of a waypoint."""
  tmpl = string.Template(templates.get_section("visual_diff_section.html"))
  return tmpl.safe_substitute(
      diff_pixels=str(visual.diff_pixels),
      base_b64=_data_uri_payload(visual.base_png_bytes),
      cand_b64=_data_uri_payload(visual.candidate_png_bytes),
      heatmap_b64=_data_uri_payload(visual.heatmap_png_bytes),
      card_id=card_id,
  )


def _render_journey_tabs(
    templates: _TemplateRegistry,
    waypoint_diffs: list[sxs_diff_engine.WaypointDiff],
) -> str:
  """Renders the navigation tab strip listing every evaluated waypoint."""
  tmpl = string.Template(templates.get_section("journey_tab.html"))
  tabs = []
  for index, diff in enumerate(waypoint_diffs):
    if diff.verdict == "SAME":
      status_class = "tab-status-pass"
    elif diff.verdict == "APPROVED":
      status_class = "tab-status-approved"
    else:
      status_class = "tab-status-diff"
    tab_title = (
        f"{html.escape(diff.journey_name)}:"
        f" {html.escape(diff.waypoint_name)}"
        + ("" if diff.verdict == "SAME" else f" ({diff.verdict})")
    )
    tabs.append(
        tmpl.safe_substitute(
            active_class=" active" if index == 0 else "",
            index=str(index),
            status_class=status_class,
            tab_title=tab_title,
        )
    )
  return "\n".join(tabs)


def _render_waypoint_card(
    templates: _TemplateRegistry,
    waypoint: sxs_diff_engine.WaypointDiff,
    card_index: int,
) -> str:
  """Renders a single waypoint comparison card."""
  card_id = f"wp_{card_index}"

  if waypoint.verdict == "SAME":
    status_html = '<span class="status-pass">PASS (Identical)</span>'
  elif waypoint.verdict == "APPROVED":
    rationale = html.escape(waypoint.approval_rationale or "Approved")
    status_html = f'<span class="status-approved">APPROVED ({rationale})</span>'
  else:
    status_html = '<span class="status-diff">DIFF DETECTED</span>'

  sections: list[str] = []

  if waypoint.dom.unified_diff:
    sections.append(_render_dom_diff(templates, waypoint.dom))

  # Gated on a measured delta, not on the heatmap existing. The engine builds a
  # heatmap for every waypoint, including byte-identical ones, so this inlined
  # five base64 PNGs (baseline and candidate twice each, plus the heatmap) for
  # waypoints with nothing to show. Across 46 waypoints that was most of a
  # 28 MB report.
  if waypoint.visual.diff_pixels or waypoint.visual.dimension_mismatch:
    sections.append(_render_visual_diff(templates, waypoint.visual, card_id))

  if waypoint.visual.dimension_mismatch:
    mismatch_tmpl = string.Template(
        templates.get_section("dimension_mismatch_section.html")
    )
    sections.append(
        mismatch_tmpl.safe_substitute(
            mismatch_text=html.escape(waypoint.visual.dimension_mismatch)
        )
    )

  if waypoint.network.status_mismatches:
    net_tmpl = string.Template(
        templates.get_section("network_diff_section.html")
    )
    sections.append(
        net_tmpl.safe_substitute(
            network_items="\n".join(
                _NETWORK_ITEM_HTML.format(html.escape(mismatch))
                for mismatch in waypoint.network.status_mismatches
            )
        )
    )

  card_tmpl = string.Template(templates.get_section("waypoint_card.html"))
  return card_tmpl.safe_substitute(
      journey_name=html.escape(waypoint.journey_name),
      waypoint_name=html.escape(waypoint.waypoint_name),
      status_html=status_html,
      sections_html=("\n" + "\n".join(sections)) if sections else "",
  )


def generate_sxs_html_report(
    waypoint_diffs: list[sxs_diff_engine.WaypointDiff],
    output_html_path: str,
    template_dir: pathlib.Path | None = None,
) -> str:
  """Renders and writes standalone HTML diff report."""
  dir_path = template_dir or _get_default_template_dir()
  templates = _TemplateRegistry(dir_path)

  has_unapproved_diffs = any(w.verdict == "CHANGED" for w in waypoint_diffs)
  has_approved_diffs = any(w.verdict == "APPROVED" for w in waypoint_diffs)

  if has_unapproved_diffs:
    badge_class = "badge-fail"
    badge_text = "DIFF DETECTED"
    summary_text = (
        f"{len(waypoint_diffs)} Waypoints Evaluated (Reviewer Action Required)"
    )
  elif has_approved_diffs:
    badge_class = "badge-approved"
    badge_text = "DIFFS APPROVED"
    summary_text = (
        f"{len(waypoint_diffs)} Waypoints Evaluated (All Diffs Approved)"
    )
  else:
    badge_class = "badge-pass"
    badge_text = "ALL WAYPOINTS IDENTICAL"
    summary_text = "100% Identical to Baseline"

  tabs_html = _render_journey_tabs(templates, waypoint_diffs)
  cards_html = "\n".join(
      _render_waypoint_card(templates, w, card_index=idx)
      for idx, w in enumerate(waypoint_diffs)
  )
  approval_portal_html = (
      templates.get_section("approval_portal.html")
      if has_unapproved_diffs
      else ""
  )

  unapproved_list = [
      {
          "key": f"{w.journey_name}:{w.waypoint_name}",
          "diff_hash": w.diff_hash,
      }
      for w in waypoint_diffs
      if w.verdict == "CHANGED"
  ]
  unapproved_json = (
      json.dumps(unapproved_list, indent=4)
      .replace("<", "\\u003c")
      .replace(">", "\\u003e")
      .replace("&", "\\u0026")
  )
  styles = templates.get("report_styles.css")

  main_tmpl = string.Template(templates.get("sxs_report_template.html"))
  full_html = main_tmpl.safe_substitute(
      styles=styles,
      summary_text=summary_text,
      badge_class=badge_class,
      badge_text=badge_text,
      tabs_html=tabs_html,
      cards_html=cards_html,
      approval_portal_html=approval_portal_html,
      unapproved_json=unapproved_json,
  )

  os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
  with open(output_html_path, "w", encoding="utf-8") as f:
    f.write(full_html)

  return output_html_path


def publish_report_artifact(
    report_path: str,
    outputs_dir: str | None,
    artifact_name: str = "sxs_report.html",
) -> str | None:
  """Copies a generated report into the Bazel undeclared outputs directory.

  The test runner packages everything under TEST_UNDECLARED_OUTPUTS_DIR into
  outputs.zip, allowing reviewers to inspect report artifacts without a local
  checkout.

  Args:
    report_path: Path to the generated HTML report.
    outputs_dir: Value of TEST_UNDECLARED_OUTPUTS_DIR, or None when the caller
      is not running under the Bazel test runner.
    artifact_name: Destination filename inside outputs_dir.

  Returns:
    The destination path, or None when no outputs directory is available.
  """
  if not outputs_dir or not os.path.isdir(outputs_dir):
    return None
  destination = os.path.join(outputs_dir, artifact_name)
  if os.path.abspath(report_path) != os.path.abspath(destination):
    shutil.copyfile(report_path, destination)
  return destination
