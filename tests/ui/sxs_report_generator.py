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
import string

# pylint: disable=g-import-not-at-top
try:
  from google3.third_party.xprof.tests.ui import sxs_diff_engine
except ImportError:
  try:
    from tests.ui import sxs_diff_engine  # pyrefly: ignore[missing-import]
  except ImportError:
    import sxs_diff_engine  # pyrefly: ignore[missing-import]


def _get_default_template_dir() -> pathlib.Path:
  """Resolves the report template directory across local and test environments."""
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


def _render_dom_diff(
    templates: _TemplateRegistry,
    unified_diff: str,
    added_lines: int,
    deleted_lines: int,
) -> str:
  """Renders a formatted DOM diff section using template."""
  diff_lines: list[str] = []
  for line in unified_diff.splitlines():
    escaped_line = html.escape(line)
    if line.startswith("+") and not line.startswith("+++"):
      diff_lines.append(f'<span class="diff-added">{escaped_line}</span>')
    elif line.startswith("-") and not line.startswith("---"):
      diff_lines.append(f'<span class="diff-deleted">{escaped_line}</span>')
    else:
      diff_lines.append(escaped_line)

  diff_html = "\n".join(diff_lines)
  tmpl = string.Template(templates.get("dom_diff_section.html"))
  return tmpl.substitute(
      added_lines=str(added_lines),
      deleted_lines=str(deleted_lines),
      diff_html=diff_html,
  )


def _render_visual_diff(
    templates: _TemplateRegistry,
    diff_pixels: int,
    base_png: bytes | None,
    candidate_png: bytes | None,
    composite_png: bytes | None,
) -> str:
  """Renders visual side-by-side composite diffs using template."""
  base_b64 = base64.b64encode(base_png).decode("ascii") if base_png else ""
  cand_b64 = (
      base64.b64encode(candidate_png).decode("ascii") if candidate_png else ""
  )
  comp_b64 = (
      base64.b64encode(composite_png).decode("ascii") if composite_png else ""
  )

  tmpl = string.Template(templates.get("visual_diff_section.html"))
  return tmpl.substitute(
      diff_pixels=str(diff_pixels),
      base_b64=base_b64,
      cand_b64=cand_b64,
      comp_b64=comp_b64,
  )


def _render_waypoint_card(
    templates: _TemplateRegistry, w: sxs_diff_engine.WaypointDiff
) -> str:
  """Renders a single waypoint comparison card."""
  journey_name = html.escape(w.journey_name)
  waypoint_name = html.escape(w.waypoint_name)

  if w.verdict == "SAME":
    status_html = '<span class="status-pass">PASS (Identical)</span>'
  elif w.verdict == "APPROVED":
    rationale = html.escape(w.approval_rationale or "Approved")
    status_html = f'<span class="status-approved">APPROVED ({rationale})</span>'
  else:
    status_html = '<span class="status-diff">DIFF DETECTED</span>'

  sections: list[str] = []

  if w.dom.unified_diff:
    sections.append(
        _render_dom_diff(
            templates,
            w.dom.unified_diff,
            w.dom.added_lines,
            w.dom.deleted_lines,
        )
    )

  if w.visual.composite_png_bytes:
    sections.append(
        _render_visual_diff(
            templates,
            w.visual.diff_pixels,
            w.visual.base_png_bytes,
            w.visual.candidate_png_bytes,
            w.visual.composite_png_bytes,
        )
    )
  elif w.visual.dimension_mismatch:
    mismatch_tmpl = string.Template(
        templates.get("dimension_mismatch_section.html")
    )
    sections.append(
        mismatch_tmpl.substitute(
            mismatch_text=html.escape(w.visual.dimension_mismatch)
        )
    )

  if w.network.status_mismatches:
    mismatch_items = "\n".join(
        '    <li class="network-item'
        f' network-item-mismatch">{html.escape(m)}</li>'
        for m in w.network.status_mismatches
    )
    net_tmpl = string.Template(templates.get("network_diff_section.html"))
    sections.append(net_tmpl.substitute(network_items=mismatch_items))

  sections_html = "\n" + "\n".join(sections) if sections else ""
  card_tmpl = string.Template(templates.get("waypoint_card.html"))
  return card_tmpl.substitute(
      journey_name=journey_name,
      waypoint_name=waypoint_name,
      status_html=status_html,
      sections_html=sections_html,
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

  badge_class = "badge-fail" if has_unapproved_diffs else "badge-pass"
  badge_text = (
      "DIFF DETECTED" if has_unapproved_diffs else "ALL WAYPOINTS IDENTICAL"
  )
  summary_text = (
      f"{len(waypoint_diffs)} Waypoints Evaluated (Reviewer Action Required)"
      if has_unapproved_diffs
      else "100% Identical to Baseline"
  )

  cards_html = "\n".join(
      _render_waypoint_card(templates, w) for w in waypoint_diffs
  )
  approval_portal_html = (
      templates.get("approval_portal.html") if has_unapproved_diffs else ""
  )

  unapproved_list = [
      {
          "key": f"{w.journey_name}:{w.waypoint_name}",
          "diff_hash": w.diff_hash,
      }
      for w in waypoint_diffs
      if w.verdict == "CHANGED"
  ]
  unapproved_json = json.dumps(unapproved_list, indent=4)
  styles = templates.get("report_styles.css")

  main_tmpl = string.Template(templates.get("sxs_report_template.html"))
  full_html = main_tmpl.substitute(
      styles=styles,
      summary_text=summary_text,
      badge_class=badge_class,
      badge_text=badge_text,
      cards_html=cards_html,
      approval_portal_html=approval_portal_html,
      unapproved_json=unapproved_json,
  )

  os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
  with open(output_html_path, "w", encoding="utf-8") as f:
    f.write(full_html)

  return output_html_path
