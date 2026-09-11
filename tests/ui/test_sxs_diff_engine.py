"""Unit tests for the multi-modal Side-by-Side (SxS) A/B diff engine."""

import io
import json
import os
import pathlib
import tempfile
import unittest

from PIL import Image

# pylint: disable=g-import-not-at-top,g-importing-member
try:
  from google3.third_party.xprof.tests.ui.sxs_diff_engine import SxsDiffEngine
  from google3.third_party.xprof.tests.ui.sxs_report_generator import generate_sxs_html_report
except ImportError:
  try:
    from tests.ui.sxs_diff_engine import SxsDiffEngine  # pyrefly: ignore[missing-import]
    from tests.ui.sxs_report_generator import generate_sxs_html_report  # pyrefly: ignore[missing-import]
  except ImportError:
    from sxs_diff_engine import SxsDiffEngine  # pyrefly: ignore[missing-import]
    from sxs_report_generator import generate_sxs_html_report  # pyrefly: ignore[missing-import]


def _create_test_image(
    color: tuple[int, int, int] | tuple[int, int, int, int],
    size: tuple[int, int] = (50, 50),
) -> bytes:
  """Creates in-memory PNG bytes of a solid color, RGBA when alpha is given."""
  img = Image.new("RGBA" if len(color) == 4 else "RGB", size, color=color)
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  return buf.getvalue()


class SxsDiffEngineTest(unittest.TestCase):
  """Tests for SxS Diff Engine and HTML report generation."""

  def test_visual_diff_identical_and_divergent(self):
    """Verifies visual diff measures pixel differences and flags size deltas."""
    engine = SxsDiffEngine()
    img_white = _create_test_image((255, 255, 255))
    img_black = _create_test_image((0, 0, 0))

    # Identical images
    diff_same = engine.compute_visual_diff(img_white, img_white)
    self.assertEqual(diff_same.diff_ratio, 0.0)
    self.assertEqual(diff_same.diff_pixels, 0)
    self.assertIsNotNone(diff_same.heatmap_png_bytes)
    self.assertIsNone(diff_same.dimension_mismatch)

    # Completely divergent images
    diff_different = engine.compute_visual_diff(img_white, img_black)
    self.assertEqual(diff_different.diff_ratio, 1.0)
    self.assertEqual(diff_different.diff_pixels, 2500)
    self.assertEqual(diff_different.total_pixels, 2500)
    self.assertIsNotNone(diff_different.heatmap_png_bytes)
    self.assertIsNone(diff_different.dimension_mismatch)

    # The heatmap is the candidate alone. The card already shows the baseline
    # and the candidate, so a wider strip would inline them a second time.
    heatmap = Image.open(io.BytesIO(diff_different.heatmap_png_bytes))
    self.assertEqual(heatmap.size, (50, 50))

    # Dimension mismatch (no silent resize)
    img_tall = _create_test_image((255, 255, 255), size=(50, 80))
    diff_mismatch = engine.compute_visual_diff(img_white, img_tall)
    self.assertIsNotNone(diff_mismatch.dimension_mismatch)
    self.assertIn("50, 50", diff_mismatch.dimension_mismatch)

  def test_visual_diff_alpha_channel_transparency(self):
    """Verifies alpha compositing prevents false divergence and detects alpha delta."""
    engine = SxsDiffEngine()
    img_trans_red = _create_test_image((255, 0, 0, 0))
    img_trans_black = _create_test_image((0, 0, 0, 0))

    # Transparent red vs transparent black (both invisible, no divergence)
    diff_trans = engine.compute_visual_diff(img_trans_red, img_trans_black)
    self.assertEqual(diff_trans.diff_pixels, 0)
    self.assertEqual(diff_trans.diff_ratio, 0.0)
    self.assertIsNone(diff_trans.dimension_mismatch)

    # Transparent red vs opaque red (element appearance/disappearance)
    img_opaque_red = _create_test_image((255, 0, 0, 255))
    diff_alpha_change = engine.compute_visual_diff(
        img_trans_red, img_opaque_red
    )
    self.assertEqual(diff_alpha_change.diff_pixels, 2500)
    self.assertEqual(diff_alpha_change.diff_ratio, 1.0)
    self.assertIsNone(diff_alpha_change.dimension_mismatch)

  def test_visual_diff_custom_background_color(self):
    """Verifies transparent pixels flatten onto the caller's background color."""
    engine = SxsDiffEngine()
    dark_bg = (32, 33, 36, 255)
    img_transparent = _create_test_image((255, 0, 0, 0))
    img_opaque_dark = _create_test_image(dark_bg)

    diff_on_dark = engine.compute_visual_diff(
        img_transparent, img_opaque_dark, background_color=dark_bg
    )
    diff_on_default_white = engine.compute_visual_diff(
        img_transparent, img_opaque_dark
    )

    # Unpainted pixels adopt the background, so the transparent capture matches
    # a dark-theme render only when the dark canvas is the one supplied.
    self.assertEqual(diff_on_dark.diff_pixels, 0)
    self.assertEqual(diff_on_dark.diff_ratio, 0.0)
    self.assertEqual(diff_on_default_white.diff_pixels, 2500)
    self.assertEqual(diff_on_default_white.diff_ratio, 1.0)

  def test_visual_diff_noise_floor_is_per_channel(self):
    """Verifies a blue-only delta is not attenuated by luminance weighting."""
    engine = SxsDiffEngine()
    img_black = _create_test_image((0, 0, 0))
    # 80/255 is a third of the channel range and far above the noise floor,
    # but luminance weights blue at 0.114, which scores it a 9.
    img_blue_shifted = _create_test_image((0, 0, 80))

    diff = engine.compute_visual_diff(img_black, img_blue_shifted)

    self.assertEqual(diff.diff_pixels, 2500)
    self.assertEqual(diff.diff_ratio, 1.0)

  def test_visual_diff_ignores_sub_threshold_noise(self):
    """Verifies deltas at or below the noise floor stay unreported."""
    engine = SxsDiffEngine()
    img_grey = _create_test_image((128, 128, 128))
    img_hinting_noise = _create_test_image((138, 138, 138))

    diff = engine.compute_visual_diff(img_grey, img_hinting_noise)

    self.assertEqual(diff.diff_pixels, 0)
    self.assertEqual(diff.diff_ratio, 0.0)

  def test_visual_diff_corrupt_and_empty_bytes_handled_gracefully(self):
    """Verifies corrupt, truncated, or empty bytes return error status without crash."""
    engine = SxsDiffEngine()
    valid_img = _create_test_image((255, 255, 255), size=(50, 50))

    # Empty image bytes
    diff_empty = engine.compute_visual_diff(b"", b"")
    self.assertIsNotNone(diff_empty.dimension_mismatch)
    self.assertEqual(diff_empty.diff_ratio, 1.0)

    # Corrupted non-image bytes vs valid image
    diff_corrupt = engine.compute_visual_diff(
        b"not_a_valid_png_payload", valid_img
    )
    self.assertIsNotNone(diff_corrupt.dimension_mismatch)
    self.assertEqual(diff_corrupt.diff_ratio, 1.0)

    # Truncated PNG header bytes vs valid image
    diff_truncated = engine.compute_visual_diff(
        b"\x89PNG\r\n\x1a\n\x00\x00", valid_img
    )
    self.assertIsNotNone(diff_truncated.dimension_mismatch)
    self.assertEqual(diff_truncated.diff_ratio, 1.0)

  def test_visual_diff_geometry_mismatch_reports_measured_divergence(self):
    """Verifies a geometry change is measured, not reported as every pixel."""
    engine = SxsDiffEngine()
    img_portrait = _create_test_image((255, 0, 0), size=(100, 200))
    img_landscape = _create_test_image((255, 0, 0), size=(200, 100))

    diff = engine.compute_visual_diff(img_portrait, img_landscape)

    self.assertIsNotNone(diff.dimension_mismatch)
    self.assertIn("100, 200", diff.dimension_mismatch)
    self.assertIn("200, 100", diff.dimension_mismatch)
    # Both are composited onto the 200x200 union over a white background. They
    # agree on the 100x100 corner they both cover and on the 100x100 corner
    # neither covers, and disagree on the other two quadrants.
    self.assertEqual(diff.total_pixels, 40000)
    self.assertEqual(diff.diff_pixels, 20000)
    self.assertEqual(diff.diff_ratio, 0.5)

  def test_dom_diff_structural_delta(self):
    """Verifies unified diff generation between DOM snapshots."""
    engine = SxsDiffEngine()
    dom_a = '<html>\n<body>\n<div id="header">Stable</div>\n</body>\n</html>'
    dom_b = (
        '<html>\n<body>\n<div id="header">Modified</div>\n<span>New'
        " Element</span>\n</body>\n</html>"
    )

    diff = engine.compute_dom_diff(dom_a, dom_b)
    self.assertTrue(diff.has_changes)
    self.assertGreater(diff.added_lines, 0)
    self.assertGreater(diff.deleted_lines, 0)
    self.assertIn('+<div id="header">Modified</div>', diff.unified_diff)

  def test_sanitize_dom_strips_angular_and_cdk_dynamic_ids(self):
    """Verifies DOM sanitizer strips dynamic Angular and Material IDs."""
    engine = SxsDiffEngine()
    dirty_dom = (
        '<div _ngcontent-c12="" _nghost-c14="" id="mat-tab-label-0-1"'
        ' id="mat-select-4" id="cdk-describedby-message-12"'
        ' id="mat-mdc-select-5"'
        ' aria-controls="mat-tab-content-0-1" aria-owns="mat-select-4-panel">'
        "<span>Stable Content</span></div>"
    )
    clean_dom = engine.sanitize_dom(dirty_dom)
    self.assertNotIn("_ngcontent", clean_dom)
    self.assertNotIn("_nghost", clean_dom)
    self.assertNotIn("mat-tab-label", clean_dom)
    self.assertNotIn("mat-select-4", clean_dom)
    self.assertNotIn("mat-mdc-select-5", clean_dom)
    self.assertNotIn("cdk-describedby", clean_dom)
    self.assertNotIn("aria-controls", clean_dom)
    self.assertNotIn("aria-owns", clean_dom)
    self.assertIn("Stable Content", clean_dom)

  def test_compute_network_diff_normalizes_query_params(self):
    """Verifies query parameters are normalized order-independently."""
    engine = SxsDiffEngine()
    reqs_a: list[dict[str, object]] = [
        {"method": "GET", "url": "/data?run=foo&tag=overview", "status": 200}
    ]
    reqs_b: list[dict[str, object]] = [
        {"method": "GET", "url": "/data?tag=overview&run=foo", "status": 200}
    ]
    diff = engine.compute_network_diff(reqs_a, reqs_b)
    self.assertFalse(diff.has_changes)
    self.assertEqual(diff.status_mismatches, [])

  def test_compute_network_diff_flags_unparseable_status(self):
    """Verifies a non-numeric status never compares equal to a real HTTP 200."""
    engine = SxsDiffEngine()
    reqs_a: list[dict[str, object]] = [
        {"method": "GET", "url": "/data", "status": 200}
    ]
    reqs_b: list[dict[str, object]] = [
        {"method": "GET", "url": "/data", "status": "ERR_CONNECTION_RESET"}
    ]

    diff = engine.compute_network_diff(reqs_a, reqs_b)

    self.assertTrue(diff.has_changes)
    self.assertIn("ERR_CONNECTION_RESET", "\n".join(diff.status_mismatches))

  def test_evaluate_waypoint_and_manifest_approval(self):
    """Verifies waypoint evaluation, content hashing, and verdict calculation."""
    engine = SxsDiffEngine()
    img_a = _create_test_image((200, 200, 200))
    img_b = _create_test_image((200, 200, 200))
    dom = "<html><body><div>Hello</div></body></html>"

    # Identical waypoint -> SAME
    waypoint_diff = engine.evaluate_waypoint(
        journey_name="journey_1",
        waypoint_name="waypoint_1",
        img_a=img_a,
        img_b=img_b,
        html_a=dom,
        html_b=dom,
        requests_a=[{"method": "GET", "url": "/data", "status": 200}],
        requests_b=[{"method": "GET", "url": "/data", "status": 200}],
    )
    self.assertEqual(waypoint_diff.visual.diff_pixels, 0)
    self.assertFalse(waypoint_diff.dom.has_changes)
    self.assertFalse(waypoint_diff.network.has_changes)
    self.assertIsNotNone(waypoint_diff.diff_hash)
    self.assertEqual(waypoint_diff.verdict, "SAME")

    # Network status mismatch (200 -> 500) -> CHANGED
    waypoint_net_diff = engine.evaluate_waypoint(
        journey_name="journey_1",
        waypoint_name="waypoint_1",
        img_a=img_a,
        img_b=img_b,
        html_a=dom,
        html_b=dom,
        requests_a=[{"method": "GET", "url": "/data", "status": 200}],
        requests_b=[{"method": "GET", "url": "/data", "status": 500}],
    )
    self.assertTrue(waypoint_net_diff.network.has_changes)
    self.assertEqual(waypoint_net_diff.verdict, "CHANGED")

    # Out-of-order network requests -> SAME
    waypoint_reordered_net = engine.evaluate_waypoint(
        journey_name="journey_1",
        waypoint_name="waypoint_1",
        img_a=img_a,
        img_b=img_b,
        html_a=dom,
        html_b=dom,
        requests_a=[
            {"method": "GET", "url": "/endpoint_1", "status": "200"},
            {"method": "GET", "url": "/endpoint_2", "status": 200},
        ],
        requests_b=[
            {"method": "GET", "url": "/endpoint_2", "status": 200},
            {"method": "GET", "url": "/endpoint_1", "status": 200},
        ],
    )
    self.assertFalse(waypoint_reordered_net.network.has_changes)
    self.assertEqual(waypoint_reordered_net.verdict, "SAME")

  def test_diff_hash_covers_dom_changes_past_the_display_cap(self):
    """Verifies the approval hash binds DOM lines the report never shows."""
    engine = SxsDiffEngine()
    img = _create_test_image((200, 200, 200))
    baseline = "\n".join(f"<div>base {i}</div>" for i in range(200))
    candidate = "\n".join(f"<div>cand {i}</div>" for i in range(200))
    late_regression = candidate.replace(
        "<div>cand 199</div>", "<div>regressed</div>"
    )

    def evaluate(html_b: str):
      return engine.evaluate_waypoint(
          journey_name="triage",
          waypoint_name="overview",
          img_a=img,
          img_b=img,
          html_a=baseline,
          html_b=html_b,
          requests_a=[],
          requests_b=[],
      )

    clean = evaluate(candidate)
    regressed = evaluate(late_regression)

    # The rendered diff is capped at 100 lines, so both waypoints display the
    # same delta. An approval recorded for one must not carry over to the
    # other regardless.
    self.assertEqual(clean.dom.unified_diff, regressed.dom.unified_diff)
    self.assertNotEqual(clean.diff_hash, regressed.diff_hash)

  def test_diff_hash_distinguishes_geometry_regressions(self):
    """Verifies equal-area geometry regressions get distinct approval hashes."""
    engine = SxsDiffEngine()
    baseline = _create_test_image((255, 0, 0), size=(50, 50))
    # Each candidate adds 30x50 of new area against the baseline, so the
    # measured pixel counts are equal and the extents are the only thing that
    # can tell a page that grew taller from one that grew wider.
    taller = _create_test_image((255, 0, 0), size=(50, 80))
    wider = _create_test_image((255, 0, 0), size=(80, 50))

    def evaluate(candidate: bytes):
      return engine.evaluate_waypoint(
          journey_name="triage",
          waypoint_name="overview",
          img_a=baseline,
          img_b=candidate,
          html_a="<div>Stable</div>",
          html_b="<div>Stable</div>",
          requests_a=[],
          requests_b=[],
      )

    grew_taller = evaluate(taller)
    grew_wider = evaluate(wider)

    self.assertEqual(
        grew_taller.visual.diff_pixels, grew_wider.visual.diff_pixels
    )
    self.assertNotEqual(grew_taller.diff_hash, grew_wider.diff_hash)

  def test_manifest_file_loading_and_verdict(self):
    """Verifies SxsDiffEngine loads and enforces approved_manifest.json."""
    img_a = _create_test_image((100, 100, 100))
    img_b = _create_test_image((200, 200, 200))
    dom_a = "<div>Before</div>"
    dom_b = "<div>After</div>"

    unapproved_engine = SxsDiffEngine()
    diff = unapproved_engine.evaluate_waypoint(
        journey_name="test_j",
        waypoint_name="test_w",
        img_a=img_a,
        img_b=img_b,
        html_a=dom_a,
        html_b=dom_b,
        requests_a=[],
        requests_b=[],
    )
    self.assertEqual(diff.verdict, "CHANGED")
    self.assertFalse(diff.is_approved)

    with tempfile.TemporaryDirectory() as tmpdir:
      manifest_path = os.path.join(tmpdir, "approved_manifest.json")
      manifest_data = {
          "approved_diffs": {
              "test_j:test_w": {
                  "diff_hash": diff.diff_hash,
                  "rationale": "Intentional UI redesign",
              }
          }
      }
      pathlib.Path(manifest_path).write_text(
          json.dumps(manifest_data), encoding="utf-8"
      )

      approved_engine = SxsDiffEngine(approved_manifest_path=manifest_path)
      approved_diff = approved_engine.evaluate_waypoint(
          journey_name="test_j",
          waypoint_name="test_w",
          img_a=img_a,
          img_b=img_b,
          html_a=dom_a,
          html_b=dom_b,
          requests_a=[],
          requests_b=[],
      )
      self.assertEqual(approved_diff.verdict, "APPROVED")
      self.assertTrue(approved_diff.is_approved)
      self.assertEqual(
          approved_diff.approval_rationale, "Intentional UI redesign"
      )

  def test_sxs_report_generation(self):
    """Verifies self-contained HTML certification report generation."""
    engine = SxsDiffEngine()
    img_a = _create_test_image((128, 128, 128))
    img_b = _create_test_image((255, 0, 0))
    dom_a = "<div>Benchmark Original</div>"
    dom_b = "<div>Benchmark Candidate</div>"

    diff_same = engine.evaluate_waypoint(
        journey_name="triage",
        waypoint_name="overview",
        img_a=img_a,
        img_b=img_a,
        html_a=dom_a,
        html_b=dom_a,
        requests_a=[],
        requests_b=[],
    )
    diff_changed = engine.evaluate_waypoint(
        journey_name="triage",
        waypoint_name="hlo_stats",
        img_a=img_a,
        img_b=img_b,
        html_a=dom_a,
        html_b=dom_b,
        requests_a=[{"method": "GET", "url": "/hlo", "status": 200}],
        requests_b=[{"method": "GET", "url": "/hlo", "status": 500}],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      out_path = os.path.join(tmpdir, "report.html")
      report_path = generate_sxs_html_report(
          [diff_same, diff_changed], out_path
      )
      self.assertEqual(report_path, out_path)
      content = pathlib.Path(report_path).read_text(encoding="utf-8")
      self.assertIn("<!DOCTYPE html>", content)
      self.assertIn(
          "OpenXLA XProf A/B User Journey Certification Report", content
      )
      self.assertIn("triage — overview", content)
      self.assertIn("triage — hlo_stats", content)
      self.assertIn("PASS (Identical)", content)
      self.assertIn("DIFF DETECTED", content)
      self.assertIn("Reviewer Action Required", content)
      self.assertIn("data:image/png;base64,", content)

  def test_sxs_report_namespaces_element_ids_per_waypoint(self):
    """Verifies each waypoint card gets a distinct element id namespace."""
    engine = SxsDiffEngine()
    img_a = _create_test_image((128, 128, 128))
    img_b = _create_test_image((255, 0, 0))
    diffs = [
        engine.evaluate_waypoint(
            journey_name="triage",
            waypoint_name=name,
            img_a=img_a,
            img_b=img_b,
            html_a="<div>Baseline</div>",
            html_b="<div>Candidate</div>",
            requests_a=[],
            requests_b=[],
        )
        for name in ("overview", "hlo_stats")
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
      out_path = os.path.join(tmpdir, "report.html")
      content = pathlib.Path(
          generate_sxs_html_report(diffs, out_path)
      ).read_text(encoding="utf-8")

    # Both cards render the slider controls, so a shared id namespace would
    # make the second card's handle drag the first card's images.
    self.assertIn('id="slider-handle-wp_0"', content)
    self.assertIn('id="slider-handle-wp_1"', content)
    self.assertIn("setVisualMode('wp_0', 'slider')", content)
    self.assertIn("setVisualMode('wp_1', 'slider')", content)

  def test_sxs_report_renders_tab_per_waypoint(self):
    """Verifies the tab strip lists every waypoint and marks the first active."""
    engine = SxsDiffEngine()
    img = _create_test_image((128, 128, 128))
    dom = "<div>Stable</div>"
    same = engine.evaluate_waypoint(
        journey_name="triage",
        waypoint_name="overview",
        img_a=img,
        img_b=img,
        html_a=dom,
        html_b=dom,
        requests_a=[],
        requests_b=[],
    )
    changed = engine.evaluate_waypoint(
        journey_name="triage",
        waypoint_name="hlo_stats",
        img_a=img,
        img_b=_create_test_image((255, 0, 0)),
        html_a=dom,
        html_b="<div>Shifted</div>",
        requests_a=[],
        requests_b=[],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      out_path = os.path.join(tmpdir, "report.html")
      content = pathlib.Path(
          generate_sxs_html_report([same, changed], out_path)
      ).read_text(encoding="utf-8")

    self.assertIn(
        '<div class="tab active" onclick="switchWaypoint(0)">', content
    )
    self.assertIn('<div class="tab" onclick="switchWaypoint(1)">', content)
    self.assertIn("tab-status-pass", content)
    self.assertIn("tab-status-diff", content)
    self.assertIn("triage: hlo_stats (CHANGED)", content)

  def test_sxs_report_renders_dimension_mismatch_banner(self):
    """Verifies the layout-shift banner renders alongside the diff heatmap."""
    engine = SxsDiffEngine()
    diff = engine.evaluate_waypoint(
        journey_name="triage",
        waypoint_name="overview",
        img_a=_create_test_image((255, 255, 255), size=(50, 50)),
        img_b=_create_test_image((255, 255, 255), size=(50, 80)),
        html_a="<div>Stable</div>",
        html_b="<div>Stable</div>",
        requests_a=[],
        requests_b=[],
    )
    self.assertIsNotNone(diff.visual.dimension_mismatch)
    self.assertIsNotNone(diff.visual.heatmap_png_bytes)

    with tempfile.TemporaryDirectory() as tmpdir:
      out_path = os.path.join(tmpdir, "report.html")
      content = pathlib.Path(
          generate_sxs_html_report([diff], out_path)
      ).read_text(encoding="utf-8")

    # The banner and the heatmap are complementary, not mutually exclusive:
    # the banner names the geometry delta, the heatmap shows where it landed.
    self.assertIn("Viewport Layout Shift Detected", content)
    self.assertIn("(50, 50) vs (50, 80)", content)
    self.assertIn("Visual Divergence", content)

  def test_sxs_report_omits_screenshots_for_identical_waypoints(self):
    """Verifies a byte-identical waypoint inlines no screenshots."""
    engine = SxsDiffEngine()
    img = _create_test_image((128, 128, 128))
    dom = "<div>Stable</div>"
    same = engine.evaluate_waypoint(
        journey_name="triage",
        waypoint_name="overview",
        img_a=img,
        img_b=img,
        html_a=dom,
        html_b=dom,
        requests_a=[],
        requests_b=[],
    )
    changed = engine.evaluate_waypoint(
        journey_name="triage",
        waypoint_name="hlo_stats",
        img_a=img,
        img_b=_create_test_image((255, 0, 0)),
        html_a=dom,
        html_b=dom,
        requests_a=[],
        requests_b=[],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      same_only = pathlib.Path(
          generate_sxs_html_report([same], os.path.join(tmpdir, "same.html"))
      ).read_text(encoding="utf-8")
      with_change = pathlib.Path(
          generate_sxs_html_report(
              [same, changed], os.path.join(tmpdir, "both.html")
          )
      ).read_text(encoding="utf-8")

    # The engine builds a heatmap unconditionally, so gating on its presence
    # inlined five screenshots per waypoint even where the two walks agreed.
    self.assertNotIn("data:image/png;base64,", same_only)
    self.assertIn("PASS (Identical)", same_only)
    self.assertIn("data:image/png;base64,", with_change)

  def test_sxs_report_survives_missing_section_templates(self):
    """Verifies an unreadable section template cannot destroy the report."""
    engine = SxsDiffEngine()
    diff = engine.evaluate_waypoint(
        journey_name="triage",
        waypoint_name="overview",
        img_a=_create_test_image((128, 128, 128)),
        img_b=_create_test_image((255, 0, 0)),
        html_a="<div>Baseline</div>",
        html_b="<div>Candidate</div>",
        requests_a=[],
        requests_b=[],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      templates = pathlib.Path(tmpdir) / "templates"
      templates.mkdir()
      # Only the outer shell exists. Every per-section template is absent,
      # which previously raised out of generation and lost the verdict for a
      # run that had already failed.
      (templates / "sxs_report_template.html").write_text(
          "<html>$styles|$summary_text|$badge_class|$badge_text|$tabs_html|"
          "$cards_html|$approval_portal_html|$unapproved_json</html>",
          encoding="utf-8",
      )
      (templates / "report_styles.css").write_text("", encoding="utf-8")
      content = pathlib.Path(
          generate_sxs_html_report(
              [diff],
              os.path.join(tmpdir, "report.html"),
              template_dir=templates,
          )
      ).read_text(encoding="utf-8")

    self.assertIn("DIFF DETECTED", content)
    self.assertIn(diff.diff_hash, content)
    self.assertIn("Section unavailable", content)

  def test_sxs_report_approval_portal_makes_no_unearned_claims(self):
    """Verifies the portal neither certifies the run nor emits a whole manifest."""
    engine = SxsDiffEngine()
    diff = engine.evaluate_waypoint(
        journey_name="triage",
        waypoint_name="overview",
        img_a=_create_test_image((128, 128, 128)),
        img_b=_create_test_image((255, 0, 0)),
        html_a="<div>Baseline</div>",
        html_b="<div>Candidate</div>",
        requests_a=[],
        requests_b=[],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      content = pathlib.Path(
          generate_sxs_html_report([diff], os.path.join(tmpdir, "report.html"))
      ).read_text(encoding="utf-8")

    # Approving in the portal writes nothing and submits nothing, so the page
    # must not repaint its own badge green, and the text it emits must not be
    # a whole manifest that overwrites approvals recorded by other reviewers.
    self.assertIn('class="badge badge-fail"', content)
    self.assertNotIn("ALL JOURNEYS CERTIFIED", content)
    self.assertNotIn("approved_diffs: {}", content)


if __name__ == "__main__":
  unittest.main()
