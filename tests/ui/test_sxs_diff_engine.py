"""Unit tests for the multi-modal Side-by-Side (SxS) A/B diff engine."""

import io
import json
import os
import pathlib
import tempfile
import unittest

# pylint: disable=g-import-not-at-top,g-importing-member
try:
  from PIL import Image

  _HAS_PIL = True
except ImportError:
  Image = None  # pyrefly: ignore[assignment]
  _HAS_PIL = False

try:
  from google3.third_party.xprof.tests.ui import sxs_diff_engine
  from google3.third_party.xprof.tests.ui.sxs_diff_engine import SxsDiffEngine
  from google3.third_party.xprof.tests.ui.sxs_report_generator import generate_sxs_html_report
except ImportError:
  try:
    from tests.ui import sxs_diff_engine  # pyrefly: ignore[missing-import]
    from tests.ui.sxs_diff_engine import SxsDiffEngine  # pyrefly: ignore[missing-import]
    from tests.ui.sxs_report_generator import generate_sxs_html_report  # pyrefly: ignore[missing-import]
  except ImportError:
    import sxs_diff_engine  # pyrefly: ignore[missing-import]
    from sxs_diff_engine import SxsDiffEngine  # pyrefly: ignore[missing-import]
    from sxs_report_generator import generate_sxs_html_report  # pyrefly: ignore[missing-import]


def _create_test_image(
    color: tuple[int, int, int], size: tuple[int, int] = (50, 50)
) -> bytes:
  """Creates in-memory PNG image bytes with solid color for visual diff tests."""
  assert Image is not None
  img = Image.new("RGB", size, color=color)
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  return buf.getvalue()


def _create_rgba_test_image(
    color: tuple[int, int, int, int], size: tuple[int, int] = (50, 50)
) -> bytes:
  """Creates in-memory RGBA PNG image bytes for visual diff tests."""
  assert Image is not None
  img = Image.new("RGBA", size, color=color)
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  return buf.getvalue()


class SxsDiffEngineTest(unittest.TestCase):
  """Tests for SxS Diff Engine and HTML report generation."""

  def test_pillow_dependency_strictly_available(self):
    """Asserts that Pillow is installed in the test execution environment."""
    self.assertTrue(
        _HAS_PIL and Image is not None,
        "PIL.Image must be installed in the test execution environment. "
        "Silent skipping in CI hides visual regressions.",
    )

  def test_visual_diff_identical_and_divergent(self):
    """Verifies visual diff measures pixel differences and flags size deltas."""
    engine = SxsDiffEngine()
    img_white = _create_test_image((255, 255, 255))
    img_black = _create_test_image((0, 0, 0))

    # Identical images
    diff_same = engine.compute_visual_diff(img_white, img_white)
    self.assertEqual(diff_same.diff_ratio, 0.0)
    self.assertEqual(diff_same.diff_pixels, 0)
    self.assertIsNotNone(diff_same.composite_png_bytes)
    self.assertIsNone(diff_same.dimension_mismatch)

    # Completely divergent images
    diff_different = engine.compute_visual_diff(img_white, img_black)
    self.assertEqual(diff_different.diff_ratio, 1.0)
    self.assertEqual(diff_different.diff_pixels, 2500)
    self.assertIsNotNone(diff_different.composite_png_bytes)

    # Dimension mismatch (no silent resize)
    img_tall = _create_test_image((255, 255, 255), size=(50, 80))
    diff_mismatch = engine.compute_visual_diff(img_white, img_tall)
    self.assertIsNotNone(diff_mismatch.dimension_mismatch)
    self.assertIn("50, 50", diff_mismatch.dimension_mismatch)

  def test_visual_diff_measures_divergence_strictly(self):
    """Verifies pixel diff calculation without conditional skip."""
    engine = sxs_diff_engine.SxsDiffEngine()
    img_white = _create_test_image((255, 255, 255), size=(50, 50))
    img_black = _create_test_image((0, 0, 0), size=(50, 50))

    diff = engine.compute_visual_diff(img_white, img_black)

    self.assertEqual(diff.diff_pixels, 2500)
    self.assertEqual(diff.total_pixels, 2500)
    self.assertEqual(diff.diff_ratio, 1.0)
    self.assertIsNotNone(diff.composite_png_bytes)
    self.assertIsNone(diff.dimension_mismatch)

  def test_visual_diff_alpha_channel_transparency(self):
    """Verifies alpha compositing prevents false divergence and detects alpha delta."""
    engine = SxsDiffEngine()
    img_trans_red = _create_rgba_test_image((255, 0, 0, 0))
    img_trans_black = _create_rgba_test_image((0, 0, 0, 0))

    # Transparent red vs transparent black (both invisible, no divergence)
    diff_trans = engine.compute_visual_diff(img_trans_red, img_trans_black)
    self.assertEqual(diff_trans.diff_pixels, 0)
    self.assertEqual(diff_trans.diff_ratio, 0.0)
    self.assertIsNone(diff_trans.dimension_mismatch)

    # Transparent red vs opaque red (element appearance/disappearance)
    img_opaque_red = _create_rgba_test_image((255, 0, 0, 255))
    diff_alpha_change = engine.compute_visual_diff(
        img_trans_red, img_opaque_red
    )
    self.assertEqual(diff_alpha_change.diff_pixels, 2500)
    self.assertEqual(diff_alpha_change.diff_ratio, 1.0)
    self.assertIsNone(diff_alpha_change.dimension_mismatch)

  def test_visual_diff_custom_background_color(self):
    """Verifies configurable background color for dark theme alpha compositing."""
    engine = SxsDiffEngine()
    img_semi_trans = _create_rgba_test_image((255, 0, 0, 128))
    dark_bg = (32, 33, 36, 255)
    diff = engine.compute_visual_diff(
        img_semi_trans, img_semi_trans, background_color=dark_bg
    )
    self.assertEqual(diff.diff_pixels, 0)
    self.assertEqual(diff.diff_ratio, 0.0)
    self.assertIsNone(diff.dimension_mismatch)

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

  def test_visual_diff_transposed_dimension_mismatch(self):
    """Verifies flipped aspect ratios with equal area flag all pixels as mismatched."""
    engine = SxsDiffEngine()
    img_portrait = _create_test_image((255, 255, 255), size=(100, 200))
    img_landscape = _create_test_image((255, 255, 255), size=(200, 100))

    diff = engine.compute_visual_diff(img_portrait, img_landscape)
    self.assertIsNotNone(diff.dimension_mismatch)
    self.assertIn("100, 200", diff.dimension_mismatch)
    self.assertIn("200, 100", diff.dimension_mismatch)
    self.assertEqual(diff.total_pixels, 20000)
    self.assertEqual(diff.diff_pixels, 20000)
    self.assertEqual(diff.diff_ratio, 1.0)

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


if __name__ == "__main__":
  unittest.main()
