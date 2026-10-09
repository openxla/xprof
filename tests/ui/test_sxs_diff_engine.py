"""Unit tests for the multi-modal Side-by-Side (SxS) A/B diff engine."""

import contextlib
import dataclasses
import enum
import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from PIL import Image

# pylint: disable=g-import-not-at-top
try:
  from google3.third_party.xprof.tests.ui import sxs_diff_engine
  from google3.third_party.xprof.tests.ui import sxs_report_generator
except ModuleNotFoundError as err:
  if not (err.name or "").startswith("google3"):
    raise
  try:
    from tests.ui import sxs_diff_engine  # pyrefly: ignore[missing-import]
    from tests.ui import sxs_report_generator  # pyrefly: ignore[missing-import]
  except ModuleNotFoundError as err2:
    if not (err2.name or "").startswith("tests"):
      raise
    import sxs_diff_engine  # pyrefly: ignore[missing-import]
    import sxs_report_generator  # pyrefly: ignore[missing-import]

approve_waypoint = sxs_diff_engine.approve_waypoint
get_default_manifest_path = sxs_diff_engine.get_default_manifest_path
sxs_main = sxs_diff_engine.main
DomDiff = sxs_diff_engine.DomDiff
NetworkDiff = sxs_diff_engine.NetworkDiff
SxsDiffEngine = sxs_diff_engine.SxsDiffEngine
VisualDiff = sxs_diff_engine.VisualDiff
WaypointDiff = sxs_diff_engine.WaypointDiff
generate_sxs_html_report = sxs_report_generator.generate_sxs_html_report
publish_report_artifact = sxs_report_generator.publish_report_artifact


def _create_test_image(
    color: tuple[int, int, int] | tuple[int, int, int, int],
    size: tuple[int, int] = (50, 50),
) -> bytes:
  """Creates in-memory PNG bytes of a solid color, RGBA when alpha is given."""
  img = Image.new("RGBA" if len(color) == 4 else "RGB", size, color=color)
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  return buf.getvalue()


def _evaluate(
    engine: SxsDiffEngine,
    img_a: bytes,
    img_b: bytes | None = None,
    *,
    html_a: str = "<div>Same</div>",
    html_b: str | None = None,
    requests_a: list[dict[str, object]] | None = None,
    requests_b: list[dict[str, object]] | None = None,
    journey: str = "triage",
    waypoint: str = "overview",
) -> WaypointDiff:
  """Evaluates a waypoint whose B side defaults to A and requests to none."""
  return engine.evaluate_waypoint(
      journey_name=journey,
      waypoint_name=waypoint,
      img_a=img_a,
      img_b=img_a if img_b is None else img_b,
      html_a=html_a,
      html_b=html_a if html_b is None else html_b,
      requests_a=[] if requests_a is None else requests_a,
      requests_b=[] if requests_b is None else requests_b,
  )


def _render_report(
    diffs: list[WaypointDiff],
    template_dir: pathlib.Path | None = None,
) -> str:
  """Renders the report into a temporary file and returns its HTML."""
  with tempfile.TemporaryDirectory() as tmpdir:
    path = generate_sxs_html_report(
        diffs,
        os.path.join(tmpdir, "report.html"),
        template_dir=template_dir,
    )
    return pathlib.Path(path).read_text(encoding="utf-8")


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
    """Verifies alpha compositing prevents false divergence."""
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
    """Verifies transparent pixels flatten onto background color."""
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

  def test_visual_diff_heatmap_paints_red_overlay_on_diverging_pixels(self):
    """Verifies visual diff heatmap paints red overlay on diverging pixels."""
    engine = SxsDiffEngine()
    img_white = _create_test_image((255, 255, 255, 255))
    img_black = _create_test_image((0, 0, 0, 255))

    diff = engine.compute_visual_diff(img_white, img_black)
    self.assertIsNotNone(diff.heatmap_png_bytes)

    heatmap_img = Image.open(io.BytesIO(diff.heatmap_png_bytes)).convert("RGBA")
    r, g, b, _ = heatmap_img.getpixel((0, 0))
    self.assertGreater(r, 200)
    self.assertLess(g, 100)
    self.assertLess(b, 100)

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
    """Verifies corrupt or empty bytes return error status without crash."""
    engine = SxsDiffEngine()
    valid_img = _create_test_image((255, 255, 255), size=(50, 50))

    # Both empty image bytes agree on no screenshot (fast path)
    diff_both_empty = engine.compute_visual_diff(b"", b"")
    self.assertIsNone(diff_both_empty.dimension_mismatch)
    self.assertEqual(diff_both_empty.diff_ratio, 0.0)

    # One empty image vs valid image returns error status
    diff_one_empty = engine.compute_visual_diff(b"", valid_img)
    self.assertIsNotNone(diff_one_empty.dimension_mismatch)
    self.assertEqual(diff_one_empty.diff_ratio, 1.0)

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
    self.assertIn(
        '\n-<div id="header">Stable</div>\n+<div id="header">Modified</div>\n',
        diff.unified_diff,
    )

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

  def test_sanitize_dom_renumbers_google_charts_renderer_ids(self):
    """Verifies renderer ids normalize to appearance order, refs intact."""
    engine = SxsDiffEngine()
    # One chart's two clip paths, numbered off a counter that is global to the
    # page load: drawn early on one walk and after other charts on the next.
    # The early walk deliberately reuses 0, the number renumbering assigns
    # first, so a rewrite that rescans its own output would collapse the two
    # ids onto each other.
    early = (
        '<svg><defs><clipPath id="_ABSTRACT_RENDERER_ID_5"></clipPath>'
        '<clipPath id="_ABSTRACT_RENDERER_ID_0"></clipPath></defs>'
        '<g clip-path="url(#_ABSTRACT_RENDERER_ID_0)"></g>'
        '<g clip-path="url(#_ABSTRACT_RENDERER_ID_5)"></g></svg>'
    )
    late = (
        '<svg><defs><clipPath id="_ABSTRACT_RENDERER_ID_16"></clipPath>'
        '<clipPath id="_ABSTRACT_RENDERER_ID_17"></clipPath></defs>'
        '<g clip-path="url(#_ABSTRACT_RENDERER_ID_17)"></g>'
        '<g clip-path="url(#_ABSTRACT_RENDERER_ID_16)"></g></svg>'
    )

    normalized = engine.sanitize_dom(early)

    self.assertEqual(normalized, engine.sanitize_dom(late))
    self.assertFalse(engine.compute_dom_diff(early, late).has_changes)
    self.assertIn('<clipPath id="_ABSTRACT_RENDERER_ID_0">', normalized)
    self.assertIn('<clipPath id="_ABSTRACT_RENDERER_ID_1">', normalized)
    self.assertIn('clip-path="url(#_ABSTRACT_RENDERER_ID_0)"', normalized)
    self.assertIn('clip-path="url(#_ABSTRACT_RENDERER_ID_1)"', normalized)

  def test_sanitize_dom_keeps_distinct_renderer_ids_distinct(self):
    """Verifies renumbering never merges two charts' ids into one."""
    engine = SxsDiffEngine()
    two_charts = (
        '<svg id="_ABSTRACT_RENDERER_ID_9"></svg>'
        '<svg id="_ABSTRACT_RENDERER_ID_4"></svg>'
    )

    normalized = engine.sanitize_dom(two_charts)

    self.assertEqual(
        normalized,
        '<svg id="_ABSTRACT_RENDERER_ID_0"></svg>'
        '<svg id="_ABSTRACT_RENDERER_ID_1"></svg>',
    )

  def test_sanitize_dom_strips_styles_and_ng_attributes(self):
    """Verifies sanitizer removes styles, stylesheets, and ng attributes."""
    engine = SxsDiffEngine()
    dom = (
        '<head><style type="text/css">.foo { color: red; }</style>'
        '<link rel="stylesheet" href="styles.css"></head>'
        "<body ng-version='17.2.0'>"
        "<div class='mat-form-field-animations-enabled' ng-reflect-name='test'"
        " ng-transition='active'>"
        "Visible Content</div>"
        "<svg><filter id='blur'>"
        "<feGaussianBlur stdDeviation='2'/></filter></svg>"
        "</body>"
    )
    clean = engine.sanitize_dom(dom)
    self.assertNotIn("color: red", clean)
    self.assertNotIn("styles.css", clean)
    self.assertNotIn("ng-version", clean)
    self.assertNotIn("ng-reflect", clean)
    self.assertNotIn("ng-transition", clean)
    self.assertNotIn("mat-form-field-animations-enabled", clean)
    self.assertNotIn("<filter", clean)
    self.assertIn("Visible Content", clean)

  def test_sanitize_dom_purges_hidden_containers_and_a11y_tables(self):
    """Verifies nested hidden containers and chart a11y tables are purged."""
    engine = SxsDiffEngine()
    dom = (
        "<div>Before</div>\n"
        "<div ng-transition='active' style='display: none'"
        " aria-hidden='true'>\n"
        "<div><span>Deeply nested hidden</span></div>\n"
        "</div>\n"
        "<div style='display: none' aria-hidden='true' />\n"
        "<div><svg><rect></rect></svg>"
        '<div aria-label="A tabular representation of the data in the chart."'
        ' style="position: absolute; left: -10000px;">'
        "<table><tr><td><div>Nested a11y</div></td></tr></table></div></div>\n"
        "<div>After</div>"
    )
    clean = engine.sanitize_dom(dom)
    self.assertNotIn("Deeply nested hidden", clean)
    self.assertNotIn("display: none", clean)
    self.assertNotIn("tabular representation", clean)
    self.assertNotIn("Nested a11y", clean)
    self.assertNotIn("</table>", clean)
    self.assertIn("<div>Before</div>", clean)
    self.assertIn("<div>After</div>", clean)
    self.assertIn("<svg><rect></rect></svg>", clean)

  def test_sanitize_dom_keeps_partially_hidden_divs(self):
    """Verifies one hiding signal alone keeps a div, but cdk overlays go."""
    engine = SxsDiffEngine()
    clean = engine.sanitize_dom(
        '<div aria-hidden="true">aria-only</div>'
        '<div style="position: absolute">chart-wrapper</div>'
        '<div class="cdk-overlay-container">overlay</div>'
    )
    self.assertIn("aria-only", clean)
    self.assertIn("chart-wrapper", clean)
    self.assertNotIn("overlay", clean)

  def test_compute_dom_diff_separates_changed_final_line(self):
    """Verifies a changed final DOM line renders as separate -/+ lines."""
    engine = SxsDiffEngine()
    diff = engine.compute_dom_diff(
        "<p>head</p>\n<div>A</div>", "<p>head</p>\n<div>B</div>"
    )
    self.assertIn("-<div>A</div>\n+<div>B</div>\n", diff.unified_diff)

  def test_sanitize_dom_filter_regex_does_not_cross_filter_tags(self):
    """Verifies <filter> regex removes blur filters without crossing tags."""
    engine = SxsDiffEngine()
    dom = (
        "<svg><filter id='f1'><feOffset dx='1' dy='1'/></filter>"
        "<filter id='f2'><feGaussianBlur stdDeviation='2'/></filter></svg>"
    )
    cleaned = engine.sanitize_dom(dom)
    self.assertIn("<filter id='f1'><feOffset dx='1' dy='1'/></filter>", cleaned)
    self.assertNotIn("feGaussianBlur", cleaned)
    self.assertNotIn("f2", cleaned)

  def test_sanitize_dom_quote_aware_tag_attributes(self):
    """Verifies element removal handles tag attributes with > in quotes."""
    engine = SxsDiffEngine()
    dom = (
        '<div data-condition="count > 5" style="display: none"'
        ' aria-hidden="true"><span>Hidden Item</span></div>'
        '<div data-condition="count > 5"><span>Visible Item</span></div>'
    )
    cleaned = engine.sanitize_dom(dom)
    self.assertNotIn("Hidden Item", cleaned)
    self.assertIn("Visible Item", cleaned)
    self.assertIn('data-condition="count > 5"', cleaned)

  def test_sanitize_dom_preserves_unmatched_closing_tag_with_matching_attrs(
      self,
  ):
    """Verifies unmatched closing tags with filter attrs are not stripped."""
    engine = SxsDiffEngine()
    dom = (
        '</script src="https://www.gstatic.com/charts/loader.js">'
        "<div>keep me</div>"
    )
    cleaned = engine.sanitize_dom(dom)
    self.assertIn("<div>keep me</div>", cleaned)

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

    # A blank value is still part of the request, so it is not dropped.
    diff_blank = engine.compute_network_diff(
        [{"method": "GET", "url": "/data?host=&run=foo", "status": 200}],
        [{"method": "GET", "url": "/data?run=foo", "status": 200}],
    )
    self.assertTrue(diff_blank.has_changes)

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
    """Verifies waypoint evaluation and verdict calculation."""
    engine = SxsDiffEngine()
    img_a = _create_test_image((200, 200, 200))
    img_b = _create_test_image((200, 200, 200))
    dom = "<html><body><div>Hello</div></body></html>"

    # Identical waypoint -> SAME
    waypoint_diff = _evaluate(
        engine,
        img_a,
        img_b,
        html_a=dom,
        requests_a=[{"method": "GET", "url": "/data", "status": 200}],
        requests_b=[{"method": "GET", "url": "/data", "status": 200}],
        journey="journey_1",
        waypoint="waypoint_1",
    )
    self.assertEqual(waypoint_diff.visual.diff_pixels, 0)
    self.assertFalse(waypoint_diff.dom.has_changes)
    self.assertFalse(waypoint_diff.network.has_changes)
    self.assertIsNotNone(waypoint_diff.diff_hash)
    self.assertEqual(waypoint_diff.verdict, "SAME")

    # Network status mismatch (200 -> 500) -> CHANGED
    waypoint_net_diff = _evaluate(
        engine,
        img_a,
        img_b,
        html_a=dom,
        requests_a=[{"method": "GET", "url": "/data", "status": 200}],
        requests_b=[{"method": "GET", "url": "/data", "status": 500}],
        journey="journey_1",
        waypoint="waypoint_1",
    )
    self.assertTrue(waypoint_net_diff.network.has_changes)
    self.assertEqual(waypoint_net_diff.verdict, "CHANGED")

    # Out-of-order network requests -> SAME
    waypoint_reordered_net = _evaluate(
        engine,
        img_a,
        img_b,
        html_a=dom,
        requests_a=[
            {"method": "GET", "url": "/endpoint_1", "status": "200"},
            {"method": "GET", "url": "/endpoint_2", "status": 200},
        ],
        requests_b=[
            {"method": "GET", "url": "/endpoint_2", "status": 200},
            {"method": "GET", "url": "/endpoint_1", "status": 200},
        ],
        journey="journey_1",
        waypoint="waypoint_1",
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
      return _evaluate(engine, img, html_a=baseline, html_b=html_b)

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
      return _evaluate(engine, baseline, candidate, html_a="<div>Stable</div>")

    grew_taller = evaluate(taller)
    grew_wider = evaluate(wider)

    self.assertEqual(
        grew_taller.visual.diff_pixels, grew_wider.visual.diff_pixels
    )
    self.assertNotEqual(grew_taller.diff_hash, grew_wider.diff_hash)

    # Even when added area matches background color (diff_pixels == 0),
    # dimension_mismatch must still hash to distinct digests.
    base_white = _create_test_image((255, 255, 255), size=(50, 50))
    taller_white = _create_test_image((255, 255, 255), size=(50, 80))
    wider_white = _create_test_image((255, 255, 255), size=(80, 50))
    w_taller = _evaluate(engine, base_white, taller_white, html_a="<p>S</p>")
    w_wider = _evaluate(engine, base_white, wider_white, html_a="<p>S</p>")
    self.assertEqual(w_taller.visual.diff_pixels, 0)
    self.assertEqual(w_wider.visual.diff_pixels, 0)
    self.assertNotEqual(w_taller.diff_hash, w_wider.diff_hash)

  def test_manifest_file_loading_and_verdict(self):
    """Verifies SxsDiffEngine loads and enforces approved_manifest.json."""
    img_a = _create_test_image((100, 100, 100))
    img_b = _create_test_image((200, 200, 200))
    dom_a = "<div>Before</div>"
    dom_b = "<div>After</div>"

    unapproved_engine = SxsDiffEngine()
    diff = _evaluate(
        unapproved_engine,
        img_a,
        img_b,
        html_a=dom_a,
        html_b=dom_b,
        journey="test_j",
        waypoint="test_w",
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
      approved_diff = _evaluate(
          approved_engine,
          img_a,
          img_b,
          html_a=dom_a,
          html_b=dom_b,
          journey="test_j",
          waypoint="test_w",
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
    dom_a = "<div>Benchmark Original</div>\n<footer></footer>"
    dom_b = "<div>Benchmark Candidate</div>\n<footer></footer>"

    diff_same = _evaluate(engine, img_a, html_a=dom_a)
    diff_changed = _evaluate(
        engine,
        img_a,
        img_b,
        html_a=dom_a,
        html_b=dom_b,
        requests_a=[{"method": "GET", "url": "/hlo", "status": 200}],
        requests_b=[{"method": "GET", "url": "/hlo", "status": 500}],
        waypoint="hlo_stats",
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
      self.assertIn("DOM Structure Divergence (+1 / -1)", content)
      self.assertIn(
          '<span class="diff-added">+&lt;div&gt;Benchmark Candidate'
          "&lt;/div&gt;</span>",
          content,
      )
      self.assertIn(
          '<span class="diff-deleted">-&lt;div&gt;Benchmark Original'
          "&lt;/div&gt;</span>",
          content,
      )
      # The +++/--- file headers are context, not changes.
      self.assertNotIn('<span class="diff-added">+++', content)
      self.assertNotIn('<span class="diff-deleted">---', content)
      self.assertIn(
          '<li class="network-item network-item-mismatch">Endpoint divergence'
          " &#x27;GET /hlo&#x27; (Status 500)",
          content,
      )

  def test_sxs_report_namespaces_element_ids_per_waypoint(self):
    """Verifies each waypoint card gets a distinct element id namespace."""
    engine = SxsDiffEngine()
    img_a = _create_test_image((128, 128, 128))
    img_b = _create_test_image((255, 0, 0))
    diffs = [
        _evaluate(
            engine,
            img_a,
            img_b,
            html_a="<div>Baseline</div>",
            html_b="<div>Candidate</div>",
            waypoint=name,
        )
        for name in ("overview", "hlo_stats")
    ]

    content = _render_report(diffs)

    # Both cards render the slider controls, so a shared id namespace would
    # make the second card's handle drag the first card's images.
    self.assertIn('id="slider-handle-wp_0"', content)
    self.assertIn('id="slider-handle-wp_1"', content)
    self.assertIn("setVisualMode('wp_0', 'slider')", content)
    self.assertIn("setVisualMode('wp_1', 'slider')", content)

  def test_sxs_report_renders_tab_per_waypoint(self):
    """Verifies the tab strip lists waypoints and marks first active."""
    engine = SxsDiffEngine()
    img = _create_test_image((128, 128, 128))
    dom = "<div>Stable</div>"
    same = _evaluate(engine, img, html_a=dom)
    changed = _evaluate(
        engine,
        img,
        _create_test_image((255, 0, 0)),
        html_a=dom,
        html_b="<div>Shifted</div>",
        waypoint="hlo_stats",
    )

    content = _render_report([same, changed])

    self.assertIn(
        '<div class="tab active" onclick="switchWaypoint(0)">', content
    )
    self.assertIn('<div class="tab" onclick="switchWaypoint(1)">', content)
    # The class names alone also appear in the inlined CSS.
    self.assertIn(
        '<div class="tab-status-pass"></div> triage: overview\n', content
    )
    self.assertIn(
        '<div class="tab-status-diff"></div> triage: hlo_stats (CHANGED)\n',
        content,
    )

  def test_sxs_report_renders_dimension_mismatch_banner(self):
    """Verifies the layout-shift banner renders alongside the diff heatmap."""
    engine = SxsDiffEngine()
    diff = _evaluate(
        engine,
        _create_test_image((255, 255, 255), size=(50, 50)),
        _create_test_image((255, 255, 255), size=(50, 80)),
        html_a="<div>Stable</div>",
    )
    self.assertIsNotNone(diff.visual.dimension_mismatch)
    self.assertIsNotNone(diff.visual.heatmap_png_bytes)

    content = _render_report([diff])

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
    same = _evaluate(engine, img, html_a=dom)
    changed = _evaluate(
        engine,
        img,
        _create_test_image((255, 0, 0)),
        html_a=dom,
        waypoint="hlo_stats",
    )

    same_only = _render_report([same])
    with_change = _render_report([same, changed])

    # The engine builds a heatmap unconditionally, so gating on its presence
    # inlined five screenshots per waypoint even where the two walks agreed.
    self.assertNotIn("data:image/png;base64,", same_only)
    self.assertIn("PASS (Identical)", same_only)
    self.assertIn("data:image/png;base64,", with_change)

  def test_sxs_report_shows_rendered_pages_without_pixel_delta(self):
    """Verifies a DOM/network-only divergence still shows both rendered pages."""
    engine = SxsDiffEngine()
    img = _create_test_image((128, 128, 128))
    diff = _evaluate(
        engine,
        img,
        html_a="<div>Baseline</div>",
        html_b="<div>Candidate</div>",
        requests_b=[{"method": "GET", "url": "/data", "status": 200}],
    )
    self.assertEqual(diff.verdict, "CHANGED")
    self.assertLessEqual(
        diff.visual.diff_ratio, sxs_diff_engine.MAX_VISUAL_DIFF_RATIO
    )

    content = _render_report([diff])

    # A reviewer asked to approve this waypoint has to be able to see it, so
    # both pages are rendered even though no pixel moved beyond the budget.
    # The heading says why the pixel-delta views are absent.
    self.assertIn(
        "Rendered Pages (DOM changed; network changed; pixels within noise"
        " budget)",
        content,
    )
    self.assertIn("Baseline Golden", content)
    self.assertIn("Candidate Run", content)
    self.assertEqual(content.count("data:image/png;base64,"), 2)
    # The pixel-delta views stay gated: no swipe slider and no heatmap. The
    # report shell builds the same ID prefixes in JavaScript, so this checks
    # for the rendered elements rather than the prefixes.
    self.assertIn('id="context-sec-wp_0"', content)
    self.assertNotIn('id="view-slider-wp_0"', content)
    self.assertNotIn('id="view-heatmap-wp_0"', content)

  def test_sxs_report_survives_missing_section_templates(self):
    """Verifies an unreadable section template cannot destroy the report."""
    engine = SxsDiffEngine()
    diff = _evaluate(
        engine,
        _create_test_image((128, 128, 128)),
        _create_test_image((255, 0, 0)),
        html_a="<div>Baseline</div>",
        html_b="<div>Candidate</div>",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      templates = pathlib.Path(tmpdir) / "templates"
      templates.mkdir()
      (templates / "sxs_report_template.html").write_text(
          "<html>$styles|$summary_text|$badge_class|$badge_text|$tabs_html|"
          "$cards_html|$approval_portal_html|$unapproved_json</html>",
          encoding="utf-8",
      )
      (templates / "report_styles.css").write_text("", encoding="utf-8")
      content = _render_report([diff], template_dir=templates)

    self.assertIn("DIFF DETECTED", content)
    self.assertIn(diff.diff_hash, content)
    self.assertIn("Section unavailable", content)

  def test_sxs_report_approval_portal_makes_no_unearned_claims(self):
    """Verifies the portal does not certify the run it reports on."""
    engine = SxsDiffEngine()
    diff = _evaluate(
        engine,
        _create_test_image((128, 128, 128)),
        _create_test_image((255, 0, 0)),
        html_a="<div>Baseline</div>",
        html_b="<div>Candidate</div>",
    )

    content = _render_report([diff])

    # Approving in the portal writes nothing and submits nothing, so the page
    # must not repaint its own badge green.
    self.assertIn('class="badge badge-fail"', content)
    self.assertNotIn("ALL JOURNEYS CERTIFIED", content)

  def test_sxs_report_builds_on_existing_approvals(self):
    """Verifies the report embeds the approvals the gate read."""
    diff = _evaluate(
        SxsDiffEngine(approved_manifest_path=""),
        _create_test_image((128, 128, 128)),
        _create_test_image((255, 0, 0)),
    )
    existing = {
        "diff_hash": "0123456789abcdef",
        "decision": "INTENTIONAL",
        "rationale": "allow </script>&",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
      manifest = os.path.join(tmpdir, "approved_manifest.json")
      pathlib.Path(manifest).write_text(
          json.dumps({"approved_diffs": {"triage:hlo_stats": existing}}),
          encoding="utf-8",
      )
      path = generate_sxs_html_report(
          [diff], os.path.join(tmpdir, "report.html"), manifest_path=manifest
      )
      content = pathlib.Path(path).read_text(encoding="utf-8")

    # The page's Approve button merges into this object, so replacing
    # approved_manifest.json with its output keeps the earlier approval.
    embedded = content.split("const currentApprovedDiffs = ", 1)[1]
    embedded = embedded.split(";\n", 1)[0]
    self.assertNotIn("</script>&", embedded)
    self.assertEqual(json.loads(embedded), {"triage:hlo_stats": existing})

  def test_sxs_report_renders_pixel_delta_views(self):
    """Verifies a real pixel delta earns the slider and heatmap views."""
    engine = SxsDiffEngine()
    diff = _evaluate(
        engine,
        _create_test_image((128, 128, 128)),
        _create_test_image((255, 0, 0)),
    )
    self.assertGreater(
        diff.visual.diff_ratio, sxs_diff_engine.MAX_VISUAL_DIFF_RATIO
    )

    content = _render_report([diff])

    # The DOM-only test asserts these same IDs are absent. Pinning them here
    # keeps that assertion honest: if the ID scheme is ever renamed, this
    # fails rather than letting the absence check pass against nothing.
    self.assertIn('id="view-slider-wp_0"', content)
    self.assertIn('id="view-heatmap-wp_0"', content)
    self.assertNotIn('id="context-sec-wp_0"', content)

  def test_sxs_report_side_labels_do_not_overlap_screenshots(self):
    """Verifies side-label headers sit above screenshots without overlapping."""
    engine = SxsDiffEngine()
    diff = _evaluate(
        engine,
        _create_test_image((128, 128, 128)),
        _create_test_image((255, 0, 0)),
    )
    content = _render_report([diff])

    side_label_rule = content.split(".side-label {", 1)[1].split("}", 1)[0]
    self.assertNotIn("position: absolute", side_label_rule)
    self.assertIn("display: flex", side_label_rule)

    self.assertIn('class="slider-header"', content)
    self.assertIn('id="slider-stage-wp_0"', content)
    slider_stage_block = content.split('id="slider-stage-wp_0"', 1)[1].split(
        "<!-- Diff Heatmap View -->", 1
    )[0]
    self.assertNotIn("side-label", slider_stage_block)

  def test_publish_report_artifact_writes_to_undeclared_outputs(self):
    """Verifies the report is copied into the Bazel undeclared outputs dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
      source = pathlib.Path(tmpdir) / "report.html"
      source.write_text("<html></html>", encoding="utf-8")
      outputs_dir = pathlib.Path(tmpdir) / "outputs"
      outputs_dir.mkdir()

      published = publish_report_artifact(str(source), str(outputs_dir))

      destination = outputs_dir / "sxs_report.html"
      self.assertEqual(published, str(destination))
      self.assertEqual(destination.read_text(encoding="utf-8"), "<html></html>")
      self.assertEqual(
          publish_report_artifact(str(destination), str(outputs_dir)),
          str(destination),
      )

  def test_publish_report_artifact_skips_unusable_outputs_dir(self):
    """Verifies publication is skipped when no outputs directory is usable."""
    with tempfile.TemporaryDirectory() as tmpdir:
      source = pathlib.Path(tmpdir) / "report.html"
      source.write_text("<html></html>", encoding="utf-8")

      self.assertIsNone(publish_report_artifact(str(source), None))
      self.assertIsNone(publish_report_artifact(str(source), ""))

  def test_spatial_diff_hash_uniqueness(self):
    """Verifies equal-count pixel changes at different positions differ."""
    engine = SxsDiffEngine()
    base_img = Image.new("RGB", (20, 20), color=(0, 0, 0))
    buf_base = io.BytesIO()
    base_img.save(buf_base, format="PNG")
    base_bytes = buf_base.getvalue()

    img_top_left = base_img.copy()
    img_top_left.putpixel((0, 0), (255, 0, 0))
    buf_tl = io.BytesIO()
    img_top_left.save(buf_tl, format="PNG")

    img_bottom_right = base_img.copy()
    img_bottom_right.putpixel((19, 19), (255, 0, 0))
    buf_br = io.BytesIO()
    img_bottom_right.save(buf_br, format="PNG")

    wp_tl = _evaluate(
        engine,
        base_bytes,
        buf_tl.getvalue(),
        html_a="<p>A</p>",
        journey="j",
        waypoint="w",
    )
    wp_br = _evaluate(
        engine,
        base_bytes,
        buf_br.getvalue(),
        html_a="<p>A</p>",
        journey="j",
        waypoint="w",
    )
    self.assertEqual(wp_tl.visual.diff_pixels, 1)
    self.assertEqual(wp_br.visual.diff_pixels, 1)
    self.assertNotEqual(wp_tl.diff_hash, wp_br.diff_hash)

    # Sub-threshold candidate noise away from the change keeps the approval.
    img_noisy = img_top_left.copy()
    img_noisy.putpixel((10, 10), (8, 8, 8))
    buf_nz = io.BytesIO()
    img_noisy.save(buf_nz, format="PNG")
    wp_noisy = _evaluate(
        engine,
        base_bytes,
        buf_nz.getvalue(),
        html_a="<p>A</p>",
        journey="j",
        waypoint="w",
    )
    self.assertEqual(wp_noisy.diff_hash, wp_tl.diff_hash)

  def test_network_diff_canonical_sig_edge_cases(self):
    """Verifies canonical_sig handles boolean status, NaN, and None fields."""
    engine = SxsDiffEngine()
    reqs_a: list[dict[str, object]] = [
        {"method": None, "url": None, "status": True},
        {"method": "POST", "url": "/api?b=1&a=2", "status": float("nan")},
    ]
    reqs_b: list[dict[str, object]] = [
        {"method": "GET", "url": "", "status": 1},
        {"method": "POST", "url": "/api?a=2&b=1", "status": "nan"},
    ]
    diff = engine.compute_network_diff(reqs_a, reqs_b)
    # True (bool) must not coerce to 1 (int); float("nan") must render as "nan"
    self.assertTrue(diff.has_changes)
    self.assertEqual(len(diff.status_mismatches), 2)
    has_status_match = any(
        "Status True" in m or "Status 1" in m for m in diff.status_mismatches
    )
    self.assertTrue(has_status_match)

    # None method/url defaults to GET and empty string
    diff_defaults = engine.compute_network_diff(
        [{"method": None, "url": None, "status": 200}],
        [{"method": "GET", "url": "", "status": 200}],
    )
    self.assertFalse(diff_defaults.has_changes)

    # Distinct URL paths are preserved and detected
    diff_paths = engine.compute_network_diff(
        [{"method": "GET", "url": "/a", "status": 200}],
        [{"method": "GET", "url": "/b", "status": 200}],
    )
    self.assertTrue(diff_paths.has_changes)

    # Distinct HTTP methods with identical URL/status are detected and formatted
    diff_methods = engine.compute_network_diff(
        [{"method": None, "url": "/a", "status": 200}],
        [{"method": "POST", "url": "/a", "status": 200}],
    )
    self.assertTrue(diff_methods.has_changes)
    self.assertIn(
        "Endpoint divergence 'GET /a' (Status 200): 1 in Baseline vs 0 in"
        " Candidate",
        diff_methods.status_mismatches,
    )
    self.assertIn(
        "Endpoint divergence 'POST /a' (Status 200): 0 in Baseline vs 1 in"
        " Candidate",
        diff_methods.status_mismatches,
    )

  def test_approve_waypoint_lifecycle(self):
    """Verifies approve_waypoint creates, merges, and updates entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
      manifest_path = os.path.join(tmpdir, "manifest.json")

      # 1. Create new manifest
      approve_waypoint(
          "triage",
          "overview",
          "hash_111",
          manifest_path=manifest_path,
          rationale="Initial spec",
      )
      data = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
      entry = data["approved_diffs"]["triage:overview"]
      self.assertEqual(entry["diff_hash"], "hash_111")
      self.assertEqual(entry["decision"], "INTENTIONAL")
      self.assertEqual(entry["rationale"], "Initial spec")
      self.assertIn("timestamp", entry)
      # Sorted keys keep re-approval diffs of the checked-in manifest minimal.
      self.assertEqual(list(entry), sorted(entry))

      # 2. Merge additional entry without clobbering existing
      approve_waypoint(
          "triage", "hlo_stats", "hash_222", manifest_path=manifest_path
      )
      data = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
      self.assertEqual(len(data["approved_diffs"]), 2)
      self.assertEqual(
          data["approved_diffs"]["triage:overview"]["diff_hash"], "hash_111"
      )
      self.assertEqual(
          data["approved_diffs"]["triage:hlo_stats"]["diff_hash"], "hash_222"
      )

      # 3. Update existing entry with new hash and rationale
      approve_waypoint(
          "triage",
          "overview",
          "hash_333",
          manifest_path=manifest_path,
          rationale="Updated spec",
      )
      data = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
      self.assertEqual(len(data["approved_diffs"]), 2)
      self.assertEqual(
          data["approved_diffs"]["triage:overview"]["diff_hash"], "hash_333"
      )
      self.assertEqual(
          data["approved_diffs"]["triage:overview"]["rationale"], "Updated spec"
      )

  def test_approve_waypoint_validation_and_recovery(self):
    """Verifies input validation and recovery on empty or corrupted files."""
    with tempfile.TemporaryDirectory() as tmpdir:
      manifest_path = os.path.join(tmpdir, "manifest.json")

      # Rejects blank and whitespace-only inputs
      for j, w, h in [
          ("", "w", "h"),
          ("j", "", "h"),
          ("j", "w", ""),
          (" ", "w", "h"),
          ("j", " ", "h"),
          ("j", "w", " "),
      ]:
        with self.assertRaises(ValueError):
          approve_waypoint(j, w, h, manifest_path=manifest_path)

      # Rejects corrupted JSON
      pathlib.Path(manifest_path).write_text("{bad: json", encoding="utf-8")
      with self.assertRaises(ValueError):
        approve_waypoint("j", "w", "h", manifest_path=manifest_path)

      # Recovers gracefully from empty file
      pathlib.Path(manifest_path).write_text("", encoding="utf-8")
      approve_waypoint("j", "w", "h", manifest_path=manifest_path)
      data = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
      self.assertEqual(data["approved_diffs"]["j:w"]["diff_hash"], "h")

      # Recovers gracefully from corrupted non-dict entry
      pathlib.Path(manifest_path).write_text(
          json.dumps({"approved_diffs": {"j:w": "corrupted_string"}}),
          encoding="utf-8",
      )
      approve_waypoint("j", "w", "new_hash", manifest_path=manifest_path)
      data = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
      self.assertEqual(data["approved_diffs"]["j:w"]["diff_hash"], "new_hash")

      # Cleans up temporary file when os.replace fails
      with mock.patch(
          "os.replace", side_effect=OSError("Simulated replace failure")
      ):
        with self.assertRaises(OSError):
          approve_waypoint("j", "w", "fail_hash", manifest_path=manifest_path)
      self.assertEqual(list(pathlib.Path(tmpdir).glob(".*.tmp.*")), [])

  def test_get_default_manifest_path_resolves(self):
    """Verifies default manifest path points to approved_manifest.json."""
    self.assertEqual(get_default_manifest_path().name, "approved_manifest.json")

  def test_cli_approve(self):
    """Verifies CLI approve updates manifest or fails on missing args."""
    with tempfile.TemporaryDirectory() as tmpdir:
      manifest_path = os.path.join(tmpdir, "cli_manifest.json")
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        exit_code = sxs_main([
            "approve",
            "--journey",
            "cli_journey",
            "--waypoint",
            "cli_wp",
            "--hash",
            "deadbeef12345678",
            "--manifest",
            manifest_path,
            "--rationale",
            "Approved in terminal",
        ])
      self.assertEqual(exit_code, 0)
      self.assertIn("Successfully approved cli_journey:cli_wp", buf.getvalue())
      data = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
      entry = data["approved_diffs"]["cli_journey:cli_wp"]
      self.assertEqual(entry["diff_hash"], "deadbeef12345678")
      self.assertEqual(entry["rationale"], "Approved in terminal")

    for missing_args in (
        [],
        ["approve", "--waypoint", "cli_wp", "--hash", "deadbeef12345678"],
        ["approve", "--journey", "cli_journey", "--hash", "deadbeef12345678"],
        ["approve", "--journey", "cli_journey", "--waypoint", "cli_wp"],
    ):
      with contextlib.redirect_stderr(io.StringIO()):
        with self.assertRaises(SystemExit):
          sxs_main(missing_args)

  def test_sxs_report_3_state_summary_banner(self):
    """Verifies report summary banner supports identical, approved, and diff."""
    engine = SxsDiffEngine()
    img_a = _create_test_image((128, 128, 128))
    img_b = _create_test_image((255, 0, 0))

    diff_same = _evaluate(engine, img_a)

    with tempfile.TemporaryDirectory() as tmpdir:
      content_identical = _render_report([diff_same])
      self.assertIn("ALL WAYPOINTS IDENTICAL", content_identical)
      self.assertIn("badge badge-pass", content_identical)
      self.assertIn("100% Identical to Baseline", content_identical)
      manifest_path = os.path.join(tmpdir, "approved_manifest.json")
      diff_changed = _evaluate(
          engine,
          img_a,
          img_b,
          html_a="<div>Base</div>",
          html_b="<div>Cand</div>",
          waypoint="hlo_stats",
      )
      approve_waypoint(
          "triage",
          "hlo_stats",
          diff_changed.diff_hash,
          manifest_path=manifest_path,
          rationale="Approved update",
      )
      approved_engine = SxsDiffEngine(approved_manifest_path=manifest_path)
      diff_approved = _evaluate(
          approved_engine,
          img_a,
          img_b,
          html_a="<div>Base</div>",
          html_b="<div>Cand</div>",
          waypoint="hlo_stats",
      )
      content_approved = _render_report([diff_same, diff_approved])
      self.assertIn("DIFFS APPROVED", content_approved)
      self.assertIn("badge badge-approved", content_approved)
      self.assertIn("All Diffs Approved", content_approved)
      self.assertIn(
          '<div class="tab-status-approved"></div>'
          " triage: hlo_stats (APPROVED)",
          content_approved,
      )
      self.assertIn(
          '<span class="status-approved">APPROVED (Approved update)</span>',
          content_approved,
      )
      content_diff = _render_report([diff_same, diff_approved, diff_changed])
      self.assertIn("DIFF DETECTED", content_diff)
      self.assertIn("badge badge-fail", content_diff)
      self.assertIn("Reviewer Action Required", content_diff)

  def test_sxs_report_unapproved_json_escapes_script_tags(self):
    """Verifies <script> JSON payload escapes <, >, and & characters."""
    engine = SxsDiffEngine()
    diff = _evaluate(
        engine,
        _create_test_image((0, 0, 0)),
        _create_test_image((255, 255, 255)),
        html_a="<div>A</div>",
        html_b="<div>B</div>",
        journey="<script>alert(1)</script>",
        waypoint="tag&test",
    )
    content = _render_report([diff])
    self.assertNotIn("<script>alert", content.split("<script>")[1])
    self.assertIn(r"\u003cscript\u003ealert(1)\u003c/script\u003e", content)
    self.assertIn(r"\u0026", content)

  def test_sub_budget_visual_noise_does_not_alter_diff_hash_or_bloat_report(
      self,
  ):
    """Verifies sub-budget pixel noise keeps diff_hash stable and skips PNGs."""
    engine = SxsDiffEngine()
    base_img = Image.new("RGB", (100, 100), (120, 120, 120))
    noisy_img = base_img.copy()
    # 1 pixel out of 10,000 = 0.0001 <= 0.001 (_MAX_VISUAL_DIFF_RATIO).
    noisy_img.putpixel((50, 50), (250, 250, 250))

    buf_a = io.BytesIO()
    base_img.save(buf_a, format="PNG")
    buf_b = io.BytesIO()
    noisy_img.save(buf_b, format="PNG")

    diff_clean = _evaluate(
        engine,
        buf_a.getvalue(),
        html_a="<div>Old</div>",
        html_b="<div>New</div>",
        journey="j",
        waypoint="w",
    )
    diff_noisy = _evaluate(
        engine,
        buf_a.getvalue(),
        buf_b.getvalue(),
        html_a="<div>Old</div>",
        html_b="<div>New</div>",
        journey="j",
        waypoint="w",
    )
    self.assertEqual(diff_clean.diff_hash, diff_noisy.diff_hash)

    same_noisy = _evaluate(
        engine,
        buf_a.getvalue(),
        buf_b.getvalue(),
        journey="j",
        waypoint="w_same",
    )
    self.assertEqual(same_noisy.verdict, "SAME")
    content = _render_report([same_noisy])
    self.assertNotIn("data:image/png;base64,", content)

  def test_evaluate_waypoint_allows_small_visual_pixel_budget(self):
    """Verifies verdicts allow <=0.1% pixel noise without DOM/net diffs."""
    engine = SxsDiffEngine()
    base_img = Image.new("RGB", (100, 10), (255, 255, 255))
    buf_base = io.BytesIO()
    base_img.save(buf_base, format="PNG")

    # 1 differing pixel out of 1000 (0.001 == _MAX_VISUAL_DIFF_RATIO) -> SAME
    within_budget_img = base_img.copy()
    within_budget_img.putpixel((0, 0), (0, 0, 0))
    buf_within = io.BytesIO()
    within_budget_img.save(buf_within, format="PNG")

    verdict_within = _evaluate(
        engine, buf_base.getvalue(), buf_within.getvalue()
    )
    self.assertEqual(verdict_within.visual.diff_pixels, 1)
    self.assertEqual(verdict_within.visual.diff_ratio, 0.001)
    self.assertEqual(verdict_within.verdict, "SAME")

    # Within visual budget but with DOM or network delta -> CHANGED
    verdict_dom_changed = _evaluate(
        engine,
        buf_base.getvalue(),
        buf_within.getvalue(),
        html_b="<div>Changed</div>",
    )
    self.assertEqual(verdict_dom_changed.verdict, "CHANGED")

    verdict_net_changed = _evaluate(
        engine,
        buf_base.getvalue(),
        buf_within.getvalue(),
        requests_b=[{"url": "http://localhost/data", "status": 500}],
    )
    self.assertEqual(verdict_net_changed.verdict, "CHANGED")

    # 2 differing pixels out of 1000 (0.002 > _MAX_VISUAL_DIFF_RATIO) -> CHANGED
    over_budget_img = within_budget_img.copy()
    over_budget_img.putpixel((1, 0), (0, 0, 0))
    buf_over = io.BytesIO()
    over_budget_img.save(buf_over, format="PNG")

    verdict_over = _evaluate(engine, buf_base.getvalue(), buf_over.getvalue())
    self.assertEqual(verdict_over.visual.diff_pixels, 2)
    self.assertEqual(verdict_over.visual.diff_ratio, 0.002)
    self.assertEqual(verdict_over.verdict, "CHANGED")

  def test_resolve_scenario_runs_maps_fixture_goto_and_hosts(self):
    """Verifies run and host resolution across logdir layouts."""

    @dataclasses.dataclass(frozen=True)
    class _Step:
      action: str
      target: str
      expected_selector: str

    @dataclasses.dataclass(frozen=True)
    class _Scenario:
      id: str
      fixture: str
      initial_tool: str
      steps: tuple[_Step, ...]

    with tempfile.TemporaryDirectory() as tmp:
      run_dir = pathlib.Path(tmp) / "plugins" / "profile" / "tpu_training"
      run_dir.mkdir(parents=True)
      (run_dir / "tpu_training.xplane.pb").write_bytes(b"")

      resolver = sxs_diff_engine.make_run_resolver(tmp)
      self.assertEqual(resolver("tpu-training"), "tpu_training")
      self.assertEqual(resolver("v6e-4-training"), "tpu_training")

      scenario = _Scenario(
          id="compiler",
          fixture="tpu-training",
          initial_tool="overview_page",
          steps=(
              _Step(
                  action="switch_tool",
                  target="overview_page",
                  expected_selector="overview-page",
              ),
              _Step(
                  action="goto",
                  target="v6e-4-training/hlo_stats",
                  expected_selector="hlo-stats",
              ),
              _Step(
                  action="select_host",
                  target="t1v-n-9bfa07b4-w-0",
                  expected_selector="overview-page",
              ),
          ),
      )

      resolved = sxs_diff_engine.resolve_scenario_runs(
          scenario, resolver, logdir=tmp
      )
      self.assertEqual(resolved.fixture, "tpu_training")
      self.assertEqual(resolved.steps[0].target, "overview_page")
      self.assertEqual(resolved.steps[1].target, "tpu_training/hlo_stats")
      self.assertEqual(resolved.steps[2].target, "tpu_training")

      # A declared host that is present is kept, even if it does not sort first.
      for host in ("t1v-n-0000-w-0", "t1v-n-9bfa07b4-w-0"):
        (run_dir / f"{host}.xplane.pb").write_bytes(b"")
      resolved = sxs_diff_engine.resolve_scenario_runs(
          scenario, resolver, logdir=tmp
      )
      self.assertEqual(resolved.steps[2].target, "t1v-n-9bfa07b4-w-0")

      # Verify symmetrical separator normalization when multiple runs exist
      # (including hyphenated directory on disk queried with underscores) and
      # goto steps without a slash.
      hyphen_dir = pathlib.Path(tmp) / "plugins" / "profile" / "gpu-profile"
      hyphen_dir.mkdir(parents=True)
      multi_resolver = sxs_diff_engine.make_run_resolver(tmp)
      self.assertEqual(multi_resolver("gpu_profile"), "gpu-profile")
      self.assertEqual(multi_resolver("tpu-training"), "tpu_training")

      slashless = _Scenario(
          id="slashless",
          fixture="gpu_profile",
          initial_tool="overview_page",
          steps=(
              _Step(
                  action="goto",
                  target="overview_page",
                  expected_selector="overview-page",
              ),
          ),
      )
      resolved_slashless = sxs_diff_engine.resolve_scenario_runs(
          slashless, multi_resolver, logdir=tmp
      )
      self.assertEqual(resolved_slashless.fixture, "gpu-profile")
      self.assertEqual(resolved_slashless.steps[0].target, "overview_page")

  def test_waypoint_names_are_identical_across_environments(self):
    """Waypoint names must not vary with the runs present in the logdir."""

    class _Action(enum.Enum):
      GOTO = "goto"
      SELECT_HOST = "select_host"
      SWITCH_TOOL = "switch_tool"

    @dataclasses.dataclass(frozen=True)
    class _Step:
      action: _Action
      target: str

    @dataclasses.dataclass(frozen=True)
    class _Scenario:
      id: str
      fixture: str
      initial_tool: str
      steps: tuple[_Step, ...]

    scenario = _Scenario(
        id="compiler.graph_and_multihost_scale",
        fixture="tpu-training",
        initial_tool="overview_page",
        steps=(
            _Step(action=_Action.SWITCH_TOOL, target="HLO Op Stats"),
            _Step(action=_Action.GOTO, target="v6e-4-training/overview_page"),
            _Step(action=_Action.SELECT_HOST, target="t1v-n-9bfa07b4-w-0"),
        ),
    )
    expected = [
        "00_overview_page",
        "01_switch_tool_HLO Op Stats",
        "02_goto_v6e-4-training/overview_page",
        "03_select_host_t1v-n-9bfa07b4-w-0",
    ]
    self.assertEqual(sxs_diff_engine.waypoint_names(scenario), expected)

    def _resolve_in_env(
        runs: tuple[str, ...], hosts: tuple[str, ...]
    ) -> _Scenario:
      """Resolves the journey against a logdir holding exactly these runs."""
      with tempfile.TemporaryDirectory() as tmp:
        for run in runs:
          run_dir = pathlib.Path(tmp) / "plugins" / "profile" / run
          run_dir.mkdir(parents=True)
          for host in hosts:
            (run_dir / f"{host}.xplane.pb").write_bytes(b"")
        return sxs_diff_engine.resolve_scenario_runs(
            scenario, sxs_diff_engine.make_run_resolver(tmp), logdir=tmp
        )

    # CI carries a single `tpu_training` run; a workstation additionally
    # carries `v6e-4-training` and a second host, so the same journey resolves
    # to different targets in the two environments.
    ci_resolved = _resolve_in_env(("tpu_training",), ("tpu_training",))
    workstation_resolved = _resolve_in_env(
        ("tpu_training", "v6e-4-training"),
        ("t1v-n-9bfa07b4-w-0", "t1v-n-9bfa07b4-w-1"),
    )

    # `_walk_journey` labels its captures from the declared scenario, so the
    # approval keys are `expected` above in both environments. The assertions
    # below show that is a real property and not a tautology: naming from the
    # resolved scenario, which is what this replaced, genuinely diverges.
    self.assertNotEqual(
        sxs_diff_engine.waypoint_names(ci_resolved),
        sxs_diff_engine.waypoint_names(workstation_resolved),
    )
    self.assertEqual(
        sxs_diff_engine.waypoint_names(ci_resolved)[2],
        "02_goto_tpu_training/overview_page",
    )
    self.assertEqual(
        sxs_diff_engine.waypoint_names(workstation_resolved)[2],
        "02_goto_v6e-4-training/overview_page",
    )

  def test_sanitize_dom_drops_external_scripts_but_keeps_app_bundles(self):
    """External loader scripts are timing noise; the app's own bundles are not."""
    engine = SxsDiffEngine(approved_manifest_path="")

    # What the two walks actually serialize when the gstatic fail-over lands
    # differently: one still holds the primary loader, the other has failed
    # over to the backup host.
    primary = (
        "<body><app></app>"
        '<script src="https://www.gstatic.com/charts/loader.js"></script>'
        '<script src="runtime.js"></script>'
        '<script src="bundle.js"></script>'
        "</body>"
    )
    failed_over = (
        "<body><app></app>"
        '<script src="https://www.gstatic.cn/charts/loader.js"></script>'
        '<script src="runtime.js"></script>'
        '<script src="bundle.js"></script>'
        "</body>"
    )
    self.assertEqual(
        engine.sanitize_dom(primary), engine.sanitize_dom(failed_over)
    )

    # The application's own bundles are relative, belong to the build under
    # test, and must survive so that dropping or renaming one is still caught.
    sanitized = engine.sanitize_dom(primary)
    self.assertIn('src="runtime.js"', sanitized)
    self.assertIn('src="bundle.js"', sanitized)
    self.assertNotIn("gstatic", sanitized)

    # A genuine change to the app's own script wiring must still diff.
    without_bundle = primary.replace('<script src="bundle.js"></script>', "")
    self.assertNotEqual(
        engine.sanitize_dom(primary), engine.sanitize_dom(without_bundle)
    )


if __name__ == "__main__":
  unittest.main()
