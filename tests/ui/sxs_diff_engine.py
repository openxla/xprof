"""Multi-modal Side-by-Side (SxS) diff engine for A/B testing."""

import collections
import dataclasses
import difflib
import functools
import hashlib
import io
import json
import pathlib
import re
import urllib.parse

from PIL import Image
from PIL import ImageChops

# Minimum 8-bit per-channel delta treated as a real divergence. Sub-pixel font
# hinting and GPU antialiasing routinely shift channels by a few levels between
# otherwise identical renders, so smaller deltas are noise rather than signal.
_MIN_CHANNEL_DELTA = 10


@dataclasses.dataclass
class VisualDiff:
  """Visual pixel delta result."""

  diff_ratio: float
  total_pixels: int
  diff_pixels: int
  heatmap_png_bytes: bytes | None = None
  dimension_mismatch: str | None = None
  base_png_bytes: bytes | None = None
  candidate_png_bytes: bytes | None = None


@dataclasses.dataclass
class DomDiff:
  """Structural DOM AST delta result."""

  has_changes: bool
  unified_diff: str
  added_lines: int
  deleted_lines: int
  # Digest of the whole delta. unified_diff is capped for display, so it cannot
  # stand in for the delta's identity.
  diff_digest: str


@dataclasses.dataclass
class NetworkDiff:
  """Network waterfall and REST schema delta result."""

  has_changes: bool
  request_count_a: int
  request_count_b: int
  status_mismatches: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class WaypointDiff:
  """Combined multi-modal delta at a specific user journey waypoint."""

  journey_name: str
  waypoint_name: str
  visual: VisualDiff
  dom: DomDiff
  network: NetworkDiff
  diff_hash: str
  is_approved: bool = False
  approval_rationale: str | None = None

  @property
  def verdict(self) -> str:
    """Determines top-level A/B certification verdict."""
    if (
        not self.visual.diff_pixels
        and not self.visual.dimension_mismatch
        and not self.dom.has_changes
        and not self.network.has_changes
    ):
      return "SAME"
    if self.is_approved:
      return "APPROVED"
    return "CHANGED"


class SxsDiffEngine:
  """Computes visual, structural DOM, and network deltas between Master and CL."""

  def __init__(self, approved_manifest_path: str | None = None):
    self.approved_manifest: dict[str, dict[str, str]] = {}
    if approved_manifest_path:
      try:
        manifest = json.loads(
            pathlib.Path(approved_manifest_path).read_text(encoding="utf-8")
        )
        self.approved_manifest = manifest.get("approved_diffs", {})
      except (json.JSONDecodeError, OSError):
        # A missing or unreadable manifest approves nothing, which keeps every
        # waypoint gated on a reviewer.
        pass

  def compute_visual_diff(
      self,
      img_bytes_a: bytes,
      img_bytes_b: bytes,
      background_color: tuple[int, int, int, int] = (255, 255, 255, 255),
  ) -> VisualDiff:
    """Computes perceptual pixel difference between two PNG screenshots."""
    try:
      img_a = Image.open(io.BytesIO(img_bytes_a)).convert("RGBA")
      img_b = Image.open(io.BytesIO(img_bytes_b)).convert("RGBA")
    except Exception as e:  # pylint: disable=broad-exception-caught
      # Pillow signals malformed payloads with OSError, UnidentifiedImageError,
      # or ValueError depending on where decoding fails. A corrupt screenshot
      # is a divergence to report, never a reason to abort the whole run.
      total_bytes = max(len(img_bytes_a), len(img_bytes_b))
      return VisualDiff(
          diff_ratio=1.0,
          total_pixels=total_bytes,
          diff_pixels=total_bytes,
          dimension_mismatch=f"Corrupt or invalid image bytes: {e}",
          base_png_bytes=img_bytes_a,
          candidate_png_bytes=img_bytes_b,
      )

    dimension_mismatch = (
        f"{img_a.size} vs {img_b.size}" if img_a.size != img_b.size else None
    )
    # Screenshots leave transparent pixels wherever the page did not paint, and
    # two runs may differ in extent. Flattening both onto identically sized
    # opaque canvases makes alpha and trailing padding comparable pixel-wise.
    target_size = (
        max(img_a.width, img_b.width),
        max(img_a.height, img_b.height),
    )
    flat_a = Image.new("RGBA", target_size, background_color)
    flat_a.alpha_composite(img_a)
    flat_b = Image.new("RGBA", target_size, background_color)
    flat_b.alpha_composite(img_b)

    # Deliberately not convert("L"): that weights the channels by luminance
    # (0.299R + 0.587G + 0.114B), which scales a blue-only delta down by ~8.8x
    # and pushes a 31% blue shift below the noise floor. The floor is a
    # per-channel one, so the largest single channel decides.
    channel_max = functools.reduce(
        ImageChops.lighter, ImageChops.difference(flat_a, flat_b).split()
    )
    mask = channel_max.point(
        lambda level: 255 if level > _MIN_CHANNEL_DELTA else 0
    )

    # Measured over the union canvas even when the extents differ. Both renders
    # were already composited onto it, so the region present in only one of them
    # is compared against the background and counted. Saturating the ratio to
    # 1.0 on any geometry change, as this previously did, painted the whole page
    # red and told the reviewer nothing about what moved. The extent change
    # still fails the waypoint: it travels in dimension_mismatch.
    total_pixels = target_size[0] * target_size[1]
    diff_pixels = mask.tobytes().count(255)
    diff_ratio = diff_pixels / total_pixels if total_pixels else 0.0

    # The candidate render with every diverging pixel painted over in red. The
    # card that displays this already shows the baseline and the candidate
    # themselves, so pasting them alongside the overlay would inline a second
    # copy of both screenshots and render each panel at a third of the width.
    heatmap = flat_b.copy()
    red_highlight = Image.new("RGBA", target_size, (235, 50, 50, 200))
    heatmap.paste(red_highlight, (0, 0), mask=mask)
    buf = io.BytesIO()
    heatmap.save(buf, format="PNG")

    return VisualDiff(
        diff_ratio=diff_ratio,
        total_pixels=total_pixels,
        diff_pixels=diff_pixels,
        heatmap_png_bytes=buf.getvalue(),
        dimension_mismatch=dimension_mismatch,
        base_png_bytes=img_bytes_a,
        candidate_png_bytes=img_bytes_b,
    )

  def sanitize_dom(self, html: str) -> str:
    """Strips non-deterministic Angular and Material IDs while preserving structure."""
    cleaned = re.sub(
        r'\s*_ng(content|host)-[a-zA-Z0-9_-]+(=["\'][^"\']*["\'])?', "", html
    )
    cleaned = re.sub(
        r' id="mat-(?:mdc-)?(tab-label|tab-content|select|option)-[0-9]+(-[0-9]+)?"',
        "",
        cleaned,
    )
    cleaned = re.sub(
        r' id="cdk-(describedby-message|overlay)-[0-9]+"', "", cleaned
    )
    cleaned = re.sub(
        r' aria-controls="mat-(?:mdc-)?tab-content-[0-9]+-[0-9]+"',
        "",
        cleaned,
    )
    cleaned = re.sub(
        r' aria-owns="mat-(?:mdc-)?select-[0-9]+-panel"', "", cleaned
    )
    return "\n".join(
        line.strip() for line in cleaned.splitlines() if line.strip()
    )

  def compute_dom_diff(self, html_a: str, html_b: str) -> DomDiff:
    """Computes line-by-line DOM structural differences."""
    clean_a = self.sanitize_dom(html_a).splitlines(keepends=True)
    clean_b = self.sanitize_dom(html_b).splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            clean_a, clean_b, fromfile="Master", tofile="Candidate", n=2
        )
    )
    added = sum(
        1 for l in diff_lines if l.startswith("+") and not l.startswith("+++")
    )
    deleted = sum(
        1 for l in diff_lines if l.startswith("-") and not l.startswith("---")
    )
    full_diff = "".join(diff_lines)
    return DomDiff(
        has_changes=bool(diff_lines),
        # Capped: the report renders this inside a fixed-height scroll box, and
        # a whole-page delta would otherwise inline megabytes of markup.
        unified_diff="".join(diff_lines[:100]),
        added_lines=added,
        deleted_lines=deleted,
        diff_digest=hashlib.sha256(full_diff.encode("utf-8")).hexdigest(),
    )

  def compute_network_diff(
      self,
      requests_a: list[dict[str, object]],
      requests_b: list[dict[str, object]],
  ) -> NetworkDiff:
    """Computes order-independent multiset network deltas."""
    mismatches = []
    if len(requests_a) != len(requests_b):
      mismatches.append(
          f"Request count mismatch: {len(requests_a)} vs {len(requests_b)}"
      )

    def _canonical_sig(req: dict[str, object]) -> tuple[str, str, str]:
      raw_status = req.get("status")
      # 200 and "200" describe the same response, so numeric statuses are
      # rendered canonically. An unparseable status is kept verbatim rather
      # than coerced, otherwise it would collapse onto a real 200 and hide the
      # very divergence this function exists to surface.
      status = (
          str(int(raw_status))
          if isinstance(raw_status, (int, float))
          else str(raw_status)
      )
      parsed = urllib.parse.urlsplit(str(req.get("url", "")))
      normalized_query = urllib.parse.urlencode(
          sorted(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
      )
      normalized_url = urllib.parse.urlunsplit(
          parsed._replace(query=normalized_query)
      )
      return (str(req.get("method", "GET")), normalized_url, status)

    counts_a = collections.Counter(_canonical_sig(r) for r in requests_a)
    counts_b = collections.Counter(_canonical_sig(r) for r in requests_b)

    for sig in sorted(counts_a.keys() | counts_b.keys()):
      ca = counts_a.get(sig, 0)
      cb = counts_b.get(sig, 0)
      if ca != cb:
        method, url, status = sig
        mismatches.append(
            f"Endpoint divergence '{method} {url}' (Status {status}): {ca} in"
            f" Baseline vs {cb} in Candidate"
        )

    return NetworkDiff(
        has_changes=bool(mismatches),
        request_count_a=len(requests_a),
        request_count_b=len(requests_b),
        status_mismatches=mismatches,
    )

  def evaluate_waypoint(
      self,
      journey_name: str,
      waypoint_name: str,
      img_a: bytes,
      img_b: bytes,
      html_a: str,
      html_b: str,
      requests_a: list[dict[str, object]],
      requests_b: list[dict[str, object]],
  ) -> WaypointDiff:
    """Evaluates multi-modal deltas for a waypoint with content-addressed hashing."""
    visual = self.compute_visual_diff(img_a, img_b)
    dom = self.compute_dom_diff(html_a, html_b)
    network = self.compute_network_diff(requests_a, requests_b)

    hasher = hashlib.sha256()
    hasher.update(f"{journey_name}:{waypoint_name}:".encode("utf-8"))
    hasher.update(dom.diff_digest.encode("utf-8"))
    if visual.diff_pixels > 0:
      diff_sig = (
          f"diff_pixels:{visual.diff_pixels}:ratio:{visual.diff_ratio:.6f}"
      )
      hasher.update(diff_sig.encode("utf-8"))
    if visual.dimension_mismatch:
      # Not an elif. A geometry change can measure zero diverging pixels when
      # what it added was blank, and two different geometry changes can measure
      # the same count, so without the extents an approval for "grew taller"
      # would also cover "grew wider".
      hasher.update(visual.dimension_mismatch.encode("utf-8"))
    for mismatch in network.status_mismatches:
      hasher.update(mismatch.encode("utf-8"))
    diff_hash = hasher.hexdigest()[:16]

    entry = self.approved_manifest.get(f"{journey_name}:{waypoint_name}", {})
    is_approved = entry.get("diff_hash") == diff_hash
    return WaypointDiff(
        journey_name=journey_name,
        waypoint_name=waypoint_name,
        visual=visual,
        dom=dom,
        network=network,
        diff_hash=diff_hash,
        is_approved=is_approved,
        approval_rationale=entry.get("rationale"),
    )
