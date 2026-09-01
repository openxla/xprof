"""Multi-modal Side-by-Side (SxS) diff engine for A/B testing."""

import collections
import dataclasses
import difflib
import hashlib
import io
import json
import pathlib
import re
import urllib.parse

# pylint: disable=g-import-not-at-top
try:
  from PIL import Image
  from PIL import ImageChops

  _HAS_PIL = True
except ImportError:
  Image = None  # pyrefly: ignore[assignment]
  ImageChops = None  # pyrefly: ignore[assignment]
  _HAS_PIL = False


@dataclasses.dataclass
class VisualDiff:
  """Visual pixel delta result."""

  diff_ratio: float
  total_pixels: int
  diff_pixels: int
  composite_png_bytes: bytes | None = None
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

  def __init__(
      self,
      approved_manifest_path: str | None = None,
      pixel_threshold: float = 0.001,
  ):
    self.approved_manifest_path = approved_manifest_path
    self.pixel_threshold = pixel_threshold
    self.approved_manifest: dict[str, dict[str, str]] = {}
    if approved_manifest_path and pathlib.Path(approved_manifest_path).exists():
      try:
        data = json.loads(
            pathlib.Path(approved_manifest_path).read_text(encoding="utf-8")
        )
        self.approved_manifest = data.get("approved_diffs", {})
      except (json.JSONDecodeError, OSError):
        self.approved_manifest = {}

  def compute_visual_diff(
      self, img_bytes_a: bytes, img_bytes_b: bytes
  ) -> VisualDiff:
    """Computes perceptual pixel difference between two PNG screenshots."""
    if not _HAS_PIL or Image is None or ImageChops is None:
      is_same = img_bytes_a == img_bytes_b
      return VisualDiff(
          diff_ratio=0.0 if is_same else 1.0,
          total_pixels=max(len(img_bytes_a), len(img_bytes_b)),
          diff_pixels=0 if is_same else max(len(img_bytes_a), len(img_bytes_b)),
          composite_png_bytes=img_bytes_a if is_same else None,
          base_png_bytes=img_bytes_a,
          candidate_png_bytes=img_bytes_b,
      )

    img_a = Image.open(io.BytesIO(img_bytes_a)).convert("RGBA")
    img_b = Image.open(io.BytesIO(img_bytes_b)).convert("RGBA")

    if img_a.size != img_b.size:
      return VisualDiff(
          diff_ratio=1.0,
          total_pixels=img_a.width * img_a.height,
          diff_pixels=abs(
              img_a.width * img_a.height - img_b.width * img_b.height
          )
          or 1,
          dimension_mismatch=f"{img_a.size} vs {img_b.size}",
      )

    diff = ImageChops.difference(img_a, img_b)
    threshold = (
        int(self.pixel_threshold * 255)
        if self.pixel_threshold < 1.0
        else int(self.pixel_threshold)
    )
    threshold = max(threshold, 10)
    mask = diff.convert("L").point(lambda p: 255 if p > threshold else 0)
    diff_pixels = mask.tobytes().count(255)
    total_pixels = img_a.width * img_a.height
    diff_ratio = diff_pixels / float(total_pixels) if total_pixels > 0 else 0.0

    diff_overlay = img_b.copy()
    red_highlight = Image.new("RGBA", img_b.size, (235, 50, 50, 200))
    diff_overlay.paste(red_highlight, (0, 0), mask=mask)

    composite = Image.new("RGBA", (img_a.width * 3, img_a.height))
    composite.paste(img_a, (0, 0))
    composite.paste(img_b, (img_a.width, 0))
    composite.paste(diff_overlay, (img_a.width * 2, 0))
    buf = io.BytesIO()
    composite.save(buf, format="PNG")

    return VisualDiff(
        diff_ratio=diff_ratio,
        total_pixels=total_pixels,
        diff_pixels=diff_pixels,
        composite_png_bytes=buf.getvalue(),
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
    return DomDiff(
        has_changes=bool(diff_lines),
        unified_diff="".join(diff_lines[:100]),
        added_lines=added,
        deleted_lines=deleted,
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

    def _canonical_sig(req: dict[str, object]) -> tuple[str, str, int]:
      status_val = req.get("status")
      if isinstance(status_val, (int, float, str)):
        try:
          status_int = int(status_val)
        except ValueError:
          status_int = 200
      else:
        status_int = 200
      parsed = urllib.parse.urlsplit(str(req.get("url", "")))
      normalized_query = urllib.parse.urlencode(
          sorted(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
      )
      normalized_url = urllib.parse.urlunsplit(
          parsed._replace(query=normalized_query)
      )
      return (
          str(req.get("method", "GET")),
          normalized_url,
          status_int,
      )

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
    hasher.update(dom.unified_diff.encode("utf-8"))
    if visual.diff_pixels > 0:
      diff_sig = (
          f"diff_pixels:{visual.diff_pixels}:ratio:{visual.diff_ratio:.6f}"
      )
      hasher.update(diff_sig.encode("utf-8"))
    elif visual.dimension_mismatch:
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
