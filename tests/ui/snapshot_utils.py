"""Cross-platform visual snapshot comparison utility using Pillow."""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import io
import os
import pathlib
import sys
from typing import Tuple
from PIL import Image
from PIL import ImageChops


def get_platform_name() -> str:
  """Returns normalized OS name: linux, macos, or windows."""
  if sys.platform == "darwin":
    return "macos"
  elif sys.platform == "win32":
    return "windows"
  return "linux"


def get_golden_path(snapshot_name: str) -> pathlib.Path:
  """Returns the platform-specific golden file path."""
  golden_dir = (
      pathlib.Path(__file__).resolve().parent / "goldens" / get_platform_name()
  )
  golden_dir.mkdir(parents=True, exist_ok=True)
  return golden_dir / f"{snapshot_name}.png"


def compare_screenshots(
    actual_png_bytes: bytes,
    snapshot_name: str,
    threshold: float = 0.02,
) -> Tuple[bool, float, str]:
  """Compares actual screenshot bytes against golden baseline image."""
  golden_file = get_golden_path(snapshot_name)
  actual_img = Image.open(io.BytesIO(actual_png_bytes)).convert("RGBA")

  if not golden_file.exists() or os.environ.get("UPDATE_GOLDENS") == "1":
    actual_img.save(golden_file)
    return True, 0.0, f"Saved new golden baseline at {golden_file}"

  golden_img = Image.open(golden_file).convert("RGBA")
  if actual_img.size != golden_img.size:
    return (
        False,
        1.0,
        f"Dimensions mismatch: {actual_img.size} vs {golden_img.size}",
    )

  diff = ImageChops.difference(actual_img, golden_img)
  mask = diff.convert("L").point(lambda p: 255 if p > 10 else 0)
  mismatched = sum(1 for p in mask.getdata() if p > 0)
  total = actual_img.size[0] * actual_img.size[1]
  diff_ratio = mismatched / total if total > 0 else 0.0

  if diff_ratio > threshold:
    diff_dir = golden_file.parent.parent / "diffs"
    diff_dir.mkdir(parents=True, exist_ok=True)
    diff.save(diff_dir / f"diff_{snapshot_name}.png")
    return (
        False,
        diff_ratio,
        (
            f"Snapshot '{snapshot_name}' differed by {diff_ratio:.2%} (max"
            f" {threshold:.2%})"
        ),
    )

  return (
      True,
      diff_ratio,
      f"Snapshot matched within tolerance: {diff_ratio:.2%}",
  )
