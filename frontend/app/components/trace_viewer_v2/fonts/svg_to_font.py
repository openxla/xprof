"""Create a font from SVGs based on a mapping JSON.

Adapted from vscode-codicons svg_to_font.py.
It reads a mapping of codepoints to icon names, loads the corresponding SVG
files, scales and positions them according to specific rules to honor the
viewBox, converts them to TrueType format, and saves the resulting font.
"""

from collections.abc import Sequence
import json
import os
from typing import Dict, cast

from absl import app
from absl import flags
from fontTools import fontBuilder
from fontTools import svgLib
from fontTools.pens import boundsPen
from fontTools.pens import cu2quPen
from fontTools.pens import ttGlyphPen

FLAGS = flags.FLAGS

flags.DEFINE_string('mapping', None, 'Path to mapping.json', required=True)
flags.DEFINE_string('icons_dir', None, 'Path to icons directory', required=True)
flags.DEFINE_string('output', None, 'Path to output font file', required=True)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Load the mapping file that associates unicode codepoints with icon names.
  with open(FLAGS.mapping) as f:
    mapping = cast(Dict[str, list[str]], json.load(f))

  # Units Per Em (UPEM). Standard for TrueType fonts is 1024 or 2048.
  # We use 1024 to match the scale of 16px icons (1024 / 16 = 64).
  upem = 1024

  glyphs = {}
  glyph_order = ['.notdef', 'space']
  glyph_lsb = {'.notdef': 0, 'space': 0}

  # Create the mandatory .notdef glyph (rendered when a character is missing).
  # We draw a simple box as the fallback glyph.
  pen = ttGlyphPen.TTGlyphPen(None)
  pen.moveTo((0, 0))
  pen.lineTo((0, upem))
  pen.lineTo((upem, upem))
  pen.lineTo((upem, 0))
  pen.closePath()
  glyphs['.notdef'] = pen.glyph()

  # Create the space glyph (empty).
  pen = ttGlyphPen.TTGlyphPen(None)
  glyphs['space'] = pen.glyph()

  # Character map: maps unicode codepoint to glyph name.
  cmap = {0x0020: 'space'}

  # Process each entry in the mapping.
  for codepoint_str, names in mapping.items():
    codepoint = int(codepoint_str)

    # Find the SVG file for this icon.
    # The mapping may list multiple names for the same glyph; we use the first
    # one that exists.
    svg_path: str | None = None
    primary_name: str | None = None
    for name in names:
      p = os.path.join(FLAGS.icons_dir, f'{name}.svg')
      if os.path.exists(p):
        svg_path = p
        primary_name = name
        break

    if not svg_path or not primary_name:
      print(f'Warning: No SVG found for codepoint {codepoint} (names: {names})')
      continue

    glyph_name = primary_name
    # Avoid duplicating glyphs if multiple codepoints point to the same icon.
    if glyph_name in glyphs:
      cmap[codepoint] = glyph_name
      continue

    try:
      # Parse the SVG file.
      svg = svgLib.SVGPath(svg_path)

      # Extract viewBox or width/height to determine the source dimensions.
      viewbox = svg.root.attrib.get('viewBox')
      if viewbox:
        parts = [float(p) for p in viewbox.split()]
        min_x = parts[0]
        min_y = parts[1]
        width = parts[2]
        height = parts[3]
      else:
        min_x = 0
        min_y = 0
        width = float(svg.root.attrib.get('width', 16))
        height = float(svg.root.attrib.get('height', 16))

      # Compute scale.
      # If the icon is larger than 16px, we scale the largest dimension down to
      # 16px (upem).
      # If it is 16px or smaller, we do NOT scale it up, preserving its
      # relative size.
      max_dim = max(width, height)
      if max_dim > 16.0:
        scale = float(upem) / max_dim
      else:
        scale = float(upem) / 16.0  # 1024 / 16 = 64

      # Compute translation to honor the viewBox.
      # We map min_x to 0 (left edge of glyph).
      # SVG Y-axis goes down, font Y-axis goes up.
      # We map min_y (top of SVG) to upem (top of font em box).
      dx = -1 * min_x * scale
      dy = upem + min_y * scale

      # Transform matrix: [a, b, c, d, e, f]
      # x' = a*x + c*y + e
      # y' = b*x + d*y + f
      # We scale and flip Y, and apply translation.
      svg.transform = (scale, 0, 0, -scale, dx, dy)

      glyph_pen = ttGlyphPen.TTGlyphPen(None)

      # SVG uses cubic Bezier curves, but TrueType fonts use quadratic curves.
      # Cu2QuPen converts cubic curves to quadratic on the fly.
      # 0.001 is the maximum allowed error for the conversion.
      cu_pen = cu2quPen.Cu2QuPen(glyph_pen, upem * 0.001)
      svg.draw(cu_pen)

      # Compute the left side bearing (LSB) based on the actual bounds of the
      # glyph. This is needed to position the glyph correctly within the
      # font. By default the left edge of the glyph is computed and set to the
      # left edge of the character. This computes the bounding box of the glyph
      # in order to reverse that by explicitly setting the left side bearing.
      bounds_pen = boundsPen.BoundsPen(None)
      svg.draw(bounds_pen)
      lsb = 0
      if bounds_pen.bounds:
        lsb = int(bounds_pen.bounds[0])

      glyphs[glyph_name] = glyph_pen.glyph()
      glyph_order.append(glyph_name)
      glyph_lsb[glyph_name] = lsb

      cmap[codepoint] = glyph_name
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Error processing {svg_path}: {e}')
      continue

  # Build the font using FontBuilder.
  fb = fontBuilder.FontBuilder(upem, isTTF=True)
  fb.setupGlyphOrder(glyph_order)
  fb.setupGlyf(glyphs)
  fb.setupCharacterMap(cmap)

  # Set up horizontal metrics (advance width and left side bearing).
  # This enables proportional widths for glyphs based on their SVG viewBox.
  metrics = {name: (upem, glyph_lsb.get(name, 0)) for name in glyph_order}
  fb.setupHorizontalMetrics(metrics)

  # Set up standard font names.
  names = {
      'familyName': 'TraceViewerIcons',
      'styleName': 'Regular',
      'uniqueFontIdentifier': 'TraceViewerIcons Regular',
      'fullName': 'TraceViewerIcons Regular',
      'version': '1.0',
      'psName': 'TraceViewerIcons-Regular',
  }
  fb.setupNameTable(names)

  # Set up horizontal header and other tables.
  fb.setupHorizontalHeader(ascent=upem, descent=0)
  fb.setupOS2()
  fb.setupPost()

  # Save the generated TrueType font.
  fb.save(FLAGS.output)
  print(f'Font saved to {FLAGS.output}')


if __name__ == '__main__':
  app.run(main)
