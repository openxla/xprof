# Trace Viewer Icons TTF Generation (`trace_viewer_icons.ttf`)

This document explains how the custom TrueType icon font
(`trace_viewer_icons.ttf`) is generated from SVG assets, compiled into a C++
header at build time, and merged into Dear ImGui's font atlas at runtime.

## 1. Overview

Trace Viewer V2 renders UI action icons (such as the track Pin/Unpin and
Hide/Unhide buttons) as font glyphs rather than procedurally drawn primitives.
The pipeline consists of three stages:

1.  **Offline TTF Generation (`svg_to_font.py`)**: Converts SVG icons and a
    Unicode Private Use Area (PUA) mapping file (`mapping.json`) into
    `trace_viewer_icons.ttf` using Python's `fontTools`.
2.  **Build-Time Header Compression (`imgui_font_headers`)**: Compiles
    `trace_viewer_icons.ttf` into a Base85-compressed C++ header
    (`trace_viewer_icons.h`) using Dear ImGui's `binary_to_compressed_c` tool.
3.  **Runtime Font Atlas Merging (`fonts.cc`)**: Merges the icon glyph range
    (`0xE000`–`0xE050`) into each loaded Roboto font style in Dear ImGui.

---

## 2. Source Assets and Codepoint Mapping

Each icon is authored as a single-path or compound-path SVG file (e.g.,
`pin.svg`, `unpin.svg`, `hide.svg`, `unhide.svg`) and assigned a Unicode
codepoint in the Private Use Area (`U+E000`–`U+E050`) via `mapping.json`:

```json
{
  "57345": ["pin"],
  "57346": ["unpin"],
  "57347": ["hide"],
  "57348": ["visible", "unhide"]
}
```

| Decimal | Hex      | UTF-8            | Icon Name(s)        | C++ Macro             |
| :------ | :------- | :--------------- | :------------------ | :-------------------- |
| `57345` | `0xE001` | `"\xee\x80\x81"` | `pin`               | `ICON_PIN_BUTTON`     |
| `57346` | `0xE002` | `"\xee\x80\x82"` | `unpin`             | `ICON_UNPIN_BUTTON`   |
| `57347` | `0xE003` | `"\xee\x80\x83"` | `hide`              | `ICON_HIDDEN_BUTTON`  |
| `57348` | `0xE004` | `"\xee\x80\x84"` | `visible`, `unhide` | `ICON_VISIBLE_BUTTON` |

---

## 3. Generating `trace_viewer_icons.ttf`

### How `svg_to_font.py` Works

The generator script uses `fontTools` (`fontBuilder`, `svgLib`, `ttGlyphPen`,
`cu2quPen`, and `boundsPen`) to build a TrueType font:

1.  **Units Per Em (UPEM)**: Set to `1024` (`64` font units per pixel on a
    `16px` grid).
2.  **Base Glyphs**: Initializes the mandatory `.notdef` fallback glyph (a
    `1024x1024` rectangle) and an empty `space` glyph (`U+0020`).
3.  **SVG ViewBox Scaling & Y-Axis Inversion**:
    *   Reads the SVG `viewBox` (`min_x`, `min_y`, `width`, `height`) or
        `width`/`height` attributes (defaulting to `16x16`).
    *   Scales icons larger than `16px` down to fit the `1024` UPEM square
        (`scale = 1024.0 / max(width, height)`), while keeping icons `<= 16px`
        at `scale = 1024.0 / 16.0 = 64.0`.
    *   Applies an affine transform `(scale, 0, 0, -scale, -min_x * scale, upem
        + min_y * scale)` to flip the downward SVG Y-axis into the upward
        TrueType Y-axis.
4.  **Cubic-to-Quadratic Curve Conversion**:
    *   SVG paths use cubic Bézier curves, whereas TrueType `glyf` tables
        require quadratic Bézier curves. `Cu2QuPen` (`max_err = upem * 0.001`)
        converts curves on the fly into `TTGlyphPen`.
    *   `BoundsPen` computes each transformed glyph's bounding box to set its
        explicit left side bearing (`LSB`) and an advance width of `1024`
        (`upem`).
5.  **Font Table Assembly**:
    *   `FontBuilder(1024, isTTF=True)` populates the `glyf`, `cmap`, `hmtx`,
        `name` (`TraceViewerIcons Regular`), `hhea` (`ascent=1024, descent=0`),
        `OS/2`, and `post` tables and writes `trace_viewer_icons.ttf`.

### Adding or Updating Icons

1.  Add or update the `.svg` file in the `icons/` directory.
2.  Add or update the codepoint entry in `icons/mapping.json` (using decimal PUA
    codepoints in `57344`–`57424` / `0xE000`–`0xE050`).
3.  Regenerate `trace_viewer_icons.ttf`:

<!-- disableFinding(LINE_OVER_80) -->
```bash
python3 svg_to_font.py \
    --mapping=icons/mapping.json \
    --icons_dir=icons \
    --output=xprof/frontend/app/components/trace_viewer_v2/fonts/trace_viewer_icons.ttf
```
<!-- enableFinding(LINE_OVER_80) -->

4.  If adding a new icon, define its UTF-8 PUA macro in
    `timeline/draw_helpers.cc` (or the relevant rendering file).

---

## 4. Build and Runtime Integration

1.  **Build-Time Header Generation (`fonts/BUILD` & `fonts/font_rules.bzl`)**:
    The `imgui_font_headers` Starlark rule runs Dear ImGui's
    `binary_to_compressed_c -base85` on `trace_viewer_icons.ttf`:

    ```python
    imgui_font_headers(
        name = "trace_viewer_icons_header",
        srcs = ["trace_viewer_icons.ttf"],
        args = ["-base85"],
    )
    ```

    This produces `trace_viewer_icons.h`, exposing
    `trace_viewer_icons_compressed_data_base85`.

2.  **Runtime Font Atlas Merging (`fonts/fonts.cc`)**: Inside `LoadFonts()`,
    after loading each Roboto font size (`body_large`, `label_large`,
    `label_medium`, `label_small`, `title_small`), the icon font is merged into
    the same `ImFont` instance:

    ```cpp
    ImFontConfig icons_config = *font_config;
    icons_config.MergeMode = true;
    icons_config.PixelSnapH = true;
    static const ImWchar icons_ranges[] = {0xe000, 0xe050, 0};
    io.Fonts->AddFontFromMemoryCompressedBase85TTF(
        trace_viewer_icons_compressed_data_base85, base_size, &icons_config,
        icons_ranges);
    ```

3.  **Drawing Glyphs (`timeline/draw_helpers.cc`)**: `DrawPinIcon()` and
    `DrawHideIcon()` measure the UTF-8 PUA string with `ImGui::CalcTextSize()`
    and draw it centered in the button bounding box via `ImDrawList::AddText()`.
