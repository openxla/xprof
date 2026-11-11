#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COUNTER_COLORS_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COUNTER_COLORS_H_

#include <optional>

#include "absl/strings/string_view.h"
#include "imgui.h"

namespace traceviewer {

// The minimum contrast ratio between the outline of a counter chart and the
// track background (WCAG 2.1 SC 1.4.11, non-text contrast).
inline constexpr float kCounterStrokeMinContrast = 3.0f;

// Colors of a counter track chart: an area filled with a vertical gradient from
// `fill_top` (at the top of the track) to `fill_bottom` (at its baseline),
// outlined along its top edge by `stroke`. All colors are opaque.
struct CounterTrackColors {
  ImU32 stroke = 0;
  ImU32 fill_top = 0;
  ImU32 fill_bottom = 0;

  friend bool operator==(const CounterTrackColors& a,
                         const CounterTrackColors& b) {
    return a.stroke == b.stroke && a.fill_top == b.fill_top &&
           a.fill_bottom == b.fill_bottom;
  }
  friend bool operator!=(const CounterTrackColors& a,
                         const CounterTrackColors& b) {
    return !(a == b);
  }
};

// Returns the identity color of a counter track of a static kernel schedule,
// or nullopt if `track_name` is not one of them:
// - "<unit> utilization" tracks (e.g. "MXU utilization") get the color of the
//   functional unit, as used for the unit in the rest of the Kernel Viewer.
// - Memory tracks (VMEM, SparseCore scratchpad spills, and register tracks
//   including "Descriptor Reg") get distinct hues, with spills in red shades.
// The colors are mid-tones designed for light backgrounds; pass them through
// MakeCounterTrackColors() to adapt them to a palette.
std::optional<ImU32> GetKernelCounterTrackBaseColor(
    absl::string_view track_name);

// Returns the chart colors of a counter track with the identity color `base`,
// drawn on `background`:
// - On dark backgrounds, `base` is lifted towards white, so it reads as a
//   luminous tint rather than a muddy mid-tone.
// - `stroke` is then pushed away from the background (darker on light
//   backgrounds, lighter on dark ones) until it reaches
//   `kCounterStrokeMinContrast` against `background`.
// - The fill gradient blends `stroke` into `background`, so it stays on-hue in
//   every palette while the outline carries the contrast.
// The alpha channels of `base` and `background` are ignored.
CounterTrackColors MakeCounterTrackColors(ImU32 base, ImU32 background);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COUNTER_COLORS_H_
