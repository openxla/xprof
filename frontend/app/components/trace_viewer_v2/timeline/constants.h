#ifndef PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_CONSTANTS_H_
#define PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_CONSTANTS_H_

#include "third_party/dear_imgui/imgui.h"

namespace traceviewer {

// ImGUI uses float for drawing, so we use float for all pixel values.
using Pixel = float;

// Default colors
// go/keep-sorted start
// Black color with 100% opacity, rgba(0, 0, 0, 1).
inline constexpr ImU32 kBlackColor = IM_COL32(0, 0, 0, 255);
// Blue color with 100% opacity, rgba(0, 0, 255, 1).
inline constexpr ImU32 kBlueColor = IM_COL32(0, 0, 255, 255);
// A light gray line,  #E3E3E3 (GM3 Grey 90)
inline constexpr ImU32 kLightGrayColor = IM_COL32(0xE3, 0xE3, 0xE3, 255);
// White color with 30% opacity, rgba(255, 255, 255, 0.3).
inline constexpr ImU32 kTransparentWhiteColor = IM_COL32(255, 255, 255, 77);
// go/keep-sorted end

// Ruler Constants
// These constants are used for drawing the timeline ruler.
// go/keep-sorted start
inline constexpr ImU32 kRulerLineColor = kBlackColor;
inline constexpr ImU32 kRulerTextColor = kBlackColor;
inline constexpr Pixel kMinTickDistancePx = 80.0f;
inline constexpr Pixel kRulerHeight = 20.0f;
// The buffer for drawing elements slightly off-screen to avoid pop-in.
inline constexpr Pixel kRulerScreenBuffer = 5.0f;
inline constexpr Pixel kRulerTextPadding = 2.0f;
inline constexpr Pixel kRulerTickHeight = 8.0f;
inline constexpr int kMinorTickDivisions = 5;
// go/keep-sorted end

// Rendering Constants
// These constants are used for rendering flame chart events and track layout.
// go/keep-sorted start
inline constexpr ImDrawFlags kImDrawFlags = ImDrawFlags_RoundCornersDefault_;
inline constexpr ImU32 kDefaultTextColor = kBlackColor;
inline constexpr Pixel kCornerRounding = 0.0f;
inline constexpr Pixel kEventHeight = 16.0f;
inline constexpr Pixel kEventMinimumDrawWidth = 2.0f;
inline constexpr Pixel kEventPaddingBottom = 1.0f;
inline constexpr Pixel kEventPaddingRight = 0.5f;
// The size of the visual indent for nested groups in the timeline, indicating
// their nesting level.
inline constexpr Pixel kIndentSize = 10.0f;
inline constexpr Pixel kMinTextWidth = 5.0f;
// Padding on the right to prevent content from touching the window edge.
inline constexpr Pixel kTimelinePaddingRight = 1.0f;
// go/keep-sorted end

// Highlighting Constants
// go/keep-sorted start
inline constexpr ImU32 kHoverMaskColor = kTransparentWhiteColor;
inline constexpr ImU32 kSelectedBorderColor = kBlueColor;
// The corner rounding applied to hovered events. Set to half of `kEventHeight`
// to create a half-circle effect on the ends of the event.
inline constexpr Pixel kHoverCornerRounding = 8.0f;
inline constexpr Pixel kSelectedBorderThickness = 2.0f;
// go/keep-sorted end

// Time Range Selection Constants
// go/keep-sorted start
// A semi-transparent blue for the curtain border. #A1C9FF at 60% opacity.
inline constexpr ImU32 kSelectedTimeRangeBorderColor = 0x99FFC9A1;
// A semi-transparent blue for the curtain highlight. #A1C9FF at 30% opacity.
inline constexpr ImU32 kSelectedTimeRangeColor = 0x4DFFC9A1;
// go/keep-sorted end

// Zooming and Panning Constants
// These constants control the zooming and panning behavior of the timeline.
// go/keep-sorted start
// The rate at which panning/zooming speed increases per second after the
// initial delay.
inline constexpr float kAccelerateRate = 10.0f;
// The delay in seconds before panning/zooming acceleration takes effect.
inline constexpr float kAccelerateThreshold = 0.1f;
// The maximum factor by which the panning/zooming speed can be accelerated.
inline constexpr float kMaxAccelerateFactor = 30.0f;
// The minimum allowed zoom factor for the timeline. This prevents zooming in
// too much that time durations (mathmatically) become zero or negative.
inline constexpr float kMinZoomFactor = 0.001f;
// The sensitivity of zooming with the mouse wheel, measured in units per pixel.
inline constexpr float kMouseWheelZoomSpeed = 0.01f;
// The base speed of timeline panning, measured in pixels per second.
inline constexpr float kPanningSpeed = 250.0f;
// The base speed of timeline scrolling, measured in pixels per second.
inline constexpr float kScrollSpeed = 160.0f;
// The base speed of timeline zooming, measured in units per second.
inline constexpr float kZoomSpeed = 0.5f;
// go/keep-sorted end

// Time Range Constants
// These constants are used for constraining the time range of the timeline.
// go/keep-sorted start
// The minimum duration of the visible time range in microseconds (= 1
// picosecond). This limits the maximum zoom-in level: time range duration
// cannot shrink below this value when zooming in.
inline constexpr double kMinDurationMicros = 1e-6;
// go/keep-sorted end
}  // namespace traceviewer

#endif  // PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_CONSTANTS_H_
