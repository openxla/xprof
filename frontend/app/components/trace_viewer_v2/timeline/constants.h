#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_CONSTANTS_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_CONSTANTS_H_

#include <cstddef>

#include "imgui.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

// ImGUI uses float for drawing, so we use float for all pixel values.
using Pixel = float;

// Default colors
// go/keep-sorted start

// Black color with 100% opacity, rgba(0, 0, 0, 1).
inline constexpr ImU32 kBlackColor = IM_COL32(0, 0, 0, 255);
// Blue color with 100% opacity, rgba(0, 0, 255, 1).
inline constexpr ImU32 kBlueColor = IM_COL32(0, 0, 255, 255);
// Dark gray color for timing marker edges.
inline constexpr ImU32 kDarkGrayColor = IM_COL32(100, 100, 100, 255);
// A light gray line,  #E3E3E3 (GM3 Grey 90)
inline constexpr ImU32 kLightGrayColor = IM_COL32(0xE3, 0xE3, 0xE3, 255);
// Red color with 100% opacity, rgba(255, 0, 0, 1).
inline constexpr ImU32 kRedColor = IM_COL32(255, 0, 0, 255);
// Black color with 30% opacity for subtle dimming.
inline constexpr ImU32 kSubtleDimmingColor = IM_COL32(0, 0, 0, 77);
// White color with 30% opacity, rgba(255, 255, 255, 0.3).
inline constexpr ImU32 kTransparentWhiteColor = IM_COL32(255, 255, 255, 77);
// White color with 100% opacity, rgba(255, 255, 255, 1).
inline constexpr ImU32 kWhiteColor = IM_COL32(255, 255, 255, 255);
// go/keep-sorted end

// Counter Track Constants
// go/keep-sorted start
inline constexpr ImU32 kCounterTrackColor = kBlue80;
inline constexpr Pixel kCounterTrackHeight = 40.0f;
// go/keep-sorted end

// Ruler Constants
// These constants are used for drawing the timeline ruler.
// go/keep-sorted start
inline constexpr ImU32 kRulerLineColor = kOutlineVariantColor;
inline constexpr ImU32 kRulerTextColor = kOutlineColor;
inline constexpr ImU32 kTraceVerticalLineColor = kInverseOnSurfaceColor;
inline constexpr Pixel kMinTickDistancePx = 80.0f;
inline constexpr Pixel kRulerHeight = 24.0f;
// The height of the minor ticks on the ruler. Major ticks span the full height
// of the ruler.
inline constexpr Pixel kRulerMinorTickHeight = 8.0f;
// The buffer for drawing elements slightly off-screen to avoid pop-in.
inline constexpr Pixel kRulerScreenBuffer = 5.0f;
inline constexpr Pixel kRulerTextPadding = 2.0f;
inline constexpr int kMinorTickDivisions = 5;
// go/keep-sorted end

// Rendering Constants
// These constants are used for rendering flame chart events and track layout.
// go/keep-sorted start
inline constexpr ImDrawFlags kImDrawFlags = ImDrawFlags_RoundCornersDefault_;
inline constexpr ImU32 kDefaultTextColor = kBlackColor;
inline constexpr Microseconds kMinVisibleEventDuration = 1000.0;
inline constexpr Pixel kCornerRounding = 0.0f;
inline constexpr Pixel kDefaultLabelWidth = 250.0f;
inline constexpr Pixel kEventHeight = 18.0f;
inline constexpr Pixel kEventMinimumDrawWidth = 2.0f;
inline constexpr Pixel kEventPaddingBottom = 1.0f;
inline constexpr Pixel kEventPaddingRight = 1.0f;
inline constexpr Pixel kInstantEventChevronHalfWidth = 4.5f;
inline constexpr Pixel kInstantEventChevronHeight = 15.5f;
inline constexpr Pixel kInstantEventHoverChevronHalfWidth = 7.0f;
inline constexpr Pixel kInstantEventHoverChevronHeight = 20.0f;
inline constexpr Pixel kMinTextWidth = 5.0f;
inline constexpr Pixel kMinUtilizationNormalization = 1.0f;
inline constexpr Pixel kProcessTrackGap = 7.0f;
inline constexpr Pixel kSplitterOffset = 4.0f;
inline constexpr Pixel kSplitterWidth = 8.0f;
inline constexpr Pixel kThreadTrackGap = 4.0f;
// Padding on the right to prevent content from touching the window edge.
inline constexpr Pixel kTimelinePaddingRight = 1.0f;
inline constexpr Pixel kToastCornerRounding = 4.0f;
inline constexpr Pixel kVirtualHeaderHeight = 30.0f;
inline constexpr double kEpsilon = 0.001;
// go/keep-sorted end

// Highlighting Constants
// go/keep-sorted start
inline constexpr ImU32 kCounterHoverColor = kBlue80;
inline constexpr ImU32 kHoverMaskColor = kTransparentWhiteColor;
inline constexpr ImU32 kSelectedBorderColor = kBlueColor;
inline constexpr Pixel kCounterHoverThickness = 3.5f;
// The arrows between the selected HLO op and its operands and consumers run
// up or down from a bar, along a lane outside the rows involved, and up or
// down again to the other bar (see RouteDependencyArrow).
inline constexpr Pixel kHloDependencyArrowCornerRadius = 6.0f;
inline constexpr Pixel kHloDependencyArrowHeadHalfWidth = 3.5f;
inline constexpr Pixel kHloDependencyArrowHeadLength = 5.0f;
// The arrows touch the selected op this far inside its near end, at most a
// quarter of the bar.
inline constexpr Pixel kHloDependencyArrowInset = 10.0f;
// The first lane is this far from the rows involved; each further arrow on the
// same side gets its own lane, another step away.
inline constexpr Pixel kHloDependencyArrowLaneGap = 8.0f;
inline constexpr Pixel kHloDependencyArrowLaneStep = 6.0f;
// Arrows between bars closer than this have no room for an arrowhead.
inline constexpr Pixel kHloDependencyArrowMinSpanForHead = 16.0f;
inline constexpr Pixel kHloDependencyArrowThickness = 1.5f;
inline constexpr Pixel kHoverCornerRounding = 4.0f;
inline constexpr Pixel kHoverPadding = 2.0f;
// The radius of the circle used to draw points in the counter track.
inline constexpr Pixel kPointRadius = 3.0f;
inline constexpr Pixel kSelectedBorderThickness = 2.0f;
inline constexpr Pixel kSelectedDataPointRadius = 4.0f;
// The opacity of the group preview (aggregated view) background.
inline constexpr float kGroupPreviewOpacity = 0.6f;
// At most this many HLO dependency arrows are drawn on each side of the
// selected op, so that an op with many consumers does not cover the timeline.
inline constexpr size_t kMaxHloDependencyArrowsPerSide = 8;
// go/keep-sorted end

// Splitter Highlighting Constants
// go/keep-sorted start
inline constexpr ImU32 kSplitterActiveColor = kBlue60;
inline constexpr ImU32 kSplitterHoverColor = kBlue80;
inline constexpr Pixel kSplitterDefaultThickness = 1.0f;
inline constexpr Pixel kSplitterHoverThickness = 2.0f;
// go/keep-sorted end

// Process Track Constants
// go/keep-sorted start
inline constexpr ImU32 kProcessTrackCollapsedColor = kInverseOnSurfaceColor;
inline constexpr ImU32 kProcessTrackExpandedColor = kSecondaryContainerColor;
inline constexpr Pixel kProcessTrackHeight = 36.0f;
// go/keep-sorted end

// Track Sidebar Constants
// These constants style the track label column (the sidebar on the left of the
// timeline): row striping, hover feedback, collapsed cues and tree guides.
// Translucent colors are derived from the palette foreground at the given
// opacity, so the sidebar adapts to both light and dark palettes.
// go/keep-sorted start
// The corner rounding of the collapsed chevron's background square.
inline constexpr Pixel kTrackChevronButtonRounding = 4.0f;
// The size of the rounded square drawn behind the chevron of collapsed rows.
inline constexpr Pixel kTrackChevronButtonSize = 14.0f;
// The size (width of the open side) of the chevron glyph.
inline constexpr Pixel kTrackChevronSize = 7.0f;
// The horizontal slot reserved for the expand/collapse chevron. Child tracks
// are indented by the same amount, so a child's chevron sits right under its
// parent's name.
inline constexpr Pixel kTrackChevronSlotWidth = 14.0f;
// The thickness of the chevron strokes.
inline constexpr Pixel kTrackChevronThickness = 1.5f;
// The horizontal inset (per layer) of the "stack of cards" hint drawn below
// collapsed rows.
inline constexpr Pixel kTrackCollapsedStackInset = 4.0f;
// The vertical spacing between the "stack of cards" hint layers.
inline constexpr Pixel kTrackCollapsedStackSpacing = 2.0f;
// The height of the count chip shown on collapsed tracks.
inline constexpr Pixel kTrackCountChipHeight = 14.0f;
// The horizontal padding inside the count chip.
inline constexpr Pixel kTrackCountChipPaddingX = 5.0f;
// The height of the drop shadow cast below process headers.
inline constexpr Pixel kTrackHeaderShadowHeight = 4.0f;
// The gap between the pin and hide icon buttons.
inline constexpr Pixel kTrackIconButtonGap = 2.0f;
// The margin between the rightmost icon button and the label column's edge.
inline constexpr Pixel kTrackIconButtonMarginRight = 4.0f;
// The size of the square pin / hide icon buttons (and their hover circle).
inline constexpr Pixel kTrackIconButtonSize = 20.0f;
// The horizontal gap between the chevron slot, the track name, and the count
// chip.
inline constexpr Pixel kTrackLabelGap = 6.0f;
// The space between the label column's left edge and the first chevron slot.
inline constexpr Pixel kTrackLabelPaddingLeft = 6.0f;
// The space between the end of a label row's content and the splitter.
inline constexpr Pixel kTrackLabelPaddingRight = 8.0f;
// The horizontal inset of the rounded hover pill from the label column edges.
inline constexpr Pixel kTrackRowPillInsetX = 4.0f;
// The vertical inset of the rounded hover pill from the row edges.
inline constexpr Pixel kTrackRowPillInsetY = 1.0f;
// The corner rounding of the hover pill.
inline constexpr Pixel kTrackRowPillRounding = 6.0f;
// The opacity of the overlay drawn on a track's canvas row while the row
// (either its label or its canvas area) is hovered.
inline constexpr float kTrackCanvasRowFocusOpacity = 0.03f;
// The opacity of the collapsed chevron's background square while hovered.
inline constexpr float kTrackCollapsedChevronButtonHoverOpacity = 0.18f;
// The opacity of the collapsed chevron's background square.
inline constexpr float kTrackCollapsedChevronButtonOpacity = 0.10f;
// The opacity of the accent-colored count chip background on collapsed rows.
inline constexpr float kTrackCollapsedChipAccentOpacity = 0.28f;
// The opacity of the top layer of the collapsed "stack of cards" hint.
inline constexpr float kTrackCollapsedStackOpacity = 0.85f;
// The opacity of the full-row overlay drawn when a label row is clicked and
// its name is copied.
inline constexpr float kTrackCopyFlashOpacity = 0.16f;
// The opacity of the count chip background drawn on collapsed tracks.
inline constexpr float kTrackCountChipOpacity = 0.10f;
// How far the process header gradient's top color moves towards the
// foreground color.
inline constexpr float kTrackHeaderGradientStrength = 0.05f;
// The opacity of the 1px inner highlight along the top of process headers.
inline constexpr float kTrackHeaderHighlightOpacity = 0.07f;
// The opacity (black) at the top of the drop shadow below process headers.
inline constexpr float kTrackHeaderShadowOpacity = 0.14f;
// The opacity of the overlay drawn on the hovered label row.
inline constexpr float kTrackHoverOpacity = 0.07f;
// The opacity of muted text (subtitles, counts, section headers).
inline constexpr float kTrackMutedTextOpacity = 0.62f;
// The opacity of the neutral accent used by tracks without a dominant color.
inline constexpr float kTrackNeutralAccentOpacity = 0.28f;
// The opacity of secondary (child track) label text.
inline constexpr float kTrackSecondaryTextOpacity = 0.86f;
// The opacity of the vertical tree guides connecting child tracks to parents.
inline constexpr float kTrackTreeGuideOpacity = 0.16f;
// The opacity of the alternating (zebra) row stripe spanning label and canvas.
inline constexpr float kTrackZebraStripeOpacity = 0.035f;
// go/keep-sorted end

// Modern Styling Constants
// CSS-like finishing touches shared by the timeline chrome.
// go/keep-sorted start
// The padding inside card-style tooltips.
inline constexpr ImVec2 kCardTooltipPadding = ImVec2(8.0f, 6.0f);
// The corner rounding of card-style tooltips.
inline constexpr Pixel kCardTooltipRounding = 6.0f;
// The extra space between letters of small upper-case section labels.
inline constexpr Pixel kSectionLabelLetterSpacing = 0.8f;
// The height of the shadow cast by the sticky ruler over scrolled tracks.
inline constexpr Pixel kStickyHeaderShadowHeight = 6.0f;
// The opacity of the faint circle behind an active (pinned / hidden) icon
// button.
inline constexpr float kIconButtonActiveOpacity = 0.08f;
// The opacity of the circular hover state behind icon buttons.
inline constexpr float kIconButtonHoverOpacity = 0.12f;
// The opacity of the circular state behind an icon button while pressed.
inline constexpr float kIconButtonPressedOpacity = 0.20f;
// The opacity (black) at the top of the sticky ruler's shadow.
inline constexpr float kStickyHeaderShadowOpacity = 0.18f;
// go/keep-sorted end

// Motion Constants
// Timing for the track sidebar's motion (see Timeline::set_animations_enabled).
// Rates are exponential smoothing rates in 1/s: ~3/rate seconds to settle.
// go/keep-sorted start
// The chevron rotation rate (~150 ms to settle).
inline constexpr float kMotionChevronRate = 20.0f;
// The duration over which the copy flash on a label row fades out.
inline constexpr float kMotionCopyFlashSeconds = 0.6f;
// The hover fade-in rate (~120 ms to settle).
inline constexpr float kMotionHoverFadeInRate = 24.0f;
// The hover fade-out rate (~250 ms to settle), slower than fade-in so the
// highlight trails the pointer.
inline constexpr float kMotionHoverFadeOutRate = 12.0f;
// The scale at which an icon button's hover circle starts growing from.
inline constexpr float kMotionIconRippleStartScale = 0.55f;
// Collapsing is shorter than expanding: things leaving should not linger.
inline constexpr float kMotionLayoutCollapseTimeScale = 0.85f;
// The longest expand / collapse transition, for very large moves.
inline constexpr float kMotionLayoutMaxSeconds = 0.34f;
// The shortest expand / collapse transition.
inline constexpr float kMotionLayoutMinSeconds = 0.18f;
// The extra transition time per pixel of travel, so bigger moves take a little
// longer and all moves feel equally fast.
inline constexpr float kMotionLayoutSecondsPerPixel = 0.00035f;
// go/keep-sorted end

// Nesting Level Constants
// go/keep-sorted start
inline constexpr int kCounterNestingLevel = 2;
inline constexpr int kHeaderNestingLevel = 0;
inline constexpr int kProcessNestingLevel = 1;
inline constexpr int kThreadNestingLevel = 2;
// go/keep-sorted end

// Virtual Header ID Constants
// go/keep-sorted start
inline constexpr int kAllHeaderId = 100000;
inline constexpr int kHiddenHeaderId = 200000;
inline constexpr int kPinnedHeaderId = 300000;
// go/keep-sorted end

// Time Range Selection Constants
// go/keep-sorted start
// A solid blue for the curtain border. #A1C9FFFF at 100% opacity.
inline constexpr ImU32 kSelectedTimeRangeColor = 0xFFFFC9A1;
inline constexpr ImU32 kSelectedTimeRangeResizeColor = kRedColor;
inline constexpr Pixel kSelectedTimeRangeTextBottomPadding = 10.0f;
inline constexpr Pixel kSelectedTimeRangeTextTopPadding = 5.0f;
// Default element dimensions
inline constexpr Pixel kSelectedTimeRangeThickness = 1.0f;
// The threshold in pixels within which the mouse is considered to be over an
// edge of a selected time range for resizing.
inline constexpr Pixel kSelectionEdgeThreshold = 5.0f;
// go/keep-sorted end

// Bookmark Constants
// go/keep-sorted start
inline constexpr ImU32 kBookmarkColor = kPink80;
inline constexpr Pixel kBookmarkLabelPadding = 4.0f;
inline constexpr Pixel kBookmarkThickness = 2.0f;
// go/keep-sorted end

// Close Button Constants
// go/keep-sorted start
inline constexpr ImU32 kCloseButtonColor = IM_COL32(128, 128, 128, 255);
inline constexpr ImU32 kCloseButtonHoverColor = IM_COL32(100, 100, 100, 255);
inline constexpr Pixel kCloseButtonPadding = 4.0f;
inline constexpr Pixel kCloseButtonSize = 14.0f;
inline constexpr Pixel kSelectedTimeRangeArrowHeadSize = 5.0f;
inline constexpr Pixel kSelectedTimeRangeArrowPadding = 4.0f;
// go/keep-sorted end

// Mouse Interaction Constants
// go/keep-sorted start
// If the mouse moves more than 5 pixels (5*5=25) between mouse down and mouse
// up, it's considered a drag, not a click.
inline constexpr float kClickDistanceThresholdSquared = 25.0f;
// go/keep-sorted end

// Zooming and Panning Constants
// These constants control the zooming and panning behavior of the timeline.
// go/keep-sorted start
// The rate at which panning/zooming speed increases per second after the
// initial delay.
inline constexpr float kAccelerateRate = 10.0f;
// The delay in seconds before panning/zooming acceleration takes effect.
inline constexpr float kAccelerateThreshold = 0.1f;
// The dominance ratio required to lock movement to a single axis
// (e.g. 2.5x).
inline constexpr float kAxisLockRatio = 2.5f;
inline constexpr float kButtonZoomInFactor = 0.8f;
inline constexpr float kButtonZoomOutFactor = 1.25f;
// The maximum factor by which the panning/zooming speed can be accelerated.
inline constexpr float kMaxAccelerateFactor = 30.0f;
// The minimum allowed zoom factor for the timeline. This prevents zooming in
// too much that time durations (mathmatically) become zero or negative.
inline constexpr float kMinZoomFactor = 0.001f;
// The sensitivity of zooming with the mouse wheel, measured in units per pixel.
inline constexpr float kMouseWheelZoomSpeed = 0.2f;
// The base speed of timeline panning, measured in pixels per second.
inline constexpr float kPanningSpeed = 1000.0f;
// The base speed of timeline scrolling, measured in pixels per second.
inline constexpr float kScrollSpeed = 160.0f;
// The multiplier applied to panning speed when Shift is held down (matches v1
// ratio: 0.5 / 0.3).
inline constexpr float kShiftPanAccelerateFactor = 0.5f / 0.3f;
// The multiplier applied to zooming speed when Shift is held down (matches v1
// ratio: 10 / 1.5).
inline constexpr float kShiftZoomAccelerateFactor = 10.0f / 1.5f;
// The deadband threshold in pixels to filter micro-displacements from wheel
// events.
inline constexpr float kWheelDeadband = 0.1f;
// The base speed of timeline zooming, measured in units per second.
inline constexpr float kZoomSpeed = 1.5f;
// go/keep-sorted end

// Time Range Constants
// These constants are used for constraining the time range of the timeline.
// go/keep-sorted start
// The minimum duration of the visible time range in microseconds (= 1
// picosecond). This limits the maximum zoom-in level: time range duration
// cannot shrink below this value when zooming in.
inline constexpr Microseconds kMinDurationMicros = 1e-6;
// The minimum duration of a fetch request in microseconds (= 1 microsecond).
// Every fetch asks the backend for the same number of resolution bins (see
// `updateUrlWithResolution` in main.ts), so the fetch window sets the shortest
// event that survives backend downsampling, e.g. window / 6680 at 1920px. Keep
// this floor well below the shortest events users zoom into, or zooming in
// stops loading more detail. 1 us gives ~0.15ns bins at 1920px, under one TPU
// cycle (e.g. LLO instruction traces), and still guards against zero-length
// fetch windows when zoomed in to kMinDurationMicros.
inline constexpr Microseconds kMinFetchDurationMicros = 1.0;
// The ratio of the viewport width to fetch data for.
inline constexpr float kFetchRatio = 3.0f;
// The ratio of the viewport width to keep data loaded for.
inline constexpr float kPreserveRatio = 2.0f;
// If user zooms in more than this ratio from the time range of fetched data,
// refetch data to get higher resolution.
inline constexpr float kRefetchZoomRatio = 8.0f;
// go/keep-sorted end

// Event Navigation Constants
// These constants are used for zooming to an event.
// go/keep-sorted start
// The maximum duration to zoom to when navigating to an event (=5s).
inline constexpr Microseconds kEventNavigationMaxDurationMicros = 5000000.0;
// The minimum duration to zoom to when navigating to an event (=10us).
inline constexpr Microseconds kEventNavigationMinDurationMicros = 10.0;
// The factor by which to multiply event duration to determine viewport size.
inline constexpr double kEventNavigationZoomFactor = 2.5;
// go/keep-sorted end

// UI Strings Constants
// go/keep-sorted start
inline constexpr char kAllHeaderName[] = "All";
inline constexpr char kCannotHideLastProcessNotification[] =
    "Cannot hide the last visible process.";
inline constexpr char kDragToResizeTooltip[] = "Drag to resize";
inline constexpr char kHiddenHeaderName[] = "Hidden";
inline constexpr char kHiddenProcessNotificationPrefix[] = "Hidden process: ";
inline constexpr char kHideTrackTooltip[] = "Hide track";
inline constexpr char kMovingTrackNotificationPrefix[] = "Moving track ";
inline constexpr char kPinTrackTooltip[] = "Pin track";
inline constexpr char kPinnedHeaderName[] = "Pinned";
inline constexpr char kPinnedProcessNotificationPrefix[] = "Pinned process: ";
inline constexpr char kProcessHeaderLabel[] = "Process";
inline constexpr char kReorderTrackTooltip[] = "Reorder track";
inline constexpr char kUnhiddenProcessNotificationPrefix[] =
    "Unhidden process: ";
inline constexpr char kUnhideTrackTooltip[] = "Unhide track";
inline constexpr char kUnpinTrackTooltip[] = "Unpin track";
inline constexpr char kUnpinnedProcessNotificationPrefix[] =
        "Unpinned process: ";
// go/keep-sorted end
}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_CONSTANTS_H_
