#include "frontend/app/components/trace_viewer_v2/color/counter_colors.h"

#include <optional>

#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"

namespace traceviewer {

namespace {

// A track (or unit) name and its color. As in colors.h, colors are 0xAABBGGRR
// and each comment gives the #RRGGBB equivalent.
struct NamedColor {
  absl::string_view name;
  ImU32 color;
};

// Functional unit colors, matching the Kernel Viewer's details panel, so a
// unit's utilization chart matches its bars there.
constexpr NamedColor kUnitColors[] = {
    {"MXU", 0xFFFF4D7C},    // #7C4DFF
    {"VPU", 0xFFE8731A},    // #1A73E8
    {"TVPU", 0xFFE8731A},   // #1A73E8
    {"XLU", 0xFF0099F2},    // #F29900
    {"XPU", 0xFF0099F2},    // #F29900
    {"EUP", 0xFFAFA412},    // #12A4AF
    {"SALU", 0xFF3E8E1E},   // #1E8E3E
    {"VLD", 0xFFBFA300},    // #00A3BF
    {"VST", 0xFF8418D0},    // #D01884
    {"SGE", 0xFFD18802},    // #0288D1
    {"DMA", 0xFF0A71E8},    // #E8710A
    {"Other", 0xFF68635F},  // #5F6368
};

// Memory track colors. Neighboring tracks (sorted by name) get contrasting
// hues, and spills are red shades because they signal register pressure.
constexpr NamedColor kMemoryColors[] = {
    {"Descriptor Reg", 0xFFD18802},   // #0288D1 cerulean
    {"Predicate Reg", 0xFF0074E3},    // #E37400 amber
    {"SCS Smem Spill", 0xFF0C36BF},   // #BF360C deep orange-red
    {"Scalar Reg", 0xFF388018},       // #188038 green
    {"Tile Smem Spill", 0xFF5B18C2},  // #C2185B crimson
    {"TileSpmem Spill", 0xFF2530D9},  // #D93025 red
    {"VMEM Scoped", 0xFFE63493},      // #9334E6 purple
    {"VMEM Spill", 0xFF2530D9},       // #D93025 red
    {"Vector Mask Reg", 0xFF837B00},  // #007B83 teal
    {"Vector Reg", 0xFFE8731A},       // #1A73E8 blue
};

constexpr absl::string_view kUtilizationSuffix = " utilization";

// How far `base` is lifted towards white on dark backgrounds.
constexpr float kDarkBackgroundTint = 0.3f;
// Each step moves the stroke this fraction of the way to black or white, until
// it reaches the minimum contrast.
constexpr float kContrastStep = 0.1f;
constexpr int kMaxContrastSteps = 20;
// The share of the stroke color in the fill at the top of the track and at its
// baseline. Dark backgrounds need a little more color to read as a tint.
constexpr float kLightFillTopStrength = 0.45f;
constexpr float kLightFillBottomStrength = 0.1f;
constexpr float kDarkFillTopStrength = 0.55f;
constexpr float kDarkFillBottomStrength = 0.15f;

// Linearly interpolates between two colors, channel by channel: `t` = 0
// returns `from` and `t` = 1 returns `to`.
ImU32 Mix(ImU32 from, ImU32 to, float t) {
  return ImGui::ColorConvertFloat4ToU32(
      ImLerp(ImGui::ColorConvertU32ToFloat4(from),
             ImGui::ColorConvertU32ToFloat4(to), t));
}

}  // namespace

std::optional<ImU32> GetKernelCounterTrackBaseColor(
    absl::string_view track_name) {
  for (const NamedColor& memory : kMemoryColors) {
    if (track_name == memory.name) return memory.color;
  }
  if (absl::ConsumeSuffix(&track_name, kUtilizationSuffix)) {
    for (const NamedColor& unit : kUnitColors) {
      if (track_name == unit.name) return unit.color;
    }
  }
  return std::nullopt;
}

CounterTrackColors MakeCounterTrackColors(ImU32 base, ImU32 background) {
  base |= IM_COL32_A_MASK;
  background |= IM_COL32_A_MASK;
  // A background is dark if white contrasts more with it than black.
  const bool dark_background =
      CalculateContrastRatio(background, IM_COL32_WHITE) >
      CalculateContrastRatio(background, IM_COL32_BLACK);
  const ImU32 away_from_background =
      dark_background ? IM_COL32_WHITE : IM_COL32_BLACK;

  ImU32 stroke =
      dark_background ? Mix(base, IM_COL32_WHITE, kDarkBackgroundTint) : base;
  for (int step = 0;
       step < kMaxContrastSteps &&
       CalculateContrastRatio(stroke, background) < kCounterStrokeMinContrast;
       ++step) {
    stroke = Mix(stroke, away_from_background, kContrastStep);
  }

  return CounterTrackColors{
      .stroke = stroke,
      .fill_top =
          Mix(background, stroke,
              dark_background ? kDarkFillTopStrength : kLightFillTopStrength),
      .fill_bottom = Mix(
          background, stroke,
          dark_background ? kDarkFillBottomStrength : kLightFillBottomStrength),
  };
}

}  // namespace traceviewer
