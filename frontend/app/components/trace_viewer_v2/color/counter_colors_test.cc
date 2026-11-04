#include "frontend/app/components/trace_viewer_v2/color/counter_colors.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "imgui.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/color/palettes.h"

namespace traceviewer {
namespace {

using ::testing::Eq;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::Lt;
using ::testing::Optional;

constexpr absl::string_view kUtilizationTracks[] = {
    "MXU utilization", "XLU utilization", "SALU utilization",
    "VPU utilization", "EUP utilization", "VLD utilization",
    "VST utilization", "DMA utilization", "Other utilization",
};

constexpr absl::string_view kMemoryTracks[] = {
    "Predicate Reg", "Scalar Reg",      "VMEM Scoped",
    "VMEM Spill",    "Vector Mask Reg", "Vector Reg",
};

constexpr absl::Span<const absl::string_view> kSections[] = {
    kUtilizationTracks,
    kMemoryTracks,
};

constexpr ImU32 kWhite = 0xFFFFFFFF;
// Dracula's background, #282A36.
constexpr ImU32 kDraculaBackground = 0xFF362A28;

TEST(GetKernelCounterTrackBaseColorTest, UtilizationTracksUseTheUnitColor) {
  // #7C4DFF and #1A73E8, the MXU and VPU colors of the Kernel Viewer.
  EXPECT_THAT(GetKernelCounterTrackBaseColor("MXU utilization"),
              Optional(Eq(0xFFFF4D7Cu)));
  EXPECT_THAT(GetKernelCounterTrackBaseColor("VPU utilization"),
              Optional(Eq(0xFFE8731Au)));
  for (absl::string_view track : kUtilizationTracks) {
    EXPECT_TRUE(GetKernelCounterTrackBaseColor(track).has_value()) << track;
  }
}

TEST(GetKernelCounterTrackBaseColorTest, VmemSpillIsRed) {
  // #D93025.
  EXPECT_THAT(GetKernelCounterTrackBaseColor("VMEM Spill"),
              Optional(Eq(0xFF2530D9u)));
}

TEST(GetKernelCounterTrackBaseColorTest, MemoryTracksHaveDistinctColors) {
  absl::flat_hash_set<ImU32> colors;
  for (absl::string_view track : kMemoryTracks) {
    const std::optional<ImU32> color = GetKernelCounterTrackBaseColor(track);
    ASSERT_TRUE(color.has_value()) << track;
    EXPECT_TRUE(colors.insert(*color).second) << track;
  }
}

TEST(GetKernelCounterTrackBaseColorTest, OtherTracksHaveNoColor) {
  EXPECT_EQ(GetKernelCounterTrackBaseColor("HBM Usage"), std::nullopt);
  EXPECT_EQ(GetKernelCounterTrackBaseColor("MXU"), std::nullopt);
  EXPECT_EQ(GetKernelCounterTrackBaseColor("utilization"), std::nullopt);
  EXPECT_EQ(GetKernelCounterTrackBaseColor(" utilization"), std::nullopt);
  EXPECT_EQ(GetKernelCounterTrackBaseColor("MXU Utilization"), std::nullopt);
  EXPECT_EQ(GetKernelCounterTrackBaseColor("TPU MXU utilization"),
            std::nullopt);
  EXPECT_EQ(GetKernelCounterTrackBaseColor("vector reg"), std::nullopt);
}

TEST(MakeCounterTrackColorsTest, KeepsBaseThatContrastsWithLightBackground) {
  // #7C4DFF on white is 4.8:1.
  EXPECT_EQ(MakeCounterTrackColors(0xFFFF4D7C, kWhite).stroke, 0xFFFF4D7Cu);
}

TEST(MakeCounterTrackColorsTest, DarkensLowContrastBaseOnLightBackground) {
  // #F29900 on white is only 2.3:1.
  const ImU32 stroke = MakeCounterTrackColors(0xFF0099F2, kWhite).stroke;

  EXPECT_THAT(CalculateContrastRatio(stroke, kWhite),
              Ge(kCounterStrokeMinContrast));
  EXPECT_THAT(CalculateLuminance(stroke), Lt(CalculateLuminance(0xFF0099F2)));
}

TEST(MakeCounterTrackColorsTest, LightensBaseOnDarkBackground) {
  // #1A73E8.
  const ImU32 stroke =
      MakeCounterTrackColors(0xFFE8731A, kDraculaBackground).stroke;

  EXPECT_THAT(CalculateLuminance(stroke), Gt(CalculateLuminance(0xFFE8731A)));
  EXPECT_THAT(CalculateContrastRatio(stroke, kDraculaBackground),
              Ge(kCounterStrokeMinContrast));
}

TEST(MakeCounterTrackColorsTest, IgnoresAlpha) {
  EXPECT_EQ(MakeCounterTrackColors(0x80E8731A, 0x00FFFFFF),
            MakeCounterTrackColors(0xFFE8731A, kWhite));
}

TEST(MakeCounterTrackColorsTest, KernelTracksContrastWithEveryPalette) {
  for (const auto& [palette_name, preset] : kPresetPalettes) {
    for (absl::Span<const absl::string_view> section : kSections) {
      for (absl::string_view track : section) {
        SCOPED_TRACE(absl::StrCat(palette_name, " / ", track));
        const std::optional<ImU32> base = GetKernelCounterTrackBaseColor(track);
        ASSERT_TRUE(base.has_value());
        const CounterTrackColors colors =
            MakeCounterTrackColors(*base, preset.background);

        const float stroke_contrast =
            CalculateContrastRatio(colors.stroke, preset.background);
        EXPECT_THAT(stroke_contrast, Ge(kCounterStrokeMinContrast));
        // The fill fades from the top of the track towards the background, and
        // never outweighs the outline.
        const float top_contrast =
            CalculateContrastRatio(colors.fill_top, preset.background);
        EXPECT_THAT(top_contrast, Lt(stroke_contrast));
        EXPECT_THAT(
            CalculateContrastRatio(colors.fill_bottom, preset.background),
            Lt(top_contrast));
      }
    }
  }
}

TEST(MakeCounterTrackColorsTest, KernelTracksInASectionStayDistinct) {
  for (const auto& [palette_name, preset] : kPresetPalettes) {
    for (absl::Span<const absl::string_view> section : kSections) {
      absl::flat_hash_set<ImU32> strokes;
      for (absl::string_view track : section) {
        const std::optional<ImU32> base = GetKernelCounterTrackBaseColor(track);
        ASSERT_TRUE(base.has_value()) << track;
        EXPECT_TRUE(
            strokes
                .insert(MakeCounterTrackColors(*base, preset.background).stroke)
                .second)
            << palette_name << " / " << track;
      }
    }
  }
}

}  // namespace
}  // namespace traceviewer
