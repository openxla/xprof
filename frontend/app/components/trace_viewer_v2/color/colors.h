#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COLORS_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COLORS_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "imgui.h"

namespace traceviewer {

// ImGui uses a 0xAABBGGRR color format. To ensure the color preview in some
// IDEs matches the actual rendered color, the hex values are defined with
// reversed relative to a standard #RRGGBB hex representation.
//
// For example, to get the color #C597FF (RR=C5, GG=97, BB=FF), the
// ImU32 value is defined as 0xFF_FF97C5 (AA=FF, BB=FF, GG=97, RR=C5).

// Static palette:
// go/keep-sorted start
// Blue 70: #7BAAF7
inline constexpr ImU32 kBlue70 = 0xFFF7AA7B;
// Blue 80: #A1C9FF
inline constexpr ImU32 kBlue80 = 0xFFFFC9A1;
// Green 80: #80DA88
inline constexpr ImU32 kGreen80 = 0xFF88DA80;
// Purple 70: #C597FF
inline constexpr ImU32 kPurple70 = 0xFFFF97C5;
// Yellow 90: #FFE07C
inline constexpr ImU32 kYellow90 = 0xFF7CE0FF;
// go/keep-sorted end

// Baseline palette:
// go/keep-sorted start
// "inverse on surface" #F2F2F2
inline constexpr ImU32 kInverseOnSurfaceColor = 0xFFF2F2F2;
// "on secondary" #004A77
inline constexpr ImU32 kOnSecondaryFixedVariantColor = 0xFF774A00;
// "on surface" #1F1F1F
inline constexpr ImU32 kOnSurfaceColor = 0xFF1F1F1F;
// "outline" #747775
inline constexpr ImU32 kOutlineColor = 0xFF757774;
// "outline variant" #C4C7C5
inline constexpr ImU32 kOutlineVariantColor = 0xFFC5C7C4;
// Color schemes/Secondary colors/Secondary container (#C2E7FF)
inline constexpr ImU32 kSecondaryContainerColor = 0xFFFFE7C2;
// go/keep-sorted end

// palette for flow categories:
// kBlue80 and kGreen80 are also used for flow categories.
// go/keep-sorted start
// Cyan80: #A1E4F2
inline constexpr ImU32 kCyan80 = 0xFFF2E4A1;
// Orange80: #FDC69C
inline constexpr ImU32 kOrange80 = 0xFF9CC6FD;
// Purple80: #D7AEFB
inline constexpr ImU32 kPurple80 = 0xFFFBAED7;
// Red80: #F6AEA9
inline constexpr ImU32 kRed80 = 0xFFA9AEF6;
// Yellow80: #FDE293
inline constexpr ImU32 kYellow80 = 0xFF93E2FD;
// go/keep-sorted end

constexpr size_t kMaxColors = 24;

// Calculate relative luminance of a color.
float CalculateLuminance(ImU32 color);

// Calculate contrast ratio between two colors.
float CalculateContrastRatio(ImU32 color1, ImU32 color2);

// Get the best text color (dark or light) for a given background color to meet
// contrast requirements.
ImU32 GetTextColorForContrast(ImU32 background_color, ImU32 on_surface_color,
                              ImU32 inverse_on_surface_color);

class ColorPalette {
 public:
  struct Preset {
    ImU32 background;
    ImU32 foreground;
    ImU32 midtone;
    ImU32 flame_header;
    ImU32 collapsed_header;
    ImU32 expanded_header;
    ImU32 subtitle;
    ImU32 ruler_text;
    ImU32 ruler_line;
    ImU32 selection;
    ImU32 on_surface;
    ImU32 inverse_on_surface;
    absl::InlinedVector<ImU32, kMaxColors> trace_colors;
    absl::InlinedVector<ImU32, kMaxColors> flow_colors;
    static Preset Default() {
      return Preset{
          .background = 0xFFFFFFFF,
          .foreground = 0xFF000000,
          .midtone = kInverseOnSurfaceColor,
          .flame_header = kBlue70,
          .collapsed_header = kInverseOnSurfaceColor,
          .expanded_header = kSecondaryContainerColor,
          .subtitle = kOnSecondaryFixedVariantColor,
          .ruler_text = kOutlineColor,
          .ruler_line = kOutlineVariantColor,
          .selection = 0xFFFFC9A1,
          .on_surface = kOnSurfaceColor,
          .inverse_on_surface = kInverseOnSurfaceColor,
          .trace_colors = {kPurple70, kGreen80, kBlue80, kYellow90},
          .flow_colors = {kCyan80, kOrange80, kPurple80, kRed80, kYellow80}};
    }
  };
  enum Key : uint16_t {
    kUnknown = 0,
    kBackground,
    kForeground,
    kMidtone,
    kFlameHeader,
    kCollapsedHeader,
    kExpandedHeader,
    kSubtitle,
    kRulerText,
    kRulerLine,
    kSelection,
    kOnSurface,
    kInverseOnSurface
  };

  explicit ColorPalette(const Preset& preset_palette);

  static ColorPalette Default() {
    return ColorPalette(Preset::Default());
  }

  ColorPalette& operator=(const Preset& preset);

  absl::Status FromPreset(absl::string_view palette_name);

  absl::StatusOr<ImU32> GetColor(Key key) const;
  absl::StatusOr<ImU32> GetTraceColor(int index) const;
  absl::Span<const ImU32> GetTraceColors() const { return trace_colors_; }
  absl::StatusOr<ImU32> GetFlowColor(int index) const;
  absl::Span<const ImU32> GetFlowColors() const { return flow_colors_; }

  absl::Status SetColor(Key key, ImU32 color);

  absl::Status SetTraceColor(int index, ImU32 color);
  absl::Status PushTraceColor(ImU32 color);
  absl::Status PopTraceColor();

  absl::Status SetFlowColor(int index, ImU32 color);
  absl::Status PushFlowColor(ImU32 color);
  absl::Status PopFlowColor();

  uint64_t GetVersion() const { return version_; }
  uint64_t GetTraceVersion() const { return trace_version_; }
  uint64_t GetFlowVersion() const { return flow_version_; }

 private:
  uint64_t version_ = 0;
  uint64_t trace_version_ = 0;
  uint64_t flow_version_ = 0;
  absl::flat_hash_map<Key, ImU32> colors_;
  absl::InlinedVector<ImU32, kMaxColors> trace_colors_;
  absl::InlinedVector<ImU32, kMaxColors> flow_colors_;
};



}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COLORS_H_
