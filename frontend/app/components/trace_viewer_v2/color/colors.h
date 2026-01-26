#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COLORS_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COLORS_H_

#include "third_party/dear_imgui/imgui.h"

namespace traceviewer {

// ImGui uses a 0xAABBGGRR color format. To ensure the color preview in some
// IDEs matches the actual rendered color, the hex values are defined with
// reversed relative to a standard #RRGGBB hex representation.
//
// For example, to get the color #C597FF (RR=C5, GG=97, BB=FF), the
// ImU32 value is defined as 0xFF_FF97C5 (AA=FF, BB=FF, GG=97, RR=C5).

// Static palette:
// go/keep-sorted start
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
// "outline" #747775
inline constexpr ImU32 kOutlineColor = 0xFF757774;
// "outline variant" #C4C7C5
inline constexpr ImU32 kOutlineVariantColor = 0xFFC5C7C4;
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

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COLORS_H_
