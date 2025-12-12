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

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COLORS_H_
