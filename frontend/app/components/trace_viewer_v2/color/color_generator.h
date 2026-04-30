#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COLOR_GENERATOR_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COLOR_GENERATOR_H_

#include <optional>

#include "absl/types/span.h"
#include "absl/strings/string_view.h"
#include "imgui.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"

namespace traceviewer {

// Generates a color for a given string ID.
ImU32 GetColorForId(absl::string_view id,
                    absl::Span<const ImU32> event_colors);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COLOR_GENERATOR_H_
