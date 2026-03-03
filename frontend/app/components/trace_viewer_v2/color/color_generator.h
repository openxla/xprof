#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COLOR_GENERATOR_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COLOR_GENERATOR_H_

#include "absl/strings/string_view.h"
#include "imgui.h"

namespace traceviewer {

// Generates a color for a given string ID.
ImU32 GetColorForId(absl::string_view id);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_COLOR_GENERATOR_H_
