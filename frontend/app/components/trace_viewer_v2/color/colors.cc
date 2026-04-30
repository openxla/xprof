#include "frontend/app/components/trace_viewer_v2/color/colors.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "imgui.h"

namespace traceviewer {

ColorPalette::ColorPalette(const ColorPalette::Preset& preset_palette) {
  colors_[Key::kBackground] = preset_palette.background;
  colors_[Key::kForeground] = preset_palette.foreground;
  colors_[Key::kMidtone] = preset_palette.midtone;
  colors_[Key::kFlameHeader] = preset_palette.flame_header;
  colors_[Key::kCollapsedHeader] = preset_palette.collapsed_header;
  colors_[Key::kExpandedHeader] = preset_palette.expanded_header;
  colors_[Key::kSubtitle] = preset_palette.subtitle;
  colors_[Key::kRulerText] = preset_palette.ruler_text;
  colors_[Key::kRulerLine] = preset_palette.ruler_line;
  colors_[Key::kSelection] = preset_palette.selection;
  trace_colors_ = preset_palette.trace_colors;
  flow_colors_ = preset_palette.flow_colors;
}

absl::StatusOr<ImU32> ColorPalette::GetColor(ColorPalette::Key key) const {
  auto it = colors_.find(key);
  if (it == colors_.end()) {
    return absl::NotFoundError(absl::StrCat("Color was incorrectly set"));
  }
  return it->second;
}

absl::StatusOr<ImU32> ColorPalette::GetTraceColor(int index) const {
  if (index < 0 || index >= trace_colors_.size()) {
    return absl::OutOfRangeError(
        absl::StrCat("Trace color index out of bounds: ", index,
                     " (size: ", trace_colors_.size(), ")"));
  }
  return trace_colors_[index];
}

absl::StatusOr<ImU32> ColorPalette::GetFlowColor(int index) const {
  if (index < 0 || index >= flow_colors_.size()) {
    return absl::OutOfRangeError(
        absl::StrCat("Flow color index out of bounds: ", index,
                     " (size: ", flow_colors_.size(), ")"));
  }
  return flow_colors_[index];
}

absl::Status ColorPalette::SetColor(ColorPalette::Key key, ImU32 color) {
  if (!colors_.contains(key)) {
    return absl::InvalidArgumentError(absl::StrCat("Color not found: ", key));
  }
  colors_[key] = color;
  version_++;
  return absl::OkStatus();
}

absl::Status ColorPalette::SetTraceColor(int index, ImU32 color) {
  if (index < 0 || index >= trace_colors_.size()) {
    return absl::OutOfRangeError(
        absl::StrCat("Trace color index out of bounds: ", index,
                     " (size: ", trace_colors_.size(), ")"));
  }
  trace_colors_[index] = color;
  trace_version_++;
  return absl::OkStatus();
}

absl::Status ColorPalette::PushTraceColor(ImU32 color) {
  if (trace_colors_.size() >= kMaxColors) {
    return absl::ResourceExhaustedError(
        absl::StrCat("Cannot add more than ", kMaxColors, " trace colors"));
  }
  trace_colors_.push_back(color);
  trace_version_++;
  return absl::OkStatus();
}

absl::Status ColorPalette::PopTraceColor() {
  if (trace_colors_.empty()) {
    return absl::FailedPreconditionError("No trace colors to remove");
  }
  if (trace_colors_.size() <= 1) {
    return absl::FailedPreconditionError("Cannot remove the last trace color");
  }
  trace_colors_.pop_back();
  trace_version_++;
  return absl::OkStatus();
}

absl::Status ColorPalette::SetFlowColor(int index, ImU32 color) {
  if (index < 0 || index >= flow_colors_.size()) {
    return absl::OutOfRangeError(
        absl::StrCat("Flow color index out of bounds: ", index,
                     " (size: ", flow_colors_.size(), ")"));
  }
  flow_colors_[index] = color;
  flow_version_++;
  return absl::OkStatus();
}

absl::Status ColorPalette::PushFlowColor(ImU32 color) {
  if (flow_colors_.size() >= kMaxColors) {
    return absl::ResourceExhaustedError(
        absl::StrCat("Cannot add more than ", kMaxColors, " flow colors"));
  }
  flow_colors_.push_back(color);
  flow_version_++;
  return absl::OkStatus();
}

absl::Status ColorPalette::PopFlowColor() {
  if (flow_colors_.empty()) {
    return absl::FailedPreconditionError("No flow colors to remove");
  }
  if (flow_colors_.size() <= 1) {
    return absl::FailedPreconditionError("Cannot remove the last flow color");
  }
  flow_colors_.pop_back();
  flow_version_++;
  return absl::OkStatus();
}

}  // namespace traceviewer
