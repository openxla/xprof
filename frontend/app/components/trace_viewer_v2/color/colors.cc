#include "frontend/app/components/trace_viewer_v2/color/colors.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "imgui.h"
#include "frontend/app/components/trace_viewer_v2/color/palettes.h"

namespace traceviewer {

ColorPalette& ColorPalette::operator=(const ColorPalette::Preset& preset) {
  colors_[Key::kBackground] = preset.background;
  colors_[Key::kForeground] = preset.foreground;
  colors_[Key::kMidtone] = preset.midtone;
  colors_[Key::kFlameHeader] = preset.flame_header;
  colors_[Key::kCollapsedHeader] = preset.collapsed_header;
  colors_[Key::kExpandedHeader] = preset.expanded_header;
  colors_[Key::kSubtitle] = preset.subtitle;
  colors_[Key::kRulerText] = preset.ruler_text;
  colors_[Key::kRulerLine] = preset.ruler_line;
  colors_[Key::kSelection] = preset.selection;
  trace_colors_ = preset.trace_colors;
  flow_colors_ = preset.flow_colors;
  version_++;
  trace_version_++;
  flow_version_++;
  return *this;
}

ColorPalette::ColorPalette(const ColorPalette::Preset& preset_palette)
    : version_(0), trace_version_(0), flow_version_(0) {
  *this = preset_palette;
}

absl::Status ColorPalette::FromPreset(absl::string_view palette_name) {
  auto it = kPresetPalettes.find(palette_name);
  if (it == kPresetPalettes.end()) {
    return absl::NotFoundError(
        absl::StrCat("Palette not found: ", palette_name));
  }
  *this = it->second;
  return absl::OkStatus();
}

absl::StatusOr<ImU32> ColorPalette::GetColor(ColorPalette::Key key) const {
  auto it = colors_.find(key);
  if (it == colors_.end()) {
    return absl::NotFoundError(absl::StrCat("Color was incorrectly set"));
  }
  return it->second;
}

absl::StatusOr<ImU32> ColorPalette::GetTraceColor(int index) const {
  if (trace_colors_.empty()) {
    return absl::FailedPreconditionError("No trace colors available");
  }
  if (index < 0) {
    return absl::InvalidArgumentError("Trace color index cannot be negative");
  }
  return trace_colors_[index % trace_colors_.size()];
}

absl::StatusOr<ImU32> ColorPalette::GetFlowColor(int index) const {
  if (flow_colors_.empty()) {
    return absl::FailedPreconditionError("No flow colors available");
  }
  if (index < 0) {
    return absl::InvalidArgumentError("Flow color index cannot be negative");
  }
  return flow_colors_[index % flow_colors_.size()];
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
