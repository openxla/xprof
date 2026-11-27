#include "frontend/app/components/trace_viewer_v2/color/colors.h"

#include <algorithm>
#include <cstdint>

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
  colors_[Key::kOnSurface] = preset.on_surface;
  colors_[Key::kInverseOnSurface] = preset.inverse_on_surface;
  colors_[Key::kHloProducerArrow] = preset.hlo_producer_arrow;
  colors_[Key::kHloConsumerArrow] = preset.hlo_consumer_arrow;
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
  current_preset_name_ = palette_name;
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

absl::Status ColorPalette::SetTraceColors(absl::Span<const ImU32> colors) {
  if (colors.empty()) {
    return absl::InvalidArgumentError("At least one trace color is required");
  }
  if (colors.size() > kMaxColors) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot set more than ", kMaxColors, " trace colors"));
  }
  trace_colors_.assign(colors.begin(), colors.end());
  trace_version_++;
  version_++;
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

namespace {

// 256-entry lookup table mapping 8-bit sRGB channel values (0..255) to linear
// float components per the IEC 61966-2-1 sRGB transfer function.
constexpr float kSrgbToLinear[256] = {
    0.00000000f, 0.00030353f, 0.00060705f, 0.00091058f, 0.00121411f,
    0.00151763f, 0.00182116f, 0.00212469f, 0.00242822f, 0.00273174f,
    0.00303527f, 0.00334654f, 0.00367651f, 0.00402472f, 0.00439144f,
    0.00477695f, 0.00518152f, 0.00560539f, 0.00604883f, 0.00651209f,
    0.00699541f, 0.00749903f, 0.00802319f, 0.00856813f, 0.00913406f,
    0.00972122f, 0.01032982f, 0.01096009f, 0.01161225f, 0.01228649f,
    0.01298303f, 0.01370208f, 0.01444384f, 0.01520851f, 0.01599629f,
    0.01680738f, 0.01764195f, 0.01850022f, 0.01938236f, 0.02028856f,
    0.02121901f, 0.02217388f, 0.02315337f, 0.02415763f, 0.02518686f,
    0.02624122f, 0.02732089f, 0.02842604f, 0.02955683f, 0.03071344f,
    0.03189603f, 0.03310477f, 0.03433981f, 0.03560131f, 0.03688945f,
    0.03820437f, 0.03954624f, 0.04091520f, 0.04231141f, 0.04373503f,
    0.04518620f, 0.04666509f, 0.04817182f, 0.04970657f, 0.05126946f,
    0.05286065f, 0.05448028f, 0.05612849f, 0.05780543f, 0.05951124f,
    0.06124605f, 0.06301002f, 0.06480327f, 0.06662594f, 0.06847817f,
    0.07036010f, 0.07227185f, 0.07421357f, 0.07618538f, 0.07818742f,
    0.08021982f, 0.08228271f, 0.08437621f, 0.08650046f, 0.08865559f,
    0.09084171f, 0.09305896f, 0.09530747f, 0.09758735f, 0.09989873f,
    0.10224173f, 0.10461648f, 0.10702310f, 0.10946171f, 0.11193243f,
    0.11443537f, 0.11697067f, 0.11953843f, 0.12213877f, 0.12477182f,
    0.12743768f, 0.13013648f, 0.13286832f, 0.13563333f, 0.13843162f,
    0.14126329f, 0.14412847f, 0.14702727f, 0.14995979f, 0.15292615f,
    0.15592646f, 0.15896084f, 0.16202938f, 0.16513219f, 0.16826940f,
    0.17144110f, 0.17464740f, 0.17788842f, 0.18116424f, 0.18447499f,
    0.18782077f, 0.19120168f, 0.19461783f, 0.19806932f, 0.20155625f,
    0.20507874f, 0.20863687f, 0.21223076f, 0.21586050f, 0.21952620f,
    0.22322796f, 0.22696587f, 0.23074005f, 0.23455058f, 0.23839757f,
    0.24228112f, 0.24620133f, 0.25015828f, 0.25415209f, 0.25818285f,
    0.26225066f, 0.26635560f, 0.27049779f, 0.27467731f, 0.27889426f,
    0.28314874f, 0.28744084f, 0.29177065f, 0.29613827f, 0.30054379f,
    0.30498731f, 0.30946892f, 0.31398871f, 0.31854678f, 0.32314321f,
    0.32777810f, 0.33245154f, 0.33716362f, 0.34191442f, 0.34670406f,
    0.35153260f, 0.35640014f, 0.36130678f, 0.36625260f, 0.37123768f,
    0.37626212f, 0.38132601f, 0.38642943f, 0.39157248f, 0.39675523f,
    0.40197778f, 0.40724021f, 0.41254261f, 0.41788507f, 0.42326767f,
    0.42869050f, 0.43415364f, 0.43965717f, 0.44520119f, 0.45078578f,
    0.45641102f, 0.46207700f, 0.46778380f, 0.47353150f, 0.47932018f,
    0.48514994f, 0.49102085f, 0.49693300f, 0.50288646f, 0.50888132f,
    0.51491767f, 0.52099557f, 0.52711513f, 0.53327640f, 0.53947949f,
    0.54572446f, 0.55201140f, 0.55834039f, 0.56471151f, 0.57112483f,
    0.57758044f, 0.58407842f, 0.59061884f, 0.59720179f, 0.60382734f,
    0.61049557f, 0.61720656f, 0.62396039f, 0.63075714f, 0.63759687f,
    0.64447968f, 0.65140564f, 0.65837482f, 0.66538730f, 0.67244316f,
    0.67954247f, 0.68668531f, 0.69387176f, 0.70110189f, 0.70837578f,
    0.71569350f, 0.72305513f, 0.73046074f, 0.73791041f, 0.74540421f,
    0.75294222f, 0.76052450f, 0.76815115f, 0.77582222f, 0.78353779f,
    0.79129794f, 0.79910274f, 0.80695226f, 0.81484657f, 0.82278575f,
    0.83076988f, 0.83879901f, 0.84687323f, 0.85499261f, 0.86315721f,
    0.87136712f, 0.87962240f, 0.88792312f, 0.89626935f, 0.90466117f,
    0.91309865f, 0.92158186f, 0.93011086f, 0.93868573f, 0.94730654f,
    0.95597335f, 0.96468625f, 0.97344529f, 0.98225055f, 0.99110210f,
    1.00000000f,
};

}  // namespace

float CalculateLuminance(ImU32 color) {
  const uint8_t r = static_cast<uint8_t>(color & 0xFF);
  const uint8_t g = static_cast<uint8_t>((color >> 8) & 0xFF);
  const uint8_t b = static_cast<uint8_t>((color >> 16) & 0xFF);

  return 0.2126f * kSrgbToLinear[r] + 0.7152f * kSrgbToLinear[g] +
         0.0722f * kSrgbToLinear[b];
}

float CalculateContrastRatio(ImU32 color1, ImU32 color2) {
  float l1 = CalculateLuminance(color1);
  float l2 = CalculateLuminance(color2);
  return (std::max(l1, l2) + 0.05f) / (std::min(l1, l2) + 0.05f);
}

ImU32 GetTextColorForContrast(ImU32 background_color, ImU32 on_surface_color,
                              ImU32 inverse_on_surface_color) {
  float contrast_on_surface =
      CalculateContrastRatio(background_color, on_surface_color);
  return contrast_on_surface >= 4.5f ? on_surface_color
                                     : inverse_on_surface_color;
}

}  // namespace traceviewer
