#include "frontend/app/components/trace_viewer_v2/application.h"

#include <dirent.h>
#include <emscripten/bind.h>
#include <emscripten/em_asm.h>
#include <emscripten/em_macros.h>
#include <emscripten/em_types.h>
#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/val.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "imgui.h"
#include "frontend/app/components/trace_viewer_v2/animation.h"
#include "frontend/app/components/trace_viewer_v2/canvas_state.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/event_data.h"
#include "frontend/app/components/trace_viewer_v2/event_manager.h"
#include "frontend/app/components/trace_viewer_v2/fonts/fonts.h"
#include "frontend/app/components/trace_viewer_v2/input_handler.h"
#include "frontend/app/components/trace_viewer_v2/scheduler.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/webgpu_render_platform.h"

namespace traceviewer {

namespace {

const char* const kWindowTarget = EMSCRIPTEN_EVENT_TARGET_WINDOW;
const char* const kCanvasTarget = "#canvas";
constexpr float kScrollbarSize = 10.0f;

EM_JS(bool, GetFeatureFlagFromJS, (const char* name), {
  if (window.getFeatureFlag) {
    return window.getFeatureFlag(UTF8ToString(name));
  }
  return false;
});

EM_BOOL OnResize(int eventType, const EmscriptenUiEvent* uiEvent,
                 void* userData) {
  Application::Instance().RequestRedraw();
  return EM_FALSE;
}

EMSCRIPTEN_KEEPALIVE void SetPalette(const std::string& palette_name) {
  ColorPalette& palette = Application::Instance().GetPalette();

  // If the new palette is the same as the current palette is, do not update it
  // again.
  if (palette.GetCurrentPresetName() == palette_name) {
    return;
  }

  absl::Status status = palette.FromPreset(palette_name);
  if (!status.ok()) {
    palette = ColorPalette::Preset::Default();
  }

  Application::Instance().RequestRedraw();
}

EMSCRIPTEN_KEEPALIVE void SetColor(ColorPalette::Key key, ImU32 color) {
  absl::Status status =
      Application::Instance().GetPalette().SetColor(key, color);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to set color: " << status;
  }
  Application::Instance().RequestRedraw();
}

EMSCRIPTEN_BINDINGS(traceviewer) {
  emscripten::function("SetPalette", &SetPalette);
  emscripten::function("SetColor", &SetColor);
}

EMSCRIPTEN_BINDINGS(colors) {
  emscripten::enum_<ColorPalette::Key>("ColorPaletteKey")
      .value("kBackground", ColorPalette::Key::kBackground)
      .value("kForeground", ColorPalette::Key::kForeground)
      .value("kMidtone", ColorPalette::Key::kMidtone)
      .value("kFlameHeader", ColorPalette::Key::kFlameHeader)
      .value("kCollapsedHeader", ColorPalette::Key::kCollapsedHeader)
      .value("kExpandedHeader", ColorPalette::Key::kExpandedHeader)
      .value("kSubtitle", ColorPalette::Key::kSubtitle)
      .value("kRulerText", ColorPalette::Key::kRulerText)
      .value("kRulerLine", ColorPalette::Key::kRulerLine)
      .value("kSelection", ColorPalette::Key::kSelection);
}

}  // namespace

// This function initializes the application, setting up the ImGui context,
// the WebGPU rendering platform, and the timeline component. This follows a
// typical pattern for game or rendering applications where resources are
// initialized before the main loop starts.
void Application::Initialize() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  // Get initial canvas state
  const CanvasState& initial_canvas_state = CanvasState::Current();

  // Load fonts, colors and styles.
  fonts::LoadFonts(initial_canvas_state.device_pixel_ratio());

  emscripten::val local_storage = emscripten::val::global("localStorage");
  std::string palette_name = "";
  if (local_storage.as<bool>()) {
    emscripten::val item = local_storage.call<emscripten::val>(
        "getItem", emscripten::val("trace_viewer_palette"));
    if (!item.isNull() && !item.isUndefined()) {
      palette_name = item.as<std::string>();
    }
  }
  absl::Status status = palette_.FromPreset(palette_name);
  if (!status.ok()) {
    palette_ = ColorPalette::Preset::Default();
    LOG(ERROR) << "Failed to load palette name: " << palette_name;
  }
  // TODO: b/450584482 - Add a dark theme for the timeline.
  ImGuiStyle& style = ImGui::GetStyle();
  style.ScrollbarSize = kScrollbarSize;
  absl::StatusOr<ImU32> color =
      palette_.GetColor(ColorPalette::Key::kBackground);
  style.Colors[ImGuiCol_WindowBg] =
      color.ok() ? ImGui::ColorConvertU32ToFloat4(*color)
                 : ImGui::ColorConvertU32ToFloat4(kWhiteColor);
  style.Colors[ImGuiCol_PopupBg] =
      color.ok() ? ImGui::ColorConvertU32ToFloat4(*color)
                 : ImGui::ColorConvertU32ToFloat4(kWhiteColor);
  color = palette_.GetColor(ColorPalette::Key::kForeground);
  style.Colors[ImGuiCol_Text] =
      color.ok() ? ImGui::ColorConvertU32ToFloat4(*color)
                 : ImGui::ColorConvertU32ToFloat4(kBlackColor);
  color = palette_.GetColor(ColorPalette::Key::kMidtone);
  style.Colors[ImGuiCol_Border] =
      color.ok() ? ImGui::ColorConvertU32ToFloat4(*color)
                 : ImGui::ColorConvertU32ToFloat4(kLightGrayColor);
  // Set the table border color to a medium gray. #666666
  // We only use this color for the vertical lines between track title and
  // framechart. Horizontal lines are rendered in timeline.
  style.Colors[ImGuiCol_TableBorderLight] = ImVec4(0.4f, 0.4f, 0.4f, 1.0f);

  // Initialize the platform
  platform_ = std::make_unique<WGPURenderPlatform>();
  platform_->Init(initial_canvas_state);
  timeline_ = std::make_unique<Timeline>(palette_);
  timeline_->set_snap_to_time_range_enabled(
      IsFeatureEnabled("snap_to_time_range"));
  timeline_->set_bookmarks_enabled(IsFeatureEnabled("bookmarks"));
  timeline_->set_event_callback(
      [](absl::string_view type, const EventData& event_data) {
        EventManager::Instance().DispatchEvent(type, event_data);
      });
  timeline_->set_redraw_callback([this]() { this->RequestRedraw(); });

  ImGuiIO& io = ImGui::GetIO();

  // Set the initial display size for ImGui.
  io.DisplayFramebufferScale = {1.0f, 1.0f};
  UpdateImGuiDisplaySize(initial_canvas_state);

  // Enable keyboard navigation for the window. This is required for the
  // timeline to handle keyboard events.
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  // Register key event handlers to the window.
  emscripten_set_keydown_callback(kWindowTarget, /*user_data=*/this,
                                  /*use_capture=*/true, HandleKeyDown);
  emscripten_set_keyup_callback(kWindowTarget, /*user_data=*/this,
                                /*use_capture=*/true, HandleKeyUp);

  // The canvas element is guaranteed to exist at this point, because
  // traceViewerV2Main() in main.ts ensures it before calling callMain(), which
  // then calls Initialize().
  // Register mouse event handlers to the canvas element.
  emscripten_set_mousemove_callback(kCanvasTarget, /*user_data=*/this,
                                    /*use_capture=*/true, HandleMouseMove);
  emscripten_set_mousedown_callback(kCanvasTarget, /*user_data=*/this,
                                    /*use_capture=*/true, HandleMouseDown);
  emscripten_set_mouseup_callback(kCanvasTarget, /*user_data=*/this,
                                  /*use_capture=*/true, HandleMouseUp);

  // Register wheel event handlers to the canvas element.
  emscripten_set_wheel_callback(kCanvasTarget, /*user_data=*/this,
                                /*use_capture=*/true, HandleWheel);
  emscripten_set_resize_callback(kWindowTarget, /*user_data=*/nullptr,
                                 /*use_capture=*/true, OnResize);

  Scheduler::Instance().SetMainLoopCallback([this]() { MainLoop(); });
}

void Application::Shutdown() {
  Scheduler::Instance().Reset();

  // Unregister event handlers. Passing nullptr unregisters the callback.
  emscripten_set_keydown_callback(kWindowTarget, /*user_data=*/nullptr,
                                  /*use_capture=*/true, nullptr);
  emscripten_set_keyup_callback(kWindowTarget, /*user_data=*/nullptr,
                                /*use_capture=*/true, nullptr);
  emscripten_set_mousemove_callback(kCanvasTarget, /*user_data=*/nullptr,
                                    /*use_capture=*/true, nullptr);
  emscripten_set_mousedown_callback(kCanvasTarget, /*user_data=*/nullptr,
                                    /*use_capture=*/true, nullptr);
  emscripten_set_mouseup_callback(kCanvasTarget, /*user_data=*/nullptr,
                                  /*use_capture=*/true, nullptr);
  emscripten_set_wheel_callback(kCanvasTarget, /*user_data=*/nullptr,
                                /*use_capture=*/true, nullptr);
  emscripten_set_resize_callback(kWindowTarget, /*user_data=*/nullptr,
                                 /*use_capture=*/true, nullptr);

  // Clean up and release memory.
  feature_flags_cache_.clear();
  timeline_.reset();
  platform_.reset();

  if (ImGui::GetCurrentContext()) {
    ImGui::DestroyContext();
  }
}

void Application::MainLoop() {
  ImGuiIO& io = ImGui::GetIO();
  io.DeltaTime = GetDeltaTime();

  Animation::UpdateAll(io.DeltaTime);

  Draw();
}

void Application::Draw() {
  UpdateApplicationColors();
  platform_->NewFrame();
  timeline_->Draw();
  UpdateMouseCursor();
  platform_->RenderFrame();

  if (Animation::HasActiveAnimations()) {
    RequestRedraw();
  }
}

void Application::UpdateMouseCursor() {
  ImGuiMouseCursor cursor = ImGui::GetMouseCursor();
  if (cursor == last_cursor_) return;
  last_cursor_ = cursor;

  EM_ASM(
      {
        // EM_ASM only supports JavaScript, not TypeScript.
        var cursor = $0;
        var cursor_css = 'default';
        if (cursor == 1)  // ImGuiMouseCursor_TextInput
          cursor_css = 'text';
        else if (cursor == 2)  // ImGuiMouseCursor_ResizeAll
          cursor_css = 'move';
        else if (cursor == 3)  // ImGuiMouseCursor_ResizeNS
          cursor_css = 'row-resize';
        else if (cursor == 4)  // ImGuiMouseCursor_ResizeEW
          cursor_css = 'col-resize';
        else if (cursor == 7)  // ImGuiMouseCursor_Hand
          cursor_css = 'pointer';

        // The canvas element is guaranteed to exist at this point, because
        // traceViewerV2Main() in main.ts ensures it before calling callMain(),
        // which then calls Initialize().
        document.getElementById('canvas').style.cursor = cursor_css;
      },
      cursor);
}

float Application::GetDeltaTime() {
  const absl::Time time_now = absl::Now();
  auto delta_time = absl::ToDoubleSeconds(time_now - last_frame_time_);
  last_frame_time_ = time_now;
  return std::min(0.1f, static_cast<float>(delta_time));
}

void Application::UpdateApplicationColors() {
  if (palette_.GetVersion() != palette_version_) {
    ImGuiStyle& style = ImGui::GetStyle();
    absl::StatusOr<ImU32> color =
        palette_.GetColor(ColorPalette::Key::kBackground);
    style.Colors[ImGuiCol_WindowBg] =
        color.ok() ? ImGui::ColorConvertU32ToFloat4(*color)
                   : ImGui::ColorConvertU32ToFloat4(kWhiteColor);
    style.Colors[ImGuiCol_PopupBg] =
        color.ok() ? ImGui::ColorConvertU32ToFloat4(*color)
                   : ImGui::ColorConvertU32ToFloat4(kWhiteColor);
    color = palette_.GetColor(ColorPalette::Key::kForeground);
    style.Colors[ImGuiCol_Text] =
        color.ok() ? ImGui::ColorConvertU32ToFloat4(*color)
                   : ImGui::ColorConvertU32ToFloat4(kBlackColor);
    color = palette_.GetColor(ColorPalette::Key::kMidtone);
    style.Colors[ImGuiCol_Border] =
        color.ok() ? ImGui::ColorConvertU32ToFloat4(*color)
                   : ImGui::ColorConvertU32ToFloat4(kLightGrayColor);
    // Set the table border color to a medium gray. #666666
    // We only use this color for the vertical line between track title and
    // framechart. Horizontal lines are rendered in timeline.
    style.Colors[ImGuiCol_TableBorderLight] = ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
    palette_version_ = palette_.GetVersion();
  }
}

void Application::UpdateImGuiDisplaySize(const CanvasState& canvas_state) {
  ImGuiIO& io = ImGui::GetIO();

  // io.DisplaySize tells ImGui the dimensions of the window in logical pixels
  // (or points). ImGui uses this for layout, window positioning, and input
  // handling in a DPI-independent manner.
  io.DisplaySize = canvas_state.logical_pixels();

  // io.DisplayFramebufferScale is the ratio between physical pixels and
  // logical pixels. ImGui uses this to scale its rendering output to match
  // the high-DPI framebuffer.
  const float dpr = canvas_state.device_pixel_ratio();
  io.DisplayFramebufferScale = {dpr, dpr};
}

void Application::SetVisibleFlowCategories(
    const emscripten::val& category_ids) {
  timeline_->SetVisibleFlowCategories(
      emscripten::vecFromJSArray<int>(category_ids));
}

bool Application::IsFeatureEnabled(const std::string& name) {
  auto it = feature_flags_cache_.find(name);
  if (it != feature_flags_cache_.end()) {
    return it->second;
  }

  bool enabled = GetFeatureFlagFromJS(name.c_str());
  feature_flags_cache_[name] = enabled;
  return enabled;
}

void Application::Resize(float dpr, int width, int height) {
  float old_dpr = CanvasState::Current().device_pixel_ratio();

  CanvasState::SetState(dpr, width, height);
  const CanvasState& canvas_state = CanvasState::Current();
  platform_->ResizeSurface(canvas_state);

  UpdateImGuiDisplaySize(canvas_state);

  if (dpr != old_dpr) {
    fonts::LoadFonts(canvas_state.device_pixel_ratio());
  }

  // Trigger an immediate synchronous redraw instead of waiting for the next
  // animation frame via RequestRedraw(). This ensures the canvas is updated in
  // the same event loop tick as the resize, preventing a blank or stretched
  // frame from being displayed (flicker).
  MainLoop();
}

}  // namespace traceviewer
