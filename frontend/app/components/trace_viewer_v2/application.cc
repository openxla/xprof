#include "xprof/frontend/app/components/trace_viewer_v2/application.h"

#include <dirent.h>
#include <emscripten/emscripten.h>  // NO_LINT
#include <emscripten/html5.h>
#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <tuple>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/dear_imgui/imgui.h"
#include "xprof/frontend/app/components/trace_viewer_v2/animation.h"
#include "xprof/frontend/app/components/trace_viewer_v2/canvas_state.h"
#include "xprof/frontend/app/components/trace_viewer_v2/event_manager.h"
#include "xprof/frontend/app/components/trace_viewer_v2/fonts/fonts.h"
#include "xprof/frontend/app/components/trace_viewer_v2/input_handler.h"  // NO_LINT
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "xprof/frontend/app/components/trace_viewer_v2/webgpu_render_platform.h"

namespace traceviewer {

namespace {

const char* const kWindowTarget = EMSCRIPTEN_EVENT_TARGET_WINDOW;
const char* const kCanvasTarget = "#canvas";
constexpr float kScrollbarSize = 10.0f;

void ApplyDefaultTraceViewerStyles() {
  ImGuiStyle& style = ImGui::GetStyle();
  style.ScrollbarSize = kScrollbarSize;
  style.WindowRounding = 0.0f;
  style.WindowPadding = ImVec2(0.0f, 0.0f);
  style.CellPadding = ImVec2(0.0f, 0.0f);
  style.ItemSpacing = ImVec2(0.0f, 0.0f);
}

void ApplyLightTheme() {
  ImGui::StyleColorsLight();
  ImGuiStyle& style = ImGui::GetStyle();
  // Set the window background color to white. #FFFFFFFF
  style.Colors[ImGuiCol_WindowBg] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

  // Set the table border color to a medium gray. #666666
  // We only use this color for the vertical lines between track title and
  // framechart. Horizontal lines are rendered in timeline.
  style.Colors[ImGuiCol_TableBorderLight] = ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
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
  CanvasState::Update();
  const CanvasState& initial_canvas_state = CanvasState::Current();

  // Load fonts, colors and styles.
  fonts::LoadFonts(initial_canvas_state.device_pixel_ratio());
  // TODO: b/450584482 - Add a dark theme for the timeline.
  ApplyLightTheme();
  ApplyDefaultTraceViewerStyles();

  // Initialize the platform
  platform_ = std::make_unique<WGPURenderPlatform>();
  platform_->Init(initial_canvas_state);
  timeline_ = std::make_unique<Timeline>();
  timeline_->set_event_callback(
      [](absl::string_view type, const EventData& event_data) {
        EventManager::Instance().DispatchEvent(type, event_data);
      });

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
}

void Application::MainLoop() {
  // TODO: b/454172203 - Replace polling `CanvasState::Update()` with a
  // push-based model. Use the `ResizeObserver` API in TypeScript to listen for
  // canvas resize events and notify the C++ application via Emscripten.
  if (CanvasState::Update()) {
    const CanvasState& canvas_state = CanvasState::Current();
    platform_->ResizeSurface(canvas_state);

    UpdateImGuiDisplaySize(canvas_state);
    fonts::LoadFonts(canvas_state.device_pixel_ratio());
  }

  ImGuiIO& io = ImGui::GetIO();
  io.DeltaTime = GetDeltaTime();

  Animation::UpdateAll(io.DeltaTime);

  platform_->NewFrame();
  timeline_->Draw();
  UpdateMouseCursor();
  platform_->RenderFrame();
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
        // TODO: b/470449677 - Support other cursor styles.
        // https://developer.mozilla.org/en-US/docs/Web/CSS/cursor
        // ImGuiMouseCursor_Hand = 7
        if (cursor == 7) {
          cursor_css = 'pointer';
        }

        // The canvas element is guaranteed to exist at this point, because
        // traceViewerV2Main() in main.ts ensures it before calling callMain(),
        // which then calls Initialize().
        document.getElementById('canvas').style.cursor = cursor_css;
      },
      cursor);
}

void Application::Main() {
  emscripten_set_main_loop_arg(
      [](void* app) {
        static_cast<traceviewer::Application*>(app)->MainLoop();
      },
      this, 0, true);
}

float Application::GetDeltaTime() {
  const absl::Time time_now = absl::Now();
  auto delta_time = absl::ToDoubleSeconds(time_now - last_frame_time_);
  last_frame_time_ = time_now;
  return std::min(0.1f, static_cast<float>(delta_time));
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

}  // namespace traceviewer
