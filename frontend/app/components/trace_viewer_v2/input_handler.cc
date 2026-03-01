#include "xprof/frontend/app/components/trace_viewer_v2/input_handler.h"

#include <emscripten/emscripten.h>
#include <emscripten/html5.h>

#include "absl/strings/string_view.h"
#include "third_party/dear_imgui/imgui.h"

namespace traceviewer {

namespace {

ImGuiKey TranslateKey(absl::string_view code) {
  if (code == "KeyA") return ImGuiKey_A;
  if (code == "KeyD") return ImGuiKey_D;
  if (code == "KeyS") return ImGuiKey_S;
  if (code == "KeyW") return ImGuiKey_W;
  if (code == "ArrowDown") return ImGuiKey_DownArrow;
  if (code == "ArrowUp") return ImGuiKey_UpArrow;
  if (code == "Escape") return ImGuiKey_Escape;
  return ImGuiKey_None;
}

void UpdateModifierKeys(const EmscriptenKeyboardEvent* event) {
  ImGuiIO& io = ImGui::GetIO();
  io.AddKeyEvent(ImGuiMod_Ctrl, event->ctrlKey);
  io.AddKeyEvent(ImGuiMod_Shift, event->shiftKey);
  io.AddKeyEvent(ImGuiMod_Alt, event->altKey);
  io.AddKeyEvent(ImGuiMod_Super, event->metaKey);
}

}  // namespace

// Returns true if the currently active element is an input element.
int IsActiveElementInput() {
  return EM_ASM_INT({
    const activeElement = document.activeElement;
    if (!activeElement) return 0;
    const tagName = activeElement.tagName.toLowerCase();
    return tagName == 'input' || tagName == 'textarea' ||
           activeElement.isContentEditable;
  });
}

EM_BOOL HandleKeyDown(int, const EmscriptenKeyboardEvent* event, void*) {
  UpdateModifierKeys(event);

  // If a native input element has focus, do not let ImGui capture the keyboard.
  if (IsActiveElementInput()) {
    return false;
  }

  ImGuiKey key = TranslateKey(event->code);
  if (key != ImGuiKey_None) {
    ImGui::GetIO().AddKeyEvent(key, true);
  }
  return ImGui::GetIO().WantCaptureKeyboard;
}

EM_BOOL HandleKeyUp(int, const EmscriptenKeyboardEvent* event, void*) {
  UpdateModifierKeys(event);

  // If a native input element has focus, do not let ImGui capture the keyboard.
  if (IsActiveElementInput()) {
    return false;
  }

  ImGuiKey key = TranslateKey(event->code);
  if (key != ImGuiKey_None) {
    ImGui::GetIO().AddKeyEvent(key, false);
  }
  return ImGui::GetIO().WantCaptureKeyboard;
}

EM_BOOL HandleMouseMove(int, const EmscriptenMouseEvent* event, void*) {
  ImGuiIO& io = ImGui::GetIO();
  io.AddMousePosEvent(event->targetX, event->targetY);
  return io.WantCaptureMouse;
}

EM_BOOL HandleMouseDown(int, const EmscriptenMouseEvent* event, void*) {
  ImGuiIO& io = ImGui::GetIO();
  io.AddMouseButtonEvent(event->button, true);
  return io.WantCaptureMouse;
}

EM_BOOL HandleMouseUp(int, const EmscriptenMouseEvent* event, void*) {
  ImGuiIO& io = ImGui::GetIO();
  io.AddMouseButtonEvent(event->button, false);
  return io.WantCaptureMouse;
}

EM_BOOL HandleWheel(int, const EmscriptenWheelEvent* event, void*) {
  ImGuiIO& io = ImGui::GetIO();

  io.AddKeyEvent(ImGuiMod_Ctrl, event->mouse.ctrlKey);
  io.AddKeyEvent(ImGuiMod_Shift, event->mouse.shiftKey);
  io.AddKeyEvent(ImGuiMod_Super, event->mouse.metaKey);

  float wheel_x = static_cast<float>(event->deltaX);
  float wheel_y = static_cast<float>(event->deltaY);

  io.AddMouseWheelEvent(wheel_x, wheel_y);

  return io.WantCaptureMouse;
}

}  // namespace traceviewer
