#include "frontend/app/components/trace_viewer_v2/input_handler.h"

#include <emscripten/em_js.h>
#include <emscripten/emscripten.h>
#include <emscripten/html5.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "absl/strings/string_view.h"
#include "imgui.h"
#include "frontend/app/components/trace_viewer_v2/scheduler.h"

namespace traceviewer {

namespace {

ImGuiKey TranslateKey(absl::string_view code) {
  if (code == "KeyA") return ImGuiKey_A;
  if (code == "KeyD") return ImGuiKey_D;
  if (code == "KeyE") return ImGuiKey_E;
  if (code == "KeyS") return ImGuiKey_S;
  if (code == "KeyW") return ImGuiKey_W;
  if (code == "Comma") return ImGuiKey_Comma;
  if (code == "KeyO") return ImGuiKey_O;
  if (code == "KeyG") return ImGuiKey_G;
  if (code == "KeyF") return ImGuiKey_F;
  if (code == "KeyM") return ImGuiKey_M;
  if (code == "ArrowDown") return ImGuiKey_DownArrow;
  if (code == "ArrowLeft") return ImGuiKey_LeftArrow;
  if (code == "ArrowRight") return ImGuiKey_RightArrow;
  if (code == "ArrowUp") return ImGuiKey_UpArrow;
  if (code == "Digit0") return ImGuiKey_0;
  if (code == "Digit1") return ImGuiKey_1;
  if (code == "Digit2") return ImGuiKey_2;
  if (code == "Digit3") return ImGuiKey_3;
  if (code == "Digit4") return ImGuiKey_4;
  if (code == "Escape") return ImGuiKey_Escape;
  if (code == "Tab") return ImGuiKey_Tab;
  return ImGuiKey_None;
}

void UpdateModifierKeys(const EmscriptenKeyboardEvent* event) {
  ImGuiIO& io = ImGui::GetIO();
    io.AddKeyEvent(ImGuiKey_ModCtrl, event->ctrlKey);
    io.AddKeyEvent(ImGuiKey_ModShift, event->shiftKey);
    io.AddKeyEvent(ImGuiKey_ModAlt, event->altKey);
    io.AddKeyEvent(ImGuiKey_ModSuper, event->metaKey);
}

// Maps W3C MouseEvent.button (0=Left, 1=Middle, 2=Right, 3=X1, 4=X2) to
// ImGuiMouseButton (0=Left, 1=Right, 2=Middle, 3=X1, 4=X2), or -1 if out of
// range.
int DomButtonToImGuiButton(uint16_t dom_button) {
  switch (dom_button) {
    case 0:
      return ImGuiMouseButton_Left;
    case 1:
      return ImGuiMouseButton_Middle;
    case 2:
      return ImGuiMouseButton_Right;
    case 3:
      return 3;
    case 4:
      return 4;
    default:
      return -1;
  }
}

// Reconciles buttons that are no longer physically held according to the W3C
// MouseEvent.buttons bitmask (1=Left, 2=Right, 4=Middle, 8=X1, 16=X2). Only
// releases buttons; never synthesizes presses so drags started outside the
// canvas do not activate canvas interactions on entry.
void ReconcileReleasedMouseButtons(uint16_t dom_buttons_mask, ImGuiIO& io) {
  struct ButtonMask {
    uint16_t mask;
    int imgui_button;
  };
  constexpr ButtonMask kButtonMasks[] = {
      {1 << 0, ImGuiMouseButton_Left},
      {1 << 1, ImGuiMouseButton_Right},
      {1 << 2, ImGuiMouseButton_Middle},
      {1 << 3, 3},
      {1 << 4, 4},
  };
  // A button that ImGui still holds while none is physically held was released
  // outside the canvas without a mouseup reaching the window. Invalidate the
  // mouse position before queuing the release, so that it is not taken for a
  // click at the last in-canvas position (an already invalid position is
  // filtered as a duplicate).
  if (dom_buttons_mask == 0) {
    for (const ButtonMask& entry : kButtonMasks) {
      if (io.MouseDown[entry.imgui_button]) {
        io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
        break;
      }
    }
  }
  for (const ButtonMask& entry : kButtonMasks) {
    if ((dom_buttons_mask & entry.mask) == 0) {
      io.AddMouseButtonEvent(entry.imgui_button, false);
    }
  }
}

// Returns true if the CSS client point (`client_x`, `client_y`) is over the
// `#canvas` element and not covered by another element (such as a menu,
// tooltip, or dialog). Returns true if there is no `#canvas`, so that releases
// are processed at the current mouse position.
EM_JS(bool, IsClientPointOverCanvas, (int client_x, int client_y), {
  const canvas = document.getElementById('canvas');
  if (!canvas) return true;
  const rect = canvas.getBoundingClientRect();
  if (client_x < rect.left || client_x >= rect.right || client_y < rect.top ||
      client_y >= rect.bottom) {
    return false;
  }
  if (typeof document.elementFromPoint !== 'function') return true;
  // elementFromPoint() returns null outside the layout viewport.
  const hit = document.elementFromPoint(client_x, client_y);
  return hit === canvas || (hit !== null && canvas.contains(hit));
});

// Normalizes browser DOM wheel deltas across pixel, line, and page delta modes
// into CSS pixel units, clamping non-finite values to zero.
ImVec2 NormalizeWheelDelta(const EmscriptenWheelEvent& event,
                           ImVec2 display_size) {
  constexpr float kWheelLineHeightPx = 16.0f;
  constexpr float kFallbackPageSizePx = 800.0f;
  float wheel_x =
      std::isfinite(event.deltaX) ? static_cast<float>(event.deltaX) : 0.0f;
  float wheel_y =
      std::isfinite(event.deltaY) ? static_cast<float>(event.deltaY) : 0.0f;
  if (event.deltaMode == DOM_DELTA_LINE) {
    wheel_x *= kWheelLineHeightPx;
    wheel_y *= kWheelLineHeightPx;
  } else if (event.deltaMode == DOM_DELTA_PAGE) {
    const float page_width =
        display_size.x > 0.0f ? display_size.x : kFallbackPageSizePx;
    const float page_height =
        display_size.y > 0.0f ? display_size.y : kFallbackPageSizePx;
    wheel_x *= page_width;
    wheel_y *= page_height;
  }
  return ImVec2(wheel_x, wheel_y);
}

}  // namespace

// Returns true if an input element has focus or DOM text is selected.
int HasDOMSelectionOrActiveInput() {
  return EM_ASM_INT({
    const selection = window.getSelection() || document.getSelection();
    if (selection) {
      if (selection.type === 'Range') {
        return 1;
      }
      const text = selection.toString();
      if (text && text.length > 0) {
        return 1;
      }
    }
    const activeElement = document.activeElement;
    if (!activeElement) return 0;
    const tagName = activeElement.tagName.toLowerCase();
    return tagName === 'input' || tagName === 'textarea' ||
           activeElement.isContentEditable ? 1 : 0;
  });
}

// Input Event Handlers
//
// Common behavior for all the following input handlers:
//
// 1. Redraw Requests:
//    All handlers unconditionally request a redraw for the next animation frame
//    via Scheduler::Instance().RequestRedraw(). Since drawing is asynchronous
//    (scheduled for the next frame), the input updates processed here will be
//    correctly applied before that frame is rendered.
//
// 2. Return Values (Event Propagation):
//    The return value (EM_BOOL) indicates whether the event was handled:
//    - EM_TRUE (true): The event was handled and should NOT be propagated.
//    - EM_FALSE (false): The event was not handled and SHOULD be propagated to
//      other listeners (e.g., browser default behavior).

EM_BOOL HandleKeyDown(int, const EmscriptenKeyboardEvent* event, void*) {
  Scheduler::Instance().RequestRedraw();
  UpdateModifierKeys(event);

  // If Cmd/Ctrl modifier shortcuts (e.g. Cmd+R, Cmd+C, Cmd+V, Cmd+F)
  // are pressed, return false to allow browser native behaviors.
  if (event->ctrlKey || event->metaKey) {
    return false;
  }

  // If a native input element has focus or DOM text is selected, do not let
  // ImGui capture the keyboard or trigger canvas redraws.
  if (HasDOMSelectionOrActiveInput()) {
    return false;
  }

  // Always let the browser/Angular handle Slash for search box focus toggle
  // and Semicolon for settings dialog toggle.
  if (absl::string_view(event->code) == "Slash" ||
      absl::string_view(event->code) == "Semicolon") {
    return false;
  }

  ImGuiKey key = TranslateKey(event->code);
  if (key != ImGuiKey_None) {
    ImGui::GetIO().AddKeyEvent(key, true);
  }
  return ImGui::GetIO().WantCaptureKeyboard;
}

EM_BOOL HandleKeyUp(int, const EmscriptenKeyboardEvent* event, void*) {
  Scheduler::Instance().RequestRedraw();
  UpdateModifierKeys(event);

  if (event->ctrlKey || event->metaKey) {
    return false;
  }

  if (HasDOMSelectionOrActiveInput()) {
    return false;
  }

  // Always let the browser/Angular handle Slash for search box focus toggle
  // and Semicolon for settings dialog toggle.
  if (absl::string_view(event->code) == "Slash" ||
      absl::string_view(event->code) == "Semicolon") {
    return false;
  }

  ImGuiKey key = TranslateKey(event->code);
  if (key != ImGuiKey_None) {
    ImGui::GetIO().AddKeyEvent(key, false);
  }
  return ImGui::GetIO().WantCaptureKeyboard;
}

EM_BOOL HandleMouseMove(int, const EmscriptenMouseEvent* event, void*) {
  if (event == nullptr || ImGui::GetCurrentContext() == nullptr) {
    return EM_FALSE;
  }
  Scheduler::Instance().RequestRedraw();
  ImGuiIO& io = ImGui::GetIO();
  // Queue button reconciliation before the new cursor position so ImGui
  // trickles any missed off-canvas release at the previous cursor position
  // (zero MouseDelta jump) before applying the re-entry position on the
  // following frame.
  ReconcileReleasedMouseButtons(event->buttons, io);
  io.AddMousePosEvent(event->targetX, event->targetY);
  return io.WantCaptureMouse;
}

EM_BOOL HandleMouseDown(int, const EmscriptenMouseEvent* event, void*) {
  if (event == nullptr || ImGui::GetCurrentContext() == nullptr) {
    return EM_FALSE;
  }
  const int imgui_button = DomButtonToImGuiButton(event->button);
  if (imgui_button < 0) return EM_FALSE;
  Scheduler::Instance().RequestRedraw();
  ImGuiIO& io = ImGui::GetIO();
  // The press carries the pointer position, so a click without a preceding
  // mousemove (e.g. right after the mouse state was cleared on window blur)
  // still lands where the pointer is. An unchanged position is filtered as a
  // duplicate, and ImGui applies the position before the press in the same
  // frame.
  io.AddMousePosEvent(event->targetX, event->targetY);
  io.AddMouseButtonEvent(imgui_button, true);
  return io.WantCaptureMouse;
}

EM_BOOL HandleMouseUp(int, const EmscriptenMouseEvent* event, void*) {
  if (event == nullptr || ImGui::GetCurrentContext() == nullptr) {
    return EM_FALSE;
  }
  const int imgui_button = DomButtonToImGuiButton(event->button);
  if (imgui_button >= 0) {
    Scheduler::Instance().RequestRedraw();
    ImGuiIO& io = ImGui::GetIO();
    // Releasing the last held button outside the canvas, or over an element
    // covering it, must not count as a click at the last in-canvas position
    // (which froze when the pointer left the canvas): invalidate the position
    // before the release. ImGui applies both in the same frame.
    if (event->buttons == 0 &&
        !IsClientPointOverCanvas(event->clientX, event->clientY)) {
      io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
    }
    io.AddMouseButtonEvent(imgui_button, false);
  }
  // Registered on window so releases outside the canvas still reach ImGui;
  // return EM_FALSE so surrounding DOM elements retain default mouseup
  // behavior.
  return EM_FALSE;
}

EM_BOOL HandleWheel(int, const EmscriptenWheelEvent* event, void*) {
  Scheduler::Instance().RequestRedraw();
  ImGuiIO& io = ImGui::GetIO();

    io.AddKeyEvent(ImGuiKey_ModCtrl, event->mouse.ctrlKey);
    io.AddKeyEvent(ImGuiKey_ModShift, event->mouse.shiftKey);
    io.AddKeyEvent(ImGuiKey_ModSuper, event->mouse.metaKey);

  const ImVec2 wheel_delta = NormalizeWheelDelta(*event, io.DisplaySize);
  io.AddMouseWheelEvent(wheel_delta.x, wheel_delta.y);

  return io.WantCaptureMouse;
}

EM_BOOL HandleWindowBlur(int, const EmscriptenFocusEvent* event, void*) {
  if (event != nullptr && event->nodeName[0] != '\0' &&
      absl::string_view(event->nodeName) != "#window") {
    return EM_FALSE;
  }
  if (ImGui::GetCurrentContext() == nullptr) {
    return EM_FALSE;
  }
  Scheduler::Instance().RequestRedraw();
  ImGuiIO& io = ImGui::GetIO();
  // Clear the input state directly instead of queuing releases: a queued
  // release would set IsMouseReleased() and commit the in-flight interaction
  // (click, selection, track drop) on focus loss, whereas cleared state is
  // seen as a cancelled interaction. This mirrors what ImGui does on
  // AddFocusEvent(false). Also drop queued events, so that nothing pressed
  // before the blur is applied after it.
  io.ClearEventsQueue();
  io.ClearInputKeys();
  io.ClearInputMouse();
  return EM_FALSE;
}

}  // namespace traceviewer
