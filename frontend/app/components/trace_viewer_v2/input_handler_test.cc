#include "frontend/app/components/trace_viewer_v2/input_handler.h"

#include <emscripten/em_js.h>
#include <emscripten/html5.h>

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>
#include "imgui.h"

namespace traceviewer {
namespace {

// The geometry of the test canvas, in CSS pixels from the top-left of the
// viewport.
constexpr int kTestCanvasWidth = 300;
constexpr int kTestCanvasHeight = 150;

// Creates the #canvas element at the top-left of the viewport, kTestCanvasWidth
// by kTestCanvasHeight CSS pixels.
EM_JS(void, CreateTestCanvas, (int width, int height), {
  let canvas = document.getElementById('canvas');
  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.id = 'canvas';
    document.body.appendChild(canvas);
  }
  canvas.style.position = 'fixed';
  canvas.style.left = '0px';
  canvas.style.top = '0px';
  canvas.style.width = width + 'px';
  canvas.style.height = height + 'px';
});

EM_JS(void, RemoveTestCanvas, (), {
  const canvas = document.getElementById('canvas');
  if (canvas) canvas.remove();
});

// Covers the test canvas with an element that receives pointer events, like an
// open menu or dialog would.
EM_JS(void, CoverTestCanvas, (int width, int height), {
  let overlay = document.getElementById('test-canvas-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'test-canvas-overlay';
    document.body.appendChild(overlay);
  }
  overlay.style.position = 'fixed';
  overlay.style.left = '0px';
  overlay.style.top = '0px';
  overlay.style.width = width + 'px';
  overlay.style.height = height + 'px';
  overlay.style.zIndex = '1000';
});

EM_JS(void, UncoverTestCanvas, (), {
  const overlay = document.getElementById('test-canvas-overlay');
  if (overlay) overlay.remove();
});

EM_JS(void, DispatchBrowserWheelEvent,
      (double dx, double dy, bool ctrl, bool shift, bool meta), {
        const el = document.getElementById('canvas');
        const event = new WheelEvent('wheel', {
          deltaX : dx,
          deltaY : dy,
          ctrlKey : ctrl,
          shiftKey : shift,
          metaKey : meta,
          bubbles : true,
          cancelable : true
        });
        el.dispatchEvent(event);
      });

class InputHandlerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(1024, 768);
    // Build font atlas to avoid assertion in NewFrame
    unsigned char* pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
  }

  void TearDown() override { ImGui::DestroyContext(); }

  // Runs one ImGui frame, so that queued input events are applied.
  void RunFrame() {
    ImGui::GetIO().DeltaTime = 1.0f / 60.0f;
    ImGui::NewFrame();
    ImGui::EndFrame();
  }

  // Moves the pointer over the canvas to (x, y) and applies it.
  void MovePointer(int x, int y) {
    EmscriptenMouseEvent move = {};
    move.targetX = x;
    move.targetY = y;
    HandleMouseMove(0, &move, nullptr);
    RunFrame();
  }

  // Presses the left button at (x, y) and applies it.
  void PressLeftButton(int x, int y) {
    EmscriptenMouseEvent down = {};
    down.button = 0;
    down.buttons = 1;
    down.targetX = x;
    down.targetY = y;
    HandleMouseDown(0, &down, nullptr);
    RunFrame();
  }

  // Returns a window-level mouseup of DOM button `button` at the CSS client
  // point (`client_x`, `client_y`), with `buttons` still held afterwards.
  static EmscriptenMouseEvent MouseUpAt(int client_x, int client_y,
                                        uint16_t button = 0,
                                        uint16_t buttons = 0) {
    EmscriptenMouseEvent up = {};
    up.button = button;
    up.buttons = buttons;
    up.clientX = client_x;
    up.clientY = client_y;
    return up;
  }
};

// Tests that need a #canvas element in the page, with a known geometry.
class InputHandlerCanvasTest : public InputHandlerTest {
 protected:
  void SetUp() override {
    InputHandlerTest::SetUp();
    CreateTestCanvas(kTestCanvasWidth, kTestCanvasHeight);
  }

  void TearDown() override {
    UncoverTestCanvas();
    RemoveTestCanvas();
    InputHandlerTest::TearDown();
  }
};

TEST_F(InputHandlerTest, HandleWheelUpdatesModifiersAndDeltas) {
  ImGuiIO& io = ImGui::GetIO();
  EmscriptenWheelEvent event;
  memset(&event, 0, sizeof(event));
  event.mouse.ctrlKey = true;
  event.mouse.shiftKey = false;
  event.mouse.metaKey = true;
  event.deltaX = 100.0;
  event.deltaY = -200.0;

  HandleWheel(0, &event, nullptr);

  // Prepare for NewFrame to process queued events
  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();

  EXPECT_TRUE(io.KeyCtrl);
  EXPECT_FALSE(io.KeyShift);
  EXPECT_TRUE(io.KeySuper);

  // Expect negated and doubled values
  EXPECT_FLOAT_EQ(io.MouseWheelH, 100.0f);
  EXPECT_FLOAT_EQ(io.MouseWheel, -200.0f);
}

TEST_F(InputHandlerTest, HandleKeyDownUpdatesModifiers) {
  ImGuiIO& io = ImGui::GetIO();
  EmscriptenKeyboardEvent event;
  memset(&event, 0, sizeof(event));
  event.ctrlKey = false;
  event.shiftKey = true;
  event.altKey = false;
  event.metaKey = true;

  HandleKeyDown(0, &event, nullptr);

  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();

  EXPECT_FALSE(io.KeyCtrl);
  EXPECT_TRUE(io.KeyShift);
  EXPECT_FALSE(io.KeyAlt);
  EXPECT_TRUE(io.KeySuper);
}

TEST_F(InputHandlerTest, HandleKeyDownTranslatesShortcutKeys) {
  ImGuiIO& io = ImGui::GetIO();

  struct TestCase {
    const char* code;
    ImGuiKey expected_key;
  };

  const TestCase test_cases[] = {
      {"KeyA", ImGuiKey_A},
      {"KeyD", ImGuiKey_D},
      {"KeyE", ImGuiKey_E},
      {"KeyS", ImGuiKey_S},
      {"KeyW", ImGuiKey_W},
      {"Comma", ImGuiKey_Comma},
      {"KeyO", ImGuiKey_O},
      {"KeyG", ImGuiKey_G},
      {"KeyF", ImGuiKey_F},
      {"KeyM", ImGuiKey_M},
      {"ArrowDown", ImGuiKey_DownArrow},
      {"ArrowLeft", ImGuiKey_LeftArrow},
      {"ArrowRight", ImGuiKey_RightArrow},
      {"ArrowUp", ImGuiKey_UpArrow},
      {"Digit0", ImGuiKey_0},
      {"Digit1", ImGuiKey_1},
      {"Digit2", ImGuiKey_2},
      {"Digit3", ImGuiKey_3},
      {"Digit4", ImGuiKey_4},
      {"Escape", ImGuiKey_Escape},
      {"Tab", ImGuiKey_Tab},
  };

  for (const auto& tc : test_cases) {
    EmscriptenKeyboardEvent event;
    memset(&event, 0, sizeof(event));
    strncpy(event.code, tc.code, sizeof(event.code) - 1);

    HandleKeyDown(0, &event, nullptr);

    io.DeltaTime = 1.0f / 60.0f;
    ImGui::NewFrame();

    EXPECT_TRUE(ImGui::IsKeyDown(tc.expected_key))
        << "Failed for code: " << tc.code;

    // Reset key state for next iteration
    ImGui::GetIO().AddKeyEvent(tc.expected_key, false);
    ImGui::EndFrame();
  }
}

TEST_F(InputHandlerTest, HandleKeyDownSlashReturnsFalse) {
  EmscriptenKeyboardEvent event;
  memset(&event, 0, sizeof(event));
  strncpy(event.code, "Slash", sizeof(event.code) - 1);

  EXPECT_FALSE(HandleKeyDown(0, &event, nullptr));
}

TEST_F(InputHandlerTest, HandleKeyDownSemicolonReturnsFalse) {
  EmscriptenKeyboardEvent event;
  memset(&event, 0, sizeof(event));
  strncpy(event.code, "Semicolon", sizeof(event.code) - 1);

  EXPECT_FALSE(HandleKeyDown(0, &event, nullptr));
}

TEST_F(InputHandlerTest, HandleKeyDownReturnsFalseWhenModifierKeysArePressed) {
  EmscriptenKeyboardEvent event;
  memset(&event, 0, sizeof(event));
  event.metaKey = true;
  strncpy(event.code, "KeyR", sizeof(event.code) - 1);

  EM_BOOL handled = HandleKeyDown(0, &event, nullptr);

  EXPECT_FALSE(handled);
}
EM_JS(void, SetDOMSelectionForTest, (const char* selectedText), {
  const range = document.createRange();
  let span = document.getElementById('test-selection-span');
  if (!span) {
    span = document.createElement('span');
    span.id = 'test-selection-span';
    document.body.appendChild(span);
  }
  span.textContent = UTF8ToString(selectedText);
  range.selectNodeContents(span);
  const selection = window.getSelection();
  if (selection) {
    selection.removeAllRanges();
    selection.addRange(range);
  }
});

EM_JS(void, ClearDOMSelectionForTest, (), {
  const selection = window.getSelection();
  if (selection) selection.removeAllRanges();
  const span = document.getElementById('test-selection-span');
  if (span) span.remove();
});

TEST_F(InputHandlerTest, HandleKeyDownReturnsFalseWhenDOMTextIsSelected) {
  SetDOMSelectionForTest("sample selected text");

  EmscriptenKeyboardEvent event;
  memset(&event, 0, sizeof(event));
  strncpy(event.code, "KeyA", sizeof(event.code) - 1);

  EM_BOOL handled = HandleKeyDown(0, &event, nullptr);

  ClearDOMSelectionForTest();

  EXPECT_FALSE(handled);
}

TEST_F(InputHandlerTest,
       HandleMouseDownAndUpMapDomButtonsAndAllowDefaultOnMouseUp) {
  ImGuiIO& io = ImGui::GetIO();

  struct ButtonMappingCase {
    uint16_t dom_button;
    int expected_imgui_button;
  };
  constexpr ButtonMappingCase kCases[] = {
      {0, ImGuiMouseButton_Left},
      {1, ImGuiMouseButton_Middle},
      {2, ImGuiMouseButton_Right},
      {3, 3},
      {4, 4},
  };

  for (const ButtonMappingCase& tc : kCases) {
    io.WantCaptureMouse = true;
    EmscriptenMouseEvent down_event = {};
    down_event.button = tc.dom_button;
    EXPECT_TRUE(HandleMouseDown(0, &down_event, nullptr));

    io.DeltaTime = 1.0f / 60.0f;
    ImGui::NewFrame();
    EXPECT_TRUE(ImGui::IsMouseDown(tc.expected_imgui_button));
    ImGui::EndFrame();

    // Window-level mouseup must return EM_FALSE even when ImGui wants the mouse
    // so surrounding DOM elements retain default click/mouseup behavior.
    io.WantCaptureMouse = true;
    EmscriptenMouseEvent up_event = {};
    up_event.button = tc.dom_button;
    EXPECT_FALSE(HandleMouseUp(0, &up_event, nullptr));

    io.DeltaTime = 1.0f / 60.0f;
    ImGui::NewFrame();
    EXPECT_TRUE(ImGui::IsMouseReleased(tc.expected_imgui_button));
    EXPECT_FALSE(ImGui::IsMouseDown(tc.expected_imgui_button));
    ImGui::EndFrame();
  }

  // Out-of-range DOM button indices (>= 5) must be ignored safely without
  // triggering ImGui bounds assertions.
  EmscriptenMouseEvent invalid_button_event = {};
  invalid_button_event.button = 5;
  EXPECT_FALSE(HandleMouseDown(0, &invalid_button_event, nullptr));
  EXPECT_FALSE(HandleMouseUp(0, &invalid_button_event, nullptr));
}

TEST_F(InputHandlerTest,
       HandleMouseMoveReconcilesReleasedButtonBeforeUpdatingPos) {
  ImGuiIO& io = ImGui::GetIO();

  MovePointer(100, 200);
  PressLeftButton(100, 200);
  ImGui::NewFrame();
  ASSERT_TRUE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  ImGui::EndFrame();

  // Simulate releasing the mouse button outside the browser/iframe (so no
  // mouseup event arrived) and re-entering the canvas at a distant Y position
  // with buttons == 0.
  EmscriptenMouseEvent reenter = {};
  reenter.targetX = 100;
  reenter.targetY = 10;
  reenter.buttons = 0;
  HandleMouseMove(0, &reenter, nullptr);

  // Frame 1: ImGui trickles the button release before the new position. The
  // release happens off-canvas (invalid position), so it is neither a click at
  // the last in-canvas position nor a MouseDelta jump.
  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_TRUE(ImGui::IsMouseReleased(ImGuiMouseButton_Left));
  EXPECT_FALSE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  EXPECT_FALSE(ImGui::IsMousePosValid());
  EXPECT_FLOAT_EQ(io.MouseDelta.x, 0.0f);
  EXPECT_FLOAT_EQ(io.MouseDelta.y, 0.0f);
  ImGui::EndFrame();

  // Frame 2: The trickled cursor position applies cleanly with the button up.
  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_FALSE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  EXPECT_FALSE(ImGui::IsMouseReleased(ImGuiMouseButton_Left));
  EXPECT_FLOAT_EQ(io.MousePos.x, 100.0f);
  EXPECT_FLOAT_EQ(io.MousePos.y, 10.0f);
  EXPECT_FLOAT_EQ(io.MouseDelta.x, 0.0f);
  EXPECT_FLOAT_EQ(io.MouseDelta.y, 0.0f);
  ImGui::EndFrame();
}

TEST_F(InputHandlerTest,
       HandleMouseMovePreservesHeldDragAndDoesNotSynthesizePress) {
  ImGuiIO& io = ImGui::GetIO();

  // Moving across the canvas with buttons == 1 when the drag started outside
  // the canvas must NOT synthesize a mouse button press inside ImGui.
  EmscriptenMouseEvent external_drag = {};
  external_drag.targetX = 50;
  external_drag.targetY = 60;
  external_drag.buttons = 1;
  HandleMouseMove(0, &external_drag, nullptr);

  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_FALSE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  ImGui::EndFrame();

  // Start a drag inside the canvas with both Left (button=0) and Right
  // (button=2) pressed.
  EmscriptenMouseEvent left_down = {};
  left_down.button = 0;
  left_down.buttons = 1;
  left_down.targetX = 50;
  left_down.targetY = 60;
  HandleMouseDown(0, &left_down, nullptr);
  EmscriptenMouseEvent right_down = {};
  right_down.button = 2;
  right_down.buttons = 3;
  right_down.targetX = 50;
  right_down.targetY = 60;
  HandleMouseDown(0, &right_down, nullptr);

  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  ASSERT_TRUE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  ASSERT_TRUE(ImGui::IsMouseDown(ImGuiMouseButton_Right));
  ImGui::EndFrame();

  // Move with buttons == 1 (Left held, Right no longer held). The position
  // stays valid, as the Left drag goes on.
  EmscriptenMouseEvent drag_move = {};
  drag_move.targetX = 120;
  drag_move.targetY = 140;
  drag_move.buttons = 1;
  HandleMouseMove(0, &drag_move, nullptr);

  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_TRUE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  EXPECT_TRUE(ImGui::IsMouseReleased(ImGuiMouseButton_Right));
  EXPECT_FALSE(ImGui::IsMouseDown(ImGuiMouseButton_Right));
  EXPECT_TRUE(ImGui::IsMousePosValid());
  ImGui::EndFrame();

  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_TRUE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  EXPECT_FLOAT_EQ(io.MousePos.x, 120.0f);
  EXPECT_FLOAT_EQ(io.MousePos.y, 140.0f);
  ImGui::EndFrame();
}

TEST_F(InputHandlerTest, HandleMouseDownSeedsMousePosWhenNoMoveCameFirst) {
  ImGuiIO& io = ImGui::GetIO();

  // No mousemove has reached the canvas yet (e.g. right after the mouse state
  // was cleared on window blur), so the position is invalid. The press still
  // carries the pointer position.
  ASSERT_FALSE(ImGui::IsMousePosValid());
  EmscriptenMouseEvent down = {};
  down.button = 0;
  down.buttons = 1;
  down.targetX = 40;
  down.targetY = 50;
  HandleMouseDown(0, &down, nullptr);

  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  // The position and the press apply in the same frame, so the click lands
  // where the pointer is.
  EXPECT_TRUE(ImGui::IsMouseClicked(ImGuiMouseButton_Left));
  EXPECT_FLOAT_EQ(io.MousePos.x, 40.0f);
  EXPECT_FLOAT_EQ(io.MousePos.y, 50.0f);
  ImGui::EndFrame();

  // The same holds after the state was cleared mid-session.
  io.ClearInputMouse();
  ASSERT_FALSE(ImGui::IsMousePosValid());
  EmscriptenMouseEvent down_again = {};
  down_again.button = 0;
  down_again.buttons = 1;
  down_again.targetX = 60;
  down_again.targetY = 70;
  HandleMouseDown(0, &down_again, nullptr);

  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_TRUE(ImGui::IsMouseClicked(ImGuiMouseButton_Left));
  EXPECT_FLOAT_EQ(io.MousePos.x, 60.0f);
  EXPECT_FLOAT_EQ(io.MousePos.y, 70.0f);
  ImGui::EndFrame();
}

TEST_F(InputHandlerCanvasTest,
       HandleMouseUpOutsideCanvasInvalidatesMousePosBeforeRelease) {
  ImGuiIO& io = ImGui::GetIO();
  MovePointer(100, 100);
  PressLeftButton(100, 100);
  ImGui::NewFrame();
  ASSERT_TRUE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  ASSERT_TRUE(ImGui::IsMousePosValid());
  ImGui::EndFrame();

  // The pointer left the canvas (no more mousemove) and the button is released
  // over the page, right of the canvas.
  EmscriptenMouseEvent up = MouseUpAt(kTestCanvasWidth + 100, 50);
  EXPECT_FALSE(HandleMouseUp(0, &up, nullptr));

  // The release applies with an invalid position: no click at the last
  // in-canvas position, no MouseDelta jump.
  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_TRUE(ImGui::IsMouseReleased(ImGuiMouseButton_Left));
  EXPECT_FALSE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  EXPECT_FALSE(ImGui::IsMousePosValid());
  EXPECT_FLOAT_EQ(io.MouseDelta.x, 0.0f);
  EXPECT_FLOAT_EQ(io.MouseDelta.y, 0.0f);
  ImGui::EndFrame();
}

TEST_F(InputHandlerCanvasTest, HandleMouseUpInsideCanvasKeepsMousePos) {
  ImGuiIO& io = ImGui::GetIO();
  MovePointer(100, 100);
  PressLeftButton(100, 100);

  EmscriptenMouseEvent up = MouseUpAt(100, 100);
  EXPECT_FALSE(HandleMouseUp(0, &up, nullptr));

  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_TRUE(ImGui::IsMouseReleased(ImGuiMouseButton_Left));
  EXPECT_TRUE(ImGui::IsMousePosValid());
  EXPECT_FLOAT_EQ(io.MousePos.x, 100.0f);
  EXPECT_FLOAT_EQ(io.MousePos.y, 100.0f);
  ImGui::EndFrame();
}

TEST_F(InputHandlerCanvasTest,
       HandleMouseUpOverElementCoveringCanvasInvalidatesMousePos) {
  ImGuiIO& io = ImGui::GetIO();
  MovePointer(100, 100);
  PressLeftButton(100, 100);

  // A menu or dialog opened over the canvas receives the pointer, so the
  // canvas position froze where the pointer went under it.
  CoverTestCanvas(kTestCanvasWidth, kTestCanvasHeight);
  EmscriptenMouseEvent up = MouseUpAt(100, 100);
  EXPECT_FALSE(HandleMouseUp(0, &up, nullptr));

  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_TRUE(ImGui::IsMouseReleased(ImGuiMouseButton_Left));
  EXPECT_FALSE(ImGui::IsMousePosValid());
  ImGui::EndFrame();
}

TEST_F(InputHandlerCanvasTest,
       HandleMouseUpOutsideCanvasKeepsMousePosWhileOtherButtonIsHeld) {
  ImGuiIO& io = ImGui::GetIO();
  MovePointer(100, 100);
  EmscriptenMouseEvent left_down = {};
  left_down.button = 0;
  left_down.buttons = 1;
  left_down.targetX = 100;
  left_down.targetY = 100;
  HandleMouseDown(0, &left_down, nullptr);
  EmscriptenMouseEvent right_down = {};
  right_down.button = 2;
  right_down.buttons = 3;
  right_down.targetX = 100;
  right_down.targetY = 100;
  HandleMouseDown(0, &right_down, nullptr);
  RunFrame();

  // Releasing Right outside the canvas while Left is still held keeps the
  // position of the ongoing Left drag.
  EmscriptenMouseEvent up =
      MouseUpAt(kTestCanvasWidth + 100, 50, /*button=*/2, /*buttons=*/1);
  EXPECT_FALSE(HandleMouseUp(0, &up, nullptr));

  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_TRUE(ImGui::IsMouseReleased(ImGuiMouseButton_Right));
  EXPECT_TRUE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  EXPECT_TRUE(ImGui::IsMousePosValid());
  EXPECT_FLOAT_EQ(io.MousePos.x, 100.0f);
  EXPECT_FLOAT_EQ(io.MousePos.y, 100.0f);
  ImGui::EndFrame();
}

TEST_F(InputHandlerTest, HandleWindowBlurClearsInputStateOnlyForWindowTarget) {
  ImGuiIO& io = ImGui::GetIO();

  MovePointer(100, 200);
  PressLeftButton(100, 200);

  EmscriptenKeyboardEvent key_down = {};
  key_down.shiftKey = true;
  strncpy(key_down.code, "KeyW", sizeof(key_down.code) - 1);
  HandleKeyDown(0, &key_down, nullptr);

  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  ASSERT_TRUE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  ASSERT_TRUE(ImGui::IsMousePosValid());
  ASSERT_TRUE(ImGui::IsKeyDown(ImGuiKey_W));
  ASSERT_TRUE(io.KeyShift);
  ImGui::EndFrame();

  // Blurring a child DOM element (such as an input box) must NOT clear canvas
  // input state.
  EmscriptenFocusEvent child_blur = {};
  strncpy(child_blur.nodeName, "INPUT", sizeof(child_blur.nodeName) - 1);
  EXPECT_FALSE(HandleWindowBlur(0, &child_blur, nullptr));

  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_TRUE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  EXPECT_TRUE(ImGui::IsMousePosValid());
  EXPECT_TRUE(ImGui::IsKeyDown(ImGuiKey_W));
  EXPECT_TRUE(io.KeyShift);
  ImGui::EndFrame();

  // Blurring the window itself clears the held mouse button, the mouse
  // position, modifiers, and shortcut keys, without a release transition that
  // would commit the interaction as a click.
  EmscriptenFocusEvent window_blur = {};
  strncpy(window_blur.nodeName, "#window", sizeof(window_blur.nodeName) - 1);
  EXPECT_FALSE(HandleWindowBlur(0, &window_blur, nullptr));

  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_FALSE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  EXPECT_FALSE(ImGui::IsMouseReleased(ImGuiMouseButton_Left));
  EXPECT_FALSE(ImGui::IsMousePosValid());
  EXPECT_FALSE(ImGui::IsKeyDown(ImGuiKey_W));
  EXPECT_FALSE(io.KeyShift);
  ImGui::EndFrame();
}

TEST_F(InputHandlerTest, HandleWindowBlurDropsQueuedInputEvents) {
  ImGuiIO& io = ImGui::GetIO();
  MovePointer(100, 200);

  // A press queued in the same tick as the blur must not be applied after it.
  EmscriptenMouseEvent down = {};
  down.button = 0;
  down.buttons = 1;
  down.targetX = 100;
  down.targetY = 200;
  HandleMouseDown(0, &down, nullptr);
  EmscriptenFocusEvent window_blur = {};
  strncpy(window_blur.nodeName, "#window", sizeof(window_blur.nodeName) - 1);
  EXPECT_FALSE(HandleWindowBlur(0, &window_blur, nullptr));

  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_FALSE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  EXPECT_FALSE(ImGui::IsMouseClicked(ImGuiMouseButton_Left));
  EXPECT_FALSE(ImGui::IsMousePosValid());
  ImGui::EndFrame();
}

TEST_F(InputHandlerTest, HandleWheelNormalizesLineAndPageDeltaModes) {
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(1000.0f, 600.0f);

  EmscriptenWheelEvent line_event = {};
  line_event.deltaMode = DOM_DELTA_LINE;
  line_event.deltaX = 2.0;
  line_event.deltaY = -3.0;

  HandleWheel(0, &line_event, nullptr);
  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_FLOAT_EQ(io.MouseWheelH, 32.0f);
  EXPECT_FLOAT_EQ(io.MouseWheel, -48.0f);
  ImGui::EndFrame();

  EmscriptenWheelEvent page_event = {};
  page_event.deltaMode = DOM_DELTA_PAGE;
  page_event.deltaX = 0.5;
  page_event.deltaY = -1.0;

  HandleWheel(0, &page_event, nullptr);
  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_FLOAT_EQ(io.MouseWheelH, 500.0f);
  EXPECT_FLOAT_EQ(io.MouseWheel, -600.0f);
  ImGui::EndFrame();

  // Non-positive DisplaySize falls back to kFallbackPageSizePx (800.0f).
  io.DisplaySize = ImVec2(0.0f, 0.0f);
  HandleWheel(0, &page_event, nullptr);
  io.DisplaySize = ImVec2(1000.0f, 600.0f);
  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_FLOAT_EQ(io.MouseWheelH, 400.0f);
  EXPECT_FLOAT_EQ(io.MouseWheel, -800.0f);
  ImGui::EndFrame();
}

TEST_F(InputHandlerTest, HandleWheelIgnoresNonFiniteDeltas) {
  ImGuiIO& io = ImGui::GetIO();
  EmscriptenWheelEvent event = {};
  event.deltaMode = DOM_DELTA_PIXEL;
  event.deltaX = std::numeric_limits<double>::quiet_NaN();
  event.deltaY = std::numeric_limits<double>::infinity();

  HandleWheel(0, &event, nullptr);
  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  EXPECT_FLOAT_EQ(io.MouseWheelH, 0.0f);
  EXPECT_FLOAT_EQ(io.MouseWheel, 0.0f);
  ImGui::EndFrame();
}

}  // namespace
}  // namespace traceviewer
