#include "frontend/app/components/trace_viewer_v2/input_handler.h"

#include <emscripten/em_js.h>
#include <emscripten/html5.h>

#include "<gtest/gtest.h>"
#include "imgui.h"

namespace traceviewer {
namespace {

EM_JS(void, CreateTestCanvas, (), {
  if (!document.getElementById('canvas')) {
    const canvas = document.createElement('canvas');
    canvas.id = 'canvas';
    document.body.appendChild(canvas);
  }
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
    // CreateTestCanvas();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(1024, 768);
    // Build font atlas to avoid assertion in NewFrame
    unsigned char* pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
  }

  void TearDown() override { ImGui::DestroyContext(); }
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

TEST_F(InputHandlerTest, HandleKeyDownReturnsFalseWhenModifierKeysArePressed) {
  EmscriptenKeyboardEvent event;
  memset(&event, 0, sizeof(event));
  event.metaKey = true;
  strncpy(event.code, "KeyR", sizeof(event.code) - 1);

  EM_BOOL handled = HandleKeyDown(0, &event, nullptr);

  EXPECT_FALSE(handled);
}
}  // namespace
}  // namespace traceviewer
