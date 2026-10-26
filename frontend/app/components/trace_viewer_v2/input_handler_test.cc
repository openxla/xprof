#include "frontend/app/components/trace_viewer_v2/input_handler.h"

#include <emscripten/em_js.h>
#include <emscripten/html5.h>

#include <gtest/gtest.h>
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

EM_JS(void, CreateShadowInputForTest, (), {
  const host = document.createElement('div');
  host.id = 'shadow-test-host';
  const shadow = host.attachShadow({mode : 'open'});
  const input = document.createElement('input');
  input.id = 'shadow-input';
  shadow.appendChild(input);
  document.body.appendChild(host);
  input.focus();
});

EM_JS(void, CreateNestedShadowInputForTest, (), {
  const outerHost = document.createElement('div');
  outerHost.id = 'nested-outer-host';
  const outerShadow = outerHost.attachShadow({mode : 'open'});
  const innerHost = document.createElement('div');
  innerHost.id = 'nested-inner-host';
  const innerShadow = innerHost.attachShadow({mode : 'open'});
  const textarea = document.createElement('textarea');
  textarea.id = 'nested-textarea';
  innerShadow.appendChild(textarea);
  outerShadow.appendChild(innerHost);
  document.body.appendChild(outerHost);
  textarea.focus();
});

EM_JS(void, CleanupShadowElementsForTest, (), {
  const host = document.getElementById('shadow-test-host');
  if (host) host.remove();
  const nested = document.getElementById('nested-outer-host');
  if (nested) nested.remove();
  if (document.activeElement && document.activeElement.blur) {
    document.activeElement.blur();
  }
});

EM_JS(void, CreateNativeDialogForTest, (), {
  const dialog = document.createElement('dialog');
  dialog.id = 'test-native-dialog';
  dialog.setAttribute('open', "");
  document.body.appendChild(dialog);
});

EM_JS(void, CreateAriaModalForTest, (), {
  const div = document.createElement('div');
  div.id = 'test-aria-modal';
  div.setAttribute('aria-modal', 'true');
  document.body.appendChild(div);
});

EM_JS(void, CreateTraceViewerHelpDialogForTest, (bool withShadowMdDialog), {
  const el = document.createElement('trace-viewer-help-dialog');
  el.id = 'test-help-dialog';
  if (withShadowMdDialog) {
    const shadow = el.attachShadow({mode : 'open'});
    const mdDialog = document.createElement('md-dialog');
    mdDialog.setAttribute('open', "");
    shadow.appendChild(mdDialog);
  } else {
    el.setAttribute('open', "");
  }
  document.body.appendChild(el);
});

EM_JS(void, CreateTraceViewerCustomizationPanelForTest, (), {
  const el = document.createElement('trace-viewer-customization-panel');
  el.id = 'test-custom-panel';
  const shadow = el.attachShadow({mode : 'open'});
  const mdDialog = document.createElement('md-dialog');
  mdDialog.setAttribute('open', "");
  shadow.appendChild(mdDialog);
  document.body.appendChild(el);
});

EM_JS(void, CleanupModalDialogsForTest, (), {
  const nativeDialog = document.getElementById('test-native-dialog');
  if (nativeDialog) nativeDialog.remove();
  const ariaModal = document.getElementById('test-aria-modal');
  if (ariaModal) ariaModal.remove();
  const help = document.getElementById('test-help-dialog');
  if (help) help.remove();
  const panel = document.getElementById('test-custom-panel');
  if (panel) panel.remove();
});

TEST_F(InputHandlerTest, HasDOMSelectionOrActiveInputDetectsShadowDOMInput) {
  CreateShadowInputForTest();

  EXPECT_EQ(HasDOMSelectionOrActiveInput(), 1);

  EmscriptenKeyboardEvent event;
  memset(&event, 0, sizeof(event));
  strncpy(event.code, "KeyW", sizeof(event.code) - 1);
  EXPECT_FALSE(HandleKeyDown(0, &event, nullptr));
  EXPECT_FALSE(HandleKeyUp(0, &event, nullptr));

  CleanupShadowElementsForTest();
}

TEST_F(InputHandlerTest,
       HasDOMSelectionOrActiveInputDetectsNestedShadowDOMInput) {
  CreateNestedShadowInputForTest();

  EXPECT_EQ(HasDOMSelectionOrActiveInput(), 1);

  CleanupShadowElementsForTest();
}

TEST_F(InputHandlerTest, IsModalDialogOpenDetectsNativeAndAriaModals) {
  EXPECT_EQ(IsModalDialogOpen(), 0);

  CreateNativeDialogForTest();
  EXPECT_EQ(IsModalDialogOpen(), 1);
  CleanupModalDialogsForTest();

  EXPECT_EQ(IsModalDialogOpen(), 0);

  CreateAriaModalForTest();
  EXPECT_EQ(IsModalDialogOpen(), 1);
  CleanupModalDialogsForTest();

  EXPECT_EQ(IsModalDialogOpen(), 0);
}

TEST_F(InputHandlerTest, IsModalDialogOpenDetectsHelpDialogAndCustomPanel) {
  EXPECT_EQ(IsModalDialogOpen(), 0);

  CreateTraceViewerHelpDialogForTest(/*withShadowMdDialog=*/false);
  EXPECT_EQ(IsModalDialogOpen(), 1);
  CleanupModalDialogsForTest();

  CreateTraceViewerHelpDialogForTest(/*withShadowMdDialog=*/true);
  EXPECT_EQ(IsModalDialogOpen(), 1);
  CleanupModalDialogsForTest();

  CreateTraceViewerCustomizationPanelForTest();
  EXPECT_EQ(IsModalDialogOpen(), 1);
  CleanupModalDialogsForTest();

  EXPECT_EQ(IsModalDialogOpen(), 0);
}

TEST_F(InputHandlerTest, ModalDialogSuppressesKeyAndWheelEvents) {
  CreateTraceViewerHelpDialogForTest(/*withShadowMdDialog=*/true);
  EXPECT_EQ(IsModalDialogOpen(), 1);

  EmscriptenKeyboardEvent key_event;
  memset(&key_event, 0, sizeof(key_event));
  strncpy(key_event.code, "KeyW", sizeof(key_event.code) - 1);

  EXPECT_FALSE(HandleKeyDown(0, &key_event, nullptr));
  EXPECT_FALSE(HandleKeyUp(0, &key_event, nullptr));

  EmscriptenWheelEvent wheel_event;
  memset(&wheel_event, 0, sizeof(wheel_event));
  wheel_event.deltaY = 100.0;
  EXPECT_FALSE(HandleWheel(0, &wheel_event, nullptr));

  CleanupModalDialogsForTest();
}

}  // namespace
}  // namespace traceviewer
