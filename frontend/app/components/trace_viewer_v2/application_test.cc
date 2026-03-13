#include "frontend/app/components/trace_viewer_v2/application.h"

#include <emscripten/bind.h>
#include <emscripten/emscripten.h>

#include "<gtest/gtest.h>"
#include "imgui.h"

namespace traceviewer {
namespace {

class ApplicationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // We can't fully initialize the application because it requires WebGPU,
    // but we can test the Shutdown logic and state cleanup.
  }

  void TearDown() override {
    // Ensure we don't leave ImGui context behind if a test fails.
    if (ImGui::GetCurrentContext()) {
      ImGui::DestroyContext();
    }
  }
};

TEST_F(ApplicationTest, ShutdownResetsState) {
  Application& app = Application::Instance();

  // Initially it should not be initialized (assuming this test runs first
  // or after a shutdown).
  app.Shutdown();

  EXPECT_FALSE(app.IsInitialized());

  // We can't easily call Initialize() here because it will try to call
  // emscripten_webgpu_get_device() and other WebGPU functions which will fail.
  // However, we can verify that Shutdown is safe to call multiple times.
  app.Shutdown();

  EXPECT_FALSE(app.IsInitialized());
}

TEST_F(ApplicationTest, ShutdownClearsImGuiContext) {
  Application& app = Application::Instance();

  // Manually create an ImGui context to simulate an initialized app.
  ImGui::CreateContext();
  EXPECT_NE(ImGui::GetCurrentContext(), nullptr);

  app.Shutdown();

  EXPECT_EQ(ImGui::GetCurrentContext(), nullptr);

  EXPECT_FALSE(app.IsInitialized());
}

}  // namespace
}  // namespace traceviewer
