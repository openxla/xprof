#include "frontend/app/components/trace_viewer_v2/application.h"

#include <emscripten/bind.h>
#include <emscripten/em_asm.h>
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

TEST_F(ApplicationTest, IsFeatureEnabledReadsFromJs) {
  Application& app = Application::Instance();
  app.Shutdown();

  EM_ASM({
    window.getFeatureFlag = (name) => {
      if (name === 'test_flag_true') return true;
      if (name === 'test_flag_false') return false;
      return false;
    };
  });

  EXPECT_TRUE(app.IsFeatureEnabled("test_flag_true"));
  EXPECT_FALSE(app.IsFeatureEnabled("test_flag_false"));
}

TEST_F(ApplicationTest, IsFeatureEnabledCachesValue) {
  Application& app = Application::Instance();
  app.Shutdown();

  EM_ASM({
    window.getFeatureFlagCallCount = 0;
    window.getFeatureFlag = (name) => {
      window.getFeatureFlagCallCount++;
      return name === 'test_flag_cached';
    };
  });

  // First call should call JS and cache the result.
  EXPECT_TRUE(app.IsFeatureEnabled("test_flag_cached"));
  EXPECT_EQ(EM_ASM_INT({ return window.getFeatureFlagCallCount; }), 1);

  // Second call should return cached result and not call JS.
  EXPECT_TRUE(app.IsFeatureEnabled("test_flag_cached"));
  EXPECT_EQ(EM_ASM_INT({ return window.getFeatureFlagCallCount; }), 1);
}

}  // namespace
}  // namespace traceviewer
