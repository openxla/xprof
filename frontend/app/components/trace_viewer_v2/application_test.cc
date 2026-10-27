#include "frontend/app/components/trace_viewer_v2/application.h"

#include <emscripten/bind.h>
#include <emscripten/em_asm.h>
#include <emscripten/emscripten.h>
#include <emscripten/val.h>

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
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
    EM_ASM({ window.localStorage.clear(); });
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

TEST_F(ApplicationTest, EnableCustomizationFlagReadsFromJs) {
  Application& app = Application::Instance();
  app.Shutdown();

  EM_ASM({
    window.getFeatureFlag = (name) => {
      return name === 'enable_customization';
    };
  });

  EXPECT_TRUE(app.IsFeatureEnabled("enable_customization"));
}

TEST_F(ApplicationTest, NavigationMethodsSafeWhenUninitialized) {
  Application& app = Application::Instance();
  app.Shutdown();

  EXPECT_FALSE(app.IsInitialized());

  // Calling search navigation methods when timeline is uninitialized should be
  // safe no-ops without crashing.
  app.NavigateToNextSearchResult();
  app.NavigateToPrevSearchResult();
}

TEST_F(ApplicationTest, SaveAndLoadTrackOrderUseOneKeyForAllSessions) {
  EXPECT_EQ(Application::LoadTrackOrderFromLocalStorage(), "");

  constexpr absl::string_view kOrder =
      R"({"mode":"alphabetical_asc","orders":[{"parent":[["TPU:0",0]],"order":[["XLU",0],["MXU",0]]}]})";
  Application::SaveTrackOrderToLocalStorage(kOrder);
  EXPECT_EQ(Application::LoadTrackOrderFromLocalStorage(), kOrder);
  // The order is saved under a single key, shared by all sessions.
  EXPECT_EQ(EM_ASM_INT({ return window.localStorage.length; }), 1);
  EXPECT_TRUE(EM_ASM_INT({
    return window.localStorage.getItem('xprof_track_order') !== null ? 1 : 0;
  }));

  // Saving no custom order (an empty string) removes the saved order.
  Application::SaveTrackOrderToLocalStorage("");
  EXPECT_EQ(Application::LoadTrackOrderFromLocalStorage(), "");
}

TEST_F(ApplicationTest, TrackOrderMethodsAreSafeWhenUninitialized) {
  Application& app = Application::Instance();
  app.Shutdown();

  EXPECT_FALSE(app.IsInitialized());
  EXPECT_FALSE(app.SetTrackOrderMode("alphabetical_asc"));
  EXPECT_FALSE(app.MoveTrack(/*source=*/1, /*target=*/0, /*after=*/false));
  app.ResetTrackOrder();
  app.SetTrackReorderingEnabled(true);
  app.RestoreSavedTrackOrder();

  EXPECT_EQ(app.GetTrackList()["length"].as<int>(), 0);
  const emscripten::val state = app.GetTrackOrderState();
  EXPECT_FALSE(state["enabled"].as<bool>());
  EXPECT_EQ(state["mode"].as<std::string>(), "custom");
  EXPECT_FALSE(state["hasCustomOrder"].as<bool>());
}

}  // namespace
}  // namespace traceviewer
