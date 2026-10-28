#include "frontend/app/components/trace_viewer_v2/application.h"

#include <emscripten/bind.h>
#include <emscripten/em_asm.h>
#include <emscripten/emscripten.h>

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

TEST_F(ApplicationTest, SaveAndLoadTrackOrderUseSeparateKeysPerViewer) {
  Application& app = Application::Instance();
  app.Shutdown();

  EXPECT_EQ(app.track_order_storage_key(),
            Application::kDefaultTrackOrderStorageKey);
  EXPECT_EQ(app.LoadTrackOrderFromLocalStorage(), "");

  constexpr absl::string_view kTraceOrder =
      R"({"orders":[{"parent":[["TPU:0",0]],"order":[["XLA Ops",0],["Steps",0]]}]})";
  app.SaveTrackOrderToLocalStorage(kTraceOrder);
  EXPECT_EQ(app.LoadTrackOrderFromLocalStorage(), kTraceOrder);
  EXPECT_EQ(EM_ASM_INT({ return window.localStorage.length; }), 1);
  EXPECT_TRUE(EM_ASM_INT({
    return window.localStorage.getItem('xprof_trace_viewer_track_order') !==
            null
        ? 1
        : 0;
  }));

  // Switching to the Kernel Viewer key isolates its saved order from the
  // Trace Viewer's saved order.
  app.SetTrackOrderStorageKey("xprof_kernel_viewer_track_order");
  EXPECT_EQ(app.track_order_storage_key(), "xprof_kernel_viewer_track_order");
  EXPECT_EQ(app.LoadTrackOrderFromLocalStorage(), "");

  constexpr absl::string_view kKernelOrder =
      R"({"orders":[{"parent":[["TPU:0",0]],"order":[["XLU",0],["MXU",0]]}]})";
  app.SaveTrackOrderToLocalStorage(kKernelOrder);
  EXPECT_EQ(app.LoadTrackOrderFromLocalStorage(), kKernelOrder);
  EXPECT_EQ(EM_ASM_INT({ return window.localStorage.length; }), 2);
  EXPECT_TRUE(EM_ASM_INT({
    return window.localStorage.getItem('xprof_kernel_viewer_track_order') !==
            null
        ? 1
        : 0;
  }));

  // Saving an empty string removes only the active viewer's key.
  app.SaveTrackOrderToLocalStorage("");
  EXPECT_EQ(app.LoadTrackOrderFromLocalStorage(), "");
  EXPECT_EQ(EM_ASM_INT({ return window.localStorage.length; }), 1);

  // Passing an empty key or shutting down resets to the default Trace Viewer
  // key, where the Trace Viewer order is still intact.
  app.SetTrackOrderStorageKey("");
  EXPECT_EQ(app.track_order_storage_key(),
            Application::kDefaultTrackOrderStorageKey);
  EXPECT_EQ(app.LoadTrackOrderFromLocalStorage(), kTraceOrder);

  app.SetTrackOrderStorageKey("xprof_kernel_viewer_track_order");
  app.Shutdown();
  EXPECT_EQ(app.track_order_storage_key(),
            Application::kDefaultTrackOrderStorageKey);
  app.SaveTrackOrderToLocalStorage("");
  EXPECT_EQ(app.LoadTrackOrderFromLocalStorage(), "");
}

TEST_F(ApplicationTest, TrackOrderMethodsAreSafeWhenUninitialized) {
  Application& app = Application::Instance();
  app.Shutdown();

  EXPECT_FALSE(app.IsInitialized());

  app.SetTrackOrderStorageKey("xprof_kernel_viewer_track_order");
  app.SetTrackReorderingEnabled(true);
  app.RestoreSavedTrackOrder();
}

}  // namespace
}  // namespace traceviewer
