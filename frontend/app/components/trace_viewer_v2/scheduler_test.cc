#include "xprof/frontend/app/components/trace_viewer_v2/scheduler.h"

#include <emscripten.h>

#include "<gtest/gtest.h>"

namespace traceviewer {

namespace {

class SchedulerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Scheduler::Instance().Reset();
  }
};

TEST_F(SchedulerTest, CallbackIsCalledOnlyOnceForMultipleRequests) {
  int call_count = 0;
  Scheduler::Instance().SetMainLoopCallback([&]() { call_count++; });

  Scheduler::Instance().RequestRedraw();
  Scheduler::Instance().RequestRedraw();
  Scheduler::Instance().RequestRedraw();

  // Sleep for 100ms to allow RAF to fire.
  // This requires Asyncify (-sASYNCIFY=1).
  emscripten_sleep(100);

  EXPECT_EQ(call_count, 1);
}

TEST_F(SchedulerTest, CallbackIsCalledAgainIfRequestedAfterFrame) {
  int call_count = 0;
  Scheduler::Instance().SetMainLoopCallback([&]() { call_count++; });

  Scheduler::Instance().RequestRedraw();

  emscripten_sleep(100);

  EXPECT_EQ(call_count, 1);

  Scheduler::Instance().RequestRedraw();
  emscripten_sleep(100);

  EXPECT_EQ(call_count, 2);
}

}  // namespace
}  // namespace traceviewer
