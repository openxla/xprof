/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xprof/convert/op_stats_to_overview_page.h"

#include "<gtest/gtest.h>"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/overview_page.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(OpStatsToOverviewPageTest, TpuDutyCycle) {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_device_type("TPU");
  op_stats.mutable_device_op_metrics_db()->set_busy_time_ps(70);
  op_stats.mutable_device_op_metrics_db()->set_idle_time_ps(30);

  OverviewPage overview_page = ConvertOpStatsToOverviewPage(op_stats);

  EXPECT_DOUBLE_EQ(overview_page.analysis().device_duty_cycle_percent(), 70.0);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
