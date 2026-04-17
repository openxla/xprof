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
TEST(OpStatsToOverviewPageTest, RooflineMetrics) {
  OpStats op_stats;
  OpMetricsDb* hlo_db = op_stats.mutable_hlo_metrics_db_complete_steps_only();
  hlo_db->set_total_time_ps(1000000);  // 1 us

  OpMetrics* metrics = hlo_db->add_metrics_db();
  metrics->set_occurrences(1);
  metrics->set_category("convolution");
  metrics->set_flops_v2(1000);
  metrics->set_time_ps(1000000);

  auto* breakdown = metrics->add_memory_accessed_breakdown();
  breakdown->set_memory_space(1);  // HBM
  breakdown->set_operation_type(OpMetrics::MemoryAccessed::READ);
  breakdown->set_bytes_accessed(500);

  breakdown = metrics->add_memory_accessed_breakdown();
  breakdown->set_memory_space(1);  // HBM
  breakdown->set_operation_type(OpMetrics::MemoryAccessed::WRITE);
  breakdown->set_bytes_accessed(500);

  OverviewPageAnalysis analysis;
  ComputeTpuAnalysisResult(op_stats, &analysis, TpuPerformanceLimits{2.0, 2.0});

  EXPECT_DOUBLE_EQ(
      analysis.flop_rate_utilization_relative_to_roofline_percent(), 50.0);
  EXPECT_DOUBLE_EQ(
      analysis.memory_bw_utilization_relative_to_hw_limit_percent(), 50.0);
}

TEST(OpStatsToOverviewPageTest, RooflineMetrics_FilteringAndAggregation) {
  OpStats op_stats;
  OpMetricsDb* hlo_db = op_stats.mutable_hlo_metrics_db_complete_steps_only();
  hlo_db->set_total_time_ps(2000000);  // 2 us

  // Valid Op 1
  OpMetrics* metrics1 = hlo_db->add_metrics_db();
  metrics1->set_occurrences(1);
  metrics1->set_category("convolution");
  metrics1->set_flops_v2(1000);
  metrics1->set_time_ps(1000000);
  auto* breakdown1 = metrics1->add_memory_accessed_breakdown();
  breakdown1->set_memory_space(1);  // HBM
  breakdown1->set_operation_type(OpMetrics::MemoryAccessed::READ);
  breakdown1->set_bytes_accessed(500);
  breakdown1 = metrics1->add_memory_accessed_breakdown();
  breakdown1->set_memory_space(1);  // HBM
  breakdown1->set_operation_type(OpMetrics::MemoryAccessed::WRITE);
  breakdown1->set_bytes_accessed(500);

  // Filtered Op (while)
  OpMetrics* metrics2 = hlo_db->add_metrics_db();
  metrics2->set_occurrences(1);
  metrics2->set_category("while");
  metrics2->set_flops_v2(5000);
  metrics2->set_time_ps(5000000);

  // Ignored Op (0 occurrences)
  OpMetrics* metrics3 = hlo_db->add_metrics_db();
  metrics3->set_occurrences(0);
  metrics3->set_category("convolution");
  metrics3->set_flops_v2(2000);

  // Valid Op 2
  OpMetrics* metrics4 = hlo_db->add_metrics_db();
  metrics4->set_occurrences(1);
  metrics4->set_category("convolution");
  metrics4->set_flops_v2(1000);
  metrics4->set_time_ps(1000000);
  auto* breakdown4 = metrics4->add_memory_accessed_breakdown();
  breakdown4->set_memory_space(1);  // HBM
  breakdown4->set_operation_type(OpMetrics::MemoryAccessed::READ);
  breakdown4->set_bytes_accessed(500);
  breakdown4 = metrics4->add_memory_accessed_breakdown();
  breakdown4->set_memory_space(1);  // HBM
  breakdown4->set_operation_type(OpMetrics::MemoryAccessed::WRITE);
  breakdown4->set_bytes_accessed(500);

  OverviewPageAnalysis analysis;
  ComputeTpuAnalysisResult(op_stats, &analysis, TpuPerformanceLimits{2.0, 2.0});

  EXPECT_DOUBLE_EQ(
      analysis.flop_rate_utilization_relative_to_roofline_percent(), 50.0);
  EXPECT_DOUBLE_EQ(
      analysis.memory_bw_utilization_relative_to_hw_limit_percent(), 50.0);
}

TEST(OpStatsToOverviewPageTest, RooflineMetrics_MemoryBound) {
  OpStats op_stats;
  OpMetricsDb* hlo_db = op_stats.mutable_hlo_metrics_db_complete_steps_only();
  hlo_db->set_total_time_ps(1000000);  // 1 us

  OpMetrics* metrics = hlo_db->add_metrics_db();
  metrics->set_occurrences(1);
  metrics->set_category("convolution");
  metrics->set_flops_v2(500);  // Low FLOPs
  metrics->set_time_ps(1000000);

  auto* breakdown = metrics->add_memory_accessed_breakdown();
  breakdown->set_memory_space(1);  // HBM
  breakdown->set_operation_type(OpMetrics::MemoryAccessed::READ);
  breakdown->set_bytes_accessed(1000);  // High bytes

  breakdown = metrics->add_memory_accessed_breakdown();
  breakdown->set_memory_space(1);  // HBM
  breakdown->set_operation_type(OpMetrics::MemoryAccessed::WRITE);
  breakdown->set_bytes_accessed(1000);

  OverviewPageAnalysis analysis;
  ComputeTpuAnalysisResult(op_stats, &analysis, TpuPerformanceLimits{2.0, 4.0});

  EXPECT_DOUBLE_EQ(
      analysis.flop_rate_utilization_relative_to_roofline_percent(), 50.0);
  EXPECT_DOUBLE_EQ(
      analysis.memory_bw_utilization_relative_to_hw_limit_percent(), 50.0);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
