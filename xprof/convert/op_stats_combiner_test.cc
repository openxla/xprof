/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/op_stats_combiner.h"

#include <cstdint>
#include <vector>

#include "<gtest/gtest.h>"
#include "absl/container/flat_hash_map.h"
#include "google/protobuf/util/message_differencer.h"
#include "xla/tsl/platform/types.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/power_metrics.pb.h"
#include "plugin/xprof/protobuf/steps_db.pb.h"
#include "xprof/utils/step_intersection.h"

namespace tensorflow {
namespace profiler {
namespace {

// Tests that the run_environment field of the combined op stats is set
// correctly.
TEST(CombineAllOpStatsTest, CombineRunEnvironment) {
  // Construct OpStatsInfo and all_op_stats_info.
  OpStats dst_op_stats, op_stats_1, op_stats_2;
  op_stats_1.mutable_run_environment()
      ->mutable_host_independent_job_info()
      ->set_profile_duration_ms(100);
  op_stats_2.mutable_run_environment()
      ->mutable_host_independent_job_info()
      ->set_profile_duration_ms(0);
  OpStatsInfo op_stats_info_1(&op_stats_1, TPU, 0),
      op_stats_info_2(&op_stats_2, TPU, 0);
  std::vector<OpStatsInfo> all_op_stats_info = {op_stats_info_1,
                                                op_stats_info_2};

  // Construct dummy step_intersection.
  StepDatabaseResult dummy_step_db_result;
  absl::flat_hash_map<uint32_t /*host_id*/, const StepDatabaseResult*> result;
  result.insert({0, &dummy_step_db_result});
  StepIntersection dummy_step_intersection = StepIntersection(1, result);

  // Combine all op stats.
  CombineAllOpStats(all_op_stats_info, dummy_step_intersection, &dst_op_stats);

  // Verify that the profile_duration_ms field of the second object is now set.
  EXPECT_EQ(100, dst_op_stats.run_environment()
                     .host_independent_job_info()
                     .profile_duration_ms());
}

TEST(CombineAllOpStatsTest, CombineRunEnvironmentWithUnknownDevice) {
  OpStats dst_op_stats, op_stats_1, op_stats_2;
  op_stats_1.mutable_run_environment()->set_device_type("TPU");
  op_stats_2.mutable_run_environment()->set_device_type("Device");
  OpStatsInfo op_stats_info_1(&op_stats_1, TPU, 0),
      op_stats_info_2(&op_stats_2, TPU, 0);
  std::vector<OpStatsInfo> all_op_stats_info = {op_stats_info_1,
                                                op_stats_info_2};

  // Construct dummy step_intersection.
  StepDatabaseResult dummy_step_db_result;
  absl::flat_hash_map<uint32_t /*host_id*/, const StepDatabaseResult*> result;
  result.insert({0, &dummy_step_db_result});
  StepIntersection dummy_step_intersection = StepIntersection(1, result);

  CombineAllOpStats(all_op_stats_info, dummy_step_intersection, &dst_op_stats);

  EXPECT_EQ("TPU", dst_op_stats.run_environment().device_type());
}

TEST(CombineAllOpStatsTest, CombinePerfEnvOrderZero) {
  // Ensure CombinePerfEnv behaves consistently regardless of order of op stats.
  OpStats dst_op_stats1, dst_op_stats2, op_stats_1, op_stats_2;
  op_stats_1.mutable_perf_env()->set_peak_tera_flops_per_second(100);
  op_stats_2.mutable_perf_env()->set_peak_tera_flops_per_second(0);
  // Construct dummy step_intersection which is required by CombineAllOpStats().
  absl::flat_hash_map<uint32_t /*host_id*/, const StepDatabaseResult*> result;
  StepIntersection dummy_step_intersection = StepIntersection(1, result);

  OpStatsInfo op_stats_info_1(&op_stats_1, TPU, 0),
      op_stats_info_2(&op_stats_2, TPU, 0);

  // Test order 1.
  std::vector<OpStatsInfo> all_op_stats_info = {op_stats_info_1,
                                                op_stats_info_2};
  CombineAllOpStats(all_op_stats_info, dummy_step_intersection, &dst_op_stats1);
  EXPECT_EQ(100, dst_op_stats1.perf_env().peak_tera_flops_per_second());

  // Test order 2.
  all_op_stats_info = {
      op_stats_info_2,
      op_stats_info_1,
  };
  CombineAllOpStats(all_op_stats_info, dummy_step_intersection, &dst_op_stats2);
  EXPECT_EQ(100, dst_op_stats2.perf_env().peak_tera_flops_per_second());
}

TEST(CombineAllOpStatsTest, CombineRunEnvironmentWithMismatchHardwareType) {
  OpStats coordinator_op_stats, device_op_stats, dst_op_stats;
  coordinator_op_stats.mutable_run_environment()->set_hardware_type(
      HardwareType::CPU_ONLY);
  device_op_stats.mutable_run_environment()->set_hardware_type(
      HardwareType::TPU);
  CombineAllOpStats({OpStatsInfo(&coordinator_op_stats, CPU_ONLY, 0),
                     OpStatsInfo(&device_op_stats, TPU, 1)},
                    StepIntersection(1, {}), &dst_op_stats);
  EXPECT_EQ(dst_op_stats.run_environment().hardware_type(), HardwareType::TPU);
}

TEST(CombineAllOpStatsTest, CombineRunEnvironmentPowerMetricsSingleHost) {
  OpStats dst_op_stats, src_op_stats;
  PowerComponentMetrics* component_metric =
      src_op_stats.mutable_run_environment()
          ->mutable_power_metrics()
          ->add_power_component_metrics();
  component_metric->set_component_name("HBM");
  component_metric->set_max_power(100);
  component_metric->set_avg_power(50);

  OpStatsInfo op_stats_info(&src_op_stats, TPU, 0);
  std::vector<OpStatsInfo> all_op_stats_info = {op_stats_info};

  // Construct dummy step_intersection.
  StepDatabaseResult dummy_step_db_result;
  absl::flat_hash_map<uint32_t /*host_id*/, const StepDatabaseResult*> result;
  result.insert({0, &dummy_step_db_result});
  StepIntersection dummy_step_intersection = StepIntersection(1, result);

  CombineAllOpStats(all_op_stats_info, dummy_step_intersection, &dst_op_stats);

  EXPECT_TRUE(
      google::protobuf::util::MessageDifferencer::Equals(dst_op_stats, src_op_stats));
}

TEST(CombineAllOpStatsTest, CombineRunEnvironmentPowerMetricsMultipleHost) {
  OpStats dst_op_stats, src_op_stats_1, src_op_stats_2;

  // Source OpStats 1
  RunEnvironment* env_1 = src_op_stats_1.mutable_run_environment();
  env_1->mutable_hostnames()->insert({"host1", true});
  PowerComponentMetrics* component_metric_1 =
      src_op_stats_1.mutable_run_environment()
          ->mutable_power_metrics()
          ->add_power_component_metrics();
  component_metric_1->set_component_name("HBM");
  component_metric_1->set_max_power(100);
  component_metric_1->set_avg_power(50);

  // Source OpStats 2
  RunEnvironment* env_2 = src_op_stats_2.mutable_run_environment();
  env_2->mutable_hostnames()->insert({"host2", true});
  PowerComponentMetrics* component_metric_2 =
      src_op_stats_2.mutable_run_environment()
          ->mutable_power_metrics()
          ->add_power_component_metrics();
  component_metric_2->set_component_name("HBM");
  component_metric_2->set_max_power(120);
  component_metric_2->set_avg_power(60);

  OpStatsInfo op_stats_info_1(&src_op_stats_1, TPU, 0);
  OpStatsInfo op_stats_info_2(&src_op_stats_2, TPU, 1);

  std::vector<OpStatsInfo> all_op_stats_info = {op_stats_info_1,
                                                op_stats_info_2};

  // Construct dummy step_intersection.
  StepDatabaseResult dummy_step_db_result;
  absl::flat_hash_map<uint32_t /*host_id*/, const StepDatabaseResult*> result;
  result.insert({0, &dummy_step_db_result});
  StepIntersection dummy_step_intersection = StepIntersection(1, result);

  CombineAllOpStats(all_op_stats_info, dummy_step_intersection, &dst_op_stats);

  const auto& dst_power_metrics =
      dst_op_stats.run_environment().power_metrics();
  EXPECT_EQ(1, dst_power_metrics.power_component_metrics_size());
  EXPECT_EQ("HBM",
            dst_power_metrics.power_component_metrics(0).component_name());
  EXPECT_EQ(120, dst_power_metrics.power_component_metrics(0).max_power());
  // Calculation:
  // Iter 1: dst={} src=op1(50) -> dst=50
  // Iter 2: dst=op1(50) src=op2(60) -> dst = 50 * 1/2 + 60 * 1/2 = 55
  EXPECT_EQ(55, dst_power_metrics.power_component_metrics(0).avg_power());
}

TEST(CombineAllOpStatsTest, CombineDisaggregatedServingStats) {
  OpStats dst_op_stats, src_op_stats_1, src_op_stats_2;

  // Source OpStats 1
  auto* serving_stats_1 =
      src_op_stats_1.mutable_disaggregated_serving_latency();
  serving_stats_1->set_num_decode_steps(10);
  serving_stats_1->mutable_decode_step_time_us()->set_avg(100.0);

  // Source OpStats 2
  auto* serving_stats_2 =
      src_op_stats_2.mutable_disaggregated_serving_latency();
  serving_stats_2->set_num_decode_steps(20);
  serving_stats_2->mutable_decode_step_time_us()->set_avg(200.0);

  OpStatsInfo op_stats_info_1(&src_op_stats_1, TPU, 0);
  OpStatsInfo op_stats_info_2(&src_op_stats_2, TPU, 1);

  std::vector<OpStatsInfo> all_op_stats_info = {op_stats_info_1,
                                                op_stats_info_2};

  // Construct dummy step_intersection.
  StepDatabaseResult dummy_step_db_result;
  absl::flat_hash_map<uint32_t /*host_id*/, const StepDatabaseResult*> result;
  result.insert({0, &dummy_step_db_result});
  StepIntersection dummy_step_intersection = StepIntersection(1, result);

  CombineAllOpStats(all_op_stats_info, dummy_step_intersection, &dst_op_stats);

  const auto& dst_serving_stats = dst_op_stats.disaggregated_serving_latency();
  EXPECT_EQ(30, dst_serving_stats.num_decode_steps());
  // (10 * 100 + 20 * 200) / 30 = 5000 / 30 = 166.666...
  EXPECT_NEAR(166.666666666, dst_serving_stats.decode_step_time_us().avg(),
              1e-6);
}

TEST(CombineAllOpStatsTest, CombineDeviceOpMetricsDbWithChildren) {
  OpStats dst_op_stats, op_stats_1, op_stats_2;

  // Set up op_stats_1 with one device op that has children.
  {
    auto* db = op_stats_1.mutable_device_op_metrics_db();
    auto* op = db->add_metrics_db();
    op->set_name("parent_op");
    op->set_hlo_module_id(123);
    op->set_occurrences(1);
    op->set_time_ps(100);

    auto* child_db = op->mutable_children();
    auto* child_op = child_db->add_metrics_db();
    child_op->set_name("child_op_1");
    child_op->set_hlo_module_id(123);
    child_op->set_occurrences(2);
    child_op->set_time_ps(50);
  }

  // Set up op_stats_2 with the same device op that has other children.
  {
    auto* db = op_stats_2.mutable_device_op_metrics_db();
    auto* op = db->add_metrics_db();
    op->set_name("parent_op");
    op->set_hlo_module_id(123);
    op->set_occurrences(2);
    op->set_time_ps(200);

    auto* child_db = op->mutable_children();
    auto* child_op = child_db->add_metrics_db();
    child_op->set_name("child_op_2");
    child_op->set_hlo_module_id(123);
    child_op->set_occurrences(3);
    child_op->set_time_ps(80);
  }

  OpStatsInfo op_stats_info_1(&op_stats_1, TPU, 0);
  OpStatsInfo op_stats_info_2(&op_stats_2, TPU, 1);

  std::vector<OpStatsInfo> all_op_stats_info = {op_stats_info_1,
                                                op_stats_info_2};

  StepDatabaseResult dummy_step_db_result;
  absl::flat_hash_map<uint32_t /*host_id*/, const StepDatabaseResult*> result;
  result.insert({0, &dummy_step_db_result});
  StepIntersection dummy_step_intersection = StepIntersection(1, result);

  CombineAllOpStats(all_op_stats_info, dummy_step_intersection, &dst_op_stats);

  // Verify that the combined op metrics has the parent_op.
  const auto& combined_db = dst_op_stats.device_op_metrics_db();
  ASSERT_EQ(combined_db.metrics_db_size(), 1);
  const auto& combined_op = combined_db.metrics_db(0);
  EXPECT_EQ(combined_op.name(), "parent_op");
  EXPECT_EQ(combined_op.occurrences(), 3);
  EXPECT_EQ(combined_op.time_ps(), 300);

  // Verify that both child_op_1 and child_op_2 are present in the children db
  // of parent_op.
  const auto& combined_children_db = combined_op.children();
  ASSERT_EQ(combined_children_db.metrics_db_size(), 2);

  // We should look up the children to verify their values since their order in
  // the repeated field might vary.
  absl::flat_hash_map<std::string, const OpMetrics*> children_map;
  for (const auto& child : combined_children_db.metrics_db()) {
    children_map[child.name()] = &child;
  }

  ASSERT_TRUE(children_map.contains("child_op_1"));
  EXPECT_EQ(children_map["child_op_1"]->occurrences(), 2);
  EXPECT_EQ(children_map["child_op_1"]->time_ps(), 50);

  ASSERT_TRUE(children_map.contains("child_op_2"));
  EXPECT_EQ(children_map["child_op_2"]->occurrences(), 3);
  EXPECT_EQ(children_map["child_op_2"]->time_ps(), 80);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
