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

#include "xprof/convert/smart_suggestion/gpu_low_occupancy_rule.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xprof/convert/smart_suggestion/mock_tool_data_provider.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "plugin/xprof/protobuf/kernel_stats.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::Return;
using ::testing::status::IsOkAndHolds;

void AddKernel(OpStats& op_stats, const std::string& name, float occupancy_pct,
               uint64_t total_duration_ns, uint32_t registers_per_thread = 0,
               uint32_t static_shmem_bytes = 0,
               uint32_t dynamic_shmem_bytes = 0) {
  KernelReport* report = op_stats.mutable_kernel_stats_db()->add_reports();
  report->set_name(name);
  report->set_occupancy_pct(occupancy_pct);
  report->set_total_duration_ns(total_duration_ns);
  report->set_registers_per_thread(registers_per_thread);
  report->set_static_shmem_bytes(static_shmem_bytes);
  report->set_dynamic_shmem_bytes(dynamic_shmem_bytes);
}

absl::StatusOr<std::optional<SmartSuggestion>> ApplyRule(
    const OpStats& op_stats) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  EXPECT_CALL(*mock_tool_data_provider, GetOpStats())
      .WillRepeatedly(Return(&op_stats));
  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  GpuLowOccupancyRule rule;
  return rule.Apply(signal_provider);
}

TEST(GpuLowOccupancyRuleTest, FiresAndListsLongestLowOccupancyKernels) {
  OpStats op_stats;
  AddKernel(op_stats, "gemm_fusion_dot_1", /*occupancy_pct=*/12.5,
            /*total_duration_ns=*/200, /*registers_per_thread=*/212,
            /*static_shmem_bytes=*/0, /*dynamic_shmem_bytes=*/98304);
  AddKernel(op_stats, "small_kernel<float>", /*occupancy_pct=*/9.375,
            /*total_duration_ns=*/100, /*registers_per_thread=*/168);
  AddKernel(op_stats, "loop_fusion", /*occupancy_pct=*/100,
            /*total_duration_ns=*/700);
  // Unknown occupancy does not count toward either share.
  AddKernel(op_stats, "unknown_occupancy", /*occupancy_pct=*/0,
            /*total_duration_ns=*/5000);

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      ApplyRule(op_stats);

  ASSERT_THAT(suggestion, IsOkAndHolds(Not(Eq(std::nullopt))));
  EXPECT_EQ((*suggestion)->rule_name(), "GpuLowOccupancyRule");
  const std::string& text = (*suggestion)->suggestion_text();
  EXPECT_THAT(text, HasSubstr("<b>30.0%</b> of GPU kernel time"));
  EXPECT_THAT(text,
              HasSubstr("<li><code>gemm_fusion_dot_1</code>: 12.5% "
                        "occupancy, 212 registers/thread, 98304 bytes "
                        "shared memory/block, 20.0% of kernel time</li>"));
  // Kernel names are HTML-escaped.
  EXPECT_THAT(text, HasSubstr("<code>small_kernel&lt;float&gt;</code>"));
  EXPECT_LT(text.find("gemm_fusion_dot_1"), text.find("small_kernel"));
  EXPECT_THAT(text, Not(HasSubstr("loop_fusion")));
  EXPECT_THAT(text, Not(HasSubstr("unknown_occupancy")));
}

TEST(GpuLowOccupancyRuleTest, ListsAtMostThreeKernels) {
  OpStats op_stats;
  for (int i = 0; i < 5; ++i) {
    AddKernel(op_stats, absl::StrCat("low_", i), /*occupancy_pct=*/10,
              /*total_duration_ns=*/100 + i);
  }

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      ApplyRule(op_stats);

  ASSERT_THAT(suggestion, IsOkAndHolds(Not(Eq(std::nullopt))));
  const std::string& text = (*suggestion)->suggestion_text();
  EXPECT_THAT(text, HasSubstr("<code>low_4</code>"));
  EXPECT_THAT(text, HasSubstr("<code>low_2</code>"));
  EXPECT_THAT(text, Not(HasSubstr("<code>low_1</code>")));
}

TEST(GpuLowOccupancyRuleTest, GroupsReportsOfTheSameKernel) {
  // One kernel launched with two configurations (or from two ops) has two
  // reports; it is listed once, with the longest report's launch parameters
  // and the time of both.
  OpStats op_stats;
  AddKernel(op_stats, "gemm", /*occupancy_pct=*/12.5,
            /*total_duration_ns=*/100, /*registers_per_thread=*/212);
  AddKernel(op_stats, "gemm", /*occupancy_pct=*/12.5,
            /*total_duration_ns=*/300, /*registers_per_thread=*/168);
  AddKernel(op_stats, "other_low", /*occupancy_pct=*/10,
            /*total_duration_ns=*/350);
  AddKernel(op_stats, "high", /*occupancy_pct=*/100,
            /*total_duration_ns=*/250);

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      ApplyRule(op_stats);

  ASSERT_THAT(suggestion, IsOkAndHolds(Not(Eq(std::nullopt))));
  const std::string& text = (*suggestion)->suggestion_text();
  EXPECT_THAT(text,
              HasSubstr("<li><code>gemm</code>: 12.5% occupancy, 168 "
                        "registers/thread, 0 bytes shared memory/block, "
                        "40.0% of kernel time</li>"));
  EXPECT_EQ(text.find("<code>gemm</code>"), text.rfind("<code>gemm</code>"));
  EXPECT_LT(text.find("<code>gemm</code>"), text.find("other_low"));
}

TEST(GpuLowOccupancyRuleTest, DoesNotFireBelowTimeShareThreshold) {
  OpStats op_stats;
  AddKernel(op_stats, "low", /*occupancy_pct=*/12.5,
            /*total_duration_ns=*/90);
  AddKernel(op_stats, "high", /*occupancy_pct=*/50,
            /*total_duration_ns=*/910);

  EXPECT_THAT(ApplyRule(op_stats), IsOkAndHolds(Eq(std::nullopt)));
}

TEST(GpuLowOccupancyRuleTest, DoesNotFireWhenOccupancyIsUnknown) {
  // Some GPUs report no occupancy, so every kernel has occupancy_pct 0.
  OpStats op_stats;
  AddKernel(op_stats, "nvjet_kernel", /*occupancy_pct=*/0,
            /*total_duration_ns=*/1000);

  EXPECT_THAT(ApplyRule(op_stats), IsOkAndHolds(Eq(std::nullopt)));
}

TEST(GpuLowOccupancyRuleTest, DoesNotFireWithoutKernels) {
  // TPU profiles have no kernel stats.
  OpStats op_stats;

  EXPECT_THAT(ApplyRule(op_stats), IsOkAndHolds(Eq(std::nullopt)));
}

TEST(GpuLowOccupancyRuleTest, DoesNotFireWhenOpStatsIsUnavailable) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  EXPECT_CALL(*mock_tool_data_provider, GetOpStats())
      .WillRepeatedly(Return(absl::NotFoundError("no op stats")));
  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  GpuLowOccupancyRule rule;

  EXPECT_THAT(rule.Apply(signal_provider), IsOkAndHolds(Eq(std::nullopt)));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
