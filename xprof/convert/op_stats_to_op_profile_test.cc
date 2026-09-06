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

#include "xprof/convert/op_stats_to_op_profile.h"

#include <cstdint>
#include <string>

#include "<gtest/gtest.h>"
#include "xprof/convert/op_profile_builder.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::op_profile::Node;

// Helper to create a dummy OpMetrics
OpMetrics CreateOpMetrics(const std::string& name, uint64_t time,
                          const std::string& category) {
  OpMetrics op;
  op.set_name(name);
  op.set_time_ps(time);
  op.set_self_time_ps(time);
  op.set_category(category);
  op.set_occurrences(1);
  return op;
}

TEST(OpStatsToOpProfileTest, SimpleProfileByCategory) {
  OpStats op_stats;
  OpMetricsDb& db = *op_stats.mutable_device_op_metrics_db();
  db.set_total_time_ps(1000);
  db.set_total_op_time_ps(800);

  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(10.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);

  auto op = CreateOpMetrics("op1", 500, "convolution");
  *db.add_metrics_db() = op;

  op_profile::Profile profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
                            OpProfileGrouping::kByCategory);

  ASSERT_TRUE(profile.has_by_category());
  const auto& by_cat = profile.by_category();
  ASSERT_EQ(by_cat.children_size(), 1);
  EXPECT_EQ(by_cat.children(0).name(), "convolution");
}

TEST(OpStatsToOpProfileTest, DeduplicationGroupingWithAndWithoutDuplicates) {
  OpStats op_stats;
  OpMetricsDb& db = *op_stats.mutable_device_op_metrics_db();
  db.set_total_time_ps(10000);
  db.set_total_op_time_ps(8000);

  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(10.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);

  // Category 1: "convolution" with duplicate ops and a single op without
  // duplicates. Duplicate ops sharing deduplicated_name "conv_dedup".
  auto conv_op1 = CreateOpMetrics("conv_op_1", 500, "convolution");
  conv_op1.set_deduplicated_name("conv_dedup");
  *db.add_metrics_db() = conv_op1;

  auto conv_op2 = CreateOpMetrics("conv_op_2", 400, "convolution");
  conv_op2.set_deduplicated_name("conv_dedup");
  *db.add_metrics_db() = conv_op2;

  // Single op with empty deduplicated_name in the same category.
  auto conv_single = CreateOpMetrics("conv_single_op", 300, "convolution");
  conv_single.set_deduplicated_name("");
  *db.add_metrics_db() = conv_single;

  // Category 2: "fusion" with mixed deduplication.
  auto fusion_op1 = CreateOpMetrics("fusion_op_1", 600, "fusion");
  fusion_op1.set_deduplicated_name("fusion_dedup");
  *db.add_metrics_db() = fusion_op1;

  auto fusion_op2 = CreateOpMetrics("fusion_op_2", 200, "fusion");
  fusion_op2.set_deduplicated_name("fusion_dedup");
  *db.add_metrics_db() = fusion_op2;

  // Single op with empty deduplicated_name in "fusion".
  auto fusion_single = CreateOpMetrics("fusion_single_op", 700, "fusion");
  fusion_single.set_deduplicated_name("");
  *db.add_metrics_db() = fusion_single;

  // Category 3: "dense" with only a single op having empty deduplicated_name.
  auto dense_single = CreateOpMetrics("dense_single_op", 800, "dense");
  dense_single.set_deduplicated_name("");
  *db.add_metrics_db() = dense_single;

  op_profile::Profile profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
                            OpProfileGrouping::kByCategory);

  ASSERT_TRUE(profile.has_by_category());
  const auto& by_cat = profile.by_category();

  // Root should have 3 categories: convolution, fusion, dense.
  ASSERT_EQ(by_cat.children_size(), 3);

  // Find category nodes.
  const Node* conv_cat = nullptr;
  const Node* fusion_cat = nullptr;
  const Node* dense_cat = nullptr;
  for (const auto& cat_node : by_cat.children()) {
    if (cat_node.name() == "convolution") conv_cat = &cat_node;
    if (cat_node.name() == "fusion") fusion_cat = &cat_node;
    if (cat_node.name() == "dense") dense_cat = &cat_node;
  }

  ASSERT_NE(conv_cat, nullptr);
  ASSERT_NE(fusion_cat, nullptr);
  ASSERT_NE(dense_cat, nullptr);

  // Validate "convolution" category:
  // Should have 2 children: 1 deduplication group node and 1 standalone single
  // op node.
  ASSERT_EQ(conv_cat->children_size(), 2);
  const Node* conv_dedup_node = nullptr;
  const Node* conv_single_node = nullptr;
  for (const auto& child : conv_cat->children()) {
    if (child.name() == "conv_op_1 and its duplicate(s)") {
      conv_dedup_node = &child;
    } else if (child.name() == "conv_single_op") {
      conv_single_node = &child;
    }
  }
  ASSERT_NE(conv_dedup_node, nullptr);
  EXPECT_EQ(conv_dedup_node->children_size(), 2);
  EXPECT_EQ(conv_dedup_node->children(0).name(), "conv_op_1");
  EXPECT_EQ(conv_dedup_node->children(1).name(), "conv_op_2");

  ASSERT_NE(conv_single_node, nullptr);
  // Single op wrapper is flattened directly into category without "... and its
  // duplicate(s)" wrapper.
  EXPECT_EQ(conv_single_node->children_size(), 0);

  // Validate "fusion" category:
  // Should have 2 children: 1 deduplication group and 1 standalone single op
  // node.
  ASSERT_EQ(fusion_cat->children_size(), 2);
  const Node* fusion_dedup_node = nullptr;
  const Node* fusion_single_node = nullptr;
  for (const auto& child : fusion_cat->children()) {
    if (child.name() == "fusion_op_1 and its duplicate(s)") {
      fusion_dedup_node = &child;
    } else if (child.name() == "fusion_single_op") {
      fusion_single_node = &child;
    }
  }
  ASSERT_NE(fusion_dedup_node, nullptr);
  EXPECT_EQ(fusion_dedup_node->children_size(), 2);
  EXPECT_EQ(fusion_dedup_node->children(0).name(), "fusion_op_1");
  EXPECT_EQ(fusion_dedup_node->children(1).name(), "fusion_op_2");

  ASSERT_NE(fusion_single_node, nullptr);
  EXPECT_EQ(fusion_single_node->children_size(), 0);

  // Validate "dense" category:
  // Should have 1 child which is flattened directly without deduplication
  // grouping node.
  ASSERT_EQ(dense_cat->children_size(), 1);
  EXPECT_EQ(dense_cat->children(0).name(), "dense_single_op");
  EXPECT_EQ(dense_cat->children(0).children_size(), 0);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
