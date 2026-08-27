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

#include <cstdint>
#include <string>

#include "<gtest/gtest.h>"
#include "xprof/convert/flat_op_stats_to_op_profile.h"
#include "xprof/convert/op_profile_builder.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace tensorflow {
namespace profiler {


namespace {

using ::tensorflow::profiler::op_profile::Node;

// Helper to create a dummy FlatOpMetrics
FlatOpMetrics CreateOpMetrics(
    const std::string& name, uint64_t id, uint64_t time,
    const std::string& category, uint64_t parent_id = 0,
    FlatOpMetrics::TpuCoreType core_type = FlatOpMetrics::TENSOR_CORE) {
  FlatOpMetrics op;
  op.set_hlo_name(name);
  op.set_op_id(id);
  op.set_time_ps(time);
  op.set_self_time_ps(time);
  op.set_category(category);
  op.set_parent_op_id(parent_id);
  op.set_core_type(core_type);
  op.set_long_name(name + "_long");
  return op;
}

TEST(FlatOpStatsToOpProfileTest, SimpleProfileByCategory) {
  OpStats op_stats;
  FlatOpMetricsDb& db = *op_stats.mutable_flat_device_op_metrics_db();
  db.set_total_time_ps(1000);
  db.set_total_op_time_ps(800);

  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(10.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);

  *db.add_op_instances() = CreateOpMetrics("op1", 1, 500, "conv");
  *db.add_op_instances() = CreateOpMetrics("op2", 2, 300, "fused");

  op_profile::Profile profile;

  ConvertFlatOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
                                OpProfileGrouping::kByCategory);

  EXPECT_EQ(profile.device_type(), "TPU");
  ASSERT_TRUE(profile.has_by_category());
  const auto& by_cat = profile.by_category();

  // Root should have children corresponding to categories
  ASSERT_EQ(by_cat.children_size(), 2);

  // Check category names
  EXPECT_TRUE(by_cat.children(0).name() == "conv" ||
              by_cat.children(1).name() == "conv");
  EXPECT_TRUE(by_cat.children(0).name() == "fused" ||
              by_cat.children(1).name() == "fused");
}

TEST(FlatOpStatsToOpProfileTest, SimpleProfileByProgram) {
  OpStats op_stats;
  FlatOpMetricsDb& db = *op_stats.mutable_flat_device_op_metrics_db();
  db.set_total_time_ps(1000);
  db.set_total_op_time_ps(800);

  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(10.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);

  auto op1 = CreateOpMetrics("op1", 1, 500, "conv");
  op1.set_hlo_module_id(10);
  *db.add_op_instances() = op1;

  auto op2 = CreateOpMetrics("op2", 2, 300, "fused");
  op2.set_hlo_module_id(20);
  *db.add_op_instances() = op2;

  (*op_stats.mutable_program_id_to_name_map())[10] = "program1";
  (*op_stats.mutable_program_id_to_name_map())[20] = "program2";

  op_profile::Profile profile;

  ConvertFlatOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
                                OpProfileGrouping::kByProgram);

  ASSERT_TRUE(profile.has_by_program());
  const auto& by_prog = profile.by_program();

  // Root should have children corresponding to programs
  EXPECT_EQ(by_prog.children_size(), 2);
}

TEST(FlatOpStatsToOpProfileTest, SparseCoreLinkageDuplication) {
  OpStats op_stats;
  FlatOpMetricsDb& db = *op_stats.mutable_flat_device_op_metrics_db();
  db.set_total_time_ps(1000);
  db.set_total_op_time_ps(800);

  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(10.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);

  // Parent TC op
  *db.add_op_instances() = CreateOpMetrics("fusion_tc", 1, 500, "fusion", 0,
                                           FlatOpMetrics::TENSOR_CORE);

  // Child SC op
  *db.add_op_instances() = CreateOpMetrics("sc_op", 2, 200, "sc_specific", 1,
                                           FlatOpMetrics::SPARSE_CORE);

  op_profile::Profile profile;

  ConvertFlatOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
                                OpProfileGrouping::kByCategory);

  ASSERT_TRUE(profile.has_by_category());
  const auto& by_cat = profile.by_category();

  // We expect both ops to be present at the top level categories,
  // AND sc_op to be duplicated inside fusion_tc.
  // Let's find fusion_tc node.
  const Node* fusion_tc_node = nullptr;
  for (const auto& cat_node : by_cat.children()) {
    if (cat_node.name() == "fusion") {
      for (const auto& op_node : cat_node.children()) {
        if (op_node.name() == "fusion_tc") {
          fusion_tc_node = &op_node;
          break;
        }
      }
    }
    if (fusion_tc_node) break;
  }

  ASSERT_NE(fusion_tc_node, nullptr);
  // Now check if fusion_tc has sc_op as child.
  bool found_sc_child = false;
  for (const auto& child : fusion_tc_node->children()) {
    if (child.name() == "sc_op") {
      found_sc_child = true;
      break;
    }
  }

  EXPECT_TRUE(found_sc_child);
}

TEST(FlatOpStatsToOpProfileTest, HighlyNestedProvenance) {
  OpStats op_stats;
  FlatOpMetricsDb& db = *op_stats.mutable_flat_device_op_metrics_db();
  db.set_total_time_ps(2000);
  db.set_total_op_time_ps(1500);

  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(10.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);

  auto op1 = CreateOpMetrics("op1", 1, 500, "conv");
  op1.set_provenance("layer1/block1/conv1");
  *db.add_op_instances() = op1;

  auto op2 = CreateOpMetrics("op2", 2, 300, "conv");
  op2.set_provenance("layer1/block1/conv2");
  *db.add_op_instances() = op2;

  auto op3 = CreateOpMetrics("op3", 3, 700, "dense");
  op3.set_provenance("layer2/dense1");
  *db.add_op_instances() = op3;

  op_profile::Profile profile;

  ConvertFlatOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
                                OpProfileGrouping::kByProvenance);

  ASSERT_TRUE(profile.has_by_provenance());
  const auto& by_prov = profile.by_provenance();

  // Root should have 1 child (program "main")
  ASSERT_EQ(by_prov.children_size(), 1);
  const auto& main_node = by_prov.children(0);
  EXPECT_EQ(main_node.name(), "main");

  // Program "main" should have children corresponding to top-level provenance
  // parts ("layer1", "layer2")
  EXPECT_EQ(main_node.children_size(), 2);

  const Node* layer1_node = nullptr;
  for (const auto& child : main_node.children()) {
    if (child.name() == "layer1") {
      layer1_node = &child;
      break;
    }
  }

  ASSERT_NE(layer1_node, nullptr);
  // layer1 should have "block1"
  ASSERT_EQ(layer1_node->children_size(), 1);
  EXPECT_EQ(layer1_node->children(0).name(), "block1");
}

TEST(FlatOpStatsToOpProfileTest,
     DeduplicationGroupingWithAndWithoutDuplicates) {
  OpStats op_stats;
  FlatOpMetricsDb& db = *op_stats.mutable_flat_device_op_metrics_db();
  db.set_total_time_ps(10000);
  db.set_total_op_time_ps(8000);

  auto* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(10.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);

  // Category 1: "convolution" with duplicate ops and a single op without
  // duplicates. Duplicate ops sharing deduplicated_name "conv_dedup".
  auto conv_op1 = CreateOpMetrics("conv_op_1", 1, 500, "convolution");
  conv_op1.set_deduplicated_name("conv_dedup");
  *db.add_op_instances() = conv_op1;

  auto conv_op2 = CreateOpMetrics("conv_op_2", 2, 400, "convolution");
  conv_op2.set_deduplicated_name("conv_dedup");
  *db.add_op_instances() = conv_op2;

  // Single op with empty deduplicated_name in the same category.
  auto conv_single = CreateOpMetrics("conv_single_op", 3, 300, "convolution");
  conv_single.set_deduplicated_name("");
  *db.add_op_instances() = conv_single;

  // Category 2: "fusion" with mixed deduplication.
  // Duplicate ops sharing deduplicated_name "fusion_dedup".
  auto fusion_op1 = CreateOpMetrics("fusion_op_1", 4, 600, "fusion");
  fusion_op1.set_deduplicated_name("fusion_dedup");
  *db.add_op_instances() = fusion_op1;

  auto fusion_op2 = CreateOpMetrics("fusion_op_2", 5, 200, "fusion");
  fusion_op2.set_deduplicated_name("fusion_dedup");
  *db.add_op_instances() = fusion_op2;

  // Single op with empty deduplicated_name in "fusion".
  auto fusion_single = CreateOpMetrics("fusion_single_op", 6, 700, "fusion");
  fusion_single.set_deduplicated_name("");
  *db.add_op_instances() = fusion_single;

  // Category 3: "dense" with only a single op having empty deduplicated_name.
  auto dense_single = CreateOpMetrics("dense_single_op", 7, 800, "dense");
  dense_single.set_deduplicated_name("");
  *db.add_op_instances() = dense_single;

  op_profile::Profile profile;
  ConvertFlatOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
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
