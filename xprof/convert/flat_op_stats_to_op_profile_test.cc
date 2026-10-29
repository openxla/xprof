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

#include "xprof/convert/flat_op_stats_to_op_profile.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tsl/platform/protobuf.h"
#include "xprof/convert/op_profile_builder.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace tensorflow {
namespace profiler {

namespace {

using ::tensorflow::profiler::op_profile::Node;
using ::testing::HasSubstr;
using ::testing::Not;

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

TEST(FlatOpStatsToOpProfileTest, EmptyFlatMetricsDbHasRootMetricsByCategory) {
  OpStats op_stats;
  op_profile::Profile profile;
  ConvertFlatOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
                                OpProfileGrouping::kByCategory);

  ASSERT_TRUE(profile.has_by_category());
  EXPECT_EQ(profile.by_category().name(), "by_category");
  EXPECT_TRUE(profile.by_category().has_metrics());
  EXPECT_DOUBLE_EQ(profile.by_category().metrics().flops(), 0.0);
  EXPECT_EQ(profile.by_category().metrics().raw_time(), 0);
  EXPECT_EQ(profile.by_category().metrics().raw_flops(), 0);
  EXPECT_EQ(profile.by_category().children_size(), 0);

  ASSERT_TRUE(profile.has_by_category_exclude_idle());
  EXPECT_EQ(profile.by_category_exclude_idle().name(), "by_category");
  EXPECT_TRUE(profile.by_category_exclude_idle().has_metrics());
  EXPECT_DOUBLE_EQ(profile.by_category_exclude_idle().metrics().flops(), 0.0);
  EXPECT_EQ(profile.by_category_exclude_idle().metrics().raw_time(), 0);
  EXPECT_EQ(profile.by_category_exclude_idle().metrics().raw_flops(), 0);
  EXPECT_EQ(profile.by_category_exclude_idle().children_size(), 0);
}

TEST(FlatOpStatsToOpProfileTest, EmptyFlatMetricsDbHasRootMetricsByProgram) {
  OpStats op_stats;
  op_profile::Profile profile;
  ConvertFlatOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
                                OpProfileGrouping::kByProgram);

  ASSERT_TRUE(profile.has_by_program());
  EXPECT_EQ(profile.by_program().name(), "by_program");
  EXPECT_TRUE(profile.by_program().has_metrics());
  EXPECT_DOUBLE_EQ(profile.by_program().metrics().flops(), 0.0);
  EXPECT_EQ(profile.by_program().metrics().raw_time(), 0);
  EXPECT_EQ(profile.by_program().metrics().raw_flops(), 0);
  EXPECT_EQ(profile.by_program().children_size(), 0);

  ASSERT_TRUE(profile.has_by_program_exclude_idle());
  EXPECT_EQ(profile.by_program_exclude_idle().name(), "by_program");
  EXPECT_TRUE(profile.by_program_exclude_idle().has_metrics());
  EXPECT_DOUBLE_EQ(profile.by_program_exclude_idle().metrics().flops(), 0.0);
  EXPECT_EQ(profile.by_program_exclude_idle().metrics().raw_time(), 0);
  EXPECT_EQ(profile.by_program_exclude_idle().metrics().raw_flops(), 0);
  EXPECT_EQ(profile.by_program_exclude_idle().children_size(), 0);
}

TEST(FlatOpStatsToOpProfileTest, EmptyFlatMetricsDbHasRootMetricsByProvenance) {
  OpStats op_stats;
  op_profile::Profile profile;
  ConvertFlatOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
                                OpProfileGrouping::kByProvenance);

  ASSERT_TRUE(profile.has_by_provenance());
  EXPECT_EQ(profile.by_provenance().name(), "by_provenance");
  EXPECT_TRUE(profile.by_provenance().has_metrics());
  EXPECT_DOUBLE_EQ(profile.by_provenance().metrics().flops(), 0.0);
  EXPECT_EQ(profile.by_provenance().metrics().raw_time(), 0);
  EXPECT_EQ(profile.by_provenance().metrics().raw_flops(), 0);
  EXPECT_EQ(profile.by_provenance().children_size(), 0);

  ASSERT_TRUE(profile.has_by_provenance_exclude_idle());
  EXPECT_EQ(profile.by_provenance_exclude_idle().name(), "by_provenance");
  EXPECT_TRUE(profile.by_provenance_exclude_idle().has_metrics());
  EXPECT_DOUBLE_EQ(profile.by_provenance_exclude_idle().metrics().flops(), 0.0);
  EXPECT_EQ(profile.by_provenance_exclude_idle().metrics().raw_time(), 0);
  EXPECT_EQ(profile.by_provenance_exclude_idle().metrics().raw_flops(), 0);
  EXPECT_EQ(profile.by_provenance_exclude_idle().children_size(), 0);
}

TEST(FlatOpStatsToOpProfileTest,
     AttachesServingMetadataToProgramNodesInProgramAndProvenanceViews) {
  OpStats op_stats;
  FlatOpMetricsDb& db = *op_stats.mutable_flat_device_op_metrics_db();
  db.set_total_time_ps(2900);
  db.set_total_op_time_ps(2800);

  PerfEnv* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(10.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);

  FlatOpMetrics op1 = CreateOpMetrics("op1", 1, 700, "conv");
  op1.set_hlo_module_id(10);
  op1.set_provenance("layer1/conv1");
  *db.add_op_instances() = op1;

  FlatOpMetrics op2 = CreateOpMetrics("op2", 2, 600, "fused");
  op2.set_hlo_module_id(20);
  op2.set_provenance("layer2/fused1");
  *db.add_op_instances() = op2;

  FlatOpMetrics op3 = CreateOpMetrics("op3", 3, 500, "dense");
  op3.set_hlo_module_id(30);
  op3.set_provenance("layer3/dense1");
  *db.add_op_instances() = op3;

  FlatOpMetrics op4 = CreateOpMetrics("op4", 4, 400, "copy");
  op4.set_hlo_module_id(40);
  op4.set_provenance("layer4/copy1");
  *db.add_op_instances() = op4;

  FlatOpMetrics op5 = CreateOpMetrics("op5", 5, 300, "transpose");
  op5.set_hlo_module_id(50);
  op5.set_provenance("layer5/transpose1");
  *db.add_op_instances() = op5;

  FlatOpMetrics op6 = CreateOpMetrics("op6", 6, 200, "reduce");
  op6.set_hlo_module_id(60);
  op6.set_provenance("layer6/reduce1");
  *db.add_op_instances() = op6;

  FlatOpMetrics op7 = CreateOpMetrics("op7", 7, 100, "pad");
  op7.set_hlo_module_id(70);
  op7.set_provenance("layer7/pad1");
  *db.add_op_instances() = op7;

  (*op_stats.mutable_program_id_to_name_map())[10] = "program1";
  (*op_stats.mutable_program_id_to_name_map())[20] = "program2";
  (*op_stats.mutable_program_id_to_name_map())[30] = "program3";
  (*op_stats.mutable_program_id_to_name_map())[40] = "program4";
  (*op_stats.mutable_program_id_to_name_map())[50] = "program5";
  (*op_stats.mutable_program_id_to_name_map())[60] = "program6";
  (*op_stats.mutable_program_id_to_name_map())[70] = "program7";

  ProgramServingMetadataMap serving_metadata;
  serving_metadata[10].set_model_id("/models/flat_model:1");
  serving_metadata[10].set_batch_size(8);
  serving_metadata[10].set_num_batches(7);
  // Entries missing a model_id, a positive batch_size, or a positive
  // num_batches are ignored (and program 70 is omitted from the map).
  serving_metadata[20].set_model_id("/models/flat_model:1");
  serving_metadata[20].set_batch_size(0);
  serving_metadata[20].set_num_batches(5);
  serving_metadata[30].set_batch_size(16);
  serving_metadata[30].set_num_batches(3);
  serving_metadata[40].set_model_id("/models/flat_model:1");
  serving_metadata[40].set_batch_size(-1);
  serving_metadata[40].set_num_batches(5);
  serving_metadata[50].set_model_id("/models/flat_model:1");
  serving_metadata[50].set_batch_size(8);
  serving_metadata[50].set_num_batches(0);
  serving_metadata[60].set_model_id("/models/flat_model:1");
  serving_metadata[60].set_batch_size(8);
  serving_metadata[60].set_num_batches(-1);

  op_profile::Profile by_prog_profile;
  ConvertFlatOpStatsToOpProfile(op_stats, HardwareType::TPU, by_prog_profile,
                                100, OpProfileGrouping::kByProgram,
                                &serving_metadata);
  ASSERT_EQ(by_prog_profile.by_program().children_size(), 7);
  EXPECT_TRUE(
      by_prog_profile.by_program().children(0).has_program_serving_metadata());
  EXPECT_EQ(by_prog_profile.by_program()
                .children(0)
                .program_serving_metadata()
                .model_id(),
            "/models/flat_model:1");
  EXPECT_EQ(by_prog_profile.by_program()
                .children(0)
                .program_serving_metadata()
                .batch_size(),
            8);
  EXPECT_EQ(by_prog_profile.by_program()
                .children(0)
                .program_serving_metadata()
                .num_batches(),
            7);
  for (int i = 1; i < 7; ++i) {
    EXPECT_FALSE(by_prog_profile.by_program()
                     .children(i)
                     .has_program_serving_metadata());
  }
  ASSERT_EQ(by_prog_profile.by_program_exclude_idle().children_size(), 7);
  EXPECT_TRUE(by_prog_profile.by_program_exclude_idle()
                  .children(0)
                  .has_program_serving_metadata());

  op_profile::Profile by_prov_profile;
  ConvertFlatOpStatsToOpProfile(op_stats, HardwareType::TPU, by_prov_profile,
                                100, OpProfileGrouping::kByProvenance,
                                &serving_metadata);
  ASSERT_EQ(by_prov_profile.by_provenance().children_size(), 7);
  EXPECT_TRUE(by_prov_profile.by_provenance()
                  .children(0)
                  .has_program_serving_metadata());
  for (int i = 1; i < 7; ++i) {
    EXPECT_FALSE(by_prov_profile.by_provenance()
                     .children(i)
                     .has_program_serving_metadata());
  }
  ASSERT_EQ(by_prov_profile.by_provenance_exclude_idle().children_size(), 7);
  EXPECT_TRUE(by_prov_profile.by_provenance_exclude_idle()
                  .children(0)
                  .has_program_serving_metadata());

  op_profile::Profile by_cat_profile;
  ConvertFlatOpStatsToOpProfile(op_stats, HardwareType::TPU, by_cat_profile,
                                100, OpProfileGrouping::kByCategory,
                                &serving_metadata);
  for (const Node* root :
       {&by_cat_profile.by_category(),
        &by_cat_profile.by_category_exclude_idle()}) {
    EXPECT_FALSE(root->has_program_serving_metadata());
    for (const Node& cat : root->children()) {
      EXPECT_FALSE(cat.has_program_serving_metadata());
    }
  }
}

TEST(FlatOpStatsToOpProfileTest,
     NullMetadataMapAndDefaultArgsProduceIdenticalJson) {
  OpStats op_stats;
  FlatOpMetricsDb& db = *op_stats.mutable_flat_device_op_metrics_db();
  db.set_total_time_ps(1000);
  db.set_total_op_time_ps(800);
  PerfEnv* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(10.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);

  FlatOpMetrics op = CreateOpMetrics("op1", 1, 800, "conv");
  op.set_hlo_module_id(77);
  *db.add_op_instances() = op;
  (*op_stats.mutable_program_id_to_name_map())[77] = "prog";

  op_profile::Profile default_profile;
  ConvertFlatOpStatsToOpProfile(op_stats, HardwareType::TPU, default_profile);

  op_profile::Profile nullptr_profile;
  ConvertFlatOpStatsToOpProfile(op_stats, HardwareType::TPU, nullptr_profile,
                                100, OpProfileGrouping::kByProgram, nullptr);

  tsl::protobuf::util::JsonPrintOptions opts;
  opts.always_print_fields_with_no_presence = true;
  std::string default_json;
  std::string nullptr_json;
  ASSERT_OK(tsl::protobuf::util::MessageToJsonString(default_profile,
                                                     &default_json, opts));
  ASSERT_OK(tsl::protobuf::util::MessageToJsonString(nullptr_profile,
                                                     &nullptr_json, opts));
  EXPECT_EQ(default_json, nullptr_json);
  EXPECT_THAT(default_json, Not(HasSubstr("programServingMetadata")));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
