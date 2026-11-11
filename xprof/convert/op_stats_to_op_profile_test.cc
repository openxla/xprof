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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tsl/platform/protobuf.h"
#include "xprof/convert/op_profile_builder.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/source_info.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::op_profile::Node;
using ::testing::HasSubstr;
using ::testing::Not;

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

TEST(OpStatsToOpProfileTest, EmptyOpStatsHasRootMetricsByCategory) {
  OpStats op_stats;
  op_profile::Profile profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
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

TEST(OpStatsToOpProfileTest, EmptyOpStatsHasRootMetricsByProgram) {
  OpStats op_stats;
  op_profile::Profile profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
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

TEST(OpStatsToOpProfileTest, EmptyOpStatsHasRootMetricsByProvenance) {
  OpStats op_stats;
  op_profile::Profile profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
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

TEST(OpStatsToOpProfileTest,
     AttachesServingMetadataInByProgramAndByProvenanceTreesOnly) {
  OpStats op_stats;
  OpMetricsDb& db = *op_stats.mutable_device_op_metrics_db();
  db.set_total_time_ps(2000);
  db.set_total_op_time_ps(1500);

  PerfEnv* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(10.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);

  OpMetrics op1 = CreateOpMetrics("op1", 1000, "convolution");
  op1.set_hlo_module_id(101);
  op1.set_provenance("model/layer1");
  *db.add_metrics_db() = op1;

  OpMetrics op2 = CreateOpMetrics("op2", 500, "fusion");
  op2.set_hlo_module_id(202);
  op2.set_provenance("model/layer2");
  *db.add_metrics_db() = op2;

  OpMetrics idle_op = CreateOpMetrics("IDLE", 500, "IDLE");
  *db.add_metrics_db() = idle_op;

  (*op_stats.mutable_program_id_to_name_map())[101] = "jit_fn";
  (*op_stats.mutable_program_id_to_name_map())[202] = "other_fn";

  ProgramServingMetadataMap serving_metadata;
  serving_metadata[101].set_model_id("/models/m1:1");
  serving_metadata[101].set_batch_size(8);
  serving_metadata[101].set_num_batches(12);

  // 1. kByProgram populates both by_program and by_program_exclude_idle.
  op_profile::Profile by_program_profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, by_program_profile,
                            100, OpProfileGrouping::kByProgram,
                            &serving_metadata);
  for (const Node* root : {&by_program_profile.by_program(),
                           &by_program_profile.by_program_exclude_idle()}) {
    bool found_101 = false;
    for (const Node& child : root->children()) {
      if (absl::StartsWith(child.name(), "jit_fn(")) {
        found_101 = true;
        ASSERT_TRUE(child.has_program_serving_metadata());
        EXPECT_EQ(child.program_serving_metadata().model_id(), "/models/m1:1");
        EXPECT_EQ(child.program_serving_metadata().batch_size(), 8);
        EXPECT_EQ(child.program_serving_metadata().num_batches(), 12);
        // Category and op children under the program node must not have
        // serving_metadata.
        for (const Node& cat : child.children()) {
          EXPECT_FALSE(cat.has_program_serving_metadata());
          for (const Node& op : cat.children()) {
            EXPECT_FALSE(op.has_program_serving_metadata());
          }
        }
      } else {
        // Program 202 (not in map) and IDLE node must not have
        // serving_metadata.
        EXPECT_FALSE(child.has_program_serving_metadata());
      }
    }
    EXPECT_TRUE(found_101);
  }

  // 2. kByProvenance populates both by_provenance and
  // by_provenance_exclude_idle.
  op_profile::Profile by_provenance_profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, by_provenance_profile,
                            100, OpProfileGrouping::kByProvenance,
                            &serving_metadata);
  for (const Node* root :
       {&by_provenance_profile.by_provenance(),
        &by_provenance_profile.by_provenance_exclude_idle()}) {
    bool found_101 = false;
    for (const Node& child : root->children()) {
      if (absl::StartsWith(child.name(), "jit_fn(")) {
        found_101 = true;
        EXPECT_TRUE(child.has_program_serving_metadata());
      } else {
        EXPECT_FALSE(child.has_program_serving_metadata());
      }
    }
    EXPECT_TRUE(found_101);
  }

  // 3. kByCategory never sets serving_metadata on any node.
  op_profile::Profile by_category_profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, by_category_profile,
                            100, OpProfileGrouping::kByCategory,
                            &serving_metadata);
  for (const Node* root : {&by_category_profile.by_category(),
                           &by_category_profile.by_category_exclude_idle()}) {
    EXPECT_FALSE(root->has_program_serving_metadata());
    for (const Node& cat : root->children()) {
      EXPECT_FALSE(cat.has_program_serving_metadata());
    }
  }
}

TEST(OpStatsToOpProfileTest,
     LabelsFallbackMainProgramWhenProgramNameMapIsEmpty) {
  OpStats op_stats;
  OpMetricsDb& db = *op_stats.mutable_device_op_metrics_db();
  db.set_total_time_ps(1000);
  db.set_total_op_time_ps(1000);
  PerfEnv* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(10.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);

  OpMetrics op = CreateOpMetrics("op1", 1000, "convolution");
  op.set_hlo_module_id(42);
  *db.add_metrics_db() = op;

  ProgramServingMetadataMap serving_metadata;
  serving_metadata[42].set_model_id("/models/fallback:1");
  serving_metadata[42].set_batch_size(16);
  serving_metadata[42].set_num_batches(5);

  op_profile::Profile profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
                            OpProfileGrouping::kByProgram, &serving_metadata);
  ASSERT_EQ(profile.by_program().children_size(), 1);
  const Node& prog_node = profile.by_program().children(0);
  EXPECT_EQ(prog_node.name(), "main");
  ASSERT_TRUE(prog_node.has_program_serving_metadata());
  EXPECT_EQ(prog_node.program_serving_metadata().model_id(),
            "/models/fallback:1");
  EXPECT_EQ(prog_node.program_serving_metadata().batch_size(), 16);
  EXPECT_EQ(prog_node.program_serving_metadata().num_batches(), 5);
}

TEST(OpStatsToOpProfileTest,
     NullMetadataMapProducesIdenticalJsonAndNoServingMetadataField) {
  OpStats op_stats;
  OpMetricsDb& db = *op_stats.mutable_device_op_metrics_db();
  db.set_total_time_ps(1000);
  db.set_total_op_time_ps(800);
  PerfEnv* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(10.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);

  OpMetrics op = CreateOpMetrics("op1", 800, "convolution");
  op.set_hlo_module_id(77);
  *db.add_metrics_db() = op;
  (*op_stats.mutable_program_id_to_name_map())[77] = "prog";

  op_profile::Profile default_profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, default_profile, 100,
                            OpProfileGrouping::kByProgram);

  op_profile::Profile nullptr_profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, nullptr_profile, 100,
                            OpProfileGrouping::kByProgram, nullptr);

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

TEST(OpStatsToOpProfileTest,
     AcceptsModelOnlyZeroBatchSizeAndIgnoresInvalidServingMetadata) {
  OpStats op_stats;
  OpMetricsDb& db = *op_stats.mutable_device_op_metrics_db();
  db.set_total_time_ps(1500);
  db.set_total_op_time_ps(1500);
  PerfEnv* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(10.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);

  OpMetrics empty_model_op = CreateOpMetrics("op1", 500, "convolution");
  empty_model_op.set_hlo_module_id(10);
  *db.add_metrics_db() = empty_model_op;

  OpMetrics zero_batch_op = CreateOpMetrics("op2", 400, "convolution");
  zero_batch_op.set_hlo_module_id(20);
  *db.add_metrics_db() = zero_batch_op;

  OpMetrics neg_batch_op = CreateOpMetrics("op3", 300, "convolution");
  neg_batch_op.set_hlo_module_id(30);
  *db.add_metrics_db() = neg_batch_op;

  OpMetrics zero_num_batches_op = CreateOpMetrics("op4", 200, "convolution");
  zero_num_batches_op.set_hlo_module_id(40);
  *db.add_metrics_db() = zero_num_batches_op;

  OpMetrics neg_num_batches_op = CreateOpMetrics("op5", 100, "convolution");
  neg_num_batches_op.set_hlo_module_id(50);
  *db.add_metrics_db() = neg_num_batches_op;

  ProgramServingMetadataMap serving_metadata;
  serving_metadata[10].set_batch_size(8);
  serving_metadata[10].set_num_batches(2);
  serving_metadata[20].set_model_id("/models/m1:1");
  serving_metadata[20].set_batch_size(0);
  serving_metadata[20].set_num_batches(2);
  serving_metadata[30].set_model_id("/models/m1:1");
  serving_metadata[30].set_batch_size(-1);
  serving_metadata[30].set_num_batches(2);
  serving_metadata[40].set_model_id("/models/m1:1");
  serving_metadata[40].set_batch_size(8);
  serving_metadata[40].set_num_batches(0);
  serving_metadata[50].set_model_id("/models/m1:1");
  serving_metadata[50].set_batch_size(8);
  serving_metadata[50].set_num_batches(-1);

  op_profile::Profile profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, profile, 100,
                            OpProfileGrouping::kByProgram, &serving_metadata);

  ASSERT_EQ(profile.by_program().children_size(), 5);
  EXPECT_FALSE(
      profile.by_program().children(0).has_program_serving_metadata());
  ASSERT_TRUE(
      profile.by_program().children(1).has_program_serving_metadata());
  EXPECT_EQ(
      profile.by_program().children(1).program_serving_metadata().model_id(),
      "/models/m1:1");
  EXPECT_EQ(
      profile.by_program().children(1).program_serving_metadata().batch_size(),
      0);
  EXPECT_EQ(
      profile.by_program().children(1).program_serving_metadata().num_batches(),
      2);
  for (int i = 2; i < 5; ++i) {
    EXPECT_FALSE(
        profile.by_program().children(i).has_program_serving_metadata());
  }
}

// The frontend derives the number of left-out children from `num_children`, so
// it must hold the child count from before pruning.
TEST(OpStatsToOpProfileTest, NumChildrenIsCountBeforePruning) {
  OpStats op_stats;
  OpMetricsDb& db = *op_stats.mutable_device_op_metrics_db();
  db.set_total_time_ps(2000);
  db.set_total_op_time_ps(1500);

  PerfEnv& perf_env = *op_stats.mutable_perf_env();
  perf_env.set_peak_tera_flops_per_second(10.0);
  perf_env.add_peak_bws_giga_bytes_per_second(100.0);
  perf_env.add_peak_bws_giga_bytes_per_second(100.0);
  perf_env.add_peak_bws_giga_bytes_per_second(100.0);

  // Five ops in one category; op_profile_limit keeps only the top two.
  for (int i = 1; i <= 5; ++i) {
    *db.add_metrics_db() =
        CreateOpMetrics("conv_op_" + std::to_string(i), 100 * i, "convolution");
  }

  op_profile::Profile by_category_profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, by_category_profile,
                            /*op_profile_limit=*/2,
                            OpProfileGrouping::kByCategory);
  ASSERT_EQ(by_category_profile.by_category().children_size(), 1);
  const Node& category = by_category_profile.by_category().children(0);
  ASSERT_EQ(category.children_size(), 2);
  EXPECT_EQ(category.num_children(), 5);
  EXPECT_EQ(category.children(0).name(), "conv_op_5");
  EXPECT_EQ(category.children(1).name(), "conv_op_4");

  op_profile::Profile by_program_profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, by_program_profile,
                            /*op_profile_limit=*/2,
                            OpProfileGrouping::kByProgram);
  ASSERT_EQ(by_program_profile.by_program().children_size(), 1);
  const Node& program = by_program_profile.by_program().children(0);
  ASSERT_EQ(program.children_size(), 1);
  const Node& program_category = program.children(0);
  ASSERT_EQ(program_category.children_size(), 2);
  EXPECT_EQ(program_category.num_children(), 5);
  EXPECT_EQ(program_category.children(0).name(), "conv_op_5");
  EXPECT_EQ(program_category.children(1).name(), "conv_op_4");
}

TEST(InternStackFramesTest, DeduplicatesStackFramesAcrossTrees) {
  op_profile::Profile profile;
  auto add_op = [](Node* parent, const std::string& name,
                   const std::string& stack_frame) {
    Node* node = parent->add_children();
    node->set_name(name);
    node->mutable_xla()->mutable_source_info()->set_file_name("f.py");
    node->mutable_xla()->mutable_source_info()->set_stack_frame(stack_frame);
    return node;
  };
  Node* by_program = profile.mutable_by_program();
  Node* program = by_program->add_children();
  add_op(program, "op1", "a.py:1\nb.py:2");
  add_op(program, "op2", "c.py:3");
  add_op(program, "op3", "");
  add_op(add_op(program, "op4", "a.py:1\nb.py:2"), "fused", "c.py:3");
  *profile.mutable_by_program_exclude_idle() = *by_program;

  InternStackFrames(&profile);

  ASSERT_EQ(profile.stack_frame_table_size(), 2);
  EXPECT_EQ(profile.stack_frame_table(0), "a.py:1\nb.py:2");
  EXPECT_EQ(profile.stack_frame_table(1), "c.py:3");
  EXPECT_FALSE(profile.has_by_category());
  EXPECT_FALSE(profile.has_by_provenance());
  for (const Node* root :
       {&profile.by_program(), &profile.by_program_exclude_idle()}) {
    const Node& prog = root->children(0);
    EXPECT_EQ(prog.children(0).xla().stack_frame_index(), 0);
    EXPECT_EQ(prog.children(1).xla().stack_frame_index(), 1);
    EXPECT_FALSE(prog.children(2).xla().has_stack_frame_index());
    EXPECT_FALSE(prog.children(2).xla().source_info().has_stack_frame());
    EXPECT_EQ(prog.children(3).xla().stack_frame_index(), 0);
    EXPECT_EQ(prog.children(3).children(0).xla().stack_frame_index(), 1);
    for (const Node& op : prog.children()) {
      EXPECT_TRUE(op.xla().source_info().stack_frame().empty());
      EXPECT_EQ(op.xla().source_info().file_name(), "f.py");
    }
  }
}

TEST(ReplaceExcludeIdleTreesWithPatchesTest, PatchesRebuildDerivedTree) {
  op_profile::Profile profile;
  Node* full = profile.mutable_by_program();
  full->set_name("by_program");
  full->set_num_children(3);
  Node* idle = full->add_children();
  idle->set_name("IDLE");
  Node* prog = full->add_children();
  prog->set_name("prog");
  prog->mutable_metrics()->set_flops(0.25);
  Node* op = prog->add_children();
  op->set_name("op");
  op->mutable_metrics()->set_raw_time(10);
  op->mutable_xla()->set_expression("e");
  Node* other = full->add_children();
  other->set_name("other");
  other->mutable_metrics()->set_flops(0.5);

  Node* excl = profile.mutable_by_program_exclude_idle();
  *excl = *full;
  excl->mutable_children()->DeleteSubrange(0, 1);  // Drop IDLE.
  excl->set_num_children(2);
  excl->mutable_children(0)->mutable_metrics()->set_flops(1.0);  // prog
  excl->mutable_children(1)->clear_metrics();  // other: presence change.

  ReplaceExcludeIdleTreesWithPatches(&profile);

  EXPECT_FALSE(profile.has_by_program_exclude_idle());
  const auto& patches = profile.exclude_idle_patch();
  ASSERT_EQ(patches.size(), 3);
  // Root: removed IDLE and changed num_children.
  EXPECT_TRUE(patches[0].path().empty());
  EXPECT_EQ(patches[0].num_children(), 2);
  ASSERT_EQ(patches[0].removed_children_size(), 1);
  EXPECT_EQ(patches[0].removed_children(0), 0);
  // prog: metrics only; "op" is unchanged and has no patch.
  ASSERT_EQ(patches[1].path_size(), 1);
  EXPECT_EQ(patches[1].path(0), 1);
  EXPECT_EQ(patches[1].metrics().flops(), 1.0);
  EXPECT_FALSE(patches[1].has_xla());
  EXPECT_FALSE(patches[1].has_name());
  // other: metrics presence changed, so the subtree is replaced.
  ASSERT_EQ(patches[2].path_size(), 1);
  EXPECT_EQ(patches[2].path(0), 2);
  EXPECT_EQ(patches[2].replacement().name(), "other");
  EXPECT_FALSE(patches[2].replacement().has_metrics());
}

TEST(ReplaceExcludeIdleTreesWithPatchesTest,
     EmitsIdentityRootPatchWhenBaseAndDerivedAreIdentical) {
  op_profile::Profile profile;
  Node* full = profile.mutable_by_program();
  full->set_name("by_program");
  full->set_num_children(1);
  Node* prog = full->add_children();
  prog->set_name("prog");
  prog->mutable_metrics()->set_flops(0.75);
  *profile.mutable_by_program_exclude_idle() = *full;

  ReplaceExcludeIdleTreesWithPatches(&profile);

  EXPECT_FALSE(profile.has_by_program_exclude_idle());
  ASSERT_EQ(profile.exclude_idle_patch_size(), 1);
  const op_profile::NodePatch& root_patch = profile.exclude_idle_patch(0);
  EXPECT_TRUE(root_patch.path().empty());
  EXPECT_FALSE(root_patch.has_replacement());
  EXPECT_FALSE(root_patch.has_name());
  EXPECT_FALSE(root_patch.has_metrics());
  EXPECT_FALSE(root_patch.has_xla());
  EXPECT_FALSE(root_patch.has_num_children());
  EXPECT_TRUE(root_patch.removed_children().empty());
}

TEST(ReplaceExcludeIdleTreesWithPatchesTest,
     ReplacesNodeWhenProgramServingMetadataDiffers) {
  op_profile::Profile profile;
  Node* full = profile.mutable_by_program();
  full->set_name("by_program");
  full->set_num_children(2);

  // Simulate two unnamed programs both falling back to node name "main" with
  // distinct serving metadata; derived drops the first "main" program.
  Node* main1 = full->add_children();
  main1->set_name("main");
  main1->mutable_metrics()->set_raw_time(100);
  main1->mutable_program_serving_metadata()->set_model_id("/models/m1:1");
  main1->mutable_program_serving_metadata()->set_batch_size(4);
  main1->mutable_program_serving_metadata()->set_num_batches(2);

  Node* main2 = full->add_children();
  main2->set_name("main");
  main2->mutable_metrics()->set_raw_time(50);
  main2->mutable_program_serving_metadata()->set_model_id("/models/m2:1");
  main2->mutable_program_serving_metadata()->set_batch_size(16);
  main2->mutable_program_serving_metadata()->set_num_batches(5);

  Node* excl = profile.mutable_by_program_exclude_idle();
  excl->set_name("by_program");
  excl->set_num_children(1);
  *excl->add_children() = *main2;

  ReplaceExcludeIdleTreesWithPatches(&profile);

  EXPECT_FALSE(profile.has_by_program_exclude_idle());
  ASSERT_EQ(profile.exclude_idle_patch_size(), 2);
  // Root removes base child index 1 after greedy alignment matches base[0] with
  // derived[0].
  EXPECT_TRUE(profile.exclude_idle_patch(0).path().empty());
  ASSERT_EQ(profile.exclude_idle_patch(0).removed_children_size(), 1);
  EXPECT_EQ(profile.exclude_idle_patch(0).removed_children(0), 1);
  // Base child 0 ("main" with m1) differs in program_serving_metadata from
  // derived child 0 ("main" with m2), so DiffNode must emit a replacement.
  ASSERT_EQ(profile.exclude_idle_patch(1).path_size(), 1);
  EXPECT_EQ(profile.exclude_idle_patch(1).path(0), 0);
  ASSERT_TRUE(profile.exclude_idle_patch(1).has_replacement());
  EXPECT_EQ(profile.exclude_idle_patch(1)
                .replacement()
                .program_serving_metadata()
                .model_id(),
            "/models/m2:1");
  EXPECT_EQ(profile.exclude_idle_patch(1)
                .replacement()
                .program_serving_metadata()
                .batch_size(),
            16);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
