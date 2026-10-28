/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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
#include "xprof/convert/inference_stats_combiner.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xprof/convert/inference_stats_grouping.h"
#include "plugin/xprof/protobuf/inference_stats.pb.h"

namespace tensorflow::profiler {
namespace {

using ::testing::ElementsAre;

void AddModelId(ModelIdDatabase* db, absl::string_view id) {
  int32_t index = db->ids_size();
  db->add_ids(std::string(id));
  (*db->mutable_id_to_index())[std::string(id)] = index;
}

TEST(InferenceStatsCombinerTest,
     RemapsBatchModelIdIndexWhenSourceModelOrdersDiffer) {
  InferenceStats src0;
  AddModelId(src0.mutable_model_id_db(), "/models/model_a:1");
  AddModelId(src0.mutable_model_id_db(), "/models/model_b:1");
  PerHostInferenceStats& host0 = (*src0.mutable_inference_stats_per_host())[0];
  RequestDetail* req0 = host0.add_request_details();
  req0->set_request_id(1);
  req0->set_model_id_index(0);
  BatchDetail* batch0 = host0.add_batch_details();
  batch0->set_batch_id(10);
  batch0->set_model_id_index(0);
  batch0->set_batch_size_after_padding(4);
  batch0->add_program_ids(1001);

  InferenceStats src1;
  // Reversed model order on host 1: index 0 is model_b, index 1 is model_a.
  AddModelId(src1.mutable_model_id_db(), "/models/model_b:1");
  AddModelId(src1.mutable_model_id_db(), "/models/model_a:1");
  PerHostInferenceStats& host1 = (*src1.mutable_inference_stats_per_host())[0];
  RequestDetail* req1 = host1.add_request_details();
  req1->set_request_id(2);
  req1->set_model_id_index(0);
  BatchDetail* batch1 = host1.add_batch_details();
  batch1->set_batch_id(20);
  batch1->set_model_id_index(0);
  batch1->set_batch_size_after_padding(8);
  batch1->add_program_ids(2002);

  InferenceStats combined;
  CombineInferenceStatsResult(0, src0, &combined);
  CombineInferenceStatsResult(1, src1, &combined);

  const int32_t model_a_index =
      combined.model_id_db().id_to_index().at("/models/model_a:1");
  const int32_t model_b_index =
      combined.model_id_db().id_to_index().at("/models/model_b:1");
  ASSERT_NE(model_a_index, model_b_index);

  const PerHostInferenceStats& combined_host0 =
      combined.inference_stats_per_host().at(0);
  ASSERT_EQ(combined_host0.request_details_size(), 1);
  EXPECT_EQ(combined_host0.request_details(0).model_id_index(), model_a_index);
  ASSERT_EQ(combined_host0.batch_details_size(), 1);
  EXPECT_EQ(combined_host0.batch_details(0).model_id_index(), model_a_index);

  const PerHostInferenceStats& combined_host1 =
      combined.inference_stats_per_host().at(1);
  ASSERT_EQ(combined_host1.request_details_size(), 1);
  EXPECT_EQ(combined_host1.request_details(0).model_id_index(), model_b_index);
  ASSERT_EQ(combined_host1.batch_details_size(), 1);
  EXPECT_EQ(combined_host1.batch_details(0).model_id_index(), model_b_index);
}

TEST(InferenceStatsCombinerTest,
     RegroupByModelPlacesBatchInRemappedModelBucket) {
  InferenceStats src0;
  AddModelId(src0.mutable_model_id_db(), "/models/model_a:1");
  AddModelId(src0.mutable_model_id_db(), "/models/model_b:1");
  PerHostInferenceStats& host0 = (*src0.mutable_inference_stats_per_host())[0];
  RequestDetail* req0 = host0.add_request_details();
  req0->set_request_id(1);
  req0->set_start_time_ps(1000);
  req0->set_end_time_ps(2000);
  req0->set_model_id_index(0);
  BatchDetail* batch0 = host0.add_batch_details();
  batch0->set_batch_id(10);
  batch0->set_start_time_ps(1000);
  batch0->set_end_time_ps(2000);
  batch0->set_model_id_index(0);
  batch0->set_batch_size_after_padding(4);
  batch0->add_program_ids(1001);

  InferenceStats src1;
  AddModelId(src1.mutable_model_id_db(), "/models/model_b:1");
  AddModelId(src1.mutable_model_id_db(), "/models/model_a:1");
  PerHostInferenceStats& host1 = (*src1.mutable_inference_stats_per_host())[0];
  RequestDetail* req1 = host1.add_request_details();
  req1->set_request_id(2);
  req1->set_start_time_ps(3000);
  req1->set_end_time_ps(4000);
  req1->set_model_id_index(0);
  BatchDetail* batch1 = host1.add_batch_details();
  batch1->set_batch_id(20);
  batch1->set_start_time_ps(3000);
  batch1->set_end_time_ps(4000);
  batch1->set_model_id_index(0);
  batch1->set_batch_size_after_padding(8);
  batch1->add_program_ids(2002);

  InferenceStats combined;
  CombineInferenceStatsResult(0, src0, &combined);
  CombineInferenceStatsResult(1, src1, &combined);
  RegroupInferenceStatsByModel(&combined);

  const int32_t model_a_index =
      combined.model_id_db().id_to_index().at("/models/model_a:1");
  const int32_t model_b_index =
      combined.model_id_db().id_to_index().at("/models/model_b:1");

  const PerModelInferenceStats& model_a_stats =
      combined.inference_stats_per_model().at(model_a_index);
  ASSERT_EQ(model_a_stats.batch_details_size(), 1);
  EXPECT_EQ(model_a_stats.batch_details(0).batch_id(), 10);
  EXPECT_THAT(model_a_stats.batch_details(0).program_ids(), ElementsAre(1001));

  const PerModelInferenceStats& model_b_stats =
      combined.inference_stats_per_model().at(model_b_index);
  ASSERT_EQ(model_b_stats.batch_details_size(), 1);
  EXPECT_EQ(model_b_stats.batch_details(0).batch_id(), 20);
  EXPECT_THAT(model_b_stats.batch_details(0).program_ids(), ElementsAre(2002));
}

TEST(InferenceStatsCombinerTest,
     AddsUnindexedSourceModelsToDestinationDatabase) {
  InferenceStats src0;
  AddModelId(src0.mutable_model_id_db(), "/models/model_a:1");
  // Add an entry to host 0's ids() without id_to_index() to verify dst indexes
  // existing ids() entries without duplicating them when another host merges.
  src0.mutable_model_id_db()->add_ids("/models/unindexed_host0:1");
  PerHostInferenceStats& host0 = (*src0.mutable_inference_stats_per_host())[0];
  RequestDetail* req0 = host0.add_request_details();
  req0->set_request_id(1);
  req0->set_start_time_ps(1000);
  req0->set_end_time_ps(2000);
  req0->set_model_id_index(1);

  InferenceStats src1;
  AddModelId(src1.mutable_model_id_db(), "/models/model_b:1");
  src1.mutable_model_id_db()->add_ids("/models/unindexed_host0:1");
  src1.mutable_model_id_db()->add_ids("/models/unindexed_host1:1");

  PerHostInferenceStats& host1 = (*src1.mutable_inference_stats_per_host())[0];
  RequestDetail* req1_shared = host1.add_request_details();
  req1_shared->set_request_id(10);
  req1_shared->set_start_time_ps(3000);
  req1_shared->set_end_time_ps(4000);
  req1_shared->set_model_id_index(1);  // "/models/unindexed_host0:1".

  RequestDetail* req1_new = host1.add_request_details();
  req1_new->set_request_id(11);
  req1_new->set_start_time_ps(5000);
  req1_new->set_end_time_ps(6000);
  req1_new->set_model_id_index(2);  // "/models/unindexed_host1:1".

  BatchDetail* batch1 = host1.add_batch_details();
  batch1->set_batch_id(20);
  batch1->set_start_time_ps(5000);
  batch1->set_end_time_ps(6000);
  batch1->set_model_id_index(2);  // "/models/unindexed_host1:1".
  batch1->add_program_ids(3003);

  InferenceStats combined;
  CombineInferenceStatsResult(0, src0, &combined);
  CombineInferenceStatsResult(1, src1, &combined);

  EXPECT_THAT(
      combined.model_id_db().ids(),
      ElementsAre("/models/model_a:1", "/models/unindexed_host0:1",
                  "/models/model_b:1", "/models/unindexed_host1:1"));
  for (int32_t i = 0; i < combined.model_id_db().ids_size(); ++i) {
    EXPECT_EQ(combined.model_id_db().id_to_index().at(
                  combined.model_id_db().ids(i)),
              i);
  }

  const int32_t host0_unindexed_idx =
      combined.model_id_db().id_to_index().at("/models/unindexed_host0:1");
  const int32_t host1_unindexed_idx =
      combined.model_id_db().id_to_index().at("/models/unindexed_host1:1");

  const PerHostInferenceStats& combined_host1 =
      combined.inference_stats_per_host().at(1);
  ASSERT_EQ(combined_host1.request_details_size(), 2);
  EXPECT_EQ(combined_host1.request_details(0).model_id_index(),
            host0_unindexed_idx);
  EXPECT_EQ(combined_host1.request_details(1).model_id_index(),
            host1_unindexed_idx);
  ASSERT_EQ(combined_host1.batch_details_size(), 1);
  EXPECT_EQ(combined_host1.batch_details(0).model_id_index(),
            host1_unindexed_idx);

  RegroupInferenceStatsByModel(&combined);
  ASSERT_TRUE(
      combined.inference_stats_per_model().contains(host0_unindexed_idx));
  EXPECT_EQ(combined.inference_stats_per_model()
                .at(host0_unindexed_idx)
                .request_details_size(),
            2);
  ASSERT_TRUE(
      combined.inference_stats_per_model().contains(host1_unindexed_idx));
  EXPECT_EQ(combined.inference_stats_per_model()
                .at(host1_unindexed_idx)
                .batch_details_size(),
            1);
}

TEST(InferenceStatsCombinerTest,
     RequestWithUnknownModelStillUpdatesTensorPattern) {
  InferenceStats src0;
  AddModelId(src0.mutable_model_id_db(), "/models/model_a:1");
  src0.mutable_tensor_pattern_db()->add_tensor_pattern("pattern_0");
  (*src0.mutable_inference_stats_per_host())[0] = PerHostInferenceStats();

  InferenceStats src1;
  AddModelId(src1.mutable_model_id_db(), "/models/model_b:1");
  // Empty string represents an unknown model and must not be added to dst.
  src1.mutable_model_id_db()->add_ids("");
  src1.mutable_tensor_pattern_db()->add_tensor_pattern("pattern_1");

  PerHostInferenceStats& host1 = (*src1.mutable_inference_stats_per_host())[0];
  RequestDetail* req1 = host1.add_request_details();
  req1->set_request_id(10);
  req1->set_model_id_index(1);  // Points to "" (unknown model).
  req1->add_tensor_event_details()->set_tensor_pattern_index(0);

  BatchDetail* batch1 = host1.add_batch_details();
  batch1->set_batch_id(20);
  batch1->set_model_id_index(0);  // Points to "/models/model_b:1".
  batch1->mutable_tensor_event_detail()->set_tensor_pattern_index(0);

  InferenceStats combined;
  CombineInferenceStatsResult(0, src0, &combined);
  CombineInferenceStatsResult(1, src1, &combined);

  EXPECT_THAT(combined.model_id_db().ids(),
              ElementsAre("/models/model_a:1", "/models/model_b:1"));
  EXPECT_FALSE(combined.model_id_db().id_to_index().contains(""));

  const PerHostInferenceStats& combined_host1 =
      combined.inference_stats_per_host().at(1);
  ASSERT_EQ(combined_host1.request_details_size(), 1);
  // The empty model ID maps to -1 (no model) instead of an unrelated model.
  EXPECT_EQ(combined_host1.request_details(0).model_id_index(), -1);
  ASSERT_EQ(combined_host1.request_details(0).tensor_event_details_size(), 1);
  // "pattern_1" is appended at index 1 in combined.tensor_pattern_db().
  EXPECT_EQ(combined_host1.request_details(0)
                .tensor_event_details(0)
                .tensor_pattern_index(),
            1);

  ASSERT_EQ(combined_host1.batch_details_size(), 1);
  EXPECT_EQ(combined_host1.batch_details(0).model_id_index(),
            combined.model_id_db().id_to_index().at("/models/model_b:1"));
  EXPECT_EQ(combined_host1.batch_details(0)
                .tensor_event_detail()
                .tensor_pattern_index(),
            1);
}

TEST(InferenceStatsCombinerTest,
     DuplicateHostIdDoesNotCorruptExistingHostEntry) {
  InferenceStats src0;
  AddModelId(src0.mutable_model_id_db(), "/models/model_a:1");
  AddModelId(src0.mutable_model_id_db(), "/models/model_b:1");
  src0.mutable_tensor_pattern_db()->add_tensor_pattern("pattern_a");
  src0.mutable_tensor_pattern_db()->add_tensor_pattern("pattern_b");
  PerHostInferenceStats& host0 = (*src0.mutable_inference_stats_per_host())[0];
  RequestDetail* req0 = host0.add_request_details();
  req0->set_request_id(1);
  req0->set_model_id_index(0);
  req0->add_tensor_event_details()->set_tensor_pattern_index(0);
  BatchDetail* batch0 = host0.add_batch_details();
  batch0->set_batch_id(10);
  batch0->set_model_id_index(0);
  batch0->mutable_tensor_event_detail()->set_tensor_pattern_index(0);

  InferenceStats duplicate_host_src;
  // Index 0 here is model_b and pattern_b, which map to index 1 in combined.
  AddModelId(duplicate_host_src.mutable_model_id_db(), "/models/model_b:1");
  AddModelId(duplicate_host_src.mutable_model_id_db(), "/models/model_c:1");
  duplicate_host_src.mutable_tensor_pattern_db()->add_tensor_pattern(
      "pattern_b");
  duplicate_host_src.mutable_tensor_pattern_db()->add_tensor_pattern(
      "pattern_c");
  PerHostInferenceStats& dup_host =
      (*duplicate_host_src.mutable_inference_stats_per_host())[0];
  dup_host.add_request_details()->set_model_id_index(0);
  dup_host.add_batch_details()->set_model_id_index(0);

  InferenceStats combined;
  CombineInferenceStatsResult(0, src0, &combined);
  // Passing src_host_id = 0 again must not re-remap host 0's existing entries.
  CombineInferenceStatsResult(0, duplicate_host_src, &combined);

  // The rejected host must not add model_c or pattern_c to the databases.
  EXPECT_EQ(combined.model_id_db().ids_size(), 2);
  EXPECT_FALSE(
      combined.model_id_db().id_to_index().contains("/models/model_c:1"));
  EXPECT_EQ(combined.tensor_pattern_db().tensor_pattern_size(), 2);

  const PerHostInferenceStats& combined_host0 =
      combined.inference_stats_per_host().at(0);
  ASSERT_EQ(combined_host0.request_details_size(), 1);
  EXPECT_EQ(combined_host0.request_details(0).model_id_index(), 0);
  EXPECT_EQ(combined_host0.request_details(0)
                .tensor_event_details(0)
                .tensor_pattern_index(),
            0);
  ASSERT_EQ(combined_host0.batch_details_size(), 1);
  EXPECT_EQ(combined_host0.batch_details(0).model_id_index(), 0);
  EXPECT_EQ(combined_host0.batch_details(0)
                .tensor_event_detail()
                .tensor_pattern_index(),
            0);
}

TEST(InferenceStatsCombinerTest,
     OutOfRangeModelIndicesMapToNoModelAndDoNotCrash) {
  InferenceStats src0;
  AddModelId(src0.mutable_model_id_db(), "/models/model_a:1");
  src0.mutable_tensor_pattern_db()->add_tensor_pattern("pattern_0");
  (*src0.mutable_inference_stats_per_host())[0] = PerHostInferenceStats();

  InferenceStats src1;
  AddModelId(src1.mutable_model_id_db(), "/models/model_b:1");
  src1.mutable_tensor_pattern_db()->add_tensor_pattern("pattern_1");
  PerHostInferenceStats& host1 = (*src1.mutable_inference_stats_per_host())[0];
  RequestDetail* req_oob = host1.add_request_details();
  req_oob->set_model_id_index(99);
  req_oob->add_tensor_event_details()->set_tensor_pattern_index(99);
  RequestDetail* req_neg = host1.add_request_details();
  req_neg->set_model_id_index(-5);
  req_neg->add_tensor_event_details()->set_tensor_pattern_index(-2);

  BatchDetail* batch_oob = host1.add_batch_details();
  batch_oob->set_model_id_index(99);
  batch_oob->mutable_tensor_event_detail()->set_tensor_pattern_index(99);
  BatchDetail* batch_neg = host1.add_batch_details();
  batch_neg->set_model_id_index(-5);

  InferenceStats combined;
  CombineInferenceStatsResult(0, src0, &combined);
  CombineInferenceStatsResult(1, src1, &combined);

  const PerHostInferenceStats& combined_host1 =
      combined.inference_stats_per_host().at(1);
  ASSERT_EQ(combined_host1.request_details_size(), 2);
  // Invalid model indices map to -1 (no model); invalid tensor pattern indices
  // are left unchanged.
  EXPECT_EQ(combined_host1.request_details(0).model_id_index(), -1);
  EXPECT_EQ(combined_host1.request_details(0)
                .tensor_event_details(0)
                .tensor_pattern_index(),
            99);
  EXPECT_EQ(combined_host1.request_details(1).model_id_index(), -1);
  EXPECT_EQ(combined_host1.request_details(1)
                .tensor_event_details(0)
                .tensor_pattern_index(),
            -2);

  ASSERT_EQ(combined_host1.batch_details_size(), 2);
  EXPECT_EQ(combined_host1.batch_details(0).model_id_index(), -1);
  EXPECT_EQ(combined_host1.batch_details(0)
                .tensor_event_detail()
                .tensor_pattern_index(),
            99);
  EXPECT_EQ(combined_host1.batch_details(1).model_id_index(), -1);
  EXPECT_FALSE(combined_host1.batch_details(1).has_tensor_event_detail());

  // Regrouping skips details with model_id_index -1 instead of indexing out of
  // bounds.
  RegroupInferenceStatsByModel(&combined);
  EXPECT_EQ(combined.inference_stats_per_model_size(), 2);
}

TEST(InferenceStatsCombinerTest,
     UnsetModelIdIndexAndIdenticalModelOrderRemainUnchanged) {
  InferenceStats src0;
  AddModelId(src0.mutable_model_id_db(), "/models/model_a:1");
  AddModelId(src0.mutable_model_id_db(), "/models/model_b:1");
  src0.mutable_tensor_pattern_db()->add_tensor_pattern("pattern_host0");
  (*src0.mutable_inference_stats_per_host())[0] = PerHostInferenceStats();

  InferenceStats src1;
  AddModelId(src1.mutable_model_id_db(), "/models/model_a:1");
  AddModelId(src1.mutable_model_id_db(), "/models/model_b:1");
  PerHostInferenceStats& host1 = (*src1.mutable_inference_stats_per_host())[0];
  host1.add_request_details()->set_model_id_index(-1);
  host1.add_request_details()->set_model_id_index(1);
  host1.add_batch_details()->set_model_id_index(-1);
  host1.add_batch_details()->set_model_id_index(1);

  // Also test when model order differs that -1 still stays -1.
  InferenceStats src2;
  AddModelId(src2.mutable_model_id_db(), "/models/model_c:1");
  PerHostInferenceStats& host2 = (*src2.mutable_inference_stats_per_host())[0];
  host2.add_request_details()->set_model_id_index(-1);
  host2.add_batch_details()->set_model_id_index(-1);

  InferenceStats combined;
  CombineInferenceStatsResult(0, src0, &combined);
  CombineInferenceStatsResult(1, src1, &combined);
  CombineInferenceStatsResult(2, src2, &combined);

  const PerHostInferenceStats& combined_host1 =
      combined.inference_stats_per_host().at(1);
  EXPECT_EQ(combined_host1.request_details(0).model_id_index(), -1);
  EXPECT_EQ(combined_host1.request_details(1).model_id_index(), 1);
  EXPECT_EQ(combined_host1.batch_details(0).model_id_index(), -1);
  EXPECT_EQ(combined_host1.batch_details(1).model_id_index(), 1);

  const PerHostInferenceStats& combined_host2 =
      combined.inference_stats_per_host().at(2);
  EXPECT_EQ(combined_host2.request_details(0).model_id_index(), -1);
  EXPECT_EQ(combined_host2.batch_details(0).model_id_index(), -1);

  // Also test when model order is identical (need_update_model_id == false)
  // while tensor patterns differ (need_update_tensor_pattern == true).
  InferenceStats src3;
  AddModelId(src3.mutable_model_id_db(), "/models/model_a:1");
  AddModelId(src3.mutable_model_id_db(), "/models/model_b:1");
  AddModelId(src3.mutable_model_id_db(), "/models/model_c:1");
  src3.mutable_tensor_pattern_db()->add_tensor_pattern("pattern_host3");
  PerHostInferenceStats& host3 = (*src3.mutable_inference_stats_per_host())[0];
  RequestDetail* req3 = host3.add_request_details();
  req3->set_model_id_index(1);
  req3->add_tensor_event_details()->set_tensor_pattern_index(0);
  BatchDetail* batch3 = host3.add_batch_details();
  batch3->set_model_id_index(2);
  batch3->mutable_tensor_event_detail()->set_tensor_pattern_index(0);

  CombineInferenceStatsResult(3, src3, &combined);
  const PerHostInferenceStats& combined_host3 =
      combined.inference_stats_per_host().at(3);
  EXPECT_EQ(combined_host3.request_details(0).model_id_index(), 1);
  EXPECT_EQ(combined_host3.request_details(0)
                .tensor_event_details(0)
                .tensor_pattern_index(),
            1);
  EXPECT_EQ(combined_host3.batch_details(0).model_id_index(), 2);
  EXPECT_EQ(combined_host3.batch_details(0)
                .tensor_event_detail()
                .tensor_pattern_index(),
            1);
}

}  // namespace
}  // namespace tensorflow::profiler
