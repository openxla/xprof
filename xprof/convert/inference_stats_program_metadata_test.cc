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
#include "xprof/convert/inference_stats_program_metadata.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/device_utils.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_test_utils.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/inference_stats.h"
#include "xprof/convert/inference_stats_grouping.h"
#include "xprof/convert/op_profile_builder.h"
#include "xprof/convert/op_stats_to_op_profile.h"
#include "xprof/convert/xplane_to_op_stats.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/inference_stats.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "xprof/utils/event_span.h"

namespace tensorflow::profiler {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::tsl::profiler::CreateXEvent;
using ::tsl::profiler::CreateXEventMetadata;
using ::tsl::profiler::DeviceType;
using ::tsl::profiler::GetOrCreateHostXPlane;
using ::tsl::profiler::GetOrCreateTpuXPlane;
using ::tsl::profiler::GroupMetadataMap;
using ::tsl::profiler::HostEventType;
using ::tsl::profiler::kXlaModuleLineName;
using ::tsl::profiler::kXlaOpLineName;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlane;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XSpace;

int32_t AddModelId(ModelIdDatabase* db, absl::string_view id) {
  int32_t index = db->ids_size();
  db->add_ids(std::string(id));
  (*db->mutable_id_to_index())[std::string(id)] = index;
  return index;
}

TEST(InferenceStatsProgramMetadataTest, SingleBatchProducesMetadata) {
  InferenceStats stats;
  int32_t model_idx = AddModelId(stats.mutable_model_id_db(), "/models/m1:1");
  BatchDetail* batch = (*stats.mutable_inference_stats_per_model())[model_idx]
                           .add_batch_details();
  batch->set_model_id_index(model_idx);
  batch->set_batch_size_after_padding(8);
  batch->add_program_ids(1001);

  ProgramServingMetadataMap metadata = BuildProgramServingMetadata(stats);
  ASSERT_EQ(metadata.size(), 1);
  ASSERT_TRUE(metadata.contains(1001));
  const op_profile::Node::ProgramServingMetadata& entry = metadata.at(1001);
  EXPECT_EQ(entry.model_id(), "/models/m1:1");
  EXPECT_EQ(entry.batch_size(), 8);
  EXPECT_EQ(entry.num_batches(), 1);
}

TEST(InferenceStatsProgramMetadataTest,
     PrefersPerModelOverPerHostAndDoesNotDoubleCount) {
  InferenceStats stats;
  int32_t model_idx = AddModelId(stats.mutable_model_id_db(), "/models/m1:1");

  BatchDetail* host_batch =
      (*stats.mutable_inference_stats_per_host())[0].add_batch_details();
  host_batch->set_model_id_index(model_idx);
  host_batch->set_batch_size_after_padding(4);
  host_batch->add_program_ids(1001);

  // When per_model is empty, per_host is used.
  ProgramServingMetadataMap from_host = BuildProgramServingMetadata(stats);
  ASSERT_TRUE(from_host.contains(1001));
  EXPECT_EQ(from_host.at(1001).batch_size(), 4);
  EXPECT_EQ(from_host.at(1001).num_batches(), 1);

  // When per_model is populated, per_model is used exclusively and per_host is
  // ignored.
  BatchDetail* model_batch =
      (*stats.mutable_inference_stats_per_model())[model_idx]
          .add_batch_details();
  model_batch->set_model_id_index(model_idx);
  model_batch->set_batch_size_after_padding(16);
  model_batch->add_program_ids(1001);

  ProgramServingMetadataMap from_model = BuildProgramServingMetadata(stats);
  ASSERT_TRUE(from_model.contains(1001));
  EXPECT_EQ(from_model.at(1001).batch_size(), 16);
  EXPECT_EQ(from_model.at(1001).num_batches(), 1);
}

TEST(InferenceStatsProgramMetadataTest,
     NegativeHostIdInPerHostStatsFallsBackToHostKey) {
  InferenceStats stats;
  AddModelId(stats.mutable_model_id_db(), "/models/a:1");
  int32_t idx_b = AddModelId(stats.mutable_model_id_db(), "/models/b:1");
  PerHostInferenceStats& host = (*stats.mutable_inference_stats_per_host())[0];
  BatchDetail* batch = host.add_batch_details();
  // An explicit negative host id must not count as a second host, so the
  // batch model index is still trusted in this single-host session.
  batch->set_host_id(-1);
  batch->set_model_id_index(idx_b);
  batch->set_batch_size_after_padding(8);
  batch->add_program_ids(1001);

  ProgramServingMetadataMap metadata = BuildProgramServingMetadata(stats);
  ASSERT_TRUE(metadata.contains(1001));
  EXPECT_EQ(metadata.at(1001).model_id(), "/models/b:1");
  EXPECT_EQ(metadata.at(1001).batch_size(), 8);
}

TEST(InferenceStatsProgramMetadataTest,
     BatchWithMultipleProgramsLabelsAllAndDeduplicatesWithinBatch) {
  InferenceStats stats;
  int32_t model_idx = AddModelId(stats.mutable_model_id_db(), "/models/m1:1");
  BatchDetail* batch = (*stats.mutable_inference_stats_per_model())[model_idx]
                           .add_batch_details();
  batch->set_model_id_index(model_idx);
  batch->set_batch_size_after_padding(8);
  batch->add_program_ids(1001);
  batch->add_program_ids(1001);  // Duplicate within same batch.
  batch->add_program_ids(2002);

  ProgramServingMetadataMap metadata = BuildProgramServingMetadata(stats);
  ASSERT_EQ(metadata.size(), 2);
  EXPECT_EQ(metadata.at(1001).num_batches(), 1);
  EXPECT_EQ(metadata.at(2002).num_batches(), 1);
}

TEST(InferenceStatsProgramMetadataTest,
     ConflictingPairsSelectsHighestCountAndLogsWarning) {
  InferenceStats stats;
  int32_t model_idx = AddModelId(stats.mutable_model_id_db(), "/models/m1:1");
  PerModelInferenceStats& per_model =
      (*stats.mutable_inference_stats_per_model())[model_idx];

  BatchDetail* bs4 = per_model.add_batch_details();
  bs4->set_model_id_index(model_idx);
  bs4->set_batch_size_after_padding(4);
  bs4->add_program_ids(1001);

  for (int i = 0; i < 3; ++i) {
    BatchDetail* bs8 = per_model.add_batch_details();
    bs8->set_model_id_index(model_idx);
    bs8->set_batch_size_after_padding(8);
    bs8->add_program_ids(1001);
  }

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(
      log, Log(absl::LogSeverity::kWarning, _,
               AllOf(HasSubstr(
                         "HLO program 1001 has multiple (model_id, batch_size) "
                         "pairs"),
                     HasSubstr("keeping (model_id=/models/m1:1, batch_size=8, "
                               "num_batches=3)"),
                     HasSubstr("(model_id=/models/m1:1, batch_size=4, "
                               "num_batches=1)"))))
      .Times(1);
  log.StartCapturingLogs();

  ProgramServingMetadataMap metadata = BuildProgramServingMetadata(stats);
  ASSERT_TRUE(metadata.contains(1001));
  const op_profile::Node::ProgramServingMetadata& entry = metadata.at(1001);
  EXPECT_EQ(entry.model_id(), "/models/m1:1");
  EXPECT_EQ(entry.batch_size(), 8);
  EXPECT_EQ(entry.num_batches(), 3);
}

TEST(InferenceStatsProgramMetadataTest, UnsetOrOutOfRangeModelIndexIsSkipped) {
  InferenceStats stats;
  AddModelId(stats.mutable_model_id_db(), "/models/m1:1");
  PerHostInferenceStats& per_host =
      (*stats.mutable_inference_stats_per_host())[0];

  BatchDetail* unset_model = per_host.add_batch_details();
  unset_model->set_model_id_index(-1);
  unset_model->set_batch_size_after_padding(8);
  unset_model->add_program_ids(1001);

  BatchDetail* oob_model = per_host.add_batch_details();
  oob_model->set_model_id_index(99);
  oob_model->set_batch_size_after_padding(16);
  oob_model->add_program_ids(2002);

  ProgramServingMetadataMap metadata = BuildProgramServingMetadata(stats);
  EXPECT_THAT(metadata, IsEmpty());
}

TEST(InferenceStatsProgramMetadataTest, PlaceholderAllModelIdIsSkipped) {
  InferenceStats stats;
  int32_t all_idx = AddModelId(stats.mutable_model_id_db(), "ALL");
  BatchDetail* batch =
      (*stats.mutable_inference_stats_per_model())[all_idx].add_batch_details();
  batch->set_model_id_index(all_idx);
  batch->set_batch_size_after_padding(32);
  batch->add_program_ids(1001);

  ProgramServingMetadataMap metadata = BuildProgramServingMetadata(stats);
  EXPECT_THAT(metadata, IsEmpty());
}

TEST(InferenceStatsProgramMetadataTest,
     SkipsProgramIdZeroAndEmptyUninformativeBatches) {
  InferenceStats stats;
  PerHostInferenceStats& per_host =
      (*stats.mutable_inference_stats_per_host())[0];

  // Program ID 0 is skipped without warning.
  BatchDetail* zero_prog = per_host.add_batch_details();
  zero_prog->set_batch_size_after_padding(0);
  zero_prog->add_program_ids(0);

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _, _)).Times(0);
  log.StartCapturingLogs();

  ProgramServingMetadataMap metadata = BuildProgramServingMetadata(stats);
  EXPECT_THAT(metadata, IsEmpty());
}

TEST(InferenceStatsProgramMetadataTest,
     NonPositiveBatchSizeOnlyProducesNoMetadataAndLogsOneSummaryWarning) {
  InferenceStats stats;
  int32_t model_idx = AddModelId(stats.mutable_model_id_db(), "/models/m1:1");
  PerModelInferenceStats& per_model =
      (*stats.mutable_inference_stats_per_model())[model_idx];

  for (int32_t bs : {0, -4, 0}) {
    BatchDetail* batch = per_model.add_batch_details();
    batch->set_model_id_index(model_idx);
    batch->set_batch_size_after_padding(bs);
    batch->add_program_ids(1001);
  }

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(
      log,
      Log(absl::LogSeverity::kWarning, _,
          HasSubstr(
              "skipped 3 batches with batch_size_after_padding <= 0 across 1 "
              "HLO programs")))
      .Times(1);
  log.StartCapturingLogs();

  ProgramServingMetadataMap metadata = BuildProgramServingMetadata(stats);
  EXPECT_THAT(metadata, IsEmpty());
}

TEST(InferenceStatsProgramMetadataTest,
     DeterministicTieBreakingByModelIdThenBatchSize) {
  InferenceStats stats;
  int32_t idx_b = AddModelId(stats.mutable_model_id_db(), "/models/b:1");
  int32_t idx_a = AddModelId(stats.mutable_model_id_db(), "/models/a:1");
  int32_t idx_z = AddModelId(stats.mutable_model_id_db(), "/models/z:1");
  PerHostInferenceStats& per_host =
      (*stats.mutable_inference_stats_per_host())[0];

  auto add_batch = [&](uint64_t program_id, int32_t model_idx, int32_t bs) {
    BatchDetail* b = per_host.add_batch_details();
    b->set_model_id_index(model_idx);
    b->set_batch_size_after_padding(bs);
    b->add_program_ids(program_id);
  };
  // Program 1001: "/models/a:1" beats "/models/b:1", and bs 4 beats bs 16.
  add_batch(1001, idx_b, 8);
  add_batch(1001, idx_a, 16);
  add_batch(1001, idx_a, 4);

  // Program 2002: unknown-model batches (-1) are skipped, leaving "/models/z:1"
  // with no conflict warning for program 2002.
  add_batch(2002, -1, 8);
  add_batch(2002, -1, 8);
  add_batch(2002, idx_z, 8);

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(
      log, Log(absl::LogSeverity::kWarning, _, HasSubstr("HLO program 1001")))
      .Times(1);
  EXPECT_CALL(
      log, Log(absl::LogSeverity::kWarning, _, HasSubstr("HLO program 2002")))
      .Times(0);
  log.StartCapturingLogs();

  ProgramServingMetadataMap metadata = BuildProgramServingMetadata(stats);
  ASSERT_TRUE(metadata.contains(1001));
  EXPECT_EQ(metadata.at(1001).model_id(), "/models/a:1");
  EXPECT_EQ(metadata.at(1001).batch_size(), 4);
  EXPECT_EQ(metadata.at(1001).num_batches(), 1);

  ASSERT_TRUE(metadata.contains(2002));
  EXPECT_EQ(metadata.at(2002).model_id(), "/models/z:1");
  EXPECT_EQ(metadata.at(2002).batch_size(), 8);
  EXPECT_EQ(metadata.at(2002).num_batches(), 1);
}

TEST(InferenceStatsProgramMetadataTest,
     SkipsZeroBatchSizeBatchesAndKeepsPositiveBatchSizeWithSummaryWarning) {
  InferenceStats stats;
  int32_t model_idx = AddModelId(stats.mutable_model_id_db(), "/models/m1:1");
  PerModelInferenceStats& per_model =
      (*stats.mutable_inference_stats_per_model())[model_idx];

  // Five batches lack padding info (batch_size_after_padding = 0), while two
  // batches report the true padded batch size (8).
  for (int i = 0; i < 5; ++i) {
    BatchDetail* b = per_model.add_batch_details();
    b->set_model_id_index(model_idx);
    b->set_batch_size_after_padding(0);
    b->add_program_ids(1001);
  }
  for (int i = 0; i < 2; ++i) {
    BatchDetail* b = per_model.add_batch_details();
    b->set_model_id_index(model_idx);
    b->set_batch_size_after_padding(8);
    b->add_program_ids(1001);
  }

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(
      log,
      Log(absl::LogSeverity::kWarning, _,
          HasSubstr(
              "skipped 5 batches with batch_size_after_padding <= 0 across 1 "
              "HLO programs")))
      .Times(1);
  // Zero-batch-size batches are skipped before pair counting, so they do not
  // trigger a conflicting-pair warning.
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                       HasSubstr("has multiple (model_id, batch_size) pairs")))
      .Times(0);
  log.StartCapturingLogs();

  ProgramServingMetadataMap metadata = BuildProgramServingMetadata(stats);
  ASSERT_TRUE(metadata.contains(1001));
  const op_profile::Node::ProgramServingMetadata& entry = metadata.at(1001);
  EXPECT_EQ(entry.model_id(), "/models/m1:1");
  EXPECT_EQ(entry.batch_size(), 8);
  EXPECT_EQ(entry.num_batches(), 2);
}

TEST(InferenceStatsProgramMetadataTest,
     ModelLookupRuleHandlesSingleSourceSingleModelAndMultiSourcePreFixCache) {
  // 1. Multi-source + single model: batch.model_id_index is trusted directly
  // without needing request lookup.
  {
    InferenceStats single_model_stats;
    int32_t m0 =
        AddModelId(single_model_stats.mutable_model_id_db(), "/models/only:1");
    PerModelInferenceStats& pm =
        (*single_model_stats.mutable_inference_stats_per_model())[m0];
    BatchDetail* b0 = pm.add_batch_details();
    b0->set_host_id(0);
    b0->set_model_id_index(m0);
    b0->set_batch_size_after_padding(4);
    b0->add_program_ids(10);
    BatchDetail* b1 = pm.add_batch_details();
    b1->set_host_id(1);
    b1->set_model_id_index(m0);
    b1->set_batch_size_after_padding(8);
    b1->add_program_ids(20);

    ProgramServingMetadataMap metadata =
        BuildProgramServingMetadata(single_model_stats);
    EXPECT_EQ(metadata.at(10).model_id(), "/models/only:1");
    EXPECT_EQ(metadata.at(20).model_id(), "/models/only:1");
  }

  // 2. Multi-source + multi-model with a stale (not remapped) batch
  // model_id_index: resolves the model via (batch.host_id, related request
  // IDs) in request_details, and skips batches whose model cannot be resolved
  // (rather than trusting the stale batch model_id_index).
  {
    InferenceStats pre_fix_cache;
    int32_t idx_a =
        AddModelId(pre_fix_cache.mutable_model_id_db(), "/models/model_a:1");
    int32_t idx_b =
        AddModelId(pre_fix_cache.mutable_model_id_db(), "/models/model_b:1");

    // Host 0 had [A, B], host 1 had [B, A]. In a cache written before the
    // combiner remapped batch model indices, host 1's
    // request was remapped to idx_b (1), while host 1's batch kept stale
    // model_id_index = 0 (and was grouped into model 0's bucket).
    PerModelInferenceStats& pm_a =
        (*pre_fix_cache.mutable_inference_stats_per_model())[idx_a];
    RequestDetail* req_host0 = pm_a.add_request_details();
    req_host0->set_host_id(0);
    req_host0->set_request_id(100);
    req_host0->set_model_id_index(idx_a);

    BatchDetail* batch_host0 = pm_a.add_batch_details();
    batch_host0->set_host_id(0);
    batch_host0->set_batch_id(1000);
    batch_host0->add_related_request_ids(100);
    batch_host0->set_model_id_index(idx_a);
    batch_host0->set_batch_size_after_padding(4);
    batch_host0->add_program_ids(1111);

    // Host 1's batch with stale model_id_index = 0 (idx_a), where the first
    // related request (199) was omitted from request_details and the second
    // related request (200) on host 1 was properly remapped to idx_b.
    BatchDetail* batch_host1_stale = pm_a.add_batch_details();
    batch_host1_stale->set_host_id(1);
    batch_host1_stale->set_batch_id(2000);
    batch_host1_stale->add_related_request_ids(199);  // Omitted request.
    batch_host1_stale->add_related_request_ids(200);
    batch_host1_stale->set_model_id_index(idx_a);  // Stale!
    batch_host1_stale->set_batch_size_after_padding(8);
    batch_host1_stale->add_program_ids(2222);

    // Batch on host 1 whose related request is missing from request_details.
    BatchDetail* batch_missing_req = pm_a.add_batch_details();
    batch_missing_req->set_host_id(1);
    batch_missing_req->set_batch_id(3000);
    batch_missing_req->add_related_request_ids(999);
    batch_missing_req->set_model_id_index(idx_a);
    batch_missing_req->set_batch_size_after_padding(16);
    batch_missing_req->add_program_ids(3333);

    PerModelInferenceStats& pm_b =
        (*pre_fix_cache.mutable_inference_stats_per_model())[idx_b];
    RequestDetail* req_host1 = pm_b.add_request_details();
    req_host1->set_host_id(1);
    req_host1->set_request_id(200);
    req_host1->set_model_id_index(idx_b);

    ProgramServingMetadataMap metadata =
        BuildProgramServingMetadata(pre_fix_cache);
    ASSERT_TRUE(metadata.contains(1111));
    EXPECT_EQ(metadata.at(1111).model_id(), "/models/model_a:1");
    EXPECT_EQ(metadata.at(1111).batch_size(), 4);

    // Program 2222 recovers "/models/model_b:1" via request (host=1, id=200)!
    ASSERT_TRUE(metadata.contains(2222));
    EXPECT_EQ(metadata.at(2222).model_id(), "/models/model_b:1");
    EXPECT_EQ(metadata.at(2222).batch_size(), 8);

    // Program 3333 has no resolved model_id (stale idx_a is not trusted), so it
    // is omitted.
    EXPECT_FALSE(metadata.contains(3333));
  }
}

TEST(InferenceStatsProgramMetadataTest,
     RealXSpaceToOpProfileAndInferenceStatsIntegration) {
  constexpr uint64_t kProgramId = 5621224425449793704ULL;
  constexpr int64_t kRequestGroupId = 101;
  constexpr int64_t kBatchGroupId = 202;

  XSpace space;
  XPlane* tpu_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/275.0,
      /*peak_hbm_bw_gigabytes_per_second=*/1200.0);
  XPlaneBuilder tpu_builder(tpu_plane);

  // XLA Modules line on device.
  XLineBuilder module_line = tpu_builder.GetOrCreateLine(0);
  module_line.SetName(kXlaModuleLineName);
  CreateXEventMetadata(
      &tpu_builder, "jit__apply(5621224425449793704)",
      {{StatType::kProgramId, static_cast<int64_t>(kProgramId)}});
  CreateXEvent(&tpu_builder, &module_line, "jit__apply(5621224425449793704)",
               /*offset_ps=*/1000, /*duration_ps=*/10000,
               {{StatType::kGroupId, kBatchGroupId}});

  // XLA Ops line on device.
  XLineBuilder op_line = tpu_builder.GetOrCreateLine(1);
  op_line.SetName(kXlaOpLineName);
  CreateXEventMetadata(
      &tpu_builder, "fusion.1",
      {{StatType::kHloCategory, "fusion"},
       {StatType::kProgramId, static_cast<int64_t>(kProgramId)},
       {StatType::kSymbolId, int64_t{1}},
       {StatType::kFlops, int64_t{1000}}});
  CreateXEvent(&tpu_builder, &op_line, "fusion.1", /*offset_ps=*/2000,
               /*duration_ps=*/8000, {{StatType::kGroupId, kBatchGroupId}});

  // Host plane with request, batch, and padding events.
  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_builder(host_plane);
  XLineBuilder host_line = host_builder.GetOrCreateLine(0);
  host_line.SetName("BatchThread");
  CreateXEvent(&host_builder, &host_line, HostEventType::kSessionRun,
               /*offset_ps=*/500, /*duration_ps=*/12000,
               {{StatType::kGroupId, kRequestGroupId},
                {StatType::kModelId, "/models/ranking_model:1"}});
  CreateXEvent(&host_builder, &host_line, HostEventType::kProcessBatch,
               /*offset_ps=*/800, /*duration_ps=*/11000,
               {{StatType::kGroupId, kBatchGroupId}});
  CreateXEvent(&host_builder, &host_line, HostEventType::kConcatInputTensors,
               /*offset_ps=*/850, /*duration_ps=*/100,
               {{StatType::kGroupId, kBatchGroupId},
                {StatType::kPaddingAmount, int64_t{0}},
                {StatType::kBatchSizeAfterPadding, int64_t{8}}});

  // 1. Convert XSpace -> OpStats.
  ASSERT_OK_AND_ASSIGN(
      OpStats op_stats,
      ConvertXSpaceToOpStats(space,
                             OpStatsOptions{.generate_op_metrics_db = true}));

  // 2. Convert same XSpace -> InferenceStats and regroup by model.
  GroupMetadataMap group_metadata_map;
  group_metadata_map[kRequestGroupId].children.insert(kBatchGroupId);
  group_metadata_map[kBatchGroupId].parents.insert(kRequestGroupId);
  std::vector<XPlane*> device_traces = {tpu_plane};
  StepEvents step_events;
  InferenceStats inference_stats;
  GenerateInferenceStats(device_traces, step_events, group_metadata_map, space,
                         DeviceType::kTpu, /*host_id=*/0, &inference_stats);
  RegroupInferenceStatsByModel(&inference_stats);

  // 3. Build program serving metadata and convert OpStats ->
  // op_profile::Profile.
  ProgramServingMetadataMap serving_metadata =
      BuildProgramServingMetadata(inference_stats);
  ASSERT_TRUE(serving_metadata.contains(kProgramId));

  // Populate program_id_to_name_map (normally populated from HloProto plane).
  (*op_stats.mutable_program_id_to_name_map())[kProgramId] = "jit__apply";

  op_profile::Profile profile;
  ConvertOpStatsToOpProfile(op_stats, HardwareType::TPU, profile,
                            /*op_profile_limit=*/100,
                            OpProfileGrouping::kByProgram, &serving_metadata);

  ASSERT_TRUE(profile.has_by_program());
  const op_profile::Node* program_node = nullptr;
  for (const op_profile::Node& child : profile.by_program().children()) {
    if (absl::StrContains(child.name(), "5621224425449793704")) {
      program_node = &child;
    } else {
      EXPECT_FALSE(child.has_program_serving_metadata());
    }
  }
  ASSERT_NE(program_node, nullptr);
  ASSERT_TRUE(program_node->has_program_serving_metadata());
  EXPECT_EQ(program_node->program_serving_metadata().model_id(),
            "/models/ranking_model:1");
  EXPECT_EQ(program_node->program_serving_metadata().batch_size(), 8);
  EXPECT_EQ(program_node->program_serving_metadata().num_batches(), 1);

  std::string json;
  tsl::protobuf::util::JsonPrintOptions opts;
  opts.always_print_fields_with_no_presence = true;
  ASSERT_OK(tsl::protobuf::util::MessageToJsonString(profile, &json, opts));
  EXPECT_THAT(json, HasSubstr("\"programServingMetadata\""));
  EXPECT_THAT(json, HasSubstr("\"/models/ranking_model:1\""));
  EXPECT_THAT(json, HasSubstr("\"batchSize\":8"));
}

// Without batching, a session has only RequestDetails. They carry no program
// ids, so the request's model cannot be matched to the program it ran.
TEST(InferenceStatsProgramMetadataTest,
     RequestOnlySessionWithoutBatchingHasNoProgramLink) {
  constexpr uint64_t kProgramId = 5621224425449793704ULL;
  constexpr int64_t kRequestGroupId = 101;

  XSpace space;
  XPlane* tpu_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/275.0,
      /*peak_hbm_bw_gigabytes_per_second=*/1200.0);
  XPlaneBuilder tpu_builder(tpu_plane);
  XLineBuilder module_line = tpu_builder.GetOrCreateLine(0);
  module_line.SetName(kXlaModuleLineName);
  CreateXEventMetadata(
      &tpu_builder, "jit__apply(5621224425449793704)",
      {{StatType::kProgramId, static_cast<int64_t>(kProgramId)}});
  // The program runs in the request's own group: there is no batch.
  CreateXEvent(&tpu_builder, &module_line, "jit__apply(5621224425449793704)",
               /*offset_ps=*/1000, /*duration_ps=*/10000,
               {{StatType::kGroupId, kRequestGroupId}});

  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_builder(host_plane);
  XLineBuilder host_line = host_builder.GetOrCreateLine(0);
  host_line.SetName("RequestThread");
  CreateXEvent(&host_builder, &host_line, HostEventType::kSessionRun,
               /*offset_ps=*/500, /*duration_ps=*/12000,
               {{StatType::kGroupId, kRequestGroupId},
                {StatType::kModelId, "/models/ranking_model:1"}});

  GroupMetadataMap group_metadata_map;
  // A request group without children, so no batches.
  group_metadata_map[kRequestGroupId];
  std::vector<XPlane*> device_traces = {tpu_plane};
  StepEvents step_events;
  InferenceStats inference_stats;
  GenerateInferenceStats(device_traces, step_events, group_metadata_map, space,
                         DeviceType::kTpu, /*host_id=*/0, &inference_stats);
  RegroupInferenceStatsByModel(&inference_stats);

  // The request knows its model and ran on device, but no batch links it to
  // the program.
  ASSERT_EQ(inference_stats.inference_stats_per_model_size(), 1);
  const PerModelInferenceStats& per_model =
      inference_stats.inference_stats_per_model().begin()->second;
  ASSERT_EQ(per_model.request_details_size(), 1);
  const RequestDetail& request = per_model.request_details(0);
  ASSERT_GE(request.model_id_index(), 0);
  ASSERT_LT(request.model_id_index(), inference_stats.model_id_db().ids_size());
  EXPECT_EQ(inference_stats.model_id_db().ids(request.model_id_index()),
            "/models/ranking_model:1");
  EXPECT_GT(request.device_time_ps(), 0);
  EXPECT_THAT(request.related_batch_ids(), IsEmpty());
  EXPECT_THAT(per_model.batch_details(), IsEmpty());

  EXPECT_THAT(BuildProgramServingMetadata(inference_stats), IsEmpty());
}

}  // namespace
}  // namespace tensorflow::profiler
