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

#include "xprof/convert/inference_stats.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/tsl/profiler/utils/device_utils.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_test_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "plugin/xprof/protobuf/inference_stats.pb.h"
#include "xprof/utils/event_span.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Property;
using ::testing::UnorderedElementsAre;
using ::tsl::profiler::CreateXEvent;
using ::tsl::profiler::CreateXEventMetadata;
using ::tsl::profiler::DeviceType;
using ::tsl::profiler::GetOrCreateHostXPlane;
using ::tsl::profiler::GetOrCreateTpuXPlane;
using ::tsl::profiler::GroupMetadata;
using ::tsl::profiler::GroupMetadataMap;
using ::tsl::profiler::HostEventType;
using ::tsl::profiler::kXlaModuleLineName;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlane;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XSpace;

TEST(InferenceStatsTest, GenerateTensorPatternWithTensorShapes) {
  XSpace space;
  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  XLineBuilder main_thread = host_plane_builder.GetOrCreateLine(0);

  constexpr int64_t kGroupId = 100;
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kSessionRun,
               1000, 5000, {{StatType::kGroupId, kGroupId}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kLinearize,
               1200, 800,
               {{StatType::kGroupId, kGroupId},
                {StatType::kTensorShapes, "[1,2,3]"},
                {StatType::kTensorLayout, "{0,1,2}"}});

  GroupMetadataMap group_metadata_map;
  group_metadata_map[kGroupId] = GroupMetadata();
  StepEvents step_events;
  InferenceStats inference_stats;

  GenerateInferenceStats(/*device_traces=*/{}, step_events, group_metadata_map,
                         space, DeviceType::kCpu, /*host_id=*/0,
                         &inference_stats);

  EXPECT_THAT(inference_stats.tensor_pattern_db().tensor_pattern(),
              ElementsAre("Linearize [1,2,3] {0,1,2}"));
}

TEST(InferenceStatsTest, GenerateTensorPatternWithDimensionsAndType) {
  XSpace space;
  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  XLineBuilder main_thread = host_plane_builder.GetOrCreateLine(0);

  constexpr int64_t kGroupId = 100;
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kSessionRun,
               1000, 5000, {{StatType::kGroupId, kGroupId}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kLinearize,
               1200, 800,
               {{StatType::kGroupId, kGroupId},
                {StatType::kDimensions, "[10,20]"},
                {StatType::kType, "F32"},
                {StatType::kTensorLayout, "{1,0}"}});

  GroupMetadataMap group_metadata_map;
  group_metadata_map[kGroupId] = GroupMetadata();
  StepEvents step_events;
  InferenceStats inference_stats;

  GenerateInferenceStats(/*device_traces=*/{}, step_events, group_metadata_map,
                         space, DeviceType::kCpu, /*host_id=*/0,
                         &inference_stats);

  EXPECT_THAT(inference_stats.tensor_pattern_db().tensor_pattern(),
              ElementsAre("Linearize f32[10,20] {1,0}"));
}

TEST(InferenceStatsTest, GenerateTensorPatternWithDimensionsOnly) {
  XSpace space;
  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  XLineBuilder main_thread = host_plane_builder.GetOrCreateLine(0);

  constexpr int64_t kGroupId = 100;
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kSessionRun,
               1000, 5000, {{StatType::kGroupId, kGroupId}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kLinearize,
               1200, 800,
               {{StatType::kGroupId, kGroupId},
                {StatType::kDimensions, "[4,8]"},
                {StatType::kTensorLayout, "{0,1}"}});

  GroupMetadataMap group_metadata_map;
  group_metadata_map[kGroupId] = GroupMetadata();
  StepEvents step_events;
  InferenceStats inference_stats;

  GenerateInferenceStats(/*device_traces=*/{}, step_events, group_metadata_map,
                         space, DeviceType::kCpu, /*host_id=*/0,
                         &inference_stats);

  EXPECT_THAT(inference_stats.tensor_pattern_db().tensor_pattern(),
              ElementsAre("Linearize [4,8] {0,1}"));
}

TEST(InferenceStatsTest, GenerateTensorPatternMultipleSorted) {
  XSpace space;
  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  XLineBuilder main_thread = host_plane_builder.GetOrCreateLine(0);

  constexpr int64_t kGroupId = 100;
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kSessionRun,
               1000, 5000, {{StatType::kGroupId, kGroupId}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kLinearize,
               1200, 800,
               {{StatType::kGroupId, kGroupId},
                {StatType::kDimensions, "[1,2]"},
                {StatType::kType, "s32"},
                {StatType::kTensorLayout, "{1,0}"}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kDelinearize,
               2000, 500,
               {{StatType::kGroupId, kGroupId},
                {StatType::kTensorShapes, "[3,4]"},
                {StatType::kTensorLayout, "{0,1}"}});

  GroupMetadataMap group_metadata_map;
  group_metadata_map[kGroupId] = GroupMetadata();
  StepEvents step_events;
  InferenceStats inference_stats;

  GenerateInferenceStats(/*device_traces=*/{}, step_events, group_metadata_map,
                         space, DeviceType::kCpu, /*host_id=*/0,
                         &inference_stats);

  // Sub-patterns should be sorted alphabetically: "Delinearize..." comes before
  // "Linearize..."
  EXPECT_THAT(
      inference_stats.tensor_pattern_db().tensor_pattern(),
      ElementsAre("Delinearize [3,4] {0,1}<br>Linearize s32[1,2] {1,0}"));
}

TEST(InferenceStatsTest, GenerateTensorPatternMissingLayoutReturnsEmpty) {
  XSpace space;
  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  XLineBuilder main_thread = host_plane_builder.GetOrCreateLine(0);

  constexpr int64_t kGroupId = 100;
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kSessionRun,
               1000, 5000, {{StatType::kGroupId, kGroupId}});
  CreateXEvent(
      &host_plane_builder, &main_thread, HostEventType::kLinearize, 1200, 800,
      {{StatType::kGroupId, kGroupId}, {StatType::kDimensions, "[1,2]"}});

  GroupMetadataMap group_metadata_map;
  group_metadata_map[kGroupId] = GroupMetadata();
  StepEvents step_events;
  InferenceStats inference_stats;

  GenerateInferenceStats(/*device_traces=*/{}, step_events, group_metadata_map,
                         space, DeviceType::kCpu, /*host_id=*/0,
                         &inference_stats);

  EXPECT_THAT(inference_stats.tensor_pattern_db().tensor_pattern(), IsEmpty());
}

TEST(InferenceStatsTest, RecordsDeduplicatedProgramIdsOnRequestDetail) {
  constexpr int64_t kRequestGroupId = 100;
  constexpr int64_t kChildGroupId = 101;
  constexpr uint64_t kProgramIdFromStat = 5621224425449793704ULL;
  constexpr uint64_t kProgramIdFromName = 7777777777777777777ULL;

  XSpace space;
  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  XLineBuilder main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kSessionRun,
               1000, 10000,
               {{StatType::kGroupId, kRequestGroupId},
                {StatType::kModelId, "/models/unbatched_model:1"}});

  XPlane* tpu_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/275.0,
      /*peak_hbm_bw_gigabytes_per_second=*/1200.0);
  XPlaneBuilder tpu_builder(tpu_plane);
  XLineBuilder module_line = tpu_builder.GetOrCreateLine(0);
  module_line.SetName(kXlaModuleLineName);
  CreateXEventMetadata(
      &tpu_builder, "jit_step",
      {{StatType::kProgramId, static_cast<int64_t>(kProgramIdFromStat)}});
  // Two executions of the same program directly under the request's group_id,
  // plus one execution parsed from the HLO module name under a child group_id.
  CreateXEvent(&tpu_builder, &module_line, "jit_step", 2000, 1000,
               {{StatType::kGroupId, kRequestGroupId}});
  CreateXEvent(&tpu_builder, &module_line, "jit_step", 3500, 1000,
               {{StatType::kGroupId, kRequestGroupId}});
  CreateXEvent(&tpu_builder, &module_line,
               "fallback_module(7777777777777777777)", 5000, 1000,
               {{StatType::kGroupId, kChildGroupId}});

  GroupMetadataMap group_metadata_map;
  group_metadata_map[kRequestGroupId].children.insert(kChildGroupId);
  group_metadata_map[kChildGroupId].parents.insert(kRequestGroupId);

  std::vector<XPlane*> device_traces = {tpu_plane};
  StepEvents step_events;
  InferenceStats inference_stats;
  GenerateInferenceStats(device_traces, step_events, group_metadata_map, space,
                         DeviceType::kTpu, /*host_id=*/0, &inference_stats);

  ASSERT_TRUE(inference_stats.inference_stats_per_host().contains(0));
  const PerHostInferenceStats& per_host =
      inference_stats.inference_stats_per_host().at(0);
  ASSERT_EQ(per_host.request_details_size(), 1);
  EXPECT_THAT(per_host.request_details(0).program_ids(),
              ElementsAre(kProgramIdFromStat, kProgramIdFromName));
}

TEST(InferenceStatsTest, UnmodeledUngroupedRequestUsesNegativeModelIdIndex) {
  XSpace space;
  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  XLineBuilder main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(
      &host_plane_builder, &main_thread, HostEventType::kSessionRun, 1000, 1000,
      {{StatType::kGroupId, 1}, {StatType::kModelId, "/models/first_model:1"}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kSessionRun,
               3000, 1000, {{StatType::kGroupId, 2}});

  GroupMetadataMap group_metadata_map;
  group_metadata_map[1];

  std::vector<XPlane*> device_traces;
  StepEvents step_events;
  InferenceStats inference_stats;
  GenerateInferenceStats(device_traces, step_events, group_metadata_map, space,
                         DeviceType::kTpu, /*host_id=*/0, &inference_stats);

  ASSERT_TRUE(inference_stats.inference_stats_per_host().contains(0));
  const PerHostInferenceStats& per_host =
      inference_stats.inference_stats_per_host().at(0);
  EXPECT_THAT(per_host.request_details(),
              UnorderedElementsAre(
                  AllOf(Property(&RequestDetail::request_id, 1),
                        Property(&RequestDetail::model_id_index, 0)),
                  AllOf(Property(&RequestDetail::request_id, 2),
                        Property(&RequestDetail::model_id_index, -1))));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
