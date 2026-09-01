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

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
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

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::tsl::profiler::CreateXEvent;
using ::tsl::profiler::DeviceType;
using ::tsl::profiler::GetOrCreateHostXPlane;
using ::tsl::profiler::GroupMetadata;
using ::tsl::profiler::GroupMetadataMap;
using ::tsl::profiler::HostEventType;
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

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
