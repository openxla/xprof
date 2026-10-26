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

#include "xprof/utils/derived_timeline.h"

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_test_utils.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Key;
using ::testing::UnorderedElementsAreArray;
using ::tsl::profiler::GetOrCreateGpuXPlane;
using ::tsl::profiler::kThreadIdCommandBufferMin;
using ::tsl::profiler::kThreadIdHloModule;
using ::tsl::profiler::kThreadIdTfNameScope;
using ::tsl::profiler::kThreadIdTfOp;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XEventVisitor;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XLineVisitor;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XPlaneVisitor;
using ::tsl::profiler::XStatValue;
using ::tsl::profiler::XStatVisitor;
using ::tsl::proto_testing::EqualsProto;

TEST(DerivedTimelineTest, EmptySpaceTest) {
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map;
  GenerateDerivedTimeLines(group_metadata_map, &space);
  EXPECT_EQ(space.planes_size(), 0);
}

// Checks that HLO module events are expanded.
TEST(DerivedTimelineTest, HloModuleNameTest) {
  const absl::string_view kHloModuleName = "hlo_module";
  const absl::string_view kKernelDetails = "kernel_details";
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map;
  XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kHloModule, kHloModuleName},
                {StatType::kKernelDetails, kKernelDetails}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300,
               {{StatType::kHloModule, kHloModuleName},
                {StatType::kKernelDetails, kKernelDetails}});
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
  // Only the hlo module line is added and other empty lines are removed at the
  // end.
  EXPECT_EQ(plane_visitor.NumLines(), 2);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (line_visitor.Id() == 0) return;
    EXPECT_EQ(line_visitor.Id() - tsl::profiler::kThreadIdDeviceDerivedMin,
              kThreadIdHloModule - kThreadIdTfNameScope);
    EXPECT_EQ(line_visitor.NumEvents(), 1);
    line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
      EXPECT_EQ(event_visitor.Name(), kHloModuleName);
    });
  });
}

// Checks that HLO module events are expanded, with both same name and scope
// range id. Note that strange XStatValue{int64_t{10}} is to handle different
// compilers behavior.
TEST(DerivedTimelineTest, HloModuleNameSameScopeRangeIdTest) {
  const absl::string_view kHloModuleName = "hlo_module";
  const absl::string_view kKernelDetails = "kernel_details";
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map;
  XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kHloModule, XStatValue{kHloModuleName}},
                {StatType::kKernelDetails, XStatValue{kKernelDetails}},
                {StatType::kScopeRangeId, XStatValue{int64_t{10}}}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300,
               {{StatType::kHloModule, XStatValue{kHloModuleName}},
                {StatType::kKernelDetails, XStatValue{kKernelDetails}},
                {StatType::kScopeRangeId, XStatValue{int64_t{10}}}});
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
  // Only the hlo module line is added and other empty lines are removed at the
  // end.
  EXPECT_EQ(plane_visitor.NumLines(), 2);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (line_visitor.Id() == 0) return;
    EXPECT_EQ(line_visitor.Id() - tsl::profiler::kThreadIdDeviceDerivedMin,
              kThreadIdHloModule - kThreadIdTfNameScope);
    EXPECT_EQ(line_visitor.NumEvents(), 1);
    line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
      EXPECT_EQ(event_visitor.Name(), kHloModuleName);
    });
  });
}

// Checks that HLO module events are expanded, with same name only,
// but different scope range id.
TEST(DerivedTimelineTest, HloModuleNameDifferentScopeRangeIdTest) {
  const absl::string_view kHloModuleName = "hlo_module";
  const absl::string_view kKernelDetails = "kernel_details";
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map;
  XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kHloModule, XStatValue{kHloModuleName}},
                {StatType::kKernelDetails, XStatValue{kKernelDetails}},
                {StatType::kScopeRangeId, XStatValue{int64_t{10}}}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300,
               {{StatType::kHloModule, XStatValue{kHloModuleName}},
                {StatType::kKernelDetails, XStatValue{kKernelDetails}},
                {StatType::kScopeRangeId, XStatValue{int64_t{20}}}});
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
  // Only the hlo module line is added and other empty lines are removed at the
  // end.
  EXPECT_EQ(plane_visitor.NumLines(), 2);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (line_visitor.Id() == 0) return;
    EXPECT_EQ(line_visitor.Id() - tsl::profiler::kThreadIdDeviceDerivedMin,
              kThreadIdHloModule - kThreadIdTfNameScope);
    EXPECT_EQ(line_visitor.NumEvents(), 2);
    line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
      EXPECT_EQ(event_visitor.Name(), kHloModuleName);
    });
  });
}

// Checks that HLO module events are expanded.
TEST(DerivedTimelineTest, NoHloModuleNameTest) {
  const absl::string_view kKernelDetails = "kernel_details";
  const uint64_t kCudaGraphExecId = 1;
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map;
  XPlane& plane = *GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(&plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kKernelDetails, kKernelDetails}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300,
               {{StatType::kKernelDetails, kKernelDetails}});
  // Also add a CudaGraph Execution event.
  CreateXEvent(&plane_builder, &line_builder, "op3", 500, 100,
               {{StatType::kCudaGraphExecId, kCudaGraphExecId}});
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(&plane);
  // Only the hlo module line is added and other empty lines are removed at the
  // end.
  EXPECT_EQ(plane_visitor.NumLines(), 1);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (line_visitor.Id() == 0) return;
    EXPECT_EQ(line_visitor.Id(), kThreadIdHloModule);
    EXPECT_EQ(line_visitor.NumEvents(), 0);
  });
}

// Checks that the TF op events are expanded.
TEST(DerivedTimelineTest, TfOpLineTest) {
  const absl::string_view kTfOpName = "mul:Mul";
  const absl::string_view kKernelDetails = "kernel_details";
  const uint64_t kCudaGraphExecId = 1;
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map;
  XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300,
               {{StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  // Also add a CudaGraph Execution event.
  CreateXEvent(&plane_builder, &line_builder, "op3", 500, 100,
               {{StatType::kTfOp, kTfOpName},
                {StatType::kCudaGraphExecId, kCudaGraphExecId}});
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
  // Only the tf op line is added and other empty lines are removed at the end.
  EXPECT_EQ(plane_visitor.NumLines(), 2);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (line_visitor.Id() == 0) return;
    EXPECT_EQ(line_visitor.Id() - tsl::profiler::kThreadIdDeviceDerivedMin,
              kThreadIdTfOp - kThreadIdTfNameScope);
    EXPECT_EQ(line_visitor.NumEvents(), 1);
    line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
      EXPECT_EQ(event_visitor.Name(), kTfOpName);
      EXPECT_EQ(event_visitor.OffsetPs(), 0);
      EXPECT_EQ(event_visitor.DurationPs(), 600);
    });
  });
}

// Checks that the dependency between the step line and the TF op line prevents
// TF op events from being expanded.
TEST(DerivedTimelineTest, DependencyTest) {
  constexpr int64_t kFirstGroupId = 0;
  constexpr int64_t kSecondGroupId = 1;

  const absl::string_view kTfOpName = "mul:Mul";
  const absl::string_view kKernelDetails = "kernel_details";
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map(
      {{0, {"train 0"}}, {1, {"train 1"}}});
  XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kGroupId, kFirstGroupId},
                {StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300,
               {{StatType::kGroupId, kSecondGroupId},
                {StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
  // The step line and the TF op line are added.
  EXPECT_EQ(plane_visitor.NumLines(), 3);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (line_visitor.Id() == 0) return;
    int64_t derived_line_id_offset =
        line_visitor.Id() - tsl::profiler::kThreadIdDeviceDerivedMin;
    EXPECT_TRUE(line_visitor.Id() == tsl::profiler::kThreadIdStepInfo ||
                derived_line_id_offset == kThreadIdTfOp - kThreadIdTfNameScope);
    EXPECT_EQ(line_visitor.NumEvents(), 2);
  });
}

// Checks that the TF op events are expanded.
TEST(DerivedTimelineTest, TfOpNameScopeTest) {
  const absl::string_view kTfOpName = "scope1/scope2/mul:Mul";
  const absl::string_view kKernelDetails = "kernel_details";
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map;
  XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300,
               {{StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
  // The TF name scope line and the TF op line are added.
  EXPECT_EQ(plane_visitor.NumLines(), 3);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    int64_t line_id = line_visitor.Id();
    if (line_id == 0) {
      return;
    } else if (line_id == kThreadIdTfNameScope) {
      EXPECT_EQ(line_visitor.NumEvents(), 2);
      line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
        EXPECT_EQ(event_visitor.OffsetPs(), 0);
        EXPECT_EQ(event_visitor.DurationPs(), 500);
      });
    } else if (line_id == kThreadIdTfOp) {
      EXPECT_EQ(line_visitor.NumEvents(), 1);
      line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
        EXPECT_EQ(event_visitor.Name(), kTfOpName);
        EXPECT_EQ(event_visitor.OffsetPs(), 0);
        EXPECT_EQ(event_visitor.DurationPs(), 500);
      });
    }
  });
}

// Checks that the TF op events are expanded.
TEST(DerivedTimelineTest, TfNameScopeMaintainsOrder) {
  const absl::string_view kTfOpName = "scope1/scope2/mul:Mul";
  const absl::string_view kKernelDetails = "kernel_details";
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map;
  XPlane* plane = tsl::profiler::GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, "TPU V4", 0, 0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 10000,
               {{StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
  // The TF name scope line and the TF op line are added.
  EXPECT_EQ(plane_visitor.NumLines(), 3);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (line_visitor.Name() == tsl::profiler::kTensorFlowNameScopeLineName) {
      EXPECT_EQ(line_visitor.NumEvents(), 2);
      uint64_t expected_duration = 10000;
      line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
        LOG(INFO) << "scope: " << event_visitor.Name();
        EXPECT_EQ(event_visitor.OffsetPs(), 0);
        EXPECT_EQ(event_visitor.DurationPs(), expected_duration);
        expected_duration -= 1000;
      });
    }
  });
}

// Checks derived events from each lines for gpu trace.
TEST(DerivedTimelineTest, OnlyDerivedEventsFromAllLines) {
  const std::string kStreamName1 = "stream1";
  const std::string kStreamName2 = "stream2";
  const std::string kScopeName1 = "scope1";
  const std::string kScopeName2 = "scope2";
  const std::string kOpName1 = "mul:Mul1";
  const std::string kOpName2 = "mul:Mul2";
  const std::string kStream1Scope1 = absl::StrCat(kStreamName1, kScopeName1);
  const std::string kStream1Scope2 = absl::StrCat(kStreamName1, kScopeName2);
  const std::string kStream2Scope1 = absl::StrCat(kStreamName2, kScopeName1);
  const std::string kStream2Scope2 = absl::StrCat(kStreamName2, kScopeName2);
  const std::string kFramWorkScopeName = "Framework Name Scope";
  const std::string kFromWorkeOpsName = "Framework Ops";
  // Adding the stream name and scope name to the tf op name to make it unique
  // and easier to identify.
  const std::string kTfOpName1 =
      absl::StrCat(kStream1Scope1, "/", kStream1Scope2, "/",
                   kOpName1);  // "stream1scope1/stream1scope2/mul:Mul1"
  const std::string kTfOpName2 =
      absl::StrCat(kStream2Scope1, "/", kStream2Scope2, "/",
                   kOpName2);  // "stream2scope1/stream2scope2/mul:Mul2"
  const absl::string_view kKernelDetails = "kernel_details";
  constexpr int64_t kEventLine0 = 0;
  constexpr int64_t kEventLine1 = 1;

  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map;
  XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(kEventLine0);
  // Add first line with two events.
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kTfOp, kTfOpName1},
                {StatType::kKernelDetails, kKernelDetails}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300,
               {{StatType::kTfOp, kTfOpName1},
                {StatType::kKernelDetails, kKernelDetails}});
  // Add second line with only one event.
  auto line_builder_2 = plane_builder.GetOrCreateLine(kEventLine1);
  CreateXEvent(&plane_builder, &line_builder_2, "op3", 50, 850,
               {{StatType::kTfOp, kTfOpName2},
                {StatType::kKernelDetails, kKernelDetails}});
  // Derive lines for the plane.
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
  // Two Events {op1, op2 in line 0} and {op3 in line 1} are added to the plane.
  // For each event, two lines are added, one for the TF name scope and one for
  // the TF op, making 6 lines in total.
  EXPECT_EQ(plane_visitor.NumLines(), 6);
  absl::flat_hash_map<
      std::string,
      absl::flat_hash_map<std::string, std::pair<uint64_t, uint64_t>>>
      expected_framework_values = {
          {"Framework Name Scope - from #0",
           {{kStream1Scope1, {0, 500}}, {kStream1Scope2, {0, 500}}}},
          {"Framework Ops - from #0", {{kTfOpName1, {0, 500}}}},
          {"Framework Name Scope - from #1",
           {{kStream2Scope1, {50, 850}}, {kStream2Scope2, {50, 850}}}},
          {"Framework Ops - from #1", {{kTfOpName2, {50, 850}}}}};
  absl::flat_hash_map<
      std::string,
      absl::flat_hash_map<std::string, std::pair<uint64_t, uint64_t>>>
      actual_framework_values;

  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (expected_framework_values.contains(line_visitor.Name())) {
      auto& actual_events = actual_framework_values[line_visitor.Name()];
      line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
        actual_events[event_visitor.Name()] = {event_visitor.OffsetPs(),
                                               event_visitor.DurationPs()};
      });
    }
  });
  // Check that every line is matched.
  EXPECT_THAT(actual_framework_values,
              UnorderedElementsAreArray(expected_framework_values));
}

// Checks that the TF op events are expanded.
TEST(DerivedTimelineTest, TfOpNameScopeShrinkTest) {
  {
    // Case 1: shirnk is possible.
    XSpace space;
    tsl::profiler::GroupMetadataMap group_metadata_map;
    XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
    XPlaneBuilder plane_builder(plane);
    auto line_builder = plane_builder.GetOrCreateLine(0);
    CreateXEvent(&plane_builder, &line_builder, "op1", 0, 10000,
                 {{StatType::kTfOp, "a/b/c/Add:Add"},
                  {StatType::kKernelDetails, "blah"}});
    CreateXEvent(
        &plane_builder, &line_builder, "op2", 20000, 30000,
        {{StatType::kTfOp, "a/d/Mul:Mul"}, {StatType::kKernelDetails, "blah"}});
    GenerateDerivedTimeLines(group_metadata_map, &space);
    XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
    // The TF name scope line and the TF op line are added.
    EXPECT_EQ(plane_visitor.NumLines(), 3);
    plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
      int64_t line_id = line_visitor.Id();
      if (line_id == 0) {
        return;
      } else if (line_id == kThreadIdTfNameScope) {
        EXPECT_EQ(line_visitor.NumEvents(), 4);
        std::map<absl::string_view, uint64_t> durations;
        line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
          durations[event_visitor.Name()] = event_visitor.DurationPs();
        });
        EXPECT_EQ(durations["a"], 50000);
        EXPECT_EQ(durations["b"], 10000);
        EXPECT_EQ(durations["c"], 9000);  // shrinked to be distinguish with b.
        EXPECT_EQ(durations["d"], 30000);
      }
    });
  }
  {
    // Case 2: shirnk is impossible due to top event is too small.
    XSpace space;
    tsl::profiler::GroupMetadataMap group_metadata_map;
    XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
    XPlaneBuilder plane_builder(plane);
    auto line_builder = plane_builder.GetOrCreateLine(0);
    CreateXEvent(&plane_builder, &line_builder, "op1", 0, 10000,
                 {{StatType::kTfOp, "a/b/c/d/e/Add:Add"},
                  {StatType::kKernelDetails, "blah"}});
    CreateXEvent(&plane_builder, &line_builder, "op2", 10000, 2000,
                 {{StatType::kTfOp, "a/b/c/d/f/Sub:Sub"},
                  {StatType::kKernelDetails, "blah"}});
    CreateXEvent(
        &plane_builder, &line_builder, "op3", 20000, 30000,
        {{StatType::kTfOp, "a/g/Mul:Mul"}, {StatType::kKernelDetails, "blah"}});
    GenerateDerivedTimeLines(group_metadata_map, &space);
    XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
    // The TF name scope line and the TF op line are added.
    EXPECT_EQ(plane_visitor.NumLines(), 3);
    bool visited_derived_line = false;
    plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
      int64_t line_id = line_visitor.Id();
      if (line_id == 0) {
        return;
      } else if (line_id - tsl::profiler::kThreadIdDeviceDerivedMin == 0) {
        visited_derived_line = true;
        EXPECT_EQ(line_visitor.NumEvents(), 7);
        std::map<absl::string_view, uint64_t> durations;
        line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
          durations[event_visitor.Name()] = event_visitor.DurationPs();
        });
        for (const auto& [name, duration] : durations) {
          LOG(ERROR) << name << ": " << duration;
        }
        EXPECT_EQ(durations["a"], 50000);
        EXPECT_EQ(durations["b"], 12000);
        EXPECT_EQ(durations["c"], 11000);  // shrinked to be distinguish with b.
        EXPECT_EQ(durations["d"], 11000);  // not shrinked because of f.
        EXPECT_EQ(durations["e"], 10000);
        EXPECT_EQ(durations["f"], 1000);
        EXPECT_EQ(durations["g"], 30000);
      }
    });
    EXPECT_TRUE(visited_derived_line);
  }
}

// Checks that XLA Ops mapping to CudaGraph launch has extra stats.
TEST(DerivedTimelineTest, XloOpHasCudaGraphStats) {
  constexpr absl::string_view kModuleName = "module";
  constexpr absl::string_view kHloOpName = "op_level_2";
  constexpr absl::string_view kKernelDetails = "kernel_details";
  constexpr int64_t kGroupIdValue = 1;
  constexpr int64_t kCorrelationIdValue = 10000;
  const uint64_t kCudaGraphIdValue = 20;
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map;

  // Build Input Plane/Line/Events and derive events from them.
  XPlane& plane = *GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(&plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kKernelDetails, kKernelDetails},
                {StatType::kGroupId, kGroupIdValue},
                {StatType::kHloModule, kModuleName},
                {StatType::kHloOp, kHloOpName},
                {StatType::kCorrelationId, kCorrelationIdValue},
                {StatType::kCudaGraphId, kCudaGraphIdValue}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300,
               {{StatType::kKernelDetails, kKernelDetails},
                {StatType::kGroupId, kGroupIdValue},
                {StatType::kHloModule, kModuleName},
                {StatType::kHloOp, kHloOpName},
                {StatType::kCorrelationId, kCorrelationIdValue},
                {StatType::kCudaGraphId, kCudaGraphIdValue}});
  GenerateDerivedTimeLines(group_metadata_map, &space);

  // Check that the HLO op line is added and has the extra stats for the first
  // derived event.
  size_t num_hlo_op_line = 0;
  size_t num_events = 0;
  std::optional<XStatVisitor> correlation_id;
  std::optional<XStatVisitor> cuda_graph_id;
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(&plane);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (line_visitor.Id() - tsl::profiler::kThreadIdDeviceDerivedMin ==
        tsl::profiler::kThreadIdHloOp - tsl::profiler::kThreadIdTfNameScope) {
      num_hlo_op_line++;
      if (num_hlo_op_line == 1) {
        num_events = line_visitor.NumEvents();
        line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
          correlation_id = event_visitor.GetStat(StatType::kCorrelationId);
          cuda_graph_id = event_visitor.GetStat(StatType::kCudaGraphId);
        });
      }
    }
  });
  EXPECT_EQ(num_hlo_op_line, 1);
  EXPECT_EQ(num_events, 1);
  ASSERT_TRUE(correlation_id.has_value());
  EXPECT_EQ(correlation_id->IntValue(), kCorrelationIdValue);
  ASSERT_TRUE(cuda_graph_id.has_value());
  EXPECT_EQ(cuda_graph_id->UintValue(), kCudaGraphIdValue);
}

TEST(DerivedTimelineTest, DeriveLinesForXlaCpuOps) {
  XPlane xplane;
  XPlaneBuilder plane_builder(&xplane);
  plane_builder.SetName(tsl::profiler::kHostThreadsPlaneName);

  absl::string_view main_line_name = "main";
  auto line_builder = plane_builder.GetOrCreateLine(0);
  line_builder.SetName(main_line_name);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kHloModule, "Module1"}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 400,
               {{StatType::kHloModule, "Module2"}});

  DeriveLinesForXlaCpuOps(&xplane);

  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(&xplane);
  EXPECT_EQ(plane_visitor.NumLines(), 2);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (line_visitor.Name() == main_line_name) return;
    line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
      if (event_visitor.Name() == "Module1") {
        EXPECT_EQ(event_visitor.DurationPs(), 100);
        EXPECT_EQ(event_visitor.OffsetPs(), 0);
      } else if (event_visitor.Name() == "Module2") {
        EXPECT_EQ(event_visitor.DurationPs(), 400);
        EXPECT_EQ(event_visitor.OffsetPs(), 200);
      } else {
        FAIL() << "Found Event " << event_visitor.Name();
      }
    });
  });
}

TEST(DerivedTimelineTest, MergeAndNoMerge) {
  constexpr absl::string_view kHloModuleName = "Framework Ops";
  static constexpr absl::string_view kTfOpName = "abc:model/layer/MatMul_1";
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map;
  XPlane* plane = tsl::profiler::GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, "DummyTPU", 1.0, 1.0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(
      &plane_builder, &line_builder, "op1", 0, 100,
      {{StatType::kHloModule, kHloModuleName}, {StatType::kTfOp, kTfOpName}});
  CreateXEvent(
      &plane_builder, &line_builder, "op2", 200, 300,
      {{StatType::kHloModule, kHloModuleName}, {StatType::kTfOp, kTfOpName}});
  // The above two events are merged into one. This event will not be merged
  // because the gap is > 2x(0..200+300) = 1000.
  CreateXEvent(
      &plane_builder, &line_builder, "op3", 1501, 300,
      {{StatType::kHloModule, kHloModuleName}, {StatType::kTfOp, kTfOpName}});
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
  // Only the hlo module line is added and other empty lines are removed at the
  // end.
  EXPECT_EQ(plane_visitor.NumLines(), 2);
  plane_visitor.ForEachLine([](const XLineVisitor& line_visitor) {
    if (line_visitor.Id() == 0) return;
    EXPECT_EQ(line_visitor.NumEvents(), 2);
    line_visitor.ForEachEvent([](const XEventVisitor& event_visitor) {
      EXPECT_EQ(event_visitor.Name(), kTfOpName);
    });
  });
}

TEST(DerivedTimelineTest, EnsureAllGpuEventsAreGrouped) {
  constexpr int64_t kFirstGroupId = 0;
  constexpr int64_t kSecondGroupId = 1;

  const absl::string_view kTfOpName = "mul:Mul";
  const absl::string_view kKernelDetails = "kernel_details";
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map(
      {{0, {"train 0"}}, {1, {"train 1"}}});
  XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kGroupId, kFirstGroupId},
                {StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300,
               {{StatType::kGroupId, kSecondGroupId},
                {StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  // Eager Op that happens after the second step.
  CreateXEvent(&plane_builder, &line_builder, "op3", 600, 100,
               {{StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
  // The step line and the TF op line are added.
  EXPECT_EQ(plane_visitor.NumLines(), 3);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
      SCOPED_TRACE(
          absl::StrCat(line_visitor.Name(), " ", event_visitor.Name()));
      EXPECT_TRUE(event_visitor.GetStat(StatType::kGroupId).has_value());
    });
  });
}

// Tests that the multi-threaded processing of Tensor Core planes works
// correctly.
TEST(DerivedTimelineTest, MultiThreadedTensorCorePlaneProcessing) {
  constexpr int kNumPlanes = 4;
  constexpr int kNumEvents = 10;
  constexpr int kEventDurationPs = 100;

  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map;

  // Create multiple Tensor Core planes, each with a line of unsorted events.
  for (int i = 0; i < kNumPlanes; ++i) {
    XPlane* plane = tsl::profiler::GetOrCreateTpuXPlane(
        &space, /*device_ordinal=*/i, "TPU V4", 0, 0);
    XPlaneBuilder plane_builder(plane);
    auto line_builder = plane_builder.GetOrCreateLine(0);
    const std::string tf_op_name = absl::StrCat("MyOp:", i);

    for (int j = 0; j < kNumEvents; ++j) {
      // Add events in reverse order to test sorting.
      int64_t offset = (kNumEvents - 1 - j) * kEventDurationPs * 2;
      CreateXEvent(&plane_builder, &line_builder, "kernel", offset,
                   kEventDurationPs, {{StatType::kTfOp, tf_op_name}});
    }
  }

  // This will trigger the multi-threaded logic you added.
  GenerateDerivedTimeLines(group_metadata_map, &space);

  // Verify that each plane was processed correctly.
  for (int i = 0; i < kNumPlanes; ++i) {
    const std::string plane_name = absl::StrCat("/device:TPU:", i);
    const XPlane* plane = tsl::profiler::FindPlaneWithName(space, plane_name);
    ASSERT_NE(plane, nullptr);
    XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);

    // 1. Verify that the events on the original line are now sorted.
    const XLine* original_line = nullptr;
    for (const auto& line : plane->lines()) {
      if (line.id() == 0) {
        original_line = &line;
        break;
      }
    }
    ASSERT_NE(original_line, nullptr);

    int64_t last_timestamp_ps = -1;
    for (const auto& event : original_line->events()) {
      ASSERT_GE(event.offset_ps(), last_timestamp_ps);
      last_timestamp_ps = event.offset_ps();
    }
    EXPECT_EQ(original_line->events_size(), kNumEvents);

    // 2. Verify that DeriveLinesFromStats created the derived TF Op line.
    bool tf_op_line_found = false;
    plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
      if (line_visitor.Name() == tsl::profiler::kTensorFlowOpLineName) {
        tf_op_line_found = true;
        EXPECT_EQ(line_visitor.NumEvents(), 1);  // Should be merged into one.
        line_visitor.ForEachEvent([&](const XEventVisitor& event) {
          EXPECT_EQ(event.Name(), absl::StrCat("MyOp:", i));
          // Check the duration of the merged event.
          int64_t expected_duration =
              (kNumEvents - 1) * kEventDurationPs * 2 + kEventDurationPs;
          EXPECT_EQ(event.DurationPs(), expected_duration);
        });
      }
    });
    EXPECT_TRUE(tf_op_line_found);
  }
}

TEST(DerivedTimelineTest, CycleDetectionTest) {
  XSpace space;
  XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);

  // Add an event with a scope_range_id.
  CreateXEvent(&plane_builder, &line_builder, "kernel", 0, 100,
               {{StatType::kHloModule, "Module"},
                {StatType::kKernelDetails, "Details"},
                {StatType::kScopeRangeId, XStatValue{int64_t{10}}}});

  ScopeRangeIdTree scope_range_id_tree;
  scope_range_id_tree[10] = 20;
  scope_range_id_tree[20] = 10;  // Cycle

  SymbolResolver symbol_resolver =
      [](std::optional<uint64_t> program_id, absl::string_view hlo_module_name,
         absl::string_view hlo_op) { return Symbol{hlo_module_name, "", ""}; };

  // This should finish without hanging/OOM.
  DeriveEventsFromAnnotations(symbol_resolver, plane, &scope_range_id_tree);

  SUCCEED();
}

constexpr int64_t kStreamId = 7;
constexpr int64_t kStepId = 1;
constexpr uint64_t kGraphA = 20;
constexpr uint64_t kGraphB = 21;
constexpr absl::string_view kModuleName = "module";

// ID of the Command Buffers line of the `rank`-th stream in processing order
// (compute streams first, then by descending number of events).
int64_t CommandBufferLineId(int rank) {
  return kThreadIdCommandBufferMin + rank;
}

// ID of the XLA Modules line of the `rank`-th stream in processing order. Each
// stream gets the derived line IDs kThreadIdTfNameScope..kThreadIdSource,
// shifted to start at kThreadIdDeviceDerivedMin + rank * 6.
int64_t HloModuleLineId(int rank) {
  constexpr int64_t kLinesPerStream =
      tsl::profiler::kThreadIdSource - kThreadIdTfNameScope + 1;
  return tsl::profiler::kThreadIdDeviceDerivedMin + rank * kLinesPerStream +
         (kThreadIdHloModule - kThreadIdTfNameScope);
}

// Returns the IDs of the lines of `plane` that have a display ID, sorted by
// display ID. Lines that the GPU stream ordering doesn't place, such as the
// Steps line, have none.
std::vector<int64_t> GetLineIdsInDisplayOrder(const XPlane& plane) {
  std::map<int64_t, int64_t> display_id_to_line_id;
  for (const auto& line : plane.lines()) {
    if (line.display_id() != 0) {
      display_id_to_line_id[line.display_id()] = line.id();
    }
  }
  std::vector<int64_t> line_ids;
  for (const auto& [display_id, line_id] : display_id_to_line_id) {
    line_ids.push_back(line_id);
  }
  return line_ids;
}

// A CUDA graph node event on a GPU stream, with the stats that the CUPTI
// collector and event grouping write for it.
struct NodeEvent {
  int64_t offset_ps;
  int64_t duration_ps;
  uint64_t graph_id;
  int64_t correlation_id;
  // kKernelDetails, kMemcpyDetails or kMemsetDetails.
  StatType details = StatType::kKernelDetails;
};

// Adds `node` to the line of `plane` with ID `stream_id`.
void AddNodeEvent(XPlaneBuilder& plane, int64_t stream_id,
                  const NodeEvent& node) {
  auto stat = [&plane](StatType type) -> const XStatMetadata& {
    return *plane.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(type));
  };
  XLineBuilder line = plane.GetOrCreateLine(stream_id);
  XEventBuilder event = line.AddEvent(*plane.GetOrCreateEventMetadata("node"));
  event.SetOffsetPs(node.offset_ps);
  event.SetDurationPs(node.duration_ps);
  event.AddStatValue(stat(node.details), "details");
  event.AddStatValue(stat(StatType::kGroupId), kStepId);
  event.AddStatValue(stat(StatType::kHloModule), kModuleName);
  event.AddStatValue(stat(StatType::kCudaGraphId), node.graph_id);
  event.AddStatValue(stat(StatType::kCorrelationId), node.correlation_id);
}

// A span on a Command Buffers line.
struct CommandBufferSpan {
  std::string name;
  int64_t offset_ps = 0;
  int64_t duration_ps = 0;
  std::optional<uint64_t> graph_id;
  std::optional<int64_t> correlation_id;
};

bool operator==(const CommandBufferSpan& a, const CommandBufferSpan& b) {
  return std::tie(a.name, a.offset_ps, a.duration_ps, a.graph_id,
                  a.correlation_id) == std::tie(b.name, b.offset_ps,
                                                b.duration_ps, b.graph_id,
                                                b.correlation_id);
}

std::ostream& operator<<(std::ostream& os, const CommandBufferSpan& span) {
  return os << "{\"" << span.name << "\", offset_ps=" << span.offset_ps
            << ", duration_ps=" << span.duration_ps
            << ", graph_id=" << ::testing::PrintToString(span.graph_id)
            << ", correlation_id="
            << ::testing::PrintToString(span.correlation_id) << "}";
}

struct CommandBufferLine {
  std::string name;
  std::vector<CommandBufferSpan> spans;
};

// Returns the Command Buffers lines of `plane`, by line ID.
std::map<int64_t, CommandBufferLine> GetCommandBufferLines(
    const XPlane& plane) {
  std::map<int64_t, CommandBufferLine> lines;
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(&plane);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (!absl::StartsWith(line_visitor.Name(),
                          tsl::profiler::kCommandBufferLineName)) {
      return;
    }
    CommandBufferLine& line = lines[line_visitor.Id()];
    line.name = std::string(line_visitor.Name());
    line_visitor.ForEachEvent([&](const XEventVisitor& event) {
      CommandBufferSpan span;
      span.name = std::string(event.Name());
      span.offset_ps = event.OffsetPs();
      span.duration_ps = event.DurationPs();
      if (std::optional<XStatVisitor> stat =
              event.GetStat(StatType::kCudaGraphId)) {
        span.graph_id = stat->IntOrUintValue();
      }
      if (std::optional<XStatVisitor> stat =
              event.GetStat(StatType::kCorrelationId)) {
        span.correlation_id = static_cast<int64_t>(stat->IntOrUintValue());
      }
      line.spans.push_back(std::move(span));
    });
  });
  return lines;
}

std::map<int64_t, std::string> GetLineNames(
    const std::map<int64_t, CommandBufferLine>& lines) {
  std::map<int64_t, std::string> names;
  for (const auto& [line_id, line] : lines) {
    names[line_id] = line.name;
  }
  return names;
}

// Step metadata doesn't affect the Command Buffers line, so pass none.
void DeriveLines(XSpace& space) {
  GenerateDerivedTimeLines(tsl::profiler::GroupMetadataMap(), &space);
}

TEST(DerivedTimelineTest, BackToBackLaunchesGetSeparateSpans) {
  XSpace space;
  XPlane& plane = *GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(&plane);
  // Two kernels of the first launch merge into one span; the second launch of
  // the same graph gets its own span.
  AddNodeEvent(plane_builder, kStreamId, {0, 100, kGraphA, 100});
  AddNodeEvent(plane_builder, kStreamId, {200, 300, kGraphA, 100});
  AddNodeEvent(plane_builder, kStreamId, {600, 150, kGraphA, 101});

  DeriveLines(space);

  std::map<int64_t, CommandBufferLine> lines = GetCommandBufferLines(plane);
  ASSERT_THAT(lines, ElementsAre(Key(CommandBufferLineId(0))));
  const CommandBufferLine& line = lines.begin()->second;
  EXPECT_EQ(line.name, "Command Buffers - from #7");
  EXPECT_THAT(
      line.spans,
      ElementsAre(
          CommandBufferSpan{"Command Buffer 20", 0, 500, kGraphA, 100},
          CommandBufferSpan{"Command Buffer 20", 600, 150, kGraphA, 101}));
}

// Graphs can interleave on one stream, e.g. a parent graph and a child graph
// on concurrent branches. A span never mixes graphs: each change of graph ID
// ends the current span, so each graph gets one span per run of events.
TEST(DerivedTimelineTest, InterleavedGraphsAlternateSpans) {
  XSpace space;
  XPlane& plane = *GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(&plane);
  AddNodeEvent(plane_builder, kStreamId, {0, 100, kGraphA, 1});
  AddNodeEvent(plane_builder, kStreamId, {50, 100, kGraphB, 2});
  AddNodeEvent(plane_builder, kStreamId, {200, 100, kGraphA, 1});
  AddNodeEvent(plane_builder, kStreamId, {300, 100, kGraphA, 1});
  AddNodeEvent(plane_builder, kStreamId, {400, 100, kGraphB, 2});

  DeriveLines(space);

  std::map<int64_t, CommandBufferLine> lines = GetCommandBufferLines(plane);
  ASSERT_THAT(lines, ElementsAre(Key(CommandBufferLineId(0))));
  EXPECT_THAT(
      lines.begin()->second.spans,
      ElementsAre(
          CommandBufferSpan{"Command Buffer 20", 0, 100, kGraphA, 1},
          CommandBufferSpan{"Command Buffer 21", 50, 100, kGraphB, 2},
          CommandBufferSpan{"Command Buffer 20", 200, 200, kGraphA, 1},
          CommandBufferSpan{"Command Buffer 21", 400, 100, kGraphB, 2}));
}

// Each stream gets its own line, named after the stream's line ID and placed
// by the stream's rank; here the ranks follow the number of events.
TEST(DerivedTimelineTest, EachStreamGetsItsOwnNamedLine) {
  XSpace space;
  XPlane& plane = *GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(&plane);
  // Stream 42 runs graph 1 three times, stream 7 graph 2 twice and stream 13
  // graph 3 once, each launch with one kernel.
  for (int i = 0; i < 3; ++i) {
    AddNodeEvent(plane_builder, 42, {i * 100, 50, 1, 10 + i});
  }
  for (int i = 0; i < 2; ++i) {
    AddNodeEvent(plane_builder, 7, {i * 100, 50, 2, 20 + i});
  }
  AddNodeEvent(plane_builder, 13, {0, 50, 3, 30});

  DeriveLines(space);

  std::map<int64_t, CommandBufferLine> lines = GetCommandBufferLines(plane);
  EXPECT_EQ(GetLineNames(lines),
            (std::map<int64_t, std::string>{
                {CommandBufferLineId(0), "Command Buffers - from #42"},
                {CommandBufferLineId(1), "Command Buffers - from #7"},
                {CommandBufferLineId(2), "Command Buffers - from #13"}}));
  EXPECT_THAT(
      lines[CommandBufferLineId(0)].spans,
      ElementsAre(CommandBufferSpan{"Command Buffer 1", 0, 50, 1, 10},
                  CommandBufferSpan{"Command Buffer 1", 100, 50, 1, 11},
                  CommandBufferSpan{"Command Buffer 1", 200, 50, 1, 12}));
  EXPECT_THAT(
      lines[CommandBufferLineId(1)].spans,
      ElementsAre(CommandBufferSpan{"Command Buffer 2", 0, 50, 2, 20},
                  CommandBufferSpan{"Command Buffer 2", 100, 50, 2, 21}));
  EXPECT_THAT(lines[CommandBufferLineId(2)].spans,
              ElementsAre(CommandBufferSpan{"Command Buffer 3", 0, 50, 3, 30}));
  // Each stream's Command Buffers line comes right after its other derived
  // lines (here only XLA Modules; empty lines are removed).
  EXPECT_THAT(GetLineIdsInDisplayOrder(plane),
              ElementsAre(42, HloModuleLineId(0), CommandBufferLineId(0), 7,
                          HloModuleLineId(1), CommandBufferLineId(1), 13,
                          HloModuleLineId(2), CommandBufferLineId(2)));
}

// Like the XLA Ops line, the Command Buffers line only uses kernels, so memcpy
// and memset nodes are left out: a span runs from the first to the last kernel
// of a launch, and a launch without kernels gets no span.
TEST(DerivedTimelineTest, MemcpyAndMemsetNodesAreExcluded) {
  XSpace space;
  XPlane& plane = *GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(&plane);
  // Graph A: memset, kernel, memcpy, kernel, memcpy.
  AddNodeEvent(plane_builder, kStreamId,
               {0, 50, kGraphA, 1, StatType::kMemsetDetails});
  AddNodeEvent(plane_builder, kStreamId, {100, 100, kGraphA, 1});
  AddNodeEvent(plane_builder, kStreamId,
               {250, 50, kGraphA, 1, StatType::kMemcpyDetails});
  AddNodeEvent(plane_builder, kStreamId, {300, 100, kGraphA, 1});
  AddNodeEvent(plane_builder, kStreamId,
               {450, 50, kGraphA, 1, StatType::kMemcpyDetails});
  // Graph B: only a memcpy and a memset.
  AddNodeEvent(plane_builder, kStreamId,
               {600, 50, kGraphB, 2, StatType::kMemcpyDetails});
  AddNodeEvent(plane_builder, kStreamId,
               {700, 50, kGraphB, 2, StatType::kMemsetDetails});

  DeriveLines(space);

  std::map<int64_t, CommandBufferLine> lines = GetCommandBufferLines(plane);
  ASSERT_THAT(lines, ElementsAre(Key(CommandBufferLineId(0))));
  EXPECT_THAT(lines.begin()->second.spans,
              ElementsAre(CommandBufferSpan{"Command Buffer 20", 100, 300,
                                            kGraphA, 1}));
}

// In GRAPH_TRACE mode CUPTI emits one event per graph launch instead of one
// per node, and the Command Buffers line repeats those events one-to-one.
TEST(DerivedTimelineTest, GraphTraceEventsMapOneToOne) {
  XSpace space;
  XPlane& plane = *GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(&plane);
  XLineBuilder line_builder = plane_builder.GetOrCreateLine(kStreamId);
  auto add_graph_launch = [&](int64_t offset_ps, uint64_t graph_id,
                              int64_t correlation_id) {
    tsl::profiler::CreateXEvent(&plane_builder, &line_builder, "CudaGraphExec",
                                offset_ps, /*duration_ps=*/300,
                                {{StatType::kCudaGraphExecId, graph_id},
                                 {StatType::kCudaGraphId, graph_id},
                                 {StatType::kCorrelationId, correlation_id},
                                 {StatType::kGroupId, kStepId},
                                 {StatType::kHloModule, kModuleName}});
  };
  add_graph_launch(0, kGraphA, 1);
  add_graph_launch(600, kGraphA, 2);
  add_graph_launch(1000, kGraphB, 3);

  DeriveLines(space);

  std::map<int64_t, CommandBufferLine> lines = GetCommandBufferLines(plane);
  ASSERT_THAT(lines, ElementsAre(Key(CommandBufferLineId(0))));
  EXPECT_THAT(
      lines.begin()->second.spans,
      ElementsAre(
          CommandBufferSpan{"Command Buffer 20", 0, 300, kGraphA, 1},
          CommandBufferSpan{"Command Buffer 20", 600, 300, kGraphA, 2},
          CommandBufferSpan{"Command Buffer 21", 1000, 300, kGraphB, 3}));
}

// Only GPU planes get the line. TPU planes only get lines derived from their
// stats, and the host plane is only read, so CUDA graph stats on either add no
// line and change nothing.
TEST(DerivedTimelineTest, NonGpuPlanesGetNoLine) {
  XSpace space;
  XPlane& gpu_plane = *GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlane& tpu_plane = *tsl::profiler::GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, "TPU V4", 0, 0);
  XPlane& host_plane = *tsl::profiler::GetOrCreateHostXPlane(&space);
  for (XPlane* plane : {&gpu_plane, &tpu_plane, &host_plane}) {
    XPlaneBuilder plane_builder(plane);
    AddNodeEvent(plane_builder, kStreamId, {0, 100, kGraphA, 1});
    AddNodeEvent(plane_builder, kStreamId, {200, 100, kGraphA, 1});
  }
  const XPlane tpu_plane_before = tpu_plane;
  const XPlane host_plane_before = host_plane;

  DeriveLines(space);

  EXPECT_THAT(GetCommandBufferLines(gpu_plane),
              ElementsAre(Key(CommandBufferLineId(0))));
  EXPECT_THAT(GetCommandBufferLines(tpu_plane), IsEmpty());
  EXPECT_THAT(GetCommandBufferLines(host_plane), IsEmpty());
  // Deriving lines from stats can add stat metadata to a TPU plane, so only
  // compare its lines.
  ASSERT_EQ(tpu_plane.lines_size(), tpu_plane_before.lines_size());
  for (int i = 0; i < tpu_plane.lines_size(); ++i) {
    EXPECT_THAT(tpu_plane.lines(i), EqualsProto(tpu_plane_before.lines(i)));
  }
  EXPECT_THAT(host_plane, EqualsProto(host_plane_before));
}

// Non-GPU planes derive their lines at the fixed IDs kThreadIdTfNameScope to
// kThreadIdSource. They get no Command Buffers line, so no derived line lands
// in the host offload op range that follows kThreadIdSource.
TEST(DerivedTimelineTest, NonGpuDerivationAddsNoCommandBufferLine) {
  XSpace space;
  XPlane& plane = *tsl::profiler::GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, "TPU V4", 0, 0);
  XPlaneBuilder plane_builder(&plane);
  AddNodeEvent(plane_builder, kStreamId, {0, 100, kGraphA, 1});
  AddNodeEvent(plane_builder, kStreamId, {200, 100, kGraphA, 1});
  SymbolResolver symbol_resolver =
      [](std::optional<uint64_t> program_id, absl::string_view hlo_module_name,
         absl::string_view hlo_op) { return Symbol{hlo_module_name, "", ""}; };

  DeriveEventsFromAnnotations(symbol_resolver, &plane);

  EXPECT_THAT(GetCommandBufferLines(plane), IsEmpty());
  std::vector<int64_t> derived_line_ids;
  for (const auto& line : plane.lines()) {
    if (tsl::profiler::IsDerivedThreadId(line.id())) {
      derived_line_ids.push_back(line.id());
    }
  }
  EXPECT_THAT(derived_line_ids, ElementsAre(kThreadIdHloModule));
  for (int64_t line_id : derived_line_ids) {
    EXPECT_FALSE(line_id >= tsl::profiler::kThreadIdHostOffloadOpStart &&
                 line_id <= tsl::profiler::kThreadIdHostOffloadOpEnd)
        << line_id;
  }
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
