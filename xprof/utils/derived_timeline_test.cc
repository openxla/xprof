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
#include <string>
#include <utility>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_test_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::UnorderedElementsAreArray;
using ::tsl::profiler::GetOrCreateGpuXPlane;
using ::tsl::profiler::kThreadIdHloModule;
using ::tsl::profiler::kThreadIdTfNameScope;
using ::tsl::profiler::kThreadIdTfOp;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventVisitor;
using ::tsl::profiler::XLineVisitor;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XPlaneVisitor;
using ::tsl::profiler::XStatValue;
using ::tsl::profiler::XStatVisitor;

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
    EXPECT_EQ(line_visitor.Id(), kThreadIdHloModule);
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
    EXPECT_EQ(line_visitor.Id(), kThreadIdHloModule);
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
    EXPECT_EQ(line_visitor.Id(), kThreadIdHloModule);
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
    EXPECT_EQ(line_visitor.Id(), kThreadIdTfOp);
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
    EXPECT_TRUE(line_visitor.Id() == tsl::profiler::kThreadIdStepInfo ||
                line_visitor.Id() == kThreadIdTfOp);
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
    plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
      int64_t line_id = line_visitor.Id();
      if (line_id == 0) {
        return;
      } else if (line_id == kThreadIdTfNameScope) {
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
    if (line_visitor.Id() == tsl::profiler::kThreadIdHloOp) {
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

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
