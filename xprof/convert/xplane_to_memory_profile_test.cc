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

#include "xprof/convert/xplane_to_memory_profile.h"

#include <cstdint>

#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/xplane_test_utils.h"
#include <gtest/gtest.h>
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "plugin/xprof/protobuf/memory_profile.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using tsl::profiler::GetOrCreateHostXPlane;

// Tests with a sample profile with multiple memory allocation and deallocation
// activities within one memory allocator captured in host trace.
TEST(ConvertXPlaneToMemoryProfile, OneAllocatorMultiActivitiesTest) {
  XSpace space;
  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  host_plane_builder.ReserveLines(1);

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "MemoryAllocation",
               40000, 1000,
               {{StatType::kBytesReserved, int64_t{2000}},
                {StatType::kBytesAllocated, int64_t{3000}},
                {StatType::kBytesAvailable, int64_t{5000}},
                {StatType::kPeakBytesInUse, int64_t{8500}},
                {StatType::kRequestedBytes, int64_t{200}},
                {StatType::kAllocationBytes, int64_t{256}},
                {StatType::kAddress, int64_t{222333}},
                {StatType::kStepId, int64_t{-93746}},
                {StatType::kDataType, int64_t{1}},
                {StatType::kAllocatorName, "GPU_0_bfc"},
                {StatType::kTfOp, "foo/bar"},
                {StatType::kRegionType, "output"},
                {StatType::kTensorShapes, "[3, 3, 512, 512]"}});

  MemoryProfile allocation_only_profile =
      ConvertXPlaneToMemoryProfile(*host_plane);
  const auto& allocation_only_summary =
      allocation_only_profile.memory_profile_per_allocator()
          .at("GPU_0_bfc")
          .profile_summary();
  EXPECT_TRUE(allocation_only_summary.has_num_allocations());
  EXPECT_EQ(allocation_only_summary.num_allocations(), 1);
  EXPECT_TRUE(allocation_only_summary.has_num_deallocations());
  EXPECT_EQ(allocation_only_summary.num_deallocations(), 0);

  CreateXEvent(&host_plane_builder, &tf_executor_thread, "MemoryDeallocation",
               50000, 1000,
               {{StatType::kBytesReserved, int64_t{2000}},
                {StatType::kBytesAllocated, int64_t{2744}},
                {StatType::kBytesAvailable, int64_t{5256}},
                {StatType::kPeakBytesInUse, int64_t{8500}},
                {StatType::kRequestedBytes, int64_t{200}},
                {StatType::kAllocationBytes, int64_t{256}},
                {StatType::kAddress, int64_t{222333}},
                {StatType::kStepId, int64_t{0}},
                {StatType::kDataType, int64_t{0}},
                {StatType::kAllocatorName, "GPU_0_bfc"},
                {StatType::kRegionType, ""},
                {StatType::kTensorShapes, ""}});

  CreateXEvent(&host_plane_builder, &tf_executor_thread, "MemoryAllocation",
               70000, 1000,
               {{StatType::kBytesReserved, int64_t{2000}},
                {StatType::kBytesAllocated, int64_t{5000}},
                {StatType::kBytesAvailable, int64_t{3000}},
                {StatType::kPeakBytesInUse, int64_t{9500}},
                {StatType::kRequestedBytes, int64_t{300}},
                {StatType::kAllocationBytes, int64_t{300}},
                {StatType::kAddress, int64_t{345678}},
                {StatType::kStepId, int64_t{-93746}},
                {StatType::kDataType, int64_t{9}},
                {StatType::kAllocatorName, "GPU_0_bfc"},
                {StatType::kTfOp, "mul_grad/Sum"},
                {StatType::kRegionType, "temp"},
                {StatType::kTensorShapes, "[1, 2]"}});

  tsl::profiler::GroupTfEvents(&space);
  MemoryProfile memory_profile = ConvertXPlaneToMemoryProfile(*host_plane);
  GenerateHloModules(space, &memory_profile);
  EXPECT_EQ(memory_profile.memory_profile_per_allocator().size(), 1);
  EXPECT_EQ(memory_profile.num_hosts(), 1);
  EXPECT_EQ(memory_profile.memory_ids_size(), 1);
  EXPECT_EQ(memory_profile.memory_profile_per_allocator().begin()->first,
            "GPU_0_bfc");
  EXPECT_EQ(memory_profile.version(), 1);
  const auto& allocator_memory_profile =
      memory_profile.memory_profile_per_allocator().begin()->second;
  EXPECT_EQ(
      allocator_memory_profile.profile_summary().peak_bytes_usage_lifetime(),
      9500);
  EXPECT_EQ(allocator_memory_profile.profile_summary()
                .peak_stats()
                .peak_bytes_in_use(),
            7000);
  EXPECT_EQ(allocator_memory_profile.profile_summary().peak_stats_time_ps(),
            70000);
  EXPECT_EQ(allocator_memory_profile.profile_summary().num_allocations(), 2);
  EXPECT_EQ(allocator_memory_profile.profile_summary().num_deallocations(), 1);
  EXPECT_EQ(allocator_memory_profile.sampled_timeline_snapshots_size(), 3);
  EXPECT_EQ(allocator_memory_profile.memory_profile_snapshots_size(), 1);
  EXPECT_EQ(allocator_memory_profile.memory_profile_snapshots()
                .at(0)
                .activity_metadata()
                .tf_op_name(),
            "mul_grad/Sum");
  EXPECT_EQ(allocator_memory_profile.active_allocations_size(), 3);
  EXPECT_EQ(
      allocator_memory_profile.active_allocations().at(2).snapshot_index(), 0);
  EXPECT_EQ(allocator_memory_profile.special_allocations_size(), 2);
  EXPECT_EQ(allocator_memory_profile.special_allocations().at(1).tf_op_name(),
            "stack");
  EXPECT_EQ(
      allocator_memory_profile.special_allocations().at(1).allocation_bytes(),
      2000);

  MemoryProfile downsampled_profile =
      ConvertXPlaneToMemoryProfile(*host_plane, /*max_num_snapshots=*/1);
  const auto& downsampled_allocator_profile =
      downsampled_profile.memory_profile_per_allocator().begin()->second;
  EXPECT_EQ(downsampled_allocator_profile.sampled_timeline_snapshots_size(), 1);
  EXPECT_EQ(downsampled_allocator_profile.memory_profile_snapshots_size(), 1);
  EXPECT_EQ(downsampled_allocator_profile.profile_summary().num_allocations(),
            2);
  EXPECT_EQ(downsampled_allocator_profile.profile_summary().num_deallocations(),
            1);
}

TEST(ConvertXPlaneToMemoryProfile,
     MultiThreadEqualPeaksAndDeallocationRequestedBytes) {
  XSpace space;
  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  host_plane_builder.ReserveLines(2);

  // Thread 0 has peak at t=70000 in step 2; Thread 1 (visited second) has
  // equal peak (with higher lifetime peak 9000 vs 5000) at earlier t=30000 in
  // step 1 and a deallocation at t=40000 without kRequestedBytes.
  auto thread_0 = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &thread_0, "MemoryAllocation", 70000, 1000,
               {{StatType::kBytesReserved, int64_t{1000}},
                {StatType::kBytesAllocated, int64_t{4000}},
                {StatType::kBytesAvailable, int64_t{5000}},
                {StatType::kPeakBytesInUse, int64_t{5000}},
                {StatType::kRequestedBytes, int64_t{400}},
                {StatType::kAllocationBytes, int64_t{512}},
                {StatType::kAddress, int64_t{999}},
                {StatType::kGroupId, int64_t{2}},
                {StatType::kDataType, int64_t{1}},
                {StatType::kAllocatorName, "GPU_0_bfc"},
                {StatType::kTfOp, "step2_op"},
                {StatType::kRegionType, "temp"},
                {StatType::kTensorShapes, "[128]"}});

  auto thread_1 = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &thread_1, "MemoryAllocation", 30000, 1000,
               {{StatType::kBytesReserved, int64_t{1000}},
                {StatType::kBytesAllocated, int64_t{4000}},
                {StatType::kBytesAvailable, int64_t{5000}},
                {StatType::kPeakBytesInUse, int64_t{9000}},
                {StatType::kRequestedBytes, int64_t{200}},
                {StatType::kAllocationBytes, int64_t{256}},
                {StatType::kAddress, int64_t{111}},
                {StatType::kGroupId, int64_t{1}},
                {StatType::kDataType, int64_t{1}},
                {StatType::kAllocatorName, "GPU_0_bfc"},
                {StatType::kTfOp, "step1_op"},
                {StatType::kRegionType, "output"},
                {StatType::kTensorShapes, "[64]"}});
  CreateXEvent(&host_plane_builder, &thread_1, "MemoryDeallocation", 40000,
               1000,
               {{StatType::kBytesReserved, int64_t{1000}},
                {StatType::kBytesAllocated, int64_t{3744}},
                {StatType::kBytesAvailable, int64_t{5256}},
                {StatType::kPeakBytesInUse, int64_t{5000}},
                {StatType::kAllocationBytes, int64_t{256}},
                {StatType::kAddress, int64_t{111}},
                {StatType::kGroupId, int64_t{1}},
                {StatType::kAllocatorName, "GPU_0_bfc"}});

  MemoryProfile memory_profile = ConvertXPlaneToMemoryProfile(*host_plane);
  const auto& allocator_profile =
      memory_profile.memory_profile_per_allocator().at("GPU_0_bfc");
  EXPECT_EQ(allocator_profile.profile_summary().num_allocations(), 2);
  EXPECT_EQ(allocator_profile.profile_summary().num_deallocations(), 1);
  EXPECT_EQ(allocator_profile.profile_summary().peak_bytes_usage_lifetime(),
            9000);
  EXPECT_EQ(allocator_profile.profile_summary().peak_stats_time_ps(), 70000);
  ASSERT_EQ(allocator_profile.memory_profile_snapshots_size(), 1);
  EXPECT_EQ(allocator_profile.memory_profile_snapshots(0)
                .activity_metadata()
                .tf_op_name(),
            "step2_op");
  ASSERT_GE(allocator_profile.sampled_timeline_snapshots_size(), 2);
  EXPECT_EQ(allocator_profile.sampled_timeline_snapshots(1)
                .activity_metadata()
                .requested_bytes(),
            200);
}

TEST(ConvertXPlaneToMemoryProfile,
     PreservesNonZeroDeallocationRequestedBytesWhenAllocIsZero) {
  XSpace space;
  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  auto line = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &line, "MemoryAllocation", 10000, 1000,
               {{StatType::kBytesReserved, int64_t{1000}},
                {StatType::kBytesAllocated, int64_t{512}},
                {StatType::kBytesAvailable, int64_t{5000}},
                {StatType::kPeakBytesInUse, int64_t{512}},
                {StatType::kAllocationBytes, int64_t{512}},
                {StatType::kAddress, int64_t{222}},
                {StatType::kAllocatorName, "GPU_0_bfc"}});
  CreateXEvent(&host_plane_builder, &line, "MemoryDeallocation", 20000, 1000,
               {{StatType::kBytesReserved, int64_t{1000}},
                {StatType::kBytesAllocated, int64_t{0}},
                {StatType::kBytesAvailable, int64_t{5512}},
                {StatType::kPeakBytesInUse, int64_t{512}},
                {StatType::kRequestedBytes, int64_t{384}},
                {StatType::kAllocationBytes, int64_t{512}},
                {StatType::kAddress, int64_t{222}},
                {StatType::kAllocatorName, "GPU_0_bfc"}});

  MemoryProfile memory_profile = ConvertXPlaneToMemoryProfile(*host_plane);
  const auto& allocator_profile =
      memory_profile.memory_profile_per_allocator().at("GPU_0_bfc");
  ASSERT_EQ(allocator_profile.sampled_timeline_snapshots_size(), 2);
  EXPECT_EQ(allocator_profile.sampled_timeline_snapshots(1)
                .activity_metadata()
                .requested_bytes(),
            384);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
