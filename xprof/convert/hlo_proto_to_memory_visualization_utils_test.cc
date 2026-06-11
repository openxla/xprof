/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/hlo_proto_to_memory_visualization_utils.h"

#include <string>

#include "<gtest/gtest.h>"
#include "absl/strings/str_format.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "plugin/xprof/protobuf/memory_viewer_preprocess.pb.h"
#include "xprof/utils/tensorflow_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

// 1 buffer allocation of 1MB
// 2 logical buffers, each is 0.5MB
static constexpr char kHLOBase[] = R"pb(
  hlo_module {
    name: "test_module"
    entry_computation_name: "test_computation"
    computations {
      name: "test_computation"
      instructions {
        name: "fusion.1"
        id: 0
        shape { tuple_shapes { element_type: U64 } }
      }
      instructions {
        name: "fusion.2"
        id: 1
        shape { tuple_shapes { element_type: U64 } }
      }
    }
  }
  buffer_assignment {
    buffer_allocations {
      index: 0
      size: 1048576
      color: 0
      assigned { logical_buffer_id: 1 offset: 0 size: 524288 }
      assigned { logical_buffer_id: 2 offset: 524288 size: 524288 }
    }
    logical_buffers {
      id: 1
      size: 524288
      color: 0
      defined_at { instruction_id: 0 shape_index: 0 }
    }
    logical_buffers {
      id: 2
      size: 524288
      color: 0
      defined_at { instruction_id: 1 shape_index: 0 }
    }
    heap_simulator_traces { %s }
  }
)pb";

TEST(MemoryViewerTest, TestHeapSimulatorTraceShareWith_1) {
  // Allocate and then share, the memory usage is not doubled.
  static constexpr char kHeapSimulatorTrace[] = R"pb(
    events { kind: ALLOC buffer_id: 1 }
    events { kind: SHARE_WITH buffer_id: 2 share_with_canonical_id: 1 }
    events { kind: FREE buffer_id: 1 }
    events { kind: FREE buffer_id: 2 }
  )pb";
  std::string hlo_string = absl::StrFormat(kHLOBase, kHeapSimulatorTrace);
  xla::HloProto hlo_proto;
  ASSERT_TRUE(ParseTextFormatFromString(hlo_string, &hlo_proto).ok());
  TF_ASSERT_OK_AND_ASSIGN(
      PreprocessResult preprocess_result,
      ConvertHloProtoToPreprocessResult(hlo_proto, {.small_buffer_size = 0}));
  EXPECT_EQ(preprocess_result.peak_heap_mib(), 0.5);
  EXPECT_EQ(preprocess_result.total_buffer_allocation_mib(), 1);
}

TEST(MemoryViewerTest, TestHeapSimulatorTraceShareWith_2) {
  // Allocate, free and then share, the memory usage is not doubled.
  static constexpr char kHeapSimulatorTrace[] = R"pb(
    events { kind: ALLOC buffer_id: 1 }
    events { kind: FREE buffer_id: 1 }
    events { kind: SHARE_WITH buffer_id: 2 share_with_canonical_id: 1 }
    events { kind: FREE buffer_id: 2 }
  )pb";
  std::string hlo_string = absl::StrFormat(kHLOBase, kHeapSimulatorTrace);
  xla::HloProto hlo_proto;
  MemoryViewerOption option;
  option.small_buffer_size = 0;
  ASSERT_TRUE(ParseTextFormatFromString(hlo_string, &hlo_proto).ok());
  TF_ASSERT_OK_AND_ASSIGN(PreprocessResult preprocess_result,
                          ConvertHloProtoToPreprocessResult(hlo_proto, option));
  EXPECT_EQ(preprocess_result.peak_heap_mib(), 0.5);
  EXPECT_EQ(preprocess_result.total_buffer_allocation_mib(), 1);
  EXPECT_FALSE(preprocess_result.allocation_timeline().empty());
}

// 1 buffer allocation of 1.5MB
// 3 logical buffers, each is 0.5MB, forming a SHARE_WITH chain: C→B→A.
static constexpr char kHLOChain[] = R"pb(
  hlo_module {
    name: "test_module"
    entry_computation_name: "test_computation"
    computations {
      name: "test_computation"
      instructions {
        name: "fusion.1"
        id: 0
        shape { tuple_shapes { element_type: U64 } }
      }
      instructions {
        name: "fusion.2"
        id: 1
        shape { tuple_shapes { element_type: U64 } }
      }
      instructions {
        name: "fusion.3"
        id: 2
        shape { tuple_shapes { element_type: U64 } }
      }
    }
  }
  buffer_assignment {
    buffer_allocations {
      index: 0
      size: 1572864
      color: 0
      assigned { logical_buffer_id: 1 offset: 0 size: 524288 }
      assigned { logical_buffer_id: 2 offset: 0 size: 524288 }
      assigned { logical_buffer_id: 3 offset: 0 size: 524288 }
    }
    logical_buffers {
      id: 1
      size: 524288
      color: 0
      defined_at { instruction_id: 0 shape_index: 0 }
    }
    logical_buffers {
      id: 2
      size: 524288
      color: 0
      defined_at { instruction_id: 1 shape_index: 0 }
    }
    logical_buffers {
      id: 3
      size: 524288
      color: 0
      defined_at { instruction_id: 2 shape_index: 0 }
    }
    heap_simulator_traces { %s }
  }
)pb";

TEST(MemoryViewerTest, TestHeapSimulatorTraceShareWithChain) {
  // Regression test for canonical chain ID mismatch bug.
  //
  // Creates a SHARE_WITH chain: C(3) shares with B(2), B(2) shares with A(1).
  //   1. ALLOC A        → A.ref_count=1, logical_buffers=[A]
  //   2. SHARE_WITH B→A → A.ref_count=2
  //   3. FREE A         → A.ref_count=1
  //   4. FREE B         → A.ref_count=0, logical_buffers=[]
  //   5. SHARE_WITH C→B → chain C→B→A, A.ref_count=1
  //                        IncreaseMemoryUsage should use A (root), not B
  //   6. FREE C         → A.ref_count=0, logical_buffers=[]
  //
  // Before the fix, step 5 pushed B's ID to logical_buffers, but step 6
  // tried to remove A's ID (via get_canonical_buffer). B's ID leaked into
  // peak_logical_buffers, creating a phantom entry in the bar chart.
  static constexpr char kHeapSimulatorTrace[] = R"pb(
    events { kind: ALLOC buffer_id: 1 }
    events { kind: SHARE_WITH buffer_id: 2 share_with_canonical_id: 1 }
    events { kind: FREE buffer_id: 1 }
    events { kind: FREE buffer_id: 2 }
    events { kind: SHARE_WITH buffer_id: 3 share_with_canonical_id: 2 }
    events { kind: FREE buffer_id: 3 }
  )pb";
  std::string hlo_string = absl::StrFormat(kHLOChain, kHeapSimulatorTrace);
  xla::HloProto hlo_proto;
  MemoryViewerOption option;
  option.small_buffer_size = 0;
  ASSERT_TRUE(ParseTextFormatFromString(hlo_string, &hlo_proto).ok());
  TF_ASSERT_OK_AND_ASSIGN(PreprocessResult preprocess_result,
                          ConvertHloProtoToPreprocessResult(hlo_proto, option));
  // Peak should be 0.5 MiB (one buffer), not 1.0 MiB (leaked phantom).
  EXPECT_EQ(preprocess_result.peak_heap_mib(), 0.5);
  // max_heap should contain exactly 1 entry (the root canonical A).
  // Before the fix, it would contain 2 entries (A from ALLOC + leaked B).
  EXPECT_EQ(preprocess_result.max_heap_size(), 1);
}

TEST(MemoryViewerTest, TestHeapSimulatorTraceShareWithPeakInstruction) {
  // Regression test for display name and lifespan bugs on shared buffers.
  //
  // Creates a trace where B(2) shares with A(1), and C(3) is allocated at the
  // same time to push the peak heap usage.
  //   1. ALLOC A        → A.ref_count=1, heap=0.5MB, peak=0.5MB
  //   2. FREE A         → A.ref_count=0, heap=0MB
  //   3. SHARE_WITH B→A → A.ref_count=1, B.span.start=2, heap=0.5MB
  //   4. ALLOC C        → C.ref_count=1, C.span.start=3, heap=1.0MB (PEAK)
  //                       peak_logical_buffers=[A, C]
  //                       peak_canonical_to_display_id=[1->2]
  //   5. FREE B         → A.ref_count=0, B.span.limit=4, heap=0.5MB
  //   6. FREE C         → C.ref_count=0, C.span.limit=5, heap=0MB
  static constexpr char kHeapSimulatorTrace[] = R"pb(
    events { kind: ALLOC buffer_id: 1 }
    events { kind: FREE buffer_id: 1 }
    events { kind: SHARE_WITH buffer_id: 2 share_with_canonical_id: 1 }
    events { kind: ALLOC buffer_id: 3 }
    events { kind: FREE buffer_id: 2 }
    events { kind: FREE buffer_id: 3 }
  )pb";
  std::string hlo_string = absl::StrFormat(kHLOChain, kHeapSimulatorTrace);
  xla::HloProto hlo_proto;
  MemoryViewerOption option;
  option.small_buffer_size = 0;
  ASSERT_TRUE(ParseTextFormatFromString(hlo_string, &hlo_proto).ok());
  TF_ASSERT_OK_AND_ASSIGN(PreprocessResult preprocess_result,
                          ConvertHloProtoToPreprocessResult(hlo_proto, option));

  EXPECT_EQ(preprocess_result.peak_heap_mib(), 1.0);
  EXPECT_EQ(preprocess_result.total_buffer_allocation_mib(), 1.5);

  // The max_heap should show B(2) and C(3).
  // Before the fix, it would show A(1) instead of B(2).
  ASSERT_EQ(preprocess_result.max_heap_size(), 2);
  EXPECT_EQ(preprocess_result.max_heap(0).logical_buffer_id(), 2);
  EXPECT_EQ(preprocess_result.max_heap(0).instruction_name(), "fusion.2{0}");
  EXPECT_EQ(preprocess_result.max_heap(1).logical_buffer_id(), 3);
  EXPECT_EQ(preprocess_result.max_heap(1).instruction_name(), "fusion.3{0}");

  // Verify the active spans.
  // Before the fix, B(2) had no span because it was shared.
  const auto& spans = preprocess_result.logical_buffer_spans();
  ASSERT_TRUE(spans.contains(2));
  EXPECT_EQ(spans.at(2).start(), 2);
  EXPECT_EQ(spans.at(2).limit(), 4);

  ASSERT_TRUE(spans.contains(3));
  EXPECT_EQ(spans.at(3).start(), 3);
  EXPECT_EQ(spans.at(3).limit(), 5);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
