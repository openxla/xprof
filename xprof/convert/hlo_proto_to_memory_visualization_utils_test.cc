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

#include <cstdio>
#include <string>

#include "<gtest/gtest.h>"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "plugin/xprof/protobuf/memory_viewer_preprocess.pb.h"
#include "xprof/utils/tensorflow_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

// 2 buffer allocations of 1MB, one of which is indefinite (constant).
// 3 logical buffers
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
      instructions {
        name: "constant.1"
        id: 2
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
    buffer_allocations {
      index: 1
      size: 1048576
      color: 0
      is_constant: true
      assigned { logical_buffer_id: 3 offset: 0 size: 1048576 }
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
      size: 1048576
      color: 0
      defined_at { instruction_id: 2 shape_index: 0 }
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
  EXPECT_EQ(preprocess_result.peak_heap_mib(), 1.5);
  // [Peak unpadded heap] = [peak of unpadded heap-simulated buffer sizes] +
  // [padded size of indefinite buffers]. In this case, the computations are on
  // a single U64 (8 bytes), and we have 1MB of indefinite buffers.
  EXPECT_EQ(preprocess_result.peak_unpadded_heap_mib(), 8.0 / (1 << 20) + 1);
  EXPECT_EQ(preprocess_result.total_buffer_allocation_mib(), 2);
  EXPECT_EQ(preprocess_result.indefinite_buffer_allocation_mib(), 1);
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
  EXPECT_EQ(preprocess_result.peak_heap_mib(), 1.5);
  // [Peak unpadded heap] = [peak of unpadded heap-simulated buffer sizes] +
  // [padded size of indefinite buffers]. In this case, the computations are on
  // a single U64 (8 bytes), and we have 1MB of indefinite buffers.
  EXPECT_EQ(preprocess_result.peak_unpadded_heap_mib(), 8.0 / (1 << 20) + 1);
  EXPECT_EQ(preprocess_result.total_buffer_allocation_mib(), 2);
  EXPECT_EQ(preprocess_result.indefinite_buffer_allocation_mib(), 1);
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

// 1 buffer allocation of 0.75 MiB.
// 4 logical buffers: A(1), B(2), C(3) share the same 0.25 MiB at offset 0;
// D(4) is an independent 0.5 MiB buffer at offset 262144.
static constexpr char kHLOChainDisplayName[] = R"pb(
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
      instructions {
        name: "fusion.4"
        id: 3
        shape { tuple_shapes { element_type: U64 } }
      }
    }
  }
  buffer_assignment {
    buffer_allocations {
      index: 0
      size: 786432
      color: 1
      assigned { logical_buffer_id: 1 offset: 0 size: 262144 }
      assigned { logical_buffer_id: 2 offset: 0 size: 262144 }
      assigned { logical_buffer_id: 3 offset: 0 size: 262144 }
      assigned { logical_buffer_id: 4 offset: 262144 size: 524288 }
    }
    logical_buffers {
      id: 1
      size: 262144
      color: 1
      defined_at { instruction_id: 0 shape_index: 0 }
    }
    logical_buffers {
      id: 2
      size: 262144
      color: 1
      defined_at { instruction_id: 1 shape_index: 0 }
    }
    logical_buffers {
      id: 3
      size: 262144
      color: 1
      defined_at { instruction_id: 2 shape_index: 0 }
    }
    logical_buffers {
      id: 4
      size: 524288
      color: 1
      defined_at { instruction_id: 3 shape_index: 0 }
    }
    heap_simulator_traces { %s }
  }
)pb";

TEST(MemoryViewerTest, TestShareWithChainDisplayName) {
  // Regression test for the display name and lifespan fix (b/522817031).
  //
  // Creates a SHARE_WITH chain C(3)→B(2)→A(1) plus an independent buffer D(4).
  // D is allocated before C reuses A's physical memory, so the peak occurs at
  // the SHARE_WITH C→B step when both D and A's memory are live.
  //
  //   1. ALLOC A(1)          → heap=0.25 MiB
  //   2. SHARE_WITH B(2)→A   → A.ref_count=2
  //   3. FREE A(1)           → A.ref_count=1
  //   4. FREE B(2)           → A.ref_count=0, heap=0
  //   5. ALLOC D(4)          → heap=0.5 MiB
  //   6. SHARE_WITH C(3)→B   → chain C→B→A, heap=0.75 MiB  ← PEAK
  //                             canonical_to_display_id[A]=C
  //   7. FREE C(3)           → heap=0.5 MiB
  //   8. FREE D(4)           → heap=0
  //
  // At peak, the bar chart entry for A's physical memory should use C's
  // metadata (instruction "fusion.3"), not A's ("fusion.1").
  static constexpr char kHeapSimulatorTrace[] = R"pb(
    events { kind: ALLOC buffer_id: 1 }
    events { kind: SHARE_WITH buffer_id: 2 share_with_canonical_id: 1 }
    events { kind: FREE buffer_id: 1 }
    events { kind: FREE buffer_id: 2 }
    events { kind: ALLOC buffer_id: 4 }
    events { kind: SHARE_WITH buffer_id: 3 share_with_canonical_id: 2 }
    events { kind: FREE buffer_id: 3 }
    events { kind: FREE buffer_id: 4 }
  )pb";
  std::string hlo_string =
      absl::StrFormat(kHLOChainDisplayName, kHeapSimulatorTrace);
  xla::HloProto hlo_proto;
  MemoryViewerOption option;
  option.memory_color = 1;  // VMEM; this bug only manifests with the explicit
                            // prefetch scheduler's VMEM heap simulator.
  option.small_buffer_size = 0;
  ASSERT_TRUE(ParseTextFormatFromString(hlo_string, &hlo_proto).ok());
  TF_ASSERT_OK_AND_ASSIGN(PreprocessResult preprocess_result,
                          ConvertHloProtoToPreprocessResult(hlo_proto, option));

  // Peak is 0.75 MiB: D (0.5 MiB) + C-via-A (0.25 MiB).
  EXPECT_EQ(preprocess_result.peak_heap_mib(), 0.75);
  // Two entries at peak: D and the canonical A (displayed as C).
  EXPECT_EQ(preprocess_result.max_heap_size(), 2);

  // Verify display names: the bar chart should show "fusion.3" (C, the current
  // sharer) for the canonical buffer's entry, not "fusion.1" (A, the root).
  bool found_fusion_3 = false;
  bool found_fusion_4 = false;
  for (const auto& heap_object : preprocess_result.max_heap()) {
    if (absl::StrContains(heap_object.instruction_name(), "fusion.3")) {
      found_fusion_3 = true;
    }
    if (absl::StrContains(heap_object.instruction_name(), "fusion.4")) {
      found_fusion_4 = true;
    }
    // fusion.1 (A's name) should NOT appear — C is the active sharer at peak.
    EXPECT_FALSE(absl::StrContains(heap_object.instruction_name(), "fusion.1"))
        << "Bar chart shows root canonical name 'fusion.1' instead of sharer "
           "name 'fusion.3'. heap_object.instruction_name()="
        << heap_object.instruction_name();
  }
  EXPECT_TRUE(found_fusion_3)
      << "Expected sharer C's instruction 'fusion.3' in the bar chart";
  EXPECT_TRUE(found_fusion_4)
      << "Expected independent buffer D's instruction 'fusion.4' in the bar "
         "chart";
}

struct DoubleRectInfo {
  std::string tooltip;
  double pos_x = 0.0;
  double pos_y = 0.0;
  double width = 0.0;
  double height = 0.0;
  uint64_t offset = 0;
  uint64_t size = 0;
};

std::vector<DoubleRectInfo> ParseLogicalBuffersFromDot(absl::string_view dot) {
  std::vector<DoubleRectInfo> buffer_rects;
  size_t offset = 0;
  while (true) {
    size_t node_start = dot.find('"', offset);
    if (node_start == absl::string_view::npos) break;
    size_t node_end = dot.find('"', node_start + 1);
    if (node_end == absl::string_view::npos) break;
    offset = node_end + 1;

    size_t open_bracket = dot.find('[', offset);
    if (open_bracket == absl::string_view::npos || open_bracket - offset > 5) {
      continue;
    }

    size_t tooltip_start = dot.find("tooltip=\"", open_bracket);
    if (tooltip_start == absl::string_view::npos) continue;
    tooltip_start += 9;
    size_t tooltip_end = dot.find('"', tooltip_start);
    if (tooltip_end == absl::string_view::npos) continue;
    std::string tooltip =
        std::string(dot.substr(tooltip_start, tooltip_end - tooltip_start));

    // Only collect logical buffers
    if (!absl::StrContains(tooltip, "buffer_id:")) {
      continue;
    }

    size_t pos_start = dot.find("pos=\"", tooltip_end);
    if (pos_start == absl::string_view::npos) continue;
    pos_start += 5;
    size_t pos_end = dot.find('!', pos_start);
    if (pos_end == absl::string_view::npos) continue;
    std::string pos_str =
        std::string(dot.substr(pos_start, pos_end - pos_start));
    double pos_x = 0.0, pos_y = 0.0;
    std::sscanf(pos_str.c_str(), "%lf,%lf", &pos_x, &pos_y);

    size_t width_start = dot.find("width=\"", pos_end);
    if (width_start == absl::string_view::npos) continue;
    width_start += 7;
    size_t width_end = dot.find_first_of("!\"", width_start);
    if (width_end == absl::string_view::npos) continue;
    std::string width_str =
        std::string(dot.substr(width_start, width_end - width_start));
    double width = std::stod(width_str);

    size_t height_start = dot.find("height=\"", width_end);
    if (height_start == absl::string_view::npos) continue;
    height_start += 8;
    size_t height_end = dot.find_first_of("!\"", height_start);
    if (height_end == absl::string_view::npos) continue;
    std::string height_str =
        std::string(dot.substr(height_start, height_end - height_start));
    double height = std::stod(height_str);

    uint64_t buffer_offset = 0;
    size_t offset_pos = tooltip.find("\noffset:");
    if (offset_pos != std::string::npos) {
      buffer_offset = std::stoull(tooltip.substr(offset_pos + 8));
    }

    uint64_t buffer_size = 0;
    size_t size_pos = tooltip.find("\nsize:");
    if (size_pos != std::string::npos) {
      buffer_size = std::stoull(tooltip.substr(size_pos + 6));
    }

    buffer_rects.push_back(
        {tooltip, pos_x, pos_y, width, height, buffer_offset, buffer_size});
    offset = height_end + 1;
  }
  return buffer_rects;
}

double GetTopBoundary(const DoubleRectInfo& rect) {
  if (rect.pos_y == static_cast<double>(static_cast<int>(rect.pos_y)) &&
      rect.height == static_cast<double>(static_cast<int>(rect.height))) {
    return static_cast<double>(static_cast<int>(rect.pos_y) +
                               static_cast<int>(rect.height) / 2);
  }
  return rect.pos_y + rect.height / 2.0;
}

double GetBottomBoundary(const DoubleRectInfo& rect) {
  if (rect.pos_y == static_cast<double>(static_cast<int>(rect.pos_y)) &&
      rect.height == static_cast<double>(static_cast<int>(rect.height))) {
    return static_cast<double>(static_cast<int>(rect.pos_y) -
                               static_cast<int>(rect.height) / 2);
  }
  return rect.pos_y - rect.height / 2.0;
}

bool RectsOverlap(const DoubleRectInfo& a, const DoubleRectInfo& b) {
  double a_left = a.pos_x - a.width / 2.0;
  double a_right = a.pos_x + a.width / 2.0;
  double a_bottom = a.pos_y - a.height / 2.0;
  double a_top = a.pos_y + a.height / 2.0;

  double b_left = b.pos_x - b.width / 2.0;
  double b_right = b.pos_x + b.width / 2.0;
  double b_bottom = b.pos_y - b.height / 2.0;
  double b_top = b.pos_y + b.height / 2.0;

  double x_overlap =
      std::max(0.0, std::min(a_right, b_right) - std::max(a_left, b_left));
  double y_overlap =
      std::max(0.0, std::min(a_top, b_top) - std::max(a_bottom, b_bottom));

  // A tolerance of 0.05 points accounts for rounding errors in the DOT file
  // (which formats coordinates to 2 decimal places, e.g. %.2f).
  constexpr double kTolerance = 0.05;
  return (x_overlap > kTolerance && y_overlap > kTolerance);
}

TEST(MemoryViewerTest, TestLogicalBuffersDoNotOverlap) {
  auto verify_no_overlap = [](absl::string_view hlo_pb) {
    xla::HloProto hlo_proto;
    MemoryViewerOption option;
    option.small_buffer_size = 0;
    option.timeline_option.render_timeline = true;
    ASSERT_TRUE(
        ParseTextFormatFromString(std::string(hlo_pb), &hlo_proto).ok());
    TF_ASSERT_OK_AND_ASSIGN(
        PreprocessResult preprocess_result,
        ConvertHloProtoToPreprocessResult(hlo_proto, option));
    EXPECT_FALSE(preprocess_result.allocation_timeline().empty());

    std::vector<DoubleRectInfo> rects =
        ParseLogicalBuffersFromDot(preprocess_result.allocation_timeline());

    ASSERT_GT(rects.size(), 0);

    for (size_t i = 0; i < rects.size(); ++i) {
      for (size_t j = i + 1; j < rects.size(); ++j) {
        EXPECT_FALSE(RectsOverlap(rects[i], rects[j]))
            << "Overlap detected between:\n"
            << "Rect " << i << ": " << rects[i].tooltip
            << " (pos: " << rects[i].pos_x << "," << rects[i].pos_y
            << " size: " << rects[i].width << "x" << rects[i].height << ")\n"
            << "Rect " << j << ": " << rects[j].tooltip
            << " (pos: " << rects[j].pos_x << "," << rects[j].pos_y
            << " size: " << rects[j].width << "x" << rects[j].height << ")";
      }
    }

    // Verify that contiguous/adjacent logical buffers have a gap of exactly 0.
    std::vector<DoubleRectInfo> sorted_rects = rects;
    std::sort(sorted_rects.begin(), sorted_rects.end(),
              [](const DoubleRectInfo& a, const DoubleRectInfo& b) {
                return a.offset < b.offset;
              });
    for (size_t i = 0; i + 1 < sorted_rects.size(); ++i) {
      if (sorted_rects[i].offset + sorted_rects[i].size ==
          sorted_rects[i + 1].offset) {
        double top = GetTopBoundary(sorted_rects[i]);
        double bottom = GetBottomBoundary(sorted_rects[i + 1]);
        EXPECT_NEAR(bottom - top, 0.0, 0.05)
            << "Non-zero gap between adjacent buffers:\n"
            << "Prev: " << sorted_rects[i].tooltip
            << " (pos_y: " << sorted_rects[i].pos_y
            << ", height: " << sorted_rects[i].height << ", top: " << top
            << ")\n"
            << "Next: " << sorted_rects[i + 1].tooltip
            << " (pos_y: " << sorted_rects[i + 1].pos_y
            << ", height: " << sorted_rects[i + 1].height
            << ", bottom: " << bottom << ")";
      }
    }
  };

  // Test Case 1: Contiguous memory buffers (Single BA)
  static constexpr char kHLOSingleBA[] = R"pb(
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
      heap_simulator_traces {
        events { kind: ALLOC buffer_id: 1 }
        events { kind: ALLOC buffer_id: 2 }
        events { kind: FREE buffer_id: 1 }
        events { kind: FREE buffer_id: 2 }
      }
    }
  )pb";
  verify_no_overlap(kHLOSingleBA);

  // Test Case 2: Contiguous buffers with customer-scale offsets/sizes
  static constexpr char kHLOCustomerScale[] = R"pb(
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
        size: 54344548352
        color: 0
        assigned { logical_buffer_id: 1 offset: 54076112896 size: 134217728 }
        assigned { logical_buffer_id: 2 offset: 54210330624 size: 134217728 }
      }
      logical_buffers {
        id: 1
        size: 134217728
        color: 0
        defined_at { instruction_id: 0 shape_index: 0 }
      }
      logical_buffers {
        id: 2
        size: 134217728
        color: 0
        defined_at { instruction_id: 1 shape_index: 0 }
      }
      heap_simulator_traces {
        events { kind: ALLOC buffer_id: 1 }
        events { kind: ALLOC buffer_id: 2 }
        events { kind: FREE buffer_id: 1 }
        events { kind: FREE buffer_id: 2 }
      }
    }
  )pb";
  verify_no_overlap(kHLOCustomerScale);

  // Test Case 3: Chronological buffers sharing offsets, but active at different
  // times
  static constexpr char kHLOChronological[] = R"pb(
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
        assigned { logical_buffer_id: 2 offset: 0 size: 524288 }
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
      heap_simulator_traces {
        events { kind: ALLOC buffer_id: 1 }
        events { kind: FREE buffer_id: 1 }
        events { kind: ALLOC buffer_id: 2 }
        events { kind: FREE buffer_id: 2 }
      }
    }
  )pb";
  verify_no_overlap(kHLOChronological);
}

// VMEM HLO proto with scoped allocation in backend_config.
// 1 buffer allocation of 1MB in VMEM (color=1).
// 2 logical buffers, each 0.5MB.
// fusion.1 has a 1MB scoped VMEM allocation.
static constexpr char kHLOVmemWithScoped[] = R"pb(
  hlo_module {
    name: "test_module"
    entry_computation_name: "test_computation"
    computations {
      name: "test_computation"
      instructions {
        name: "fusion.1"
        id: 0
        backend_config: "{\"used_scoped_memory_configs\":[{\"memory_space\":\"1\",\"size\":\"1048576\"}]}"
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
      color: 1
      assigned { logical_buffer_id: 1 offset: 0 size: 524288 }
      assigned { logical_buffer_id: 2 offset: 524288 size: 524288 }
    }
    logical_buffers {
      id: 1
      size: 524288
      color: 1
      defined_at { instruction_id: 0 shape_index: 0 }
    }
    logical_buffers {
      id: 2
      size: 524288
      color: 1
      defined_at { instruction_id: 1 shape_index: 0 }
    }
    heap_simulator_traces {
      events { kind: ALLOC buffer_id: 1 }
      events { kind: ALLOC buffer_id: 2 }
      events { kind: FREE buffer_id: 1 }
      events { kind: FREE buffer_id: 2 }
    }
  }
)pb";

TEST(MemoryViewerTest, ScopedVmemAllocation_SingleInstruction) {
  xla::HloProto hlo_proto;
  ASSERT_TRUE(ParseTextFormatFromString(kHLOVmemWithScoped, &hlo_proto).ok());
  MemoryViewerOption option;
  option.small_buffer_size = 0;
  option.memory_color = 1;
  TF_ASSERT_OK_AND_ASSIGN(PreprocessResult result,
                       ConvertHloProtoToPreprocessResult(hlo_proto, option));
  EXPECT_EQ(result.max_scoped_vmem_allocation_mib(), 1.0);
  EXPECT_EQ(result.max_scoped_vmem_instruction_name(), "fusion.1");
}

TEST(MemoryViewerTest, ScopedVmemAllocation_MaxAcrossInstructions) {
  // Two instructions with different scoped VMEM sizes.
  // fusion.1 has 1MB, fusion.2 has 2MB. Should pick fusion.2.
  static constexpr char kHLOVmemTwoScoped[] = R"pb(
    hlo_module {
      name: "test_module"
      entry_computation_name: "test_computation"
      computations {
        name: "test_computation"
        instructions {
          name: "fusion.1"
          id: 0
          backend_config: "{\"used_scoped_memory_configs\":[{\"memory_space\":\"1\",\"size\":\"1048576\"}]}"
          shape { tuple_shapes { element_type: U64 } }
        }
        instructions {
          name: "fusion.2"
          id: 1
          backend_config: "{\"used_scoped_memory_configs\":[{\"memory_space\":\"1\",\"size\":\"2097152\"}]}"
          shape { tuple_shapes { element_type: U64 } }
        }
      }
    }
    buffer_assignment {
      buffer_allocations {
        index: 0
        size: 1048576
        color: 1
        assigned { logical_buffer_id: 1 offset: 0 size: 524288 }
        assigned { logical_buffer_id: 2 offset: 524288 size: 524288 }
      }
      logical_buffers {
        id: 1
        size: 524288
        color: 1
        defined_at { instruction_id: 0 shape_index: 0 }
      }
      logical_buffers {
        id: 2
        size: 524288
        color: 1
        defined_at { instruction_id: 1 shape_index: 0 }
      }
      heap_simulator_traces {
        events { kind: ALLOC buffer_id: 1 }
        events { kind: ALLOC buffer_id: 2 }
        events { kind: FREE buffer_id: 1 }
        events { kind: FREE buffer_id: 2 }
      }
    }
  )pb";
  xla::HloProto hlo_proto;
  ASSERT_TRUE(
      ParseTextFormatFromString(kHLOVmemTwoScoped, &hlo_proto).ok());
  MemoryViewerOption option;
  option.small_buffer_size = 0;
  option.memory_color = 1;
  TF_ASSERT_OK_AND_ASSIGN(PreprocessResult result,
                       ConvertHloProtoToPreprocessResult(hlo_proto, option));
  EXPECT_EQ(result.max_scoped_vmem_allocation_mib(), 2.0);
  EXPECT_EQ(result.max_scoped_vmem_instruction_name(), "fusion.2");
}

TEST(MemoryViewerTest, ScopedVmemAllocation_NoScopedConfigs) {
  // VMEM HLO with no used_scoped_memory_configs in backend_config.
  static constexpr char kHLOVmemNoScoped[] = R"pb(
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
        color: 1
        assigned { logical_buffer_id: 1 offset: 0 size: 524288 }
        assigned { logical_buffer_id: 2 offset: 524288 size: 524288 }
      }
      logical_buffers {
        id: 1
        size: 524288
        color: 1
        defined_at { instruction_id: 0 shape_index: 0 }
      }
      logical_buffers {
        id: 2
        size: 524288
        color: 1
        defined_at { instruction_id: 1 shape_index: 0 }
      }
      heap_simulator_traces {
        events { kind: ALLOC buffer_id: 1 }
        events { kind: ALLOC buffer_id: 2 }
        events { kind: FREE buffer_id: 1 }
        events { kind: FREE buffer_id: 2 }
      }
    }
  )pb";
  xla::HloProto hlo_proto;
  ASSERT_TRUE(
      ParseTextFormatFromString(kHLOVmemNoScoped, &hlo_proto).ok());
  MemoryViewerOption option;
  option.small_buffer_size = 0;
  option.memory_color = 1;
  TF_ASSERT_OK_AND_ASSIGN(PreprocessResult result,
                       ConvertHloProtoToPreprocessResult(hlo_proto, option));
  EXPECT_EQ(result.max_scoped_vmem_allocation_mib(), 0);
  EXPECT_TRUE(result.max_scoped_vmem_instruction_name().empty());
}

TEST(MemoryViewerTest, ScopedVmemAllocation_HBMIgnoresScoped) {
  // Even if instructions have scoped configs, HBM (color=0) should not report
  // scoped allocations because ComputeMaxScopedAllocationBytes is VMEM-only.
  static constexpr char kHeapSimulatorTrace[] = R"pb(
    events { kind: ALLOC buffer_id: 1 }
    events { kind: ALLOC buffer_id: 2 }
    events { kind: FREE buffer_id: 1 }
    events { kind: FREE buffer_id: 2 }
  )pb";
  std::string hlo_string = absl::StrFormat(kHLOBase, kHeapSimulatorTrace);
  xla::HloProto hlo_proto;
  ASSERT_TRUE(ParseTextFormatFromString(hlo_string, &hlo_proto).ok());
  // Inject a backend_config with scoped VMEM into the first instruction.
  hlo_proto.mutable_hlo_module()
      ->mutable_computations(0)
      ->mutable_instructions(0)
      ->set_backend_config(
          "{\"used_scoped_memory_configs\":[{\"memory_space\":\"1\","
          "\"size\":\"1048576\"}]}");
  MemoryViewerOption option;
  option.small_buffer_size = 0;
  option.memory_color = 0;  // HBM
  TF_ASSERT_OK_AND_ASSIGN(PreprocessResult result,
                       ConvertHloProtoToPreprocessResult(hlo_proto, option));
  EXPECT_EQ(result.max_scoped_vmem_allocation_mib(), 0);
  EXPECT_TRUE(result.max_scoped_vmem_instruction_name().empty());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
