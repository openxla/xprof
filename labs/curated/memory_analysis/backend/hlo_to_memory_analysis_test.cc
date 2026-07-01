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

#include "xprof/labs/curated/memory_analysis/backend/hlo_to_memory_analysis.h"

#include <cstdint>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"  // from @com_github_nlohmann_json
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/utils/tensorflow_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::status::IsOk;
using ::testing::UnorderedElementsAre;

// Simple HLO featuring an All-Gather node.
constexpr absl::string_view kAllGatherHlo = R"pb(
  hlo_module {
    name: "test_module"
    entry_computation_name: "test_computation"
    entry_computation_id: 1
    host_program_shape {
      parameters { element_type: F32 dimensions: 128 dimensions: 128 }
      parameters { element_type: F32 dimensions: 256 dimensions: 128 }
      result { element_type: F32 dimensions: 256 dimensions: 128 }
      parameter_names: "weight_param"
      parameter_names: "activation_node"
    }
    computations {
      name: "test_computation"
      id: 1
      root_id: 4
      instructions {
        name: "weight_param"
        id: 1
        opcode: "parameter"
        shape { element_type: F32 dimensions: 128 dimensions: 128 }
        parameter_number: 0
      }
      instructions {
        name: "gather_node"
        id: 2
        opcode: "all-gather"
        shape { element_type: F32 dimensions: 256 dimensions: 128 }
        operand_ids: 1
        dimensions: 0
      }
      instructions {
        name: "activation_node"
        id: 3
        opcode: "parameter"
        shape { element_type: F32 dimensions: 256 dimensions: 128 }
        parameter_number: 1
      }
      instructions {
        name: "multiply_node"
        id: 4
        opcode: "multiply"
        shape { element_type: F32 dimensions: 256 dimensions: 128 }
        operand_ids: 2
        operand_ids: 3
      }
    }
  }
  buffer_assignment {
    buffer_allocations {
      index: 0
      size: 327680
      color: 0
      assigned { logical_buffer_id: 1 offset: 0 size: 65536 }
      assigned { logical_buffer_id: 2 offset: 65536 size: 131072 }
      assigned { logical_buffer_id: 3 offset: 196608 size: 131072 }
    }
    logical_buffers {
      id: 1
      size: 65536
      color: 0
      defined_at { instruction_id: 1 shape_index: 0 }
    }
    logical_buffers {
      id: 2
      size: 131072
      color: 0
      defined_at { instruction_id: 2 shape_index: 0 }
    }
    logical_buffers {
      id: 3
      size: 131072
      color: 0
      defined_at { instruction_id: 3 shape_index: 0 }
    }
    heap_simulator_traces {
      events { kind: ALLOC buffer_id: 1 }
      events { kind: ALLOC buffer_id: 2 }
      events { kind: ALLOC buffer_id: 3 }
      events { kind: FREE buffer_id: 1 }
      events { kind: FREE buffer_id: 2 }
      events { kind: FREE buffer_id: 3 }
      buffer_allocation_index: 0
    }
  }
)pb";

TEST(HloToMemoryAnalysisTest, TestTraceGatheredWeights) {
  xla::HloProto hlo_proto;
  ASSERT_THAT(ParseTextFormatFromString(std::string(kAllGatherHlo), &hlo_proto),
              IsOk());

  TF_ASSERT_OK_AND_ASSIGN(absl::flat_hash_set<std::string> gathered_weights,
                          TraceGatheredWeights(hlo_proto));

  // 'gather_node' should be traced as gathered weights, along with derived
  // nodes like 'multiply_node'. The input 'weight_param' is
  // pre-gathered/sharded, so it is not a gathered weight. 'activation_node' is
  // also not a gathered weight.
  EXPECT_THAT(gathered_weights,
              UnorderedElementsAre("gather_node", "multiply_node"));
}

TEST(HloToMemoryAnalysisTest, TestConvertHloProtoToMemoryAnalysisJson) {
  xla::HloProto hlo_proto;
  ASSERT_THAT(ParseTextFormatFromString(std::string(kAllGatherHlo), &hlo_proto),
              IsOk());

  TF_ASSERT_OK_AND_ASSIGN(
      std::string json_str,
      ConvertHloProtoToMemoryAnalysisJson(hlo_proto, "test_module"));

  nlohmann::json analysis_json = nlohmann::json::parse(json_str);

  EXPECT_EQ(analysis_json["moduleName"], "test_module");
  EXPECT_TRUE(analysis_json.contains("summary"));
  EXPECT_TRUE(analysis_json.contains("peakHeapBuffers"));

  bool found_gather_node = false;
  for (const nlohmann::json& buf : analysis_json["peakHeapBuffers"]) {
    std::string label = buf.value("label", "");
    if (absl::StrContains(label, "gather_node")) {
      found_gather_node = true;
      EXPECT_EQ(buf["category"], "Weight");
      EXPECT_EQ(buf["subCategory"], "Gathered");
    }
  }
  EXPECT_TRUE(found_gather_node);
}

// Simple HLO featuring multi-color (multi-space) buffer allocations.
constexpr absl::string_view kMultiColorHlo = R"pb(
  hlo_module {
    name: "test_module"
    entry_computation_name: "test_computation"
    entry_computation_id: 1
    host_program_shape {
      parameters { element_type: F32 dimensions: 128 dimensions: 128 }
      parameters { element_type: F32 dimensions: 256 dimensions: 128 }
      result { element_type: F32 dimensions: 256 dimensions: 128 }
      parameter_names: "weight_param"
      parameter_names: "activation_node"
    }
    computations {
      name: "test_computation"
      id: 1
      root_id: 4
      instructions {
        name: "weight_param"
        id: 1
        opcode: "parameter"
        shape { element_type: F32 dimensions: 128 dimensions: 128 }
        parameter_number: 0
      }
      instructions {
        name: "gather_node"
        id: 2
        opcode: "all-gather"
        shape { element_type: F32 dimensions: 256 dimensions: 128 }
        operand_ids: 1
        dimensions: 0
      }
      instructions {
        name: "activation_node"
        id: 3
        opcode: "parameter"
        shape { element_type: F32 dimensions: 256 dimensions: 128 }
        parameter_number: 1
      }
      instructions {
        name: "multiply_node"
        id: 4
        opcode: "multiply"
        shape { element_type: F32 dimensions: 256 dimensions: 128 }
        operand_ids: 2
        operand_ids: 3
      }
    }
  }
  buffer_assignment {
    buffer_allocations {
      index: 0
      size: 196608
      color: 0
      assigned { logical_buffer_id: 1 offset: 0 size: 65536 }
      assigned { logical_buffer_id: 2 offset: 65536 size: 131072 }
    }
    buffer_allocations {
      index: 1
      size: 131072
      color: 1
      assigned { logical_buffer_id: 3 offset: 0 size: 131072 }
    }
    logical_buffers {
      id: 1
      size: 65536
      color: 0
      defined_at { instruction_id: 1 shape_index: 0 }
    }
    logical_buffers {
      id: 2
      size: 131072
      color: 0
      defined_at { instruction_id: 2 shape_index: 0 }
    }
    logical_buffers {
      id: 3
      size: 131072
      color: 1
      defined_at { instruction_id: 3 shape_index: 0 }
    }
    heap_simulator_traces {
      events { kind: ALLOC buffer_id: 1 }
      events { kind: ALLOC buffer_id: 2 }
      events { kind: FREE buffer_id: 1 }
      events { kind: FREE buffer_id: 2 }
      buffer_allocation_index: 0
    }
    heap_simulator_traces {
      events { kind: ALLOC buffer_id: 3 }
      events { kind: FREE buffer_id: 3 }
      buffer_allocation_index: 1
    }
  }
)pb";

TEST(HloToMemoryAnalysisTest,
     TestConvertMultiColorHloProtoToMemoryAnalysisJson) {
  xla::HloProto hlo_proto;
  ASSERT_THAT(
      ParseTextFormatFromString(std::string(kMultiColorHlo), &hlo_proto),
      IsOk());

  TF_ASSERT_OK_AND_ASSIGN(
      std::string json_str,
      ConvertHloProtoToMemoryAnalysisJson(hlo_proto, "test_module"));

  nlohmann::json analysis_json = nlohmann::json::parse(json_str);

  EXPECT_EQ(analysis_json["moduleName"], "test_module");
  EXPECT_TRUE(analysis_json.contains("summary"));
  EXPECT_TRUE(analysis_json.contains("memorySpaceBreakdown"));
  EXPECT_TRUE(analysis_json.contains("peakHeapBuffers"));

  // Check memorySpaceBreakdown values
  nlohmann::json breakdown = analysis_json["memorySpaceBreakdown"];
  EXPECT_TRUE(breakdown.contains("HBM"));
  EXPECT_TRUE(breakdown.contains("VMem"));
  EXPECT_GE(breakdown["HBM"].get<int64_t>(), 0);
  EXPECT_GE(breakdown["VMem"].get<int64_t>(), 0);

  // Verify individual buffers are tagged with correct memorySpace
  int found_hbm = 0;
  int found_vmem = 0;
  for (const nlohmann::json& buf : analysis_json["peakHeapBuffers"]) {
    std::string label = buf.value("label", "");
    std::string memory_space = buf.value("memorySpace", "");
    if (absl::StrContains(label, "weight_param") ||
        absl::StrContains(label, "gather_node")) {
      EXPECT_EQ(memory_space, "HBM");
      ++found_hbm;
    } else if (absl::StrContains(label, "activation_node")) {
      EXPECT_EQ(memory_space, "VMem");
      ++found_vmem;
    }
  }
  EXPECT_EQ(found_hbm, 2);
  EXPECT_EQ(found_vmem, 1);
}

// Complex HLO featuring multiple All-Gather nodes and various embedding ops to
// test DAG memoization and category precedence.
constexpr absl::string_view kComplexMemoAndCategorizationHlo = R"pb(
  hlo_module {
    name: "complex_test_module"
    entry_computation_name: "test_computation"
    entry_computation_id: 1
    host_program_shape {
      parameters { element_type: F32 dimensions: 128 dimensions: 128 }
      parameters {
        element_type: F32
        dimensions: 128
        dimensions: 128
        layout {
          minor_to_major: 1
          minor_to_major: 0
          tail_padding_alignment_in_elements: 1024
        }
      }
      parameters { element_type: F32 dimensions: 128 dimensions: 128 }
      parameters { element_type: F32 dimensions: 128 dimensions: 128 }
      parameters { element_type: F32 dimensions: 128 dimensions: 128 }
      result {
        element_type: TUPLE
        tuple_shapes { element_type: F32 dimensions: 256 dimensions: 128 }
        tuple_shapes { element_type: F32 dimensions: 256 dimensions: 128 }
        tuple_shapes { element_type: F32 dimensions: 128 dimensions: 128 }
        tuple_shapes { element_type: F32 dimensions: 128 dimensions: 128 }
        tuple_shapes {
          element_type: F32
          dimensions: 128
          dimensions: 128
          layout {
            minor_to_major: 1
            minor_to_major: 0
            tail_padding_alignment_in_elements: 1024
          }
        }
        tuple_shapes { element_type: F32 dimensions: 128 dimensions: 128 }
        tuple_shapes { element_type: F32 dimensions: 128 dimensions: 128 }
        tuple_shapes { element_type: F32 dimensions: 128 dimensions: 128 }
      }
      parameter_names: "param_base"
      parameter_names: "param_sparsecore"
      parameter_names: "param_act"
      parameter_names: "param_grad"
      parameter_names: "param_emb"
    }
    computations {
      name: "test_computation"
      id: 1
      root_id: 12
      instructions {
        name: "param_base"
        id: 1
        opcode: "parameter"
        shape { element_type: F32 dimensions: 128 dimensions: 128 }
        parameter_number: 0
      }
      instructions {
        name: "reshape1"
        id: 2
        opcode: "reshape"
        shape { element_type: F32 dimensions: 128 dimensions: 128 }
        operand_ids: 1
      }
      instructions {
        name: "gather1"
        id: 3
        opcode: "all-gather"
        shape { element_type: F32 dimensions: 256 dimensions: 128 }
        operand_ids: 2
        dimensions: 0
      }
      instructions {
        name: "gather2"
        id: 4
        opcode: "all-gather"
        shape { element_type: F32 dimensions: 256 dimensions: 128 }
        operand_ids: 2
        dimensions: 0
      }
      instructions {
        name: "slice1"
        id: 5
        opcode: "slice"
        shape { element_type: F32 dimensions: 64 dimensions: 128 }
        operand_ids: 1
        slice_dimensions { start: 0 limit: 64 stride: 1 }
        slice_dimensions { start: 0 limit: 128 stride: 1 }
      }
      instructions {
        name: "gather3"
        id: 6
        opcode: "all-gather"
        shape { element_type: F32 dimensions: 128 dimensions: 128 }
        operand_ids: 5
        dimensions: 0
      }
      instructions {
        name: "gather4"
        id: 7
        opcode: "all-gather"
        shape { element_type: F32 dimensions: 128 dimensions: 128 }
        operand_ids: 5
        dimensions: 0
      }
      instructions {
        name: "param_sparsecore"
        id: 8
        opcode: "parameter"
        shape {
          element_type: F32
          dimensions: 128
          dimensions: 128
          layout {
            minor_to_major: 1
            minor_to_major: 0
            tail_padding_alignment_in_elements: 1024
          }
        }
        parameter_number: 1
        metadata { op_name: "func/embedding_activations" }
      }
      instructions {
        name: "param_act"
        id: 9
        opcode: "parameter"
        shape { element_type: F32 dimensions: 128 dimensions: 128 }
        parameter_number: 2
        metadata { op_name: "func/embedding_activations" }
      }
      instructions {
        name: "param_grad"
        id: 10
        opcode: "parameter"
        shape { element_type: F32 dimensions: 128 dimensions: 128 }
        parameter_number: 3
        metadata { op_name: "func/embedding_gradients" }
      }
      instructions {
        name: "param_emb"
        id: 11
        opcode: "parameter"
        shape { element_type: F32 dimensions: 128 dimensions: 128 }
        parameter_number: 4
        metadata { op_name: "func/embedding_table" }
      }
      instructions {
        name: "root_tuple"
        id: 12
        opcode: "tuple"
        shape {
          element_type: TUPLE
          tuple_shapes { element_type: F32 dimensions: 256 dimensions: 128 }
          tuple_shapes { element_type: F32 dimensions: 256 dimensions: 128 }
          tuple_shapes { element_type: F32 dimensions: 128 dimensions: 128 }
          tuple_shapes { element_type: F32 dimensions: 128 dimensions: 128 }
          tuple_shapes {
            element_type: F32
            dimensions: 128
            dimensions: 128
            layout {
              minor_to_major: 1
              minor_to_major: 0
              tail_padding_alignment_in_elements: 1024
            }
          }
          tuple_shapes { element_type: F32 dimensions: 128 dimensions: 128 }
          tuple_shapes { element_type: F32 dimensions: 128 dimensions: 128 }
          tuple_shapes { element_type: F32 dimensions: 128 dimensions: 128 }
        }
        operand_ids: 3
        operand_ids: 4
        operand_ids: 6
        operand_ids: 7
        operand_ids: 8
        operand_ids: 9
        operand_ids: 10
        operand_ids: 11
      }
    }
  }
  buffer_assignment {
    buffer_allocations {
      index: 0
      size: 1048576
      color: 0
      is_entry_computation_parameter: true
      assigned { logical_buffer_id: 8 offset: 0 size: 65536 }
    }
    buffer_allocations {
      index: 1
      size: 1048576
      color: 1
      assigned { logical_buffer_id: 9 offset: 0 size: 65536 }
    }
    buffer_allocations {
      index: 2
      size: 1048576
      color: 2
      assigned { logical_buffer_id: 10 offset: 0 size: 65536 }
    }
    buffer_allocations {
      index: 3
      size: 1048576
      color: 3
      assigned { logical_buffer_id: 11 offset: 0 size: 65536 }
    }
    logical_buffers {
      id: 8
      size: 65536
      color: 0
      defined_at { instruction_id: 8 shape_index: 0 }
    }
    logical_buffers {
      id: 9
      size: 65536
      color: 1
      defined_at { instruction_id: 9 shape_index: 0 }
    }
    logical_buffers {
      id: 10
      size: 65536
      color: 2
      defined_at { instruction_id: 10 shape_index: 0 }
    }
    logical_buffers {
      id: 11
      size: 65536
      color: 3
      defined_at { instruction_id: 11 shape_index: 0 }
    }
    heap_simulator_traces {
      events { kind: ALLOC buffer_id: 8 }
      events { kind: FREE buffer_id: 8 }
      buffer_allocation_index: 0
    }
    heap_simulator_traces {
      events { kind: ALLOC buffer_id: 9 }
      events { kind: FREE buffer_id: 9 }
      buffer_allocation_index: 1
    }
    heap_simulator_traces {
      events { kind: ALLOC buffer_id: 10 }
      events { kind: FREE buffer_id: 10 }
      buffer_allocation_index: 2
    }
    heap_simulator_traces {
      events { kind: ALLOC buffer_id: 11 }
      events { kind: FREE buffer_id: 11 }
      buffer_allocation_index: 3
    }
  }
)pb";

TEST(HloToMemoryAnalysisTest, TestDAGMemoizationAndCategorization) {
  // Arrange
  xla::HloProto hlo_proto;
  ASSERT_THAT(ParseTextFormatFromString(
                  std::string(kComplexMemoAndCategorizationHlo), &hlo_proto),
              IsOk());

  // Act
  TF_ASSERT_OK_AND_ASSIGN(
      std::string json_str,
      ConvertHloProtoToMemoryAnalysisJson(hlo_proto, "complex_test_module"));
  nlohmann::json analysis_json = nlohmann::json::parse(json_str);

  // Assert
  EXPECT_EQ(analysis_json["moduleName"], "complex_test_module");
  EXPECT_TRUE(analysis_json.contains("summary"));
  EXPECT_TRUE(analysis_json.contains("peakHeapBuffers"));

  bool found_sparsecore_param = false;
  bool found_act_param = false;
  bool found_grad_param = false;
  bool found_emb_param = false;

  for (const nlohmann::json& buf : analysis_json["peakHeapBuffers"]) {
    std::string label = buf.value("label", "");
    std::string category = buf.value("category", "");
    std::string sub_category = buf.value("subCategory", "");

    if (absl::StrContains(label, "param_sparsecore")) {
      found_sparsecore_param = true;
      EXPECT_EQ(category, "Input");
      EXPECT_EQ(sub_category, "SparseCore");
    } else if (absl::StrContains(label, "param_act")) {
      found_act_param = true;
      EXPECT_EQ(category, "SparseCore");
      EXPECT_EQ(sub_category, "Activation");
    } else if (absl::StrContains(label, "param_grad")) {
      found_grad_param = true;
      EXPECT_EQ(category, "SparseCore");
      EXPECT_EQ(sub_category, "Gradient");
    } else if (absl::StrContains(label, "param_emb")) {
      found_emb_param = true;
      EXPECT_EQ(category, "SparseCore");
      EXPECT_EQ(sub_category, "Other");
    }
  }

  EXPECT_TRUE(found_sparsecore_param);
  EXPECT_TRUE(found_act_param);
  EXPECT_TRUE(found_grad_param);
  EXPECT_TRUE(found_emb_param);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
