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

#include <set>
#include <string>

#include "<gtest/gtest.h>"
#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"  // from @com_github_nlohmann_json
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/utils/tensorflow_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

// Simple HLO featuring an All-Gather node.
static constexpr char kAllGatherHlo[] = R"pb(
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
  ASSERT_TRUE(ParseTextFormatFromString(kAllGatherHlo, &hlo_proto).ok());

  TF_ASSERT_OK_AND_ASSIGN(std::set<std::string> gathered_weights,
                          TraceGatheredWeights(hlo_proto));

  // 'gather_node' should be traced as gathered weights.
  // The input 'weight_param' is pre-gathered/sharded, so it is not a gathered
  // weight. 'activation_node' and 'multiply_node' should not.
  EXPECT_TRUE(gathered_weights.find("gather_node") != gathered_weights.end());
  EXPECT_TRUE(gathered_weights.find("weight_param") == gathered_weights.end());
  EXPECT_TRUE(gathered_weights.find("activation_node") ==
              gathered_weights.end());
}

TEST(HloToMemoryAnalysisTest, TestConvertHloProtoToMemoryAnalysisJson) {
  xla::HloProto hlo_proto;
  ASSERT_TRUE(ParseTextFormatFromString(kAllGatherHlo, &hlo_proto).ok());

  TF_ASSERT_OK_AND_ASSIGN(
      std::string json_str,
      ConvertHloProtoToMemoryAnalysisJson(hlo_proto, "test_module"));

  nlohmann::json analysis_json = nlohmann::json::parse(json_str);

  EXPECT_EQ(analysis_json["moduleName"], "test_module");
  EXPECT_TRUE(analysis_json.contains("summary"));
  EXPECT_TRUE(analysis_json.contains("peakHeapBuffers"));

  // Traced weights should end up classified under "Weight"
  // gather_node (131072 bytes) and weight_param (65536 bytes)
  // Wait, the heap simulator trace allocates:
  // buffer 1 (defined_at: weight_param): size 65536, group is set during
  // Preprocess. Let's verify how the buffers are annotated inside
  // peakHeapBuffers.
  bool found_gather_node = false;
  for (const auto& buf : analysis_json["peakHeapBuffers"]) {
    std::string label = buf.value("label", "");
    if (label.find("gather_node") != std::string::npos) {
      found_gather_node = true;
      EXPECT_EQ(buf["category"], "Weight");
      EXPECT_EQ(buf["subCategory"], "Gathered");
    }
  }
  EXPECT_TRUE(found_gather_node);
}

// Simple HLO featuring multi-color (multi-space) buffer allocations.
static constexpr char kMultiColorHlo[] = R"pb(
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
  ASSERT_TRUE(ParseTextFormatFromString(kMultiColorHlo, &hlo_proto).ok());

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
  for (const auto& buf : analysis_json["peakHeapBuffers"]) {
    std::string label = buf.value("label", "");
    std::string memory_space = buf.value("memorySpace", "");
    if (label.find("weight_param") != std::string::npos ||
        label.find("gather_node") != std::string::npos) {
      EXPECT_EQ(memory_space, "HBM");
      found_hbm++;
    } else if (label.find("activation_node") != std::string::npos) {
      EXPECT_EQ(memory_space, "VMem");
      found_vmem++;
    }
  }
  EXPECT_EQ(found_hbm, 2);
  EXPECT_EQ(found_vmem, 1);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
