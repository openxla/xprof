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

#include <iostream>
#include <string>

#include "absl/log/log.h"
#include "google/protobuf/text_format.h"
#include "xla/service/hlo.pb.h"
#include "xprof/convert/hlo_proto_to_graph_view.h"

int main(int argc, char* argv[]) {
  // Use a string representation of Mock HLO Proto.
  std::string hlo_proto_text = R"pb(
    hlo_module {
      name: "compare_reduce_fusion"
      id: 10
      entry_computation_name: "fused_computation"
      entry_computation_id: 1
      computations {
        name: "add_comp"
        id: 2
        root_id: 102
        instructions {
          name: "p0"
          opcode: "parameter"
          id: 100
          shape { element_type: PRED }
          parameter_number: 0
        }
        instructions {
          name: "p1"
          opcode: "parameter"
          id: 101
          shape { element_type: PRED }
          parameter_number: 1
        }
        instructions {
          name: "or"
          opcode: "or"
          id: 102
          shape { element_type: PRED }
          operand_ids: 100
          operand_ids: 101
        }
        program_shape {
          parameters { element_type: PRED }
          parameters { element_type: PRED }
          result { element_type: PRED }
          parameter_names: "p0"
          parameter_names: "p1"
        }
      }
      computations {
        name: "fused_computation"
        id: 1
        root_id: 9
        instructions {
          name: "param_0"
          opcode: "parameter"
          id: 0
          shape { element_type: S32 dimensions: 1 dimensions: 16384 }
          parameter_number: 0
        }
        instructions {
          name: "param_1"
          opcode: "parameter"
          id: 1
          shape { element_type: S32 }
          parameter_number: 1
        }
        instructions {
          name: "param_2"
          opcode: "parameter"
          id: 2
          shape { element_type: PRED dimensions: 1 dimensions: 16384 }
          parameter_number: 2
        }
        instructions {
          name: "dynamic-slice.1"
          opcode: "dynamic-slice"
          id: 3
          shape { element_type: S32 dimensions: 1 dimensions: 4096 }
          operand_ids: 0
          operand_ids: 1
          operand_ids: 1
        }
        instructions {
          name: "dynamic-slice.2"
          opcode: "dynamic-slice"
          id: 4
          shape { element_type: PRED dimensions: 1 dimensions: 4096 }
          operand_ids: 2
          operand_ids: 1
          operand_ids: 1
        }
        instructions {
          name: "add"
          opcode: "add"
          id: 5
          shape { element_type: S32 dimensions: 1 dimensions: 4096 }
          operand_ids: 3
          operand_ids: 3
        }
        instructions {
          name: "select"
          opcode: "select"
          id: 6
          shape { element_type: S32 dimensions: 1 dimensions: 4096 }
          operand_ids: 4
          operand_ids: 5
          operand_ids: 3
        }
        instructions {
          name: "compare"
          opcode: "compare"
          id: 7
          shape { element_type: PRED dimensions: 1 dimensions: 4096 }
          operand_ids: 6
          operand_ids: 6
          comparison_direction: "NE"
        }
        instructions {
          name: "reduce"
          opcode: "reduce"
          id: 8
          shape { element_type: PRED dimensions: 4096 }
          operand_ids: 7
          operand_ids: 7
          called_computation_ids: 2
        }
        instructions {
          name: "tuple"
          opcode: "tuple"
          id: 9
          shape {
            element_type: TUPLE
            tuple_shapes { element_type: PRED dimensions: 4096 }
            tuple_shapes { element_type: S32 dimensions: 1 dimensions: 4096 }
            tuple_shapes { element_type: PRED dimensions: 1 dimensions: 4096 }
          }
          operand_ids: 8
          operand_ids: 6
          operand_ids: 4
        }
      }
      host_program_shape {
        parameters { element_type: S32 dimensions: 1 dimensions: 16384 }
        parameters { element_type: S32 }
        parameters { element_type: PRED dimensions: 1 dimensions: 16384 }
        result {
          element_type: TUPLE
          tuple_shapes { element_type: PRED dimensions: 4096 }
          tuple_shapes { element_type: S32 dimensions: 1 dimensions: 4096 }
          tuple_shapes { element_type: PRED dimensions: 1 dimensions: 4096 }
        }
        parameter_names: "param_0"
        parameter_names: "param_1"
        parameter_names: "param_2"
      }
    }
  )pb";

  xla::HloProto hlo_proto;
  if (!google::protobuf::TextFormat::ParseFromString(hlo_proto_text, &hlo_proto)) {
    LOG(ERROR) << "Failed to parse hlo proto text";
    return 1;
  }

  // Convert the HloProto into graph HTML.
  absl::StatusOr<std::string> graph_html_or =
      tensorflow::profiler::ConvertHloProtoToGraph(
          hlo_proto, "tuple", 10, xla::HloRenderOptions(),
          xla::RenderedGraphFormat::kHtml);

  if (!graph_html_or.ok()) {
    LOG(ERROR) << "Failed to generate graph HTML: " << graph_html_or.status();
    return 1;
  }

  std::cout << *graph_html_or;
  return 0;
}
