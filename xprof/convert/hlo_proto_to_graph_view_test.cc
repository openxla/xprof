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

#include "xprof/convert/hlo_proto_to_graph_view.h"

#include <string>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "google/protobuf/text_format.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xprof/convert/tool_options.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::HasSubstr;
using ::testing::StartsWith;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(GraphViewerParamsTest, GraphType) {
  // Default for graph type.
  ToolOptions options1;
  options1["type"] = "graph";
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params1,
                          ParseGraphViewerParams(options1));
  EXPECT_EQ(params1.type, "graph");
  EXPECT_EQ(params1.node_name, "");
  EXPECT_EQ(params1.graph_width, 3);
  EXPECT_EQ(params1.render_options.show_backend_config, false);
  EXPECT_EQ(params1.render_options.show_fusion_subcomputations, true);
  EXPECT_EQ(params1.format, xla::RenderedGraphFormat::kUrl);

  // User defined options for graph type.
  ToolOptions options2;
  options2["type"] = "graph";
  options2["node_name"] = "fusion.111";
  options2["graph_width"] = 10;
  options2["show_metadata"] = 1;
  options2["merge_fusion"] = 1;
  options2["format"] = "html";
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params2,
                          ParseGraphViewerParams(options2));
  EXPECT_EQ(params2.type, "graph");
  EXPECT_EQ(params2.node_name, "fusion.111");
  EXPECT_EQ(params2.graph_width, 10);
  EXPECT_EQ(params2.render_options.show_backend_config, true);
  EXPECT_EQ(params2.render_options.show_fusion_subcomputations, false);
  EXPECT_EQ(params2.format, xla::RenderedGraphFormat::kHtml);
}

TEST(GraphViewerParamsTest, ShortTxtType) {
  // Default for short txt type.
  ToolOptions options1;
  options1["type"] = "short_txt";
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params1,
                          ParseGraphViewerParams(options1));
  EXPECT_EQ(params1.type, "short_txt");
  EXPECT_EQ(params1.verbose, false);
  EXPECT_EQ(params1.show_metadata, false);

  // User defined options for short txt type.
  ToolOptions options2;
  options2["type"] = "short_txt";
  options2["show_metadata"] = 1;
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params2,
                          ParseGraphViewerParams(options2));
  EXPECT_EQ(params2.type, "short_txt");
  EXPECT_EQ(params2.verbose, false);
  EXPECT_EQ(params2.show_metadata, true);
}

TEST(GraphViewerParamsTest, LongTxtType) {
  // Default for long txt type.
  ToolOptions options1;
  options1["type"] = "long_txt";
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params1,
                          ParseGraphViewerParams(options1));
  EXPECT_EQ(params1.type, "long_txt");
  EXPECT_EQ(params1.verbose, true);
  EXPECT_EQ(params1.show_metadata, false);

  // User defined options for long txt type.
  ToolOptions options2;
  options2["type"] = "long_txt";
  options2["show_metadata"] = 1;
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params2,
                          ParseGraphViewerParams(options2));
  EXPECT_EQ(params2.type, "long_txt");
  EXPECT_EQ(params2.verbose, true);
  EXPECT_EQ(params2.show_metadata, true);
}

TEST(GraphViewerParamsTest, AdjNodesType) {
  ToolOptions options1;
  options1["type"] = "adj_nodes";
  options1["node_name"] = "fusion.111";
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params1,
                          ParseGraphViewerParams(options1));
  EXPECT_EQ(params1.type, "adj_nodes");
  EXPECT_EQ(params1.node_name, "fusion.111");
}

TEST(GraphViewerParamsTest, OtherTypes) {
  ToolOptions options1;
  EXPECT_THAT(ParseGraphViewerParams(options1),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("Graph viewer must provide a type option")));

  ToolOptions options2;
  options2["type"] = "abcd";
  EXPECT_THAT(ParseGraphViewerParams(options2),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("Unknown graph viewer type option: abcd")));
}

TEST(HloProtoToGraphViewTest, GetAdjacentNodesParameterInsideFusion) {
  xla::HloProto hlo_proto;
  const std::string hlo_text = R"pb(
    hlo_module {
      name: "test_fusion_module"
      entry_computation_name: "entry"
      entry_computation_id: 1
      id: 100
      computations {
        id: 2
        name: "fused_computation"
        instructions {
          id: 10
          name: "fused_param.0"
          opcode: "parameter"
          shape {
            element_type: F32
            layout { format: DENSE }
          }
          parameter_number: 0
        }
        instructions {
          id: 11
          name: "fused_negate.1"
          opcode: "negate"
          shape {
            element_type: F32
            layout { format: DENSE }
          }
          operand_ids: 10
        }
        program_shape {
          parameters {
            element_type: F32
            layout { format: DENSE }
          }
          result {
            element_type: F32
            layout { format: DENSE }
          }
          parameter_names: "fused_param.0"
        }
        root_id: 11
      }
      computations {
        id: 1
        name: "entry"
        instructions {
          id: 1
          name: "input_param.0"
          opcode: "parameter"
          shape {
            element_type: F32
            layout { format: DENSE }
          }
          parameter_number: 0
        }
        instructions {
          id: 2
          name: "fusion.1"
          opcode: "fusion"
          shape {
            element_type: F32
            layout { format: DENSE }
          }
          operand_ids: 1
          called_computation_ids: 2
          fusion_kind: "kCustom"
        }
        program_shape {
          parameters {
            element_type: F32
            layout { format: DENSE }
          }
          result {
            element_type: F32
            layout { format: DENSE }
          }
          parameter_names: "input_param.0"
        }
        root_id: 2
      }
      host_program_shape {
        parameters {
          element_type: F32
          layout { format: DENSE }
        }
        result {
          element_type: F32
          layout { format: DENSE }
        }
        parameter_names: "input_param.0"
      }
    }
  )pb";
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(hlo_text, &hlo_proto));

  TF_ASSERT_OK_AND_ASSIGN(std::string adj_json,
                          GetAdjacentNodes(hlo_proto, "fused_param.0"));
  EXPECT_THAT(adj_json, HasSubstr("\"operand_names\":[\"input_param.0\"]"));
  EXPECT_THAT(adj_json, HasSubstr("\"consumer_names\":[\"fused_negate.1\"]"));
}

TEST(HloProtoToGraphViewTest, ConvertHloProtoToMeGraphWithExtraNodes) {
  xla::HloProto hlo_proto;
  const std::string hlo_text = R"pb(
    hlo_module {
      name: "test_extra_nodes_module"
      entry_computation_name: "entry"
      entry_computation_id: 1
      id: 101
      computations {
        id: 1
        name: "entry"
        instructions {
          id: 1
          name: "param_a"
          opcode: "parameter"
          shape {
            element_type: F32
            layout { format: DENSE }
          }
          parameter_number: 0
        }
        instructions {
          id: 2
          name: "param_b"
          opcode: "parameter"
          shape {
            element_type: F32
            layout { format: DENSE }
          }
          parameter_number: 1
        }
        instructions {
          id: 3
          name: "extra_isolated"
          opcode: "parameter"
          shape {
            element_type: F32
            layout { format: DENSE }
          }
          parameter_number: 2
        }
        instructions {
          id: 4
          name: "add.1"
          opcode: "add"
          shape {
            element_type: F32
            layout { format: DENSE }
          }
          operand_ids: 1
          operand_ids: 2
        }
        program_shape {
          parameters {
            element_type: F32
            layout { format: DENSE }
          }
          parameters {
            element_type: F32
            layout { format: DENSE }
          }
          parameters {
            element_type: F32
            layout { format: DENSE }
          }
          result {
            element_type: F32
            layout { format: DENSE }
          }
          parameter_names: "param_a"
          parameter_names: "param_b"
          parameter_names: "extra_isolated"
        }
        root_id: 4
      }
      host_program_shape {
        parameters {
          element_type: F32
          layout { format: DENSE }
        }
        parameters {
          element_type: F32
          layout { format: DENSE }
        }
        parameters {
          element_type: F32
          layout { format: DENSE }
        }
        result {
          element_type: F32
          layout { format: DENSE }
        }
        parameter_names: "param_a"
        parameter_names: "param_b"
        parameter_names: "extra_isolated"
      }
    }
  )pb";
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(hlo_text, &hlo_proto));

  TF_ASSERT_OK_AND_ASSIGN(
      std::string graph_json,
      ConvertHloProtoToMeGraph(hlo_proto, "add.1", 1, {"extra_isolated"}));
  EXPECT_THAT(graph_json, HasSubstr("extra_isolated"));
  EXPECT_THAT(graph_json, HasSubstr("add.1"));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
