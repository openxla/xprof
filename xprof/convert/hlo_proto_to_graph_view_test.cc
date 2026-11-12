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
#include "nlohmann/json_fwd.hpp"  // from @com_github_nlohmann_json
#include "nlohmann/json.hpp"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xla/xla_data.pb.h"
#include "xprof/convert/tool_options.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::StartsWith;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;
using json = nlohmann::json;

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

TEST(HloProtoToGraphViewTest, GetAdjacentNodesMetadata) {
  xla::HloProto hlo_proto;
  xla::HloModuleProto* module = hlo_proto.mutable_hlo_module();
  module->set_name("test_module");
  module->set_id(1);
  module->set_entry_computation_name("entry");
  module->set_entry_computation_id(10);

  auto set_f32_2x4_shape = [](xla::ShapeProto* shape) {
    shape->set_element_type(xla::F32);
    shape->add_dimensions(2);
    shape->add_dimensions(4);
    shape->mutable_layout()->add_minor_to_major(1);
    shape->mutable_layout()->add_minor_to_major(0);
  };

  xla::HloComputationProto* comp = module->add_computations();
  comp->set_name("entry");
  comp->set_id(10);
  comp->set_root_id(2);

  xla::HloInstructionProto* param = comp->add_instructions();
  param->set_name("arg0.1");
  param->set_opcode("parameter");
  param->set_parameter_number(0);
  param->set_id(0);
  set_f32_2x4_shape(param->mutable_shape());

  xla::HloInstructionProto* negate = comp->add_instructions();
  negate->set_name("negate.2");
  negate->set_opcode("negate");
  negate->set_id(1);
  negate->add_operand_ids(0);
  set_f32_2x4_shape(negate->mutable_shape());

  xla::HloInstructionProto* add = comp->add_instructions();
  add->set_name("add.3");
  add->set_opcode("add");
  add->set_id(2);
  add->add_operand_ids(0);
  add->add_operand_ids(1);
  set_f32_2x4_shape(add->mutable_shape());

  xla::ProgramShapeProto* program_shape = module->mutable_host_program_shape();
  set_f32_2x4_shape(program_shape->add_parameters());
  program_shape->add_parameter_names("arg0");
  set_f32_2x4_shape(program_shape->mutable_result());

  TF_ASSERT_OK_AND_ASSIGN(std::string response,
                          GetAdjacentNodes(hlo_proto, "negate.2"));
  json j = json::parse(response);
  EXPECT_THAT(j["operand_names"], Eq(json::array({"arg0.1"})));
  EXPECT_THAT(j["consumer_names"], Eq(json::array({"add.3"})));
  EXPECT_THAT(j["center_node"], Eq(json{{"name", "negate.2"},
                                        {"opcode", "negate"},
                                        {"shape", "f32[2,4]"}}));
  EXPECT_THAT(j["operands"], Eq(json::array({{{"name", "arg0.1"},
                                              {"opcode", "parameter"},
                                              {"shape", "f32[2,4]"}}})));
  EXPECT_THAT(
      j["consumers"],
      Eq(json::array(
          {{{"name", "add.3"}, {"opcode", "add"}, {"shape", "f32[2,4]"}}})));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
