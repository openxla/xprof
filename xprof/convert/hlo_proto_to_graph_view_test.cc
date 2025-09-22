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

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
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

TEST(ConvertHloModuleProtoToGraphTest, NodeNotFound) {
  xla::HloModuleProto hlo_module_proto;
  hlo_module_proto.set_name("test_module");
  hlo_module_proto.mutable_host_program_shape();
  auto* computation = hlo_module_proto.add_computations();
  computation->set_name("test_module");
  auto* instruction = computation->add_instructions();
  instruction->set_id(0);
  instruction->set_name("constant.0");
  instruction->set_opcode("constant");
  instruction->mutable_shape()->set_element_type(xla::F32);
  computation->set_root_id(0);
  hlo_module_proto.set_entry_computation_name("test_module");
  std::string node_name = "non_existent_node";
  int graph_width = 3;
  xla::HloRenderOptions render_options;
  xla::RenderedGraphFormat format = xla::RenderedGraphFormat::kUrl;

  auto result = ConvertHloModuleProtoToGraph(
      hlo_module_proto, node_name, graph_width, render_options, format);
  EXPECT_THAT(result,
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("Couldn't find HloInstruction or "
                                 "HloComputation named non_existent_node.")));
}

TEST(ConvertHloModuleProtoToGraphTest, NodeFound) {
  xla::HloModuleProto hlo_module_proto;
  hlo_module_proto.set_name("test_module");
  hlo_module_proto.mutable_host_program_shape();
  auto* computation = hlo_module_proto.add_computations();
  computation->set_name("test_module");
  auto* instruction = computation->add_instructions();
  instruction->set_id(0);
  instruction->set_name("constant.0");
  instruction->set_opcode("constant");
  instruction->mutable_shape()->set_element_type(xla::F32);
  computation->set_root_id(0);
  hlo_module_proto.set_entry_computation_name("test_module");
  std::string node_name = "constant.0";  // This node exists.
  int graph_width = 3;
  xla::HloRenderOptions render_options;
  xla::RenderedGraphFormat format = xla::RenderedGraphFormat::kDot;

  auto result = ConvertHloModuleProtoToGraph(
      hlo_module_proto, node_name, graph_width, render_options, format);
  // Expect an OK status and a DOT graph to be returned.
  EXPECT_THAT(result, IsOkAndHolds(StartsWith("digraph")));
}

TEST(ConvertHloModuleProtoToStringViewTest, AllTypes) {
  xla::HloModuleProto hlo_module_proto;
  hlo_module_proto.set_name("test_module");
  hlo_module_proto.mutable_host_program_shape();
  auto* computation = hlo_module_proto.add_computations();
  computation->set_name("test_module");
  auto* instruction = computation->add_instructions();
  instruction->set_id(0);
  instruction->set_name("constant.0");
  instruction->set_opcode("constant");
  instruction->mutable_shape()->set_element_type(xla::F32);
  computation->set_root_id(0);
  hlo_module_proto.set_entry_computation_name("test_module");
  instruction->mutable_metadata()->set_op_name("my_op");

  // JSON type.
  TF_ASSERT_OK_AND_ASSIGN(auto json_str, ConvertHloModuleProtoToStringView(
                                             hlo_module_proto, kJsonTypeName,
                                             /*verbose=*/false,
                                             /*metadata=*/false));
  EXPECT_THAT(json_str, HasSubstr("\"name\": \"test_module\""));

  // Proto type.
  TF_ASSERT_OK_AND_ASSIGN(auto proto_str, ConvertHloModuleProtoToStringView(
                                              hlo_module_proto, kProtoTypeName,
                                              /*verbose=*/false,
                                              /*metadata=*/false));
  EXPECT_EQ(proto_str, hlo_module_proto.SerializeAsString());

  // Proto text type.
  TF_ASSERT_OK_AND_ASSIGN(
      auto proto_text_str,
      ConvertHloModuleProtoToStringView(hlo_module_proto, kProtoTextTypeName,
                                        /*verbose=*/false,
                                        /*metadata=*/false));
  EXPECT_THAT(proto_text_str, HasSubstr("hlo_module"));
  EXPECT_THAT(proto_text_str, HasSubstr("name: \"test_module\""));

  // Text type (short), without metadata.
  TF_ASSERT_OK_AND_ASSIGN(
      auto short_txt_str,
      ConvertHloModuleProtoToStringView(hlo_module_proto, "unknown_type",
                                        /*verbose=*/false,
                                        /*metadata=*/false));
  EXPECT_THAT(short_txt_str, HasSubstr("HloModule test_module"));
  EXPECT_THAT(short_txt_str, HasSubstr("ENTRY test_module"));
  EXPECT_THAT(short_txt_str, Not(HasSubstr("my_op")));

  // Text type (short), with metadata.
  TF_ASSERT_OK_AND_ASSIGN(
      auto short_txt_meta_str,
      ConvertHloModuleProtoToStringView(hlo_module_proto, "unknown_type",
                                        /*verbose=*/false,
                                        /*metadata=*/true));
  EXPECT_THAT(short_txt_meta_str, HasSubstr("HloModule test_module"));
  EXPECT_THAT(short_txt_meta_str, HasSubstr("ENTRY test_module"));
  EXPECT_THAT(short_txt_meta_str, HasSubstr("my_op"));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
