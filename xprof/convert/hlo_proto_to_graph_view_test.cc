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

#include <cstddef>
#include <string>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "nlohmann/json_fwd.hpp"  // from @com_github_nlohmann_json
#include "nlohmann/json.hpp"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xla/xla_data.pb.h"
#include "xprof/convert/graphviz_helper.h"
#include "xprof/convert/tool_options.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Not;
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

  // Boolean query string options from HTTP query params ("true" / "false").
  ToolOptions options3;
  options3["type"] = "graph";
  options3["node_name"] = "fusion.111";
  options3["graph_width"] = "2";
  options3["show_metadata"] = "true";
  options3["merge_fusion"] = "true";
  options3["format"] = "html";
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params3,
                          ParseGraphViewerParams(options3));
  EXPECT_EQ(params3.graph_width, 2);
  EXPECT_EQ(params3.render_options.show_backend_config, true);
  EXPECT_EQ(params3.render_options.show_fusion_subcomputations, false);

  ToolOptions options4;
  options4["type"] = "graph";
  options4["show_metadata"] = "false";
  options4["merge_fusion"] = "false";
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params4,
                          ParseGraphViewerParams(options4));
  EXPECT_EQ(params4.render_options.show_backend_config, false);
  EXPECT_EQ(params4.render_options.show_fusion_subcomputations, true);

  // Malformed boolean strings fall back to defaults.
  ToolOptions options5;
  options5["type"] = "graph";
  options5["show_metadata"] = "garbage";
  options5["merge_fusion"] = "garbage";
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params5,
                          ParseGraphViewerParams(options5));
  EXPECT_EQ(params5.render_options.show_backend_config, false);
  EXPECT_EQ(params5.render_options.show_fusion_subcomputations, true);
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

  ToolOptions options3;
  options3["type"] = "short_txt";
  options3["show_metadata"] = "true";
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params3,
                          ParseGraphViewerParams(options3));
  EXPECT_EQ(params3.show_metadata, true);
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

  ToolOptions options3;
  options3["type"] = "long_txt";
  options3["show_metadata"] = "true";
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params3,
                          ParseGraphViewerParams(options3));
  EXPECT_EQ(params3.show_metadata, true);
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

TEST(GraphvizHelperTest, JsStringEscapeSpecialCharacters) {
  EXPECT_THAT(WrapDotInHtml(""), HasSubstr(R"(const data = "";)"));
  EXPECT_THAT(WrapDotInHtml("digraph G {}"),
              HasSubstr(R"(const data = "digraph G {}";)"));
  EXPECT_THAT(WrapDotInHtml("a\"b\\c"),
              HasSubstr(R"(const data = "a\"b\\c";)"));
  EXPECT_THAT(WrapDotInHtml("a\nb\rc\td\be\ff"),
              HasSubstr(R"(const data = "a\nb\rc\td\be\ff";)"));
  EXPECT_THAT(WrapDotInHtml(absl::string_view("a\x00\x01\x1f"
                                              "b",
                                              5)),
              HasSubstr(R"(const data = "a\u0000\u0001\u001fb";)"));
  EXPECT_THAT(
      WrapDotInHtml("<script>&</script>"),
      HasSubstr(
          R"(const data = "\u003cscript\u003e\u0026\u003c/script\u003e";)"));
  EXPECT_THAT(WrapDotInHtml("${alert(1)}`"),
              HasSubstr(R"(const data = "${alert(1)}`";)"));
  EXPECT_THAT(WrapDotInHtml("\xe2\x80\xa8"),
              HasSubstr(R"(const data = "\u2028";)"));
  EXPECT_THAT(WrapDotInHtml("\xe2\x80\xa9"),
              HasSubstr(R"(const data = "\u2029";)"));
  EXPECT_THAT(WrapDotInHtml("\xe2\x80\xa7"),
              HasSubstr("const data = \"\xe2\x80\xa7\";"));
  EXPECT_THAT(WrapDotInHtml("\xe2\x82\xac"),
              HasSubstr("const data = \"\xe2\x82\xac\";"));
  EXPECT_THAT(WrapDotInHtml("\xe2"), HasSubstr("const data = \"\xe2\";"));
}

TEST(GraphvizHelperTest, WrapDotInHtmlEscapesTemplateLiteralAndScriptBreakout) {
  std::string html1 =
      WrapDotInHtml(R"(digraph G { n0 [tooltip="${alert(1)}"]; })");
  EXPECT_THAT(html1, Not(HasSubstr("const data = `")));
  EXPECT_THAT(
      html1,
      HasSubstr(
          R"(const data = "digraph G { n0 [tooltip=\"${alert(1)}\"]; }";)"));

  std::string html2 =
      WrapDotInHtml(R"(digraph G { n0 [tooltip="`;alert(2);//"]; })");
  EXPECT_THAT(html2, Not(HasSubstr("const data = `")));
  EXPECT_THAT(
      html2,
      HasSubstr(
          R"(const data = "digraph G { n0 [tooltip=\"`;alert(2);//\"]; }";)"));

  std::string html3 = WrapDotInHtml(
      R"(digraph G { n0 [tooltip="</script><script>alert(3)</script>"]; })");
  EXPECT_THAT(html3, Not(HasSubstr("const data = `")));
  EXPECT_THAT(html3, Not(HasSubstr("<script>alert(3)</script>")));
  EXPECT_THAT(
      html3,
      HasSubstr(
          R"(\u003c/script\u003e\u003cscript\u003ealert(3)\u003c/script\u003e)"));

  std::string html4 =
      WrapDotInHtml("digraph G {\n  n0 -> n1;\n}", R"(dot");alert(4);//)");
  EXPECT_THAT(html4, Not(HasSubstr("const data = `")));
  EXPECT_THAT(html4,
              HasSubstr(R"(const data = "digraph G {\n  n0 -\u003e n1;\n}";)"));
  EXPECT_THAT(
      html4,
      HasSubstr(
          R"(hpccWasm.graphviz.layout(dot_data, "svg", "dot\");alert(4);//"))"));
}

TEST(HloProtoToGraphViewTest,
     ConvertHloProtoToGraphHtmlEscapesOpMetadataXssPayloads) {
  xla::HloProto hlo_proto;
  xla::HloModuleProto* module = hlo_proto.mutable_hlo_module();
  module->set_name("test_module");
  module->set_id(1);
  module->set_entry_computation_name("c");
  module->set_entry_computation_id(10);

  xla::ShapeProto shape;
  shape.set_element_type(xla::F32);
  shape.add_dimensions(2);
  shape.mutable_layout()->add_minor_to_major(0);

  xla::HloComputationProto* comp = module->add_computations();
  comp->set_name("c");
  comp->set_id(10);
  comp->set_root_id(0);

  xla::HloInstructionProto* param = comp->add_instructions();
  param->set_name("p");
  param->set_opcode("parameter");
  param->set_parameter_number(0);
  param->set_id(0);
  *param->mutable_shape() = shape;
  param->mutable_metadata()->set_op_name("${MANTIS_XSS_MARKER}");
  param->mutable_metadata()->set_source_file(
      "`;\"</script><script>MANTIS_XSS_MARKER</script>;//");
  param->mutable_metadata()->set_source_line(1);

  xla::ProgramShapeProto* program_shape = module->mutable_host_program_shape();
  *program_shape->add_parameters() = shape;
  program_shape->add_parameter_names("p");
  *program_shape->mutable_result() = shape;

  TF_ASSERT_OK_AND_ASSIGN(
      std::string html,
      ConvertHloProtoToGraph(hlo_proto, "c", /*graph_width=*/3,
                             xla::HloRenderOptions(),
                             xla::RenderedGraphFormat::kHtml));
  EXPECT_THAT(html, HasSubstr("const data = \"digraph "));
  EXPECT_THAT(html, Not(HasSubstr("const data = `")));
  EXPECT_THAT(html,
              Not(HasSubstr("</script><script>MANTIS_XSS_MARKER</script>")));
  EXPECT_THAT(html, Not(HasSubstr("`;\"</script>")));
  EXPECT_THAT(
      html,
      HasSubstr(
          R"(\"\u003c/script\u003e\u003cscript\u003eMANTIS_XSS_MARKER\u003c/script\u003e;//)"));
}

std::string ExtractAndUnescapeJsData(absl::string_view html) {
  constexpr absl::string_view kPrefix = "const data = \"";
  constexpr absl::string_view kSuffix = "\";\n";
  const size_t start = html.find(kPrefix);
  if (start == absl::string_view::npos) {
    return "";
  }
  const size_t content_start = start + kPrefix.size();
  const size_t end = html.find(kSuffix, content_start);
  if (end == absl::string_view::npos) {
    return "";
  }
  absl::string_view escaped = html.substr(content_start, end - content_start);
  std::string out;
  out.reserve(escaped.size());
  for (size_t i = 0; i < escaped.size(); ++i) {
    if (escaped[i] != '\\' || i + 1 >= escaped.size()) {
      out.push_back(escaped[i]);
      continue;
    }
    ++i;
    switch (escaped[i]) {
      case '\\':
        out.push_back('\\');
        break;
      case '"':
        out.push_back('"');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'u':
        if (i + 4 < escaped.size()) {
          absl::string_view hex = escaped.substr(i + 1, 4);
          if (hex == "003c") {
            out.push_back('<');
          } else if (hex == "003e") {
            out.push_back('>');
          } else if (hex == "0026") {
            out.push_back('&');
          } else if (hex == "2028") {
            out.append("\xe2\x80\xa8");
          } else if (hex == "2029") {
            out.append("\xe2\x80\xa9");
          }
          i += 4;
        }
        break;
      default:
        out.push_back(escaped[i]);
        break;
    }
  }
  return out;
}

TEST(HloProtoToGraphViewTest,
     ConvertHloProtoToGraphHtmlPreservesDotAndStylesheet) {
  xla::HloProto hlo_proto;
  xla::HloModuleProto* module = hlo_proto.mutable_hlo_module();
  module->set_name("test_module");
  module->set_id(1);
  module->set_entry_computation_name("entry");
  module->set_entry_computation_id(10);

  xla::ShapeProto shape;
  shape.set_element_type(xla::F32);
  shape.add_dimensions(2);
  shape.add_dimensions(4);
  shape.mutable_layout()->add_minor_to_major(1);
  shape.mutable_layout()->add_minor_to_major(0);

  xla::HloComputationProto* comp = module->add_computations();
  comp->set_name("entry");
  comp->set_id(10);
  comp->set_root_id(2);

  xla::HloInstructionProto* param = comp->add_instructions();
  param->set_name("arg0.1");
  param->set_opcode("parameter");
  param->set_parameter_number(0);
  param->set_id(0);
  *param->mutable_shape() = shape;
  param->mutable_metadata()->set_op_type("Parameter");
  param->mutable_metadata()->set_op_name("inputs\\layer_1\\arg0");
  param->mutable_metadata()->set_source_file("C:\\models\\train\\step.py");
  param->mutable_metadata()->set_source_line(42);

  xla::HloInstructionProto* negate = comp->add_instructions();
  negate->set_name("negate.2");
  negate->set_opcode("negate");
  negate->set_id(1);
  negate->add_operand_ids(0);
  *negate->mutable_shape() = shape;
  negate->mutable_metadata()->set_op_type("Neg");
  negate->mutable_metadata()->set_op_name("block_a\\neg_op\\n1");
  negate->mutable_metadata()->set_source_file("src/ops/math.py");
  negate->mutable_metadata()->set_source_line(108);

  xla::HloInstructionProto* add = comp->add_instructions();
  add->set_name("add.3");
  add->set_opcode("add");
  add->set_id(2);
  add->add_operand_ids(0);
  add->add_operand_ids(1);
  *add->mutable_shape() = shape;
  add->mutable_metadata()->set_op_type("AddV2");
  add->mutable_metadata()->set_op_name("residual_add");

  xla::ProgramShapeProto* program_shape = module->mutable_host_program_shape();
  *program_shape->add_parameters() = shape;
  program_shape->add_parameter_names("arg0");
  *program_shape->mutable_result() = shape;

  xla::HloRenderOptions render_options;
  render_options.show_backend_config = true;

  TF_ASSERT_OK_AND_ASSIGN(
      std::string dot,
      ConvertHloProtoToGraph(hlo_proto, "negate.2", /*graph_width=*/3,
                             render_options, xla::RenderedGraphFormat::kDot));
  EXPECT_EQ(ExtractAndUnescapeJsData(WrapDotInHtml(dot)), dot);

  TF_ASSERT_OK_AND_ASSIGN(
      std::string html,
      ConvertHloProtoToGraph(hlo_proto, "negate.2", /*graph_width=*/3,
                             render_options, xla::RenderedGraphFormat::kHtml));

  std::string decoded_js_data = ExtractAndUnescapeJsData(html);
  EXPECT_THAT(decoded_js_data, StartsWith("digraph G {\n"));
  EXPECT_THAT(decoded_js_data, HasSubstr("stylesheet=<\n  data:text/css,"));
  EXPECT_THAT(decoded_js_data, HasSubstr("\n>\n"));
  EXPECT_THAT(decoded_js_data, HasSubstr("inputs\\layer_1\\arg0"));
  EXPECT_THAT(decoded_js_data, HasSubstr("C:\\models\\train\\step.py:42"));
  EXPECT_THAT(decoded_js_data, HasSubstr("block_a\\neg_op\\n1"));
  EXPECT_THAT(decoded_js_data, HasSubstr("arg0.1 -> negate.2"));
  EXPECT_THAT(decoded_js_data, HasSubstr("negate.2 -> add.3"));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
