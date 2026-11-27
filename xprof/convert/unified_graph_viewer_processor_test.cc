/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "file/base/filesystem.h"
#include "file/base/options.h"
#include "file/base/path.h"
#include "net/proto2/contrib/parse_proto/parse_text_proto.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xla/service/hlo.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor.h"
#include "xprof/convert/unified_profile_processor_factory.h"
#include "xprof/convert/unified_tools_registration.h"

namespace xprof {
namespace {

using ::google::protobuf::contrib::parse_proto::ParseTextProtoOrDie;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;

class UnifiedGraphViewerProcessorTest : public testing::Test {
 protected:
  void SetUp() override {
    session_dir_ = file::JoinPath(testing::TempDir(),
                                  "unified_graph_viewer_processor_test");
    file::RecursivelyDelete(session_dir_, file::Defaults()).IgnoreError();
    CHECK_OK(file::CreateDir(session_dir_, file::Defaults()));
    RegisterUnifiedToolRegistrations();
  }

  void TearDown() override {
    file::RecursivelyDelete(session_dir_, file::Defaults()).IgnoreError();
  }

  std::string session_dir_;
  ToolOptions options_;
};

TEST_F(UnifiedGraphViewerProcessorTest, HandleMissingHloProto) {
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("graph_viewer",
                                                           options_);
  ASSERT_NE(processor, nullptr);

  std::string xspace_path = file::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  std::vector<std::string> xspace_paths = {xspace_path};
  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));

  // Should fail because HLO proto is missing.
  EXPECT_FALSE(processor->ProcessSession(session_snapshot, options_).ok());
}

TEST_F(UnifiedGraphViewerProcessorTest, HandleSuccess) {
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("graph_viewer",
                                                           options_);
  ASSERT_NE(processor, nullptr);

  xla::HloProto dummy_hlo = ParseTextProtoOrDie(R"pb(
    hlo_module {
      name: "my_module"
      entry_computation_name: "my_module"
      entry_computation_id: 1
      host_program_shape {
        result {
          element_type: F32
          dimensions: [ 1 ]
        }
      }
      computations {
        name: "my_module"
        id: 1
        root_id: 10
        instructions {
          name: "constant"
          opcode: "constant"
          id: 10
          shape {
            element_type: F32
            dimensions: [ 1 ]
            layout { minor_to_major: 0 }
          }
          literal {
            shape {
              element_type: F32
              dimensions: [ 1 ]
              layout { minor_to_major: 0 }
            }
            f32s: 1.0
          }
        }
        program_shape {
          result {
            element_type: F32
            dimensions: [ 1 ]
          }
        }
      }
    }
  )pb");

  std::string hlo_path = file::JoinPath(session_dir_, "my_module.hlo_proto.pb");
  ASSERT_OK(WriteBinaryProto(hlo_path, dummy_hlo));

  std::string xspace_path = file::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  std::vector<std::string> xspace_paths = {xspace_path};
  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));

  ToolOptions options;
  options["module_name"] = "my_module";
  options["type"] = "short_txt";
  options["graph_width"] = "1";
  options["merge_fusion"] = "false";

  ASSERT_OK(processor->ProcessSession(session_snapshot, options));
  EXPECT_FALSE(processor->GetData().empty());
}

TEST_F(UnifiedGraphViewerProcessorTest, HandleOpStyles) {
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("graph_viewer",
                                                           options_);
  ASSERT_NE(processor, nullptr);

  std::string xspace_path = file::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  std::vector<std::string> xspace_paths = {xspace_path};
  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));

  ToolOptions options;
  options["type"] = "op_styles";

  ASSERT_OK(processor->ProcessSession(session_snapshot, options));
  EXPECT_EQ(processor->GetContentType(), "application/json");
  EXPECT_THAT(processor->GetData(), ::testing::HasSubstr("\"kBlue\""));
}

TEST_F(UnifiedGraphViewerProcessorTest, HandleMeGraph) {
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("graph_viewer",
                                                           options_);
  ASSERT_NE(processor, nullptr);

  xla::HloProto dummy_hlo = ParseTextProtoOrDie(R"pb(
    hlo_module {
      name: "my_module"
      entry_computation_name: "my_module"
      entry_computation_id: 1
      host_program_shape {
        result {
          element_type: F32
          dimensions: [ 1 ]
        }
      }
      computations {
        name: "my_module"
        id: 1
        root_id: 10
        instructions {
          name: "constant"
          opcode: "constant"
          id: 10
          shape {
            element_type: F32
            dimensions: [ 1 ]
            layout { minor_to_major: 0 }
          }
          literal {
            shape {
              element_type: F32
              dimensions: [ 1 ]
              layout { minor_to_major: 0 }
            }
            f32s: 1.0
          }
        }
        program_shape {
          result {
            element_type: F32
            dimensions: [ 1 ]
          }
        }
      }
    }
  )pb");

  std::string hlo_path = file::JoinPath(session_dir_, "my_module.hlo_proto.pb");
  ASSERT_OK(WriteBinaryProto(hlo_path, dummy_hlo));

  std::string xspace_path = file::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  std::vector<std::string> xspace_paths = {xspace_path};
  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));

  ToolOptions options;
  options["module_name"] = "my_module";
  options["type"] = "me_graph";
  options["node_name"] = "constant";

  ASSERT_OK(processor->ProcessSession(session_snapshot, options));
  EXPECT_EQ(processor->GetContentType(), "application/json");
  EXPECT_THAT(processor->GetData(),
              ::testing::AllOf(::testing::HasSubstr("\"subgraphs\""),
                               ::testing::HasSubstr("\"nodes\""),
                               ::testing::HasSubstr("\"constant\"")));
}

TEST_F(UnifiedGraphViewerProcessorTest,
       HandleBooleanStringShowMetadataAndMergeFusion) {
  xla::HloProto dummy_hlo = ParseTextProtoOrDie(R"pb(
    hlo_module {
      name: "my_module"
      entry_computation_name: "entry"
      entry_computation_id: 1
      host_program_shape {
        parameters {
          element_type: F32
          dimensions: [ 2 ]
          layout { minor_to_major: 0 }
        }
        parameter_names: "p0"
        result {
          element_type: F32
          dimensions: [ 2 ]
          layout { minor_to_major: 0 }
        }
      }
      computations {
        name: "fused_comp"
        id: 2
        root_id: 21
        instructions {
          name: "fused_param"
          opcode: "parameter"
          parameter_number: 0
          id: 20
          shape {
            element_type: F32
            dimensions: [ 2 ]
            layout { minor_to_major: 0 }
          }
        }
        instructions {
          name: "fused_negate"
          opcode: "negate"
          id: 21
          operand_ids: 20
          shape {
            element_type: F32
            dimensions: [ 2 ]
            layout { minor_to_major: 0 }
          }
        }
        program_shape {
          parameters {
            element_type: F32
            dimensions: [ 2 ]
            layout { minor_to_major: 0 }
          }
          parameter_names: "fp0"
          result {
            element_type: F32
            dimensions: [ 2 ]
            layout { minor_to_major: 0 }
          }
        }
      }
      computations {
        name: "entry"
        id: 1
        root_id: 11
        instructions {
          name: "param.0"
          opcode: "parameter"
          parameter_number: 0
          id: 10
          shape {
            element_type: F32
            dimensions: [ 2 ]
            layout { minor_to_major: 0 }
          }
        }
        instructions {
          name: "fusion.1"
          opcode: "fusion"
          fusion_kind: "kLoop"
          id: 11
          operand_ids: 10
          called_computation_ids: 2
          backend_config: "custom_backend_cfg"
          metadata { op_name: "source_op_meta" }
          shape {
            element_type: F32
            dimensions: [ 2 ]
            layout { minor_to_major: 0 }
          }
        }
        program_shape {
          parameters {
            element_type: F32
            dimensions: [ 2 ]
            layout { minor_to_major: 0 }
          }
          parameter_names: "p0"
          result {
            element_type: F32
            dimensions: [ 2 ]
            layout { minor_to_major: 0 }
          }
        }
      }
    }
  )pb");

  std::string hlo_path = file::JoinPath(session_dir_, "my_module.hlo_proto.pb");
  ASSERT_OK(WriteBinaryProto(hlo_path, dummy_hlo));

  std::string xspace_path = file::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create({xspace_path}, /*xspaces=*/std::nullopt));

  auto render = [&](const ToolOptions& opts) -> absl::StatusOr<std::string> {
    std::unique_ptr<UnifiedProfileProcessor> processor =
        UnifiedProfileProcessorFactory::GetInstance().Create("graph_viewer",
                                                             opts);
    if (processor == nullptr) {
      return absl::InternalError("Failed to create graph_viewer processor");
    }
    absl::Status status = processor->ProcessSession(session_snapshot, opts);
    if (!status.ok()) {
      return status;
    }
    return processor->GetData();
  };

  ToolOptions options;
  options["module_name"] = "my_module";
  options["type"] = "graph";
  options["format"] = "dot";
  options["node_name"] = "fusion.1";
  options["graph_width"] = "1";

  // Scenario 1: Default/false flags expand fusion and omit backend_config.
  options["show_metadata"] = "false";
  options["merge_fusion"] = "false";
  ASSERT_OK_AND_ASSIGN(std::string dot_default, render(options));
  EXPECT_THAT(dot_default, ::testing::HasSubstr("fused_negate"));
  EXPECT_THAT(dot_default,
              ::testing::Not(::testing::HasSubstr("custom_backend_cfg")));

  // Scenario 2: show_metadata="true" renders backend_config on the node.
  options["show_metadata"] = "true";
  options["merge_fusion"] = "false";
  ASSERT_OK_AND_ASSIGN(std::string dot_metadata, render(options));
  EXPECT_THAT(dot_metadata, ::testing::HasSubstr("fused_negate"));
  EXPECT_THAT(dot_metadata, ::testing::HasSubstr("custom_backend_cfg"));

  // Scenario 3: merge_fusion="true" keeps fusion.1 while collapsing fused ops.
  options["show_metadata"] = "false";
  options["merge_fusion"] = "true";
  ASSERT_OK_AND_ASSIGN(std::string dot_merged, render(options));
  EXPECT_THAT(
      dot_merged,
      ::testing::AllOf(::testing::HasSubstr("fusion.1"),
                       ::testing::Not(::testing::HasSubstr("fused_negate"))));

  // Scenario 4: short_txt with show_metadata="false" renders HLO without meta.
  ToolOptions txt_options;
  txt_options["module_name"] = "my_module";
  txt_options["type"] = "short_txt";
  txt_options["show_metadata"] = "false";
  ASSERT_OK_AND_ASSIGN(std::string txt_no_meta, render(txt_options));
  EXPECT_THAT(
      txt_no_meta,
      ::testing::AllOf(::testing::HasSubstr("fusion.1"),
                       ::testing::Not(::testing::HasSubstr("source_op_meta"))));

  // Scenario 5: short_txt with show_metadata="true" includes op metadata.
  txt_options["show_metadata"] = "true";
  ASSERT_OK_AND_ASSIGN(std::string txt_with_meta, render(txt_options));
  EXPECT_THAT(txt_with_meta,
              ::testing::AllOf(::testing::HasSubstr("fusion.1"),
                               ::testing::HasSubstr("source_op_meta")));
}

TEST_F(UnifiedGraphViewerProcessorTest, HandleContentTypesByTypeAndFormat) {
  xla::HloProto dummy_hlo = ParseTextProtoOrDie(R"pb(
    hlo_module {
      name: "my_module"
      entry_computation_name: "entry"
      entry_computation_id: 1
      host_program_shape { result { element_type: F32 } }
      computations {
        name: "entry"
        id: 1
        root_id: 10
        instructions {
          name: "constant.1"
          opcode: "constant"
          id: 10
          shape {
            element_type: F32
            layout {}
          }
          literal {
            shape {
              element_type: F32
              layout {}
            }
            f32s: 1.0
          }
        }
        program_shape { result { element_type: F32 } }
      }
    }
  )pb");
  ASSERT_OK(WriteBinaryProto(
      file::JoinPath(session_dir_, "my_module.hlo_proto.pb"), dummy_hlo));
  std::string xspace_path = file::JoinPath(session_dir_, "test_host.xplane.pb");
  ASSERT_OK(WriteBinaryProto(xspace_path, XSpace()));
  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot snapshot,
      SessionSnapshot::Create({xspace_path}, /*xspaces=*/std::nullopt));

  struct Case {
    const char* type;
    const char* format;
    const char* mime;
    const char* substr;
  };
  const Case kCases[] = {
      {"adj_nodes", "", "application/json", "\"constant.1\""},
      {"graph", "html", "text/html", ""},
      {"graph", "dot", "text/plain", "digraph"},
      {"short_txt", "html", "text/plain", ""},
  };
  for (const auto& tc : kCases) {
    ToolOptions opts = {{"module_name", std::string("my_module")},
                        {"node_name", std::string("constant.1")},
                        {"graph_width", std::string("1")},
                        {"type", std::string(tc.type)}};
    if (*tc.format != '\0') opts["format"] = std::string(tc.format);
    auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
        "graph_viewer", opts);
    ASSERT_NE(processor, nullptr);
    ASSERT_OK(processor->ProcessSession(snapshot, opts));
    EXPECT_EQ(processor->GetContentType(), tc.mime) << tc.type;
    if (*tc.substr != '\0') {
      EXPECT_THAT(processor->GetData(), ::testing::HasSubstr(tc.substr));
    }
  }
}

}  // namespace
}  // namespace xprof
