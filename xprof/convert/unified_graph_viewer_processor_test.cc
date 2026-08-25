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
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/log/check.h"
#include "absl/status/status.h"
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

}  // namespace
}  // namespace xprof
