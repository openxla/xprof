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

#include <cstddef>
#include <memory>
#include <string>

#include "net/proto2/contrib/parse_proto/parse_text_proto.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "xla/service/hlo.pb.h"
#include "xprof/convert/base_hlo_processor.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor.h"
#include "xprof/convert/unified_profile_processor_factory.h"
#include "xprof/convert/unified_session_snapshot.h"

namespace xprof {
namespace {

using ::google::protobuf::contrib::parse_proto::ParseTextProtoOrDie;
using ::testing::IsEmpty;
using ::testing::Not;

class MockXprofSessionSnapshot : public XprofSessionSnapshot {
 public:
  MOCK_METHOD(size_t, XSpaceSize, (), (const, override));
  MOCK_METHOD((absl::StatusOr<tensorflow::profiler::XSpace*>), GetXSpace,
              (size_t index, google::protobuf::Arena* arena), (const, override));
  MOCK_METHOD(std::string, GetHostname, (size_t index), (const, override));
  MOCK_METHOD(absl::string_view, GetSessionRunDir, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::string>, GetHostDataFileName,
              (tensorflow::profiler::StoredDataType data_type,
               absl::string_view host),
              (const, override));
};

TEST(UnifiedMemoryViewerProcessorTest, ProcessHloJsonTest) {
  tensorflow::profiler::ToolOptions options;
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("memory_viewer",
                                                         options);
  ASSERT_NE(processor, nullptr);

  auto* memory_viewer_processor =
      dynamic_cast<BaseHloProcessor*>(processor.get());
  ASSERT_NE(memory_viewer_processor, nullptr);

  xla::HloProto hlo_proto = ParseTextProtoOrDie(R"pb(
    hlo_module {
      name: "test_module"
    }
  )pb");

  testing::NiceMock<MockXprofSessionSnapshot> session_snapshot;
  ASSERT_OK(memory_viewer_processor->ProcessHlo(session_snapshot, hlo_proto,
                                                options));

  EXPECT_EQ(memory_viewer_processor->GetContentType(), "application/json");
  EXPECT_THAT(memory_viewer_processor->GetData(), Not(IsEmpty()));
}

TEST(UnifiedMemoryViewerProcessorTest, ProcessHloHtmlTest) {
  tensorflow::profiler::ToolOptions options;
  options["view_memory_allocation_timeline"] = true;
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("memory_viewer",
                                                         options);
  ASSERT_NE(processor, nullptr);

  auto* memory_viewer_processor =
      dynamic_cast<BaseHloProcessor*>(processor.get());
  ASSERT_NE(memory_viewer_processor, nullptr);

  xla::HloProto hlo_proto = ParseTextProtoOrDie(R"pb(
    hlo_module {
      name: "test_module"
    }
  )pb");

  testing::NiceMock<MockXprofSessionSnapshot> session_snapshot;
  ASSERT_OK(memory_viewer_processor->ProcessHlo(session_snapshot, hlo_proto,
                                                options));

  EXPECT_EQ(memory_viewer_processor->GetContentType(), "text/html");
  EXPECT_THAT(memory_viewer_processor->GetData(), Not(IsEmpty()));
}

}  // namespace
}  // namespace xprof
