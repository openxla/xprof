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

#include "xprof/convert/base_hlo_processor.h"

#include <cstddef>
#include <string>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "xla/service/hlo.pb.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_session_snapshot.h"

namespace xprof {
namespace {

using ::testing::HasSubstr;
using ::testing::status::StatusIs;

class DummyHloProcessor : public BaseHloProcessor {
 public:
  using BaseHloProcessor::BaseHloProcessor;

  absl::Status ProcessHlo(
      const XprofSessionSnapshot& session_snapshot,
      const xla::HloProto& hlo_proto,
      const tensorflow::profiler::ToolOptions& options) override {
    return absl::OkStatus();
  }
};

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

TEST(BaseHloProcessorTest, MapIsUnimplemented) {
  tensorflow::profiler::ToolOptions options;
  DummyHloProcessor processor(options);
  testing::NiceMock<MockXprofSessionSnapshot> snapshot;
  tensorflow::profiler::XSpace xspace;

  EXPECT_THAT(processor.Map(snapshot, "localhost", xspace),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Map not implemented for BaseHloProcessor")));
}

TEST(BaseHloProcessorTest, ReduceIsUnimplemented) {
  tensorflow::profiler::ToolOptions options;
  DummyHloProcessor processor(options);
  testing::NiceMock<MockXprofSessionSnapshot> snapshot;

  EXPECT_THAT(
      processor.Reduce(snapshot, {}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Reduce not implemented for BaseHloProcessor")));
}

TEST(BaseHloProcessorTest,
     ProcessSessionReturnsInvalidArgumentForNonProfilerSnapshot) {
  tensorflow::profiler::ToolOptions options;
  DummyHloProcessor processor(options);
  testing::NiceMock<MockXprofSessionSnapshot> snapshot;

  EXPECT_THAT(processor.ProcessSession(snapshot, options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("session_snapshot is not a "
                                 "tensorflow::profiler::SessionSnapshot")));
}

}  // namespace
}  // namespace xprof
