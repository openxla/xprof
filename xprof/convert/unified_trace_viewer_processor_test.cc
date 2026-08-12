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

#include "xprof/convert/unified_trace_viewer_processor.h"

#include <cstddef>
#include <string>

#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_session_snapshot.h"

namespace xprof {
namespace {

class MockXprofSessionSnapshot : public XprofSessionSnapshot {
 public:
  size_t XSpaceSize() const override { return 0; }
  absl::StatusOr<tensorflow::profiler::XSpace*> GetXSpace(
      size_t index, google::protobuf::Arena* arena) const override {
    return absl::NotFoundError("No XSpace");
  }
  std::string GetHostname(size_t index) const override { return ""; }
  absl::string_view GetSessionRunDir() const override { return ""; }
  absl::StatusOr<std::string> GetHostDataFileName(
      tensorflow::profiler::StoredDataType data_type,
      absl::string_view host) const override {
    return absl::NotFoundError("No File");
  }
};

TEST(UnifiedTraceViewerProcessorTest, InvalidArgument) {
  tensorflow::profiler::ToolOptions options;
  UnifiedTraceViewerProcessor processor(options);
  MockXprofSessionSnapshot session_snapshot;
  absl::Status status = processor.ProcessSession(session_snapshot, options);
  EXPECT_FALSE(status.ok());
}

}  // namespace
}  // namespace xprof

