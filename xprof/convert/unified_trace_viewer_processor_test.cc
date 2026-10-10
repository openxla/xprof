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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor.h"
#include "xprof/convert/unified_profile_processor_factory.h"
#include "xprof/convert/unified_session_snapshot.h"
#include "xprof/convert/unified_tools_registration.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

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

TEST(UnifiedTraceViewerProcessorTest, ProcessSessionJsonSuccess) {
  RegisterUnifiedToolRegistrations();
  ToolOptions options;
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("trace_viewer",
                                                           options);
  ASSERT_NE(processor, nullptr);

  std::string session_dir = tsl::io::JoinPath(
      testing::TempDir(), "unified_trace_viewer_processor_test_json");
  tsl::Env::Default()->RecursivelyCreateDir(session_dir).IgnoreError();
  std::string xspace_path =
      tsl::io::JoinPath(session_dir, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  std::vector<std::string> xspace_paths = {xspace_path};
  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));

  EXPECT_OK(processor->ProcessSession(session_snapshot, options));
  EXPECT_EQ(processor->GetContentType(), "application/json");
  EXPECT_THAT(processor->GetData(), Not(IsEmpty()));

  ToolOptions mpmd_options;
  mpmd_options["mpmd_pipeline_view"] = true;
  mpmd_options["mpmd_single_device_per_stage"] = true;
  std::unique_ptr<UnifiedProfileProcessor> mpmd_processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("trace_viewer",
                                                           mpmd_options);
  ASSERT_NE(mpmd_processor, nullptr);
  EXPECT_OK(mpmd_processor->ProcessSession(session_snapshot, mpmd_options));
  EXPECT_EQ(mpmd_processor->GetContentType(), "application/json");
  const std::string& mpmd_data = mpmd_processor->GetData();
  EXPECT_THAT(
      mpmd_data,
      HasSubstr(R"({"name":"mpmd_single_device_per_stage","value":true})"));
}

TEST(UnifiedTraceViewerProcessorTest, ProcessSessionPbSuccess) {
  RegisterUnifiedToolRegistrations();
  ToolOptions options;
  options["format"] = "pb";
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("trace_viewer",
                                                           options);
  ASSERT_NE(processor, nullptr);

  std::string session_dir = tsl::io::JoinPath(
      testing::TempDir(), "unified_trace_viewer_processor_test_pb");
  tsl::Env::Default()->RecursivelyCreateDir(session_dir).IgnoreError();
  std::string xspace_path =
      tsl::io::JoinPath(session_dir, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  std::vector<std::string> xspace_paths = {xspace_path};
  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));

  EXPECT_OK(processor->ProcessSession(session_snapshot, options));
  EXPECT_EQ(processor->GetContentType(), "application/octet-stream");
  EXPECT_THAT(processor->GetData(), Not(IsEmpty()));

  XSpace tpu_space;
  tensorflow::profiler::XPlane* tpu_plane = tpu_space.add_planes();
  tpu_plane->set_name("/device:TPU:0");
  tensorflow::profiler::XLine* line = tpu_plane->add_lines();
  line->set_id(1);
  tensorflow::profiler::XEvent* event = line->add_events();
  event->set_offset_ps(100);
  event->set_duration_ps(50);
  std::string tpu_xspace_path =
      tsl::io::JoinPath(session_dir, "tpu_host.xplane.pb");
  ASSERT_OK(WriteBinaryProto(tpu_xspace_path, tpu_space));

  std::vector<std::string> tpu_xspace_paths = {tpu_xspace_path};
  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot tpu_session_snapshot,
      SessionSnapshot::Create(tpu_xspace_paths, /*xspaces=*/std::nullopt));

  EXPECT_OK(processor->ProcessSession(tpu_session_snapshot, options));
  EXPECT_EQ(processor->GetContentType(), "application/octet-stream");
  EXPECT_THAT(processor->GetData(), Not(IsEmpty()));

  ToolOptions mpmd_options;
  mpmd_options["format"] = "pb";
  mpmd_options["mpmd_pipeline_view"] = true;
  mpmd_options["mpmd_single_device_per_stage"] = true;
  std::unique_ptr<UnifiedProfileProcessor> mpmd_processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("trace_viewer",
                                                           mpmd_options);
  ASSERT_NE(mpmd_processor, nullptr);
  EXPECT_OK(mpmd_processor->ProcessSession(tpu_session_snapshot, mpmd_options));
  EXPECT_EQ(mpmd_processor->GetContentType(), "application/octet-stream");
  const std::string& mpmd_pb_data = mpmd_processor->GetData();
  EXPECT_THAT(mpmd_pb_data, Not(IsEmpty()));
}

// The streaming trace viewer must keep using the legacy
// StreamingTraceViewerProcessor: the unified processor is non-streaming and
// does not honor the streaming protocol (viewport, resolution, search, uid
// lookups). An unregistered tool makes the dispatcher fall back to legacy.
TEST(UnifiedTraceViewerProcessorTest, StreamingTraceViewerIsNotRegistered) {
  RegisterUnifiedToolRegistrations();
  ToolOptions options;
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("trace_viewer@",
                                                           options);
  EXPECT_EQ(processor, nullptr);
}

}  // namespace
}  // namespace xprof
