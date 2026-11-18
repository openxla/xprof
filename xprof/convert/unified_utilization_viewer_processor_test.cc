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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "file/base/filesystem.h"
#include "file/base/options.h"
#include "file/base/path.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/jsoncpp/include/json/reader.h"
#include "third_party/jsoncpp/include/json/value.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor.h"
#include "xprof/convert/unified_profile_processor_factory.h"
#include "xprof/convert/unified_session_snapshot.h"
#include "xprof/convert/unified_tools_registration.h"
#include "xprof/utils/tpu_counter_ids_v7x.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;
using ::testing::_;
using ::testing::EqualsProto;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Return;
using ::testing::status::StatusIs;
using ::tsl::profiler::GetStatTypeStr;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XPlaneBuilder;

void PopulateTpuV7xXSpace(XSpace& xspace) {
  xspace.add_hostnames("test_host");
  XPlane* plane = xspace.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v7x");

  auto line_builder = builder.GetOrCreateLine(0);

  using Tpu7x = TpuCounterIdsTpu7x;
  uint64_t cycles_id = Tpu7x::
      VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT
  uint64_t scalar_inst_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_SCALAR_ALU_INSTRUCTION_0;  // NOLINT

  {
    auto event_builder =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("CYCLES"));
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                   StatType::kPerformanceCounterId)),
                               cycles_id);
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kCounterValue)),
                               1000.0);
  }
  {
    auto event_builder = line_builder.AddEvent(
        *builder.GetOrCreateEventMetadata("SCALAR_ALU_0"));
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                   StatType::kPerformanceCounterId)),
                               scalar_inst_id);
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kCounterValue)),
                               500.0);
  }
}

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

class UnifiedUtilizationViewerProcessorTest : public testing::Test {
 protected:
  void SetUp() override {
    session_dir_ =
        file::JoinPath(testing::TempDir(), "unified_utilization_viewer_test");
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

TEST_F(UnifiedUtilizationViewerProcessorTest, EmptyXSpaceTest) {
  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "utilization_viewer", options_);
  ASSERT_TRUE(processor != nullptr);

  std::string xspace_path = file::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  CHECK_OK(WriteBinaryProto(xspace_path, dummy_space));

  ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot,
                       SessionSnapshot::Create({xspace_path}, std::nullopt));

  ASSERT_OK(processor->ProcessSession(session_snapshot, options_));

  std::string output_str = processor->GetData();
  EXPECT_THAT(output_str, Not(IsEmpty()));
  EXPECT_EQ(processor->GetContentType(), "application/json");

  Json::Value json;
  Json::Reader reader;
  ASSERT_TRUE(reader.parse(output_str, json));
  ASSERT_TRUE(json.isMember("rows"));
  EXPECT_THAT(json["rows"], IsEmpty());
}

TEST_F(UnifiedUtilizationViewerProcessorTest, NoXSpaceTest) {
  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "utilization_viewer", options_);
  ASSERT_TRUE(processor != nullptr);

  testing::NiceMock<MockXprofSessionSnapshot> session_snapshot;
  EXPECT_CALL(session_snapshot, XSpaceSize()).WillRepeatedly(Return(0));

  EXPECT_THAT(processor->ProcessSession(session_snapshot, options_),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("No XSpace found in the session.")));
}

TEST_F(UnifiedUtilizationViewerProcessorTest,
       PreloadedXSpaceIsNotMutatedInPlace) {
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("utilization_viewer",
                                                           options_);
  ASSERT_NE(processor, nullptr);

  XSpace preloaded_xspace;
  PopulateTpuV7xXSpace(preloaded_xspace);
  XSpace original_xspace = preloaded_xspace;

  testing::NiceMock<MockXprofSessionSnapshot> session_snapshot;
  EXPECT_CALL(session_snapshot, XSpaceSize()).WillRepeatedly(Return(1));
  EXPECT_CALL(session_snapshot, GetXSpace(0, _))
      .WillOnce(Return(&preloaded_xspace));

  ASSERT_OK(processor->ProcessSession(session_snapshot, options_));
  EXPECT_THAT(preloaded_xspace, EqualsProto(original_xspace));
  EXPECT_EQ(processor->GetContentType(), "application/json");

  Json::Value json;
  Json::Reader reader;
  ASSERT_TRUE(reader.parse(processor->GetData(), json));
  ASSERT_TRUE(json.isMember("rows"));
  EXPECT_THAT(json["rows"], Not(IsEmpty()));
}

TEST_F(UnifiedUtilizationViewerProcessorTest,
       ArenaOwnedXSpaceIsProcessedInPlace) {
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("utilization_viewer",
                                                           options_);
  ASSERT_NE(processor, nullptr);

  testing::NiceMock<MockXprofSessionSnapshot> session_snapshot;
  EXPECT_CALL(session_snapshot, XSpaceSize()).WillRepeatedly(Return(1));
  EXPECT_CALL(session_snapshot, GetXSpace(0, _))
      .WillOnce([](size_t /*index*/, google::protobuf::Arena* arena) {
        XSpace* arena_xspace = google::protobuf::Arena::Create<XSpace>(arena);
        PopulateTpuV7xXSpace(*arena_xspace);
        return arena_xspace;
      });

  ASSERT_OK(processor->ProcessSession(session_snapshot, options_));
  EXPECT_EQ(processor->GetContentType(), "application/json");

  Json::Value json;
  Json::Reader reader;
  ASSERT_TRUE(reader.parse(processor->GetData(), json));
  ASSERT_TRUE(json.isMember("rows"));
  EXPECT_THAT(json["rows"], Not(IsEmpty()));
}

}  // namespace
}  // namespace xprof
