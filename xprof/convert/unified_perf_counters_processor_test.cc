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

namespace xprof {
namespace {

using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Return;
using ::testing::status::StatusIs;
using ::tsl::profiler::GetStatTypeStr;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlaneBuilder;

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

class UnifiedPerfCountersProcessorTest : public testing::Test {
 protected:
  void SetUp() override {
    session_dir_ =
        file::JoinPath(testing::TempDir(), "unified_perf_counters_test");
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

TEST_F(UnifiedPerfCountersProcessorTest, EmptyXSpaceTest) {
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("perf_counters",
                                                           options_);
  ASSERT_NE(processor, nullptr);

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

TEST_F(UnifiedPerfCountersProcessorTest, MultiHostTpuPerfCountersTest) {
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("perf_counters",
                                                           options_);
  ASSERT_NE(processor, nullptr);

  auto build_tpu_xspace = [](int64_t chip_id, absl::string_view kernel_name,
                             absl::string_view counter_name, uint64_t val) {
    XSpace space;
    XPlaneBuilder device_plane(space.add_planes());
    device_plane.SetName(std::string(tsl::profiler::kTpuPlanePrefix) + "0");
    device_plane.AddStatValue(
        *device_plane.GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kGlobalChipId)),
        chip_id);
    XLineBuilder line = device_plane.GetOrCreateLine(0);
    line.SetName(kernel_name);
    XEventBuilder event =
        line.AddEvent(*device_plane.GetOrCreateEventMetadata(counter_name));
    event.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kCounterValue)),
                       val);
    event.AddStatValue(
        *device_plane.GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kPerformanceCounterDescription)),
        "TPU TC Counter");
    event.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kPerformanceCounterSets)),
                       "TPU_TC_SET");
    return space;
  };

  std::string host0_path = file::JoinPath(session_dir_, "host0.xplane.pb");
  std::string host1_path = file::JoinPath(session_dir_, "host1.xplane.pb");
  CHECK_OK(WriteBinaryProto(
      host0_path, build_tpu_xspace(0, "fusion.1", "MXU_CYCLES", 255ULL)));
  CHECK_OK(WriteBinaryProto(
      host1_path, build_tpu_xspace(1, "fusion.2", "HBM_READ_BYTES", 4096ULL)));

  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create({host0_path, host1_path}, std::nullopt));

  ASSERT_OK(processor->ProcessSession(session_snapshot, options_));
  std::string output_str = processor->GetData();

  EXPECT_THAT(output_str, HasSubstr("host0"));
  EXPECT_THAT(output_str, HasSubstr("host1"));
  EXPECT_THAT(output_str, HasSubstr("mxu_cycles"));
  EXPECT_THAT(output_str, HasSubstr("hbm_read_bytes"));
  EXPECT_THAT(output_str, HasSubstr("0xff"));
  EXPECT_THAT(output_str, HasSubstr("0x1000"));
}

TEST_F(UnifiedPerfCountersProcessorTest, NoXSpaceTest) {
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("perf_counters",
                                                           options_);
  ASSERT_NE(processor, nullptr);

  testing::NiceMock<MockXprofSessionSnapshot> session_snapshot;
  EXPECT_CALL(session_snapshot, XSpaceSize()).WillRepeatedly(Return(0));

  EXPECT_THAT(processor->ProcessSession(session_snapshot, options_),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("No XSpace found in the session.")));
}

}  // namespace
}  // namespace xprof
