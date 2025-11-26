#include "xprof/convert/streaming_trace_viewer_processor.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "file/base/path.h"
#include "file/util/temp_path.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_statistics.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"

namespace xprof {
using internal::GetTraceViewOption;
using internal::TraceViewOption;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XEvent;
using ::tensorflow::profiler::XEventMetadata;
using ::tensorflow::profiler::XLine;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;
using ::tensorflow::profiler::XStat;
using ::tensorflow::profiler::XStatMetadata;
using ::testing::HasSubstr;
using ::testing::status::StatusIs;
using ::tsl::profiler::kHostThreadsPlaneName;  // Standard plane name


// Helper function to create a simple XSpace for testing
XSpace CreateTestXSpace(int num_events) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name(kHostThreadsPlaneName);  // Use standard plane name

  // Setup Event Metadata
  int64_t event1_id =
      static_cast<int64_t>(tsl::profiler::HostEventType::kTraceContext);
  XEventMetadata& event1_metadata =
      (*plane->mutable_event_metadata())[event1_id];
  event1_metadata.set_id(event1_id);
  event1_metadata.set_name(
      GetHostEventTypeStr(tsl::profiler::HostEventType::kTraceContext));

  int64_t event2_id =
      static_cast<int64_t>(tsl::profiler::HostEventType::kSessionRun);
  XEventMetadata& event2_metadata =
      (*plane->mutable_event_metadata())[event2_id];
  event2_metadata.set_id(event2_id);
  event2_metadata.set_name(
      GetHostEventTypeStr(tsl::profiler::HostEventType::kSessionRun));

  // Setup Stat Metadata
  const int64_t kGroupIdType =
      static_cast<int64_t>(tsl::profiler::StatType::kGroupId);
  XStatMetadata& group_id_metadata =
      (*plane->mutable_stat_metadata())[kGroupIdType];
  group_id_metadata.set_id(kGroupIdType);
  group_id_metadata.set_name(GetStatTypeStr(tsl::profiler::StatType::kGroupId));

  XLine* line = plane->add_lines();
  line->set_id(1);
  line->set_name("Test Line");

  if (num_events > 0) {
    XEvent* event = line->add_events();
    event->set_metadata_id(event1_id);
    event->set_offset_ps(1000000000);
    event->set_duration_ps(100000000);
    XStat* stat = event->add_stats();
    stat->set_metadata_id(kGroupIdType);
    stat->set_int64_value(123);
  }
  if (num_events > 1) {
    XEvent* event2 = line->add_events();
    event2->set_metadata_id(event2_id);
    event2->set_offset_ps(1200000000);
    event2->set_duration_ps(50000000);
    XStat* stat2 = event2->add_stats();
    stat2->set_metadata_id(kGroupIdType);
    stat2->set_int64_value(456);
  }
  return space;
}

XSpace CreateSingleEventXSpace() {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name(kHostThreadsPlaneName);

  int64_t event2_id =
      static_cast<int64_t>(tsl::profiler::HostEventType::kSessionRun);
  XEventMetadata& event2_metadata =
      (*plane->mutable_event_metadata())[event2_id];
  event2_metadata.set_id(event2_id);
  event2_metadata.set_name(
      GetHostEventTypeStr(tsl::profiler::HostEventType::kSessionRun));

  XLine* line = plane->add_lines();
  line->set_id(1);
  line->set_name("Test Line");

  XEvent* event2 = line->add_events();
  event2->set_metadata_id(event2_id);
  event2->set_offset_ps(1200000000);
  event2->set_duration_ps(50000000);

  return space;
}

class StreamingTraceViewerProcessorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a temporary directory for test files.
    temp_path_ = std::make_unique<TempPath>(TempPath::Local);
    session_dir_ = file::JoinPath(temp_path_->path(), "session");
    TF_CHECK_OK(tsl::Env::Default()->CreateDir(session_dir_));
  }

  // Helper to create a SessionSnapshot by writing XSpaces to temp files
  absl::StatusOr<SessionSnapshot> CreateSnapshot(
      const absl::flat_hash_map<std::string, XSpace>& host_xspaces) {
    std::vector<std::string> xspace_paths;
    for (const auto& pair : host_xspaces) {
      const std::string& host_name = pair.first;
      const XSpace& xspace = pair.second;
      std::string xspace_path =
          file::JoinPath(session_dir_, host_name + ".xspace");
      TF_RETURN_IF_ERROR(
          tsl::WriteBinaryProto(tsl::Env::Default(), xspace_path, xspace));
      xspace_paths.push_back(xspace_path);
    }
    std::sort(xspace_paths.begin(), xspace_paths.end());
    return SessionSnapshot::Create(std::move(xspace_paths),
                                   /*xspaces=*/std::nullopt);
  }

  std::string session_dir_;
  std::unique_ptr<TempPath> temp_path_;
};

namespace {

// GET_TRACE_VIEW_OPTION TESTS
TEST_F(StreamingTraceViewerProcessorTest, GetTraceViewOptionValid) {
  ToolOptions options;
  options["start_time_ms"] = "100.5";
  options["end_time_ms"] = "200.0";
  options["resolution"] = "1000";
  options["event_name"] = "test_event";
  options["search_prefix"] = "prefix";
  options["duration_ms"] = "10.0";
  options["unique_id"] = "12345";

  TF_ASSERT_OK_AND_ASSIGN(TraceViewOption trace_option,
                          GetTraceViewOption(options));

  EXPECT_DOUBLE_EQ(trace_option.start_time_ms, 100.5);
  EXPECT_DOUBLE_EQ(trace_option.end_time_ms, 200.0);
  EXPECT_EQ(trace_option.resolution, 1000);
  EXPECT_EQ(trace_option.event_name, "test_event");
  EXPECT_EQ(trace_option.search_prefix, "prefix");
  EXPECT_DOUBLE_EQ(trace_option.duration_ms, 10.0);
  EXPECT_EQ(trace_option.unique_id, 12345);
}

TEST_F(StreamingTraceViewerProcessorTest, GetTraceViewOptionDefaults) {
  ToolOptions options;
  TF_ASSERT_OK_AND_ASSIGN(TraceViewOption trace_option,
                          GetTraceViewOption(options));

  EXPECT_DOUBLE_EQ(trace_option.start_time_ms, 0.0);
  EXPECT_DOUBLE_EQ(trace_option.end_time_ms, 0.0);
  EXPECT_EQ(trace_option.resolution, 0);
  EXPECT_EQ(trace_option.event_name, "");
  EXPECT_EQ(trace_option.search_prefix, "");
  EXPECT_DOUBLE_EQ(trace_option.duration_ms, 0.0);
  EXPECT_EQ(trace_option.unique_id, 0);
}

TEST_F(StreamingTraceViewerProcessorTest, GetTraceViewOptionInvalidNumber) {
  ToolOptions options;
  options["resolution"] = "not_a_number";
  EXPECT_THAT(GetTraceViewOption(options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("wrong arguments")));
}

// MAP TESTS
TEST_F(StreamingTraceViewerProcessorTest, MapCreatesFiles) {
  XSpace space = CreateTestXSpace(2);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {{"host1", space}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));

  ToolOptions empty_options;
  StreamingTraceViewerProcessor processor(empty_options);

  TF_ASSERT_OK_AND_ASSIGN(std::string map_output,
                          processor.Map(snapshot, "host1", space));
}

TEST_F(StreamingTraceViewerProcessorTest, MapIsIdempotent) {
  XSpace space = CreateTestXSpace(2);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {{"host1", space}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));

  ToolOptions empty_options;
  StreamingTraceViewerProcessor processor(empty_options);

  TF_ASSERT_OK_AND_ASSIGN(std::string map_output_path,
                          processor.Map(snapshot, "host1", space));

  tsl::FileStatistics stat1;
  TF_ASSERT_OK(tsl::Env::Default()->Stat(map_output_path, &stat1));

  // Call Map again.
  TF_ASSERT_OK_AND_ASSIGN(std::string map_output_path_2,
                          processor.Map(snapshot, "host1", space));

  tsl::FileStatistics stat2;
  TF_ASSERT_OK(tsl::Env::Default()->Stat(map_output_path_2, &stat2));

  // Check that the file was not modified.
  EXPECT_EQ(stat1.mtime_nsec, stat2.mtime_nsec);
}

// REDUCE TESTS
TEST_F(StreamingTraceViewerProcessorTest, ReduceEmptyMapOutput) {
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {
      {"host1", CreateTestXSpace(0)}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));
  ToolOptions empty_options;
  StreamingTraceViewerProcessor processor(empty_options);
  EXPECT_THAT(processor.Reduce(snapshot, {}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("map_output_files cannot be empty")));
}

TEST_F(StreamingTraceViewerProcessorTest, ReduceSingleHost) {
  XSpace space = CreateTestXSpace(2);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {{"host1", space}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));

  ToolOptions tool_options;
  tool_options["end_time_ms"] = "2000.0";  // 2s
  tool_options["resolution"] = "1";
  StreamingTraceViewerProcessor processor(tool_options);

  TF_ASSERT_OK_AND_ASSIGN(std::string map_output,
                          processor.Map(snapshot, "host1", space));

  TF_EXPECT_OK(processor.Reduce(snapshot, {map_output}));

  const std::string& json_output = processor.GetData();
  ASSERT_FALSE(json_output.empty());

  nlohmann::json parsed_json =
      nlohmann::json::parse(json_output, nullptr, false);
  ASSERT_FALSE(parsed_json.is_discarded());

  // Check for top-level keys.
  EXPECT_TRUE(parsed_json.contains("traceEvents"));
  EXPECT_TRUE(parsed_json.contains("displayTimeUnit"));

  const auto& trace_events = parsed_json["traceEvents"];
  EXPECT_TRUE(trace_events.is_array());
  // We expect more than 2 events because of metadata events.
  EXPECT_GT(trace_events.size(), 2);

  // Find the "SessionRun" event and check its properties.
  bool session_run_event_found = false;
  for (const auto& event : trace_events) {
    if (event.value("name", "") == "SessionRun") {
      session_run_event_found = true;
      EXPECT_EQ(event.value("ph", ""), "X");
      // Timestamps are in microseconds.
      EXPECT_NEAR(event.value("ts", 0.0), 1200.0, 1.0);
      EXPECT_NEAR(event.value("dur", 0.0), 50.0, 1.0);
      break;
    }
  }
  EXPECT_TRUE(session_run_event_found);
}


TEST_F(StreamingTraceViewerProcessorTest, ReduceMultiHost) {
  XSpace space1 = CreateTestXSpace(1);
  XSpace space2 = CreateTestXSpace(1);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {{"host1", space1},
                                                           {"host2", space2}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));

  ToolOptions tool_options;
  tool_options["end_time_ms"] = "2000.0";
  tool_options["resolution"] = "1";
  StreamingTraceViewerProcessor processor(tool_options);

  TF_ASSERT_OK_AND_ASSIGN(std::string map_output1,
                          processor.Map(snapshot, "host1", space1));
  TF_ASSERT_OK_AND_ASSIGN(std::string map_output2,
                          processor.Map(snapshot, "host2", space2));

  TF_EXPECT_OK(processor.Reduce(snapshot, {map_output1, map_output2}));

  const std::string& json_output = processor.GetData();
  ASSERT_FALSE(json_output.empty());

  nlohmann::json parsed_json =
      nlohmann::json::parse(json_output, nullptr, false);
  ASSERT_FALSE(parsed_json.is_discarded());
  EXPECT_TRUE(parsed_json.contains("traceEvents"));
}

TEST_F(StreamingTraceViewerProcessorTest, ReduceWithMissingFiles) {
  XSpace space = CreateTestXSpace(2);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {{"host1", space}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));

  ToolOptions tool_options;
  StreamingTraceViewerProcessor processor(tool_options);

  TF_ASSERT_OK_AND_ASSIGN(std::string map_output,
                          processor.Map(snapshot, "host1", space));

  // Delete the metadata file to simulate a missing file scenario.
  auto metadata_path = snapshot.MakeHostDataFilePath(
      tensorflow::profiler::StoredDataType::TRACE_EVENTS_METADATA_LEVELDB,
      "host1");
  if (metadata_path.has_value()) {
    TF_ASSERT_OK(tsl::Env::Default()->DeleteFile(*metadata_path));
  }

  EXPECT_THAT(
      processor.Reduce(snapshot, {map_output}),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Could not find metadata file for host: host1")));
}

TEST_F(StreamingTraceViewerProcessorTest, ReduceWithSearchPrefix) {
  XSpace space = CreateTestXSpace(2);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {{"host1", space}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));

  ToolOptions tool_options;
  tool_options["search_prefix"] = "Sess";  // Updated prefix
  StreamingTraceViewerProcessor processor(tool_options);

  TF_ASSERT_OK_AND_ASSIGN(std::string map_output,
                          processor.Map(snapshot, "host1", space));
  TF_EXPECT_OK(processor.Reduce(snapshot, {map_output}));

  const std::string& json_output = processor.GetData();
  ASSERT_FALSE(json_output.empty());

  nlohmann::json parsed_json =
      nlohmann::json::parse(json_output, nullptr, false);
  ASSERT_FALSE(parsed_json.is_discarded());
  EXPECT_TRUE(parsed_json.contains("traceEvents"));
}

// PROCESS SESSION TEST
TEST_F(StreamingTraceViewerProcessorTest, ProcessSessionEndToEnd) {
  XSpace space1 = CreateTestXSpace(1);
  XSpace space2 = CreateTestXSpace(1);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {{"host1", space1},
                                                           {"host2", space2}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));

  ToolOptions tool_options;
  tool_options["end_time_ms"] = "2000.0";
  tool_options["resolution"] = "1";
  StreamingTraceViewerProcessor processor(tool_options);

  TF_EXPECT_OK(processor.ProcessSession(snapshot, tool_options));
}

TEST_F(StreamingTraceViewerProcessorTest, ProcessSessionSingleHost) {
  XSpace space = CreateTestXSpace(1);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {{"host1", space}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));

  ToolOptions tool_options;
  tool_options["end_time_ms"] = "2000.0";
  tool_options["resolution"] = "1";
  StreamingTraceViewerProcessor processor(tool_options);

  TF_EXPECT_OK(processor.ProcessSession(snapshot, tool_options));
}

}  // namespace
}  // namespace xprof
