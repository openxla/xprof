#include "xprof/convert/streaming_trace_viewer_processor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"
#include "google/protobuf/arena.h"
#include "google/protobuf/util/message_differencer.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/lib/io/iterator.h"
#include "xla/tsl/lib/io/table.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_statistics.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/trace_view_options.h"
#include "xprof/convert/trace_viewer/lite_trace_events.h"

#include "xprof/convert/unified_session_snapshot.h"
#include "xprof/convert/xplane_to_trace_container.h"
#include "xprof/convert/xprof_thread_pool_executor.h"

namespace xprof {
using ::tensorflow::profiler::GetTraceViewOption;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::TraceViewOption;
using ::tensorflow::profiler::XEvent;
using ::tensorflow::profiler::XEventMetadata;
using ::tensorflow::profiler::XLine;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;
using ::tensorflow::profiler::XStat;
using ::tensorflow::profiler::XStatMetadata;
using ::testing::HasSubstr;
using ::testing::status::StatusIs;
using ::tsl::profiler::kHostThreadsPlaneName;

// Helper function to create a simple XSpace for testing
XSpace CreateTestXSpace(int num_events) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name(kHostThreadsPlaneName);

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
    session_dir_ = tsl::io::JoinPath(
        testing::TempDir(),
        absl::StrCat("session_",
                     testing::UnitTest::GetInstance()->current_test_info()->name()));
    TF_CHECK_OK(tsl::Env::Default()->RecursivelyCreateDir(session_dir_));
  }

  // Helper to create a SessionSnapshot by writing XSpaces to temp files
  absl::StatusOr<SessionSnapshot> CreateSnapshot(
      const absl::flat_hash_map<std::string, XSpace>& host_xspaces) {
    std::vector<std::string> xspace_paths;
    for (const auto& pair : host_xspaces) {
      const std::string& host_name = pair.first;
      const XSpace& xspace = pair.second;
      std::string xspace_path =
          tsl::io::JoinPath(session_dir_, host_name + ".xspace");
      TF_RETURN_IF_ERROR(
          xprof::WriteBinaryProto(xspace_path, xspace));
      xspace_paths.push_back(xspace_path);
    }
    std::sort(xspace_paths.begin(), xspace_paths.end());
    return SessionSnapshot::Create(std::move(xspace_paths),
                                   /*xspaces=*/std::nullopt);
  }

  std::string session_dir_;
};

namespace {

TEST_F(StreamingTraceViewerProcessorTest, GetTraceViewOptionValid) {
  ToolOptions options;
  options["start_time_ms"] = "100.5";
  options["end_time_ms"] = "200.0";
  options["resolution"] = "1000";
  options["event_name"] = "test_event";
  options["search_prefix"] = "prefix";
  options["duration_ms"] = "10.0";
  options["unique_id"] = "12345";
  options["search_metadata"] = false;

  TF_ASSERT_OK_AND_ASSIGN(TraceViewOption trace_option,
                          GetTraceViewOption(options));

  EXPECT_DOUBLE_EQ(trace_option.start_time_ms, 100.5);
  EXPECT_DOUBLE_EQ(trace_option.end_time_ms, 200.0);
  EXPECT_EQ(trace_option.resolution, 1000);
  EXPECT_EQ(trace_option.event_name, "test_event");
  EXPECT_EQ(trace_option.search_prefix, "prefix");
  EXPECT_DOUBLE_EQ(trace_option.duration_ms, 10.0);
  EXPECT_EQ(trace_option.unique_id, 12345);
  EXPECT_FALSE(trace_option.search_metadata);
}

TEST_F(StreamingTraceViewerProcessorTest,
       GetTraceViewOptionSearchMetadataTrue) {
  ToolOptions options;
  options["search_metadata"] = true;

  TF_ASSERT_OK_AND_ASSIGN(TraceViewOption trace_option,
                          GetTraceViewOption(options));

  EXPECT_TRUE(trace_option.search_metadata);
}

TEST_F(StreamingTraceViewerProcessorTest, GetTraceViewOptionSearchMetadata) {
  ToolOptions options;
  options["search_prefix"] = "prefix";
  options["search_metadata"] = true;

  TF_ASSERT_OK_AND_ASSIGN(TraceViewOption trace_option,
                          GetTraceViewOption(options));

  EXPECT_EQ(trace_option.search_prefix, "prefix");
  EXPECT_TRUE(trace_option.search_metadata);
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
  EXPECT_FALSE(trace_option.search_metadata);
}

TEST_F(StreamingTraceViewerProcessorTest, GetTraceViewOptionFloatFormatted) {
  ToolOptions options;
  options["start_time_ms"] = "100.5";
  options["end_time_ms"] = "200.0";
  options["resolution"] = "1000.000000";
  options["event_name"] = "test_event";
  options["search_prefix"] = "prefix";
  options["duration_ms"] = "10.0";
  options["unique_id"] = "12345.000000";

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

TEST_F(StreamingTraceViewerProcessorTest, GetTraceViewOptionInvalidNumber) {
  ToolOptions options;
  options["resolution"] = "not_a_number";
  EXPECT_THAT(GetTraceViewOption(options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("resolution must be a number")));
}

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

  TF_ASSERT_OK_AND_ASSIGN(std::string map_output_path_2,
                          processor.Map(snapshot, "host1", space));

  tsl::FileStatistics stat2;
  TF_ASSERT_OK(tsl::Env::Default()->Stat(map_output_path_2, &stat2));

  EXPECT_EQ(stat1.mtime_nsec, stat2.mtime_nsec);
}

TEST_F(StreamingTraceViewerProcessorTest, MapWithPath) {
  XSpace space = CreateTestXSpace(2);
  std::string host_name = "host1";
  std::string xspace_path =
      tsl::io::JoinPath(session_dir_, host_name + ".xplane.pb");
  TF_ASSERT_OK(tsl::WriteBinaryProto(tsl::Env::Default(), xspace_path, space));

  ToolOptions empty_options;
  StreamingTraceViewerProcessor processor(empty_options);

  TF_ASSERT_OK_AND_ASSIGN(
      SessionSnapshot snapshot_for_paths,
      SessionSnapshot::Create({xspace_path}, /*xspaces=*/std::nullopt));
  std::optional<std::string> trace_events_sstable_path =
      snapshot_for_paths.MakeHostDataFilePath(
          tensorflow::profiler::StoredDataType::TRACE_LEVELDB, host_name);
  ASSERT_TRUE(trace_events_sstable_path.has_value());
  EXPECT_TRUE(absl::IsNotFound(
      tsl::Env::Default()->FileExists(*trace_events_sstable_path)));

  TF_ASSERT_OK_AND_ASSIGN(std::string map_output, processor.Map(xspace_path));

  TF_EXPECT_OK(tsl::Env::Default()->FileExists(*trace_events_sstable_path));
  EXPECT_EQ(map_output, *trace_events_sstable_path);

  tsl::FileStatistics stat1;
  TF_ASSERT_OK(tsl::Env::Default()->Stat(map_output, &stat1));
  TF_ASSERT_OK_AND_ASSIGN(std::string map_output2, processor.Map(xspace_path));
  tsl::FileStatistics stat2;
  TF_ASSERT_OK(tsl::Env::Default()->Stat(map_output2, &stat2));
  EXPECT_EQ(stat1.mtime_nsec, stat2.mtime_nsec);
  EXPECT_EQ(map_output2, *trace_events_sstable_path);
}

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
  tool_options["end_time_ms"] = "2000.0";
  tool_options["resolution"] = "1";
  StreamingTraceViewerProcessor processor(tool_options);

  TF_ASSERT_OK_AND_ASSIGN(std::string map_output,
                          processor.Map(snapshot, "host1", space));

  TF_EXPECT_OK(processor.Reduce(snapshot, {map_output}));

  const std::string& json_output = processor.GetData();
  ASSERT_FALSE(json_output.empty());

  nlohmann::json parsed_json = nlohmann::json::parse(json_output);
  EXPECT_TRUE(parsed_json.contains("traceEvents"));

  const auto& trace_events = parsed_json["traceEvents"];
  bool session_run_event_found = false;
  for (const auto& event : trace_events) {
    if (event.value("name", "") == "SessionRun") {
      session_run_event_found = true;
      EXPECT_NEAR(event.value("ts", 0.0), 1200.0, 1.0);
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
  EXPECT_TRUE(nlohmann::json::parse(json_output).contains("traceEvents"));
}

TEST_F(StreamingTraceViewerProcessorTest, ReduceMultiHostStressTest) {
  const int kNumHosts = 10;
  absl::flat_hash_map<std::string, XSpace> host_xspaces;
  std::vector<std::string> host_names;

  for (int i = 0; i < kNumHosts; ++i) {
    std::string host_name = absl::StrCat("host", i);
    host_names.push_back(host_name);

    XSpace space = CreateSingleEventXSpace();
    XEventMetadata& metadata =
        (*space.mutable_planes(0)
              ->mutable_event_metadata())[static_cast<int64_t>(
            tsl::profiler::HostEventType::kSessionRun)];
    metadata.set_name(absl::StrCat("EventFrom", host_name));

    host_xspaces[host_name] = std::move(space);
  }

  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));

  ToolOptions tool_options;
  tool_options["end_time_ms"] = "5000.0";
  tool_options["resolution"] = "1";
  StreamingTraceViewerProcessor processor(tool_options);

  std::vector<std::string> map_outputs;
  for (const auto& host_name : host_names) {
    TF_ASSERT_OK_AND_ASSIGN(
        std::string path,
        processor.Map(snapshot, host_name, host_xspaces[host_name]));
    map_outputs.push_back(path);
  }

  TF_EXPECT_OK(processor.Reduce(snapshot, map_outputs));

  const std::string& json_output = processor.GetData();
  ASSERT_FALSE(json_output.empty());

  nlohmann::json parsed_json = nlohmann::json::parse(json_output);
  const auto& trace_events = parsed_json["traceEvents"];

  for (const auto& host_name : host_names) {
    std::string expected_event_name = absl::StrCat("EventFrom", host_name);
    bool found = false;
    for (const auto& event : trace_events) {
      if (event.value("name", "") == expected_event_name) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Could not find event from " << host_name;
  }
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

  std::optional<std::string> metadata_path = snapshot.MakeHostDataFilePath(
      tensorflow::profiler::StoredDataType::TRACE_EVENTS_METADATA_LEVELDB,
      "host1");
  if (metadata_path.has_value()) {
    TF_ASSERT_OK(tsl::Env::Default()->DeleteFile(*metadata_path));
  }

  EXPECT_THAT(
      processor.Reduce(snapshot, {map_output}),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("No hosts with valid trace data")));
}

TEST_F(StreamingTraceViewerProcessorTest, ReduceWithSearchMetadata) {
  XSpace space = CreateTestXSpace(2);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {{"host1", space}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));

  ToolOptions tool_options;
  tool_options["search_prefix"] = "Sess";
  tool_options["search_metadata"] = true;
  StreamingTraceViewerProcessor processor(tool_options);

  TF_ASSERT_OK_AND_ASSIGN(std::string map_output,
                          processor.Map(snapshot, "host1", space));
  TF_EXPECT_OK(processor.Reduce(snapshot, {map_output}));

  const std::string& json_output = processor.GetData();
  ASSERT_FALSE(json_output.empty());
  EXPECT_TRUE(nlohmann::json::parse(json_output).contains("traceEvents"));
}

TEST_F(StreamingTraceViewerProcessorTest, ReduceWithSearchPrefix) {
  XSpace space = CreateTestXSpace(2);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {{"host1", space}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));

  ToolOptions tool_options;
  tool_options["search_prefix"] = "Sess";
  StreamingTraceViewerProcessor processor(tool_options);

  TF_ASSERT_OK_AND_ASSIGN(std::string map_output,
                          processor.Map(snapshot, "host1", space));
  TF_EXPECT_OK(processor.Reduce(snapshot, {map_output}));

  const std::string& json_output = processor.GetData();
  ASSERT_FALSE(json_output.empty());
  EXPECT_TRUE(nlohmann::json::parse(json_output).contains("traceEvents"));
}

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

TEST_F(StreamingTraceViewerProcessorTest, ReduceWithEventName) {
  XSpace space = CreateTestXSpace(2);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {{"host1", space}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));

  ToolOptions tool_options;
  tool_options["event_name"] = "SessionRun";
  tool_options["start_time_ms"] = "1.2";
  tool_options["duration_ms"] = "0.05";
  tool_options["unique_id"] = "0";
  StreamingTraceViewerProcessor processor(tool_options);

  TF_ASSERT_OK_AND_ASSIGN(std::string map_output,
                          processor.Map(snapshot, "host1", space));
  TF_EXPECT_OK(processor.Reduce(snapshot, {map_output}));

  const std::string& json_output = processor.GetData();
  ASSERT_FALSE(json_output.empty());

  nlohmann::json parsed_json = nlohmann::json::parse(json_output);
  ASSERT_TRUE(parsed_json.contains("traceEvents"));

  const auto& trace_events = parsed_json["traceEvents"];
  int complete_event_count = 0;
  bool session_run_found = false;

  for (const auto& event : trace_events) {
    if (event.value("ph", "") == "X") {
      complete_event_count++;
      if (event.value("name", "") == "SessionRun") {
        session_run_found = true;
      }
    }
  }

  EXPECT_EQ(complete_event_count, 1);
  EXPECT_TRUE(session_run_found);
}

TEST_F(StreamingTraceViewerProcessorTest, ProcessSessionWithSearchPrefix) {
  XSpace space = CreateTestXSpace(2);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {{"host1", space}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));

  ToolOptions tool_options;
  tool_options["search_prefix"] = "Sess";
  StreamingTraceViewerProcessor processor(tool_options);

  TF_EXPECT_OK(processor.ProcessSession(snapshot, tool_options));

  const std::string& json_output = processor.GetData();
  ASSERT_FALSE(json_output.empty());
  EXPECT_TRUE(nlohmann::json::parse(json_output).contains("traceEvents"));
}

TEST_F(StreamingTraceViewerProcessorTest, ShouldUseWorkerService) {
  StreamingTraceViewerProcessor processor({});

  // Verify false for a single-host snapshot.
  absl::flat_hash_map<std::string, XSpace> single_host = {{"host1", XSpace()}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot1,
                           CreateSnapshot(single_host));
  EXPECT_FALSE(processor.ShouldUseWorkerService(snapshot1, {}));

  // Verify true for a multi-host snapshot.
  absl::flat_hash_map<std::string, XSpace> multi_host = {
      {"host1", XSpace()}, {"host2", XSpace()}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot2,
                           CreateSnapshot(multi_host));
  EXPECT_TRUE(processor.ShouldUseWorkerService(snapshot2, {}));
}

TEST_F(StreamingTraceViewerProcessorTest, ReduceWithCorruptedMapOutput) {
  XSpace space = CreateTestXSpace(1);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {{"host1", space}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                           CreateSnapshot(host_xspaces));

  StreamingTraceViewerProcessor processor({});
  TF_ASSERT_OK_AND_ASSIGN(std::string map_output,
                          processor.Map(snapshot, "host1", space));

  TF_ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), map_output,
                                      "this is not a valid sstable file"));

  EXPECT_FALSE(processor.Reduce(snapshot, {map_output}).ok());
}

TEST_F(StreamingTraceViewerProcessorTest, MapWithEmptyXSpace) {
  // Use helper to avoid segfault in ProcessMegascaleDcn.
  XSpace empty_space = CreateTestXSpace(0);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {
      {"host1", empty_space}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                           CreateSnapshot(host_xspaces));

  StreamingTraceViewerProcessor processor({});

  TF_ASSERT_OK_AND_ASSIGN(std::string map_output,
                          processor.Map(snapshot, "host1", empty_space));
  TF_EXPECT_OK(processor.Reduce(snapshot, {map_output}));

  const std::string& json_output = processor.GetData();
  EXPECT_FALSE(json_output.empty());
  EXPECT_TRUE(nlohmann::json::parse(json_output).contains("traceEvents"));
}

TEST_F(StreamingTraceViewerProcessorTest, ReduceSingleHostPbFormat) {
  XSpace space = CreateTestXSpace(2);
  absl::flat_hash_map<std::string, XSpace> host_xspaces = {{"host1", space}};
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot snapshot,
                          CreateSnapshot(host_xspaces));

  ToolOptions tool_options;
  tool_options["format"] = "pb";
  StreamingTraceViewerProcessor processor(tool_options);

  TF_ASSERT_OK_AND_ASSIGN(std::string map_output,
                          processor.Map(snapshot, "host1", space));

  TF_EXPECT_OK(processor.Reduce(snapshot, {map_output}));

  const std::string& pb_output = processor.GetData();
  ASSERT_FALSE(pb_output.empty());

  // Verify it is not valid JSON
  EXPECT_FALSE(nlohmann::json::accept(pb_output));
}

class StringBackedWritableFile : public tsl::WritableFile {
 public:
  explicit StringBackedWritableFile(std::string* external_buffer)
      : buffer_(external_buffer) {}
  ~StringBackedWritableFile() override = default;

  absl::Status Append(absl::string_view data) override {
    buffer_->append(data.data(), data.size());
    return absl::OkStatus();
  }
  absl::Status Close() override { return absl::OkStatus(); }
  absl::Status Flush() override { return absl::OkStatus(); }
  absl::Status Name(absl::string_view* result) const override {
    return absl::OkStatus();
  }
  absl::Status Sync() override { return absl::OkStatus(); }
  absl::Status Tell(int64_t* position) override {
    *position = buffer_->size();
    return absl::OkStatus();
  }

 private:
  std::string* buffer_;
};

std::optional<tensorflow::profiler::Trace> ExtractTraceFromTable(
    tsl::table::Table* table) {
  std::unique_ptr<tsl::table::Iterator> it(table->NewIterator());

  it->Seek("/trace");
  if (it->Valid() && it->key() == "/trace") {
    tensorflow::profiler::Trace trace;
    if (trace.ParseFromString(std::string(it->value()))) {
      return trace;
    }
  }

  it->Seek("trace");
  if (it->Valid() && it->key() == "trace") {
    tensorflow::profiler::Trace trace;
    if (trace.ParseFromString(std::string(it->value()))) {
      return trace;
    }
  }

  return std::nullopt;
}

// Helper to compare two LevelDB tables logically using Iterators
void CompareLevelDbTables(
    const std::string& legacy_file, const std::string& lite_file,
    bool compare_trace,
    const std::function<void(absl::string_view key,
                             absl::string_view legacy_val,
                             absl::string_view lite_val)>& val_assert_fn) {
  tsl::Env* env = tsl::Env::Default();

  // Open Legacy Table
  tsl::table::Table* legacy_table = nullptr;
  std::unique_ptr<tsl::RandomAccessFile> legacy_raf;
  uint64_t legacy_size;
  TF_CHECK_OK(env->GetFileSize(legacy_file, &legacy_size));
  TF_CHECK_OK(env->NewRandomAccessFile(legacy_file, &legacy_raf));
  TF_CHECK_OK(tsl::table::Table::Open(tsl::table::Options(), legacy_raf.get(),
                                      legacy_size, &legacy_table));
  std::unique_ptr<tsl::table::Table> legacy_table_deleter(legacy_table);

  // Open Lite Table
  tsl::table::Table* lite_table = nullptr;
  std::unique_ptr<tsl::RandomAccessFile> lite_raf;
  uint64_t lite_size;
  TF_CHECK_OK(env->GetFileSize(lite_file, &lite_size));
  TF_CHECK_OK(env->NewRandomAccessFile(lite_file, &lite_raf));
  TF_CHECK_OK(tsl::table::Table::Open(tsl::table::Options(), lite_raf.get(),
                                      lite_size, &lite_table));
  std::unique_ptr<tsl::table::Table> lite_table_deleter(lite_table);

  // Instantiate Iterators
  std::unique_ptr<tsl::table::Iterator> legacy_it(legacy_table->NewIterator());
  std::unique_ptr<tsl::table::Iterator> lite_it(lite_table->NewIterator());

  legacy_it->SeekToFirst();
  lite_it->SeekToFirst();

  // Side-by-side record traversal
  while (legacy_it->Valid() && lite_it->Valid()) {
    // Skip the "trace" metadata key during the loop to avoid block layout
    // differences
    if (legacy_it->key() == "/trace") {
      legacy_it->Next();
      continue;
    }
    if (lite_it->key() == "trace") {
      lite_it->Next();
      continue;
    }

    // Assert that keys match lexicographically
    ASSERT_EQ(legacy_it->key(), lite_it->key());
    // Perform logical value comparisons
    val_assert_fn(legacy_it->key(), legacy_it->value(), lite_it->value());

    legacy_it->Next();
    lite_it->Next();
  }

  if (lite_it->Valid() && lite_it->key() == "trace") {
    lite_it->Next();
  }

  // Verify both iterators reached the end symmetrically
  EXPECT_FALSE(legacy_it->Valid());
  EXPECT_FALSE(lite_it->Valid());

  // Deep logical Protobuf Comparison
  if (compare_trace) {
    auto legacy_trace_opt = ExtractTraceFromTable(legacy_table);
    auto lite_trace_opt = ExtractTraceFromTable(lite_table);

    ASSERT_TRUE(legacy_trace_opt.has_value())
        << "Legacy table missing Trace metadata!";
    ASSERT_TRUE(lite_trace_opt.has_value())
        << "Lite table missing Trace metadata!";

    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(*lite_trace_opt,
                                                         *legacy_trace_opt))
        << "Logical unified Trace metadata mismatch!";
  }
}

class LiteTraceEventsParityTest : public ::testing::TestWithParam<std::string> {
};

TEST_P(LiteTraceEventsParityTest, TestLiteTraceEventsByLevel) {
  std::string xplane_path =
      devtools_build::GetDataDependencyFilepath(GetParam());
  XSpace space;
  TF_CHECK_OK(xprof::ReadBinaryProto(xplane_path, &space));

  std::vector<std::vector<const tensorflow::profiler::TraceEvent*>>
      legacy_events_by_level;
  std::vector<std::vector<const tensorflow::profiler::TraceEventLite*>>
      lite_events_by_level;
  tensorflow::profiler::TraceEventsContainer legacy_container;
  tensorflow::profiler::TraceEventLiteContainer lite_container;

  tensorflow::profiler::XprofThreadPoolExecutor executor("ParityTestPool", 2);
  executor.Execute([&]() {
    tensorflow::profiler::ConvertXSpaceToTraceEventsContainer(
        "localhost", space, &legacy_container);
    legacy_events_by_level = legacy_container.GetTraceEventsByLevel();
  });
  executor.Execute([&]() {
    tensorflow::profiler::ConvertXSpaceToLiteTraceEventsContainer(
        "localhost", space, &lite_container);
    lite_events_by_level =
        tensorflow::profiler::LiteTraceEventsByLevel(&lite_container);
  });
  executor.JoinAll();

  // 1. Verify total number of events matches perfectly using NumEvents()
  EXPECT_EQ(legacy_container.NumEvents(),
            tensorflow::profiler::NumEvents(lite_container));

  // 2. Verify number of visibility zoom levels matches
  EXPECT_EQ(legacy_events_by_level.size(), lite_events_by_level.size());

  // 3. Verify number of events at each level matches
  for (size_t i = 0;
       i < std::min(legacy_events_by_level.size(), lite_events_by_level.size());
       ++i) {
    if (legacy_events_by_level[i].size() != lite_events_by_level[i].size()) {
      LOG(ERROR) << "Level " << i << " MISMATCH DETAILS:";
      LOG(ERROR) << "Legacy size: " << legacy_events_by_level[i].size()
                 << ", Lite size: " << lite_events_by_level[i].size();
    }

    EXPECT_EQ(legacy_events_by_level[i].size(), lite_events_by_level[i].size())
        << "Mismatch at level " << i;
  }
}

TEST_P(LiteTraceEventsParityTest, TestEndToEndTableParity) {
  std::string xplane_path =
      devtools_build::GetDataDependencyFilepath(GetParam());
  tensorflow::profiler::XSpace space;
  TF_CHECK_OK(xprof::ReadBinaryProto(xplane_path, &space));

  // 1. Generate local temporary file paths on disk
  std::string legacy_fast_file, legacy_metadata_file, legacy_trie_file;
  std::string lite_fast_file, lite_metadata_file, lite_trie_file;

  tsl::Env* env = tsl::Env::Default();
  CHECK(env->LocalTempFilename(&legacy_fast_file));
  CHECK(env->LocalTempFilename(&legacy_metadata_file));
  CHECK(env->LocalTempFilename(&legacy_trie_file));
  CHECK(env->LocalTempFilename(&lite_fast_file));
  CHECK(env->LocalTempFilename(&lite_metadata_file));
  CHECK(env->LocalTempFilename(&lite_trie_file));

  // 2. Instantiate standard production WritableFile instances directly on disk!
  std::unique_ptr<tsl::WritableFile> legacy_fast;
  std::unique_ptr<tsl::WritableFile> legacy_metadata;
  std::unique_ptr<tsl::WritableFile> legacy_trie;
  TF_CHECK_OK(env->NewWritableFile(legacy_fast_file, &legacy_fast));
  TF_CHECK_OK(env->NewWritableFile(legacy_metadata_file, &legacy_metadata));
  TF_CHECK_OK(env->NewWritableFile(legacy_trie_file, &legacy_trie));

  std::unique_ptr<tsl::WritableFile> lite_fast;
  std::unique_ptr<tsl::WritableFile> lite_metadata;
  std::unique_ptr<tsl::WritableFile> lite_trie;
  TF_CHECK_OK(env->NewWritableFile(lite_fast_file, &lite_fast));
  TF_CHECK_OK(env->NewWritableFile(lite_metadata_file, &lite_metadata));
  TF_CHECK_OK(env->NewWritableFile(lite_trie_file, &lite_trie));

  // 3. Serialize Legacy sequentially directly to disk
  {
    tensorflow::profiler::TraceEventsContainer legacy_container;
    tensorflow::profiler::ConvertXSpaceToTraceEventsContainer(
        "localhost", space, &legacy_container);
    TF_CHECK_OK(legacy_container.StoreAsLevelDbTables(
        std::move(legacy_fast), std::move(legacy_metadata),
        std::move(legacy_trie)));
  }

  // 4. Serialize Lite sequentially directly to disk
  {
    tensorflow::profiler::TraceEventLiteContainer lite_container;
    tensorflow::profiler::ConvertXSpaceToLiteTraceEventsContainer(
        "localhost", space, &lite_container);

    auto converter_fn =
        [&](const tsl::profiler::XEventVisitor& event_visitor,
            const tensorflow::profiler::TraceEventLite& lite_event,
            absl::flat_hash_map<uint64_t, std::string>* local_name_table,
            tensorflow::profiler::TraceEvent* full_event,
            google::protobuf::Arena* arena) -> absl::Status {
      tensorflow::profiler::ConvertLiteTraceEventToFullTraceEvent(
          event_visitor, lite_event, lite_container, local_name_table,
          full_event, arena);
      return absl::OkStatus();
    };

    TF_CHECK_OK(tensorflow::profiler::StoreLiteEventsAsLevelDbTables(
        &lite_container, converter_fn, lite_fast, lite_metadata,
        lite_trie));
  }

  // 5. Verify the Fast Event Tables logically (comparing serialized events and
  // Trace metadata)
  CompareLevelDbTables(legacy_fast_file, lite_fast_file, /*compare_trace=*/true,
                       [](absl::string_view key, absl::string_view legacy_val,
                          absl::string_view lite_val) {
                         // Symmetrical event record assertion
                         EXPECT_EQ(legacy_val, lite_val);
                       });

  // 6. Verify the Metadata Tables logically (comparing stat arguments)
  CompareLevelDbTables(
      legacy_metadata_file, lite_metadata_file, /*compare_trace=*/false,
      [](absl::string_view key, absl::string_view legacy_val,
         absl::string_view lite_val) { EXPECT_EQ(legacy_val, lite_val); });

  // 7. Verify the Prefix Trie Indexes logically
  CompareLevelDbTables(legacy_trie_file, lite_trie_file,
                       /*compare_trace=*/false,
                       [](absl::string_view key, absl::string_view legacy_val,
                          absl::string_view lite_val) {
                         // Symmetrical prefix trie index assertion
                         EXPECT_EQ(legacy_val, lite_val);
                       });

  // 8. Cleanup temp files on disk
  env->DeleteFile(legacy_fast_file).IgnoreError();
  env->DeleteFile(legacy_metadata_file).IgnoreError();
  env->DeleteFile(legacy_trie_file).IgnoreError();
  env->DeleteFile(lite_fast_file).IgnoreError();
  env->DeleteFile(lite_metadata_file).IgnoreError();
  env->DeleteFile(lite_trie_file).IgnoreError();
}

INSTANTIATE_TEST_SUITE_P(
    TraceViewerParity, LiteTraceEventsParityTest,
    ::testing::Values("google3/third_party/xprof/convert/test_xplanes/"
                      "gpu_training_2.xplane.pb"));

}  // namespace



}  // namespace xprof
