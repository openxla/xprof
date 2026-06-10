#include "xprof/convert/xplane_to_tools_data.h"

#include <cstdint>
#include <optional>
#include <string>

#include "file/base/filesystem.h"
#include "file/base/options.h"
#include "file/base/path.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/statusor.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::tsl::profiler::kHostThreadsPlaneName;

XSpace CreateTestXSpace() {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name(kHostThreadsPlaneName);

  // Setup Event Metadata
  int64_t event_id =
      static_cast<int64_t>(tsl::profiler::HostEventType::kTraceContext);
  XEventMetadata& event_metadata = (*plane->mutable_event_metadata())[event_id];
  event_metadata.set_id(event_id);
  event_metadata.set_name(
      GetHostEventTypeStr(tsl::profiler::HostEventType::kTraceContext));

  // Setup Stat Metadata
  const int64_t kGroupIdType =
      static_cast<int64_t>(tsl::profiler::StatType::kGroupId);
  XStatMetadata& group_id_metadata =
      (*plane->mutable_stat_metadata())[kGroupIdType];
  group_id_metadata.set_id(kGroupIdType);
  group_id_metadata.set_name("MetadataWithFoo");

  XLine* line = plane->add_lines();
  line->set_id(1);
  line->set_name("Test Line");

  XEvent* event = line->add_events();
  event->set_metadata_id(event_id);
  event->set_offset_ps(1000000000);
  event->set_duration_ps(100000000);
  XStat* stat = event->add_stats();
  stat->set_metadata_id(kGroupIdType);
  stat->set_int64_value(123);

  return space;
}

TEST(XplaneToToolsDataTest, TraceViewerSearchMetadataOption) {
  // Arrange
  std::string session_dir =
      file::JoinPath(testing::TempDir(), "xplane_to_tools_data_test");
  ASSERT_OK(file::CreateDir(session_dir, file::Defaults()));

  std::string xspace_path = file::JoinPath(session_dir, "test_host.xplane.pb");
  XSpace space = CreateTestXSpace();
  ASSERT_OK(xprof::WriteBinaryProto(xspace_path, space));

  absl::StatusOr<SessionSnapshot> session_snapshot =
      SessionSnapshot::Create({xspace_path}, std::nullopt);
  ASSERT_OK(session_snapshot);

  ToolOptions options;
  // Searching for "TraceContext" should match the event.
  // With search_metadata=true, it should load and return the metadata (e.g.
  // MetadataWithFoo)
  options["search_prefix"] = "TraceContext";
  options["search_metadata"] = true;
  options["resolution"] = "0";
  options["start_time_ms"] = "0.0";
  options["end_time_ms"] = "0.0";
  options["duration_ms"] = "0.0";

  // Act
  absl::StatusOr<std::string> tool_data = ConvertMultiXSpacesToToolData(
      *session_snapshot, "trace_viewer@", options);

  // Assert
  ASSERT_OK(tool_data);
  EXPECT_THAT(*tool_data, Not(IsEmpty()));
  EXPECT_THAT(*tool_data, HasSubstr("TraceContext"));
  EXPECT_THAT(*tool_data, HasSubstr("MetadataWithFoo"));

  // Arrange (No Metadata)
  // With search_metadata=false, it should still match the event but not return
  // its metadata
  ToolOptions options_no_metadata = options;
  options_no_metadata["search_metadata"] = false;

  // Act
  absl::StatusOr<std::string> tool_data_no_metadata =
      ConvertMultiXSpacesToToolData(*session_snapshot, "trace_viewer@",
                                    options_no_metadata);

  // Assert
  ASSERT_OK(tool_data_no_metadata);
  EXPECT_THAT(*tool_data_no_metadata, HasSubstr("TraceContext"));
  EXPECT_THAT(*tool_data_no_metadata, Not(HasSubstr("MetadataWithFoo")));

  // Clean up
  ASSERT_OK(file::RecursivelyDelete(session_dir, file::Defaults()));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
