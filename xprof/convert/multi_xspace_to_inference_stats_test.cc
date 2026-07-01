#include "xprof/convert/multi_xspace_to_inference_stats.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"

namespace tensorflow {
namespace profiler {
namespace {

class ConvertMultiXSpaceToInferenceStatsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up mock XSpace data here.
    xspace_ = std::make_unique<XSpace>();
  XPlane* plane = xspace_->add_planes();
  plane->set_name(tsl::profiler::kHostThreadsPlaneName);
  // Add more lines and events to simulate real data
  XLine* line = plane->add_lines();
  line->set_name("MyThread");
  XEvent* event = line->add_events();
  event->set_offset_ps(1000);
  event->set_duration_ps(2000);
  // Add stats to the event
  XStat* stat = event->add_stats();
  stat->set_int64_value(12345);
  }

  std::unique_ptr<XSpace> xspace_;
};

TEST_F(ConvertMultiXSpaceToInferenceStatsTest, TestWithMultipleXSpaces) {
  std::string test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  std::string path = absl::StrCat("ram://", test_name, "/");
  std::vector<std::string> paths = {absl::StrCat(path, "hostname1.xplane.pb"),
                                    absl::StrCat(path, "hostname2.xplane.pb")};

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::make_unique<XSpace>());
  xspaces.push_back(std::make_unique<XSpace>());

  absl::StatusOr<SessionSnapshot> session_snapshot_status =
      SessionSnapshot::Create(paths, std::move(xspaces));
  SessionSnapshot session_snapshot = std::move(session_snapshot_status.value());

  InferenceStats inference_stats;
  absl::Status status = ConvertMultiXSpaceToInferenceStats(
      session_snapshot, "request", "batch", &inference_stats);

  EXPECT_OK(status);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
