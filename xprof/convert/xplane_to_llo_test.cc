/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/xplane_to_llo.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::testing::Not;
using ::testing::Property;
using ::testing::SizeIs;
using ::testing::StartsWith;

constexpr char kHost[] = "host_a";

// Adds a metadata plane with `num_modules` LLO modules (stored as opaque bytes
// on kernel event metadata) and an unrelated event metadata to `xspace`.
void AddLloModules(XSpace* xspace, int num_modules) {
  XPlane* plane = xspace->add_planes();
  tsl::profiler::XPlaneBuilder builder(plane);
  builder.SetName(tsl::profiler::kMetadataPlaneName);
  const tsl::profiler::XStatMetadata* llo_proto_stat =
      builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kLloProto));
  const tsl::profiler::XStatMetadata* other_stat =
      builder.GetOrCreateStatMetadata("other");
  for (int i = 0; i < num_modules; ++i) {
    tsl::profiler::XEventMetadata* kernel =
        builder.GetOrCreateEventMetadata(absl::StrCat("module(1)::kernel", i));
    // The profiler stores each module as the serialized proto (bytes).
    XStat* llo_stat = kernel->add_stats();
    llo_stat->set_metadata_id(llo_proto_stat->id());
    llo_stat->set_bytes_value(absl::StrCat("llo-bytes-", i));
    tsl::profiler::XStatsBuilder<tsl::profiler::XEventMetadata> stats(kernel,
                                                                      &builder);
    stats.AddStatValue(*other_stat, int64_t{7});
  }
  tsl::profiler::XEventMetadata* unrelated =
      builder.GetOrCreateEventMetadata("not a kernel");
  tsl::profiler::XStatsBuilder<tsl::profiler::XEventMetadata> stats(unrelated,
                                                                    &builder);
  stats.AddStatValue(*other_stat, int64_t{1});
}

// Creates a session with a writable run directory holding `xspace` as
// `<kHost>.xplane.pb`.
absl::StatusOr<SessionSnapshot> CreateFileSession(const XSpace& xspace) {
  const std::string run_dir = absl::StrCat(
      testing::TempDir(), "/",
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
  TF_RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(run_dir));
  const std::string path = absl::StrCat(run_dir, "/", kHost, ".xplane.pb");
  TF_RETURN_IF_ERROR(tsl::WriteBinaryProto(tsl::Env::Default(), path, xspace));
  return SessionSnapshot::Create({path}, std::nullopt);
}

// Creates a session without an accessible run directory holding `xspace`.
absl::StatusOr<SessionSnapshot> CreateInMemorySession(XSpace xspace) {
  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::make_unique<XSpace>(std::move(xspace)));
  return SessionSnapshot::Create(
      {absl::StrCat("ram://in_memory/", kHost, ".xplane.pb")},
      std::move(xspaces));
}

TEST(XPlaneToLloTest, DetectsLloModulesOnTheMetadataPlane) {
  XSpace xspace;
  EXPECT_FALSE(HasLloProtoMetadata(xspace));
  AddLloModules(&xspace, /*num_modules=*/0);
  EXPECT_FALSE(HasLloProtoMetadata(xspace));

  XSpace xspace_with_modules;
  AddLloModules(&xspace_with_modules, /*num_modules=*/2);
  EXPECT_TRUE(HasLloProtoMetadata(xspace_with_modules));
}

TEST(XPlaneToLloTest, ExtractsOnlyTheLloModules) {
  XSpace xspace;
  AddLloModules(&xspace, /*num_modules=*/2);

  XPlane llo_plane;
  EXPECT_EQ(ExtractLloXPlane(xspace, &llo_plane), 2);

  EXPECT_EQ(llo_plane.name(), tsl::profiler::kMetadataPlaneName);
  EXPECT_THAT(llo_plane.event_metadata(), SizeIs(2));
  EXPECT_THAT(llo_plane.stat_metadata(), SizeIs(1));
  for (const auto& [id, event_metadata] : llo_plane.event_metadata()) {
    EXPECT_THAT(event_metadata.stats(), SizeIs(1));
    EXPECT_THAT(event_metadata.stats(0).bytes_value(),
                StartsWith("llo-bytes-"));
  }
}

TEST(XPlaneToLloTest, CachesTheLloXPlaneOfAFileSession) {
  XSpace xspace;
  AddLloModules(&xspace, /*num_modules=*/3);
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot session, CreateFileSession(xspace));
  const std::string cache_path =
      absl::StrCat(session.GetSessionRunDir(), "/", kHost, kLloXPlaneSuffix);
  ASSERT_FALSE(tsl::Env::Default()->FileExists(cache_path).ok());

  // The first read of the host writes the cache file; later reads use it.
  absl::StatusOr<XPlane> llo_plane = GetLloXPlaneByHostName(session, kHost);
  ASSERT_THAT(llo_plane, IsOk());
  EXPECT_THAT(llo_plane->event_metadata(), SizeIs(3));
  EXPECT_TRUE(tsl::Env::Default()->FileExists(cache_path).ok());

  EXPECT_THAT(GetLloXPlaneByHostName(session, /*host_name=*/""),
              IsOkAndHolds(Property(&XPlane::event_metadata, SizeIs(3))));
  EXPECT_THAT(GetLloXPlaneByHostName(session, "unknown_host"), Not(IsOk()));
}

TEST(XPlaneToLloTest, MaybeCacheLloXPlaneWritesOnce) {
  XSpace xspace;
  AddLloModules(&xspace, /*num_modules=*/1);
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot session, CreateFileSession(xspace));

  EXPECT_THAT(MaybeCacheLloXPlane(session, kHost, xspace), IsOk());
  const std::string cache_path =
      absl::StrCat(session.GetSessionRunDir(), "/", kHost, kLloXPlaneSuffix);
  EXPECT_TRUE(tsl::Env::Default()->FileExists(cache_path).ok());
  // A second call keeps the existing file.
  EXPECT_THAT(MaybeCacheLloXPlane(session, kHost, XSpace()), IsOk());
  EXPECT_THAT(GetLloXPlaneByHostName(session, kHost),
              IsOkAndHolds(Property(&XPlane::event_metadata, SizeIs(1))));
}

TEST(XPlaneToLloTest, ReadsInMemorySessionsWithoutCaching) {
  XSpace xspace;
  AddLloModules(&xspace, /*num_modules=*/2);
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot session,
                          CreateInMemorySession(xspace));

  EXPECT_THAT(MaybeCacheLloXPlane(session, kHost, xspace), IsOk());
  EXPECT_THAT(GetLloXPlaneByHostName(session, kHost),
              IsOkAndHolds(Property(&XPlane::event_metadata, SizeIs(2))));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
