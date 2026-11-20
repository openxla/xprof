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

#ifndef _WIN32
#include <sys/stat.h>
#endif

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/unified_session_snapshot.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::IsEmpty;
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

TEST(XPlaneToLloTest, ExtractsAcrossMultipleMetadataPlanesWithoutIdCollisions) {
  XSpace xspace;

  // First metadata plane carries only HLO metadata (no `kLloProto` stat).
  {
    tsl::profiler::XPlaneBuilder hlo_builder(xspace.add_planes());
    hlo_builder.SetName(tsl::profiler::kMetadataPlaneName);
    const tsl::profiler::XStatMetadata* hlo_stat_metadata =
        hlo_builder.GetOrCreateStatMetadata(
            tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloProto));
    XStat* hlo_stat =
        hlo_builder.GetOrCreateEventMetadata("hlo_mod(1)")->add_stats();
    hlo_stat->set_metadata_id(hlo_stat_metadata->id());
    hlo_stat->set_bytes_value("hlo-bytes");
  }
  EXPECT_FALSE(HasLloProtoMetadata(xspace));

  // Second metadata plane carries `kLloProto` at plane-local stat id 1 and
  // event metadata id 1.
  {
    tsl::profiler::XPlaneBuilder builder_a(xspace.add_planes());
    builder_a.SetName(tsl::profiler::kMetadataPlaneName);
    const tsl::profiler::XStatMetadata* llo_stat_a =
        builder_a.GetOrCreateStatMetadata(
            tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kLloProto));
    ASSERT_EQ(llo_stat_a->id(), 1);
    tsl::profiler::XEventMetadata* event_a =
        builder_a.GetOrCreateEventMetadata("mod_a(1)::k0");
    ASSERT_EQ(event_a->id(), 1);
    XStat* stat = event_a->add_stats();
    stat->set_metadata_id(llo_stat_a->id());
    stat->set_bytes_value("llo-bytes-a");
  }

  // Third metadata plane has a different stat id for `kLloProto` (id 2) and a
  // colliding plane-local event metadata id (id 1).
  {
    tsl::profiler::XPlaneBuilder builder_b(xspace.add_planes());
    builder_b.SetName(tsl::profiler::kMetadataPlaneName);
    builder_b.GetOrCreateStatMetadata("unrelated_stat");
    const tsl::profiler::XStatMetadata* llo_stat_b =
        builder_b.GetOrCreateStatMetadata(
            tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kLloProto));
    ASSERT_EQ(llo_stat_b->id(), 2);
    tsl::profiler::XEventMetadata* event_b =
        builder_b.GetOrCreateEventMetadata("mod_b(2)::k1");
    ASSERT_EQ(event_b->id(), 1);
    XStat* stat = event_b->add_stats();
    stat->set_metadata_id(llo_stat_b->id());
    stat->set_bytes_value("llo-bytes-b");
  }

  EXPECT_TRUE(HasLloProtoMetadata(xspace));

  XPlane llo_plane;
  EXPECT_EQ(ExtractLloXPlane(xspace, &llo_plane), 2);
  ASSERT_THAT(llo_plane.stat_metadata(), SizeIs(1));
  const int64_t extracted_stat_id = llo_plane.stat_metadata().begin()->first;
  ASSERT_THAT(llo_plane.event_metadata(), SizeIs(2));
  ASSERT_EQ(llo_plane.event_metadata().count(1), 1);
  ASSERT_EQ(llo_plane.event_metadata().count(2), 1);
  EXPECT_EQ(llo_plane.event_metadata().at(1).name(), "mod_a(1)::k0");
  EXPECT_EQ(llo_plane.event_metadata().at(1).stats(0).metadata_id(),
            extracted_stat_id);
  EXPECT_EQ(llo_plane.event_metadata().at(1).stats(0).bytes_value(),
            "llo-bytes-a");
  EXPECT_EQ(llo_plane.event_metadata().at(2).name(), "mod_b(2)::k1");
  EXPECT_EQ(llo_plane.event_metadata().at(2).stats(0).metadata_id(),
            extracted_stat_id);
  EXPECT_EQ(llo_plane.event_metadata().at(2).stats(0).bytes_value(),
            "llo-bytes-b");
}

TEST(XPlaneToLloTest, ExtractLloXPlaneHandlesNullAndClearsExistingPlane) {
  XSpace xspace;
  AddLloModules(&xspace, /*num_modules=*/1);
  EXPECT_EQ(ExtractLloXPlane(xspace, nullptr), 0);

  XPlane llo_plane;
  llo_plane.set_name("stale_plane");
  llo_plane.add_lines()->set_name("stale_line");
  llo_plane.add_stats()->set_int64_value(42);
  (*llo_plane.mutable_event_metadata())[99].set_name("stale_event");
  (*llo_plane.mutable_stat_metadata())[99].set_name("stale_stat");

  EXPECT_EQ(ExtractLloXPlane(xspace, &llo_plane), 1);
  EXPECT_EQ(llo_plane.name(), tsl::profiler::kMetadataPlaneName);
  EXPECT_THAT(llo_plane.lines(), IsEmpty());
  EXPECT_THAT(llo_plane.stats(), IsEmpty());
  EXPECT_THAT(llo_plane.event_metadata(), SizeIs(1));
  EXPECT_EQ(llo_plane.event_metadata().count(99), 0);
  EXPECT_THAT(llo_plane.stat_metadata(), SizeIs(1));
  EXPECT_EQ(llo_plane.stat_metadata().count(99), 0);

  // Extracting from an XSpace without LLO modules also clears stale contents.
  EXPECT_EQ(ExtractLloXPlane(XSpace(), &llo_plane), 0);
  EXPECT_THAT(llo_plane.event_metadata(), IsEmpty());
  EXPECT_THAT(llo_plane.stat_metadata(), IsEmpty());
}

TEST(XPlaneToLloTest, CachesTheLloXPlaneOfAFileSession) {
  XSpace xspace;
  AddLloModules(&xspace, /*num_modules=*/3);
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot session, CreateFileSession(xspace));
  ASSERT_THAT(HostDataFileExists(session, StoredDataType::LLO_XPLANE, kHost),
              IsOkAndHolds(false));

  // The first read of the host writes the cache file; later reads use it.
  absl::StatusOr<XPlane> llo_plane = GetLloXPlaneByHostName(session, kHost);
  ASSERT_THAT(llo_plane, IsOk());
  EXPECT_THAT(llo_plane->event_metadata(), SizeIs(3));
  EXPECT_THAT(HostDataFileExists(session, StoredDataType::LLO_XPLANE, kHost),
              IsOkAndHolds(true));

  EXPECT_THAT(GetLloXPlaneByHostName(session, /*host_name=*/""),
              IsOkAndHolds(Property(&XPlane::event_metadata, SizeIs(3))));
  EXPECT_THAT(GetLloXPlaneByHostName(session, "unknown_host"), Not(IsOk()));

  ASSERT_THAT(session.ClearCacheFiles(), IsOk());
  EXPECT_THAT(HostDataFileExists(session, StoredDataType::LLO_XPLANE, kHost),
              IsOkAndHolds(false));
}

TEST(XPlaneToLloTest, RecoversFromCorruptCacheFile) {
  XSpace xspace;
  AddLloModules(&xspace, /*num_modules=*/2);
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot session, CreateFileSession(xspace));
  const std::string cache_path =
      absl::StrCat(session.GetSessionRunDir(), "/", kHost, ".llo_xplane.pb");
  ASSERT_THAT(tsl::WriteStringToFile(tsl::Env::Default(), cache_path,
                                     "not-a-valid-binary-proto"),
              IsOk());

  EXPECT_THAT(GetLloXPlaneByHostName(session, kHost),
              IsOkAndHolds(Property(&XPlane::event_metadata, SizeIs(2))));

  XPlane rewritten_cache;
  ASSERT_THAT(ReadBinaryProto(session, StoredDataType::LLO_XPLANE, kHost,
                              &rewritten_cache),
              IsOk());
  EXPECT_THAT(rewritten_cache.event_metadata(), SizeIs(2));
}

TEST(XPlaneToLloTest, RejectsPathTraversalAndNonSessionHostNames) {
  XSpace xspace;
  AddLloModules(&xspace, /*num_modules=*/1);
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot session, CreateFileSession(xspace));

  // Create a sibling directory holding a `.llo_xplane.pb` file and a stale
  // cache file in the session run dir for an excluded host.
  const std::string other_dir = absl::StrCat(testing::TempDir(), "/other_dir");
  ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(other_dir), IsOk());
  XPlane forged_plane;
  (*forged_plane.mutable_event_metadata())[1].set_name("forged");
  ASSERT_THAT(
      xprof::WriteBinaryProto(absl::StrCat(other_dir, "/outside.llo_xplane.pb"),
                              forged_plane),
      IsOk());
  ASSERT_THAT(
      xprof::WriteBinaryProto(absl::StrCat(session.GetSessionRunDir(),
                                           "/excluded_host.llo_xplane.pb"),
                              forged_plane),
      IsOk());

  EXPECT_THAT(GetLloXPlaneByHostName(session, "../other_dir/outside"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetLloXPlaneByHostName(session, "excluded_host"),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(MaybeCacheLloXPlane(session, "../other_dir/evil", xspace),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(MaybeCacheLloXPlane(session, "excluded_host", xspace),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_FALSE(tsl::Env::Default()
                   ->FileExists(absl::StrCat(other_dir, "/evil.llo_xplane.pb"))
                   .ok());
}

#ifndef _WIN32
TEST(XPlaneToLloTest, ReturnsExtractedLloXPlaneWhenRunDirIsReadOnly) {
  XSpace xspace;
  AddLloModules(&xspace, /*num_modules=*/2);
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot session, CreateFileSession(xspace));
  const std::string run_dir(session.GetSessionRunDir());

  ASSERT_EQ(::chmod(run_dir.c_str(), 0555), 0);
  absl::Cleanup restore_permissions = [&run_dir] {
    ::chmod(run_dir.c_str(), 0755);
  };

  EXPECT_THAT(GetLloXPlaneByHostName(session, kHost),
              IsOkAndHolds(Property(&XPlane::event_metadata, SizeIs(2))));
  EXPECT_THAT(HostDataFileExists(session, StoredDataType::LLO_XPLANE, kHost),
              IsOkAndHolds(false));
}
#endif

TEST(XPlaneToLloTest, MaybeCacheLloXPlaneWritesOnce) {
  XSpace xspace;
  AddLloModules(&xspace, /*num_modules=*/1);
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot session, CreateFileSession(xspace));

  EXPECT_THAT(MaybeCacheLloXPlane(session, kHost, xspace), IsOk());
  EXPECT_THAT(HostDataFileExists(session, StoredDataType::LLO_XPLANE, kHost),
              IsOkAndHolds(true));
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
