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

#include "xprof/utils/xplane_hlo_fixer.h"

#include <cstdint>
#include <string>

#include "<gtest/gtest.h>"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xprof {
namespace {

using tensorflow::profiler::XSpace;
using tsl::profiler::GetStatTypeStr;
using tsl::profiler::kHloProto;
using tsl::profiler::kMetadataPlaneName;
using tsl::profiler::XPlaneBuilder;


TEST(XPlaneHloFixerTest, FixesHloMetadataSuccess) {
  XSpace space;
  XPlaneBuilder metadata_plane(space.add_planes());
  metadata_plane.SetName(kMetadataPlaneName);


  // Add legacy HLO Proto metadata.
  auto* hlo_metadata = metadata_plane.GetOrCreateStatMetadata("HLO Proto");
  int64_t hlo_stat_id = hlo_metadata->id();

  // Add event metadata with HLO stat and program ID in name.
  uint64_t prog_id = 12345;
  std::string event_name = "train_step (12345)";
  auto* event_meta = metadata_plane.GetOrCreateEventMetadata(event_name);
  auto* stat = event_meta->add_stats();
  stat->set_metadata_id(hlo_stat_id);
  stat->set_str_value("hlo_content");
  int64_t original_event_id = event_meta->id();

  ASSERT_NE(original_event_id, static_cast<int64_t>(prog_id));

  FixHloMetadataInXSpace(&space);

  // Verify HLO Proto name is updated.
  EXPECT_EQ(hlo_metadata->name(), GetStatTypeStr(kHloProto));

  // Verify event metadata ID is updated to program ID.
  auto event_meta_map = space.planes(0).event_metadata();
  EXPECT_EQ(event_meta_map.size(), 1);
  EXPECT_TRUE(event_meta_map.contains(prog_id));
  EXPECT_EQ(event_meta_map.at(prog_id).name(), event_name);
  EXPECT_EQ(event_meta_map.at(prog_id).id(), prog_id);
}

TEST(XPlaneHloFixerTest, DoesNotFixIfNoLegacyHloStat) {
  XSpace space;
  XPlaneBuilder metadata_plane(space.add_planes());
  metadata_plane.SetName(kMetadataPlaneName);

  // Add event metadata.
  std::string event_name = "train_step (12345)";
  auto* event_meta = metadata_plane.GetOrCreateEventMetadata(event_name);
  int64_t original_event_id = event_meta->id();

  FixHloMetadataInXSpace(&space);

  // Verify event metadata ID is NOT updated.
  auto event_meta_map = space.planes(0).event_metadata();
  EXPECT_TRUE(event_meta_map.contains(original_event_id));
}

TEST(XPlaneHloFixerTest, DoesNotFixIfHloStatAlreadyHasExpectedName) {
  XSpace space;
  XPlaneBuilder metadata_plane(space.add_planes());
  metadata_plane.SetName(kMetadataPlaneName);

  // Add expected HLO Proto metadata.
  auto* hlo_metadata =
      metadata_plane.GetOrCreateStatMetadata(GetStatTypeStr(kHloProto));
  int64_t hlo_stat_id = hlo_metadata->id();

  // Add event metadata.
  uint64_t prog_id = 12345;
  std::string event_name = "train_step (12345)";
  auto* event_meta = metadata_plane.GetOrCreateEventMetadata(event_name);
  auto* stat = event_meta->add_stats();
  stat->set_metadata_id(hlo_stat_id);
  stat->set_str_value("hlo_content");
  int64_t original_event_id = event_meta->id();

  FixHloMetadataInXSpace(&space);

  // Verify event metadata ID is NOT updated.
  auto event_meta_map = space.planes(0).event_metadata();
  EXPECT_TRUE(event_meta_map.contains(original_event_id));
  EXPECT_FALSE(event_meta_map.contains(prog_id));
}

TEST(XPlaneHloFixerTest, SkipsIfNoHloStatInEvent) {
  XSpace space;
  XPlaneBuilder metadata_plane(space.add_planes());
  metadata_plane.SetName(kMetadataPlaneName);

  // Add some other stat.
  auto* other_stat = metadata_plane.GetOrCreateStatMetadata("Other Stat");

  uint64_t prog_id = 12345;
  std::string event_name = "train_step (12345)";
  auto* event_meta = metadata_plane.GetOrCreateEventMetadata(event_name);
  auto* stat = event_meta->add_stats();
  stat->set_metadata_id(other_stat->id());
  stat->set_str_value("value");
  int64_t original_event_id = event_meta->id();

  FixHloMetadataInXSpace(&space);

  // Verify event metadata ID is NOT updated.
  auto event_meta_map = space.planes(0).event_metadata();
  EXPECT_TRUE(event_meta_map.contains(original_event_id));
  EXPECT_FALSE(event_meta_map.contains(prog_id));
}

}  // namespace
}  // namespace xprof
