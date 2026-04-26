#include "xprof/utils/hlo_proto_map.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "xla/backends/profiler/cpu/metadata_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;
using ::testing::status::StatusIs;

TEST(HloProtoMapTest, GetOriginalModuleList) {
  HloProtoMap hlo_proto_map;
  EXPECT_THAT(hlo_proto_map.GetModuleList(true), IsEmpty());

  auto hlo_proto_1 = std::make_unique<xla::HloProto>();
  hlo_proto_1->mutable_hlo_module()->set_name("module1");
  hlo_proto_map.AddOriginalHloProto(1, std::move(hlo_proto_1));

  auto hlo_proto_2 = std::make_unique<xla::HloProto>();
  hlo_proto_2->mutable_hlo_module()->set_name("module2");
  hlo_proto_map.AddOriginalHloProto(2, std::move(hlo_proto_2));

  EXPECT_THAT(hlo_proto_map.GetModuleList(true),
              UnorderedElementsAre("module1(1)", "module2(2)"));
}

TEST(HloProtoMapTest, GetOriginalHloProto) {
  HloProtoMap hlo_proto_map;
  auto hlo_proto = std::make_unique<xla::HloProto>();
  hlo_proto->mutable_hlo_module()->set_name("module");
  hlo_proto_map.AddOriginalHloProto(1, std::move(hlo_proto));

  // Test GetOriginalHloProtoByProgramId
  ASSERT_OK_AND_ASSIGN(const xla::HloProto* result_by_id,
                       hlo_proto_map.GetOriginalHloProtoByProgramId(1));
  EXPECT_EQ(result_by_id->hlo_module().name(), "module");

  EXPECT_THAT(hlo_proto_map.GetOriginalHloProtoByProgramId(2),
              StatusIs(absl::StatusCode::kNotFound));

  // Test GetOriginalHloProtoByModuleName
  ASSERT_OK_AND_ASSIGN(
      const xla::HloProto* result_by_name,
      hlo_proto_map.GetOriginalHloProtoByModuleName("module(1)"));
  EXPECT_EQ(result_by_name->hlo_module().name(), "module");

  EXPECT_THAT(hlo_proto_map.GetOriginalHloProtoByModuleName("module(2)"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(hlo_proto_map.GetOriginalHloProtoByModuleName("module2(1)"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(HloProtoMapTest, ParseOriginalHloProtosFromXSpace) {
  XSpace space;
  XPlane* raw_plane = space.add_planes();
  tsl::profiler::XPlaneBuilder plane_builder(raw_plane);
  plane_builder.SetName(std::string(tsl::profiler::kMetadataPlaneName));

  xla::profiler::MetadataXPlaneBuilder metadata_builder(raw_plane);

  xla::HloProto hlo_proto;
  hlo_proto.mutable_hlo_module()->set_name("my_module");

  // Add original hlo proto with program_id = 100
  metadata_builder.AddOriginalHloProto(100, hlo_proto);

  HloProtoMap hlo_proto_map;
  hlo_proto_map.AddHloProtosFromXSpace(space);

  // Verify that the original hlo proto is parsed correctly
  ASSERT_OK_AND_ASSIGN(const xla::HloProto* result_by_id,
                       hlo_proto_map.GetOriginalHloProtoByProgramId(100));
  EXPECT_EQ(result_by_id->hlo_module().name(), "my_module");

  // Also verify that the optimized hlo proto is not magically added
  EXPECT_THAT(hlo_proto_map.GetHloProtoByProgramId(100),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
