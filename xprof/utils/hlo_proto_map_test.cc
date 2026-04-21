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

TEST(HloProtoMapTest, AddHloProtosFromXSpace) {
  tensorflow::profiler::XSpace space;
  tensorflow::profiler::XPlane* metadata_plane = space.add_planes();
  tsl::profiler::XPlaneBuilder plane_builder(metadata_plane);
  plane_builder.SetName(tsl::profiler::kMetadataPlaneName);

  xla::profiler::MetadataXPlaneBuilder metadata_builder(metadata_plane);

  xla::HloProto optimized_proto;
  optimized_proto.mutable_hlo_module()->set_name("optimized_module");
  metadata_builder.AddHloProto(1, optimized_proto);

  xla::HloProto original_proto;
  original_proto.mutable_hlo_module()->set_name("original_module");
  metadata_builder.AddOriginalHloProto(1, original_proto);

  HloProtoMap hlo_proto_map;
  hlo_proto_map.AddHloProtosFromXSpace(space);

  ASSERT_OK_AND_ASSIGN(const xla::HloProto* opt_result,
                       hlo_proto_map.GetHloProtoByProgramId(1));
  EXPECT_EQ(opt_result->hlo_module().name(), "optimized_module");

  ASSERT_OK_AND_ASSIGN(const xla::HloProto* orig_result,
                       hlo_proto_map.GetOriginalHloProtoByProgramId(1));
  EXPECT_EQ(orig_result->hlo_module().name(), "original_module");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
