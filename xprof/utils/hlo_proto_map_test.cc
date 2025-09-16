#include "xprof/utils/hlo_proto_map.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;
using ::testing::status::StatusIs;

TEST(HloProtoMapTest, GetOriginalModuleList) {
  HloProtoMap hlo_proto_map;
  EXPECT_THAT(hlo_proto_map.GetOriginalModuleList(), IsEmpty());

  auto hlo_module_1 = std::make_unique<xla::HloModuleProto>();
  hlo_module_1->set_name("module1");
  hlo_proto_map.AddOriginalHloProto(1, std::move(hlo_module_1));

  auto hlo_module_2 = std::make_unique<xla::HloModuleProto>();
  hlo_module_2->set_name("module2");
  hlo_proto_map.AddOriginalHloProto(2, std::move(hlo_module_2));

  EXPECT_THAT(hlo_proto_map.GetOriginalModuleList(),
              UnorderedElementsAre("module1(1)", "module2(2)"));
}

TEST(HloProtoMapTest, GetOriginalHloProto) {
  HloProtoMap hlo_proto_map;
  auto hlo_module = std::make_unique<xla::HloModuleProto>();
  hlo_module->set_name("module");
  hlo_proto_map.AddOriginalHloProto(1, std::move(hlo_module));

  // Test GetOriginalHloProtoByProgramId
  ASSERT_OK_AND_ASSIGN(const xla::HloModuleProto* result_by_id,
                       hlo_proto_map.GetOriginalHloProtoByProgramId(1));
  EXPECT_EQ(result_by_id->name(), "module");

  EXPECT_THAT(hlo_proto_map.GetOriginalHloProtoByProgramId(2),
              StatusIs(absl::StatusCode::kNotFound));

  // Test GetOriginalHloProtoByModuleName
  ASSERT_OK_AND_ASSIGN(
      const xla::HloModuleProto* result_by_name,
      hlo_proto_map.GetOriginalHloProtoByModuleName("module(1)"));
  EXPECT_EQ(result_by_name->name(), "module");

  EXPECT_THAT(hlo_proto_map.GetOriginalHloProtoByModuleName("module(2)"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(hlo_proto_map.GetOriginalHloProtoByModuleName("module2(1)"),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
