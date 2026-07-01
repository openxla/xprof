#include "xprof/utils/xla_op_utils.h"

#include <string>

#include "gtest/gtest.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(XlaOpUtilsTest, ExtractXprofKernelMetadataTest) {
  std::string hlo_expression = R"(
  %blah.1 = bf16[4096,2048]{1,0:T(8,128)(2,1)}
    custom-call(s32[]{:T(128)} %bitcast), custom_call_target="tpu_custom_call",
    frontend_attributes={
      kernel_metadata={"xprof_metadata": "{\"whatever\": {\"I\": \"want\"}}"}}
)";
  std::string expected = R"({"whatever":{"I":"want"}})";
  EXPECT_EQ(ExtractXprofKernelMetadata(hlo_expression), expected);
}

TEST(XlaOpUtilsTest, ExtractXprofKernelMetadataTest_InvalidHlo) {
  std::string hlo_expression = R"(
  %blah.1 = bf16[4096,2048]{1,0:T(8,128)(2,1)}
    custom-call(s32[]{:T(128)} %bitcast), custom_call_target="tpu_custom_call",
    frontend_attributes={
      kernel_metadata={"xprof_metadata": "{broken_json\"}}"}}
)";
  EXPECT_EQ(ExtractXprofKernelMetadata(hlo_expression), "");
}

TEST(XlaOpUtilsTest, ExtractXprofKernelMetadataTest_NoXprofMetadata) {
  std::string hlo_expression = R"(
  %blah.1 = bf16[4096,2048]{1,0:T(8,128)(2,1)}
    custom-call(s32[]{:T(128)} %bitcast), custom_call_target="tpu_custom_call",
    frontend_attributes={
      kernel_metadata={}"}}
)";
  EXPECT_EQ(ExtractXprofKernelMetadata(hlo_expression), "");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
