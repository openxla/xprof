#include "xprof/convert/trace_viewer/trace_utils.h"

#include "<gtest/gtest.h>"

namespace tensorflow {
namespace profiler {
namespace {

TEST(TraceOptionsTest, IsTpuCoreDeviceNameTest) {
  EXPECT_TRUE(IsTpuCoreDeviceName("/device:TPU:0"));
  EXPECT_TRUE(IsTpuCoreDeviceName("TensorNode"));
  EXPECT_TRUE(IsTpuCoreDeviceName("TPU Core"));
  EXPECT_FALSE(IsTpuCoreDeviceName("GPU"));
  EXPECT_FALSE(IsTpuCoreDeviceName("Host Interface"));
}

TEST(TraceOptionsTest, MaybeTpuHostInterfaceDeviceNameTest) {
  EXPECT_TRUE(MaybeTpuHostInterfaceDeviceName("TPU v2 Host Interface"));
  EXPECT_FALSE(MaybeTpuHostInterfaceDeviceName("TPU v2"));
}

TEST(TraceOptionsTest, IsTpuHbmDeviceNameTest) {
  EXPECT_TRUE(IsTpuHbmDeviceName("TPU v2 HBM"));
  EXPECT_FALSE(IsTpuHbmDeviceName("TPU v2"));
}

TEST(TraceOptionsTest, IsTpuIciRouterDeviceNameTest) {
  EXPECT_TRUE(IsTpuIciRouterDeviceName("TPU v2 ICI Router"));
  EXPECT_FALSE(IsTpuIciRouterDeviceName("TPU v2"));
}

TEST(TraceOptionsTest, MaybeTpuNonCoreDeviceNameTest) {
  EXPECT_TRUE(MaybeTpuNonCoreDeviceName("#Chip TPU Non-Core HBM"));
  EXPECT_TRUE(MaybeTpuNonCoreDeviceName("#Chip TPU Non-Core Host Interface"));
  EXPECT_TRUE(MaybeTpuNonCoreDeviceName("#Chip TPU Non-Core ICI Router"));
  EXPECT_FALSE(MaybeTpuNonCoreDeviceName("TPU v2"));
  EXPECT_FALSE(MaybeTpuNonCoreDeviceName("TPU Non-Core Other"));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
