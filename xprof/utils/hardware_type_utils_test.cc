/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/utils/hardware_type_utils.h"

#include "<gtest/gtest.h>"
#include "xla/tsl/profiler/utils/math_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(HardwareTypeUtilsTest, B200PeakComputTFlops) {
  DeviceCapabilities device_cap;
  // For NVIDIA B200, according to:
  // https://resources.nvidia.com/en-us-blackwell-architecture?ncid=pa-srch-goog-585983-Intel-Brand-Broad
  // https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
  device_cap.set_clock_rate_in_ghz(1.830);
  device_cap.set_num_cores(148);
  device_cap.set_memory_size_in_bytes(
      tsl::profiler::GibiToGiga(tsl::profiler::GigaToUni(180)));
  device_cap.set_memory_bandwidth(tsl::profiler::GigaToUni(7.68 * 1024));
  device_cap.set_device_vendor("Nvidia");
  device_cap.mutable_compute_capability()->set_major(10);
  device_cap.mutable_compute_capability()->set_minor(0);

  // Get target TFLOPS per SM and check.
  double peak_tflops =
      GetFlopMaxThroughputPerSM(device_cap) * device_cap.num_cores() / 1000.0;
  EXPECT_NEAR(peak_tflops, 2218, /*abs_error=*/1.0);
}

// It should fall back to the highest compute cap less than 10.9.
// Currently it is 10.0.
TEST(HardwareTypeUtilsTest, FutureBlackwellPeakComputTFlops) {
  DeviceCapabilities device_cap;
  device_cap.set_clock_rate_in_ghz(1.830);
  device_cap.set_num_cores(148);
  device_cap.set_memory_size_in_bytes(
      tsl::profiler::GibiToGiga(tsl::profiler::GigaToUni(180)));
  device_cap.set_memory_bandwidth(tsl::profiler::GigaToUni(7.68 * 1024));
  device_cap.set_device_vendor("Nvidia");
  // Fake compute cap 10.9.
  device_cap.mutable_compute_capability()->set_major(10);
  device_cap.mutable_compute_capability()->set_minor(9);

  // Get target TFLOPS per SM and check.
  double peak_tflops =
      GetFlopMaxThroughputPerSM(device_cap) * device_cap.num_cores() / 1000.0;
  EXPECT_NEAR(peak_tflops, 2218, /*abs_error=*/1.0);
}

TEST(HardwareTypeUtilsTest, H100PeakComputTFlops) {
  DeviceCapabilities device_cap;
  // For NVIDIA H100 PCIe 80 GB, according to
  // https://resources.nvidia.com/en-us-data-center-overview/gtc22-whitepaper-hopper
  // https://www.techpowerup.com/gpu-specs/h100-pcie-80-gb.c3899
  device_cap.set_clock_rate_in_ghz(1.620);
  device_cap.set_num_cores(114);
  device_cap.set_memory_size_in_bytes(
      tsl::profiler::GibiToGiga(tsl::profiler::GigaToUni(80)));
  device_cap.set_memory_bandwidth(tsl::profiler::GigaToUni(2.04 * 1024));
  device_cap.set_device_vendor("Nvidia");
  device_cap.mutable_compute_capability()->set_major(9);
  device_cap.mutable_compute_capability()->set_minor(0);

  // Get target TFLOPS per SM and check.
  double peak_tflops =
      GetFlopMaxThroughputPerSM(device_cap) * device_cap.num_cores() / 1000.0;
  EXPECT_NEAR(peak_tflops, 756, /*abs_error=*/1.0);
}

TEST(HardwareTypeUtilsTest, A100PeakComputTFlops) {
  DeviceCapabilities device_cap;
  // For NVIDIA A100 SXM4 80 GB, according to:
  // https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
  // https://www.techpowerup.com/gpu-specs/a100-sxm4-80-gb.c3746
  device_cap.set_clock_rate_in_ghz(1.410);
  device_cap.set_num_cores(108);
  device_cap.set_memory_size_in_bytes(
      tsl::profiler::GibiToGiga(tsl::profiler::GigaToUni(80)));
  device_cap.set_memory_bandwidth(tsl::profiler::GigaToUni(2.04 * 1024));
  device_cap.set_device_vendor("Nvidia");
  device_cap.mutable_compute_capability()->set_major(8);
  device_cap.mutable_compute_capability()->set_minor(0);

  double peak_tflops =
      GetFlopMaxThroughputPerSM(device_cap) * device_cap.num_cores() / 1000.0;
  EXPECT_NEAR(peak_tflops, 312, /*abs_error=*/1.0);
}

TEST(HardwareTypeUtilsTest, Mi300XPeakComputeTFlops) {
  DeviceCapabilities device_cap;
  // MI300X: 304 CU at 2.1 GHz, published dense FP16/BF16 matrix 1307.4 TFLOPS.
  device_cap.set_clock_rate_in_ghz(2.1);
  device_cap.set_num_cores(304);
  device_cap.set_device_vendor("AMD");
  device_cap.set_device_name("gfx942:sramecc+:xnack-");

  double peak_tflops =
      GetFlopMaxThroughputPerSM(device_cap) * device_cap.num_cores() / 1000.0;
  // Guards the bf16 trap: if the matrix rate were stored only in bf16_tflops,
  // the max would skip it and report the vector fp32 peak of 163.4 instead.
  EXPECT_NEAR(peak_tflops, 1307.4, /*abs_error=*/1.0);
}

TEST(HardwareTypeUtilsTest, Mi100PeakComputeTFlops) {
  DeviceCapabilities device_cap;
  // MI100: 120 CU at 1.502 GHz, published FP16 matrix 184.6 TFLOPS. CDNA 1 is
  // the one architecture where bf16 MFMA is half rate, so fp16 is the headline.
  device_cap.set_clock_rate_in_ghz(1.502);
  device_cap.set_num_cores(120);
  device_cap.set_device_vendor("AMD");
  device_cap.set_device_name("gfx908");

  double peak_tflops =
      GetFlopMaxThroughputPerSM(device_cap) * device_cap.num_cores() / 1000.0;
  EXPECT_NEAR(peak_tflops, 184.6, /*abs_error=*/1.0);
}

TEST(HardwareTypeUtilsTest, Mi250XPeakComputeTFlopsPerGcd) {
  DeviceCapabilities device_cap;
  // MI250X: 104 CU per GCD at 1.7 GHz, published FP16/BF16 362.1 TFLOPS per GCD.
  device_cap.set_clock_rate_in_ghz(1.7);
  device_cap.set_num_cores(104);
  device_cap.set_device_vendor("AMD");
  device_cap.set_device_name("gfx90a");

  double peak_tflops =
      GetFlopMaxThroughputPerSM(device_cap) * device_cap.num_cores() / 1000.0;
  EXPECT_NEAR(peak_tflops, 362.1, /*abs_error=*/1.0);
}

TEST(HardwareTypeUtilsTest, Mi355XPeakComputeTFlops) {
  DeviceCapabilities device_cap;
  // MI355X: 256 CU at 2.4 GHz, published dense FP16/BF16 matrix 2.5 PFLOPS.
  device_cap.set_clock_rate_in_ghz(2.4);
  device_cap.set_num_cores(256);
  device_cap.set_device_vendor("AMD");
  device_cap.set_device_name("gfx950");

  double peak_tflops =
      GetFlopMaxThroughputPerSM(device_cap) * device_cap.num_cores() / 1000.0;
  EXPECT_NEAR(peak_tflops, 2516.6, /*abs_error=*/1.0);
}

TEST(HardwareTypeUtilsTest, AmbiguousAmdComputeCapabilityReportsNoPeak) {
  // (9, 0) is gfx908 or gfx90a, whose rates differ by 2x. Without a device name
  // to disambiguate, report nothing rather than pick one.
  DeviceCapabilities device_cap;
  device_cap.set_clock_rate_in_ghz(1.7);
  device_cap.set_num_cores(104);
  device_cap.set_device_vendor("AMD");
  device_cap.mutable_compute_capability()->set_major(9);
  device_cap.mutable_compute_capability()->set_minor(0);

  EXPECT_EQ(GetFlopMaxThroughputPerSM(device_cap), 0.0);
}

TEST(HardwareTypeUtilsTest, Mi300XSharedMemoryBandwidth) {
  DeviceCapabilities device_cap;
  // MI300X: 304 CU at 2.1 GHz, CDNA 3 (32 LDS banks x 4 B = 128 B/clock/CU).
  device_cap.set_clock_rate_in_ghz(2.1);
  device_cap.set_num_cores(304);
  device_cap.set_device_vendor("AMD");
  device_cap.set_device_name("gfx942");

  double aggregate_giga_bytes_per_second =
      device_cap.num_cores() *
      tsl::profiler::UniToGiga(GetSharedMemoryBandwidthPerSM(device_cap));
  // 304 * 128 B * 2.1e9 = 81,715.2 GB/s. Notably half of what the Nvidia bank
  // model would report, since CDNA has the same 32 banks at half the width.
  EXPECT_NEAR(aggregate_giga_bytes_per_second, 81715.2, /*abs_error=*/1.0);
}

TEST(HardwareTypeUtilsTest, Mi355XSharedMemoryBandwidthDoubles) {
  DeviceCapabilities device_cap;
  // MI355X: 256 CU at 2.4 GHz, CDNA 4 (64 LDS banks x 4 B = 256 B/clock/CU).
  device_cap.set_clock_rate_in_ghz(2.4);
  device_cap.set_num_cores(256);
  device_cap.set_device_vendor("AMD");
  device_cap.set_device_name("gfx950");

  double aggregate_giga_bytes_per_second =
      device_cap.num_cores() *
      tsl::profiler::UniToGiga(GetSharedMemoryBandwidthPerSM(device_cap));
  EXPECT_NEAR(aggregate_giga_bytes_per_second, 157286.4, /*abs_error=*/1.0);
}

TEST(HardwareTypeUtilsTest, SharedMemoryBandwidthFromComputeCapability) {
  // Traces captured before the collector emitted a device name still resolve,
  // because (9, 4) identifies CDNA 3 unambiguously.
  DeviceCapabilities device_cap;
  device_cap.set_clock_rate_in_ghz(2.1);
  device_cap.set_num_cores(304);
  device_cap.set_device_vendor("AMD");
  device_cap.mutable_compute_capability()->set_major(9);
  device_cap.mutable_compute_capability()->set_minor(4);

  double aggregate_giga_bytes_per_second =
      device_cap.num_cores() *
      tsl::profiler::UniToGiga(GetSharedMemoryBandwidthPerSM(device_cap));
  EXPECT_NEAR(aggregate_giga_bytes_per_second, 81715.2, /*abs_error=*/1.0);
}

TEST(HardwareTypeUtilsTest, SharedMemoryBandwidthUnknownAmdArchReportsNothing) {
  // (9, 0) is gfx908 or gfx90a, whose rates differ. Report nothing rather than
  // pick one.
  DeviceCapabilities device_cap;
  device_cap.set_clock_rate_in_ghz(1.7);
  device_cap.set_num_cores(104);
  device_cap.set_device_vendor("AMD");
  device_cap.mutable_compute_capability()->set_major(9);
  device_cap.mutable_compute_capability()->set_minor(0);

  EXPECT_EQ(GetSharedMemoryBandwidthPerSM(device_cap), 0.0);
}

TEST(HardwareTypeUtilsTest, GpuModelNameIgnoresNvidiaProductName) {
  DeviceCapabilities device_cap;
  device_cap.set_device_vendor("Nvidia");
  device_cap.mutable_compute_capability()->set_major(9);
  // CUPTI reports a product name, which contains no "GPU" substring. Returning
  // it would make ParseHardwareType, the GPU roofline table selection and the
  // frontend all stop recognising the device.
  device_cap.set_device_name("NVIDIA H100 80GB HBM3");

  EXPECT_EQ(GpuModelName(device_cap), "Nvidia GPU (Hopper)");
  EXPECT_EQ(ParseHardwareType(GpuModelName(device_cap)), HardwareType::GPU);
}

TEST(HardwareTypeUtilsTest, GpuModelNameFallsBackToFamilyWithoutName) {
  DeviceCapabilities device_cap;
  device_cap.set_device_vendor("Nvidia");
  device_cap.mutable_compute_capability()->set_major(9);

  EXPECT_EQ(GpuModelName(device_cap), "Nvidia GPU (Hopper)");
}

TEST(HardwareTypeUtilsTest, GpuModelNameReportsAmdArchAndStripsFeatures) {
  DeviceCapabilities device_cap;
  device_cap.set_device_vendor("AMD");
  device_cap.mutable_compute_capability()->set_major(9);
  device_cap.mutable_compute_capability()->set_minor(4);
  device_cap.set_device_name("gfx942:sramecc+:xnack-");

  // Must remain more specific than the family fallback, and must still contain
  // "GPU" because ParseHardwareType and the frontend both key on that.
  EXPECT_EQ(GpuModelName(device_cap), "AMD GPU - gfx942");
  EXPECT_EQ(ParseHardwareType(GpuModelName(device_cap)), HardwareType::GPU);
}

TEST(HardwareTypeUtilsTest, GpuModelNameFallsBackToAmdFamilyWithoutName) {
  DeviceCapabilities device_cap;
  device_cap.set_device_vendor("AMD");
  device_cap.mutable_compute_capability()->set_major(9);

  EXPECT_EQ(GpuModelName(device_cap), "AMD GPU - gfx-9XX series");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
