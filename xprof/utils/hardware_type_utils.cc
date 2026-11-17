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

#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "xprof/utils/cuda_type_utils.h"
#include "xprof/utils/rocm_type_utils.h"

namespace tensorflow {
namespace profiler {

double GetFlopMaxThroughputPerCore(const DeviceCapabilities& device_cap) {
  if (device_cap.device_vendor() == tsl::profiler::kDeviceVendorNvidia) {
    return cuda::GetFlopMaxThroughputPerCore(device_cap);
  }
  if (device_cap.device_vendor() == tsl::profiler::kDeviceVendorAMD) {
    return rocm::GetFlopMaxThroughputPerCore(device_cap);
  }
  LOG(WARNING) << "Unsupported device vendor " << device_cap.device_vendor();
  return 0.0;
}

double GetFlopRateAdjustmentToBf16(
    int bitwidth, const std::optional<DeviceCapabilities>& device_cap) {
  std::optional<double> adjustment;
  if (device_cap.has_value()) {
    if (device_cap->device_vendor() == tsl::profiler::kDeviceVendorNvidia) {
      adjustment = cuda::GetFlopRateAdjustmentToBf16(bitwidth, *device_cap);
    } else if (device_cap->device_vendor() == tsl::profiler::kDeviceVendorAMD) {
      adjustment = rocm::GetFlopRateAdjustmentToBf16(bitwidth, *device_cap);
    } else if (!device_cap->device_vendor().empty()) {
      LOG_FIRST_N(WARNING, 1)
          << "Unsupported device vendor " << device_cap->device_vendor();
    }
  }
  if (adjustment.has_value()) {
    return *adjustment;
  }
  // Fall back to bitwise normalization relative to 16-bit when device
  // capabilities are unavailable or unrecognized.
  switch (bitwidth) {
    case 8:
      return 2.0;
    case 4:
      return 4.0;
    default:
      return 1.0;
  }
}

double GetSharedMemoryBandwidthPerCore(const DeviceCapabilities& device_cap) {
  if (device_cap.device_vendor() == tsl::profiler::kDeviceVendorNvidia) {
    return cuda::GetSharedMemoryBandwidthPerCore(device_cap);
  }
  if (device_cap.device_vendor() == tsl::profiler::kDeviceVendorAMD) {
    return rocm::GetSharedMemoryBandwidthPerCore(device_cap);
  }
  // Report zero rather than a value borrowed from another vendor.
  return 0.0;
}

std::string GpuModelName(const DeviceCapabilities& device_cap) {
  if (device_cap.device_vendor() == tsl::profiler::kDeviceVendorNvidia) {
    return cuda::GpuModelName(device_cap);
  }
  if (device_cap.device_vendor() == tsl::profiler::kDeviceVendorAMD) {
    return rocm::GpuModelName(device_cap);
  }
  LOG(ERROR) << "Unknown device vendor " << device_cap.device_vendor();
  return "";
}

HardwareType ParseHardwareType(absl::string_view device_type) {
  if (absl::StrContains(device_type, "GPU")) return HardwareType::GPU;
  if (device_type == "CPU") return HardwareType::CPU_ONLY;
  if (absl::StrContains(device_type, "TPU")) return HardwareType::TPU;
  return HardwareType::UNKNOWN_HARDWARE;
}

HardwareType GetHardwareType(const RunEnvironment& run_environment) {
  if (run_environment.hardware_type() != HardwareType::UNKNOWN_HARDWARE) {
    return run_environment.hardware_type();
  }
  return ParseHardwareType(run_environment.device_type());
}

bool HasDevice(HardwareType x) { return x > tensorflow::profiler::CPU_ONLY; }

}  // namespace profiler
}  // namespace tensorflow
