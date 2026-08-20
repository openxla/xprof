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

#include <string>

#include "xprof/utils/cuda_type_utils.h"
#include "xprof/utils/rocm_type_utils.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"

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

bool HasDevice(HardwareType x) { return x > tensorflow::profiler::CPU_ONLY; }

}  // namespace profiler
}  // namespace tensorflow
