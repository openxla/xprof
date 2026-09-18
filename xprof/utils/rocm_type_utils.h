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

#ifndef XPROF_UTILS_ROCM_TYPE_UTILS_H_
#define XPROF_UTILS_ROCM_TYPE_UTILS_H_

#include <string>

#include "absl/strings/string_view.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"

namespace tensorflow {
namespace profiler {
namespace rocm {

// Peak throughput in GFLOPs per second per CU, at the reported clock.
double GetFlopMaxThroughputPerCore(const DeviceCapabilities& device_cap);

// LDS bandwidth in Bytes per second per CU. LDS is the AMD analogue of Nvidia
// shared memory.
double GetSharedMemoryBandwidthPerCore(const DeviceCapabilities& device_cap);

// GCN architecture, e.g. "AMD GPU - gfx942".
std::string GpuModelName(const DeviceCapabilities& device_cap);

// True if kernel name matches against known Matrix Core naming patterns.
bool IsKernelUsingMatrixCore(absl::string_view kernel_name);

}  // namespace rocm
}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_UTILS_ROCM_TYPE_UTILS_H_
