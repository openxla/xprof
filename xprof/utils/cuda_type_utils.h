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

#ifndef XPROF_UTILS_CUDA_TYPE_UTILS_H_
#define XPROF_UTILS_CUDA_TYPE_UTILS_H_

#include <string>

#include "plugin/xprof/protobuf/hardware_types.pb.h"

namespace tensorflow {
namespace profiler {
namespace cuda {

// Peak throughput in GFLOPs per second per SM, at the reported clock.
double GetFlopMaxThroughputPerCore(const DeviceCapabilities& device_cap);

// Shared memory bandwidth in Bytes per second per SM.
double GetSharedMemoryBandwidthPerCore(const DeviceCapabilities& device_cap);

// Microarchitecture family, e.g. "Nvidia GPU (Hopper)".
std::string GpuModelName(const DeviceCapabilities& device_cap);

}  // namespace cuda
}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_UTILS_CUDA_TYPE_UTILS_H_
