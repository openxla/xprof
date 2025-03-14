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

#ifndef XPROF_UTILS_DEVICE_CAPS_UTILS_H_
#define XPROF_UTILS_DEVICE_CAPS_UTILS_H_

#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xprof/protobuf/hardware_types.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

using ::tensorflow::profiler::DeviceCapabilities;

void SetDeviceCaps(const DeviceCapabilities& caps, XPlane* plane);
DeviceCapabilities GetDeviceCaps(const XPlane& plane);

}  // namespace profiler
}  // namespace tsl

#endif  // XPROF_UTILS_DEVICE_CAPS_UTILS_H_
