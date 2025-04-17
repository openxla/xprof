/* Copyright 2025 The XProf Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_UTILS_TENSORFLOW_UTILS_H_
#define THIRD_PARTY_XPROF_UTILS_TENSORFLOW_UTILS_H_

#include <string>

#include "absl/status/status.h"
#include "google/protobuf/message.h"
#include "xla/tsl/platform/types.h"
#include "plugin/tensorboard_plugin_profile/protobuf/tensorflow_datatypes.pb.h"

namespace tensorflow {
namespace profiler {
enum { kDataTypeRefOffset = 100 };
inline bool IsRefType(TensorflowDataType dtype) {
  return dtype > static_cast<TensorflowDataType>(kDataTypeRefOffset);
}
tsl::string DataTypeString(TensorflowDataType dtype);
tsl::string DataTypeStringInternal(TensorflowDataType dtype);
absl::Status ParseTextFormatFromString(std::string input,
                                       google::protobuf::Message* output);
}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_UTILS_TENSORFLOW_UTILS_H_
