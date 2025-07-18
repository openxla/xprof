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

#ifndef XPROF_CONVERT_XPLANE_TO_MEMORY_PROFILE_H_
#define XPROF_CONVERT_XPLANE_TO_MEMORY_PROFILE_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "plugin/xprof/protobuf/memory_profile.pb.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::StatType;
using tsl::profiler::XEventVisitor;
using tsl::profiler::XLineVisitor;
using tsl::profiler::XPlaneBuilder;
using tsl::profiler::XPlaneVisitor;
using tsl::profiler::XStatVisitor;

// Process the host threads XPlane and generate MemoryProfile result; at most
// max_num_snapshots will be displayed on the UI.
// REQUIRED: host_plane should have been grouped by calling GroupTfEvents().
MemoryProfile ConvertXPlaneToMemoryProfile(const XPlane& host_plane,
                                           int64_t max_num_snapshots = 1000);

absl::Status ConvertXSpaceToMemoryProfileJson(const XSpace& xspace,
                                              std::string* json_output);
}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_XPLANE_TO_MEMORY_PROFILE_H_
