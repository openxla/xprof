/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_CONVERT_XPLANE_TO_PERF_COUNTERS_H_
#define THIRD_PARTY_XPROF_CONVERT_XPLANE_TO_PERF_COUNTERS_H_

#include <string>

#include "absl/status/statusor.h"
#include "xprof/convert/repository.h"

namespace tensorflow {
namespace profiler {

// Converts performance counter events in XSpace to a DataTable JSON string for
// the perf_counters tool.
absl::StatusOr<std::string> ConvertMultiXSpacesToPerfCounters(
    const SessionSnapshot& session_snapshot);

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_XPLANE_TO_PERF_COUNTERS_H_
