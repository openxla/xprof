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

#ifndef THIRD_PARTY_XPROF_CONVERT_XPLANE_TO_FLAT_OP_METRICS_DB_H_
#define THIRD_PARTY_XPROF_CONVERT_XPLANE_TO_FLAT_OP_METRICS_DB_H_

#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"

namespace tensorflow {
namespace profiler {

void ConvertSparseCoreDeviceTraceXPlaneToFlatOpMetricsDb(
    const XPlane& device_trace,
    absl::flat_hash_map<std::pair<uint64_t, uint64_t>, FlatOpMetricsDb>&
        sparse_core_metrics_map);

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_XPLANE_TO_FLAT_OP_METRICS_DB_H_
