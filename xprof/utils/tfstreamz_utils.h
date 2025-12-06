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
#ifndef XPROF_UTILS_TFSTREAMZ_UTILS_H_
#define XPROF_UTILS_TFSTREAMZ_UTILS_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "xla/tsl/lib/monitoring/collected_metrics.h"
#include "xla/tsl/platform/types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

struct TfStreamzSnapshot {
  std::unique_ptr<tsl::monitoring::CollectedMetrics> metrics;
  uint64_t start_time_ns;  // time before collection.
  uint64_t end_time_ns;    // time after collection.
};

absl::Status SerializeToXPlane(const std::vector<TfStreamzSnapshot>& snapshots,
                               XPlane* plane, uint64_t line_start_time_ns);

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_UTILS_TFSTREAMZ_UTILS_H_
