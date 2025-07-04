/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XPROF_CONVERT_XPLANE_TO_OP_METRICS_DB_H_
#define XPROF_CONVERT_XPLANE_TO_OP_METRICS_DB_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/op_metrics_db_combiner.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "xprof/utils/op_utils.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::GetDeviceEventTimespan;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XLineVisitor;
using ::tsl::profiler::XPlaneVisitor;
using ::tsl::profiler::XStatVisitor;

// Data per host thread for TensorFlow Op Metrics Database.
struct TfMetricsDbData {
  // A database of TF-Op metrics for this core.
  OpMetricsDb tf_metrics_db;
  HostOpMetricsDbBuilder tf_metrics_db_builder{&tf_metrics_db};
};

absl::flat_hash_map<int64_t, tsl::profiler::TfOp>
CollectTfOpsFromHostThreadsXPlane(const XPlane& host_trace);

TfMetricsDbData ConvertHostThreadsXLineToTfMetricsDbData(
    const XLineVisitor& line,
    const absl::flat_hash_map<int64_t, tsl::profiler::TfOp>& tf_ops);

void ConsumeTfMetricsDbData(TfMetricsDbData src, OpMetricsDbCombiner* dst);

OpMetricsDb ConvertHostThreadsXPlaneToOpMetricsDb(const XPlane& host_trace);

// Converts GPU device trace to OpMetricsDb.
// Will use HloModuleMap to source performance info for cost analysis.
OpMetricsDb ConvertDeviceTraceXPlaneToOpMetricsDb(
    const XPlane& device_trace, const HloModuleMap& hlo_module_map);

// Convert TPU DeviceTrace XPlane to OpMetricDb
OpMetricsDb ConvertTpuDeviceTraceXPlaneToOpMetricsDb(
    const XPlane& device_trace);

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_XPLANE_TO_OP_METRICS_DB_H_
