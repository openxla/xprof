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

#ifndef XPROF_CONVERT_XPLANE_TO_KERNEL_STATS_DB_H_
#define XPROF_CONVERT_XPLANE_TO_KERNEL_STATS_DB_H_

#include <functional>
#include <memory>
#include <string>

#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/data_table_utils.h"
#include "plugin/tensorboard_plugin_profile/protobuf/kernel_stats.pb.h"
#include "xprof/utils/gpu_event_stats.h"
#include "xprof/utils/kernel_stats_utils.h"

namespace tensorflow {
namespace profiler {

void ConvertDeviceTraceXPlaneToKernelReports(
    const XPlane& device_trace,
    const std::function<void(const GpuEventStats&, KernelReport*)>&
        on_kernel_fn,
    KernelReportMap* reports);

std::unique_ptr<DataTable> GenerateKernelStatsDataTable(
    const KernelStatsDb& kernel_stats_db);

std::string KernelStatsToDataTableJson(const KernelStatsDb& kernel_stats_db);

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_XPLANE_TO_KERNEL_STATS_DB_H_
