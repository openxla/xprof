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

#include "xprof/convert/op_stats_to_pod_stats.h"

#include <algorithm>
#include <initializer_list>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/gtl/map_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "plugin/xprof/protobuf/steps_db.pb.h"
#include "xprof/utils/diagnostics.h"
#include "xprof/utils/event_span.h"

namespace tensorflow {
namespace profiler {
using tsl::uint64;

namespace {

PodStatsRecord CreatePodStatsRecord(absl::string_view host_name,
                                    const StepInfoResult& step_info) {
  PodStatsRecord record;
  GenericStepBreakdown generic;
  bool success = step_info.step_breakdown().UnpackTo(&generic);
  DCHECK(success);
  record.set_host_name(tsl::string(host_name));
  record.set_step_num(step_info.step_num());
  record.set_total_duration_us(
      tsl::profiler::PicoToMicro(step_info.duration_ps()));
  auto& step_breakdown_map = *record.mutable_step_breakdown_us();
  std::vector<std::pair<uint64, absl::string_view>> metrics;

  auto add_event = [&](GenericEventType type,
                       std::initializer_list<EventType> event_list) {
    uint64 ps = 0;
    for (const auto& event_type : event_list) {
      ps +=
          tsl::gtl::FindWithDefault(generic.type_ps(), event_type, /*value=*/0);
    }
    step_breakdown_map[type] = tsl::profiler::PicoToMicro(ps);
    metrics.emplace_back(ps, GetGenericEventTypeStr(type));
  };

  add_event(kDeviceCompute, {DEVICE_COMPUTE_32, DEVICE_COMPUTE_16});
  add_event(kDeviceToDevice, {DEVICE_TO_DEVICE, DEVICE_WAIT_DEVICE});
  add_event(kDeviceCollectives, {DEVICE_COLLECTIVES});
  add_event(kHostCompute, {HOST_COMPUTE});
  add_event(kHostPrepare, {HOST_PREPARE});
  add_event(kInput, {HOST_WAIT_INPUT, HOST_TO_DEVICE, DEVICE_WAIT_HOST});
  add_event(kOutput, {DEVICE_TO_HOST});
  add_event(kCompile, {HOST_COMPILE});
  add_event(kAllOthers, {UNKNOWN_TIME});

  std::sort(metrics.begin(), metrics.end());
  record.set_bottleneck(metrics.back().second.data(),
                        metrics.back().second.size());
  return record;
}

}  // namespace

PodStatsDatabase ConvertOpStatsToPodStats(const OpStats& op_stats) {
  PodStatsDatabase pod_stats_db;
  const auto& core_id_map = op_stats.core_id_to_details();
  for (int i = GenericEventType::kFirstGenericEventType;
       i <= GenericEventType::kLastGenericEventType; i++) {
    auto& event = *pod_stats_db.add_step_breakdown_events();
    event.set_id(i);
    absl::string_view type_str =
        GetGenericEventTypeStr(static_cast<GenericEventType>(i));
    event.set_name(type_str.data(), type_str.size());
  }

  for (const auto& step_sequence : op_stats.step_db().step_sequence()) {
    for (const auto& entry : step_sequence.step_info_per_core()) {
      if (!core_id_map.contains(entry.first)) {
        LOG(WARNING) << "core_id_map does not contain " << entry.first;
        continue;
      }
      const CoreDetails& details = core_id_map.at(entry.first);
      *pod_stats_db.add_pod_stats_record() =
          CreatePodStatsRecord(details.hostname(), entry.second);
    }
  }
  PopulateStepDiagnostics(op_stats, pod_stats_db.mutable_diagnostics());
  return pod_stats_db;
}

}  // namespace profiler
}  // namespace tensorflow
