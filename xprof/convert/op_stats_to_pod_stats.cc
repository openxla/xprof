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
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/gtl/map_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xprof/convert/profile_time_breakdown.h"
#include "plugin/xprof/protobuf/steps_db.pb.h"
#include "xprof/utils/diagnostics.h"
#include "xprof/utils/event_span.h"
#include "xprof/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {

namespace {

PodStatsRecord CreatePodStatsRecord(absl::string_view host_name,
                                    const StepInfoResult& step_info) {
  PodStatsRecord record;
  GenericStepBreakdown generic;
  bool success = step_info.step_breakdown().UnpackTo(&generic);
  DCHECK(success);
  record.set_host_name(host_name);
  record.set_step_num(step_info.step_num());
  record.set_total_duration_us(
      tsl::profiler::PicoToMicro(step_info.duration_ps()));
  auto& step_breakdown_map = *record.mutable_step_breakdown_us();
  std::vector<std::pair<uint64_t, absl::string_view>> metrics;

  auto add_event_ps = [&](GenericEventType type, uint64_t ps) {
    step_breakdown_map[type] = tsl::profiler::PicoToMicro(ps);
    metrics.emplace_back(ps, GetGenericEventTypeStr(type));
  };

  if (generic.type_ps().empty() && !generic.category_ps().empty()) {
    ProfileTimeBreakdown time_breakdown;
    uint64_t total_category_ps = 0;
    for (const auto& [category, time_ps] : generic.category_ps()) {
      if (category == kIdle) continue;
      time_breakdown.IncrementCategoryTimePs(category, time_ps);
      total_category_ps += time_ps;
    }
    time_breakdown.SetProfileTimePs(
        std::max(step_info.duration_ps(), total_category_ps));
    time_breakdown.BreakdownSparseCoreV0Infeed();

    uint64_t infeed_ps = time_breakdown.InfeedTimePs() +
                         time_breakdown.HostRecvTimePs() +
                         time_breakdown.SparseCoreV0InfeedWaitTimePs() +
                         time_breakdown.SparseCoreV0InfeedTransformTimePs();
    uint64_t outfeed_ps = time_breakdown.OutfeedTimePs() +
                          time_breakdown.HostSendTimePs() +
                          time_breakdown.SparseCoreV0OutfeedTimePs();
    uint64_t collectives_ps = time_breakdown.AllReduceOrAllToAllTimePs();
    uint64_t d2d_ps = time_breakdown.SendTimePs() +
                      time_breakdown.RecvTimePs() +
                      time_breakdown.MegacoreFusionTimePs();
    uint64_t non_compute_busy_ps =
        collectives_ps + time_breakdown.SendTimePs() +
        time_breakdown.RecvTimePs() +
        time_breakdown.SparseCoreV0InfeedTransformTimePs() +
        time_breakdown.SparseCoreV0OutfeedTimePs();
    uint64_t busy_ps = time_breakdown.TensorCoreBusyTimePs();
    uint64_t compute_ps =
        busy_ps > non_compute_busy_ps ? busy_ps - non_compute_busy_ps : 0;
    uint64_t idle_ps = time_breakdown.IdleTimePs();

    add_event_ps(kDeviceCompute, compute_ps);
    add_event_ps(kDeviceToDevice, d2d_ps);
    add_event_ps(kDeviceCollectives, collectives_ps);
    add_event_ps(kHostCompute, 0);
    add_event_ps(kHostPrepare, 0);
    add_event_ps(kInput, infeed_ps);
    add_event_ps(kOutput, outfeed_ps);
    add_event_ps(kCompile, 0);
    add_event_ps(kAllOthers, idle_ps);
  } else {
    auto add_event = [&](GenericEventType type,
                         std::initializer_list<EventType> event_list) {
      uint64_t ps = 0;
      for (const auto& event_type : event_list) {
        ps += tsl::gtl::FindWithDefault(generic.type_ps(), event_type,
                                        /*value=*/0);
      }
      add_event_ps(type, ps);
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
  }

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
