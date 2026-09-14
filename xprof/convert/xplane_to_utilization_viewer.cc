/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include "xprof/convert/xplane_to_utilization_viewer.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/overload.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/data_table_utils.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/tpu_counter_util.h"
#include "xprof/convert/tpu_generic_utilization_utils.h"
#include "xprof/utils/tpu_counter_ids_v6e.h"
#include "xprof/utils/tpu_counter_ids_v7x.h"

namespace xprof {

// Minimal definition for DeviceType enum after removing device_type_utils.h
// TODO(cliveverghese) : Adopt a generic approach for device types.
enum class ViewerDeviceType {
  UNKNOWN_DEVICE = 0,
  TPU_V7X = 12,
  TPU_V6E = 13,  // NOTE: For counter calculations, uses TPUv7.
};

namespace {

using ::nlohmann::json;
using ::tensorflow::profiler::DataTable;
using ::tensorflow::profiler::GetParam;
using ::tensorflow::profiler::TableColumn;
using ::tensorflow::profiler::ToolOptions;
using ::tsl::profiler::CreateTfXPlaneVisitor;
using ::tsl::profiler::GetStatTypeStr;
using ::tsl::profiler::kTpuPlanePrefix;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XPlane;
using ::tsl::profiler::XSpace;

// Hardcoded values from device_type_utils.cc/h
double GetTensorCoreFrequencyHz(ViewerDeviceType device_type) {
  switch (device_type) {
    case ViewerDeviceType::TPU_V6E:
      return 1.75e9;
    case ViewerDeviceType::TPU_V7X:
      return 1.9e9;
    default:
      return 1.0e9;  // Default
  }
}

bool IsTpuV6e(absl::string_view device_type) {
  return absl::StrContains(device_type, "TPU v6 Lite");
}

bool IsTpuV7x(absl::string_view device_type) {
  return absl::StrContains(device_type, "TPU v7x");
}

double GetPeakHbmBandwidthBps(ViewerDeviceType device_type) {
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  switch (device_type) {
    case ViewerDeviceType::TPU_V6E:
      return 1525.5 * kGiB;
    case ViewerDeviceType::TPU_V7X:
      return 3433.0 * kGiB;
    default:
      return 1.2e12;  // Default
  }
}

int GetNumMxus(ViewerDeviceType device_type) { return 2; }

int GetCyclesPerXlu(ViewerDeviceType device_type) {
  switch (device_type) {
    case ViewerDeviceType::TPU_V6E:
      return 4;
    case ViewerDeviceType::TPU_V7X:
      return 1;
    default:
      return 1;
  }
}

int GetTcCoreCount(ViewerDeviceType device_type) {
  switch (device_type) {
    case ViewerDeviceType::TPU_V6E:
      return 1;
    case ViewerDeviceType::TPU_V7X:
      return 2;
    default:
      return 1;
  }
}

int GetNumDies(ViewerDeviceType device_type) {
  switch (device_type) {
    case ViewerDeviceType::TPU_V6E:
      return 1;
    case ViewerDeviceType::TPU_V7X:
      return 2;
    default:
      return 1;
  }
}

int GetNumScCoresPerDie(ViewerDeviceType device_type) { return 2; }

// Helper to determine if we should process this device.
bool ShouldProcessDevice(absl::string_view device_type) {
  static const auto* const kSupportedDevices =
      new absl::flat_hash_set<std::string>{
          "TPU v7x",
          "TPU v6 Lite",
      };
  return kSupportedDevices->contains(device_type);
}

uint64_t GetCounterValue(std::variant<double, uint64_t> counter_value) {
  return std::visit(
      absl::Overload{
          [](double arg) -> uint64_t {
            if (std::isnan(arg)) return 0;
            // Ensure arg stays in bounds.
            if (arg <
                static_cast<double>(std::numeric_limits<uint64_t>::min())) {
              return 0;
            } else if (arg >= static_cast<double>(
                                  std::numeric_limits<uint64_t>::max())) {
              return std::numeric_limits<uint64_t>::max();
            } else {
              return static_cast<uint64_t>(arg);
            }
          },
          [](uint64_t arg) -> uint64_t { return arg; },
      },
      counter_value);
}

void ComputeAllTpuGenericUtilizations(const TpuCounterUtil& tpu_counters,
                                      ViewerDeviceType device_type_enum,
                                      UtilizationCounters* utilization) {
  xprof::TpuGenericUtilizationOptions options;
  options.num_mxu_per_tensor_core = GetNumMxus(device_type_enum);
  options.cycles_per_xlu_instruction = GetCyclesPerXlu(device_type_enum);
  options.is_tpu6e = (device_type_enum == ViewerDeviceType::TPU_V6E);
  options.frequency_hz = GetTensorCoreFrequencyHz(device_type_enum);
  options.peak_hbm_bw_bps = GetPeakHbmBandwidthBps(device_type_enum);

  // Iterate cores based on architecture.
  int num_tc_cores = GetTcCoreCount(device_type_enum);

  for (int core = 0; core < num_tc_cores; ++core) {
    // 1. Process TC Core
    xprof::ComputeTpuGenericTcUnitUtilization(tpu_counters, options, core,
                                              utilization);

    // 2. Process Bandwidth (HBM)
    xprof::ComputeTpuGenericBandwidthUtilization(tpu_counters, options, core,
                                                 utilization);

    // 3. Process ICI Bandwidth (Per Device/Chip)
    xprof::ComputeTpuGenericIciBandwidthUtilization(tpu_counters, options,
                                                    utilization);
  }

  // 4. Process SC Cores (interleaved per die/core)
  if (device_type_enum == ViewerDeviceType::TPU_V7X) {
    for (int die = 0; die < GetNumDies(device_type_enum); ++die) {
      for (int sc_core = 0; sc_core < GetNumScCoresPerDie(device_type_enum);
           ++sc_core) {
        xprof::ComputeTpuv7xScUnitUtilization(tpu_counters, die, sc_core,
                                              utilization);
      }
    }
  } else if (device_type_enum == ViewerDeviceType::TPU_V6E) {
    for (int sc_core = 0; sc_core < GetNumScCoresPerDie(device_type_enum);
         ++sc_core) {
      xprof::ComputeTpuv6eScUnitUtilization(tpu_counters, 0, sc_core,
                                            utilization);
    }
  }
}

}  // namespace

absl::StatusOr<std::string> ConvertXSpaceToUtilizationViewer(
    const XSpace& space) {
  DataTable data_table;
  // Columns matching UtilizationViewer::kColumns
  std::vector<TableColumn> columns = {
      TableColumn("host", "number", "Host"),
      TableColumn("device", "number", "Device"),
      TableColumn("sample", "number", "Sample"),
      TableColumn("node", "number", "Node"),
      TableColumn("name", "string", "Name"),
      TableColumn("achieved", "number", "Achieved"),
      TableColumn("peak", "number", "Peak"),
      TableColumn("unit", "string", "Unit"),
  };
  for (const auto& col : columns) {
    data_table.AddColumn(col);
  }

  for (const XPlane& plane : space.planes()) {
    if (!absl::StartsWith(plane.name(), kTpuPlanePrefix)) {
      continue;
    }

    auto visitor = CreateTfXPlaneVisitor(&plane);

    std::string device_type;
    int64_t host_id = 0;
    int64_t device_id = -1;

    visitor.ForEachStat([&](const tsl::profiler::XStatVisitor& stat) {
      if (stat.Type() == StatType::kDeviceId ||
          stat.Name() == GetStatTypeStr(StatType::kDeviceId)) {
        device_id = stat.IntOrUintValue();
      } else if (stat.Type() == StatType::kDeviceTypeString ||
                 stat.Name() == GetStatTypeStr(StatType::kDeviceTypeString)) {
        device_type = std::string(stat.StrOrRefValue());
      }
    });

    if (device_id == -1 || !ShouldProcessDevice(device_type)) {
      continue;
    }

    // Simplified device type logic checks
    ViewerDeviceType device_type_enum = ViewerDeviceType::UNKNOWN_DEVICE;
    if (IsTpuV7x(device_type)) {
      device_type_enum = ViewerDeviceType::TPU_V7X;
    } else if (IsTpuV6e(device_type)) {
      device_type_enum = ViewerDeviceType::TPU_V6E;
    }

    visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
      int64_t sample_id = line.Id();
      absl::flat_hash_map<uint64_t, uint64_t> counters_map;

      line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
        uint64_t counter_id = 0;
        std::variant<double, uint64_t> counter_value = 0.0;
        bool found_value = false;

        // 1. Extract Counter ID
        // Try precise StatType first
        auto id_stat = event.GetStat(StatType::kPerformanceCounterId);
        if (!id_stat) {
          id_stat = event.Metadata().GetStat(StatType::kPerformanceCounterId);
        }

        if (id_stat) {
          counter_id = static_cast<uint64_t>(id_stat->IntOrUintValue());
        }

        // Fallback to EventId if still 0
        if (counter_id == 0) counter_id = event.Id();

        // 2. Extract Counter Value
        auto val_stat = event.GetStat(StatType::kCounterValue);
        if (!val_stat) {
          val_stat = event.Metadata().GetStat(StatType::kCounterValue);
        }

        if (val_stat) {
          // IntOrUintValue fallback added here
          double double_value = val_stat->DoubleValue();
          if (double_value == 0.0) {
            counter_value = val_stat->IntOrUintValue();
          } else {
            counter_value = double_value;
          }
          found_value = true;
        }

        if (found_value && counter_id != 0) {
          counters_map[counter_id] = GetCounterValue(counter_value);
        }
      });

      if (counters_map.empty()) return;

      TpuCounterUtil tpu_counters(host_id, device_id, sample_id,
                                  std::move(counters_map));
      UtilizationCounters utilization;
      utilization.host_id = host_id;
      utilization.device_id = device_id;
      utilization.correlation_id = sample_id;

      ComputeAllTpuGenericUtilizations(tpu_counters, device_type_enum,
                                       &utilization);

      // Add metrics to table
      for (const auto& metric : utilization.metrics) {
        auto* row = data_table.AddRow();
        row->AddNumberCell(utilization.host_id);
        row->AddNumberCell(utilization.device_id);
        row->AddNumberCell(utilization.correlation_id);
        row->AddNumberCell(metric.node_id);
        row->AddTextCell(metric.metric);
        row->AddNumberCell(metric.achieved);
        row->AddNumberCell(metric.peak);
        row->AddTextCell(metric.unit);
      }
    });
  }

  return data_table.ToJson();
}

namespace {

std::string ExtractTimelineKernelName(
    const tsl::profiler::XPlaneVisitor& visitor) {
  std::string fallback_kernel_name = "";
  visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
    absl::string_view lname = line.Name();
    if (lname == "PALLAS" || lname == "XLA OPS" || lname == "Pallas" ||
        lname == "XLA Ops") {
      line.ForEachEvent([&](const tsl::profiler::XEventVisitor& ev) {
        if (!ev.Name().empty() && fallback_kernel_name.empty()) {
          fallback_kernel_name = std::string(ev.Name());
        }
      });
    }
  });
  return fallback_kernel_name;
}

absl::flat_hash_map<uint64_t, uint64_t> ParseCountersFromLine(
    const tsl::profiler::XLineVisitor& line, double* max_event_duration_us) {
  absl::flat_hash_map<uint64_t, uint64_t> counters_map;
  double event_max_duration_us = 0.0;

  line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
    double ev_dur = event.DurationNs() / 1000.0;
    if (ev_dur > event_max_duration_us) event_max_duration_us = ev_dur;

    uint64_t counter_id = 0;
    std::variant<double, uint64_t> counter_value = 0.0;
    bool found_value = false;

    auto id_stat = event.GetStat(StatType::kPerformanceCounterId);
    if (!id_stat) {
      id_stat = event.Metadata().GetStat(StatType::kPerformanceCounterId);
    }
    if (id_stat) {
      counter_id = static_cast<uint64_t>(id_stat->IntOrUintValue());
    }
    if (counter_id == 0) counter_id = event.Id();

    auto val_stat = event.GetStat(StatType::kCounterValue);
    if (!val_stat) {
      val_stat = event.Metadata().GetStat(StatType::kCounterValue);
    }
    if (val_stat) {
      double double_value = val_stat->DoubleValue();
      counter_value =
          (double_value == 0.0) ? val_stat->IntOrUintValue() : double_value;
      found_value = true;
    }

    if (found_value && counter_id != 0) {
      counters_map[counter_id] += GetCounterValue(counter_value);
    }
  });

  if (max_event_duration_us != nullptr) {
    *max_event_duration_us = event_max_duration_us;
  }
  return counters_map;
}

void ScaleNominalCycleCounters(
    ViewerDeviceType device_type_enum, double effective_duration_us,
    bool force_duration_override,
    absl::flat_hash_map<uint64_t, uint64_t>& counters_map) {
  double freq_hz = GetTensorCoreFrequencyHz(device_type_enum);
  double nominal_cycles_per_core = effective_duration_us * (freq_hz / 1e6);
  if (nominal_cycles_per_core <= 0.0) return;

  if (device_type_enum == ViewerDeviceType::TPU_V7X) {
    uint64_t c0 = TpuCounterIdsTpu7x::
        VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT
    uint64_t c1 = TpuCounterIdsTpu7x::
        VF_CHIP_DIE1_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT
    if (counters_map[c0] == 0 || force_duration_override) {
      counters_map[c0] = GetCounterValue(nominal_cycles_per_core);
    }
    if (counters_map[c1] == 0 || force_duration_override) {
      counters_map[c1] = GetCounterValue(nominal_cycles_per_core);
    }
  } else if (device_type_enum == ViewerDeviceType::TPU_V6E) {
    uint64_t c0 = TpuCounterIdsTpu6e::
        VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES;  // NOLINT
    if (counters_map[c0] == 0 || force_duration_override) {
      counters_map[c0] = GetCounterValue(nominal_cycles_per_core);
    }
  }
}

json ComputePrecisionBreakdown(double bf16, double i8, double i4, double fp8) {
  double total_mxu_cycles = bf16 + i8 + i4 + fp8;
  json precision_breakdown = json::object();
  if (total_mxu_cycles > 0) {
    precision_breakdown["BF16"] =
        std::round((bf16 / total_mxu_cycles) * 10000.0) / 100.0;
    precision_breakdown["Int8"] =
        std::round((i8 / total_mxu_cycles) * 10000.0) / 100.0;
    precision_breakdown["Int4"] =
        std::round((i4 / total_mxu_cycles) * 10000.0) / 100.0;
    precision_breakdown["FP8"] =
        std::round((fp8 / total_mxu_cycles) * 10000.0) / 100.0;
  } else {
    precision_breakdown["BF16"] = 0.0;
    precision_breakdown["Int8"] = 0.0;
    precision_breakdown["Int4"] = 0.0;
    precision_breakdown["FP8"] = 0.0;
  }
  return precision_breakdown;
}

struct MetricAgg {
  double achieved = 0.0;
  double peak = 0.0;
};

json AggregateKernelMetrics(
    const UtilizationCounters& utilization,
    double& avg_mxu_achieved, double& avg_mxu_peak,
    double& bf16_achieved, double& i8_achieved,
    double& i4_achieved, double& fp8_achieved) {
  absl::flat_hash_map<std::string, MetricAgg> other_metrics_map;

  for (const auto& metric : utilization.metrics) {
    if (metric.metric == "Avg MXU Busy") {
      avg_mxu_achieved += metric.achieved;
      avg_mxu_peak += metric.peak;
    } else if (metric.metric == "MXU BF16") {
      bf16_achieved += metric.achieved;
    } else if (metric.metric == "MXU I8") {
      i8_achieved += metric.achieved;
    } else if (metric.metric == "MXU I4") {
      i4_achieved += metric.achieved;
    } else if (metric.metric == "MXU E4M3 + E5M2") {
      fp8_achieved += metric.achieved;
    } else if (metric.peak > 0) {
      std::string metric_name = metric.metric;
      if (absl::StartsWith(metric_name, "HBM Rd+Wr")) {
        metric_name = "HBM Bandwidth Utilization";
      }
      auto& agg = other_metrics_map[metric_name];
      agg.achieved += metric.achieved;
      agg.peak += metric.peak;
    }
  }

  json other_metrics = json::object();
  for (const auto& [metric_name, agg] : other_metrics_map) {
    if (agg.peak > 0) {
      other_metrics[metric_name] =
          std::round((agg.achieved / agg.peak) * 10000.0) / 100.0;
    }
  }
  return other_metrics;
}

std::optional<json> ProcessDevicePlane(
    const XPlane& plane, const std::string& kernel_filter,
    double duration_us_param, bool force_duration_override,
    int64_t device_filter) {
  if (!absl::StartsWith(plane.name(), kTpuPlanePrefix)) return std::nullopt;
  auto visitor = CreateTfXPlaneVisitor(&plane);

  std::string device_type;
  int64_t host_id = 0;
  int64_t device_id = -1;

  visitor.ForEachStat([&](const tsl::profiler::XStatVisitor& stat) {
    if (stat.Type() == StatType::kDeviceId ||
        stat.Name() == GetStatTypeStr(StatType::kDeviceId)) {
      device_id = stat.IntOrUintValue();
    } else if (stat.Type() == StatType::kDeviceTypeString ||
               stat.Name() == GetStatTypeStr(StatType::kDeviceTypeString)) {
      device_type = std::string(stat.StrOrRefValue());
    }
  });

  // Fallback: parse device ID from plane name /device:TPU:<id>
  if (device_id == -1) {
    absl::string_view plane_name = plane.name();
    static constexpr absl::string_view kPrefix = "/device:TPU:";
    if (absl::StartsWith(plane_name, kPrefix)) {
      if (!absl::SimpleAtoi(plane_name.substr(kPrefix.size()), &device_id)) {
        device_id = -1;
      }
    }
  }

  if (device_id == -1 || !ShouldProcessDevice(device_type)) return std::nullopt;
  if (device_filter >= 0 && device_id != device_filter) return std::nullopt;

  ViewerDeviceType device_type_enum = ViewerDeviceType::UNKNOWN_DEVICE;
  if (IsTpuV7x(device_type)) {
    device_type_enum = ViewerDeviceType::TPU_V7X;
  } else if (IsTpuV6e(device_type)) {
    device_type_enum = ViewerDeviceType::TPU_V6E;
  }

  std::string fallback_kernel_name = ExtractTimelineKernelName(visitor);

  json device_json;
  device_json["device_id"] = device_id;
  device_json["device_type"] = device_type.empty() ? "TPU" : device_type;
  device_json["kernels"] = json::array();

  visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
    int64_t sample_id = line.Id();
    absl::string_view line_name = line.Name();
    std::string kernel_name = "";

    if (absl::StartsWith(line_name, "counters_")) {
      kernel_name = std::string(line_name.substr(9));
      if (kernel_name == "0") kernel_name = fallback_kernel_name;
    } else if (line_name == "_counters_" || line_name == "counters") {
      kernel_name = fallback_kernel_name;
    }
    if (kernel_name.empty()) kernel_name = "default_kernel";

    if (!kernel_filter.empty() &&
        !absl::StrContains(kernel_name, kernel_filter)) {
      return;
    }

    double event_max_duration_us = 0.0;
    auto counters_map = ParseCountersFromLine(line, &event_max_duration_us);
    if (counters_map.empty()) return;

    double effective_duration_us =
        (duration_us_param > 0.0) ? duration_us_param : event_max_duration_us;

    ScaleNominalCycleCounters(device_type_enum, effective_duration_us,
                              force_duration_override, counters_map);

    TpuCounterUtil tpu_counters(host_id, device_id, sample_id,
                                std::move(counters_map));
    UtilizationCounters utilization;
    utilization.host_id = host_id;
    utilization.device_id = device_id;
    utilization.correlation_id = sample_id;

    ComputeAllTpuGenericUtilizations(tpu_counters, device_type_enum,
                                     &utilization);

    json kernel_json;
    kernel_json["kernel_name"] = kernel_name;
    kernel_json["duration_us"] = effective_duration_us;

    double avg_mxu_achieved = 0.0, avg_mxu_peak = 0.0;
    double bf16_achieved = 0.0, i8_achieved = 0.0, i4_achieved = 0.0,
           fp8_achieved = 0.0;

    json other_metrics = AggregateKernelMetrics(
        utilization, avg_mxu_achieved, avg_mxu_peak, bf16_achieved,
        i8_achieved, i4_achieved, fp8_achieved);

    json precision_breakdown = ComputePrecisionBreakdown(
        bf16_achieved, i8_achieved, i4_achieved, fp8_achieved);

    double mxu_util =
        (avg_mxu_peak > 0)
            ? std::round((avg_mxu_achieved / avg_mxu_peak) * 10000.0) / 100.0
            : 0.0;

    kernel_json["mxu_utilization"] = mxu_util;
    kernel_json["mxu_is_anomaly"] = (mxu_util > 100.0);
    kernel_json["mxu_cycles_breakdown"] = precision_breakdown;
    kernel_json["other_metrics"] = other_metrics;

    device_json["kernels"].push_back(kernel_json);
  });

  return device_json;
}

}  // namespace

absl::StatusOr<std::string> ConvertXSpaceToKernelUtilization(
    const XSpace& space, const ToolOptions& options) {
  std::string kernel_filter = "";
  if (auto k = GetParam<std::string>(options, "kernel")) {
    kernel_filter = *k;
  } else if (auto kn = GetParam<std::string>(options, "kernel_name")) {
    kernel_filter = *kn;
  }

  double duration_us_param = 0.0;
  if (auto di = GetParam<int>(options, "duration_us")) {
    duration_us_param = static_cast<double>(*di);
  } else if (auto ds = GetParam<std::string>(options, "duration_us")) {
    if (!absl::SimpleAtod(*ds, &duration_us_param)) {
      duration_us_param = 0.0;
    }
  }

  bool force_duration_override = false;
  if (auto f = GetParam<bool>(options, "force_duration")) {
    force_duration_override = *f;
  }

  int64_t device_filter = -1;
  if (auto dev = GetParam<int>(options, "device_id")) {
    device_filter = *dev;
  } else if (auto devs = GetParam<std::string>(options, "device_id")) {
    if (!absl::SimpleAtoi(*devs, &device_filter)) {
      device_filter = -1;
    }
  }

  json root_json;
  root_json["status"] = "SUCCESS";
  root_json["devices"] = json::array();

  for (const XPlane& plane : space.planes()) {
    auto device_json = ProcessDevicePlane(
        plane, kernel_filter, duration_us_param, force_duration_override,
        device_filter);
    if (device_json.has_value()) {
      root_json["devices"].push_back(std::move(*device_json));
    }
  }

  return root_json.dump(2);
}

}  // namespace xprof
