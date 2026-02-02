#include "xprof/convert/xplane_to_utilization_viewer.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xprof/convert/data_table_utils.h"
#include "xprof/convert/tpu_counter_util.h"
#include "xprof/convert/tpuv7generic_utilization_utils.h"

namespace xprof {

// Minimal definition for DeviceType enum after removing device_type_utils.h
// TODO(cliveverghese) : Adopt a generic approach for device types.
enum class ViewerDeviceType {
  UNKNOWN_DEVICE = 0,
  TPU_V7X = 12,
};

namespace {

using ::tensorflow::profiler::DataTable;
using ::tensorflow::profiler::TableColumn;
using ::tsl::profiler::CreateTfXPlaneVisitor;
using ::tsl::profiler::kTpuPlanePrefix;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XPlane;
using ::tsl::profiler::XSpace;

// Hardcoded values from device_type_utils.cc/h
double GetTensorCoreFrequencyHz(ViewerDeviceType device_type) {
  if (device_type == ViewerDeviceType::TPU_V7X) {
    return 1.9e9;
  }
  return 1.0e9;  // Default
}

double GetPeakHbmBandwidthBps(ViewerDeviceType device_type) {
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  if (device_type == ViewerDeviceType::TPU_V7X) {
    return 3433.0 * kGiB;
  }
  return 1.2e12;  // Default
}

int GetNumMxus(ViewerDeviceType device_type) { return 2; }

int GetCyclesPerXlu(ViewerDeviceType device_type) {
  return 1;  // TPU v7x and default
}

int GetTcCoreCount(ViewerDeviceType device_type) { return 2; }

bool IsTpuV7x(absl::string_view device_type) {
  return absl::StrContains(device_type, "TPU v7x");
}

// Helper to determine if we should process this device.
bool ShouldProcessDevice(absl::string_view device_type) {
  return IsTpuV7x(device_type);
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

    // Create visitor first to use helper methods
    auto visitor = CreateTfXPlaneVisitor(&plane);

    if (!absl::StartsWith(plane.name(), kTpuPlanePrefix)) {
      continue;
    }

    std::string device_type;
    int64_t host_id = 0;
    int64_t device_id = -1;

    visitor.ForEachStat([&](const tsl::profiler::XStatVisitor& stat) {
      if (stat.Type() == StatType::kDeviceId ||
          stat.Name() == GetStatTypeStr(StatType::kDeviceId)) {
        device_id = stat.IntOrUintValue();
      } else if (stat.Type() == StatType::kDeviceTypeString) {
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
    }

    visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
      int64_t sample_id = line.Id();
      absl::flat_hash_map<uint64_t, uint64_t> counters_map;

      line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
        uint64_t counter_id = 0;
        double counter_value = 0.0;
        bool found_value = false;

        // 1. Extract Counter ID
        // Try precise StatType first
        auto id_stat = event.GetStat(StatType::kPerformanceCounterId);
        if (!id_stat) {
          id_stat = event.Metadata().GetStat(StatType::kPerformanceCounterId);
        }

        counter_id = static_cast<uint64_t>(id_stat->IntOrUintValue());

        // Fallback to EventId if still 0
        if (counter_id == 0) counter_id = event.Id();

        // 2. Extract Counter Value
        auto val_stat = event.GetStat(StatType::kCounterValue);
        if (!val_stat) {
          val_stat = event.Metadata().GetStat(StatType::kCounterValue);
        }

        if (val_stat) {
          // IntOrUintValue fallback added here
          counter_value = val_stat->DoubleValue();
          if (counter_value == 0.0) {
            counter_value = static_cast<double>(val_stat->IntOrUintValue());
          }
          found_value = true;
        }

        if (found_value && counter_id != 0) {
          counters_map[counter_id] = static_cast<uint64_t>(counter_value);
        }
      });

      if (counters_map.empty()) return;

      TpuCounterUtil tpu_counters(host_id, device_id, sample_id,
                                  std::move(counters_map));
      UtilizationCounters utilization;
      utilization.host_id = host_id;
      utilization.device_id = device_id;
      utilization.correlation_id = sample_id;

      xprof::Tpuv7GenericUtilizationOptions options;
      options.num_mxu_per_tensor_core = GetNumMxus(device_type_enum);
      options.cycles_per_xlu_instruction = GetCyclesPerXlu(device_type_enum);
      options.is_tpu7 = (device_type_enum != ViewerDeviceType::TPU_V7X);
      options.frequency_hz = GetTensorCoreFrequencyHz(device_type_enum);
      options.peak_hbm_bw_bps = GetPeakHbmBandwidthBps(device_type_enum);

      // Iterate cores based on architecture.
      int num_tc_cores = GetTcCoreCount(device_type_enum);

      for (int core = 0; core < num_tc_cores; ++core) {
        // 1. Process TC Core (Generic V7 logic applies to V7 and V7x TC)
        xprof::ComputeTpuv7GenericTcUnitUtilization(tpu_counters, options, core,
                                                    &utilization);

        // 2. Process Bandwidth (HBM)
        xprof::ComputeTpuv7GenericBandwidthUtilization(tpu_counters, options,
                                                       core, &utilization);

        // 3. Process ICI Bandwidth (Per Device/Chip)
        // ExtractUtilizationCounters calls this inside the loop, so we
        // duplicate to match regression test behavior.
        xprof::ComputeTpuv7GenericIciBandwidthUtilization(tpu_counters, options,
                                                          &utilization);
      }

      // 4. Process SC Cores (V7x only, interleaved per die/core)
      // Must come AFTER TC/HBM metrics to match ExtractUtilizationCounters
      // (tpu_counter_util.cc)
      if (device_type_enum == ViewerDeviceType::TPU_V7X) {
        for (int die = 0; die < 2; ++die) {
          for (int sc_core = 0; sc_core < 2; ++sc_core) {
            xprof::ComputeTpuv7xScUnitUtilization(tpu_counters, die, sc_core,
                                                  &utilization);
          }
        }
      }

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

}  // namespace xprof
