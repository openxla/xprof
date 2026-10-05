#include "xprof/convert/xplane_to_perf_counters.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/data_table_utils.h"
#include "xprof/convert/unified_session_snapshot.h"
#ifdef EMBEDDED_FEATURES_ENABLED
#include "xprof/embedded/perf_counters/perf_counters_db.h"
#endif

namespace tensorflow {
namespace profiler {

namespace {
using ::tsl::profiler::CreateTfXPlaneVisitor;
using ::tsl::profiler::kGpuPlanePrefix;
using ::tsl::profiler::kTpuPlanePrefix;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventVisitor;
using ::tsl::profiler::XLineVisitor;
using ::tsl::profiler::XPlaneVisitor;
using ::tsl::profiler::XStatVisitor;
}  // namespace

void ConvertXSpaceToPerfCounters(const XSpace* space,
                                 absl::string_view hostname,
                                 DataTable* data_table) {
  if (data_table->GetColumns().empty()) {
    data_table->AddColumn(TableColumn("Host", "string", "Host"));
    data_table->AddColumn(TableColumn("Chip", "number", "Chip"));
    data_table->AddColumn(TableColumn("Kernel", "string", "Kernel"));
    data_table->AddColumn(TableColumn("Sample", "number", "Sample"));
    data_table->AddColumn(TableColumn("Counter", "string", "Counter"));
    data_table->AddColumn(TableColumn("Value", "number", "Value (Hex)"));
    data_table->AddColumn(TableColumn("Description", "string", "Description"));
    data_table->AddColumn(TableColumn("Set", "string", "Set"));
  }

  for (const XPlane& plane : space->planes()) {
    if (!absl::StartsWith(plane.name(), kTpuPlanePrefix) &&
        !absl::StartsWith(plane.name(), kGpuPlanePrefix)) {
      continue;
    }

    XPlaneVisitor visitor = CreateTfXPlaneVisitor(&plane);
    int64_t chip_id = -1;
    absl::string_view device_type;
    visitor.ForEachStat([&](const XStatVisitor& stat) {
      if (stat.Type() == StatType::kGlobalChipId) {
        chip_id = stat.IntOrUintValue();
      } else if (stat.Type() == StatType::kDeviceTypeString) {
        device_type = absl::string_view(stat.StrOrRefValue());
      }
    });
    if (chip_id == -1) {
      visitor.ForEachStat([&](const XStatVisitor& stat) {
        if (stat.Type() == StatType::kDeviceId) {
          chip_id = stat.IntOrUintValue();
        }
      });
    }
    if (chip_id == -1) continue;

#ifdef EMBEDDED_FEATURES_ENABLED
    const xprof::embedded::PerfCounterMap& counter_map =
        xprof::embedded::GetPerfCounterMapForDevice(device_type);
#endif

    visitor.ForEachLine([&](const XLineVisitor& line) {
      line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
        std::optional<XStatVisitor> counter_value_stat =
            event.GetStat(StatType::kCounterValue);
        if (!counter_value_stat) return;

        uint64_t value = counter_value_stat->IntOrUintValue();

        std::optional<XStatVisitor> id_stat =
            event.GetStat(StatType::kPerformanceCounterId);
        if (!id_stat) {
          id_stat = event.Metadata().GetStat(StatType::kPerformanceCounterId);
        }

        absl::string_view description;
        absl::string_view counter_sets;
        absl::string_view counter_name = event.Name();
#ifdef EMBEDDED_FEATURES_ENABLED
        if (id_stat) {
          uint64_t counter_id =
              static_cast<uint64_t>(id_stat->IntOrUintValue());
          if (auto it = counter_map.find(counter_id); it != counter_map.end()) {
            description = it->second.description;
            counter_sets = it->second.counter_sets;
          }
        }
        // Fallback to read the XStats for the description and counter sets
        // if they are not found in the embedded database. We can safely
        // deprecate this in the future.
#endif
        if (description.empty()) {
          std::optional<XStatVisitor> description_stat =
              event.GetStat(StatType::kPerformanceCounterDescription);
          if (!description_stat) {
            description_stat = event.Metadata().GetStat(
                StatType::kPerformanceCounterDescription);
          }
          if (description_stat) {
            description = description_stat->StrOrRefValue();
          }
        }
        if (counter_sets.empty()) {
          std::optional<XStatVisitor> set_stat =
              event.GetStat(StatType::kPerformanceCounterSets);
          if (!set_stat) {
            set_stat =
                event.Metadata().GetStat(StatType::kPerformanceCounterSets);
          }
          if (set_stat) {
            counter_sets = set_stat->StrOrRefValue();
          }
        }

        data_table->AddRow()
            ->AddTextCell(hostname)
            .AddNumberCell(chip_id)
            .AddTextCell(line.Name())
            .AddNumberCell(line.Id())
            .AddTextCell(absl::AsciiStrToLower(counter_name))
            .AddHexCell(value)
            .AddTextCell(description)
            .AddTextCell(counter_sets);
      });
    });
  }
}

absl::StatusOr<std::string> ConvertMultiXSpacesToPerfCounters(
    const xprof::XprofSessionSnapshot& session_snapshot) {
  DataTable data_table;

  // Ensure columns are added even if there are no XSpaces.
  if (data_table.GetColumns().empty()) {
    data_table.AddColumn(TableColumn("Host", "string", "Host"));
    data_table.AddColumn(TableColumn("Chip", "number", "Chip"));
    data_table.AddColumn(TableColumn("Kernel", "string", "Kernel"));
    data_table.AddColumn(TableColumn("Sample", "number", "Sample"));
    data_table.AddColumn(TableColumn("Counter", "string", "Counter"));
    data_table.AddColumn(TableColumn("Value", "number", "Value (Hex)"));
    data_table.AddColumn(TableColumn("Description", "string", "Description"));
    data_table.AddColumn(TableColumn("Set", "string", "Set"));
  }

  for (size_t i = 0; i < session_snapshot.XSpaceSize(); ++i) {
    google::protobuf::Arena arena;
    auto xspace_or = session_snapshot.GetXSpace(i, &arena);
    if (!xspace_or.ok()) continue;
    XSpace* space = xspace_or.value();
    std::string hostname = session_snapshot.GetHostname(i);

    ConvertXSpaceToPerfCounters(space, hostname, &data_table);
  }

  return data_table.ToJson();
}

}  // namespace profiler
}  // namespace tensorflow
