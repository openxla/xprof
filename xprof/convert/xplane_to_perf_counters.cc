#include "xprof/convert/xplane_to_perf_counters.h"

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
#include "xprof/convert/repository.h"

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
    visitor.ForEachStat([&](const XStatVisitor& stat) {
      if (stat.Type() == StatType::kGlobalChipId) {
        chip_id = stat.IntOrUintValue();
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

    visitor.ForEachLine([&](const XLineVisitor& line) {
      line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
        std::optional<XStatVisitor> counter_value_stat =
            event.GetStat(StatType::kCounterValue);
        if (!counter_value_stat) return;

        uint64_t value = counter_value_stat->IntOrUintValue();
        // Assuming we want to show all including zeros unless filtered by
        // frontend But existing code filtered zeros. Let's keep it consistent
        // if needed, but frontend usually handles filtering via args. For
        // now, let's include everything to be safe.

        std::optional<XStatVisitor> description_stat =
            event.GetStat(StatType::kPerformanceCounterDescription);
        std::optional<XStatVisitor> set_stat =
            event.GetStat(StatType::kPerformanceCounterSets);

        data_table->AddRow()
            ->AddTextCell(hostname)
            .AddNumberCell(chip_id)
            .AddTextCell(line.Name())
            .AddNumberCell(line.Id())
            .AddTextCell(absl::AsciiStrToLower(event.Name()))
            .AddHexCell(value)
            .AddTextCell(description_stat ? description_stat->StrOrRefValue()
                                          : "")
            .AddTextCell(set_stat ? set_stat->StrOrRefValue() : "");
      });
    });
  }
}

absl::StatusOr<std::string> ConvertMultiXSpacesToPerfCounters(
    const SessionSnapshot& session_snapshot) {
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

  for (int i = 0; i < session_snapshot.XSpaceSize(); ++i) {
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
