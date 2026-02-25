#include "xprof/convert/megascale_perfetto/xspace_loader.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <list>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/megascale_perfetto/xprof_trace.h"

namespace xprof::megascale {
namespace {

using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventVisitor;
using ::tsl::profiler::XLineVisitor;
using ::tsl::profiler::XPlaneVisitor;
using ::tsl::profiler::XSpace;
using ::tsl::profiler::XStatVisitor;

bool SkipStat(absl::string_view stat_name) {
  using tsl::profiler::GetStatTypeStr;
  static const auto* const ignored_stats =
      new absl::flat_hash_set<absl::string_view>{
          GetStatTypeStr(StatType::kConsumerId),
          GetStatTypeStr(StatType::kConsumerType),
          GetStatTypeStr(StatType::kFlops),
          GetStatTypeStr(StatType::kFlow),
          GetStatTypeStr(StatType::kIsAsync),
          GetStatTypeStr(StatType::kIsRoot),
          GetStatTypeStr(StatType::kKernelDetails),
          GetStatTypeStr(StatType::kProducerId),
          GetStatTypeStr(StatType::kProducerType),
          GetStatTypeStr(StatType::kProgramId),
          GetStatTypeStr(StatType::kSymbolId),
          GetStatTypeStr(StatType::kSourceInfo),
          GetStatTypeStr(StatType::kSourceStack),
      };
  return ignored_stats->contains(stat_name);
}

bool IsTpuLineAllowed(const XLineVisitor& line) {
  const std::vector<std::string> allowed_lines = {"Steps", "XLA Modules",
                                                  "XLA Ops", "XLA TraceMe"};
  for (const auto& allowed_name : allowed_lines) {
    if (line.Name() == allowed_name) {
      return true;
    }
  }
  return false;
}

int64_t ExtractTpuIdFromPlaneName(absl::string_view plane_name) {
  int64_t tpu_id;
  if (RE2::FullMatch(plane_name, "/device:TPU:(\\d+)", &tpu_id)) {
    return tpu_id;
  }
  return -1;
}

// Extracts the raw Device ID (e.g. 200005) from the graph key.
// Input: "device_200005_gid_..."
int64_t ExtractDeviceIdFromGraphKey(absl::string_view graph_key) {
  static constexpr LazyRE2 kDeviceRe = {R"(device_(\d+)_gid_)"};
  int64_t device_id = -1;
  if (!RE2::PartialMatch(graph_key, *kDeviceRe, &device_id)) {
    return -1;
  }
  return device_id;
}

absl::string_view ExtractShortGraphKey(absl::string_view graph_key) {
  // Remove launch_id and iteration count.
  size_t dollar_pos = graph_key.find('$');
  if (dollar_pos != absl::string_view::npos) {
    return graph_key.substr(0, dollar_pos);
  }
  return graph_key;
}

void ExtractEventArgs(const XEventVisitor& event, StringTable& string_table,
                      std::vector<Arg>& args) {
  auto for_each_stat = [&](const tsl::profiler::XStatVisitor& stat) {
    if (SkipStat(stat.Name())) {
      return;
    }
    Arg arg;
    arg.key = string_table.Intern(stat.Name());
    switch (stat.ValueCase()) {
      case tsl::profiler::XStat::kInt64Value:
        arg.value = stat.IntValue();
        break;
      case tsl::profiler::XStat::kUint64Value:
        arg.value = stat.UintValue();
        break;
      case tsl::profiler::XStat::kDoubleValue:
        arg.value = stat.DoubleValue();
        break;
      case tsl::profiler::XStat::kBytesValue:
      case tsl::profiler::XStat::kRefValue:
      case tsl::profiler::XStat::kStrValue:
        arg.value = string_table.Intern(stat.StrOrRefValue());
        break;
      default:
        // Other cases are not supported.
        return;
    }
    args.push_back(arg);
  };
  event.Metadata().ForEachStat(for_each_stat);
  event.ForEachStat(for_each_stat);
}

struct GraphKeyInfo {
  std::string short_name;
  int64_t device_id = 0;
  int64_t iteration = 0;

  bool operator<(const GraphKeyInfo& other) const {
    if (short_name != other.short_name) {
      return short_name < other.short_name;
    }
    if (device_id != other.device_id) {
      return device_id < other.device_id;
    }
    return iteration < other.iteration;
  }
};

GraphKeyInfo ParseGraphKey(absl::string_view graph_key) {
  GraphKeyInfo info;
  static constexpr LazyRE2 kGraphKeyRe = {
      R"(device_(\d+)_gid_([^\$]+)\$.*\^i(\d+).*)"};
  RE2::FullMatch(graph_key, *kGraphKeyRe, &info.device_id, &info.short_name,
                 &info.iteration);
  return info;
}

absl::string_view GetGraphKey(const Event& event, const XprofTrace& trace) {
  for (const auto& arg : event.args) {
    if (trace.string_table.Get(arg.key) == "graph_key") {
      if (std::holds_alternative<StringId>(arg.value)) {
        return trace.string_table.Get(std::get<StringId>(arg.value));
      }
    }
  }
  return "";
}

}  // namespace

XprofTrace XSpaceLoader::Load(const tsl::profiler::XSpace& space) {
  XprofTrace trace;

  // Stores unique TPU IDs found in the XSpace (e.g. 0, 1, 2, 3).
  std::vector<int64_t> tpu_ids;

  // ---------------------------------------------------------------------------
  // 1. Load TPU Planes
  // ---------------------------------------------------------------------------
  for (const auto& plane : space.planes()) {
    int64_t tpu_id = ExtractTpuIdFromPlaneName(plane.name());
    if (tpu_id == -1) {
      continue;
    }

    tpu_ids.push_back(tpu_id);

    // Direct access to the fragment map
    std::vector<Track>& tracks = trace.tpu_fragments[tpu_id];
    XPlaneVisitor plane_visitor(&plane, {tsl::profiler::FindStatType});

    plane_visitor.ForEachLine([&](const XLineVisitor& line) {
      if (!IsTpuLineAllowed(line)) {
        return;
      }

      Track track;
      track.name = line.Name();

      line.ForEachEvent([&](const XEventVisitor& event_visitor) {
        Event event;
        event.timestamp_ps = event_visitor.TimestampPs();
        event.duration_ps = event_visitor.DurationPs();

        if (event_visitor.HasDisplayName()) {
          event.name = event_visitor.DisplayName();
          event.args.push_back(
              {trace.string_table.Intern("long_name"),
               trace.string_table.Intern(event_visitor.Name())});
        } else {
          event.name = event_visitor.Name();
        }

        ExtractEventArgs(event_visitor, trace.string_table, event.args);
        track.events.push_back(std::move(event));
      });

      tracks.push_back(std::move(track));
    });
  }

  // Sort TPU IDs to prepare for matching.
  std::sort(tpu_ids.begin(), tpu_ids.end());

  // ---------------------------------------------------------------------------
  // 2. Buffer Megascale Data (Grouped by Raw Device ID)
  // ---------------------------------------------------------------------------
  // Key: Raw Device ID (e.g. 200005) -> Vector of Tracks
  absl::flat_hash_map<int64_t, std::list<Track>> raw_megascale_fragments;

  for (const auto& plane : space.planes()) {
    if (plane.name() != "/device:CUSTOM:Megascale Trace") {
      continue;
    }

    XPlaneVisitor plane_visitor(&plane, {tsl::profiler::FindStatType});

    // Local lookup: track_name -> Track*
    absl::flat_hash_map<std::string, Track*> track_ptr_map;

    plane_visitor.ForEachLine([&](const XLineVisitor& line) {
      line.ForEachEvent([&](const XEventVisitor& event_visitor) {
        // Extract raw device ID from graph_key
        absl::string_view graph_key;
        event_visitor.ForEachStat([&](const XStatVisitor& stat) {
          if (stat.Name() == "graph_key") {
            graph_key = stat.StrOrRefValue();
          }
        });

        if (graph_key.empty()) {
          return;
        }

        int64_t raw_device_id = ExtractDeviceIdFromGraphKey(graph_key);
        if (raw_device_id == -1) {
          return;
        }

        // Find or create track in the BUFFER
        Track*& track = track_ptr_map[graph_key];
        if (track == nullptr) {
          std::string track_name(ExtractShortGraphKey(graph_key));
          std::list<Track>& fragments = raw_megascale_fragments[raw_device_id];
          fragments.push_back(Track{std::move(track_name), {}});
          track = &fragments.back();
        }

        Event event;
        event.timestamp_ps = event_visitor.TimestampPs();
        event.duration_ps = event_visitor.DurationPs();

        if (event_visitor.HasDisplayName()) {
          event.name = event_visitor.DisplayName();
          event.args.push_back(
              {trace.string_table.Intern("long_name"),
               trace.string_table.Intern(event_visitor.Name())});
        } else {
          event.name = event_visitor.Name();
        }

        ExtractEventArgs(event_visitor, trace.string_table, event.args);

        track->events.push_back(std::move(event));
      });
    });
  }

  // ---------------------------------------------------------------------------
  // 3. Map Raw Device IDs to TPU IDs
  // ---------------------------------------------------------------------------
  if (raw_megascale_fragments.empty()) {
    return trace;
  }

  std::vector<int64_t> raw_device_ids;
  for (const auto& pair : raw_megascale_fragments) {
    raw_device_ids.push_back(pair.first);
  }
  // Map iteration order is already sorted by key (int64_t), but explicit sort
  // is safe.
  std::sort(raw_device_ids.begin(), raw_device_ids.end());

  // Validate
  if (tpu_ids.size() != raw_device_ids.size()) {
    LOG(WARNING)
        << "Mismatch in device counts! "
        << "TPU Planes: " << tpu_ids.size() << ", "
        << "Megascale Devices: " << raw_device_ids.size() << ". "
        << "Megascale assignment may be incorrect (performing best-effort "
           "mapping).";
  }

  // Create mapping: sorted raw ID [i] -> sorted TPU ID [i]
  absl::flat_hash_map<int64_t, int64_t> device_id_to_tpu_id;
  size_t limit = std::min(tpu_ids.size(), raw_device_ids.size());
  for (size_t i = 0; i < limit; ++i) {
    device_id_to_tpu_id[raw_device_ids[i]] = tpu_ids[i];
  }

  // ---------------------------------------------------------------------------
  // 3.5. Sort Tracks based on Graph Key (short_name, device_id, iteration)
  // ---------------------------------------------------------------------------
  for (auto& [raw_id, tracks] : raw_megascale_fragments) {
    tracks.sort([&](const Track& a, const Track& b) {
      if (a.events.empty()) return true;
      if (b.events.empty()) return false;
      GraphKeyInfo info_a = ParseGraphKey(GetGraphKey(a.events[0], trace));
      GraphKeyInfo info_b = ParseGraphKey(GetGraphKey(b.events[0], trace));
      return info_a < info_b;
    });
  }

  // ---------------------------------------------------------------------------
  // 4. Finalize (Move Buffer to Trace)
  // ---------------------------------------------------------------------------
  for (auto& [raw_id, tracks] : raw_megascale_fragments) {
    auto it = device_id_to_tpu_id.find(raw_id);
    if (it != device_id_to_tpu_id.end()) {
      int64_t target_tpu_id = it->second;
      trace.megascale_fragments[target_tpu_id].assign(
          std::make_move_iterator(tracks.begin()),
          std::make_move_iterator(tracks.end()));
    } else {
      LOG(WARNING) << "Dropping Megascale data for raw device ID " << raw_id
                   << " (could not map to any TPU plane).";
    }
  }

  return trace;
}

absl::StatusOr<XprofTrace> XSpaceLoader::LoadFromFile(
    const std::string& file_path) {
  tsl::profiler::XSpace xspace;
  TF_RETURN_IF_ERROR(xprof::ReadBinaryProto(file_path, &xspace));
  return Load(xspace);
}

}  // namespace xprof::megascale
