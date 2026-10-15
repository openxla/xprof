#include "xprof/convert/trace_viewer/delta_series/trace_data_to_compressed_delta_series_proto.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/lib/context_types.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

constexpr absl::string_view kXlaOpsThreadName = "XLA Ops";
constexpr absl::string_view kComputeUtilizationRooflineThreadName =
    "Compute Utilization/Roofline Efficiency";
constexpr absl::string_view kComputeUtilizationThreadName =
    "Compute Utilization";
constexpr absl::string_view kDataMotionLayersLowerThreadName =
    "Data motion layers utilization";
constexpr absl::string_view kDataMotionLayersTitleThreadName =
    "Data Motion Layers Utilization";
constexpr absl::string_view kHloOpArgName = "hlo_op";
constexpr absl::string_view kNameArgName = "Name";
constexpr absl::string_view kHloModuleArgName = "hlo_module";
constexpr absl::string_view kHloModuleIdArgName = "hlo_module_id";
constexpr absl::string_view kProgramIdArgName = "program_id";
constexpr absl::string_view kKernelDetailsArgName = "kernel_details";

absl::string_view ResolveEventName(const Trace& trace,
                                   const TraceEvent& event) {
  if (event.has_name_ref()) {
    const auto it = trace.name_table().find(event.name_ref());
    if (it != trace.name_table().end()) {
      return it->second;
    }
    return "";
  }
  return event.name();
}

absl::string_view ExtractModuleIdFromKernelDetails(
    absl::string_view kernel_details) {
  static const absl::NoDestructor<RE2> kModuleRe(R"(module:.*_(\d+))");
  static const absl::NoDestructor<RE2> kModuleParenRe(
      R"(module:[^\s(]+\(([^)]+)\))");
  absl::string_view module_id;
  if (RE2::PartialMatch(kernel_details, *kModuleRe, &module_id)) {
    return module_id;
  }
  if (RE2::PartialMatch(kernel_details, *kModuleParenRe, &module_id)) {
    return module_id;
  }
  return "";
}

}  // namespace

DeltaSeriesProtoConverter::DeltaSeriesProtoConverter(
    const Trace* trace, CounterExtractor counter_extractor,
    const DeltaSeriesProtoConversionOptions& options)
    : trace_(trace),
      counter_extractor_(std::move(counter_extractor)),
      options_(options) {
  // Reserve index 0 for empty strings to prevent default 0 name_refs
  // from pointing to valid strings.
  interned_strings_.push_back("");
  interned_strings_map_[""] = 0;
}

uint32_t DeltaSeriesProtoConverter::MaybeInternString(absl::string_view str) {
  const auto it = interned_strings_map_.find(str);
  if (it != interned_strings_map_.end()) {
    return it->second;
  }
  uint32_t interned_string_id = interned_strings_.size();
  interned_strings_.push_back(std::string(str));
  interned_strings_map_[str] = interned_string_id;
  return interned_string_id;
}

xprof::TraceMetadata DeltaSeriesProtoConverter::GetTraceMetadata() const {
  xprof::TraceMetadata metadata;
  std::vector<uint32_t> sorted_device_ids;
  sorted_device_ids.reserve(trace_->devices_size());
  for (const auto& [device_id, _] : trace_->devices()) {
    if (options_.mpmd_single_device_per_stage &&
        deduplicated_device_ids_.contains(device_id)) {
      continue;
    }
    sorted_device_ids.push_back(device_id);
  }
  absl::c_sort(sorted_device_ids);

  for (const uint32_t device_id : sorted_device_ids) {
    const auto dev_it = trace_->devices().find(device_id);
    if (dev_it == trace_->devices().end()) continue;
    const Device& device = dev_it->second;

    xprof::Process* process = metadata.add_processes();
    process->set_id(device_id);
    if (device.has_name()) {
      process->set_name(device.name());
    }
    if (!options_.mpmd_pipeline_view) {
      process->set_sort_index(device_id);
    } else if (!mpmd_sort_indices_.empty()) {
      const auto it = mpmd_sort_indices_.find(device_id);
      process->set_sort_index(it != mpmd_sort_indices_.end()
                                  ? it->second
                                  : kMpmdUnrankedSortIndexBase + device_id);
    }

    std::vector<uint32_t> sorted_resource_ids;
    sorted_resource_ids.reserve(device.resources_size());
    for (const auto& [resource_id, _] : device.resources()) {
      sorted_resource_ids.push_back(resource_id);
    }
    absl::c_sort(sorted_resource_ids);

    for (const uint32_t resource_id : sorted_resource_ids) {
      const auto res_it = device.resources().find(resource_id);
      if (res_it == device.resources().end()) continue;
      const Resource& resource = res_it->second;

      xprof::Thread* thread = process->add_threads();
      thread->set_id(resource_id);
      if (resource.has_name()) {
        thread->set_name(resource.name());
      }
      if (!options_.sort_resources_by_name.contains(device_id)) {
        thread->set_sort_index(resource_id);
      }
    }
  }
  return metadata;
}

void DeltaSeriesProtoConverter::PopulateTraceEventMetadata(
    const TraceEvent& event, xprof::TraceEventMetadata* metadata) {
  if (event.has_flow_id()) {
    metadata->set_flow_id(event.flow_id());
  }
  if (event.has_flow_category()) {
    tsl::profiler::ContextType type =
        tsl::profiler::GetSafeContextType(event.flow_category());
    if (type != tsl::profiler::ContextType::kGeneric &&
        type != tsl::profiler::ContextType::kLegacy) {
      const char* category = tsl::profiler::GetContextTypeString(type);
      metadata->set_flow_category(MaybeInternString(category));
    }
  }
  if (event.has_group_id()) {
    metadata->set_group_id(event.group_id());
  }
  if (event.has_serial()) {
    metadata->set_serial(event.serial());
  }
}

bool DeltaSeriesProtoConverter::ExtractHloMetadataFromRawData(
    const TraceEvent& event, bool is_xla_ops_thread, bool is_data_motion_layer,
    absl::string_view event_name, uint32_t* hlo_op_ref,
    uint32_t* hlo_module_ref) {
  *hlo_op_ref = 0;
  *hlo_module_ref = 0;
  if (!event.has_raw_data()) return false;
  RawData raw_data;
  if (!raw_data.ParseFromString(event.raw_data()) || !raw_data.has_args()) {
    return false;
  }

  std::string hlo_op;
  std::string name_arg;
  std::string hlo_module;
  std::string hlo_module_id;
  std::string program_id;
  std::string kernel_details;

  for (const TraceEventArguments::Argument& arg : raw_data.args().arg()) {
    const absl::string_view arg_name = arg.name();
    std::string str_val;
    if (arg.has_str_value()) {
      str_val = arg.str_value();
    } else if (arg.has_ref_value()) {
      const auto it = trace_->name_table().find(arg.ref_value());
      if (it != trace_->name_table().end()) {
        str_val = it->second;
      }
    } else if (arg.has_int_value()) {
      str_val = absl::StrCat(arg.int_value());
    } else if (arg.has_uint_value()) {
      str_val = absl::StrCat(arg.uint_value());
    }
    if (str_val.empty()) continue;

    if (arg_name == kHloOpArgName) {
      hlo_op = std::move(str_val);
    } else if (arg_name == kNameArgName) {
      name_arg = std::move(str_val);
    } else if (arg_name == kHloModuleArgName) {
      hlo_module = std::move(str_val);
    } else if (arg_name == kHloModuleIdArgName) {
      hlo_module_id = std::move(str_val);
    } else if (arg_name == kProgramIdArgName) {
      program_id = std::move(str_val);
    } else if (arg_name == kKernelDetailsArgName) {
      kernel_details = std::move(str_val);
    }
  }

  const bool has_hlo_in_args = !hlo_op.empty() && !hlo_module.empty();
  if (!is_xla_ops_thread && !is_data_motion_layer && !has_hlo_in_args) {
    return false;
  }

  std::string hlo_op_str;
  if (is_data_motion_layer) {
    hlo_op_str = !name_arg.empty() ? std::move(name_arg) : std::move(hlo_op);
  } else if (!hlo_op.empty() && (!is_xla_ops_thread || hlo_op != event_name)) {
    hlo_op_str = std::move(hlo_op);
  }

  std::string hlo_module_str;
  if (!hlo_module.empty()) {
    if (!hlo_module_id.empty()) {
      hlo_module_str = absl::StrCat(hlo_module, "(", hlo_module_id, ")");
    } else if (!program_id.empty()) {
      hlo_module_str = absl::StrCat(hlo_module, "(", program_id, ")");
    } else {
      const absl::string_view mod_id =
          ExtractModuleIdFromKernelDetails(kernel_details);
      if (!mod_id.empty()) {
        hlo_module_str = absl::StrCat(hlo_module, "(", mod_id, ")");
      } else {
        hlo_module_str = std::move(hlo_module);
      }
    }
  }

  if (!hlo_op_str.empty()) {
    *hlo_op_ref = MaybeInternString(hlo_op_str);
  }
  if (!hlo_module_str.empty()) {
    *hlo_module_ref = MaybeInternString(hlo_module_str);
  }
  return (*hlo_op_ref != 0 || *hlo_module_ref != 0);
}

void DeltaSeriesProtoConverter::AddCompleteEventTrack(
    uint32_t pid, uint64_t tid, const TraceEventTrack& events,
    xprof::TraceDataResponse* response) {
  xprof::TraceEventSeries* series = response->add_complete_events();
  xprof::TraceEventSeriesMetadata* series_metadata = series->mutable_metadata();
  series_metadata->set_process_id(pid);
  series_metadata->set_thread_id(tid);

  if (!options_.sparse_event_metadata) {
    uint64_t last_timestamp = 0;
    for (const TraceEvent* event : events) {
      const uint64_t current_timestamp = event->timestamp_ps();
      series->add_deltas(current_timestamp - last_timestamp);
      last_timestamp = current_timestamp;

      series->add_durations(event->duration_ps());
      series->add_name_refs(
          MaybeInternString(ResolveEventName(*trace_, *event)));
      PopulateTraceEventMetadata(*event, series->add_event_metadata());
    }
    return;
  }

  absl::string_view res_name;
  if (const auto dev_it = trace_->devices().find(pid);
      dev_it != trace_->devices().end()) {
    if (const auto res_it = dev_it->second.resources().find(tid);
        res_it != dev_it->second.resources().end() &&
        res_it->second.has_name()) {
      res_name = res_it->second.name();
    }
  }
  const bool is_xla_ops_thread = (res_name == kXlaOpsThreadName);
  const bool is_data_motion_layer =
      (res_name == kComputeUtilizationRooflineThreadName ||
       res_name == kComputeUtilizationThreadName ||
       res_name == kDataMotionLayersLowerThreadName ||
       res_name == kDataMotionLayersTitleThreadName);
  bool should_extract_hlo_raw_data = !events.empty();
  if (is_xla_ops_thread && events.size() > 64) {
    uint32_t probe_op = 0;
    uint32_t probe_mod = 0;
    const absl::string_view first_name = ResolveEventName(*trace_, *events[0]);
    should_extract_hlo_raw_data = ExtractHloMetadataFromRawData(
        *events[0], is_xla_ops_thread, is_data_motion_layer, first_name,
        &probe_op, &probe_mod);
  }

  uint64_t last_timestamp = 0;
  int64_t current_group_id_run_val = 0;
  uint32_t current_group_id_run_len = 0;
  uint32_t emitted_index = 0;

  for (size_t i = 0; i < events.size(); ++i) {
    const TraceEvent* event = events[i];
    const uint64_t current_timestamp = event->timestamp_ps();
    series->add_deltas(current_timestamp - last_timestamp);
    last_timestamp = current_timestamp;

    series->add_durations(event->duration_ps());

    const absl::string_view event_name = ResolveEventName(*trace_, *event);
    series->add_name_refs(MaybeInternString(event_name));

    const int64_t group_id = event->has_group_id() ? event->group_id() : -1;
    if (current_group_id_run_len == 0) {
      current_group_id_run_val = group_id;
      current_group_id_run_len = 1;
    } else if (group_id == current_group_id_run_val) {
      ++current_group_id_run_len;
    } else {
      series->add_group_id_run_values(current_group_id_run_val);
      series->add_group_id_run_lengths(current_group_id_run_len);
      current_group_id_run_val = group_id;
      current_group_id_run_len = 1;
    }

    uint32_t hlo_op_ref = 0;
    uint32_t hlo_module_ref = 0;
    const bool has_hlo_refs =
        should_extract_hlo_raw_data &&
        ExtractHloMetadataFromRawData(*event, is_xla_ops_thread,
                                      is_data_motion_layer, event_name,
                                      &hlo_op_ref, &hlo_module_ref);
    const bool has_non_generic_category =
        event->has_flow_category() &&
        tsl::profiler::GetSafeContextType(event->flow_category()) !=
            tsl::profiler::ContextType::kGeneric &&
        tsl::profiler::GetSafeContextType(event->flow_category()) !=
            tsl::profiler::ContextType::kLegacy;
    if (event->flow_id() != 0 || event->serial() != 0 ||
        has_non_generic_category || has_hlo_refs) {
      xprof::TraceEventMetadata* meta = series->add_event_metadata();
      PopulateTraceEventMetadata(*event, meta);
      if (hlo_op_ref != 0) meta->set_hlo_op_ref(hlo_op_ref);
      if (hlo_module_ref != 0) meta->set_hlo_module_ref(hlo_module_ref);
      series->add_event_metadata_indices(emitted_index);
    }
    ++emitted_index;
  }
  if (current_group_id_run_len > 0) {
    series->add_group_id_run_values(current_group_id_run_val);
    series->add_group_id_run_lengths(current_group_id_run_len);
  }
}

void DeltaSeriesProtoConverter::AddAsyncEventTrack(
    uint32_t pid, absl::string_view name, const TraceEventTrack& events,
    xprof::TraceDataResponse* response) {
  xprof::TraceEventSeries* series = response->add_async_events();
  xprof::TraceEventSeriesMetadata* series_metadata = series->mutable_metadata();
  series_metadata->set_process_id(pid);
  series_metadata->set_name_ref(MaybeInternString(name));

  if (!options_.sparse_event_metadata) {
    uint64_t last_timestamp = 0;
    for (const TraceEvent* event : events) {
      const uint64_t current_timestamp = event->timestamp_ps();
      series->add_deltas(current_timestamp - last_timestamp);
      last_timestamp = current_timestamp;

      series->add_durations(event->duration_ps());
      PopulateTraceEventMetadata(*event, series->add_event_metadata());
    }
    return;
  }

  uint64_t last_timestamp = 0;
  int64_t current_group_id_run_val = 0;
  uint32_t current_group_id_run_len = 0;
  uint32_t emitted_index = 0;

  for (size_t i = 0; i < events.size(); ++i) {
    const TraceEvent* event = events[i];
    const uint64_t current_timestamp = event->timestamp_ps();
    series->add_deltas(current_timestamp - last_timestamp);
    last_timestamp = current_timestamp;

    series->add_durations(event->duration_ps());

    const int64_t group_id = event->has_group_id() ? event->group_id() : -1;
    if (current_group_id_run_len == 0) {
      current_group_id_run_val = group_id;
      current_group_id_run_len = 1;
    } else if (group_id == current_group_id_run_val) {
      ++current_group_id_run_len;
    } else {
      series->add_group_id_run_values(current_group_id_run_val);
      series->add_group_id_run_lengths(current_group_id_run_len);
      current_group_id_run_val = group_id;
      current_group_id_run_len = 1;
    }

    uint32_t hlo_op_ref = 0;
    uint32_t hlo_module_ref = 0;
    const bool has_hlo_refs = ExtractHloMetadataFromRawData(
        *event, /*is_xla_ops_thread=*/false, /*is_data_motion_layer=*/false,
        name, &hlo_op_ref, &hlo_module_ref);
    const bool has_non_generic_category =
        event->has_flow_category() &&
        tsl::profiler::GetSafeContextType(event->flow_category()) !=
            tsl::profiler::ContextType::kGeneric &&
        tsl::profiler::GetSafeContextType(event->flow_category()) !=
            tsl::profiler::ContextType::kLegacy;
    if ((event->duration_ps() == 0 && event->flow_id() != 0) ||
        event->serial() != 0 || has_non_generic_category || has_hlo_refs) {
      xprof::TraceEventMetadata* meta = series->add_event_metadata();
      PopulateTraceEventMetadata(*event, meta);
      if (hlo_op_ref != 0) meta->set_hlo_op_ref(hlo_op_ref);
      if (hlo_module_ref != 0) meta->set_hlo_module_ref(hlo_module_ref);
      series->add_event_metadata_indices(emitted_index);
    }
    ++emitted_index;
  }
  if (current_group_id_run_len > 0) {
    series->add_group_id_run_values(current_group_id_run_val);
    series->add_group_id_run_lengths(current_group_id_run_len);
  }
}

void DeltaSeriesProtoConverter::AddCounterEventTrack(
    uint32_t pid, absl::string_view name, const TraceEventTrack& events,
    xprof::TraceDataResponse* response) {
  xprof::TraceEventSeries* series = response->add_counter_events();
  xprof::TraceEventSeriesMetadata* series_metadata = series->mutable_metadata();
  series_metadata->set_process_id(pid);
  series_metadata->set_name_ref(MaybeInternString(name));

  bool has_series_name = false;
  std::string series_name;
  uint64_t last_timestamp = 0;
  for (const TraceEvent* event : events) {
    const uint64_t current_timestamp = event->timestamp_ps();
    series->add_deltas(current_timestamp - last_timestamp);
    last_timestamp = current_timestamp;

    xprof::TraceEventMetadata* event_metadata = series->add_event_metadata();
    if (event->has_raw_data()) {
      counter_extractor_(event->raw_data(), event_metadata,
                         has_series_name ? nullptr : &series_name);
      if (!has_series_name && !series_name.empty()) {
        series_metadata->set_event_stats_ref(MaybeInternString(series_name));
        has_series_name = true;
      }
    }
  }
}

namespace {

absl::Status AppendCompressedStreamFrame(const xprof::TraceStreamMessage& msg,
                                         std::string* out) {
  std::string serialized;
  if (!msg.SerializeToString(&serialized)) {
    return absl::InternalError("Failed to serialize TraceStreamMessage.");
  }
  std::string compressed;
  TF_ASSIGN_OR_RETURN(compressed, ZstdCompression::Compress(serialized));
  const uint32_t frame_len = static_cast<uint32_t>(compressed.size());
  const char len_bytes[4] = {
      static_cast<char>(frame_len & 0xFF),
      static_cast<char>((frame_len >> 8) & 0xFF),
      static_cast<char>((frame_len >> 16) & 0xFF),
      static_cast<char>((frame_len >> 24) & 0xFF),
  };
  out->append(len_bytes, 4);
  out->append(compressed);
  return absl::OkStatus();
}

struct StreamCompleteEvent {
  uint64_t ts_ps = 0;
  uint64_t dur_ps = 0;
  uint32_t name_ref = 0;
  bool has_rle_group_id = false;
  int64_t rle_group_id = -1;
  bool is_sparse_meta = false;
  std::optional<xprof::TraceEventMetadata> meta;
};

struct StreamAsyncEvent {
  uint64_t ts_ps = 0;
  uint64_t dur_ps = 0;
  uint32_t name_ref = 0;
  uint64_t flow_id = 0;
  int32_t group_id = -1;
  uint32_t serial = 0;
  uint32_t hlo_op_ref = 0;
  uint32_t hlo_module_ref = 0;
};

struct StreamCounterEvent {
  uint64_t ts_ps = 0;
  xprof::TraceEventMetadata meta;
};

struct PreparedStreamTrack {
  xprof::TrackManifest::TrackKind kind =
      xprof::TrackManifest::TRACK_KIND_UNSPECIFIED;
  uint32_t process_id = 0;
  uint64_t thread_id = 0;
  uint32_t name_ref = 0;
  std::optional<uint32_t> event_stats_ref;
  bool ordered = true;
  std::vector<StreamCompleteEvent> complete_events;
  std::vector<StreamAsyncEvent> async_events;
  std::vector<StreamCounterEvent> counter_events;

  size_t event_count() const {
    switch (kind) {
      case xprof::TrackManifest::TRACK_KIND_COMPLETE:
        return complete_events.size();
      case xprof::TrackManifest::TRACK_KIND_ASYNC:
        return async_events.size();
      case xprof::TrackManifest::TRACK_KIND_COUNTER:
        return counter_events.size();
      default:
        return 0;
    }
  }
};

}  // namespace

absl::StatusOr<std::string> ConvertTraceDataResponseToCompressedStreamProto(
    const xprof::TraceDataResponse& response,
    absl::FunctionRef<size_t(uint32_t track_id, size_t remaining_events)>
        next_segment_size_fn) {
  std::vector<std::string> interned_strings(response.interned_strings().begin(),
                                            response.interned_strings().end());
  if (interned_strings.empty()) {
    interned_strings.push_back("");
  }
  absl::flat_hash_map<std::string, uint32_t> interned_map;
  interned_map.reserve(interned_strings.size());
  for (uint32_t i = 0; i < interned_strings.size(); ++i) {
    interned_map.try_emplace(interned_strings[i], i);
  }
  const auto get_interned_string =
      [&interned_strings](uint32_t ref) -> absl::string_view {
    return ref < interned_strings.size()
               ? absl::string_view(interned_strings[ref])
               : "";
  };
  const auto intern_string = [&interned_strings,
                              &interned_map](absl::string_view s) -> uint32_t {
    const auto it = interned_map.find(s);
    if (it != interned_map.end()) return it->second;
    const uint32_t id = static_cast<uint32_t>(interned_strings.size());
    interned_strings.emplace_back(s);
    interned_map.emplace(interned_strings.back(), id);
    return id;
  };

  // Map (pid, tid) -> thread_name to identify XLA Modules tracks (P6).
  absl::flat_hash_map<std::pair<uint32_t, uint64_t>, std::string> thread_names;
  for (const xprof::Process& proc : response.metadata().processes()) {
    for (const xprof::Thread& thr : proc.threads()) {
      if (!thr.name().empty()) {
        thread_names[{proc.id(), thr.id()}] = thr.name();
      }
    }
  }

  // 1. Unpack and group complete event tracks by (pid, tid).
  absl::btree_map<uint32_t,
                  absl::btree_map<uint64_t, std::vector<StreamCompleteEvent>>>
      complete_by_pid_tid;
  uint64_t max_complete_end_ps = 0;
  for (const xprof::TraceEventSeries& series : response.complete_events()) {
    const int n = series.deltas_size();
    if (n == 0) continue;
    const uint32_t pid = series.metadata().process_id();
    const uint64_t tid = series.metadata().thread_id();
    std::vector<StreamCompleteEvent>& out = complete_by_pid_tid[pid][tid];
    out.reserve(out.size() + static_cast<size_t>(n));

    uint64_t cur_ts_ps = 0;
    const bool has_sparse = series.event_metadata_indices_size() > 0;
    int sparse_cursor = 0;
    const int sparse_count = std::min(series.event_metadata_indices_size(),
                                      series.event_metadata_size());
    const bool has_rle =
        series.group_id_run_values_size() > 0 &&
        series.group_id_run_values_size() == series.group_id_run_lengths_size();
    int rle_idx = 0;
    uint32_t rle_rem = has_rle ? series.group_id_run_lengths(0) : 0;

    for (int i = 0; i < n; ++i) {
      const uint64_t delta_ps = series.deltas(i);
      const int64_t signed_delta = static_cast<int64_t>(delta_ps);
      if (signed_delta < 0) {
        const uint64_t abs_d = static_cast<uint64_t>(-(signed_delta + 1)) + 1;
        cur_ts_ps = (cur_ts_ps >= abs_d) ? (cur_ts_ps - abs_d) : 0;
      } else {
        cur_ts_ps += delta_ps;
      }
      const uint64_t dur_ps =
          i < series.durations_size() ? series.durations(i) : 0;
      max_complete_end_ps = std::max(max_complete_end_ps, cur_ts_ps + dur_ps);
      const uint32_t name_ref = i < series.name_refs_size()
                                    ? series.name_refs(i)
                                    : series.metadata().name_ref();

      StreamCompleteEvent ev;
      ev.ts_ps = cur_ts_ps;
      ev.dur_ps = dur_ps;
      ev.name_ref = name_ref;
      ev.has_rle_group_id = has_rle;
      ev.is_sparse_meta = has_sparse;

      if (has_rle) {
        while (rle_rem == 0 &&
               rle_idx + 1 < series.group_id_run_lengths_size()) {
          ++rle_idx;
          rle_rem = series.group_id_run_lengths(rle_idx);
        }
        if (rle_rem > 0) {
          ev.rle_group_id = series.group_id_run_values(rle_idx);
          --rle_rem;
        }
      }

      if (has_sparse) {
        while (sparse_cursor < sparse_count &&
               series.event_metadata_indices(sparse_cursor) <
                   static_cast<uint32_t>(i)) {
          ++sparse_cursor;
        }
        if (sparse_cursor < sparse_count &&
            series.event_metadata_indices(sparse_cursor) ==
                static_cast<uint32_t>(i)) {
          ev.meta = series.event_metadata(sparse_cursor);
          ++sparse_cursor;
        }
      } else if (i < series.event_metadata_size()) {
        ev.meta = series.event_metadata(i);
      }
      out.push_back(std::move(ev));
    }
  }

  // 2. Unpack and canonicalize counter tracks per (pid, name) (P8).
  struct CounterSeriesRef {
    double first_ts_us = 0.0;
    const xprof::TraceEventSeries* series = nullptr;
  };
  absl::btree_map<uint32_t,
                  absl::btree_map<std::string, std::vector<CounterSeriesRef>>>
      counter_series_by_pid_name;
  uint64_t max_counter_end_ps = 0;
  for (const xprof::TraceEventSeries& series : response.counter_events()) {
    const uint32_t pid = series.metadata().process_id();
    const std::string name(get_interned_string(series.metadata().name_ref()));
    const double first_ts_us = series.deltas_size() > 0
                                   ? (series.deltas(0) / 1000000.0)
                                   : std::numeric_limits<double>::max();
    counter_series_by_pid_name[pid][name].push_back(
        {.first_ts_us = first_ts_us, .series = &series});
    uint64_t cur_ts_ps = 0;
    for (int i = 0; i < series.deltas_size(); ++i) {
      cur_ts_ps += series.deltas(i);
    }
    if (series.deltas_size() > 0) {
      max_counter_end_ps = std::max(max_counter_end_ps, cur_ts_ps);
    }
  }

  // 3. Pair async events on the server (P7).
  struct AsyncTrackBucket {
    uint64_t tid = 0;
    uint32_t name_ref = 0;
    std::vector<StreamAsyncEvent> events;
  };
  absl::btree_map<uint32_t, absl::btree_map<std::string, AsyncTrackBucket>>
      async_by_pid_name;
  absl::btree_set<uint32_t> async_pids;
  struct OpenAsync {
    StreamAsyncEvent ev;
    uint32_t pid = 0;
    uint64_t tid = 0;
    std::string name;
  };
  absl::flat_hash_map<std::tuple<uint32_t, uint32_t, uint64_t>, OpenAsync>
      open_async;
  std::vector<OpenAsync> unclosed_async;
  uint64_t max_async_end_ps = 0;

  for (const xprof::TraceEventSeries& series : response.async_events()) {
    const xprof::TraceEventSeriesMetadata& meta = series.metadata();
    const uint32_t pid = meta.process_id();
    const uint64_t tid = meta.thread_id();
    async_pids.insert(pid);

    const bool has_per_event_names = series.name_refs_size() > 0;
    const absl::string_view series_name = get_interned_string(meta.name_ref());
    uint64_t cur_ts_ps = 0;
    const int n = series.deltas_size();
    const bool has_sparse = series.event_metadata_indices_size() > 0;
    int sparse_cursor = 0;
    const int sparse_count = std::min(series.event_metadata_indices_size(),
                                      series.event_metadata_size());
    const bool has_rle =
        series.group_id_run_values_size() > 0 &&
        series.group_id_run_values_size() == series.group_id_run_lengths_size();
    int rle_idx = 0;
    uint32_t rle_rem = has_rle ? series.group_id_run_lengths(0) : 0;

    for (int i = 0; i < n; ++i) {
      cur_ts_ps += series.deltas(i);
      const uint64_t dur_ps =
          i < series.durations_size() ? series.durations(i) : 0;
      uint64_t flow_id = 0;
      int32_t group_id = -1;
      uint32_t serial = 0;
      uint32_t hlo_op_ref = 0;
      uint32_t hlo_module_ref = 0;

      if (has_rle) {
        while (rle_rem == 0 &&
               rle_idx + 1 < series.group_id_run_lengths_size()) {
          ++rle_idx;
          rle_rem = series.group_id_run_lengths(rle_idx);
        }
        if (rle_rem > 0) {
          const int64_t val = series.group_id_run_values(rle_idx);
          if (val >= 0) group_id = static_cast<int32_t>(val);
          --rle_rem;
        }
      }

      const xprof::TraceEventMetadata* ev_meta = nullptr;
      if (has_sparse) {
        while (sparse_cursor < sparse_count &&
               series.event_metadata_indices(sparse_cursor) <
                   static_cast<uint32_t>(i)) {
          ++sparse_cursor;
        }
        if (sparse_cursor < sparse_count &&
            series.event_metadata_indices(sparse_cursor) ==
                static_cast<uint32_t>(i)) {
          ev_meta = &series.event_metadata(sparse_cursor);
          ++sparse_cursor;
        }
      } else if (i < series.event_metadata_size()) {
        ev_meta = &series.event_metadata(i);
      }

      if (ev_meta != nullptr) {
        flow_id = ev_meta->flow_id();
        if (group_id < 0 && ev_meta->group_id() != 0) {
          group_id = static_cast<int32_t>(ev_meta->group_id());
        }
        serial = ev_meta->serial();
        if (ev_meta->has_hlo_op_ref() &&
            ev_meta->hlo_op_ref() < interned_strings.size()) {
          hlo_op_ref = ev_meta->hlo_op_ref();
        }
        if (ev_meta->has_hlo_module_ref() &&
            ev_meta->hlo_module_ref() < interned_strings.size()) {
          hlo_module_ref = ev_meta->hlo_module_ref();
        }
      }

      const uint32_t raw_name_ref =
          (has_per_event_names && i < series.name_refs_size())
              ? series.name_refs(i)
              : meta.name_ref();
      const std::string ev_name(has_per_event_names
                                    ? get_interned_string(raw_name_ref)
                                    : series_name);
      const uint32_t canon_name_ref = intern_string(ev_name);
      const StreamAsyncEvent ev{
          .ts_ps = cur_ts_ps,
          .dur_ps = dur_ps,
          .name_ref = canon_name_ref,
          .flow_id = flow_id,
          .group_id = group_id,
          .serial = serial,
          .hlo_op_ref = hlo_op_ref,
          .hlo_module_ref = hlo_module_ref,
      };

      if (dur_ps > 0) {
        max_async_end_ps = std::max(max_async_end_ps, cur_ts_ps + dur_ps);
        AsyncTrackBucket& bucket = async_by_pid_name[pid][ev_name];
        if (bucket.events.empty()) {
          bucket.tid = tid;
          bucket.name_ref = canon_name_ref;
        }
        bucket.events.push_back(ev);
      } else {
        max_async_end_ps = std::max(max_async_end_ps, cur_ts_ps);
        const std::tuple<uint32_t, uint32_t, uint64_t> key(pid, raw_name_ref,
                                                           flow_id);
        const auto [it, inserted] = open_async.try_emplace(
            key, OpenAsync{.ev = ev, .pid = pid, .tid = tid, .name = ev_name});
        if (!inserted) {
          OpenAsync& begin = it->second;
          if (ev.ts_ps > begin.ev.ts_ps) {
            begin.ev.dur_ps = ev.ts_ps - begin.ev.ts_ps;
          }
          if (begin.ev.serial == 0 && ev.serial != 0) {
            begin.ev.serial = ev.serial;
          }
          if (begin.ev.group_id < 0 && ev.group_id >= 0) {
            begin.ev.group_id = ev.group_id;
          }
          if (begin.ev.hlo_op_ref == 0 && ev.hlo_op_ref != 0) {
            begin.ev.hlo_op_ref = ev.hlo_op_ref;
          }
          if (begin.ev.hlo_module_ref == 0 && ev.hlo_module_ref != 0) {
            begin.ev.hlo_module_ref = ev.hlo_module_ref;
          }
          AsyncTrackBucket& bucket = async_by_pid_name[pid][begin.name];
          if (bucket.events.empty()) {
            bucket.tid = begin.tid;
            bucket.name_ref = begin.ev.name_ref;
          }
          bucket.events.push_back(begin.ev);
          open_async.erase(it);
        }
      }
    }

    if (!has_per_event_names) {
      if (const auto it = open_async.find({pid, meta.name_ref(), 0});
          it != open_async.end()) {
        unclosed_async.push_back(it->second);
        open_async.erase(it);
      }
    } else {
      for (int i = 0; i < series.name_refs_size(); ++i) {
        if (const auto it = open_async.find({pid, series.name_refs(i), 0});
            it != open_async.end()) {
          unclosed_async.push_back(it->second);
          open_async.erase(it);
        }
      }
    }
  }

  if (!open_async.empty()) {
    std::vector<OpenAsync> remaining;
    remaining.reserve(open_async.size());
    for (const auto& [_, open_ev] : open_async) {
      remaining.push_back(open_ev);
    }
    absl::c_sort(remaining, [](const OpenAsync& a, const OpenAsync& b) {
      return std::tie(a.pid, a.tid, a.ev.ts_ps, a.ev.flow_id, a.ev.name_ref) <
             std::tie(b.pid, b.tid, b.ev.ts_ps, b.ev.flow_id, b.ev.name_ref);
    });
    for (const OpenAsync& open_ev : remaining) {
      unclosed_async.push_back(open_ev);
    }
    open_async.clear();
  }

  if (!unclosed_async.empty()) {
    const uint64_t full_end_ps = response.has_full_timespan_end_ps()
                                     ? response.full_timespan_end_ps()
                                     : 0;
    const uint64_t trace_end_ps =
        std::max({max_async_end_ps, max_counter_end_ps, full_end_ps,
                  max_complete_end_ps});
    for (OpenAsync& unclosed : unclosed_async) {
      unclosed.ev.dur_ps = trace_end_ps > unclosed.ev.ts_ps
                               ? (trace_end_ps - unclosed.ev.ts_ps)
                               : 0;
      AsyncTrackBucket& bucket = async_by_pid_name[unclosed.pid][unclosed.name];
      if (bucket.events.empty()) {
        bucket.tid = unclosed.tid;
        bucket.name_ref = unclosed.ev.name_ref;
      }
      bucket.events.push_back(unclosed.ev);
    }
  }

  // 4. Assemble PreparedStreamTrack list in deterministic order, placing
  // XLA Modules tracks before other complete tracks on each pid (P6).
  absl::btree_set<uint32_t> all_pids;
  for (const auto& [pid, _] : complete_by_pid_tid) all_pids.insert(pid);
  for (const auto& [pid, _] : counter_series_by_pid_name) all_pids.insert(pid);
  for (const uint32_t pid : async_pids) all_pids.insert(pid);

  std::vector<PreparedStreamTrack> tracks;
  for (const uint32_t pid : all_pids) {
    if (const auto it_cnt = counter_series_by_pid_name.find(pid);
        it_cnt != counter_series_by_pid_name.end()) {
      for (auto& [name, refs] : it_cnt->second) {
        absl::c_stable_sort(
            refs, [](const CounterSeriesRef& a, const CounterSeriesRef& b) {
              return a.first_ts_us < b.first_ts_us;
            });
        PreparedStreamTrack track;
        track.kind = xprof::TrackManifest::TRACK_KIND_COUNTER;
        track.process_id = pid;
        track.name_ref = intern_string(name);
        for (const CounterSeriesRef& ref : refs) {
          if (!track.event_stats_ref.has_value() &&
              ref.series->metadata().has_event_stats_ref()) {
            track.event_stats_ref = ref.series->metadata().event_stats_ref();
          }
          uint64_t cur_ts_ps = 0;
          for (int i = 0; i < ref.series->deltas_size(); ++i) {
            cur_ts_ps += ref.series->deltas(i);
            xprof::TraceEventMetadata meta;
            if (i < ref.series->event_metadata_size()) {
              meta = ref.series->event_metadata(i);
            }
            track.counter_events.push_back(
                {.ts_ps = cur_ts_ps, .meta = std::move(meta)});
          }
        }
        track.ordered = absl::c_is_sorted(
            track.counter_events,
            [](const StreamCounterEvent& a, const StreamCounterEvent& b) {
              return a.ts_ps < b.ts_ps;
            });
        tracks.push_back(std::move(track));
      }
    }

    if (const auto it_comp = complete_by_pid_tid.find(pid);
        it_comp != complete_by_pid_tid.end()) {
      for (int pass = 0; pass < 2; ++pass) {
        for (auto& [tid, evs] : it_comp->second) {
          const auto thr_it = thread_names.find({pid, tid});
          const bool is_xla_modules =
              (thr_it != thread_names.end() && thr_it->second == "XLA Modules");
          if ((pass == 0) != is_xla_modules) continue;

          bool sorted_ts_dur = true;
          for (size_t i = 1; i < evs.size(); ++i) {
            if (evs[i].ts_ps < evs[i - 1].ts_ps ||
                (evs[i].ts_ps == evs[i - 1].ts_ps &&
                 evs[i].dur_ps > evs[i - 1].dur_ps)) {
              sorted_ts_dur = false;
              break;
            }
          }
          PreparedStreamTrack track;
          track.kind = xprof::TrackManifest::TRACK_KIND_COMPLETE;
          track.process_id = pid;
          track.thread_id = tid;
          track.ordered = sorted_ts_dur;
          track.complete_events = std::move(evs);
          tracks.push_back(std::move(track));
        }
      }
    }

    if (const auto it_async = async_by_pid_name.find(pid);
        it_async != async_by_pid_name.end()) {
      for (auto& [name, bucket] : it_async->second) {
        absl::c_stable_sort(bucket.events, [](const StreamAsyncEvent& a,
                                              const StreamAsyncEvent& b) {
          if (a.ts_ps != b.ts_ps) return a.ts_ps < b.ts_ps;
          return a.dur_ps > b.dur_ps;
        });
        PreparedStreamTrack track;
        track.kind = xprof::TrackManifest::TRACK_KIND_ASYNC;
        track.process_id = pid;
        track.thread_id = bucket.tid;
        track.name_ref = bucket.name_ref;
        track.ordered = true;
        track.async_events = std::move(bucket.events);
        tracks.push_back(std::move(track));
      }
    } else if (async_pids.contains(pid)) {
      PreparedStreamTrack track;
      track.kind = xprof::TrackManifest::TRACK_KIND_ASYNC;
      track.process_id = pid;
      track.name_ref = 0;
      track.ordered = true;
      tracks.push_back(std::move(track));
    }
  }

  std::string stream_bytes;
  // Emit Frame 0: TraceStreamHeader.
  {
    xprof::TraceStreamMessage header_msg;
    xprof::TraceStreamHeader* header = header_msg.mutable_header();
    for (const xprof::DetailItem& detail : response.details()) {
      *header->add_details() = detail;
    }
    if (response.has_full_timespan_start_ps()) {
      header->set_full_timespan_start_ps(response.full_timespan_start_ps());
    }
    if (response.has_full_timespan_end_ps()) {
      header->set_full_timespan_end_ps(response.full_timespan_end_ps());
    }
    if (response.has_metadata()) {
      *header->mutable_metadata() = response.metadata();
    }
    for (const std::string& s : interned_strings) {
      header->add_interned_strings(s);
    }
    for (uint32_t t = 0; t < tracks.size(); ++t) {
      xprof::TrackManifest* tm = header->add_tracks();
      tm->set_track_id(t);
      tm->set_kind(tracks[t].kind);
      tm->set_process_id(tracks[t].process_id);
      tm->set_thread_id(tracks[t].thread_id);
      tm->set_name_ref(tracks[t].name_ref);
      tm->set_event_count(tracks[t].event_count());
      tm->set_ordered(tracks[t].ordered);
    }
    if (absl::Status s = AppendCompressedStreamFrame(header_msg, &stream_bytes);
        !s.ok()) {
      return s;
    }
  }

  // Emit TraceEventChunk frames and compute P10 trailer counts/checksums.
  std::vector<uint64_t> track_counts(tracks.size(), 0);
  std::vector<uint64_t> track_checksums(tracks.size(), 0);
  uint64_t total_events = 0;

  constexpr size_t kMaxEventsPerChunkFrame = 65536;
  xprof::TraceStreamMessage chunk_msg;
  size_t current_chunk_events = 0;
  const auto flush_chunk_if_non_empty = [&]() -> absl::Status {
    if (chunk_msg.has_chunk() &&
        (chunk_msg.chunk().segments_size() > 0 ||
         chunk_msg.chunk().sealed_track_ids_size() > 0)) {
      if (absl::Status s =
              AppendCompressedStreamFrame(chunk_msg, &stream_bytes);
          !s.ok()) {
        return s;
      }
      chunk_msg.Clear();
      current_chunk_events = 0;
    }
    return absl::OkStatus();
  };

  for (uint32_t t = 0; t < tracks.size(); ++t) {
    const PreparedStreamTrack& track = tracks[t];
    const size_t n = track.event_count();
    if (n == 0) {
      chunk_msg.mutable_chunk()->add_sealed_track_ids(t);
      continue;
    }

    size_t offset = 0;
    while (offset < n) {
      const size_t remaining = n - offset;
      const size_t requested = next_segment_size_fn(t, remaining);
      const size_t seg_len = std::clamp<size_t>(
          requested == 0 ? remaining : requested, 1, remaining);

      if (current_chunk_events > 0 &&
          current_chunk_events + seg_len > kMaxEventsPerChunkFrame) {
        if (absl::Status s = flush_chunk_if_non_empty(); !s.ok()) {
          return s;
        }
      }

      xprof::TrackSegment* seg = chunk_msg.mutable_chunk()->add_segments();
      seg->set_track_id(t);
      seg->set_first_event_ordinal(offset);
      xprof::TraceEventSeries* series = seg->mutable_series();
      series->mutable_metadata()->set_process_id(track.process_id);
      series->mutable_metadata()->set_thread_id(track.thread_id);
      series->mutable_metadata()->set_name_ref(track.name_ref);

      if (track.kind == xprof::TrackManifest::TRACK_KIND_COMPLETE) {
        int64_t run_val = 0;
        uint32_t run_len = 0;
        const bool use_rle = track.complete_events[offset].has_rle_group_id;
        const bool use_sparse = track.complete_events[offset].is_sparse_meta;
        for (size_t k = 0; k < seg_len; ++k) {
          const StreamCompleteEvent& ev = track.complete_events[offset + k];
          const uint64_t prev_ts_ps =
              (k == 0) ? 0 : track.complete_events[offset + k - 1].ts_ps;
          const uint64_t delta_ps = ev.ts_ps - prev_ts_ps;
          series->add_deltas(delta_ps);
          series->add_durations(ev.dur_ps);
          series->add_name_refs(ev.name_ref);
          track_checksums[t] = UpdateTrackEventChecksum(
              track_checksums[t], ev.ts_ps, ev.dur_ps, ev.name_ref);

          if (use_rle) {
            if (run_len == 0) {
              run_val = ev.rle_group_id;
              run_len = 1;
            } else if (ev.rle_group_id == run_val) {
              ++run_len;
            } else {
              series->add_group_id_run_values(run_val);
              series->add_group_id_run_lengths(run_len);
              run_val = ev.rle_group_id;
              run_len = 1;
            }
          }
          if (use_sparse) {
            if (ev.meta.has_value()) {
              *series->add_event_metadata() = *ev.meta;
              series->add_event_metadata_indices(static_cast<uint32_t>(k));
            }
          } else {
            *series->add_event_metadata() =
                ev.meta.value_or(xprof::TraceEventMetadata{});
          }
        }
        if (use_rle && run_len > 0) {
          series->add_group_id_run_values(run_val);
          series->add_group_id_run_lengths(run_len);
        }
      } else if (track.kind == xprof::TrackManifest::TRACK_KIND_ASYNC) {
        int64_t run_val = 0;
        uint32_t run_len = 0;
        for (size_t k = 0; k < seg_len; ++k) {
          const StreamAsyncEvent& ev = track.async_events[offset + k];
          const uint64_t delta_ps =
              (k == 0) ? ev.ts_ps
                       : (ev.ts_ps - track.async_events[offset + k - 1].ts_ps);
          series->add_deltas(delta_ps);
          series->add_durations(ev.dur_ps);
          track_checksums[t] = UpdateTrackEventChecksum(
              track_checksums[t], ev.ts_ps, ev.dur_ps, track.name_ref);

          const int64_t gid = ev.group_id;
          if (run_len == 0) {
            run_val = gid;
            run_len = 1;
          } else if (gid == run_val) {
            ++run_len;
          } else {
            series->add_group_id_run_values(run_val);
            series->add_group_id_run_lengths(run_len);
            run_val = gid;
            run_len = 1;
          }

          if (ev.flow_id != 0 || ev.serial != 0 || ev.hlo_op_ref != 0 ||
              ev.hlo_module_ref != 0) {
            xprof::TraceEventMetadata* meta = series->add_event_metadata();
            if (ev.flow_id != 0) meta->set_flow_id(ev.flow_id);
            if (ev.serial != 0) meta->set_serial(ev.serial);
            if (ev.hlo_op_ref != 0) meta->set_hlo_op_ref(ev.hlo_op_ref);
            if (ev.hlo_module_ref != 0) {
              meta->set_hlo_module_ref(ev.hlo_module_ref);
            }
            series->add_event_metadata_indices(static_cast<uint32_t>(k));
          }
        }
        if (run_len > 0) {
          series->add_group_id_run_values(run_val);
          series->add_group_id_run_lengths(run_len);
        }
      } else if (track.kind == xprof::TrackManifest::TRACK_KIND_COUNTER) {
        if (offset == 0 && track.event_stats_ref.has_value()) {
          series->mutable_metadata()->set_event_stats_ref(
              *track.event_stats_ref);
        }
        for (size_t k = 0; k < seg_len; ++k) {
          const StreamCounterEvent& ev = track.counter_events[offset + k];
          const uint64_t prev_ts_ps =
              (k == 0) ? 0 : track.counter_events[offset + k - 1].ts_ps;
          const uint64_t delta_ps = ev.ts_ps - prev_ts_ps;
          series->add_deltas(delta_ps);
          *series->add_event_metadata() = ev.meta;
          track_checksums[t] = UpdateTrackEventChecksum(
              track_checksums[t], ev.ts_ps, /*duration_ps=*/0, track.name_ref);
        }
      }

      offset += seg_len;
      current_chunk_events += seg_len;
      if (offset == n) {
        chunk_msg.mutable_chunk()->add_sealed_track_ids(t);
      } else {
        // Flush multi-segment tracks into distinct chunk frames so streaming
        // clients exercise incremental multi-frame ingestion.
        if (absl::Status s = flush_chunk_if_non_empty(); !s.ok()) {
          return s;
        }
      }
    }
    track_counts[t] = n;
    total_events += n;
  }
  if (absl::Status s = flush_chunk_if_non_empty(); !s.ok()) {
    return s;
  }

  // Emit Final Frame: TraceStreamTrailer (P10).
  {
    xprof::TraceStreamMessage trailer_msg;
    xprof::TraceStreamTrailer* trailer = trailer_msg.mutable_trailer();
    for (uint32_t t = 0; t < tracks.size(); ++t) {
      trailer->add_track_event_counts(track_counts[t]);
      trailer->add_track_event_checksums(track_checksums[t]);
    }
    trailer->set_total_events(total_events);
    trailer->set_final_string_count(
        static_cast<uint32_t>(interned_strings.size()));
    if (absl::Status s =
            AppendCompressedStreamFrame(trailer_msg, &stream_bytes);
        !s.ok()) {
      return s;
    }
  }

  return stream_bytes;
}

absl::StatusOr<std::string> ConvertTraceDataResponseToCompressedStreamProto(
    const xprof::TraceDataResponse& response, size_t max_events_per_segment) {
  const size_t cap =
      max_events_per_segment == 0 ? 65536 : max_events_per_segment;
  return ConvertTraceDataResponseToCompressedStreamProto(
      response,
      [cap](uint32_t /*track_id*/, size_t remaining_events) -> size_t {
        return std::min(remaining_events, cap);
      });
}

}  // namespace profiler
}  // namespace tensorflow
