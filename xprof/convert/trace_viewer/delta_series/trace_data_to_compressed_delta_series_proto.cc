#include "xprof/convert/trace_viewer/delta_series/trace_data_to_compressed_delta_series_proto.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"
#include "tsl/profiler/lib/context_types.h"
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

}  // namespace profiler
}  // namespace tensorflow
