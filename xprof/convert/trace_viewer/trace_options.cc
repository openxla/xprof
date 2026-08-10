#include "xprof/convert/trace_viewer/trace_options.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "google/protobuf/json/json.h"
#include "re2/re2.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/trace_viewer/trace_events_filter_interface.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"
#include "xprof/convert/trace_viewer/trace_utils.h"
#include "plugin/xprof/protobuf/trace_filter_config.pb.h"

namespace tensorflow {
namespace profiler {
namespace filter_internal {

struct CompiledEventFilter {
  std::string field_name;
  tensorflow::profiler::TraceEventFilter::Operator op_id;
  std::string str_value;
  std::shared_ptr<RE2> regex;
  double double_value = 0.0;
  bool has_double_value = false;
};

// TraceEventsFilter is used to filter trace events based on TraceOptions.
class TraceEventsFilter : public TraceEventsFilterInterface {
 public:
  explicit TraceEventsFilter(const TraceOptions& options) : options_(options) {}

  void SetUp(const Trace& trace) override;

  bool Filter(const TraceEvent& event) override;

 private:
  const TraceOptions options_;

  TraceDeviceType device_type_ = TraceDeviceType::kUnknownDevice;
  absl::flat_hash_set<uint32_t /*device_id*/> tpu_noncore_devices_;
  absl::flat_hash_set<uint32_t /*device_id*/> tpu_core_devices_;

  bool has_filters_ = false;
  bool has_resource_filters_ = false;
  absl::flat_hash_set<uint32_t /*device_id*/> matching_device_ids_;
  absl::flat_hash_set<uint64_t /*packed host/thread key*/>
      matching_resource_keys_;

  const Trace* trace_ptr_ = nullptr;
  bool is_negation_ = false;
  std::vector<CompiledEventFilter> compiled_event_filters_;
};

void TraceEventsFilter::SetUp(const Trace& trace) {
  trace_ptr_ = &trace;
  for (const auto& [device_id, device] : trace.devices()) {
    if (IsTpuCoreDeviceName(device.name())) {
      device_type_ = TraceDeviceType::kTpu;
      tpu_core_devices_.insert(device_id);
    } else if (MaybeTpuNonCoreDeviceName(device.name())) {
      tpu_noncore_devices_.insert(device_id);
    }
  }

  if (options_.trace_filter_config.has_value()) {
    const auto& config = *options_.trace_filter_config;
    is_negation_ = config.negation();

    compiled_event_filters_.reserve(config.trace_event_filters().size());
    for (const auto& event_filter : config.trace_event_filters()) {
      CompiledEventFilter compiled;
      compiled.field_name = event_filter.field_name();
      compiled.op_id = event_filter.op_id();
      if (event_filter.has_str_value()) {
        compiled.str_value = event_filter.str_value();
      }
      if (event_filter.has_regex_value()) {
        compiled.regex = std::make_shared<RE2>(event_filter.regex_value());
      } else if (compiled.op_id ==
                     tensorflow::profiler::TraceEventFilter::OP_REGEX &&
                 event_filter.has_str_value()) {
        compiled.regex = std::make_shared<RE2>(event_filter.str_value());
      }
      if (event_filter.has_double_value()) {
        compiled.double_value = event_filter.double_value();
        compiled.has_double_value = true;
      }
      compiled_event_filters_.push_back(std::move(compiled));
    }

    std::vector<std::unique_ptr<RE2>> device_matchers;
    device_matchers.reserve(config.device_regexes().size());
    for (const auto& regex : config.device_regexes()) {
      device_matchers.push_back(std::make_unique<RE2>(regex));
    }
    std::vector<std::unique_ptr<RE2>> resource_matchers;
    for (const auto& regex : config.resource_regexes()) {
      resource_matchers.push_back(std::make_unique<RE2>(regex));
    }

    if (!device_matchers.empty() || !resource_matchers.empty()) {
      has_filters_ = true;
      if (!resource_matchers.empty()) {
        has_resource_filters_ = true;
      }
      for (const auto& [device_id, device] : trace.devices()) {
        bool device_match = device_matchers.empty();
        for (const auto& matcher : device_matchers) {
          if (RE2::PartialMatch(device.name(), *matcher)) {
            device_match = true;
            break;
          }
        }

        for (const auto& [resource_id, resource] : device.resources()) {
          bool resource_match = resource_matchers.empty();
          for (const auto& matcher : resource_matchers) {
            if (RE2::PartialMatch(resource.name(), *matcher)) {
              resource_match = true;
              break;
            }
          }

          if (device_match && resource_match) {
            matching_device_ids_.insert(device_id);
            matching_resource_keys_.insert(
                (static_cast<uint64_t>(device_id) << 32) | resource_id);
          }
        }
      }
    }
  }
}

namespace {

bool MatchCompiledFilter(const TraceEvent& event,
                         const CompiledEventFilter& filter,
                         const Trace& trace) {
  if (filter.field_name == "name") {
    const std::string& event_name =
        event.has_name_ref() ? trace.name_table().at(event.name_ref())
                             : event.name();

    if (filter.op_id == tensorflow::profiler::TraceEventFilter::OP_EQ) {
      if (filter.regex != nullptr) {
        return RE2::PartialMatch(event_name, *filter.regex);
      }
      return event_name == filter.str_value;
    } else if (filter.op_id ==
               tensorflow::profiler::TraceEventFilter::OP_REGEX) {
      if (filter.regex != nullptr) {
        return RE2::PartialMatch(event_name, *filter.regex);
      }
      return false;
    }
  } else if (filter.field_name == "duration") {
    double duration_ms = static_cast<double>(event.duration_ps()) / 1e9;
    if (filter.has_double_value) {
      double val = filter.double_value;
      switch (filter.op_id) {
        case tensorflow::profiler::TraceEventFilter::OP_EQ:
          return std::abs(duration_ms - val) < 1e-6;
        case tensorflow::profiler::TraceEventFilter::OP_LT:
          return duration_ms < val;
        case tensorflow::profiler::TraceEventFilter::OP_GT:
          return duration_ms > val;
        case tensorflow::profiler::TraceEventFilter::OP_LE:
          return duration_ms <= val;
        case tensorflow::profiler::TraceEventFilter::OP_GE:
          return duration_ms >= val;
        default:
          return false;
      }
    }
  }
  return true;
}

}  // namespace

bool TraceEventsFilter::Filter(const TraceEvent& event) {
  if (has_filters_) {
    if (!matching_device_ids_.contains(event.device_id())) {
      return true;
    }
    if (has_resource_filters_ &&
        !matching_resource_keys_.contains(
            (static_cast<uint64_t>(event.device_id()) << 32) |
            event.resource_id())) {
      return true;
    }
  }

  if (trace_ptr_ != nullptr && !compiled_event_filters_.empty()) {
    bool event_matched = true;
    for (const auto& filter : compiled_event_filters_) {
      if (!MatchCompiledFilter(event, filter, *trace_ptr_)) {
        event_matched = false;
        break;
      }
    }
    if (is_negation_ == event_matched) {
      return true;  // filter out
    }
  }

  switch (device_type_) {
    case TraceDeviceType::kUnknownDevice:
      break;
    case TraceDeviceType::kTpu:
      if (tpu_noncore_devices_.contains(event.device_id()) ||
          tpu_core_devices_.contains(event.device_id())) {
        // Filter intermediate DMA flow events unless "Full DMA" is checked.
        if (IsFlowMid(event)) return !options_.full_dma;
      }
      break;
    case TraceDeviceType::kGpu:
      break;
  }
  return false;
}

}  // namespace filter_internal

TraceOptions TraceOptionsFromToolOptions(const ToolOptions& tool_options) {
  TraceOptions options;
  options.full_dma =
      GetParamWithDefault<bool>(tool_options, kFullDma, options.full_dma);
  options.enable_legacy_dcn = GetParamWithDefault<bool>(
      tool_options, kEnableLegacyDcn, options.enable_legacy_dcn);

  if (auto filter_config_param =
          GetParam<std::string>(tool_options, "trace_filter_config")) {
    tensorflow::profiler::TraceFilterConfig trace_filter_config;
    google::protobuf::util::JsonParseOptions json_options;
    json_options.ignore_unknown_fields = true;
    if (google::protobuf::util::JsonStringToMessage(*filter_config_param,
                                          &trace_filter_config, json_options)
            .ok()) {
      options.trace_filter_config = trace_filter_config;
    }
  }
  return options;
}

JsonTraceOptions::Details TraceOptionsToDetails(TraceDeviceType device_type,
                                                const TraceOptions& options) {
  JsonTraceOptions::Details details;
  switch (device_type) {
    case TraceDeviceType::kUnknownDevice:
      break;
    case TraceDeviceType::kTpu:
      details.push_back({kFullDma, options.full_dma});
      break;
    case TraceDeviceType::kGpu:
      break;
  }
  return details;
}

std::unique_ptr<tensorflow ::profiler::TraceEventsFilterInterface>
CreateTraceEventsFilterFromTraceOptions(const TraceOptions& options) {
  return std::make_unique<filter_internal::TraceEventsFilter>(options);
}

}  // namespace profiler
}  // namespace tensorflow
