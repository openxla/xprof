/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_EVENTS_TO_JSON_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_EVENTS_TO_JSON_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/macros.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/time/time.h"
#include "re2/re2.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/context_types.h"
#include "xprof/convert/trace_viewer/trace_events_util.h"
#include "xprof/convert/trace_viewer/trace_viewer_color.h"
#include "plugin/xprof/protobuf/task.pb.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace tensorflow {
namespace profiler {

namespace internal {

// MPMD module information extracted from an event name.
struct MpmdModuleInfo {
  std::string program_key;
  int group_id = 0;
  int loop_id = 0;
  int min_layer = 0;
  bool has_explicit_layer = false;
};

// Normalizes an MPMD raw program name by stripping dynamic sequence-length
// bucket suffixes (e.g. "_bucket_128k", "_bucket_512k").
inline std::string NormalizeMpmdProgramKey(absl::string_view raw_program) {
  static const LazyRE2 kBucketSuffixRe = {R"((_bucket_[A-Za-z0-9_]+)$)"};
  std::string normalized(raw_program);
  RE2::Replace(&normalized, *kBucketSuffixRe, "");
  return normalized;
}

// Extracts MPMD module information from an event name.
// Supports both Shardy "XLA Modules" naming conventions and legacy patterns.
inline std::optional<MpmdModuleInfo> ExtractMpmdModuleInfo(
    absl::string_view event_name) {
  // First check general Shardy "XLA Modules" format:
  // Examples:
  //   p0_loop_0_layer_0_0.inc_prefill_step_32k_chunk_4096(12345)
  //   p24_loop_1_layer_24_24.inc_prefill_step_32k_chunk_4096(12345)
  //   p1_inferred.inc_prefill_session_32k(12345)
  //   p0_inferred.inc_prefill_final_32k(12345)
  //   p0_stage0.inc_prefill_step_4k_bucket_128k(12345)
  static const LazyRE2 kShardyModuleRe = {
      R"(^p(\d+)_([A-Za-z0-9_.\-]*?)\.+([^\.\(]+)(?:\(|$))"};
  std::string group_str;
  std::string descriptor;
  std::string raw_program;
  if (RE2::PartialMatch(event_name, *kShardyModuleRe, &group_str, &descriptor,
                        &raw_program)) {
    MpmdModuleInfo info;
    if (!absl::SimpleAtoi(group_str, &info.group_id)) {
      return std::nullopt;
    }
    info.program_key = NormalizeMpmdProgramKey(raw_program);

    // Extract loop_id from descriptor if present, e.g. "loop_0" or "_loop_0".
    static const LazyRE2 kLoopRe = {R"((?:^|_)loop_(\d+))"};
    int loop_id = 0;
    if (RE2::PartialMatch(descriptor, *kLoopRe, &loop_id)) {
      info.loop_id = loop_id;
    } else {
      info.loop_id = 0;
    }

    // Check for layer or stage in descriptor.
    static const LazyRE2 kLayerRangeRe = {
        R"((?:^|_)(?:layer_(\d+)_(\d+)|layer_(\d+)|stage(\d+)))"};
    std::string layer1_str;
    std::string layer2_str;
    std::string single_layer_str;
    std::string stage_str;
    if (RE2::PartialMatch(descriptor, *kLayerRangeRe, &layer1_str, &layer2_str,
                          &single_layer_str, &stage_str)) {
      info.has_explicit_layer = true;
      if (!layer1_str.empty()) {
        int l1 = 0;
        int l2 = 0;
        absl::SimpleAtoi(layer1_str, &l1);
        absl::SimpleAtoi(layer2_str, &l2);
        info.min_layer = std::min(l1, l2);
      } else if (!single_layer_str.empty()) {
        absl::SimpleAtoi(single_layer_str, &info.min_layer);
      } else {
        absl::SimpleAtoi(stage_str, &info.min_layer);
      }
    } else {
      // Non-layer Shardy programs (e.g. p1_inferred.inc_prefill_session_32k).
      info.has_explicit_layer = false;
      info.min_layer = info.group_id;
    }
    return info;
  }

  // Fallback: check legacy _stage(\d+) or _layer_(\d+)(?:_(\d+))? anywhere in
  // event_name for backward compatibility with synthetic unit tests.
  static const LazyRE2 kLegacyRe = {
      R"((?:^|_)(?:layer_(\d+)(?:_(\d+))?|stage(\d+))(?:\.([^\(\)]+))?)"};
  std::string legacy_l1;
  std::string legacy_l2;
  std::string legacy_stage;
  std::string legacy_prog;
  if (RE2::PartialMatch(event_name, *kLegacyRe, &legacy_l1, &legacy_l2,
                        &legacy_stage, &legacy_prog)) {
    MpmdModuleInfo info;
    info.has_explicit_layer = true;
    info.loop_id = 0;
    if (!legacy_prog.empty()) {
      info.program_key = NormalizeMpmdProgramKey(legacy_prog);
    } else {
      info.program_key = "";
    }
    if (!legacy_l1.empty()) {
      int l1 = 0;
      absl::SimpleAtoi(legacy_l1, &l1);
      if (!legacy_l2.empty()) {
        int l2 = 0;
        absl::SimpleAtoi(legacy_l2, &l2);
        info.min_layer = std::min(l1, l2);
      } else {
        info.min_layer = l1;
      }
    } else {
      absl::SimpleAtoi(legacy_stage, &info.min_layer);
    }
    info.group_id = info.min_layer;
    return info;
  }

  return std::nullopt;
}

}  // namespace internal

// The JSON parser's 700MB limit is tested empirically to hold up to 16M
// counter events. (go/xprof-event-counter-fix). Conservatively setting
// this to 10M toward room for other events.
inline constexpr size_t kMaxCounterEvents = 10'000'000;

// JSON generation options.
struct JsonTraceOptions {
  using Details = std::vector<std::pair<std::string, bool>>;

  // Options and values for filtering based on the "details" menu.
  Details details;

  // Device IDs of devices whose resources should be sorted by name instead of
  // by resource ID.
  absl::flat_hash_set<uint32_t /*device_id*/> sort_resources_by_name;

  // Returns the color for an event.
  TraceEventsColorerInterface* colorer = nullptr;

  bool generate_stack_frames = true;
  bool use_new_backend = false;
  bool mpmd_pipeline_view = false;
  std::string code_link;
  // The absolute walltime timestamp in nanoseconds used as the baseline for
  // this snapshot's trace events (computed via `xprof::GetHostStartNs`). This
  // is set only by the live trace viewer to allow the frontend to align
  // timestamps across live streaming snapshots.
  std::optional<uint64_t> snapshot_baseline_ns;
};

// Counts generated JSON events by type.
class JsonEventCounter {
 public:
  JsonEventCounter() : event_count_(kNumEventTypes, 0) {}
  ~JsonEventCounter() { LOG(INFO) << ToString(); }

  // Types of JSON events (bit.ly/trace-event-format)
  enum EventType {
    kCompleteEvent,
    kCompleteEventWithFlow,
    kCounterEvent,
    kAsyncEvent,
  };

  void Inc(EventType e) { ++event_count_[e]; }

  std::string ToString() const {
    std::string output = "Generated JSON events:";
    for (size_t i = 0; i < event_count_.size(); ++i) {
      absl::StrAppend(&output, " ", kEventTypeName[i], ": ", event_count_[i]);
    }
    return output;
  }

  size_t GetCounterEventCount() const { return event_count_[kCounterEvent]; }

 private:
  static constexpr absl::string_view kEventTypeName[] = {
      "complete",
      "complete+flow",
      "counter",
      "async",
  };

  static constexpr size_t kNumEventTypes = std::size(kEventTypeName);

  absl::FixedArray<size_t> event_count_;
};

// Adds a separator between elements of a JSON array or object.
template <typename IOBuffer>
class JsonSeparator {
 public:
  explicit JsonSeparator(IOBuffer* output) : output_(output) {}

  // Does nothing on the first call; adds a comma to the output on subsequent
  // calls.
  void Add() {
    output_->Append(sep_);
    sep_ = ",";
  }

 private:
  IOBuffer* output_;
  absl::string_view sep_;
};

// Converts picoseconds to microseconds.
inline double PicosToMicros(uint64_t ps) { return ps / 1E6; }

// Escapes the contents of "raw" in JSON style.
// Also adds double quotes to the beginning and end of the string.
std::string JsonEscape(absl::string_view raw);

std::string ProtoString(const tsl::protobuf::Message& pb);

template <typename RawDataType, typename IOBuffer>
void WriteTpuData(const RawDataType& data, JsonSeparator<IOBuffer>* separator,
                  IOBuffer* output) {}

// Writes JSON events from a TraceEvent.
template <typename IOBuffer, typename RawDataType>
class JsonEventWriter {
 public:
  JsonEventWriter(const TraceEventsColorerInterface* colorer,
                  const Trace& trace,
                  const absl::btree_map<uint64_t, uint64_t>& references,
                  IOBuffer* output)
      : colorer_(colorer),
        trace_(trace),
        references_(references),
        output_(output) {}

  void WriteEvent(const TraceEvent& event) const {
    std::optional<TraceEvent> async_event;
    output_->Append(R"({"pid":)", event.device_id());
    if (event.has_resource_id()) {
      output_->Append(R"(,"tid":)", event.resource_id());
    }
    const std::string& event_name =
        event.has_name_ref() ? trace_.name_table().at(event.name_ref())
                             : event.name();
    output_->Append(R"(,"name":)", JsonEscape(event_name));
    tsl::profiler::Timespan span = EventSpan(event);
    // "%.17g" is the default double format in google::protobuf::util::JsonFormat.
    absl::Format(output_, R"(,"ts":%.17g)", PicosToMicros(span.begin_ps()));
    JsonEventCounter::EventType event_type = JsonEventCounter::kCounterEvent;
    if (event.has_resource_id()) {
      event_type = event.has_flow_id()
                       ? JsonEventCounter::kCompleteEventWithFlow
                       : JsonEventCounter::kCompleteEvent;
      // A complete event must have a duration, otherwise trace-viewer will
      // extend the event to the end of the trace and append "(Did Not Finish)"
      // to its name. Make the minimum duration 1 picosecond.
      uint64_t duration_ps = std::max(span.duration_ps(), uint64_t{1});
      absl::Format(output_, R"(,"dur":%.17g)", PicosToMicros(duration_ps));

      if (std::optional<uint32_t> color_id = colorer_->GetColor(event)) {
        output_->Append(R"(,"cname":)", TraceViewerColorName(*color_id));
      }

      // FlowV2
      if (event_type == JsonEventCounter::kCompleteEventWithFlow) {
        output_->Append(R"(,"bind_id":)", event.flow_id());
        if (event.has_flow_category()) {
          tsl::profiler::ContextType type =
              tsl::profiler::GetSafeContextType(event.flow_category());
          if (type != tsl::profiler::ContextType::kGeneric &&
              type != tsl::profiler::ContextType::kLegacy) {
            const char* category = tsl::profiler::GetContextTypeString(type);
            output_->Append(R"(,"cat":")", category, R"(")");
          }
        }
        switch (event.flow_entry_type()) {
          case TraceEvent::FLOW_NONE:
            // The caller prevents this case from happening.
            break;
          case TraceEvent::FLOW_START:
            output_->Append(R"(,"flow_out":true)");
            break;
          case TraceEvent::FLOW_MID:
            output_->Append(R"(,"flow_in":true,"flow_out":true)");
            break;
          case TraceEvent::FLOW_END:
            output_->Append(R"(,"flow_in":true)");
            break;
        }
      }
      output_->Append(R"(,"ph":"X")");
    } else {
      event_type = event.has_flow_id() ? JsonEventCounter::kAsyncEvent
                                       : JsonEventCounter::kCounterEvent;
      if (event_type == JsonEventCounter::kCounterEvent) {
        output_->Append(R"(,"ph":"C")");
      } else {  // async events
        output_->Append(R"(,"id":)", event.flow_id());
        if (event.has_flow_category()) {
          tsl::profiler::ContextType type =
              tsl::profiler::GetSafeContextType(event.flow_category());
          const char* category = tsl::profiler::GetContextTypeString(type);
          output_->Append(R"(,"cat":")", category, R"(")");
        }
        switch (event.flow_entry_type()) {
          case TraceEvent::FLOW_NONE:
            // The caller prevents this case from happening.
            break;
          case TraceEvent::FLOW_START:
            output_->Append(R"(,"ph":"b")");
            break;
          case TraceEvent::FLOW_END:
            output_->Append(R"(,"ph":"e")");
            break;
          case TraceEvent::FLOW_MID:
            output_->Append(R"(,"ph":"b")");
            async_event.emplace(event);
            async_event->set_flow_entry_type(TraceEvent::FLOW_END);
            async_event->set_timestamp_ps(event.timestamp_ps() +
                                          event.duration_ps());
            async_event->clear_raw_data();
            break;
        }
      }
    }
    WriteArgs(event);
    if (event.has_serial()) {
      output_->Append(R"(,"z":)", event.serial());
    }

    output_->Append("}");
    counter_.Inc(event_type);
    if (async_event) {
      output_->Append(",");
      WriteEvent(*async_event);
    }
  }

  size_t GetCounterEventCount() const {
    return counter_.GetCounterEventCount();
  }

  bool isMatchingLastCounterEvent(const TraceEvent& event) const {
    const std::string& event_name =
        event.has_name_ref() ? trace_.name_table().at(event.name_ref())
                             : event.name();
    auto key = std::make_pair(event.device_id(), event_name);
    return last_counter_event_key_ == key;
  }

  void AddCounterEvent(const TraceEvent& event) {
    counter_.Inc(JsonEventCounter::kCounterEvent);
    const std::string& event_name =
        event.has_name_ref() ? trace_.name_table().at(event.name_ref())
                             : event.name();
    auto key = std::make_pair(event.device_id(), event_name);
    if (last_counter_event_key_ != key) {
      last_counter_event_key_ = key;

      std::string event_stats_str = "";
      if (event.has_raw_data()) {
        RawDataType data;
        if (data.ParseFromString(event.raw_data()) && data.has_args()) {
          if (data.args().arg_size() > 0) {
            event_stats_str = absl::StrFormat(R"(,"event_stats":"%s")",
                                              data.args().arg(0).name());
          }
        }
      }
      output_->Append(
          absl::StrFormat(R"({"pid":%d,"name":"%s","ph":"C"%s,"entries":[)",
                          event.device_id(), event_name, event_stats_str));
    }

    std::vector<std::string> entry_values;
    if (event.has_raw_data()) {
      RawDataType data;
      if (!data.ParseFromString(event.raw_data())) {
        LOG(WARNING) << "Failed to parse raw data for event: " << event_name;
        return;
      }
      if (!data.has_args()) {
        return;
      }
      for (const auto& arg : data.args().arg()) {
        entry_values.push_back(GetArgValue(arg));
      }
    }
    if (entry_values.empty()) return;
    output_->Append(absl::StrFormat(R"([%.17g,%s])",
                                    PicosToMicros(event.timestamp_ps()),
                                    absl::StrJoin(entry_values, ",")));
  }

 private:
  std::string GetArgValue(const TraceEventArguments::Argument& arg) {
    switch (arg.value_case()) {
      case TraceEventArguments::Argument::kStrValue:
        return JsonEscape(arg.str_value());
      case TraceEventArguments::Argument::kIntValue:
        return absl::StrCat(arg.int_value());
      case TraceEventArguments::Argument::kUintValue:
        return absl::StrCat(arg.uint_value());
      case TraceEventArguments::Argument::kDoubleValue:
        // Displaying two decimal places for Perf Counter based stats.
        if (IsPerfCounterBasedStats(arg.name())) {
          return absl::StrFormat("%.2f", arg.double_value());
        }
        return absl::StrFormat("%.17g", arg.double_value());
      case TraceEventArguments::Argument::kRefValue: {
        const auto& it = trace_.name_table().find(arg.ref_value());
        if (it != trace_.name_table().end()) {
          return JsonEscape(it->second);
        }
        return "";
      }
      case TraceEventArguments::Argument::VALUE_NOT_SET:
        LOG(WARNING) << "Value not set for argument: " << arg.name();
        return "";
      default:
        LOG(WARNING) << "Unexpected value type for argument: " << arg.name();
        return "";
    }
  }
  void WriteArgs(const TraceEvent& event) const {
    if (!event.has_group_id() && !event.has_raw_data()) {
      return;
    }
    output_->Append(R"(,"args":{)");
    std::optional<uint64_t> stack_frames;
    JsonSeparator<IOBuffer> separator(output_);
    if (event.has_group_id()) {
      separator.Add();
      output_->Append(R"("group_id":)", event.group_id());
    }
    if (event.has_raw_data()) {
      RawDataType data;
      data.ParseFromString(event.raw_data());
      switch (data.raw_data_case()) {
        case RawDataType::RAW_DATA_NOT_SET:
          break;
        case RawDataType::kTpuData:
          WriteTpuData<RawDataType, IOBuffer>(data, &separator, output_);
          break;
        case RawDataType::kDmaActivity:
          separator.Add();
          output_->Append(R"("DMA activity":)",
                          ProtoString(data.dma_activity()));
          break;
        case RawDataType::kArgs:
          for (const auto& arg : data.args().arg()) {
            switch (arg.value_case()) {
              case TraceEventArguments::Argument::kStrValue:
                separator.Add();
                WriteArg(arg.name(), arg.str_value());
                break;
              case TraceEventArguments::Argument::kIntValue:
                separator.Add();
                WriteArg(arg.name(), arg.int_value());
                break;
              case TraceEventArguments::Argument::kUintValue:
                separator.Add();
                WriteArg(arg.name(), arg.uint_value());
                break;
              case TraceEventArguments::Argument::kDoubleValue:
                separator.Add();
                WriteArg(arg.name(), arg.double_value());
                break;
              case TraceEventArguments::Argument::kRefValue: {
                const auto& it = trace_.name_table().find(arg.ref_value());
                if (it != trace_.name_table().end()) {
                  // Each event could only have one stack frame.
                  if (absl::StartsWith(it->second, "@@") && !stack_frames) {
                    stack_frames = arg.ref_value();
                  } else {
                    separator.Add();
                    WriteArg(arg.name(), it->second);
                  }
                }
                break;
              }
              case TraceEventArguments::Argument::VALUE_NOT_SET:
                break;
            }
          }
          break;
      }
    }
    output_->Append("}");

    // Write the optional stack frame.
    if (stack_frames.has_value()) {
      output_->Append(R"(,"sf":)", references_.at(*stack_frames), R"()");
    }
  }
  void WriteArg(absl::string_view name, absl::string_view value) const {
    output_->Append(JsonEscape(name), ":", JsonEscape(value));
  }
  void WriteArg(absl::string_view name, uint64_t value) const {
    // Limit beyond which integers converted to 64-bit IEEE floating point may
    // lose accuracy. JavaScript stores all numbers as doubles, quote the value
    // to preserve accuracy.
    // https://en.wikipedia.org/wiki/Double-precision_floating-point_format
    constexpr uint64_t kIeeeLimit = 1ULL << 53;
    if (value > kIeeeLimit) {
      output_->Append(JsonEscape(name), ":\"", value, "\"");
    } else {
      output_->Append(JsonEscape(name), ":", value);
    }
  }
  void WriteArg(absl::string_view name, int64_t value) const {
    // Limit beyond which integers converted to 64-bit IEEE floating point may
    // lose accuracy. JavaScript stores all numbers as doubles, quote the value
    // to preserve accuracy.
    // https://en.wikipedia.org/wiki/Double-precision_floating-point_format
    constexpr uint64_t kIeeeLimit = 1ULL << 53;
    if (abs(value) > kIeeeLimit) {
      output_->Append(JsonEscape(name), ":\"", value, "\"");
    } else {
      output_->Append(JsonEscape(name), ":", value);
    }
  }
  void WriteArg(absl::string_view name, double value) const {
    if (std::isfinite(value)) {
      output_->Append(JsonEscape(name));
      // "%.17g" is the default double format in google::protobuf::util::JsonFormat.
      if (IsPerfCounterBasedStats(name)) {
        absl::Format(output_, ":%.2f", value);
      } else {
        absl::Format(output_, ":%.17g", value);
      }
    } else if (std::isinf(value)) {
      output_->Append(JsonEscape(name), R"(:"Infinity")");
    } else if (std::isinf(-value)) {
      output_->Append(JsonEscape(name), R"(:"-Infinity")");
    } else {
      output_->Append(JsonEscape(name), R"(:"NaN")");
    }
  }

  const TraceEventsColorerInterface* colorer_;
  const Trace& trace_;
  const absl::btree_map<uint64_t, uint64_t>& references_;
  IOBuffer* output_;
  mutable JsonEventCounter counter_;
  std::pair<uint32_t, std::string> last_counter_event_key_ = {0, ""};
};

template <typename IOBuffer>
void WriteTasks(const Trace& trace, IOBuffer* output) {
  const auto& tasks = trace.tasks();
  if (tasks.empty()) return;
  output->Append(R"("tasks":[)");
  JsonSeparator<IOBuffer> task_separator(output);
  absl::btree_map<uint32_t, Task> ordered_tasks(tasks.begin(), tasks.end());
  for (const auto& entry : ordered_tasks) {
    const uint32_t host_id = entry.first;
    const auto& task = entry.second;

    task_separator.Add();
    output->Append("{");
    JsonSeparator<IOBuffer> field_separator(output);
    field_separator.Add();
    output->Append(R"("host_id":)", host_id);
    if (task.has_changelist()) {
      field_separator.Add();
      output->Append(R"("changelist":)", task.changelist());
    }
    if (task.has_clean_build()) {
      field_separator.Add();
      output->Append(R"("clean_build":)", task.clean_build());
    }
    if (task.has_build_time()) {
      field_separator.Add();
      output->Append(
          R"("build_time":)",
          JsonEscape(absl::FormatTime(absl::FromUnixNanos(task.build_time()),
                                      absl::UTCTimeZone())));
    }
    if (task.has_build_target()) {
      field_separator.Add();
      output->Append(R"("build_target":)", JsonEscape(task.build_target()));
    }
    if (task.has_command_line()) {
      field_separator.Add();
      output->Append(R"("command_line":)", JsonEscape(task.command_line()));
    }
    if (task.has_start_time()) {
      field_separator.Add();
      output->Append(
          R"("start_time":)",
          JsonEscape(absl::FormatTime(absl::FromUnixNanos(task.start_time()),
                                      absl::UTCTimeZone())));
    }
    if (task.has_gtc_freq_hz()) {
      field_separator.Add();
      output->Append(R"("gtc_freq_hz":)", task.gtc_freq_hz());
    }
    if (task.has_tensor_core_freq_hz()) {
      field_separator.Add();
      output->Append(R"("tensor_core_freq_hz":)", task.tensor_core_freq_hz());
    }
    if (task.has_sparse_core_freq_hz()) {
      field_separator.Add();
      output->Append(R"("sparse_core_freq_hz":)", task.sparse_core_freq_hz());
    }
    output->Append("}");
  }
  output->Append("],");
}

template <typename IOBuffer>
void WriteStackFrames(const Trace& trace,
                      const absl::btree_map<uint64_t, uint64_t>& references,
                      IOBuffer* output) {
  const auto& name_table = trace.name_table();
  output->Append(R"("stackFrames":{)");
  JsonSeparator<IOBuffer> separator(output);
  for (const auto& [fp, name] : name_table) {
    if (!absl::StartsWith(name, "@@")) continue;
    separator.Add();
    std::string_view name_view = name;
    absl::ConsumePrefix(&name_view, "@@");
    output->Append(R"(")", references.at(fp), R"(":{"name":)",
                   JsonEscape(name_view), R"(})");
  }
  output->Append("},");
}

template <typename IOBuffer>
void WriteDetails(const JsonTraceOptions::Details& details, IOBuffer* output) {
  if (details.empty()) return;
  output->Append(R"("details":[)");
  JsonSeparator<IOBuffer> separator(output);
  for (const auto& detail : details) {
    separator.Add();
    output->Append(R"({"name":)", JsonEscape(detail.first), R"(,"value":)",
                   detail.second ? "true" : "false", "}");
  }
  output->Append("],");
}

absl::btree_map<uint64_t, uint64_t> BuildStackFrameReferences(
    const Trace& trace);

template <typename IOBuffer>
void WriteReturnedEventsSize(const int events_size, IOBuffer* output) {
  output->Append(R"("returnedEventsSize":)", events_size, R"(,)");
}

template <typename IOBuffer>
void WriteFilteredByVisibility(bool filtered_by_visibility, IOBuffer* output) {
  absl::string_view filtered_by_visibility_str =
      filtered_by_visibility ? "true" : "false";
  output->Append(R"("filteredByVisibility":)", filtered_by_visibility_str,
                 R"(,)");
}

template <typename IOBuffer>
void WriteTraceFullTimespan(const Trace* trace, IOBuffer* output) {
  auto start_time_ms = trace->min_timestamp_ps() / 1000000000.0;
  auto end_time_ms = trace->max_timestamp_ps() / 1000000000.0;
  output->Append(R"("fullTimespan":[)", start_time_ms, R"(,)", end_time_ms,
                 R"(],)");
}

// Base offset for process sort indices of devices not ranked into an MPMD
// stage. Offsets unranked devices past stage ranks with headroom for PID
// tie-breaking.
inline constexpr uint32_t kMpmdUnrankedSortIndexBase = 1u << 30;

// Template functions are implicitly inline when defined in a header.
// TODO(b/483237058): Cache the results of SortMpmdDevices.
template <typename TraceEventsContainer>
void SortMpmdDevices(
    const TraceEventsContainer& events,
    absl::flat_hash_map<uint32_t, uint32_t>& device_to_sort_index) {
  device_to_sort_index.clear();

  const Trace& trace = events.trace();
  absl::flat_hash_set<std::pair<uint32_t, uint64_t>> xla_modules_resources;
  bool has_any_xla_modules_resource = false;
  for (const auto& [device_id, device] : trace.devices()) {
    for (const auto& [resource_id, resource] : device.resources()) {
      if (resource.name() == "XLA Modules") {
        xla_modules_resources.insert({device_id, resource_id});
        has_any_xla_modules_resource = true;
      }
    }
  }

  // Per-device statistics per (program_key, has_explicit_layer).
  struct DeviceProgramStats {
    uint64_t min_timestamp_ps = std::numeric_limits<uint64_t>::max();
    std::pair<int, int> min_stage = {std::numeric_limits<int>::max(),
                                     std::numeric_limits<int>::max()};
  };

  using ProgramKeyTier = std::pair<std::string, bool>;
  absl::flat_hash_map<uint32_t,
                      absl::flat_hash_map<ProgramKeyTier, DeviceProgramStats>>
      device_program_stats;
  absl::flat_hash_map<ProgramKeyTier, uint64_t> global_program_min_timestamp_ps;

  events.ForAllEvents([&](const TraceEvent& event) {
    if (has_any_xla_modules_resource &&
        !xla_modules_resources.contains(
            {event.device_id(), event.resource_id()})) {
      return;
    }
    const std::string& event_name =
        event.has_name_ref() ? trace.name_table().at(event.name_ref())
                             : event.name();
    if (const std::optional<internal::MpmdModuleInfo> module_info =
            internal::ExtractMpmdModuleInfo(event_name);
        module_info.has_value()) {
      const ProgramKeyTier key_tier{module_info->program_key,
                                    module_info->has_explicit_layer};
      const uint64_t ts = event.timestamp_ps();
      const std::pair<int, int> stage{module_info->loop_id,
                                      module_info->min_layer};

      DeviceProgramStats& dev_stats =
          device_program_stats[event.device_id()][key_tier];
      dev_stats.min_timestamp_ps = std::min(dev_stats.min_timestamp_ps, ts);
      dev_stats.min_stage = std::min(dev_stats.min_stage, stage);

      const auto [global_it, inserted] =
          global_program_min_timestamp_ps.try_emplace(key_tier, ts);
      if (!inserted) {
        global_it->second = std::min(global_it->second, ts);
      }
    }
  });

  if (device_program_stats.empty()) {
    return;
  }

  // Device entry used for final sorting.
  struct DeviceSortEntry {
    uint64_t global_program_min_timestamp_ps = 0;
    std::string program_key;
    int loop_id = 0;
    int min_layer = 0;
    uint32_t device_id = 0;
  };

  std::vector<DeviceSortEntry> sorted_devices;
  sorted_devices.reserve(device_program_stats.size());

  for (const auto& [device_id, program_map] : device_program_stats) {
    bool has_tier1 = false;
    for (const auto& [key_tier, _] : program_map) {
      if (key_tier.second) {
        has_tier1 = true;
        break;
      }
    }

    const ProgramKeyTier* best_key_tier = nullptr;
    const DeviceProgramStats* best_stats = nullptr;

    for (const auto& [key_tier, stats] : program_map) {
      if (has_tier1 && !key_tier.second) {
        continue;
      }
      if (best_stats == nullptr ||
          std::tie(stats.min_timestamp_ps, key_tier.first) <
              std::tie(best_stats->min_timestamp_ps, best_key_tier->first)) {
        best_key_tier = &key_tier;
        best_stats = &stats;
      }
    }

    DeviceSortEntry entry;
    entry.device_id = device_id;
    entry.program_key = best_key_tier->first;
    entry.loop_id = best_stats->min_stage.first;
    entry.min_layer = best_stats->min_stage.second;
    entry.global_program_min_timestamp_ps =
        global_program_min_timestamp_ps.at(*best_key_tier);
    sorted_devices.push_back(entry);
  }

  absl::c_sort(sorted_devices,
               [](const DeviceSortEntry& a, const DeviceSortEntry& b) {
                 if (a.global_program_min_timestamp_ps !=
                     b.global_program_min_timestamp_ps) {
                   return a.global_program_min_timestamp_ps <
                          b.global_program_min_timestamp_ps;
                 }
                 if (a.program_key != b.program_key) {
                   return a.program_key < b.program_key;
                 }
                 if (a.loop_id != b.loop_id) {
                   return a.loop_id < b.loop_id;
                 }
                 if (a.min_layer != b.min_layer) {
                   return a.min_layer < b.min_layer;
                 }
                 return a.device_id < b.device_id;
               });

  for (size_t i = 0; i < sorted_devices.size(); ++i) {
    device_to_sort_index[sorted_devices[i].device_id] =
        static_cast<uint32_t>(i);
  }
}

template <typename IOBuffer, typename TraceEventsContainer,
          typename RawDataType>
void TraceEventsToJson(const JsonTraceOptions& options,
                       const TraceEventsContainer& events, IOBuffer* output) {
  // Set the displayTimeUnit to nanoseconds (default is milliseconds), so the UI
  // uses higher-precision when manipulating event times. Note that the
  // timestamps of trace events are always given in microseconds.
  output->Append(
      R"({"displayTimeUnit":"ns","metadata":{"highres-ticks":true}, "codeLink":")",
      options.code_link, R"(",)");

  if (options.snapshot_baseline_ns.has_value()) {
    output->Append(R"("snapshot_baseline_ns":)", *options.snapshot_baseline_ns,
                   R"(,)");
  }

  output->Append(absl::StrFormat(R"("useNewBackend": %s,)",
                                 options.use_new_backend ? "true" : "false"));
  absl::flat_hash_map<uint32_t, uint32_t> device_to_sort_index;
  if (options.mpmd_pipeline_view) {
    output->Append(
        absl::StrFormat(R"("mpmdPipelineView": %s,)",
                        options.mpmd_pipeline_view ? "true" : "false"));
    SortMpmdDevices(events, device_to_sort_index);
  }

  WriteDetails(options.details, output);
  WriteReturnedEventsSize(events.NumEvents(), output);
  WriteFilteredByVisibility(events.FilterByVisibility(), output);
  WriteTraceFullTimespan(&events.trace(), output);

  const Trace& trace = events.trace();
  WriteTasks(trace, output);

  auto references = BuildStackFrameReferences(trace);
  if (options.generate_stack_frames) {
    WriteStackFrames(trace, references, output);
  }

  output->Append(R"("traceEvents":[)");
  JsonSeparator<IOBuffer> separator(output);
  absl::btree_map<uint32_t, Device> ordered_devices(trace.devices().begin(),
                                                    trace.devices().end());
  for (const auto& [device_id, device] : ordered_devices) {
    if (device.has_name()) {
      separator.Add();
      output->Append(R"({"args":{"name":)", JsonEscape(device.name()),
                     R"(},"name":"process_name","ph":"M","pid":)", device_id,
                     R"(,"thread_count":)", device.resources_size(), "}");
    }
    if (!options.mpmd_pipeline_view) {
      separator.Add();
      output->Append(R"({"args":{"sort_index":)", device_id,
                     R"(},"name":"process_sort_index","ph":"M","pid":)",
                     device_id, "}");
    } else if (!device_to_sort_index.empty()) {
      const auto it = device_to_sort_index.find(device_id);
      const uint32_t sort_index = it != device_to_sort_index.end()
                                      ? it->second
                                      : kMpmdUnrankedSortIndexBase + device_id;
      separator.Add();
      output->Append(R"({"args":{"sort_index":)", sort_index,
                     R"(},"name":"process_sort_index","ph":"M","pid":)",
                     device_id, "}");
    }
    absl::btree_map<uint64_t, Resource> ordered_resources(
        device.resources().begin(), device.resources().end());
    for (const auto& [resource_id, resource] : ordered_resources) {
      if (resource.has_name()) {
        separator.Add();
        output->Append(R"({"args":{"name":)", JsonEscape(resource.name()),
                       R"(},"name":"thread_name","ph":"M","pid":)", device_id,
                       R"(,"tid":)", resource_id, "}");
      }
      if (!options.sort_resources_by_name.count(device_id)) {
        separator.Add();
        output->Append(R"({"args":{"sort_index":)", resource_id,
                       R"(},"name":"thread_sort_index","ph":"M","pid":)",
                       device_id, R"(,"tid":)", resource_id, "}");
      }
    }
  }

  TraceEventsColorerInterface* colorer = options.colorer;
  DefaultTraceEventsColorer default_colorer;
  if (colorer == nullptr) colorer = &default_colorer;
  colorer->SetUp(trace);

  // Write events.
  JsonEventWriter<IOBuffer, RawDataType> writer(colorer, trace, references,
                                                output);
  bool prev_was_counter = false;
  events.ForAllEvents([&](const TraceEvent& event) {
    bool is_counter_event = !event.has_resource_id() && !event.has_flow_id();
    if ((prev_was_counter && !is_counter_event) ||
        (!writer.isMatchingLastCounterEvent(event) && is_counter_event &&
         prev_was_counter)) {
      output->Append("]}");
    }
    separator.Add();
    if (is_counter_event) {
      writer.AddCounterEvent(event);
    } else {
      writer.WriteEvent(event);
    }
    prev_was_counter = is_counter_event;
  });
  if (prev_was_counter) {
    output->Append("]}");
  }
  size_t counter_event_count = writer.GetCounterEventCount();
  VLOG(1) << "Counter event count: " << counter_event_count;
  if (counter_event_count == tensorflow::profiler::kMaxCounterEvents) {
    output->Append(
        R"(], "showCounterMessage": "Only )",
        tensorflow::profiler::kMaxCounterEvents,
        R"( counter events are shown. Zoom in or pan to see more." )");
  } else {
    output->Append(R"(], "showCounterMessage": "" )");
  }
  output->Append(R"(,"totalCounterEvents":)", counter_event_count);
  output->Append(R"(})");
}

class IOBufferAdapter {
 public:
  explicit IOBufferAdapter(std::string* output) : output_(output) {}

  template <typename... AV>
  inline void Append(AV&&... args) {
    absl::StrAppend(output_, std::forward<AV>(args)...);
  }

  // Support IOBufferAdapter as a sink object for absl::Format.
  friend void AbslFormatFlush(IOBufferAdapter* buffer, absl::string_view s) {
    absl::StrAppend(buffer->output_, s);
  }

 private:
  std::string* output_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_EVENTS_TO_JSON_H_
