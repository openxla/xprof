/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

#include "xprof/convert/events_db/gpu_trace_parser.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"
#include "xprof/utils/gpu_event_stats.h"
#include "xprof/utils/hlo_module_map.h"
#include "xprof/utils/performance_info_wrapper.h"

namespace xprof::events_db::internal {
namespace {

bool IsXlaLine(absl::string_view line_name) {
  return absl::StartsWith(line_name, tsl::profiler::kXlaModuleLineName) ||
         absl::StartsWith(line_name, tsl::profiler::kXlaOpLineName);
}

template <typename T, typename U>
void AssignIfNonEmpty(Record& record, TypedFieldIndex<T> field, U&& value) {
  if (!value.empty()) {
    if constexpr (std::is_assignable_v<T&, U>) {
      record[field] = std::forward<U>(value);
    } else {
      record[field].assign(std::begin(value), std::end(value));
    }
  }
}

absl::StatusOr<StepControl> ParseXlaLineEvent(
    Record& record, absl::string_view device_name,
    const tsl::profiler::XLineVisitor& line,
    const tsl::profiler::XEventVisitor& event, const FieldIndices& indices,
    RecordConsumerRef consumer) {
  record.clear();
  internal::ExtractCommonInfo(device_name, line, event, indices, record);
  if (absl::StartsWith(line.Name(), tsl::profiler::kXlaModuleLineName)) {
    record[indices.kernel_name] = absl::StrCat("HLO Module:", event.Name());
  } else {
    record[indices.kernel_name] = event.Name();
  }
  return consumer(record);
}

std::vector<std::pair<absl::string_view, std::string>> BuildTraceArgs(
    const tsl::profiler::XEventVisitor& event) {
  std::vector<std::pair<absl::string_view, std::string>> key_value_pairs;
  event.ForEachStat([&](const tsl::profiler::XStatVisitor& stat) {
    if (stat.Type().has_value()) {
      switch (stat.Type().value()) {
        case tsl::profiler::StatType::kTfOp:
        case tsl::profiler::StatType::kHloOp:
        case tsl::profiler::StatType::kHloModule:
        case tsl::profiler::StatType::kProgramId:
        case tsl::profiler::StatType::kGroupId:
        case tsl::profiler::StatType::kCorrelationId:
        case tsl::profiler::StatType::kKernelDetails:
        case tsl::profiler::StatType::kTensorShapes:
          break;
        default:
          key_value_pairs.emplace_back(stat.Name(), stat.ToString());
          break;
      }
    } else {
      key_value_pairs.emplace_back(stat.Name(), stat.ToString());
    }
  });
  return key_value_pairs;
}

absl::StatusOr<StepControl> ParseDeviceLineEvent(
    Record& record, absl::string_view device_name,
    const tsl::profiler::XLineVisitor& line,
    const tsl::profiler::XEventVisitor& event,
    const tsl::profiler::GroupMetadataMap& group_metadata_map,
    const tensorflow::profiler::HloModuleMap& hlo_module_map,
    const FieldIndices& indices, RecordConsumerRef consumer) {
  record.clear();

  record[indices.device] = device_name;
  record[indices.category] = "device";
  record[indices.kernel_name] = event.Name();
  record[indices.stream_id] = static_cast<uint32_t>(line.Id());
  record[indices.start_ns] = static_cast<uint64_t>(event.TimestampNs());
  record[indices.end_ns] = static_cast<uint64_t>(event.EndTimestampNs());
  record[indices.self_time_ns] = static_cast<uint64_t>(event.DurationNs());

  tensorflow::profiler::GpuEventStats stats(&event);
  if (stats.group_id.has_value()) {
    const auto it = group_metadata_map.find(stats.group_id.value());
    if (it != group_metadata_map.end()) record[indices.step] = it->second.name;
  }
  if (stats.correlation_id.has_value()) {
    record[indices.correlation_id] =
        static_cast<uint32_t>(stats.correlation_id.value());
  }
  if (stats.IsKernel()) {
    record[indices.kernel_details] = stats.kernel_details;
  }
  if (stats.IsXlaOp()) {
    // Consider only the innermost HLO op.
    record[indices.hlo_op] = stats.hlo_op_names.back();
    record[indices.hlo_module] = stats.hlo_module_name;
    const tensorflow::profiler::HloInstructionWrapper* hlo_instruction =
        tensorflow::profiler::GetHloInstruction(
            hlo_module_map, stats.program_id, stats.hlo_op_names.back());
    if (hlo_instruction != nullptr) {
      const tensorflow::profiler::PerformanceInfoWrapper* perf_info =
          hlo_instruction->GetPerformanceInfoWrapper();
      if (perf_info != nullptr) {
        record[indices.flops] = static_cast<uint64_t>(perf_info->flops());
        record[indices.memory_accessed] =
            static_cast<uint64_t>(perf_info->bytes_accessed());
      }
      // GPU/XLA's tf_op is always XlaRun, we can get individual op name
      // from symbol table.
      const tsl::profiler::TfOp tf_op =
          tsl::profiler::ParseTfOpFullname(hlo_instruction->TfOpName());
      record[indices.tf_op_name] = tf_op.name;
      record[indices.tf_op_type] = tf_op.type;
      record[indices.hlo_fingerprint] = hlo_instruction->Fingerprint();
      AssignIfNonEmpty(record, indices.input_tensors,
                       hlo_instruction->InputTensors());
      AssignIfNonEmpty(record, indices.output_tensors,
                       hlo_instruction->OutputTensors());
    }
  } else if (stats.IsTfOp()) {
    const tsl::profiler::TfOp tf_op =
        tsl::profiler::ParseTfOpFullname(stats.tf_op_fullname);
    record[indices.tf_op_name] = tf_op.name;
    record[indices.tf_op_type] = tf_op.type;
    AssignIfNonEmpty(record, indices.input_tensors,
                     tsl::profiler::ParseTensorShapes(stats.tensor_shapes));
  }

  std::vector<std::pair<absl::string_view, std::string>> trace_args =
      BuildTraceArgs(event);
  if (!trace_args.empty()) {
    record[indices.trace_args] = FormatTraceArgs(trace_args);
  }

  return consumer(record);
}

}  // namespace

absl::StatusOr<ParseStatus> ParseGpuTrace(
    const tensorflow::profiler::XPlane& device_trace,
    const tensorflow::profiler::HloModuleMap& hlo_module_map,
    const tsl::profiler::GroupMetadataMap& group_metadata_map,
    const FieldIndices& indices, RecordConsumerRef consumer) {
  const tsl::profiler::XPlaneVisitor plane =
      tsl::profiler::CreateTfXPlaneVisitor(&device_trace);
  const std::string device_name = absl::StrCat("gpu:", plane.Id());

  Record record;
  for (const tensorflow::profiler::XLine& line : device_trace.lines()) {
    const tsl::profiler::XLineVisitor line_visitor(&plane, &line);
    // Perf Counter events are handled separately.
    if (line_visitor.Name() == tsl::profiler::kCounterEventsLineName) continue;

    if (IsXlaLine(line_visitor.Name())) {
      for (const tensorflow::profiler::XEvent& event : line.events()) {
        TF_ASSIGN_OR_RETURN(const StepControl status,
                            ParseXlaLineEvent(record, device_name, line_visitor,
                                              tsl::profiler::XEventVisitor(
                                                  &plane, &line, &event),
                                              indices, consumer));
        if (status == StepControl::kStop) return ParseStatus::kStoppedEarly;
      }
      continue;
    }

    if (tsl::profiler::IsDerivedThreadId(line_visitor.Id())) continue;

    for (const tensorflow::profiler::XEvent& event : line.events()) {
      TF_ASSIGN_OR_RETURN(
          const StepControl status,
          ParseDeviceLineEvent(
              record, device_name, line_visitor,
              tsl::profiler::XEventVisitor(&plane, &line, &event),
              group_metadata_map, hlo_module_map, indices, consumer));
      if (status == StepControl::kStop) return ParseStatus::kStoppedEarly;
    }
  }

  return ParseStatus::kComplete;
}

}  // namespace xprof::events_db::internal
