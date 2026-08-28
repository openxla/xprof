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

#include "xprof/convert/events_db/tpu_trace_parser.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/tpu_xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/parent_event_tracker.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"
#include "xprof/convert/events_db/tpu_component.h"
#include "xprof/utils/hlo_module_map.h"
#include "xprof/utils/performance_info_wrapper.h"

namespace xprof::events_db::internal {
namespace {

constexpr uint32_t kRecordBufferCapacity = 64;

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

bool PopulateTensorCoreEventRecord(
    int64_t line_id, tsl::profiler::XEventContextTracker& step_context,
    tsl::profiler::XEventContextTracker& hlo_module_context,
    const tensorflow::profiler::HloModuleMap& hlo_module_map,
    uint64_t& last_hlo_module_id, const tsl::profiler::XEventVisitor& event,
    const tensorflow::profiler::HloModuleWrapper*& current_hlo_map,
    const FieldIndices& indices, Record& record) {
  switch (line_id) {
    case TpuComponent::kTensorCoreTraceMe:
    case TpuComponent::kTensorCore:
      record[indices.kernel_name] = event.Name();
      return true;
    case TpuComponent::kTensorCoreStepCounter:
      // These events are on-device-loop step marker.
      record[indices.kernel_name] = absl::StrCat("step:", event.Name());
      return true;
    case TpuComponent::kTensorCoreHloModule:
      // These events are HLO module events.
      record[indices.kernel_name] = absl::StrCat("HLO Module:", event.Name());
      return true;
    case TpuComponent::kTensorCoreHLO: {
      record[indices.kernel_name] = event.Name();
      record[indices.hlo_op] = event.DisplayName();
      std::optional<tsl::profiler::XEventVisitor> step_event =
          step_context.GetContainingEvent(event.GetTimespan());
      std::optional<tsl::profiler::XEventVisitor> hlo_module_event =
          hlo_module_context.GetContainingEvent(event.GetTimespan());
      if (step_event) {
        record[indices.step] = absl::StrCat("step:", step_event->Name());
      }
      if (hlo_module_event) {
        record[indices.hlo_module] = hlo_module_event->Name();

        if (hlo_module_event->Id() != last_hlo_module_id) {
          last_hlo_module_id = hlo_module_event->Id();
          current_hlo_map = nullptr;
          if (const std::optional<uint64_t> program_id =
                  GetProgramIdFromHloModuleName(hlo_module_event->Name())) {
            current_hlo_map = tensorflow::profiler::GetHloModule(
                &hlo_module_map, *program_id);
          }
        }
        if (current_hlo_map && !record[indices.hlo_op].empty()) {
          const auto* hlo_instruction =
              current_hlo_map->GetHloInstruction(record[indices.hlo_op]);
          if (hlo_instruction) {
            record[indices.hlo_fingerprint] = hlo_instruction->Fingerprint();
            auto perf_info = hlo_instruction->GetPerformanceInfoWrapper();
            if (perf_info != nullptr) {
              record[indices.flops] = perf_info->flops();
              record[indices.memory_accessed] = perf_info->bytes_accessed();
            }
            AssignIfNonEmpty(record, indices.input_tensors,
                             hlo_instruction->InputTensors());
            AssignIfNonEmpty(record, indices.output_tensors,
                             hlo_instruction->OutputTensors());
          }
        }
      }
      return true;
    }
    case TpuComponent::kTensorCoreOverlay:
      record[indices.kernel_name] = absl::StrCat("TC Overlay:", event.Name());
      return true;
    default:
      return false;
  }
}

absl::StatusOr<ParseStatus> ParseTpuTensorCoreTrace(
    const tensorflow::profiler::XPlane& device_trace,
    const tensorflow::profiler::HloModuleMap& hlo_module_map,
    const FieldIndices& indices, RecordConsumerRef consumer) {
  const tsl::profiler::XPlaneVisitor plane =
      tsl::profiler::CreateTfXPlaneVisitor(&device_trace);
  const absl::string_view device_name = GetTpuDeviceName(plane.Name());
  const tensorflow::profiler::HloModuleWrapper* current_hlo_map = nullptr;

  // Cache to avoid parsing program name multiple times. If the current program
  // event's metadata id is same as `last_hlo_module_id`, use `current_hlo_map`.
  uint64_t last_hlo_module_id = static_cast<uint64_t>(-1);

  tsl::profiler::XEventContextTracker step_context(
      &plane, tsl::profiler::FindLineWithId(
                  device_trace, TpuComponent::kTensorCoreStepCounter));
  tsl::profiler::XEventContextTracker hlo_module_context(
      &plane, tsl::profiler::FindLineWithId(
                  device_trace, TpuComponent::kTensorCoreHloModule));

  ParentEventTracker parent_tracker(kRecordBufferCapacity);
  for (const tensorflow::profiler::XLine& line : device_trace.lines()) {
    const tsl::profiler::XLineVisitor line_visitor(&plane, &line);
    parent_tracker.Reset();
    for (const tensorflow::profiler::XEvent& event : line.events()) {
      const tsl::profiler::XEventVisitor event_visitor(&plane, &line, &event);
      Record record = parent_tracker.GetOrCreateRecord();
      ExtractCommonInfo(device_name, line_visitor, event_visitor, indices,
                        record);
      if (!PopulateTensorCoreEventRecord(line_visitor.Id(), step_context,
                                         hlo_module_context, hlo_module_map,
                                         last_hlo_module_id, event_visitor,
                                         current_hlo_map, indices, record)) {
        continue;
      }
      ASSIGN_OR_RETURN(const StepControl control,
                       parent_tracker.AddRecord(
                           std::move(record),
                           static_cast<uint64_t>(event_visitor.TimestampNs()),
                           static_cast<uint64_t>(event_visitor.DurationNs()),
                           indices, consumer));
      if (control == StepControl::kStop) return ParseStatus::kStoppedEarly;
    }
    ASSIGN_OR_RETURN(const StepControl control, parent_tracker.Flush(consumer));
    if (control == StepControl::kStop) return ParseStatus::kStoppedEarly;
  }

  return ParseStatus::kComplete;
}

void PopulateSparseCoreEventRecord(int64_t line_id,
                                   const tsl::profiler::XEventVisitor& event,
                                   const FieldIndices& indices,
                                   Record& record) {
  switch (line_id) {
    case TpuComponent::kSparseCoreModule:
      record[indices.kernel_name] = absl::StrCat("HLO Module:", event.Name());
      break;
    case TpuComponent::kSparseCoreOps:
    case TpuComponent::kSparseCoreSyncs:
      record[indices.kernel_name] = event.Name();
      break;
    case TpuComponent::kSparseCoreStepCounter:
      record[indices.kernel_name] = absl::StrCat("step:", event.Name());
      break;
    case TpuComponent::kSparseCoreOverlay:
      record[indices.kernel_name] = absl::StrCat("SC Overlay:", event.Name());
      break;
    default:
      // Add all events on TEC.
      if (line_id >= TpuComponent::kSparseCoreTecBase &&
          line_id <= TpuComponent::kSparseCoreTec15) {
        record[indices.kernel_name] = event.Name();
      }
  }
}

absl::StatusOr<ParseStatus> ParseTpuSparseCoreTrace(
    const tensorflow::profiler::XPlane& device_trace,
    const FieldIndices& indices, RecordConsumerRef consumer) {
  const tsl::profiler::XPlaneVisitor plane =
      tsl::profiler::CreateTfXPlaneVisitor(&device_trace);
  const absl::string_view device_name = GetTpuDeviceName(plane.Name());
  Record record;
  for (const tensorflow::profiler::XLine& line : device_trace.lines()) {
    const tsl::profiler::XLineVisitor line_visitor(&plane, &line);
    for (const tensorflow::profiler::XEvent& event : line.events()) {
      const tsl::profiler::XEventVisitor event_visitor(&plane, &line, &event);
      record.clear();
      ExtractCommonInfo(device_name, line_visitor, event_visitor, indices,
                        record);
      PopulateSparseCoreEventRecord(line_visitor.Id(), event_visitor, indices,
                                    record);
      ASSIGN_OR_RETURN(const StepControl control, consumer(record));
      if (control == StepControl::kStop) return ParseStatus::kStoppedEarly;
    }
  }
  return ParseStatus::kComplete;
}

absl::StatusOr<ParseStatus> ParseTpuNonCoreTrace(
    const tensorflow::profiler::XPlane& device_trace,
    const FieldIndices& indices, RecordConsumerRef consumer) {
  const tsl::profiler::XPlaneVisitor plane =
      tsl::profiler::CreateTfXPlaneVisitor(&device_trace);
  const absl::string_view device_name = GetTpuDeviceName(plane.Name());
  Record record;
  for (const tensorflow::profiler::XLine& line : device_trace.lines()) {
    const tsl::profiler::XLineVisitor line_visitor(&plane, &line);
    for (const tensorflow::profiler::XEvent& event : line.events()) {
      const tsl::profiler::XEventVisitor event_visitor(&plane, &line, &event);
      record.clear();
      ExtractCommonInfo(device_name, line_visitor, event_visitor, indices,
                        record);
      ASSIGN_OR_RETURN(const StepControl control, consumer(record));
      if (control == StepControl::kStop) return ParseStatus::kStoppedEarly;
    }
  }
  return ParseStatus::kComplete;
}

}  // namespace

absl::StatusOr<ParseStatus> ParseTpuTrace(
    const tensorflow::profiler::XPlane& device_trace,
    const tensorflow::profiler::HloModuleMap& hlo_module_map,
    const FieldIndices& indices, RecordConsumerRef consumer) {
  if (tsl::profiler::GetTensorCoreId(device_trace.name()).has_value()) {
    return ParseTpuTensorCoreTrace(device_trace, hlo_module_map, indices,
                                   consumer);
  }
  if (tsl::profiler::GetSparseCoreId(device_trace.name()).has_value()) {
    return ParseTpuSparseCoreTrace(device_trace, indices, consumer);
  }
  return ParseTpuNonCoreTrace(device_trace, indices, consumer);
}

}  // namespace xprof::events_db::internal
