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

#include "xprof/convert/events_db/custom_trace_parser.h"

#include <string>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/record_consumer.h"

namespace xprof::events_db::internal {

absl::StatusOr<ParseStatus> ParseCustomTrace(
    const tensorflow::profiler::XPlane& custom_trace,
    const FieldIndices& indices, RecordConsumerRef consumer) {
  const tsl::profiler::XPlaneVisitor plane =
      tsl::profiler::CreateTfXPlaneVisitor(&custom_trace);

  Record record;
  for (const tensorflow::profiler::XLine& line : custom_trace.lines()) {
    const tsl::profiler::XLineVisitor line_visitor(&plane, &line);
    for (const tensorflow::profiler::XEvent& event : line.events()) {
      const tsl::profiler::XEventVisitor event_visitor(&plane, &line, &event);
      record.clear();
      ExtractCommonInfo(plane.Name(), line_visitor, event_visitor, indices,
                        record);
      record[indices.kernel_name] = event_visitor.Name();

      ASSIGN_OR_RETURN(const StepControl control, consumer(record));
      if (control == StepControl::kStop) return ParseStatus::kStoppedEarly;
    }
  }

  return ParseStatus::kComplete;
}

}  // namespace xprof::events_db::internal
