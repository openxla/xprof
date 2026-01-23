/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_CONVERT_XSPACE_TO_EVENT_TIME_FRACTION_ANALYZER_H_
#define THIRD_PARTY_XPROF_CONVERT_XSPACE_TO_EVENT_TIME_FRACTION_ANALYZER_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"
#include "plugin/xprof/protobuf/event_time_fraction_analyzer.pb.h"

namespace tensorflow {
namespace profiler {

// Converts XSpace to EventTimeFractionAnalyzerResult for the given event.
absl::StatusOr<EventTimeFractionAnalyzerResult>
ConvertXSpaceToEventTimeFractionAnalyzerResult(
    const XSpace& xspace, absl::string_view target_event_name);

// Converts multiple XSpaces to EventTimeFractionAnalyzerResult for the given
// event.
absl::StatusOr<EventTimeFractionAnalyzerResult>
ConvertMultiXSpacesToEventTimeFractionAnalyzerResult(
    const SessionSnapshot& session_snapshot,
    absl::string_view target_event_name);

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_XSPACE_TO_EVENT_TIME_FRACTION_ANALYZER_H_
