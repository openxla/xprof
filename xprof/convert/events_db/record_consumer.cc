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

#include "xprof/convert/events_db/record_consumer.h"

#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace xprof::events_db {

std::string_view StepControlToString(StepControl step_control) {
  switch (step_control) {
    case StepControl::kContinue:
      return "CONTINUE";
    case StepControl::kStop:
      return "STOP";
    default:
      return "UNKNOWN";
  }
}

absl::StatusOr<StepControl> StepControlFromString(std::string_view name) {
  if (name == StepControlToString(StepControl::kContinue))
    return StepControl::kContinue;
  if (name == StepControlToString(StepControl::kStop))
    return StepControl::kStop;
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown StepControl: '", name, "'"));
}

std::string_view ParseStatusToString(ParseStatus parse_status) {
  switch (parse_status) {
    case ParseStatus::kComplete:
      return "COMPLETE";
    case ParseStatus::kStoppedEarly:
      return "STOPPED_EARLY";
    default:
      return "UNKNOWN";
  }
}

absl::StatusOr<ParseStatus> ParseStatusFromString(std::string_view name) {
  if (name == ParseStatusToString(ParseStatus::kComplete))
    return ParseStatus::kComplete;
  if (name == ParseStatusToString(ParseStatus::kStoppedEarly))
    return ParseStatus::kStoppedEarly;
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown ParseStatus: '", name, "'"));
}

}  // namespace xprof::events_db
