/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XPROF_CONVERT_STEP_EVENTS_TO_STEPS_DB_H_
#define XPROF_CONVERT_STEP_EVENTS_TO_STEPS_DB_H_

#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/types.h"
#include "plugin/xprof/protobuf/steps_db.pb.h"
#include "xprof/utils/event_span.h"

namespace tensorflow {
namespace profiler {

TF_CONST_INIT extern const tsl::uint32 kDefaultGpuLocalCoreId;

// Converts from overlapped Step-Events to StepDatabaseResult.
StepDatabaseResult ConvertStepEventsToStepDb(
    bool has_device, bool maybe_drop_incomplete_steps,
    StepEvents& overlapped_step_events);

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_STEP_EVENTS_TO_STEPS_DB_H_
