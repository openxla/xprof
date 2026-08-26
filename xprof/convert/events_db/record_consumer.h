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

#ifndef THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_RECORD_CONSUMER_H_
#define THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_RECORD_CONSUMER_H_

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"

namespace xprof::events_db {

class Record;

// Control-flow decision returned by the record consumer callback after
// processing each streamed `Record`.
enum class StepControl {
  kContinue,  // Continue parsing subsequent events.
  kStop,      // Clean early stop requested (e.g. limit reached or match found).
};

// Final outcome of the entire parsing operation.
enum class ParseStatus {
  kComplete,      // Scanned the entire trace to completion.
  kStoppedEarly,  // Parsing stopped early and cleanly because consumer returned
                  // `kStop`.
};

// Non-owning callable reference for streaming parsed `Record`s. Functions that
// receive this reference as an argument must not store it beyond the immediate
// function call.
//
// The caller retains ownership of the `Record` (e.g. to reuse its capacity
// across iterations). The callee does not own the `Record` and must not retain
// references or pointers to it after returning. However, moving elements or
// field values (such as `std::string` or `std::vector`) out of the `Record`
// is permitted and encouraged to avoid unnecessary copying.
//
// - Returning `StepControl::kContinue` continues ingestion.
// - Returning `StepControl::kStop` stops ingestion early and returns
//   `ParseStatus::kStoppedEarly`.
// - Returning an error status immediately aborts ingestion and propagates the
//   error.
using RecordConsumerRef =
    absl::FunctionRef<absl::StatusOr<StepControl>(Record&)>;

}  // namespace xprof::events_db

#endif  // THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_RECORD_CONSUMER_H_
