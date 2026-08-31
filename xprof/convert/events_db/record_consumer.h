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

#include <type_traits>

#include "absl/status/status.h"
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

// Non-owning view of a record consumer and its completion lifecycle. Functions
// that receive this reference as an argument must not store it beyond the
// immediate function call.
//
// The parser retains ownership of the `Record` to reuse allocated capacity
// across iterations. The consumer must not retain references or pointers to the
// `Record` after returning. However, moving elements or field values (such as
// `std::string` or `std::vector`) out of the `Record` is permitted and
// encouraged to avoid unnecessary copying.
//
// ### Record Consumption
// A consumer receives streamed `Record`s during parsing via one of the
// following methods (only the first one is called when both are available):
//   - `Consume(Record&)` -> `absl::StatusOr<StepControl>`
//   - `operator()(Record&)` -> `absl::StatusOr<StepControl>` (e.g. lambdas)
//
// ### Completion & Finalization
// After all parsing threads complete, the parser invokes an overload of the
// `Finalize` method, if defined, on the main thread (only the first one is
// called when both are available):
//   - `Finalize(const absl::StatusOr<ParseStatus>& result)` -> `absl::Status`
//     or `void`:
//     Receives the final parsing outcome. This allows consumers to commit
//     transactions on success, record metadata, or rollback and clean up
//     temporary files on failure.
//   - `Finalize()` -> `absl::Status` or `void`:
//     Is called only on successful parsing (`result.ok()`).
class RecordConsumerRef {
 public:
  template <typename T, typename = std::enable_if_t<!std::is_same_v<
                            RecordConsumerRef, std::remove_cvref_t<T>>>>
  RecordConsumerRef(T&& target) noexcept
      : target_(const_cast<void*>(static_cast<const void*>(&target))),
        consume_fn_([](void* ptr, Record& r) -> absl::StatusOr<StepControl> {
          using RawT = std::remove_reference_t<T>;
          RawT* target = static_cast<RawT*>(ptr);
          constexpr bool kHasConsume = requires { target->Consume(r); };
          static_assert(kHasConsume || std::is_invocable_v<RawT&, Record&>,
                        "RecordConsumerRef requires a consumer with either "
                        "Consume(Record&) or operator()(Record&) "
                        "method.");
          if constexpr (kHasConsume) {
            return target->Consume(r);
          } else {
            return target->operator()(r);
          }
        }),
        finalize_fn_(
            [](void* ptr,
               const absl::StatusOr<ParseStatus>& result) -> absl::Status {
              using RawT = std::remove_reference_t<T>;
              RawT* target = static_cast<RawT*>(ptr);
              constexpr bool kHasFinalizeWithResult =
                  requires { target->Finalize(result); };
              constexpr bool kHasFinalize = requires { target->Finalize(); };
              if constexpr (kHasFinalizeWithResult) {
                constexpr bool kIsVoid =
                    std::is_void_v<decltype(target->Finalize(result))>;
                if constexpr (kIsVoid) {
                  target->Finalize(result);
                  return absl::OkStatus();
                } else {
                  return target->Finalize(result);
                }
              } else if constexpr (kHasFinalize) {
                constexpr bool kIsVoid =
                    std::is_void_v<decltype(target->Finalize())>;
                if (result.ok()) {
                  if constexpr (kIsVoid) {
                    target->Finalize();
                    return absl::OkStatus();
                  } else {
                    return target->Finalize();
                  }
                }
                return absl::OkStatus();
              } else {
                return absl::OkStatus();
              }
            }) {}

  RecordConsumerRef(const RecordConsumerRef&) = default;
  RecordConsumerRef(RecordConsumerRef&&) = default;

  RecordConsumerRef& operator=(const RecordConsumerRef&) = default;
  RecordConsumerRef& operator=(RecordConsumerRef&&) = default;

  absl::StatusOr<StepControl> operator()(Record& record) const {
    return consume_fn_(target_, record);
  }
  absl::StatusOr<StepControl> Consume(Record& record) const {
    return consume_fn_(target_, record);
  }
  absl::Status Finalize(const absl::StatusOr<ParseStatus>& result) const {
    return finalize_fn_(target_, result);
  }

 private:
  void* target_;
  absl::StatusOr<StepControl> (*consume_fn_)(void*, Record&);
  absl::Status (*finalize_fn_)(void*, const absl::StatusOr<ParseStatus>&);
};

static_assert(std::is_trivially_copyable_v<RecordConsumerRef>);
static_assert(std::is_copy_constructible_v<RecordConsumerRef>);
static_assert(std::is_move_constructible_v<RecordConsumerRef>);

}  // namespace xprof::events_db

#endif  // THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_RECORD_CONSUMER_H_
