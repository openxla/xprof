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

#ifndef THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_ARROW_UTILS_H_
#define THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_ARROW_UTILS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/arrow/api.h"
#include "third_party/arrow/result.h"
#include "third_party/arrow/status.h"
#include "third_party/arrow/type.h"

namespace xprof::events_db::internal {

// Converts an `arrow::Status` to an `absl::Status`.
absl::Status ToAbslStatus(const arrow::Status& status);

// Converts an `absl::Status` to an `arrow::Status`.
arrow::Status ToArrowStatus(const absl::Status& status);

// Converts an `arrow::Result<T>` to an `absl::StatusOr<T>`.
template <typename T>
absl::StatusOr<T> ToAbslStatusOr(arrow::Result<T> result) {
  if (result.ok()) return std::move(result).ValueUnsafe();
  return ToAbslStatus(result.status());
}

// Converts an `absl::StatusOr<T>` to an `arrow::Result<T>`.
template <typename T>
arrow::Result<T> ToArrowResult(absl::StatusOr<T> status) {
  if (status.ok()) return std::move(*status);
  return ToArrowStatus(status.status());
}

// Type trait to identify `std::vector` types.
template <typename>
struct is_std_vector : std::false_type {};

template <typename... Args>
struct is_std_vector<std::vector<Args...>> : std::true_type {};

template <typename T>
constexpr bool is_std_vector_v = is_std_vector<T>::value;

template <typename>
constexpr bool always_false_v = false;

// Returns the corresponding Arrow DataType for a C++ type T supported by
// Events DB.
template <typename T>
std::shared_ptr<arrow::DataType> GetArrowType() {
  using CleanT = std::remove_cvref_t<T>;
  if constexpr (is_std_vector_v<CleanT>) {
    static_assert(
        !is_std_vector_v<std::remove_cvref_t<typename CleanT::value_type>>,
        "Nested vectors are not supported.");
  }
  if constexpr (std::is_same_v<CleanT, std::monostate>)
    return arrow::null();
  else if constexpr (std::is_same_v<CleanT, bool>)
    return arrow::boolean();
  else if constexpr (std::is_same_v<CleanT, int32_t>)
    return arrow::int32();
  else if constexpr (std::is_same_v<CleanT, uint32_t>)
    return arrow::uint32();
  else if constexpr (std::is_same_v<CleanT, int64_t>)
    return arrow::int64();
  else if constexpr (std::is_same_v<CleanT, uint64_t>)
    return arrow::uint64();
  else if constexpr (std::is_same_v<CleanT, double>)
    return arrow::float64();
  else if constexpr (std::is_same_v<CleanT, std::string>)
    return arrow::utf8();
  else if constexpr (is_std_vector_v<CleanT>)
    return arrow::list(GetArrowType<typename CleanT::value_type>());
  else
    static_assert(always_false_v<T>,
                  "Unsupported type for Arrow Parquet export.");
}

}  // namespace xprof::events_db::internal

#endif  // THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_ARROW_UTILS_H_
